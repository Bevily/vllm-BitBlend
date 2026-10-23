# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blackwell SM120 MX backend: cuDNN Low, fused CUTLASS Mid and High."""

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch

from vllm import envs
from vllm.model_executor.kernels.linear.bitblend.base import (
    BackendRequest,
    BitBlendBackend,
    KernelInvocation,
    PreparedLayer,
    SupportResult,
)
from vllm.model_executor.layers.quantization.bitblend.profile import (
    MX_PROFILE,
    TierSpec,
)


@dataclass(frozen=True)
class _State:
    weights: dict[str, torch.Tensor]
    scales: dict[str, torch.Tensor]
    database: Any = None


class Sm120Backend(BitBlendBackend):
    supports_graphs = True
    supports_mixed_graphs = True

    def create_graph_workspace(self, layers, max_tokens, device):
        from vllm.model_executor.kernels.linear.bitblend.sm120.graph import (
            MixedGraphWorkspace,
        )

        geometries = {
            (
                p.request.geometry.local_input_size,
                p.request.geometry.local_output_size,
            )
            for p in layers
        }
        return MixedGraphWorkspace(
            geometries, max_tokens, device, database=layers[0].state.database
        )

    @torch.compiler.disable
    @torch.no_grad()
    def run_grouped(self, prepared, x, spans, bias, workspace):
        from vllm.model_executor.kernels.linear.bitblend.sm120.activation import (
            prepare_activation,
        )
        from vllm.model_executor.kernels.linear.bitblend.sm120.ops import (
            prepare_grouped,
        )

        state = prepared.state
        if (
            x.dtype != torch.bfloat16
            or x.device != state.weights["p0"].device
            or x.ndim != 2
            or x.shape[1] != prepared.request.geometry.local_input_size
        ):
            raise ValueError("Invalid grouped BF16 input")
        from vllm.model_executor.kernels.linear.bitblend.sm120.graph import (
            MixedGraphWorkspace,
        )

        if isinstance(workspace, MixedGraphWorkspace):
            return workspace.run(prepared, x, bias)
        groups, covered = [], 0
        with torch.cuda.device(x.device):
            for span in spans:
                if span.start != covered or span.stop <= span.start:
                    raise ValueError("Grouped spans must cover input in order")
                self.resolve_invocation(
                    prepared,
                    prepared.request.profile.tier(span.tier),
                    (span.stop - span.start, x.shape[1]),
                )
                groups.append((span.tier, span.stop - span.start))
                covered = span.stop
            if covered != x.shape[0]:
                raise ValueError("Grouped spans must cover all input tokens")
            groups = prepare_activation(x.contiguous(), groups).run()
            config = None
            if state.database is not None:
                config = state.database.lookup_grouped(
                    [(s.tier, s.stop - s.start) for s in spans],
                    prepared.request.geometry.local_output_size,
                    x.shape[1],
                )
            return prepare_grouped(
                groups,
                state.weights,
                state.scales,
                config=config,
                bias=bias,
            ).run()

    @classmethod
    def supports(cls, request: BackendRequest) -> SupportResult:
        if request.profile != MX_PROFILE:
            return SupportResult(False, "sm120 supports mx_residual_3plane only")
        if request.dtype != "bfloat16":
            return SupportResult(False, "sm120 requires BF16 activations")
        device = torch.device(request.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return SupportResult(False, "sm120 requires CUDA")
        if torch.cuda.get_device_capability(device) != (12, 0):
            return SupportResult(False, "sm120 requires compute capability 12.0")
        geometry = request.geometry
        if geometry.local_input_size % 128 or geometry.local_output_size % 8:
            return SupportResult(False, "sm120 requires K divisible by 128 and N by 8")
        if not request.tiers or any(
            tier not in ("low", "mid", "high") for tier in request.tiers
        ):
            return SupportResult(False, "sm120 requires Low, Mid or High tiers")
        if importlib.util.find_spec("flashinfer") is None:
            return SupportResult(
                False, "sm120 requires FlashInfer with CUDA JIT support"
            )
        from vllm.utils.flashinfer import has_flashinfer_cutedsl

        if not has_flashinfer_cutedsl():
            return SupportResult(False, "sm120 requires FlashInfer CuTe-DSL")
        return SupportResult(True)

    def prepare(self, layer: torch.nn.Module, request: BackendRequest) -> PreparedLayer:
        from flashinfer.quantization import block_scale_interleave

        from vllm.model_executor.kernels.linear.bitblend.sm120 import activation
        from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
            GROUPED_DEFAULTS,
        )
        from vllm.model_executor.kernels.linear.bitblend.sm120.jit import (
            grouped_module,
            warp_module,
        )

        support = self.supports(request)
        if not support.supported:
            raise ValueError(support.reason)
        device = torch.device(request.device)
        weights, scales = {}, {}
        database = None
        from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
            load_database,
            resolve_database_path,
        )

        with torch.cuda.device(device):
            database_path = resolve_database_path(envs.VLLM_BITBLEND_SM120_TACTICS)
            if database_path:
                database = load_database(database_path)
        for plane in request.profile.required_planes(request.tiers):
            payload = getattr(layer, f"weight_{plane.name}").detach()
            scale = getattr(layer, f"weight_scale_{plane.name}").detach()
            expected_weight, expected_scale = plane.shapes(
                request.geometry.local_output_size, request.geometry.local_input_size
            )
            if (
                payload.device != device
                or scale.device != device
                or payload.dtype != torch.uint8
                or scale.dtype != torch.uint8
                or tuple(payload.shape) != expected_weight
                or tuple(scale.shape) != expected_scale
            ):
                raise ValueError(f"Invalid sm120 checkpoint plane {plane.name}")
            if bool((scale == 255).any()):
                raise ValueError("UE8M0 scale byte 255 is reserved for NaN")
            if plane.bits == 8:
                payload = payload.contiguous().view(torch.float8_e4m3fn)
                if not bool(torch.isfinite(payload.float()).all()):
                    raise ValueError("MXFP8 weight payload contains NaN")
            weights[plane.name] = payload.contiguous()
            scales[plane.name] = block_scale_interleave(scale.contiguous()).flatten()
        # Compile and check resource limits during loading, before any forward.
        with torch.cuda.device(device):
            if set(request.tiers) == {"low", "mid", "high"}:
                for high_threads in (2, 4):
                    activation.quantize_kernel(
                        request.geometry.local_input_size, high_threads
                    )
                for config in GROUPED_DEFAULTS:
                    grouped_module(config.variant)
                if database is not None:
                    for config in database.grouped_configurations(
                        request.geometry.local_output_size,
                        request.geometry.local_input_size,
                    ):
                        module = grouped_module(config.variant)
                        if not (module.resources() >> 8) & 255:
                            raise ValueError(
                                "Verified grouped kernel has zero occupancy"
                            )
            for tier in request.tiers:
                if tier in ("mid", "high"):
                    warp_module(tier).initialize(torch.cuda.current_device())
                    if database is not None:
                        for config, _ in database.configurations(
                            tier,
                            request.geometry.local_output_size,
                            request.geometry.local_input_size,
                        ):
                            warp_module(tier, *config.variant)
        return PreparedLayer(
            "sm120_mx_128x4_v1", _State(weights, scales, database), request
        )

    def resolve_invocation(
        self,
        prepared: PreparedLayer,
        tier: TierSpec,
        shape: tuple[int, ...],
        *,
        graph_mode: bool = False,
    ) -> KernelInvocation:
        request = prepared.request
        if tier.name not in request.tiers or tier != request.profile.tier(tier.name):
            raise ValueError("Tier does not match prepared sm120 weights")
        if not shape or shape[-1] != request.geometry.local_input_size:
            raise ValueError("Activation shape does not match local K")
        return KernelInvocation(tier, shape, graph_mode=graph_mode)

    @torch.compiler.disable
    @torch.no_grad()
    def run(
        self,
        prepared: PreparedLayer,
        x: torch.Tensor,
        invocation: KernelInvocation,
        bias: torch.Tensor | None,
        workspace: Any,
    ) -> torch.Tensor:
        from vllm.model_executor.kernels.linear.bitblend.sm120.ops import gemm, quantize

        self.resolve_invocation(
            prepared, invocation.tier, tuple(x.shape), graph_mode=invocation.graph_mode
        )
        state = prepared.state
        request = prepared.request
        device = state.weights["p0"].device
        if (
            x.device != device
            or x.dtype != torch.bfloat16
            or tuple(x.shape) != invocation.shape
        ):
            raise ValueError("Activation dtype/device/shape does not match invocation")
        n = request.geometry.local_output_size
        if bias is not None and (
            bias.shape != (n,) or bias.dtype != x.dtype or bias.device != device
        ):
            raise ValueError("Bias must be BF16 [local N] on the activation device")
        with torch.cuda.device(device):
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            if flat.shape[0] == 0:
                return x.new_empty((*x.shape[:-1], n))
            activation, scale = quantize(flat, invocation.tier.name)
            selected = None
            if state.database is not None and invocation.tier.name in ("mid", "high"):
                selected = state.database.lookup(
                    invocation.tier.name, flat.shape[0], n, flat.shape[1]
                )
            config, tactic = selected if selected is not None else (None, 0)
            output = gemm(
                invocation.tier.name,
                activation,
                scale,
                state.weights,
                state.scales,
                config=config,
                tactic=tactic,
            )[: flat.shape[0]]
            if bias is not None:
                output = output + bias
            return output.reshape(*x.shape[:-1], n)
