# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit-only eager numerical oracle, with decoded FP32 weights and GEMMs.

This intentionally allocates intermediates and synchronizes for validation.
It neither emulates Tensor Core accumulation nor implements a serving backend.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.kernels.linear.bitblend.base import (
    BackendRequest,
    BitBlendBackend,
    KernelInvocation,
    PreparedLayer,
    SupportResult,
)
from vllm.model_executor.layers.quantization.bitblend.profile import (
    FP8_PROFILE,
    MX_PROFILE,
    PlaneSpec,
    TierSpec,
)

_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def decode_plane(
    payload: torch.Tensor, scales: torch.Tensor, plane: PlaneSpec
) -> torch.Tensor:
    """Decode canonical checkpoint bytes, without runtime scale swizzles."""
    if payload.dtype != torch.uint8 or payload.ndim != 2:
        raise ValueError("Plane payload must be a two-dimensional uint8 tensor")
    n, packed_k = payload.shape
    k = packed_k * 8 // plane.bits
    weight_shape, scale_shape = plane.shapes(n, k)
    scale_dtype = torch.uint8 if plane.scale_block else torch.float32
    if (
        tuple(payload.shape) != weight_shape
        or tuple(scales.shape) != scale_shape
        or scales.dtype != scale_dtype
        or scales.device != payload.device
    ):
        raise ValueError(f"Invalid payload/scale contract for {plane.name}")
    if plane.scale_block:
        if bool((scales == 255).any()):
            raise ValueError("UE8M0 scale byte 255 is reserved for NaN")
        # ldexp also represents UE8M0 byte 0 (2**-127) correctly.
        scale = torch.ldexp(
            torch.ones_like(scales, dtype=torch.float32), scales.int() - 127
        ).repeat_interleave(plane.scale_block, dim=-1)
    else:
        _finite(scales, "Weight scales")
        if bool((scales <= 0).any()):
            raise ValueError("FP8 channel scales must be positive")
        scale = scales
    if plane.bits == 4:
        codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
        table = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=payload.device)
        values = table[(codes & 7).long()]
        values = torch.where((codes & 8) != 0, -values, values)
    else:
        values = payload.contiguous().view(torch.float8_e4m3fn).float()
    _finite(values, "Weight payload")
    result = values * scale
    _finite(result, "Decoded weights")
    return result


def quantize_activation(
    x: torch.Tensor, activation: str, *, fp4_scale_rule: str = "rounded-amax"
) -> torch.Tensor:
    """Return quantized/dequantized FP32 activations, once per layer call.

    FP4 defaults to vLLM's rounded-amax rule; 'ceil' validates CuTe-DSL.
    Both use nearest-even E2M1 values.
    MXFP8 uses ceil(log2(amax/448)); channel-profile FP8 uses amax/448
    per token (scale 1 for zero tokens). Nonfinite inputs are rejected.
    """
    if fp4_scale_rule not in ("rounded-amax", "ceil"):
        raise ValueError(f"Unsupported FP4 scale rule: {fp4_scale_rule}")
    values = x.float()
    _finite(values, "Activations")
    if activation == "fp8_per_token":
        amax = values.abs().amax(dim=-1, keepdim=True)
        scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
        scale = torch.where(amax == 0, 1.0, scale)
        quantized = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return quantized.float() * scale
    if activation not in ("mxfp4_block32", "mxfp8_block32"):
        raise ValueError(f"Unsupported reference activation format: {activation}")
    if values.shape[-1] % 32:
        raise ValueError("MX activation K must be divisible by 32")
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    if activation == "mxfp4_block32" and fp4_scale_rule == "ceil":
        exponent = torch.ceil(torch.log2(amax / 6.0))
    elif activation == "mxfp4_block32":
        # Match _compute_quant_and_scale in vLLM's mxfp4_utils.py.
        rounded = ((amax.view(torch.int32) + 0x200000) & 0x7F800000).view(torch.float32)
        exponent = (
            torch.floor(torch.log2(rounded.clamp_min(torch.finfo(torch.float32).tiny)))
            - 2
        )
    else:
        exponent = torch.ceil(torch.log2(amax / 448.0))
    exponent = exponent.clamp(-127, 127).int()
    normalized = torch.ldexp(blocks, -exponent)
    if activation == "mxfp4_block32":
        # Odd codes lose exact midpoint ties to their even neighbours.
        table = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=x.device)
        priority = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
        distances = (normalized.abs().unsqueeze(-1) - table[priority]).abs()
        quantized = table[priority[distances.argmin(dim=-1)]]
        quantized = torch.copysign(quantized, normalized)
    else:
        quantized = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return torch.ldexp(quantized, exponent).reshape(x.shape)


@dataclass(frozen=True)
class _ReferenceState:
    weights: dict[str, torch.Tensor]


class ReferenceBackend(BitBlendBackend):
    """CPU/CUDA PyTorch oracle, selectable only with backend='reference'."""

    @classmethod
    def supports(cls, request: BackendRequest) -> SupportResult:
        if request.profile not in (MX_PROFILE, FP8_PROFILE):
            return SupportResult(False, "unknown profile")
        if request.dtype != "bfloat16":
            return SupportResult(False, "only BF16 activations are supported")
        if torch.device(request.device).type not in ("cpu", "cuda"):
            return SupportResult(False, "only CPU and CUDA eager execution")
        try:
            if not request.tiers:
                raise ValueError("at least one tier is required")
            for tier in request.tiers:
                request.profile.tier(tier)
            for plane in request.profile.required_planes(request.tiers):
                plane.shapes(
                    request.geometry.local_output_size,
                    request.geometry.local_input_size,
                )
        except ValueError as error:
            return SupportResult(False, str(error))
        return SupportResult(True)

    def prepare(self, layer: torch.nn.Module, request: BackendRequest) -> PreparedLayer:
        support = self.supports(request)
        if not support.supported:
            raise ValueError(support.reason)
        weights = {}
        for plane in request.profile.required_planes(request.tiers):
            payload = getattr(layer, f"weight_{plane.name}")
            scales = getattr(layer, f"weight_scale_{plane.name}")
            if payload.device != torch.device(request.device):
                raise ValueError("Plane device does not match backend request")
            expected, _ = plane.shapes(
                request.geometry.local_output_size, request.geometry.local_input_size
            )
            if tuple(payload.shape) != expected:
                raise ValueError(f"Plane {plane.name} does not match local geometry")
            weights[plane.name] = decode_plane(payload.detach(), scales.detach(), plane)
        return PreparedLayer("reference_fp32_v1", _ReferenceState(weights), request)

    def resolve_invocation(
        self,
        prepared: PreparedLayer,
        tier: TierSpec,
        shape: tuple[int, ...],
        *,
        graph_mode: bool = False,
    ) -> KernelInvocation:
        if graph_mode:
            raise NotImplementedError("Reference backend supports eager execution only")
        request = prepared.request
        if tier.name not in request.tiers or tier != request.profile.tier(tier.name):
            raise ValueError("Tier does not match prepared reference weights")
        if not shape or shape[-1] != request.geometry.local_input_size:
            raise ValueError("Activation shape does not match local K")
        return KernelInvocation(tier, shape)

    @torch.no_grad()
    def run(
        self,
        prepared: PreparedLayer,
        x: torch.Tensor,
        invocation: KernelInvocation,
        bias: torch.Tensor | None,
        workspace: Any,
    ) -> torch.Tensor:
        self.resolve_invocation(
            prepared, invocation.tier, tuple(x.shape), graph_mode=invocation.graph_mode
        )
        state = prepared.state
        device = next(iter(state.weights.values())).device
        if (
            x.dtype != torch.bfloat16
            or x.device != device
            or tuple(x.shape) != invocation.shape
        ):
            raise ValueError("Activation dtype/device/shape does not match invocation")
        if bias is not None and (
            bias.shape != (prepared.request.geometry.local_output_size,)
            or bias.device != device
            or bias.dtype != torch.bfloat16
        ):
            raise ValueError("Bias must be BF16 [local N] on the activation device")
        activation = quantize_activation(x, invocation.tier.activation)
        first, *residuals = invocation.tier.planes
        output = activation @ state.weights[first].T
        for name in residuals:
            output = output + activation @ state.weights[name].T
        if bias is not None:
            output = output + bias.float()
        return output.to(torch.bfloat16)
