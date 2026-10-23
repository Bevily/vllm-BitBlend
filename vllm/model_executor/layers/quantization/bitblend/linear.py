# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.model_executor.kernels.linear import select_bitblend_backend
from vllm.model_executor.kernels.linear.bitblend.base import (
    BackendRequest,
    LayerGeometry,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.bitblend.runtime import (
    BatchExecutionPlan,
    get_execution_plan,
    get_workspace,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)

if TYPE_CHECKING:
    from .config import BitBlendConfig


@register_weight_loader_v2_supported_method
class BitBlendLinearMethod(LinearMethodBase):
    def __init__(self, config: "BitBlendConfig"):
        self.config = config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        geometry = LayerGeometry(
            input_size,
            output_size,
            input_size_per_partition,
            tuple(output_partition_sizes),
        )
        n, k = geometry.local_output_size, geometry.local_input_size
        loader = extra_weight_attrs["weight_loader"]
        layer.bitblend_geometry = geometry
        for plane in self.config.profile.planes:
            weight_shape, scale_shape = plane.shapes(n, k)
            attrs = dict(input_dim=1, output_dim=0, weight_loader=loader)
            data = torch.empty(weight_shape, dtype=torch.uint8)
            weight = (
                PackedvLLMParameter(data=data, packed_dim=1, packed_factor=2, **attrs)
                if plane.bits == 4
                else ModelWeightParameter(data=data, **attrs)
            )
            scale_dtype = torch.uint8 if plane.scale_block else torch.float32
            scale_data = torch.empty(scale_shape, dtype=scale_dtype)
            scale = (
                GroupQuantScaleParameter(data=scale_data, **attrs)
                if plane.scale_block
                else ChannelQuantScaleParameter(
                    data=scale_data, output_dim=0, weight_loader=loader
                )
            )
            layer.register_parameter(f"weight_{plane.name}", weight)
            layer.register_parameter(f"weight_scale_{plane.name}", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = getattr(layer, f"weight_{self.config.profile.planes[0].name}")
        config = get_current_vllm_config_or_none()
        graph_mode = (
            config is not None
            and not config.model_config.enforce_eager
            and config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        )
        request = BackendRequest(
            self.config.profile,
            layer.bitblend_geometry,
            tuple(tier.name for tier in self.config.profile.tiers),
            str(weight.device),
            "bfloat16",
            mixed_batch=self.config.mixed_batch,
            graph_mode=graph_mode,
        )
        backend = select_bitblend_backend(self.config.backend, request)
        layer.bitblend_prepared = backend.prepare(layer, request)
        layer.bitblend_backend = backend

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan = get_execution_plan()
        if plan is not None and plan.profile != self.config.profile.name:
            raise ValueError("BitBlend execution plan does not match loaded weights")
        if not hasattr(layer, "bitblend_backend"):
            raise RuntimeError("BitBlend weights have not been prepared")
        backend = layer.bitblend_backend
        prepared = layer.bitblend_prepared
        if isinstance(plan, BatchExecutionPlan):
            flat = x.reshape(-1, x.shape[-1])
            if flat.shape[0] != plan.num_tokens:
                raise ValueError("BitBlend token plan does not match linear input")
            if self.config.mixed_batch:
                output = backend.run_grouped(
                    prepared, flat, plan.spans, bias, get_workspace()
                )
                return output.reshape(*x.shape[:-1], output.shape[-1])
            tier_name = plan.spans[0].tier
        else:
            tier_name = plan.tier if plan else self.config.tier
        tier = self.config.profile.tier(tier_name)
        invocation = backend.resolve_invocation(prepared, tier, tuple(x.shape))
        return backend.run(prepared, x, invocation, bias, get_workspace())
