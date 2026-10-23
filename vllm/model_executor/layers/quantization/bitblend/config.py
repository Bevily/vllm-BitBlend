# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.bitblend.profile import (
    MX_PROFILE,
    get_profile,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def validate_mixed_batch_config(config: "VllmConfig") -> None:
    """Validate shared runner constraints; backend selection checks hardware."""
    quant = config.quant_config
    if not isinstance(quant, BitBlendConfig) or not quant.mixed_batch:
        return
    if config.device_config.device_type != "cuda":
        raise ValueError("BitBlend mixed_batch requires the V1 GPU model runner")
    model, parallel = config.model_config, config.parallel_config
    if not model.enforce_eager:
        from vllm.config import CompilationMode, CUDAGraphMode

        compilation = config.compilation_config
        if (
            compilation.mode != CompilationMode.NONE
            or compilation.cudagraph_mode
            not in (
                CUDAGraphMode.NONE,
                CUDAGraphMode.FULL,
                CUDAGraphMode.FULL_DECODE_ONLY,
            )
        ):
            raise ValueError(
                "BitBlend mixed_batch graphs require compilation mode=0 "
                "and cudagraph_mode=FULL or FULL_DECODE_ONLY"
            )
    if config.use_v2_model_runner:
        raise ValueError("BitBlend mixed_batch requires VLLM_USE_V2_MODEL_RUNNER=0")
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.enable_dbo
    ):
        raise ValueError("BitBlend mixed_batch currently requires a single GPU")
    if (
        model.runner_type != "generate"
        or model.is_encoder_decoder
        or model.is_moe
        or model.supports_multimodal_inputs
        or config.speculative_config is not None
        or config.lora_config is not None
        or config.kv_transfer_config is not None
        or config.ec_transfer_config is not None
    ):
        raise ValueError(
            "BitBlend mixed_batch supports dense decoder-only text generation "
            "without speculative decoding, LoRA or KV/encoder transfer"
        )


class BitBlendConfig(QuantizationConfig):
    """Version-1 packed BitBlend checkpoint configuration.

    All declared planes are loaded, including for fixed-tier execution.
    """

    def __init__(
        self,
        profile: str = MX_PROFILE.name,
        tier: str = "low",
        backend: str = "auto",
        ignored_layers: tuple[str, ...] = (),
        mixed_batch: bool = False,
    ) -> None:
        super().__init__()
        self.profile = get_profile(profile)
        self.profile.tier(tier)
        if not backend:
            raise ValueError("BitBlend backend must not be empty")
        self.tier = tier
        self.backend = backend
        self.ignored_layers = tuple(ignored_layers)
        if type(mixed_batch) is not bool:
            raise ValueError("BitBlend mixed_batch must be a boolean")
        self.mixed_batch = mixed_batch

    def request_tier(self, extra_args: dict[str, Any] | None) -> str:
        tier = (extra_args or {}).get("bitblend_tier", self.tier)
        if not isinstance(tier, str):
            raise ValueError("bitblend_tier must be a tier name")
        self.profile.tier(tier)
        if not self.mixed_batch and tier != self.tier:
            raise ValueError("Per-request BitBlend tiers require mixed_batch=true")
        return tier

    @classmethod
    def get_name(cls) -> str:
        return "bitblend"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Hardware constraints belong to the backend, not the checkpoint.
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["bitblend_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitBlendConfig":
        if config.get("schema_version", 1) != 1:
            raise ValueError("Unsupported BitBlend schema_version")
        ignored = config.get("ignored_layers", ())
        if not isinstance(ignored, (list, tuple)) or not all(
            isinstance(x, str) for x in ignored
        ):
            raise ValueError("ignored_layers must be a sequence of layer names")
        return cls(
            profile=config.get("profile", MX_PROFILE.name),
            tier=config.get("tier", "low"),
            backend=config.get("backend", "auto"),
            ignored_layers=tuple(ignored),
            mixed_batch=config.get("mixed_batch", False),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if not isinstance(layer, LinearBase):
            return None
        if prefix.split(".")[-1] == "lm_head" or is_layer_skipped(
            prefix, list(self.ignored_layers), self.packed_modules_mapping
        ):
            return UnquantizedLinearMethod()
        from .linear import BitBlendLinearMethod

        return BitBlendLinearMethod(self)


class BitBlendMXConfig(BitBlendConfig):
    """Compatibility name for the legacy three-plane MX checkpoint tensors."""

    @classmethod
    def get_name(cls) -> str:
        return "bitblend_mx"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitBlendConfig":
        result = super().from_config(config)
        if result.profile != MX_PROFILE:
            raise ValueError("bitblend_mx requires mx_residual_3plane")
        return result
