# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MX activation quantization and three-tier GEMM dispatch."""

from dataclasses import dataclass

import torch

from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
    TIERS,
    GroupedConfig,
    HighWarpConfig,
    MidWarpConfig,
    grouped_default_config,
)
from vllm.model_executor.kernels.linear.bitblend.sm120.jit import (
    grouped_module,
    warp_module,
)


def warp_tile(m: int, n: int, k: int, *, tier: str = "high") -> tuple[int, int]:
    """Choose a compiled token/channel tile without tuning in forward."""
    if tier == "mid" and m >= 512 and n >= 2048:
        return (128, 64)
    if m <= 32:
        return (8 if m <= 8 else 16 if m <= 16 else 32, 64)
    if m <= 64 and n >= 2048:
        return (32, 64)
    # Wider tiles amortize plane loads when the output grid has enough CTAs.
    if m >= 1024 and n >= 2048:
        return (128, 128)
    return (64, 64)


def warp_splits(
    m: int, n: int, k: int, tile: tuple[int, int] | None = None, *, tier: str = "high"
) -> int:
    """Bound split-K for the unified dual-plane kernel's output grid."""
    if m <= 0 or n <= 0 or k <= 0 or n % 8 or k % 128:
        raise ValueError(
            "Dual-plane GEMM requires positive M/N/K, N divisible by 8 and K by 128"
        )
    tokens, channels = tile or warp_tile(m, n, k, tier=tier)
    blocks = ((m + tokens - 1) // tokens) * ((n + channels - 1) // channels)
    splits = 1
    # Avoid a reduction when the output grid already exposes enough parallelism.
    while blocks * splits < 96 and splits < 8 and splits * 2 <= k // 128:
        splits *= 2
    return splits


def quantize(x: torch.Tensor, tier: str) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp4_quantize, mxfp8_quantize

    if tier in ("low", "mid"):
        return mxfp4_quantize(x, backend="cute-dsl")
    if tier == "high":
        return mxfp8_quantize(x, backend="cute-dsl")
    raise ValueError(f"Unknown MX tier: {tier}")


def gemm(
    tier: str,
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weights: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    *,
    config: HighWarpConfig | MidWarpConfig | None = None,
    tactic: int = 0,
) -> torch.Tensor:
    m = activation.shape[0]
    n, packed_k = weights["p0"].shape
    k = packed_k * 2
    if tier == "low":
        from flashinfer import mm_fp4

        return mm_fp4(
            activation,
            weights["p0"].T,
            activation_scale.view((m + 127) // 128 * 128, k // 32),
            scales["p0"].view((n + 127) // 128 * 128, k // 32).T,
            block_size=32,
            use_nvfp4=False,
            backend="cudnn",
            out_dtype=torch.bfloat16,
        )
    if tier not in ("mid", "high"):
        raise ValueError(f"Unknown MX tier: {tier}")
    config_type = MidWarpConfig if tier == "mid" else HighWarpConfig
    if (config is not None and type(config) is not config_type) or tactic != 0:
        raise ValueError(f"{tier} requires {config_type.__name__} and tactic=0")
    selected = config or config_type(
        *warp_tile(m, n, k, tier=tier), warp_splits(m, n, k, tier=tier)
    )
    selected.validate_shape(m, n, k)
    splits = selected.splits
    output = torch.empty((m, n), dtype=torch.bfloat16, device=activation.device)
    scratch = torch.empty(
        0 if splits == 1 else splits * m * n,
        dtype=torch.float32,
        device=activation.device,
    )
    extension = warp_module(tier, *selected.variant)
    residual = "p1" if tier == "mid" else "p2"
    getattr(extension, f"{tier}_warp")(
        activation.contiguous(),
        activation_scale.reshape(-1).contiguous(),
        weights["p0"],
        scales["p0"],
        weights[residual],
        scales[residual],
        output,
        scratch,
        splits,
        selected.tokens,
        selected.channels,
    )
    return output


@dataclass
class GroupedCall:
    """Own pointer targets until the asynchronous grouped launch completes.

    Construction performs validation and descriptor upload; run() is allocation
    free, so kernel benchmarks can capture it without timing preparation.
    """

    module: object
    descriptors: torch.Tensor
    activations: tuple
    arguments: tuple
    output: torch.Tensor

    def run(self):
        self.module.grouped_gemm(self.descriptors, *self.arguments)
        return self.output


def prepare_grouped(
    groups,
    weights: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    *,
    config: GroupedConfig | None = None,
    bias: torch.Tensor | None = None,
) -> GroupedCall:
    """Build a grouped call from ordered (tier, quantized X, scale, M) tuples.

    Groups retain their local scale origin, including repeated tiers and M tails.
    The trusted device descriptor ABI is internal; callers use this validator.
    """
    if not groups:
        raise ValueError("Grouped GEMM requires at least one nonempty group")
    config = config or grouped_default_config(groups)
    n, packed_k = weights["p0"].shape
    k = packed_k * 2
    device = weights["p0"].device
    if device.type != "cuda":
        raise ValueError("Grouped GEMM requires CUDA tensors")
    for plane, dtype, width in (
        ("p0", torch.uint8, k // 2),
        ("p1", torch.uint8, k // 2),
        ("p2", torch.float8_e4m3fn, k),
    ):
        w, s = weights[plane], scales[plane]
        if (
            w.device != device
            or s.device != device
            or w.dtype != dtype
            or s.dtype != torch.uint8
            or tuple(w.shape) != (n, width)
            or s.numel() != ((n + 127) // 128) * 128 * (k // 32)
            or not w.is_contiguous()
            or not s.is_contiguous()
        ):
            raise ValueError(f"Invalid grouped weight/scale layout: {plane}")
    if bias is not None and (
        bias.device != device
        or bias.dtype != torch.bfloat16
        or tuple(bias.shape) != (n,)
        or not bias.is_contiguous()
    ):
        raise ValueError("Grouped bias must be contiguous BF16 [N]")
    rows, tasks, scratch_size = 0, 0, 0
    descriptors, keepalive = [], []
    for tier, activation, scale, m in groups:
        if tier not in TIERS:
            raise ValueError("Unknown grouped MX tier")
        selected = getattr(config, tier)
        selected.validate_shape(m, n, k)
        dtype = torch.float8_e4m3fn if tier == "high" else torch.uint8
        width = k if tier == "high" else k // 2
        if (
            activation.device != device
            or scale.device != device
            or activation.dtype != dtype
            or scale.dtype != torch.uint8
            or activation.ndim != 2
            or activation.shape[0] < m
            or activation.shape[1] != width
            or scale.numel() != ((m + 127) // 128) * 128 * (k // 32)
            or not activation.is_contiguous()
            or not scale.is_contiguous()
        ):
            raise ValueError("Invalid grouped activation/scale layout")
        mt = (m + selected.tokens - 1) // selected.tokens
        nt = (n + selected.channels - 1) // selected.channels
        group_tasks = mt * nt * selected.splits
        if group_tasks >= 2**31:
            raise ValueError("Grouped task count exceeds signed 32-bit indexing")
        tasks += group_tasks
        descriptors.append(
            [
                activation.data_ptr(),
                scale.data_ptr(),
                m,
                rows,
                TIERS.index(tier),
                selected.splits,
                scratch_size,
                tasks,
                *_fast_divisor(mt * nt),
                *_fast_divisor(mt if config.m_major else nt),
            ]
        )
        rows += m
        if selected.splits > 1:
            scratch_size += selected.splits * m * n
        keepalive.extend((activation, scale))
    with torch.cuda.device(device):
        module = grouped_module(config.variant)
        order_grouped_descriptors(descriptors, config, k)
        host = torch.tensor(descriptors, dtype=torch.int64, pin_memory=True)
        descriptor = host.to(device, non_blocking=True)
        output = torch.empty((rows, n), device=device, dtype=torch.bfloat16)
        scratch = torch.empty(scratch_size, device=device, dtype=torch.float32)
        bias_tensor = bias if bias is not None else output.new_empty(0)
        arguments = (
            weights["p0"],
            scales["p0"],
            weights["p1"],
            scales["p1"],
            weights["p2"],
            scales["p2"],
            output,
            scratch,
            bias_tensor,
            tasks,
            config.waves,
            config.m_major,
            scratch_size > 0,
        )
        return GroupedCall(module, descriptor, tuple(keepalive), arguments, output)


def order_grouped_descriptors(rows, config: GroupedConfig, k: int) -> None:
    """Order heavy jobs first while preserving output and scratch offsets."""
    if not config.heavy_first:
        return
    rows.sort(
        key=lambda row: (
            (k // getattr(config, TIERS[row[4]]).block_k)
            * (1 if row[4] == 0 else 2)
            * getattr(config, TIERS[row[4]]).tokens
            / (row[5] * getattr(config, TIERS[row[4]]).token_warps)
        ),
        reverse=True,
    )
    tasks = 0
    for row in rows:
        tasks += row[8] * row[5]
        row[7] = tasks


def _fast_divisor(divisor: int):
    """CUTLASS FastDivmod parameters for nonnegative signed 32-bit tasks."""
    if divisor == 1:
        return 1, 0, 0
    p = 31 + (divisor - 1).bit_length()
    return divisor, ((1 << p) + divisor - 1) // divisor, p - 32
