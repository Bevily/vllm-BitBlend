# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable compile configurations and constrained SM120 search space."""

from dataclasses import dataclass, replace

M_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

WARP_TILES = tuple((t, c) for t in (8, 16, 32, 64, 128) for c in (64, 128))


@dataclass(frozen=True)
class WarpConfig:
    tokens: int
    channels: int
    splits: int
    block_k: int = 128
    stages: int = 2
    token_warps: int = 1
    matrix_load: int = 0
    register_prefetch: int = 0
    vector_epilogue: int = 0

    @property
    def variant(self):
        return (
            self.block_k,
            self.stages,
            self.token_warps,
            self.matrix_load,
            self.register_prefetch,
            self.vector_epilogue,
        )

    def shared_bytes(self, tier: str):
        scale_slots = max(4, self.block_k // 32)
        if tier not in ("low", "mid", "high"):
            raise ValueError("Unknown MX tier")
        planes = 1 if tier == "low" else 2
        payload = self.block_k * (
            (self.channels + self.tokens) / 2
            if tier == "low"
            else self.channels + self.tokens / 2
            if tier == "mid"
            else 1.5 * self.channels + self.tokens
        )
        return max(
            int(
                self.stages
                * (payload + scale_slots * (planes * self.channels + self.tokens))
            ),
            self.tokens * self.channels * 4 if self.vector_epilogue else 0,
        )

    def __post_init__(self):
        if any(
            type(v) is not int
            for v in (self.tokens, self.channels, self.splits, *self.variant)
        ):
            raise ValueError("Dual-plane tile and splits must be integers")
        if (
            self.block_k not in (64, 128, 256)
            or self.stages not in range(1, 9)
            or self.token_warps not in (1, 2, 4)
            or any(
                v not in (0, 1)
                for v in (
                    self.matrix_load,
                    self.register_prefetch,
                    self.vector_epilogue,
                )
            )
            or self.tokens % (8 * self.token_warps)
        ):
            raise ValueError("Unsupported dual-plane pipeline configuration")
        if (self.tokens, self.channels) not in WARP_TILES or self.splits < 1:
            raise ValueError("Unsupported unified dual-plane configuration")

    def validate_shape(self, m: int, n: int, k: int):
        if (
            min(m, n, k) <= 0
            or n % 8
            or k % 128
            or (k % self.block_k or self.splits > k // self.block_k)
        ):
            raise ValueError(
                "Unified dual-plane configuration is invalid for this shape"
            )


@dataclass(frozen=True)
class HighWarpConfig(WarpConfig):
    """FP8 activation with FP4/FP8 residual planes."""


@dataclass(frozen=True)
class MidWarpConfig(WarpConfig):
    """Packed FP4 activation with two FP4 weight planes."""


def warp_candidates(tier: str, m: int, n: int, k: int):
    """Enumerate the supported tile product and every legal K split."""
    if tier not in ("mid", "high"):
        raise ValueError("Dual-plane candidates require Mid or High")
    config_type = MidWarpConfig if tier == "mid" else HighWarpConfig
    config_type(8, 64, 1).validate_shape(m, n, k)
    return tuple(
        config_type(tokens, channels, splits, block_k, stages, q, matrix)
        for block_k in (64, 128, 256)
        if k % block_k == 0
        for stages in range(1, 9)
        for q in (1, 2, 4)
        for matrix in ((0,) if tier == "mid" else (0, 1))
        for tokens, channels in WARP_TILES
        if tokens % (8 * q) == 0
        if config_type(tokens, channels, 1, block_k, stages, q, matrix).shared_bytes(
            tier
        )
        <= 101376
        for splits in range(1, k // block_k + 1)
    )


@dataclass(frozen=True)
class LowWarpConfig(WarpConfig):
    """Packed FP4 activation with one FP4 weight plane."""


@dataclass(frozen=True)
class GroupedConfig:
    """One launch's compatible tier specializations and task ordering."""

    low: LowWarpConfig = LowWarpConfig(8, 64, 1)
    mid: MidWarpConfig = MidWarpConfig(8, 64, 1)
    high: HighWarpConfig = HighWarpConfig(8, 64, 1)
    waves: int = 0
    m_major: bool = False
    heavy_first: bool = False
    min_blocks: int = 1

    def __post_init__(self):
        if (
            type(self.low) is not LowWarpConfig
            or type(self.mid) is not MidWarpConfig
            or type(self.high) is not HighWarpConfig
            or type(self.waves) is not int
            or self.waves not in (0, 1, 2, 4)
            or type(self.m_major) is not bool
            or type(self.heavy_first) is not bool
            or type(self.min_blocks) is not int
            or self.min_blocks not in (1, 2, 3, 4, 6, 8)
        ):
            raise ValueError("Invalid grouped configuration")
        threads = {c.channels * 2 * c.token_warps for c in self.tiers}
        if len(threads) != 1 or max(threads) > 1024:
            raise ValueError("Grouped tiers must have the same CTA thread count")
        if max(threads) * self.min_blocks > 1536:
            raise ValueError("Grouped residency target exceeds SM120 thread capacity")
        if any(c.shared_bytes(t) > 101376 for t, c in zip(TIERS, self.tiers)):
            raise ValueError("Grouped configuration exceeds SM120 shared memory")

    @property
    def tiers(self):
        return self.low, self.mid, self.high

    @property
    def variant(self):
        # Split counts and task ordering do not change compiled instructions.
        return (
            *((c.tokens, c.channels, *c.variant) for c in self.tiers),
            self.min_blocks,
        )

    @classmethod
    def from_dict(cls, value):
        return cls(
            **{
                tier: kind(**value[tier])
                for tier, kind in zip(
                    TIERS, (LowWarpConfig, MidWarpConfig, HighWarpConfig)
                )
            },
            waves=value.get("waves", 0),
            m_major=value.get("m_major", False),
            heavy_first=value.get("heavy_first", False),
            min_blocks=value.get("min_blocks", 1),
        )


TIERS = ("low", "mid", "high")


GROUPED_DEFAULTS = tuple(
    GroupedConfig(
        **{
            tier: kind(tokens, 64, 1, token_warps=warps)
            for tier, kind in zip(TIERS, (LowWarpConfig, MidWarpConfig, HighWarpConfig))
        }
    )
    for tokens, warps in ((8, 1), (32, 1), (64, 1), (128, 2))
)


def grouped_default_config(groups):
    """Bounded precompiled defaults; exact verified tactics take precedence."""
    largest = max(group[3] for group in groups)
    return GROUPED_DEFAULTS[
        0 if largest <= 8 else 1 if largest <= 32 else 2 if largest <= 128 else 3
    ]


def grouped_tier_candidates(tier: str, m: int, n: int, k: int, threads: int):
    """Lazy full per-tier space, constrained by a launch's common CTA size.

    Search each common thread family and coordinate-sweep these candidates;
    independent optima cannot simply be combined into a grouped launch.
    """
    if tier not in TIERS or threads not in (128, 256, 512, 1024):
        raise ValueError("Invalid grouped tier or CTA size")
    kind = dict(low=LowWarpConfig, mid=MidWarpConfig, high=HighWarpConfig)[tier]
    kind(8, 64, 1).validate_shape(m, n, k)
    for block_k in (64, 128, 256):
        if k % block_k:
            continue
        for stages in range(1, 9):
            for tokens, channels in WARP_TILES:
                for q in (1, 2, 4):
                    if channels * 2 * q != threads or tokens % (8 * q):
                        continue
                    for matrix in (0, 1) if tier == "high" else (0,):
                        for prefetch in (0, 1):
                            for epilogue in (0, 1):
                                base = kind(
                                    tokens,
                                    channels,
                                    1,
                                    block_k,
                                    stages,
                                    q,
                                    matrix,
                                    prefetch,
                                    epilogue,
                                )
                                if base.shared_bytes(tier) > 101376:
                                    continue
                                for splits in range(1, k // block_k + 1):
                                    yield replace(base, splits=splits)


def grouped_coordinate_candidates(base: WarpConfig, tier: str, m: int, n: int, k: int):
    """Round-robin parameter axes so a finite budget reaches different mechanisms.

    Repeated sweeps compose changes; exhaustive per-tier products remain available
    through grouped_tier_candidates. Neither search guarantees a global optimum.
    """
    from itertools import zip_longest

    threads = base.channels * 2 * base.token_warps
    axes = [
        [dict(tokens=t) for t in (32, 64, 128, 16, 8)],
        [dict(block_k=b) for b in (256, 64, 128)],
        [dict(stages=s) for s in (3, 1, 4, 2, 5, 6, 7, 8)],
        [dict(register_prefetch=p) for p in (1, 0)],
        [dict(vector_epilogue=e) for e in (1, 0)],
        [dict(matrix_load=load) for load in ((1, 0) if tier == "high" else (0,))],
        [dict(splits=s) for s in range(1, k // base.block_k + 1)],
        [
            dict(channels=c, token_warps=q)
            for c in (64, 128)
            for q in (1, 2, 4)
            if c * 2 * q == threads
        ],
    ]
    seen = {base}
    for changes in zip_longest(*axes):
        for change in changes:
            if change is None:
                continue
            try:
                candidate = replace(base, **change)
                candidate.validate_shape(m, n, k)
            except ValueError:
                continue
            if candidate not in seen and candidate.shared_bytes(tier) <= 101376:
                seen.add(candidate)
                yield candidate
