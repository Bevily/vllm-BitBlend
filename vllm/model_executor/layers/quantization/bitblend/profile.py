# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Literal

Tier = Literal["low", "mid", "high"]


@dataclass(frozen=True)
class PlaneSpec:
    name: str
    bits: int
    scale_block: int | None
    residual_of: str | None = None

    def shapes(self, n: int, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
        alignment = self.scale_block or 1
        if n <= 0 or k <= 0 or k % alignment or k * self.bits % 8:
            raise ValueError(f"Invalid {self.name} geometry: {(n, k)}")
        return (n, k * self.bits // 8), (n, k // alignment if self.scale_block else 1)


@dataclass(frozen=True)
class TierSpec:
    name: Tier
    planes: tuple[str, ...]
    activation: str


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    planes: tuple[PlaneSpec, ...]
    tiers: tuple[TierSpec, ...]

    def tier(self, name: str) -> TierSpec:
        for tier in self.tiers:
            if tier.name == name:
                return tier
        raise ValueError(f"Tier {name!r} is not defined by {self.name}")

    def required_planes(self, tiers: tuple[str, ...]) -> tuple[PlaneSpec, ...]:
        required = {plane for tier in tiers for plane in self.tier(tier).planes}
        return tuple(plane for plane in self.planes if plane.name in required)


MX_PROFILE = ProfileSpec(
    "mx_residual_3plane",
    (
        PlaneSpec("p0", 4, 32),
        PlaneSpec("p1", 4, 32, "p0"),
        PlaneSpec("p2", 8, 32, "p0"),
    ),
    (
        TierSpec("low", ("p0",), "mxfp4_block32"),
        TierSpec("mid", ("p0", "p1"), "mxfp4_block32"),
        TierSpec("high", ("p0", "p2"), "mxfp8_block32"),
    ),
)
FP8_PROFILE = ProfileSpec(
    "fp8_residual_2plane",
    (PlaneSpec("h0", 8, None), PlaneSpec("h1", 8, None, "h0")),
    (
        TierSpec("low", ("h0",), "fp8_per_token"),
        TierSpec("high", ("h0", "h1"), "fp8_per_token"),
    ),
)


def get_profile(name: str) -> ProfileSpec:
    for profile in (MX_PROFILE, FP8_PROFILE):
        if profile.name == name:
            return profile
    raise ValueError(f"Unknown BitBlend profile: {name!r}")
