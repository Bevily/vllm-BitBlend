# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from vllm.model_executor.layers.quantization.bitblend.profile import get_profile


@dataclass(frozen=True)
class ExecutionPlan:
    """Explicit fixed-tier override for numerical checks and direct layer calls."""

    profile: str
    tier: str

    def __post_init__(self) -> None:
        get_profile(self.profile).tier(self.tier)


@dataclass(frozen=True)
class TierSpan:
    tier: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.stop) is not int
            or self.start < 0
            or self.stop <= self.start
        ):
            raise ValueError("BitBlend token spans must be nonempty half-open ranges")


@dataclass(frozen=True)
class BatchExecutionPlan:
    """Device-independent token ranges in the final model input order."""

    profile: str
    spans: tuple[TierSpan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spans", tuple(self.spans))
        spec = get_profile(self.profile)
        end = 0
        for span in self.spans:
            spec.tier(span.tier)
            if span.start != end:
                raise ValueError("BitBlend token spans must cover a contiguous prefix")
            end = span.stop
        if not self.spans:
            raise ValueError("BitBlend batch plan must contain tokens")

    @property
    def num_tokens(self) -> int:
        return self.spans[-1].stop


_plan: ContextVar[ExecutionPlan | BatchExecutionPlan | None] = ContextVar(
    "bitblend_plan", default=None
)
_workspace: ContextVar[Any] = ContextVar("bitblend_workspace", default=None)


def get_execution_plan() -> ExecutionPlan | BatchExecutionPlan | None:
    return _plan.get()


def get_workspace() -> Any:
    return _workspace.get()


@contextmanager
def execution_context(
    plan: ExecutionPlan | BatchExecutionPlan, *, workspace: Any = None
) -> Iterator[None]:
    """Bind one forward; graph adapters update device metadata before replay."""
    token = _plan.set(plan)
    workspace_token = _workspace.set(workspace)
    try:
        yield
    finally:
        _workspace.reset(workspace_token)
        _plan.reset(token)
