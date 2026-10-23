# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol

from vllm.model_executor.layers.quantization.bitblend.profile import (
    ProfileSpec,
    TierSpec,
)

if TYPE_CHECKING:
    import torch

    from vllm.model_executor.layers.quantization.bitblend.runtime import (
        BatchExecutionPlan,
    )


@dataclass(frozen=True)
class LayerGeometry:
    input_size: int
    output_size: int
    local_input_size: int
    output_partitions: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            min(self.input_size, self.output_size, self.local_input_size) <= 0
            or not self.output_partitions
            or min(self.output_partitions) <= 0
            or self.local_input_size > self.input_size
            or self.local_output_size > self.output_size
        ):
            raise ValueError("Invalid BitBlend layer geometry")

    @property
    def local_output_size(self) -> int:
        return sum(self.output_partitions)


@dataclass(frozen=True)
class BackendRequest:
    profile: ProfileSpec
    geometry: LayerGeometry
    tiers: tuple[str, ...]
    device: str
    dtype: str
    mixed_batch: bool = False
    graph_mode: bool = False


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason: str = ""


@dataclass(frozen=True)
class PreparedLayer:
    """Backend-owned packed weights and stable descriptors; no shared scratch."""

    layout: str
    state: Any
    request: BackendRequest


class MixedGraphWorkspace(Protocol):
    """Runner-owned storage; the backend owns device descriptors and launches.

    Updates and forwards are serialized on the runner's compute stream. Captured
    addresses/capacities stay stable when tier counts change between replays.
    """

    def update(self, plan: "BatchExecutionPlan") -> None: ...

    def run(
        self,
        prepared: PreparedLayer,
        x: "torch.Tensor",
        bias: "torch.Tensor | None",
    ) -> "torch.Tensor": ...


@dataclass(frozen=True)
class KernelInvocation:
    tier: TierSpec
    shape: tuple[int, ...]
    graph_mode: bool = False


class BitBlendBackend(ABC):
    """A run shares activation quantization and combines independently scaled planes.

    Implementations own tactic selection and temporary allocations. Mixed graph
    storage belongs to the runner-owned backend workspace. Runtime layouts,
    dtypes, shape constraints and dependency probes belong in supports/prepare.
    Built-in implementations are discovered lazily during backend selection.
    """

    supports_graphs: bool = False
    supports_mixed_graphs: bool = False

    @classmethod
    def check_support(cls, request: BackendRequest) -> SupportResult:
        """Combine execution capabilities with backend-specific device checks."""
        if request.graph_mode:
            if not cls.supports_graphs:
                return SupportResult(False, "CUDA Graph execution is not supported")
            if request.mixed_batch and not cls.supports_mixed_graphs:
                return SupportResult(
                    False, "mixed CUDA Graph execution is not supported"
                )
        return cls.supports(request)

    @classmethod
    @abstractmethod
    def supports(cls, request: BackendRequest) -> SupportResult:
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self, layer: "torch.nn.Module", request: BackendRequest
    ) -> PreparedLayer:
        raise NotImplementedError

    @abstractmethod
    def resolve_invocation(
        self,
        prepared: PreparedLayer,
        tier: TierSpec,
        shape: tuple[int, ...],
        *,
        graph_mode: bool = False,
    ) -> KernelInvocation:
        """Use prepared tactics or bounded rules; never benchmark on this path."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        prepared: PreparedLayer,
        x: "torch.Tensor",
        invocation: KernelInvocation,
        bias: "torch.Tensor | None",
        workspace: Any,
    ) -> "torch.Tensor":
        raise NotImplementedError

    def run_grouped(self, prepared, x, spans, bias, workspace):
        """Numerical backends may execute spans separately for comparison."""
        output = x.new_empty((x.shape[0], prepared.request.geometry.local_output_size))
        for span in spans:
            part = x[span.start : span.stop]
            tier = prepared.request.profile.tier(span.tier)
            invocation = self.resolve_invocation(prepared, tier, tuple(part.shape))
            output[span.start : span.stop].copy_(
                self.run(prepared, part, invocation, bias, workspace)
            )
        return output

    def create_graph_workspace(
        self,
        layers: list[PreparedLayer],
        max_tokens: int,
        device: "torch.device",
    ) -> MixedGraphWorkspace:
        """Allocate stable mixed metadata before native model graph capture."""
        raise NotImplementedError("This BitBlend backend does not support mixed graphs")


_backends: dict[str, type[BitBlendBackend]] = {}
_BUILTIN_BACKENDS = {
    "sm120": "vllm.model_executor.kernels.linear.bitblend.sm120.backend.Sm120Backend",
}


def register_backend(name: str, backend: type[BitBlendBackend]) -> None:
    """Register during worker startup, before model loading."""
    if not name or name in ("auto", "reference") or name in _backends:
        raise ValueError(f"Invalid or duplicate BitBlend backend: {name!r}")
    if not issubclass(backend, BitBlendBackend):
        raise TypeError("Expected a BitBlendBackend subclass")
    _backends[name] = backend


def select_backend(name: str, request: BackendRequest) -> BitBlendBackend:
    # The numerical oracle is explicit-only, never an automatic fallback.
    if name == "reference":
        from vllm.model_executor.kernels.linear.bitblend.reference import (
            ReferenceBackend,
        )

        support = ReferenceBackend.check_support(request)
        if not support.supported:
            raise NotImplementedError(
                f"Unsupported BitBlend reference: {support.reason}"
            )
        return ReferenceBackend()
    candidates = (
        dict.fromkeys((*_backends, *_BUILTIN_BACKENDS))
        if name == "auto"
        else {name: None}
    )
    reasons = []
    for candidate in candidates:
        if candidate not in _backends and candidate in _BUILTIN_BACKENDS:
            module, attribute = _BUILTIN_BACKENDS[candidate].rsplit(".", 1)
            register_backend(candidate, getattr(import_module(module), attribute))
        cls = _backends.get(candidate)
        if cls is None:
            reasons.append(f"{candidate}: not registered")
            continue
        support = cls.check_support(request)
        if support.supported:
            return cls()
        reasons.append(f"{candidate}: {support.reason}")
    detail = "; ".join(reasons) or "no hardware kernels are implemented"
    raise NotImplementedError(
        f"No BitBlend backend for {request.profile.name}: {detail}"
    )
