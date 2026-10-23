# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt native request batches to device-independent BitBlend token plans."""

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

from vllm.model_executor.layers.quantization.bitblend.runtime import (
    BatchExecutionPlan,
    TierSpan,
    execution_context,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.bitblend.config import BitBlendConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


def reorder_tier_batch(
    config: "BitBlendConfig",
    batch: "InputBatch",
    requests: dict[str, "CachedRequestState"],
    scheduled: dict[str, int],
    decode_threshold: int | None,
) -> None:
    """Group tiers while preserving the attention backend's four regions."""
    order = {tier.name: i for i, tier in enumerate(config.profile.tiers)}

    def key(req_id: str) -> tuple[int, int]:
        index = batch.req_id_to_index[req_id]
        region = 0
        if decode_threshold is not None:
            computed = batch.num_computed_tokens_cpu[index]
            if computed == 0:
                region = 3
            elif scheduled[req_id] > decode_threshold:
                region = 2
            elif computed < batch.num_prompt_tokens[index]:
                region = 1
        params = requests[req_id].sampling_params
        tier = config.request_tier(params.extra_args if params else None)
        return region, order[tier]

    desired = sorted(batch.req_ids, key=key)
    for index, req_id in enumerate(desired):
        current = batch.req_id_to_index[req_id]
        if current != index:
            batch.swap_states(index, current)


def execution_context_for_batch(
    config: "BitBlendConfig | None",
    batch: "InputBatch",
    requests: dict[str, "CachedRequestState"],
    scheduled: dict[str, int],
    num_tokens: int,
    workspace: Any = None,
) -> AbstractContextManager[None]:
    if config is None:
        return nullcontext()
    spans: list[TierSpan] = []
    offset = 0
    for req_id in batch.req_ids:
        count = scheduled[req_id]
        if count <= 0:
            raise ValueError("BitBlend batch contains an unscheduled request")
        params = requests[req_id].sampling_params
        tier = config.request_tier(params.extra_args if params else None)
        if spans and spans[-1].tier == tier:
            previous = spans.pop()
            spans.append(TierSpan(tier, previous.start, offset + count))
        else:
            spans.append(TierSpan(tier, offset, offset + count))
        offset += count
    if not spans or offset > num_tokens:
        raise ValueError("BitBlend batch plan does not match model token count")
    if offset < num_tokens:
        last = spans.pop()
        spans.append(TierSpan(last.tier, last.start, num_tokens))
    plan = BatchExecutionPlan(config.profile.name, tuple(spans))
    if workspace is not None:
        workspace.update(plan)
    return execution_context(plan, workspace=workspace)


def execution_context_for_dummy(
    config: "BitBlendConfig | None", num_tokens: int, workspace: Any
) -> AbstractContextManager[None]:
    """Warm up and capture the same mixed linears used by actual requests."""
    if config is None or workspace is None:
        return nullcontext()
    plan = BatchExecutionPlan(
        config.profile.name, (TierSpan(config.tier, 0, num_tokens),)
    )
    workspace.update(plan)
    return execution_context(plan, workspace=workspace)


def create_graph_workspace(model, max_tokens: int, device):
    """Delegate storage/layout decisions to the model's prepared backend."""
    backend, layers = None, []
    for layer in model.modules():
        prepared = getattr(layer, "bitblend_prepared", None)
        if prepared is not None:
            candidate = layer.bitblend_backend
            if backend is not None and type(candidate) is not type(backend):
                raise ValueError("Mixed graphs require a common BitBlend backend")
            backend = candidate
            layers.append(prepared)
    if backend is None:
        raise ValueError("Mixed graphs require prepared BitBlend layers")
    for prepared in layers:
        support = backend.check_support(prepared.request)
        if not support.supported:
            raise ValueError(f"Unsupported BitBlend graph workspace: {support.reason}")
    return backend.create_graph_workspace(layers, max_tokens, device)
