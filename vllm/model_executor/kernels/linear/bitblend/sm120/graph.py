# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-owned mixed descriptors shared by native full-model CUDA graphs."""

from dataclasses import replace

import torch

from vllm.model_executor.kernels.linear.bitblend.sm120.activation import quantize_kernel
from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
    TIERS,
    grouped_default_config,
)
from vllm.model_executor.kernels.linear.bitblend.sm120.jit import grouped_module
from vllm.model_executor.kernels.linear.bitblend.sm120.ops import (
    _fast_divisor,
    order_grouped_descriptors,
)
from vllm.model_executor.layers.quantization.bitblend.runtime import BatchExecutionPlan


class MixedGraphWorkspace:
    """Reuse quantization storage sequentially, with one metadata copy per forward.

    Graphs capture descriptor addresses and launch capacities, never tier counts.
    GEMM outputs remain graph-pool allocations: residuals must not alias reusable
    workspace. The owning runner serializes updates and model forwards on its
    compute stream, just like its native input and attention metadata buffers.
    """

    def __init__(self, geometries, max_tokens, device, max_spans=12, database=None):
        if max_tokens <= 0 or max_spans <= 0:
            raise ValueError("Mixed graph capacities must be positive")
        self.geometries = tuple(sorted(set(geometries)))  # (local K, local N)
        if not self.geometries:
            raise ValueError("Mixed graph workspace requires prepared SM120 layers")
        self.max_tokens, self.max_spans = max_tokens, max_spans
        self.device = torch.device(device)
        self.num_tokens = 0
        self.database = database
        self.configurations = {}
        ks = sorted({k for k, _ in self.geometries})
        self.quantized, self.scales, self.high_threads = {}, {}, {}
        self.quant_descriptors, self.gemm_descriptors = {}, {}
        fields = len(ks) * max_spans * 6 + len(self.geometries) * max_spans * 14
        self.descriptors = torch.empty(fields, device=device, dtype=torch.int64)
        self.device = self.descriptors.device
        offset = 0
        with torch.cuda.device(device):
            self.blocks = (
                torch.cuda.get_device_properties(device).multi_processor_count * 4
            )
            for k in ks:
                scale_rows = max_tokens + 127 * max_spans
                if k % 128 or scale_rows * (k // 32) >= 2**31:
                    raise ValueError("Mixed graph capacity exceeds MX scale indexing")
                self.quantized[k] = torch.empty(
                    max_tokens * k, device=device, dtype=torch.uint8
                )
                self.scales[k] = torch.empty(
                    scale_rows * (k // 32), device=device, dtype=torch.uint8
                )
                self.high_threads[k] = 2 if max_tokens * (k // 32) >= 65536 else 4
                quantize_kernel(k, self.high_threads[k])
                size = max_spans * 6
                self.quant_descriptors[k] = self.descriptors[
                    offset : offset + size
                ].view(max_spans, 6)
                offset += size
            for k, n in self.geometries:
                if n % 8:
                    raise ValueError("Mixed graph local N must be divisible by 8")
                size = max_spans * 14
                self.gemm_descriptors[k, n] = self.descriptors[
                    offset : offset + size
                ].view(max_spans, 14)
                offset += size

    def config_for(self, m, n, k):
        """Resolve once per capture shape; never depend on current tier ratios."""
        key = m, n, k
        if key not in self.configurations:
            selected = (
                self.database.lookup_graph(m, n, k, self.max_spans)
                if self.database is not None
                else None
            )
            self.configurations[key] = selected or replace(
                grouped_default_config(((None, None, None, m),)), waves=1
            )
        return self.configurations[key]

    def update(self, plan: BatchExecutionPlan):
        """Upload all geometry descriptors before entering the model graph."""
        if plan.num_tokens > self.max_tokens or len(plan.spans) > self.max_spans:
            raise ValueError("BitBlend token plan exceeds mixed graph capacity")
        data, groups = [], {}
        for k, activation in self.quantized.items():
            rows, payload_offset, scale_offset, tasks = [], 0, 0, 0
            entries = []
            for span in plan.spans:
                m = span.stop - span.start
                high = span.tier == "high"
                padded = (m + 127) // 128 * 128
                units = 256 // self.high_threads[k] if high else 256
                tasks += (padded * (k // 32) + units - 1) // units
                a = activation.data_ptr() + payload_offset
                s = self.scales[k].data_ptr() + scale_offset
                rows.append([span.start, m, a, s, tasks, int(high)])
                entries.append((span, a, s))
                payload_offset += m * (k if high else k // 2)
                scale_offset += padded * (k // 32)
            rows.extend([[0, 0, 0, 0, tasks, 0]] * (self.max_spans - len(rows)))
            data.extend(v for row in rows for v in row)
            groups[k] = entries
        for k, n in self.geometries:
            config = self.config_for(plan.num_tokens, n, k)
            max_splits = max(c.splits for c in config.tiers)
            rows, tasks = [], 0
            for span, a, s in groups[k]:
                c = getattr(config, span.tier)
                m = span.stop - span.start
                mt = (m + c.tokens - 1) // c.tokens
                nt = (n + c.channels - 1) // c.channels
                tasks += mt * nt * c.splits
                rows.append(
                    [
                        a,
                        s,
                        m,
                        span.start,
                        TIERS.index(span.tier),
                        c.splits,
                        span.start * n * max_splits,
                        tasks,
                        *_fast_divisor(mt * nt),
                        *_fast_divisor(mt if config.m_major else nt),
                    ]
                )
            order_grouped_descriptors(rows, config, k)
            if tasks >= 2**31:
                raise ValueError("Mixed graph capacity exceeds GEMM task indexing")
            rows.extend(
                [[0, 0, 0, 0, 0, 1, 0, tasks, 1, 0, 0, 1, 0, 0]]
                * (self.max_spans - len(rows))
            )
            data.extend(v for row in rows for v in row)
        # A fresh pinned allocation prevents CPU writes racing an earlier async
        # upload. PyTorch's pinned allocator retains it until the copy completes.
        host = torch.tensor(data, dtype=torch.int64, pin_memory=True)
        self.descriptors.copy_(host, non_blocking=True)
        self.num_tokens = plan.num_tokens

    def run(self, prepared, x, bias):
        """Capture/run a linear using the runner's current descriptor addresses."""
        m, k = x.shape
        state = prepared.state
        n = prepared.request.geometry.local_output_size
        if (
            m != self.num_tokens
            or x.device != self.device
            or not x.is_contiguous()
            or x.data_ptr() % 16
        ):
            raise ValueError("Mixed graph input does not match the prepared plan")
        if bias is not None and (
            bias.shape != (n,)
            or bias.dtype != x.dtype
            or bias.device != x.device
            or not bias.is_contiguous()
        ):
            raise ValueError("Mixed graph bias must be contiguous BF16 [N]")
        config = self.config_for(m, n, k)
        threads = self.high_threads[k]
        groups = min(m, self.max_spans)
        padded = ((m + 127) // 128 + groups - 1) * 128
        units = 256 // threads
        tasks = (padded * (k // 32) + units - 1) // units
        quantize_kernel(k, threads)(
            x.view(-1),
            self.quant_descriptors[k],
            self.max_spans,
            tasks,
            min(tasks, self.blocks),
        )
        output = x.new_empty((m, n))
        splits = max(c.splits for c in config.tiers)
        scratch = output.new_empty(
            m * n * splits if splits > 1 else 0, dtype=torch.float32
        )
        bias_tensor = bias if bias is not None else output.new_empty(0)
        # Bound any partition into <=max_spans, including unequal tier tiles.
        smallest_tile = min(c.tokens for c in config.tiers)
        tasks = (
            ((m + smallest_tile - 1) // smallest_tile + groups - 1)
            * max((n + c.channels - 1) // c.channels for c in config.tiers)
            * splits
        )
        grouped_module(config.variant).grouped_gemm(
            self.gemm_descriptors[k, n],
            state.weights["p0"],
            state.scales["p0"],
            state.weights["p1"],
            state.scales["p1"],
            state.weights["p2"],
            state.scales["p2"],
            output,
            scratch,
            bias_tensor,
            tasks,
            config.waves,
            config.m_major,
            splits > 1,
        )
        return output
