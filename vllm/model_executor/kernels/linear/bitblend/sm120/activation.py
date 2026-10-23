# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch mixed MX quantization using FlashInfer's CuTe DSL primitives.

Reference: FlashInfer 0.7.2 quantization/kernels/mxfp{4,8}_quantize.py and
quantization_cute_dsl_utils.py (Apache-2.0, Copyright FlashInfer team).
"""

from dataclasses import dataclass
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8
from flashinfer.cute_dsl.fp4_common import (
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_u64,
)
from flashinfer.quantization.quantization_cute_dsl_utils import (
    bfloat2_hmax_reduce_to_f32,
    bfloat2_max_abs_4,
    bfloat2_max_abs_8,
    bfloat2x4_to_fp8x8_packed,
    compute_sf_index_swizzled_128x4_gpu,
    float_to_ue8m0_fast,
    process_mxfp4_block_bfloat,
    reduce_max_2threads,
    reduce_max_4threads,
    ue8m0_to_inv_scale_fast,
)


class _MixedQuantize:
    def __init__(self, k, high_threads, threads=256):
        self.k = k
        self.high_threads = high_threads
        self.threads = threads

    @cute.jit
    def __call__(self, x, groups, count: Int32, tasks: Int32, blocks: Int32, stream):
        self.kernel(x, groups, count, tasks).launch(
            grid=[blocks, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, groups: cute.Tensor, count: Int32, tasks: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        stride, _, _ = cute.arch.grid_dim()
        cols = self.k // 32
        tasks = cutlass.min(tasks, Int32(groups[count - 1, 4]))
        while task < tasks:
            group = Int32(0)
            while group + 1 < count and task >= groups[group, 4]:
                group += 1
            first = Int32(0)
            if group > 0:
                first = Int32(groups[group - 1, 4])
            local = task - first
            start = groups[group, 0]
            m = Int32(groups[group, 1])
            output = groups[group, 2]
            scales = cute.make_tensor(
                cute.make_ptr(Uint8, groups[group, 3], cute.AddressSpace.gmem),
                cute.make_layout((((m + 127) // 128) * 128 * cols,)),
            )
            high = groups[group, 5] != 0
            # A whole CTA takes the same format branch; no mixed-format warp.
            if high:
                sf = (
                    local * (self.threads // self.high_threads)
                    + tid // self.high_threads
                )
                row = sf // cols
                col = sf % cols
                lane = tid % self.high_threads
                if row < m:
                    elements = 32 // self.high_threads
                    offset = col * 32 + lane * elements
                    ptr = get_ptr_as_int64(x, Int32(0)) + (start + Int64(row)) * Int64(
                        self.k * 2
                    )
                    inp = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((self.k,)),
                    )
                    v0, v1, v2, v3 = ld_global_v4_u32(get_ptr_as_int64(inp, offset))
                    maximum = bfloat2_hmax_reduce_to_f32(
                        bfloat2_max_abs_4(v0, v1, v2, v3)
                    )
                    if cutlass.const_expr(self.high_threads == 2):
                        v4, v5, v6, v7 = ld_global_v4_u32(
                            get_ptr_as_int64(inp, offset + 8)
                        )
                        maximum = bfloat2_hmax_reduce_to_f32(
                            bfloat2_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7)
                        )
                        maximum = reduce_max_2threads(maximum)
                    else:
                        maximum = reduce_max_4threads(maximum)
                    scale = float_to_ue8m0_fast(maximum * Float32(1.0 / 448.0))
                    inv = ue8m0_to_inv_scale_fast(scale)
                    out = cute.make_tensor(
                        cute.make_ptr(
                            Uint8,
                            output + Int64(row) * self.k,
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        ),
                        cute.make_layout((self.k,)),
                    )
                    packed = bfloat2x4_to_fp8x8_packed(v0, v1, v2, v3, inv)
                    st_global_u64(get_ptr_as_int64(out, offset), packed)
                    if cutlass.const_expr(self.high_threads == 2):
                        packed_hi = bfloat2x4_to_fp8x8_packed(v4, v5, v6, v7, inv)
                        st_global_u64(get_ptr_as_int64(out, offset + 8), packed_hi)
                    if lane == 0:
                        scales[
                            compute_sf_index_swizzled_128x4_gpu(row, col, Int32(cols))
                        ] = scale.to(Uint8)
            else:
                sf = local * self.threads + tid
                row = sf // cols
                col = sf % cols
                if row < m:
                    ptr = get_ptr_as_int64(x, Int32(0)) + (start + Int64(row)) * Int64(
                        self.k * 2
                    )
                    inp = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((self.k,)),
                    )
                    _, scale, lo, hi = process_mxfp4_block_bfloat(inp, col * 32)
                    out = cute.make_tensor(
                        cute.make_ptr(
                            Uint8,
                            output + Int64(row) * (self.k // 2),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        ),
                        cute.make_layout((self.k // 2,)),
                    )
                    st_global_u64(get_ptr_as_int64(out, col * 16), lo)
                    st_global_u64(get_ptr_as_int64(out, col * 16 + 8), hi)
                    scales[
                        compute_sf_index_swizzled_128x4_gpu(row, col, Int32(cols))
                    ] = scale
            # Parallelize padded rows as well, as in the official swizzled path.
            pad = local * self.threads + tid
            leader = True
            if high:
                pad = (
                    local * (self.threads // self.high_threads)
                    + tid // self.high_threads
                )
                leader = tid % self.high_threads == 0
            end = ((m + 127) // 128) * 128 * cols
            if pad >= m * cols and pad < end and leader:
                scales[
                    compute_sf_index_swizzled_128x4_gpu(
                        pad // cols, pad % cols, Int32(cols)
                    )
                ] = Uint8(0)
            task += stride


@cache
def quantize_kernel(k: int, high_threads: int = 4):
    """Compile once per K/cooperation size; span counts and M stay dynamic."""
    return cute.compile(
        _MixedQuantize(k, high_threads),
        cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (cute.sym_int(),), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            Int64, (cute.sym_int(), 6), stride_order=(1, 0), assumed_align=16
        ),
        Int32(1),
        Int32(1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@dataclass
class ActivationCall:
    """Prepared outputs and descriptor ownership; run is allocation free."""

    kernel: object
    x: torch.Tensor
    descriptors: torch.Tensor
    groups: tuple
    tasks: int
    blocks: int

    def run(self):
        self.kernel(
            self.x.view(-1), self.descriptors, len(self.groups), self.tasks, self.blocks
        )
        return self.groups


def prepare_activation(x: torch.Tensor, spans) -> ActivationCall:
    """Prepare contiguous BF16 [M,K] and ordered nonempty (tier,M) spans."""
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_contiguous()
        or x.shape[1] % 128
        or min(x.shape) <= 0
        or x.data_ptr() % 16
    ):
        raise ValueError(
            "Mixed activation requires aligned contiguous CUDA BF16 [M,K], K % 128 = 0"
        )
    spans = tuple(spans)
    if not spans or any(
        t not in ("low", "mid", "high") or type(m) is not int or m <= 0
        for t, m in spans
    ):
        raise ValueError("Mixed activation requires nonempty known tier spans")
    m, k = x.shape
    if (
        sum(rows for _, rows in spans) != m
        or ((m + 127) // 128) * 128 * (k // 32) >= 2**31
    ):
        raise ValueError(
            "Mixed activation spans must cover input within int32 scale indexing"
        )
    high_threads = (
        2 if sum(rows for t, rows in spans if t == "high") * (k // 32) >= 65536 else 4
    )
    groups, descriptors = [], []
    start, tasks = 0, 0
    with torch.cuda.device(x.device):
        for tier, rows in spans:
            high = tier == "high"
            activation = torch.empty(
                (rows, k if high else k // 2),
                device=x.device,
                dtype=torch.float8_e4m3fn if high else torch.uint8,
            )
            scale = torch.empty(
                ((rows + 127) // 128) * 128 * (k // 32),
                device=x.device,
                dtype=torch.uint8,
            )
            units = 256 // high_threads if high else 256
            padded = (rows + 127) // 128 * 128
            tasks += (padded * (k // 32) + units - 1) // units
            descriptors.append(
                [start, rows, activation.data_ptr(), scale.data_ptr(), tasks, int(high)]
            )
            groups.append((tier, activation, scale, rows))
            start += rows
        host = torch.tensor(descriptors, dtype=torch.int64, pin_memory=True)
        descriptor = host.to(x.device, non_blocking=True)
        kernel = quantize_kernel(k, high_threads)
        blocks = min(
            tasks, torch.cuda.get_device_properties(x.device).multi_processor_count * 4
        )
        return ActivationCall(kernel, x, descriptor, tuple(groups), tasks, blocks)
