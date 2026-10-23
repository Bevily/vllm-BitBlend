# SM120 BitBlend kernels

`mid_warp.cu` and `high_warp.cu` instantiate the same `dual_warp.cuh` mainloop,
loader, epilogue and deterministic split-K reduction. Every Mid/High shape uses
its respective specialization; there is no legacy-kernel fallback. Low uses
FlashInfer/cuDNN. The GB10 reference repository is not a runtime dependency.

## Official references and retained BitBlend semantics

- CUTLASS 4.5 `cute/arch/mma_sm120.hpp` and `cute/atom/mma_traits_sm120.hpp`:
  native block-scaled MMA, register coordinates and scale distribution.
- FlashInfer `gemm/kernels/sm12x_cute/independent_small.py`: operand-swapped
  narrow tiles, `cp.async` double buffering and Mid's `ldmatrix.x4` / `.x2`.
- FlashInfer `gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py`: compile tile
  variants within SM120 resource limits, then choose a shape-compatible tile.
- The GB10 compact dual-plane implementation: load one activation stream for
  both weight planes, keep independent plane scales and accumulate in FP32.

Mid computes `A4*P0 + A4*P1` with two native `m16n8k64` FP4/FP4 MMAs. All
operands remain packed FP4, including shared memory and MMA registers. Each
instruction consumes two independent block-32 UE8M0 scales per operand.
High computes `A8*P0 + A8*P2` with `m16n8k32` F4/F8 and F8/F8 MMAs. Its
register expansion follows the mixed MMA encoding; no expanded weight copy
is stored. Both tiers round once to BF16 after FP32 accumulation. The official
NVFP4 kernel's block-16/E4M3 scaling is **not** substituted for MX block-32/UE8M0.
Activation quantization remains FlashInfer CuTe-DSL with ceil scaling.
Mixed batches use `sm120/activation.py`: one CuTe DSL launch dispatches whole
CTAs to MXFP4 or MXFP8 while reusing FlashInfer's vector loads, reductions,
conversions and 128x4 scale indexing. Per-span buffers remain independent;
padding scales are zeroed in the same launch. The quantizer is not fused into
the GEMM mainloop. Fixed-tier quantization still uses FlashInfer's public API.

## Implemented parameter domain

Both tiers support ten token/channel pairs: `tokens={8,16,32,64,128}` and
`channels={64,128}`. The pipeline chooses `block_k={64,128,256}`,
`stages=1..8`, and `token_warps={1,2,4}`. Token warps partition each tile's
token panels; CTA threads are `2*channels*token_warps`. Token panels must divide
evenly between token warps. High additionally chooses direct shared gathers or
`ldmatrix.x4/.x2`; Mid always uses matrix loads. Both support optional register
prefetch of the next MMA weight fragment and a vectorized epilogue.

`cp.async` copies 16-byte payloads and aligned 4-byte scale groups. K=64
stages select the appropriate half of an aligned scale group, and use narrower
XOR swizzles. A ring of up to eight stages overlaps copies and compute, with
explicit tail draining and barriers before buffer reuse.

With `L=block_k` and `Q=max(4,L/32)`, one stage occupies
`L*(channels+tokens/2)+Q*(2*channels+tokens)` bytes for Mid, or
`L*(1.5*channels+tokens)+Q*(2*channels+tokens)` for High. The vector epilogue
reuses stage storage after all copies and MMA reads complete. Its allocation
is `max(stages*stage_bytes,4*tokens*channels)`, not the sum. It stores BF16
pairs or FP32 split partials in aligned four-element vectors. Resource queries
use compiled register/local-memory attributes and CUDA's occupancy API.

M/N tails are masked; K must be divisible by 128 and by the selected block K,
and N by 8. Any integer split count in `[1,K/block_k]` is supported, including
uneven partitions. Split=1 writes BF16 directly. Larger splits use
`splits*M*N` FP32 scratch elements and one deterministic reduction, without
atomics. Inference uses an exact-shape verified tactic or the bounded heuristic
in `sm120/ops.py`; it never tunes during forward. Preparation preloads every
selected specialization before graph capture. TMA and producer/consumer warp
specialization are not implemented. Fixed-tier launches use ordinary grids. See
`docs/bitblend-sm120-database.md` for staged search and verification.

Mixed execution uses `grouped.cu` with the same device mainloop, including a
one-plane Low specialization. Its CTA-uniform tier dispatch writes directly to
the original output row range. Ordinary and persistent round-robin grids,
heavy-first group order, two M/N traversal orders and residency launch bounds
are tunable. Shared memory is the maximum tier requirement, not the sum.
CUTLASS `FastDivmod` parameters are prepared on the host; per-group task counts
must fit signed 32-bit indexing. Split-K adds one grouped deterministic reduction
that also accepts reordered descriptors. The scalar epilogue skips the final
shared-memory barrier; persistent tasks still synchronize before storage reuse.
For mixed full-model graphs, `sm120/graph.py` owns runner-lifetime payload/scale
arenas and stable per-geometry descriptors. One metadata upload precedes each
forward. Captured quantization/GEMM launches read dynamic task counts from the
descriptor tails, bounded by capture capacities. Native token buckets share
these buffers; GEMM outputs remain distinct graph-pool allocations. This path
loads exact-token-bucket `graph_selected` tactics independently of eager span
tables, with bounded defaults on misses. Tuned tiers may use distinct tiles and
Split-K counts; capture reserves the worst-case grid and FP32 scratch for any
ratio. The benchmark's `--search-mixed-graph` / `--verify-mixed-graph` modes time
this same path while updating ratios without recapture.

Compilation uses FlashInfer's bundled CUTLASS headers and TVM-FFI. Source
fingerprints cover both `.cu` entry points and the shared `.cuh`; packaged
builds include these files. CUDA Graph replay is handled by native vLLM and
PyTorch graph memory pools, without a separate per-layer workspace manager.
