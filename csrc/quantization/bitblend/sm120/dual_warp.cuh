// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Register coordinates follow CUTLASS mma_traits_sm120.hpp / mma_traits_sm80.hpp.
// The narrow, operand-swapped tile and async staging follow FlashInfer's
// sm12x_cute/independent_small.py; this kernel uses MX scales and two planes.

#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>

#include "cute/arch/mma_sm120.hpp"
#include "tvm_ffi_utils.h"

namespace bitblend::BITBLEND_NAMESPACE {

#ifndef BITBLEND_BLOCK_K
#define BITBLEND_BLOCK_K 128
#define BITBLEND_STAGES 2
#define BITBLEND_TOKEN_WARPS 1
#define BITBLEND_MATRIX_LOAD 0
#endif
#ifndef BITBLEND_REGISTER_PREFETCH
#define BITBLEND_REGISTER_PREFETCH 0
#define BITBLEND_VECTOR_EPILOGUE 0
#endif
constexpr int kBlockK = BITBLEND_BLOCK_K;
constexpr int kStages = BITBLEND_STAGES;
constexpr int kTokenWarps = BITBLEND_TOKEN_WARPS;
constexpr int kScales = kBlockK < 128 ? 4 : kBlockK / 32;
#ifndef BITBLEND_LOW
#define BITBLEND_LOW 0
#endif
constexpr bool kLow = BITBLEND_LOW;
constexpr bool kMid = BITBLEND_MID;
constexpr int kActivationBytes = kMid ? kBlockK / 2 : kBlockK;
constexpr int kResidualBytes = kMid ? kBlockK / 2 : kBlockK;

// Byte layouts equivalent to CuTe Swizzle<3,4,3> (F8) and <2,4,3> (F4).
// Each permutation preserves the 16-byte alignment of cp.async destinations.
__device__ __forceinline__ int col8(int row, int col) {
  return col ^ ((row & (kBlockK < 128 ? 3 : 7)) * 16);
}

__device__ __forceinline__ int col4(int row, int col) {
  return col ^ (((row >> 1) & (kBlockK < 128 ? 1 : 3)) * 16);
}

template <int Tokens, int Channels>
struct alignas(16) Stage {
  uint8_t p0[Channels][kBlockK / 2];
#if !BITBLEND_LOW
  uint8_t p2[Channels][kResidualBytes];
#endif
  uint8_t x[Tokens][kActivationBytes];
  uint8_t s0[Channels][kScales];
#if !BITBLEND_LOW
  uint8_t s2[Channels][kScales];
#endif
  uint8_t sx[Tokens][kScales];
};

__device__ __forceinline__ int scale_offset(int row, int group, int k) {
  return (row / 128) * (k / 32) * 128 + (group / 4) * 512 +
         (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}

template <int Bytes>
__device__ __forceinline__ void copy_async(void* dst, void const* src, bool valid) {
  auto address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  if constexpr (Bytes == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                 "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::
                 "r"(address), "l"(src), "r"(valid ? 4 : 0) : "memory");
  }
}

template <int Tokens, int Channels>
__device__ __forceinline__ void load_stage(
    Stage<Tokens, Channels>& s, uint8_t const* x, uint8_t const* sx, uint8_t const* p0,
    uint8_t const* s0, uint8_t const* p2, uint8_t const* s2,
    int m, int n, int k, int m0, int n0, int tile) {
  int tid = threadIdx.x;
  constexpr int Threads = Channels * 2 * kTokenWarps;
  int offset = tile * kBlockK;
#pragma unroll
  for (int i = tid; i < Channels * kBlockK / 32; i += Threads) {
    int row = i / (kBlockK / 32);
    int col = (i % (kBlockK / 32)) * 16;
    bool valid = n0 + row < n;
    auto src = valid ? p0 + int64_t(n0 + row) * (k / 2) + offset / 2 + col : p0;
    copy_async<16>(&s.p0[row][col4(row, col)], src, valid);
  }
#if !BITBLEND_LOW
#pragma unroll
  for (int i = tid; i < Channels * kResidualBytes / 16; i += Threads) {
    int row = i / (kResidualBytes / 16);
    int col = (i % (kResidualBytes / 16)) * 16;
    bool valid = n0 + row < n;
    auto src = valid ? p2 + int64_t(n0 + row) * (kMid ? k / 2 : k) +
                           offset / (kMid ? 2 : 1) + col
                     : p2;
    copy_async<16>(&s.p2[row][(kMid ? col4(row, col) : col8(row, col))], src, valid);
  }
#endif
#pragma unroll
  for (int i = tid; i < Tokens * kActivationBytes / 16; i += Threads) {
    int row = i / (kActivationBytes / 16);
    int col = (i % (kActivationBytes / 16)) * 16;
    bool valid = m0 + row < m;
    auto src = valid ? x + int64_t(m0 + row) * (kMid ? k / 2 : k) +
                           offset / (kMid ? 2 : 1) + col
                     : x;
    copy_async<16>(&s.x[row][(kMid ? col4(row, col) : col8(row, col))], src, valid);
  }
#pragma unroll
  for (int i = tid; i < Channels * (kScales / 4); i += Threads) {
    int row = i / (kScales / 4), chunk = i % (kScales / 4);
    bool valid = n0 + row < n;
    int index = valid ? scale_offset(n0 + row, (offset / 32 & ~3) + chunk * 4, k) : 0;
    copy_async<4>(&s.s0[row][chunk * 4], s0 + index, valid);
#if !BITBLEND_LOW
    copy_async<4>(&s.s2[row][chunk * 4], s2 + index, valid);
#endif
  }
#pragma unroll
  for (int i = tid; i < Tokens * (kScales / 4); i += Threads) {
    int row = i / (kScales / 4), chunk = i % (kScales / 4);
    bool valid = m0 + row < m;
    int index = valid ? scale_offset(m0 + row, (offset / 32 & ~3) + chunk * 4, k) : 0;
    copy_async<4>(&s.sx[row][chunk * 4], sx + index, valid);
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t unpack4(uint8_t const* ptr) {
  uint32_t v = *reinterpret_cast<uint16_t const*>(ptr);
  // Mixed F4/F8 MMA takes four F4 values in the high bits of four bytes.
  return ((v & 0xf) | ((v & 0xf0) << 4) | ((v & 0xf00) << 8) |
          ((v & 0xf000) << 12)) << 2;
}

__device__ __forceinline__ uint32_t word(uint8_t const* ptr) {
  return *reinterpret_cast<uint32_t const*>(ptr);
}

__device__ __forceinline__ void load4(uint8_t const* ptr, uint32_t& a0,
    uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  auto address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(address) : "memory");
}

__device__ __forceinline__ void load2(uint8_t const* ptr, uint32_t& b0, uint32_t& b1) {
  auto address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(b0), "=r"(b1) : "r"(address) : "memory");
}

using MmaPacked = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue8m0_t, 32>;

using Mma4 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e4m3_t, float,
    cutlass::float_ue8m0_t, 32>;
using Mma8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
    cutlass::float_ue8m0_t, 32>;

struct WeightRegisters {
  uint32_t a[4], w[4], s0, sr;
};

template <int Tokens, int Channels>
__device__ __forceinline__ WeightRegisters load_weights(
    Stage<Tokens, Channels> const& s, int warp, int lane, int kb, int scale_base) {
  WeightRegisters r;
  int row = warp * 16 + lane / 4;
  int scale_row = row + (lane % 2) * 8;
  if constexpr (kMid) {
    r.s0 = *reinterpret_cast<uint16_t const*>(&s.s0[scale_row][scale_base + kb * 2]);
#if !BITBLEND_LOW
    r.sr = *reinterpret_cast<uint16_t const*>(&s.s2[scale_row][scale_base + kb * 2]);
#endif
    int ar = warp * 16 + lane % 16;
    int ac = kb * 32 + (lane / 16) * 16;
    load4(&s.p0[ar][col4(ar, ac)], r.a[0], r.a[1], r.a[2], r.a[3]);
#if !BITBLEND_LOW
    load4(&s.p2[ar][col4(ar, ac)], r.w[0], r.w[1], r.w[2], r.w[3]);
#endif
  } else {
    r.s0 = s.s0[scale_row][scale_base + kb];
#if !BITBLEND_LOW
    r.sr = s.s2[scale_row][scale_base + kb];
#endif
    int col = kb * 32 + (lane % 4) * 4;
    r.a[0] = unpack4(&s.p0[row][col4(row, col / 2)]);
    r.a[1] = unpack4(&s.p0[row + 8][col4(row + 8, col / 2)]);
    r.a[2] = unpack4(&s.p0[row][col4(row, (col + 16) / 2)]);
    r.a[3] = unpack4(&s.p0[row + 8][col4(row + 8, (col + 16) / 2)]);
#if !BITBLEND_LOW
    if constexpr (BITBLEND_MATRIX_LOAD) {
      int ar = warp * 16 + lane % 16;
      int ac = kb * 32 + (lane / 16) * 16;
      load4(&s.p2[ar][col8(ar, ac)], r.w[0], r.w[1], r.w[2], r.w[3]);
    } else {
      r.w[0] = word(&s.p2[row][col8(row, col)]);
      r.w[1] = word(&s.p2[row + 8][col8(row + 8, col)]);
      r.w[2] = word(&s.p2[row][col8(row, col + 16)]);
      r.w[3] = word(&s.p2[row + 8][col8(row + 8, col + 16)]);
    }
#endif
  }
  return r;
}

template <int Tokens, int Channels>
constexpr int shared_bytes() {
  constexpr int stage_bytes = sizeof(Stage<Tokens, Channels>) * kStages;
  constexpr int output_bytes = BITBLEND_VECTOR_EPILOGUE ? Tokens * Channels * 4 : 0;
  return stage_bytes > output_bytes ? stage_bytes : output_bytes;
}

__device__ __forceinline__ __nv_bfloat16 finish(float value, __nv_bfloat16 const* bias,
                                               int channel) {
  auto rounded = __float2bfloat16_rn(value);
  return bias ? __float2bfloat16_rn(__bfloat162float(rounded) +
                                   __bfloat162float(bias[channel])) : rounded;
}

template <int Tokens, int Channels>
__device__ __forceinline__ void compute_tile(
    uint8_t const* __restrict__ x, uint8_t const* __restrict__ sx,
    uint8_t const* __restrict__ p0, uint8_t const* __restrict__ s0,
    uint8_t const* __restrict__ p2, uint8_t const* __restrict__ s2,
    __nv_bfloat16* __restrict__ out, float* __restrict__ partial,
    int m, int n, int k, int splits, int m0, int n0, int split,
    uint8_t* storage, __nv_bfloat16 const* bias = nullptr) {
  auto* stages = reinterpret_cast<Stage<Tokens, Channels>*>(storage);
  int lane = threadIdx.x % 32;
  int warp = (threadIdx.x / 32) % (Channels / 16);
  constexpr int Panels = Tokens / (8 * kTokenWarps);
  int panel0 = ((threadIdx.x / 32) / (Channels / 16)) * Panels;
  int group = lane / 4;
  int quarter = lane % 4;
  int begin = 0, end = k / kBlockK;
  if (splits != 1) {
    begin = end * split / splits;
    end = end * (split + 1) / splits;
  }
  float c[Panels][4] = {};
#pragma unroll
  for (int i = 0; i < kStages; ++i) {
    if (begin + i < end)
      load_stage(stages[i], x, sx, p0, s0, p2, s2, m, n, k, m0, n0, begin + i);
  }
  for (int tile = begin; tile < end; ++tile) {
    int buffer = (tile - begin) % kStages;
    int pending = min(kStages - 1, end - tile - 1);
    // PTX wait depth is immediate; the tail drains only outstanding groups.
    switch (pending) {
      case 7: asm volatile("cp.async.wait_group 7;" ::: "memory"); break;
      case 6: asm volatile("cp.async.wait_group 6;" ::: "memory"); break;
      case 5: asm volatile("cp.async.wait_group 5;" ::: "memory"); break;
      case 4: asm volatile("cp.async.wait_group 4;" ::: "memory"); break;
      case 3: asm volatile("cp.async.wait_group 3;" ::: "memory"); break;
      case 2: asm volatile("cp.async.wait_group 2;" ::: "memory"); break;
      case 1: asm volatile("cp.async.wait_group 1;" ::: "memory"); break;
      default: asm volatile("cp.async.wait_group 0;" ::: "memory"); break;
    }
    __syncthreads();
    Stage<Tokens, Channels> const& s = stages[buffer];
    int scale_base = kBlockK == 64 ? (tile % 2) * 2 : 0;
    constexpr int Steps = kBlockK / (kMid ? 64 : 32);
    WeightRegisters prefetched[2];
    if constexpr (BITBLEND_REGISTER_PREFETCH)
      prefetched[0] = load_weights(s, warp, lane, 0, scale_base);
#pragma unroll
    for (int kb = 0; kb < Steps; ++kb) {
      WeightRegisters r;
      if constexpr (BITBLEND_REGISTER_PREFETCH) {
        r = prefetched[kb % 2];
        if (kb + 1 < Steps)
          prefetched[(kb + 1) % 2] = load_weights(s, warp, lane, kb + 1, scale_base);
      } else {
        r = load_weights(s, warp, lane, kb, scale_base);
      }
#pragma unroll
      for (int panel = 0; panel < Panels; ++panel) {
        int token = (panel0 + panel) * 8 + group;
        uint32_t b0, b1;
        float* d = c[panel];
        if constexpr (kMid) {
          int br = (panel0 + panel) * 8 + lane % 8;
          int bc = kb * 32 + ((lane / 8) % 2) * 16;
          load2(&s.x[br][col4(br, bc)], b0, b1);
          uint16_t scale_x = *reinterpret_cast<uint16_t const*>(
              &s.sx[token][scale_base + kb * 2]);
          MmaPacked::fma(d[0], d[1], d[2], d[3], r.a[0], r.a[1], r.a[2], r.a[3], b0, b1,
                        d[0], d[1], d[2], d[3], r.s0, scale_x);
          if constexpr (!kLow)
            MmaPacked::fma(d[0], d[1], d[2], d[3], r.w[0], r.w[1], r.w[2], r.w[3], b0, b1,
                          d[0], d[1], d[2], d[3], r.sr, scale_x);
        } else {
          if constexpr (BITBLEND_MATRIX_LOAD) {
            int br = (panel0 + panel) * 8 + lane % 8;
            int bc = kb * 32 + ((lane / 8) % 2) * 16;
            load2(&s.x[br][col8(br, bc)], b0, b1);
          } else {
            int col = kb * 32 + quarter * 4;
            b0 = word(&s.x[token][col8(token, col)]);
            b1 = word(&s.x[token][col8(token, col + 16)]);
          }
          uint8_t scale_x = s.sx[token][scale_base + kb];
          Mma4::fma(d[0], d[1], d[2], d[3], r.a[0], r.a[1], r.a[2], r.a[3], b0, b1,
                    d[0], d[1], d[2], d[3], r.s0, scale_x);
          Mma8::fma(d[0], d[1], d[2], d[3], r.w[0], r.w[1], r.w[2], r.w[3], b0, b1,
                    d[0], d[1], d[2], d[3], r.sr, scale_x);
        }
      }
    }
    // A scalar epilogue does not reuse shared memory after the final tile.
    if (tile + 1 < end || BITBLEND_VECTOR_EPILOGUE) __syncthreads();
    if (tile + kStages < end)
      load_stage(stages[buffer], x, sx, p0, s0, p2, s2,
                 m, n, k, m0, n0, tile + kStages);
  }
#pragma unroll
  for (int panel = 0; panel < Panels; ++panel) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int token = m0 + (panel0 + panel) * 8 + quarter * 2 + i % 2;
      int channel = n0 + warp * 16 + group + (i / 2) * 8;
      if constexpr (BITBLEND_VECTOR_EPILOGUE) {
        reinterpret_cast<float*>(storage)[(token - m0) * Channels + channel - n0] = c[panel][i];
      } else if (token < m && channel < n) {
        int64_t index = int64_t(token) * n + channel;
        if (splits == 1) {
          out[index] = finish(c[panel][i], bias, channel);
        } else {
          partial[int64_t(split) * m * n + index] = c[panel][i];
        }
      }
    }
  }
  if constexpr (BITBLEND_VECTOR_EPILOGUE) {
    __syncthreads();
    constexpr int Threads = Channels * 2 * kTokenWarps;
    auto* values = reinterpret_cast<float const*>(storage);
    if (splits == 1) {
      for (int i = threadIdx.x; i < Tokens * Channels / 2; i += Threads) {
        int token = m0 + i * 2 / Channels, channel = n0 + i * 2 % Channels;
        if (token < m && channel < n)
          *reinterpret_cast<__nv_bfloat162*>(out + int64_t(token) * n + channel) =
              __halves2bfloat162(finish(values[i * 2], bias, channel),
                                 finish(values[i * 2 + 1], bias, channel + 1));
      }
    } else {
      for (int i = threadIdx.x; i < Tokens * Channels / 4; i += Threads) {
        int token = m0 + i * 4 / Channels, channel = n0 + i * 4 % Channels;
        if (token < m && channel < n)
          *reinterpret_cast<float4*>(partial + int64_t(split) * m * n +
                                    int64_t(token) * n + channel) =
              reinterpret_cast<float4 const*>(values)[i];
      }
    }
  }

}

#ifndef BITBLEND_DEVICE_ONLY
// Fixed-tier launches use the same tile routine as the heterogeneous kernel.
template <int Tokens, int Channels>
__global__ __launch_bounds__(Channels * 2 * kTokenWarps) void kernel(
    uint8_t const* x, uint8_t const* sx, uint8_t const* p0, uint8_t const* s0,
    uint8_t const* p2, uint8_t const* s2, __nv_bfloat16* out, float* partial,
    int m, int n, int k, int splits) {
  extern __shared__ __align__(16) uint8_t storage[];
  compute_tile<Tokens, Channels>(x, sx, p0, s0, p2, s2, out, partial,
      m, n, k, splits, blockIdx.y * Tokens, blockIdx.x * Channels, blockIdx.z, storage);
}

template <int Tokens, int Channels>
void launch(uint8_t const* x, uint8_t const* sx, uint8_t const* p0,
            uint8_t const* s0, uint8_t const* p2, uint8_t const* s2,
            __nv_bfloat16* out, float* partial, int m, int n, int k,
            int splits, cudaStream_t stream) {
  if constexpr (Tokens % (8 * kTokenWarps) != 0 ||
                shared_bytes<Tokens, Channels>() > 101376) {
    throw std::invalid_argument("Unsupported dual-plane resource configuration");
  } else {
  dim3 grid((n + Channels - 1) / Channels, (m + Tokens - 1) / Tokens, splits);
  kernel<Tokens, Channels><<<grid, Channels * 2 * kTokenWarps,
                            shared_bytes<Tokens, Channels>(), stream>>>(
      x, sx, p0, s0, p2, s2, out, partial, m, n, k, splits);
  }
}

template <int Tokens, int Channels>
void initialize_tile() {
  if constexpr (Tokens % (8 * kTokenWarps) == 0 &&
                shared_bytes<Tokens, Channels>() <= 101376) {
  auto status = cudaFuncSetAttribute(kernel<Tokens, Channels>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes<Tokens, Channels>());
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
}

__global__ void reduce(float const* partial, __nv_bfloat16* out,
                       int64_t elements, int splits) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    float sum = 0;
    for (int i = 0; i < splits; ++i) sum += partial[int64_t(i) * elements + index];
    out[index] = __float2bfloat16_rn(sum);
  }
}

#endif  // BITBLEND_DEVICE_ONLY
}  // namespace bitblend::BITBLEND_NAMESPACE

#ifndef BITBLEND_DEVICE_ONLY

namespace {

void initialize(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  using namespace bitblend::BITBLEND_NAMESPACE;
  initialize_tile<8, 64>();
  initialize_tile<16, 64>();
  initialize_tile<32, 64>();
  initialize_tile<8, 128>();
  initialize_tile<16, 128>();
  initialize_tile<32, 128>();
  initialize_tile<64, 64>();
  initialize_tile<64, 128>();
  initialize_tile<128, 64>();
  initialize_tile<128, 128>();
}

template <int Tokens, int Channels>
int64_t tile_resources() {
  using namespace bitblend::BITBLEND_NAMESPACE;
  if constexpr (Tokens % (8 * kTokenWarps) != 0 ||
                shared_bytes<Tokens, Channels>() > 101376) return 0;
  else {
    cudaFuncAttributes attr{};
    int blocks = 0;
    auto status = cudaFuncGetAttributes(&attr, kernel<Tokens, Channels>);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
        kernel<Tokens, Channels>, Channels * 2 * kTokenWarps,
        shared_bytes<Tokens, Channels>());
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return attr.numRegs + (int64_t(blocks) << 8) + (int64_t(attr.localSizeBytes) << 16);
  }
}

int64_t resources(int64_t tokens, int64_t channels) {
  if (tokens == 8 && channels == 64) return tile_resources<8, 64>();
  if (tokens == 8 && channels == 128) return tile_resources<8, 128>();
  if (tokens == 16 && channels == 64) return tile_resources<16, 64>();
  if (tokens == 16 && channels == 128) return tile_resources<16, 128>();
  if (tokens == 32 && channels == 64) return tile_resources<32, 64>();
  if (tokens == 32 && channels == 128) return tile_resources<32, 128>();
  if (tokens == 64 && channels == 64) return tile_resources<64, 64>();
  if (tokens == 64 && channels == 128) return tile_resources<64, 128>();
  if (tokens == 128 && channels == 64) return tile_resources<128, 64>();
  if (tokens == 128 && channels == 128) return tile_resources<128, 128>();
  throw std::invalid_argument("Unsupported dual-plane tile");
}

void dual_warp(TensorView x, TensorView sx, TensorView p0, TensorView s0,
               TensorView p2, TensorView s2, TensorView out, TensorView scratch,
               int64_t splits, int64_t tokens, int64_t channels) {
#if BITBLEND_MID
  CHECK_INPUT_AND_TYPE(x, dl_uint8);
#else
  CHECK_INPUT_AND_TYPE(x, dl_float8_e4m3fn);
#endif
  CHECK_INPUT_AND_TYPE(sx, dl_uint8);
  CHECK_INPUT_AND_TYPE(p0, dl_uint8);
  CHECK_INPUT_AND_TYPE(s0, dl_uint8);
#if BITBLEND_MID
  CHECK_INPUT_AND_TYPE(p2, dl_uint8);
#else
  CHECK_INPUT_AND_TYPE(p2, dl_float8_e4m3fn);
#endif
  CHECK_INPUT_AND_TYPE(s2, dl_uint8);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(scratch, dl_float32);
  TVM_FFI_ICHECK_EQ(x.ndim(), 2);
  TVM_FFI_ICHECK_EQ(p0.ndim(), 2);
  TVM_FFI_ICHECK_EQ(p2.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  int64_t m = x.size(0), k = x.size(1) * (BITBLEND_MID ? 2 : 1);
  int64_t n = p0.size(0);
  TVM_FFI_ICHECK_GT(m, 0);
  TVM_FFI_ICHECK_GT(n, 0);
  TVM_FFI_ICHECK_GT(k, 0);
  TVM_FFI_ICHECK_EQ(k % 128, 0);
  TVM_FFI_ICHECK_EQ(n % 8, 0);
  TVM_FFI_ICHECK_GE(splits, 1);
  TVM_FFI_ICHECK_LE(splits, k / bitblend::BITBLEND_NAMESPACE::kBlockK);
  TVM_FFI_ICHECK_EQ(p0.size(1) * 2, k);
  TVM_FFI_ICHECK_EQ(p2.size(0), n);
  TVM_FFI_ICHECK_EQ(p2.size(1) * (BITBLEND_MID ? 2 : 1), k);
  TVM_FFI_ICHECK_EQ(out.size(0), m);
  TVM_FFI_ICHECK_EQ(out.size(1), n);
  TVM_FFI_ICHECK_EQ(sx.numel(), ((m + 127) / 128) * 128 * (k / 32));
  TVM_FFI_ICHECK_EQ(s0.numel(), ((n + 127) / 128) * 128 * (k / 32));
  TVM_FFI_ICHECK_EQ(s2.numel(), s0.numel());
  TVM_FFI_ICHECK_GE(scratch.numel(), splits == 1 ? 0 : splits * m * n);
  for (auto t : {sx, p0, s0, p2, s2, out, scratch}) {
    TVM_FFI_ICHECK_EQ(t.device().device_id, x.device().device_id);
  }
  ffi::CUDADeviceGuard guard(x.device().device_id);
  auto stream = get_stream(x.device());
  using namespace bitblend::BITBLEND_NAMESPACE;
  auto dispatch = launch<8, 64>;
  if (channels == 64) {
    switch (tokens) {
      case 8: dispatch = launch<8, 64>; break;
      case 16: dispatch = launch<16, 64>; break;
      case 32: dispatch = launch<32, 64>; break;
      case 64: dispatch = launch<64, 64>; break;
      case 128: dispatch = launch<128, 64>; break;
      default: throw std::invalid_argument("Unsupported dual-plane token tile");
    }
  } else if (channels == 128) {
    switch (tokens) {
      case 8: dispatch = launch<8, 128>; break;
      case 16: dispatch = launch<16, 128>; break;
      case 32: dispatch = launch<32, 128>; break;
      case 64: dispatch = launch<64, 128>; break;
      case 128: dispatch = launch<128, 128>; break;
      default: throw std::invalid_argument("Unsupported dual-plane token tile");
    }
  } else {
    throw std::invalid_argument("Unsupported dual-plane channel tile");
  }
  dispatch(
      static_cast<uint8_t const*>(x.data_ptr()),
      static_cast<uint8_t const*>(sx.data_ptr()),
      static_cast<uint8_t const*>(p0.data_ptr()),
      static_cast<uint8_t const*>(s0.data_ptr()),
      static_cast<uint8_t const*>(p2.data_ptr()),
      static_cast<uint8_t const*>(s2.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<float*>(scratch.data_ptr()), m, n, k, splits, stream);
  if (splits > 1) {
    reduce<<<(m * n + 255) / 256, 256, 0, stream>>>(
        static_cast<float const*>(scratch.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()), m * n, splits);
  }
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

}  // namespace

#if BITBLEND_MID
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mid_warp, dual_warp);
#else
TVM_FFI_DLL_EXPORT_TYPED_FUNC(high_warp, dual_warp);
#endif
TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize, initialize);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(resources, resources);

#endif  // BITBLEND_DEVICE_ONLY
