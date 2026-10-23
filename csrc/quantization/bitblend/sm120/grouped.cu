// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Round-robin persistent task assignment follows CUTLASS grouped_scheduler.md.
// SM120 MMA, swizzles and async pipelines share the fixed-tier tile routines.
#define BITBLEND_DEVICE_ONLY
#include "cutlass/fast_math.h"

#define BITBLEND_NAMESPACE grouped_low
#define BITBLEND_MID 1
#define BITBLEND_LOW 1
#define BITBLEND_BLOCK_K GROUPED_LOW_BLOCK_K
#define BITBLEND_STAGES GROUPED_LOW_STAGES
#define BITBLEND_TOKEN_WARPS GROUPED_LOW_TOKEN_WARPS
#define BITBLEND_MATRIX_LOAD GROUPED_LOW_MATRIX_LOAD
#define BITBLEND_REGISTER_PREFETCH GROUPED_LOW_REGISTER_PREFETCH
#define BITBLEND_VECTOR_EPILOGUE GROUPED_LOW_VECTOR_EPILOGUE
#include "dual_warp.cuh"
#undef BITBLEND_NAMESPACE
#undef BITBLEND_MID
#undef BITBLEND_LOW
#undef BITBLEND_BLOCK_K
#undef BITBLEND_STAGES
#undef BITBLEND_TOKEN_WARPS
#undef BITBLEND_MATRIX_LOAD
#undef BITBLEND_REGISTER_PREFETCH
#undef BITBLEND_VECTOR_EPILOGUE

#define BITBLEND_NAMESPACE grouped_mid
#define BITBLEND_MID 1
#define BITBLEND_LOW 0
#define BITBLEND_BLOCK_K GROUPED_MID_BLOCK_K
#define BITBLEND_STAGES GROUPED_MID_STAGES
#define BITBLEND_TOKEN_WARPS GROUPED_MID_TOKEN_WARPS
#define BITBLEND_MATRIX_LOAD GROUPED_MID_MATRIX_LOAD
#define BITBLEND_REGISTER_PREFETCH GROUPED_MID_REGISTER_PREFETCH
#define BITBLEND_VECTOR_EPILOGUE GROUPED_MID_VECTOR_EPILOGUE
#include "dual_warp.cuh"
#undef BITBLEND_NAMESPACE
#undef BITBLEND_MID
#undef BITBLEND_LOW
#undef BITBLEND_BLOCK_K
#undef BITBLEND_STAGES
#undef BITBLEND_TOKEN_WARPS
#undef BITBLEND_MATRIX_LOAD
#undef BITBLEND_REGISTER_PREFETCH
#undef BITBLEND_VECTOR_EPILOGUE

#define BITBLEND_NAMESPACE grouped_high
#define BITBLEND_MID 0
#define BITBLEND_LOW 0
#define BITBLEND_BLOCK_K GROUPED_HIGH_BLOCK_K
#define BITBLEND_STAGES GROUPED_HIGH_STAGES
#define BITBLEND_TOKEN_WARPS GROUPED_HIGH_TOKEN_WARPS
#define BITBLEND_MATRIX_LOAD GROUPED_HIGH_MATRIX_LOAD
#define BITBLEND_REGISTER_PREFETCH GROUPED_HIGH_REGISTER_PREFETCH
#define BITBLEND_VECTOR_EPILOGUE GROUPED_HIGH_VECTOR_EPILOGUE
#include "dual_warp.cuh"
#undef BITBLEND_NAMESPACE
#undef BITBLEND_MID
#undef BITBLEND_LOW
#undef BITBLEND_BLOCK_K
#undef BITBLEND_STAGES
#undef BITBLEND_TOKEN_WARPS
#undef BITBLEND_MATRIX_LOAD
#undef BITBLEND_REGISTER_PREFETCH
#undef BITBLEND_VECTOR_EPILOGUE

namespace bitblend::grouped {
constexpr int kThreads = GROUPED_LOW_CHANNELS * 2 * GROUPED_LOW_TOKEN_WARPS;
static_assert(kThreads == GROUPED_MID_CHANNELS * 2 * GROUPED_MID_TOKEN_WARPS);
static_assert(kThreads == GROUPED_HIGH_CHANNELS * 2 * GROUPED_HIGH_TOKEN_WARPS);
constexpr int kLowShared = grouped_low::shared_bytes<GROUPED_LOW_TOKENS, GROUPED_LOW_CHANNELS>();
constexpr int kMidShared = grouped_mid::shared_bytes<GROUPED_MID_TOKENS, GROUPED_MID_CHANNELS>();
constexpr int kHighShared = grouped_high::shared_bytes<GROUPED_HIGH_TOKENS, GROUPED_HIGH_CHANNELS>();
constexpr int kShared = kLowShared > kMidShared ?
    (kLowShared > kHighShared ? kLowShared : kHighShared) :
    (kMidShared > kHighShared ? kMidShared : kHighShared);
static_assert(kShared <= 101376);

// Eight base fields followed by two host-precomputed CUTLASS dividers.
struct Group {
  int64_t x, sx, m, row, tier, splits, scratch, task_end;
  int64_t tiles, tiles_multiplier, tiles_shift;
  int64_t extent, extent_multiplier, extent_shift;
};
static_assert(sizeof(Group) == 14 * sizeof(int64_t));

__device__ __forceinline__ void divmod(int& q, int& r, int value,
                                      int64_t d, int64_t mul, int64_t shift) {
  cutlass::FastDivmod divisor;
  divisor.divisor = d;
  divisor.multiplier = mul;
  divisor.shift_right = shift;
  divisor(q, r, value);
}

__global__ __launch_bounds__(kThreads, GROUPED_MIN_BLOCKS) void kernel(
    Group const* groups, int count, int64_t tasks, int n, int k,
    uint8_t const* p0, uint8_t const* s0, uint8_t const* p1, uint8_t const* s1,
    uint8_t const* p2, uint8_t const* s2, __nv_bfloat16* out, float* scratch,
    __nv_bfloat16 const* bias, bool m_major) {
  extern __shared__ __align__(16) uint8_t storage[];
  // A captured launch supplies a capacity; descriptors carry this batch's work.
  tasks = tasks < groups[count - 1].task_end ? tasks : groups[count - 1].task_end;
  for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    int index = 0;
    while (index + 1 < count && task >= groups[index].task_end) ++index;
    Group g = groups[index];
    int local = task - (index ? groups[index - 1].task_end : 0);
    int tm = g.tier == 0 ? GROUPED_LOW_TOKENS :
             g.tier == 1 ? GROUPED_MID_TOKENS : GROUPED_HIGH_TOKENS;
    int tn = g.tier == 0 ? GROUPED_LOW_CHANNELS :
             g.tier == 1 ? GROUPED_MID_CHANNELS : GROUPED_HIGH_CHANNELS;
    int split, tile, outer, inner;
    divmod(split, tile, local, g.tiles, g.tiles_multiplier, g.tiles_shift);
    divmod(outer, inner, tile, g.extent, g.extent_multiplier, g.extent_shift);
    int m0 = (m_major ? inner : outer) * tm;
    int n0 = (m_major ? outer : inner) * tn;
    auto* x = reinterpret_cast<uint8_t const*>(g.x);
    auto* sx = reinterpret_cast<uint8_t const*>(g.sx);
    auto* y = out + g.row * n;
    auto* partial = scratch ? scratch + g.scratch : nullptr;
    if (g.tier == 0) {
      grouped_low::compute_tile<GROUPED_LOW_TOKENS, GROUPED_LOW_CHANNELS>(
          x, sx, p0, s0, nullptr, nullptr, y, partial, g.m, n, k,
          g.splits, m0, n0, split, storage, bias);
    } else if (g.tier == 1) {
      grouped_mid::compute_tile<GROUPED_MID_TOKENS, GROUPED_MID_CHANNELS>(
          x, sx, p0, s0, p1, s1, y, partial, g.m, n, k,
          g.splits, m0, n0, split, storage, bias);
    } else {
      grouped_high::compute_tile<GROUPED_HIGH_TOKENS, GROUPED_HIGH_CHANNELS>(
          x, sx, p0, s0, p2, s2, y, partial, g.m, n, k,
          g.splits, m0, n0, split, storage, bias);
    }
    // All async copies have drained; also protect vector-epilogue shared reads
    // before the next job reuses the same storage, possibly for another tier.
    if (task + gridDim.x < tasks) __syncthreads();
  }
}

__global__ void reduce(Group const* groups, int count, float const* scratch,
                       __nv_bfloat16* out, int64_t elements, int n,
                       __nv_bfloat16 const* bias) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  int64_t row = index / n;
  int group = 0;
  while (group + 1 < count &&
         (row < groups[group].row || row >= groups[group].row + groups[group].m)) ++group;
  Group g = groups[group];
  if (g.splits == 1) return;
  int64_t local = index - g.row * n;
  float value = 0;
  for (int split = 0; split < g.splits; ++split)
    value += scratch[g.scratch + int64_t(split) * g.m * n + local];
  out[index] = grouped_low::finish(value, bias, index % n);
}
}  // namespace bitblend::grouped

namespace {
void initialize(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  auto status = cudaFuncSetAttribute(bitblend::grouped::kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bitblend::grouped::kShared);
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int64_t resources() {
  using namespace bitblend::grouped;
  cudaFuncAttributes attr{};
  int blocks = 0;
  auto status = cudaFuncGetAttributes(&attr, kernel);
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kThreads, kShared);
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  return attr.numRegs + (int64_t(blocks) << 8) + (int64_t(attr.localSizeBytes) << 16);
}

void grouped_gemm(TensorView groups, TensorView p0, TensorView s0,
                  TensorView p1, TensorView s1, TensorView p2, TensorView s2,
                  TensorView out, TensorView scratch, TensorView bias,
                  int64_t tasks, int64_t waves, bool m_major, bool split_k) {
  using namespace bitblend::grouped;
  CHECK_INPUT_AND_TYPE(groups, dl_int64);
  CHECK_INPUT_AND_TYPE(p0, dl_uint8);
  CHECK_INPUT_AND_TYPE(s0, dl_uint8);
  CHECK_INPUT_AND_TYPE(p1, dl_uint8);
  CHECK_INPUT_AND_TYPE(s1, dl_uint8);
  CHECK_INPUT_AND_TYPE(s2, dl_uint8);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(scratch, dl_float32);
  CHECK_INPUT_AND_TYPE(bias, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(p2, dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(groups.ndim(), 2);
  TVM_FFI_ICHECK_EQ(groups.size(1), 14);
  TVM_FFI_ICHECK_GT(groups.size(0), 0);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(p0.ndim(), 2);
  int64_t n = p0.size(0), k = p0.size(1) * 2;
  TVM_FFI_ICHECK_EQ(out.size(1), n);
  TVM_FFI_ICHECK_EQ(p1.size(0), n);
  TVM_FFI_ICHECK_EQ(p1.size(1) * 2, k);
  TVM_FFI_ICHECK_EQ(p2.size(0), n);
  TVM_FFI_ICHECK_EQ(p2.size(1), k);
  TVM_FFI_ICHECK_EQ(n % 8, 0);
  TVM_FFI_ICHECK_EQ(k % 128, 0);
  TVM_FFI_ICHECK_GT(tasks, 0);
  TVM_FFI_ICHECK_GE(waves, 0);
  TVM_FFI_ICHECK(bias.numel() == 0 || bias.numel() == n);
  for (auto t : {p0, s0, p1, s1, p2, s2, out, scratch, bias})
    TVM_FFI_ICHECK_EQ(t.device().device_id, groups.device().device_id);
  ffi::CUDADeviceGuard guard(groups.device().device_id);
  auto stream = get_stream(groups.device());
  int blocks = tasks;
  if (waves) {
    int sm_count = 0, active = (resources() >> 8) & 255;
    TVM_FFI_ICHECK_GT(active, 0);
    auto status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                         groups.device().device_id);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    blocks = std::min<int64_t>(tasks, int64_t(sm_count) * active * waves);
  }
  auto* bias_ptr = bias.numel() ? static_cast<__nv_bfloat16 const*>(bias.data_ptr()) : nullptr;
  kernel<<<blocks, kThreads, kShared, stream>>>(
      static_cast<Group const*>(groups.data_ptr()), groups.size(0), tasks, n, k,
      static_cast<uint8_t const*>(p0.data_ptr()), static_cast<uint8_t const*>(s0.data_ptr()),
      static_cast<uint8_t const*>(p1.data_ptr()), static_cast<uint8_t const*>(s1.data_ptr()),
      static_cast<uint8_t const*>(p2.data_ptr()), static_cast<uint8_t const*>(s2.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<float*>(scratch.data_ptr()),
      bias_ptr, m_major);
  if (split_k)
    reduce<<<(out.numel() + 255) / 256, 256, 0, stream>>>(
        static_cast<Group const*>(groups.data_ptr()), groups.size(0),
        static_cast<float const*>(scratch.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
        out.numel(), n, bias_ptr);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize, initialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(resources, resources);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(grouped_gemm, grouped_gemm);
