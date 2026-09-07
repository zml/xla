/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp8.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/fp8_block_gemv_kernel.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"

namespace xla::gpu::kernel {
namespace {

__device__ __forceinline__ void Fp8x16ToF32(const uint4& wv, float* o) {
  const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __half2 a = __nv_cvt_fp8x2_to_halfraw2(w32[j] & 0xffff, __NV_E4M3);
    __half2 b = __nv_cvt_fp8x2_to_halfraw2(w32[j] >> 16, __NV_E4M3);
    float2 fa = __half22float2(a);
    float2 fb = __half22float2(b);
    o[j * 4 + 0] = fa.x;
    o[j * 4 + 1] = fa.y;
    o[j * 4 + 2] = fb.x;
    o[j * 4 + 3] = fb.y;
  }
}

template <int WARPS, int R, int UNROLL>
__global__ __launch_bounds__(WARPS * 32) void Fp8BlockGemv(
    const __nv_bfloat16* __restrict__ x, const __nv_fp8_e4m3* __restrict__ W,
    const __nv_bfloat16* __restrict__ Ws, __nv_bfloat16* __restrict__ out,
    int N, int K, int ldsc) {
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);  // [K], scaled

  const int tid = threadIdx.x;
  const int nthreads = WARPS * 32;
  const int n_base = blockIdx.x * (WARPS * R);
  const __nv_bfloat16* scrow = Ws + static_cast<size_t>(n_base >> 7) * ldsc;

  for (int i = tid; i < K; i += nthreads) {
    xs[i] = __float2bfloat16(__bfloat162float(x[i]) *
                             __bfloat162float(scrow[i >> 7]));
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int n0 = n_base + warp * R;
  if (n0 >= N) return;

  const int nvec = K >> 4;
  const uint4* Wrows[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    Wrows[r] = reinterpret_cast<const uint4*>(W + static_cast<size_t>(n0 + r) * K);
  }

  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;

  int v = lane;
  for (; v + 32 * (UNROLL - 1) < nvec; v += 32 * UNROLL) {
    uint4 wv[R][UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int r = 0; r < R; ++r) wv[r][u] = Wrows[r][v + u * 32];
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const __nv_bfloat16* xp = xs + ((v + u * 32) << 4);
      float xr[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) xr[i] = __bfloat162float(xp[i]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float wf[16];
        Fp8x16ToF32(wv[r][u], wf);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[r] = fmaf(wf[i], xr[i], acc[r]);
      }
    }
  }
  for (; v < nvec; v += 32) {
    const __nv_bfloat16* xp = xs + (v << 4);
    float xr[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) xr[i] = __bfloat162float(xp[i]);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      uint4 wv = Wrows[r][v];
      float wf[16];
      Fp8x16ToF32(wv, wf);
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[r] = fmaf(wf[i], xr[i], acc[r]);
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
    }
    if (lane == 0 && n0 + r < N) out[n0 + r] = __float2bfloat16(acc[r]);
  }
}

void* KernelSymbol(const Fp8BlockGemvKernelConfig& c) {
#define XLA_FP8_GEMV_CASE(W, R, U)                       \
  if (c.num_warps == (W) && c.rows_per_warp == (R) && c.unroll == (U)) { \
    return absl::bit_cast<void*>(&Fp8BlockGemv<W, R, U>);                \
  }
  XLA_FP8_GEMV_CASE(2, 2, 4)
  XLA_FP8_GEMV_CASE(2, 2, 8)
  XLA_FP8_GEMV_CASE(2, 4, 4)
  XLA_FP8_GEMV_CASE(2, 4, 8)
  XLA_FP8_GEMV_CASE(4, 2, 4)
  XLA_FP8_GEMV_CASE(4, 2, 8)
  XLA_FP8_GEMV_CASE(4, 4, 4)
  XLA_FP8_GEMV_CASE(4, 4, 8)
  XLA_FP8_GEMV_CASE(8, 2, 4)
  XLA_FP8_GEMV_CASE(8, 2, 8)
  XLA_FP8_GEMV_CASE(8, 4, 4)
  XLA_FP8_GEMV_CASE(8, 4, 8)
  XLA_FP8_GEMV_CASE(16, 2, 4)
  XLA_FP8_GEMV_CASE(16, 2, 8)
  XLA_FP8_GEMV_CASE(16, 4, 4)
  XLA_FP8_GEMV_CASE(16, 4, 8)
#undef XLA_FP8_GEMV_CASE
  return nullptr;
}

}  // namespace

bool IsSupportedFp8BlockGemvKernelConfig(
    const Fp8BlockGemvKernelConfig& config) {
  return KernelSymbol(config) != nullptr;
}

absl::StatusOr<CustomKernel> GetFp8BlockGemvKernel(
    const Fp8BlockGemvKernelConfig& config, int64_t activation_arg,
    int64_t weight_arg, int64_t scale_arg, int64_t output_arg, int64_t n,
    int64_t k) {
  void* symbol = KernelSymbol(config);
  if (symbol == nullptr) {
    return Internal("No fp8 block gemv kernel for %d warps, %d rows, unroll %d",
                    config.num_warps, config.rows_per_warp, config.unroll);
  }
  if (k % 16 != 0) {
    return Internal("fp8 block gemv needs k %% 16 == 0, got %d", k);
  }
  const int64_t rows_per_block = config.num_warps * config.rows_per_warp;
  if (128 % rows_per_block != 0) {
    return Internal(
        "fp8 block gemv needs the rows a block owns to divide the 128-row "
        "scale block, got %d",
        rows_per_block);
  }

  se::KernelArgsPackingSpec packing;
  packing.AddAddressArgument(activation_arg);
  packing.AddAddressArgument(weight_arg);
  packing.AddAddressArgument(scale_arg);
  packing.AddAddressArgument(output_arg);
  packing.AddConstantArgument(static_cast<int32_t>(n));
  packing.AddConstantArgument(static_cast<int32_t>(k));
  packing.AddConstantArgument(static_cast<int32_t>(k / 128));

  se::KernelLoaderSpec spec = se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      symbol, "fp8_block_gemv", /*arity=*/7, std::move(packing));

  return CustomKernel(
      absl::StrCat("fp8_block_gemv_", n, "x", k, "_w", config.num_warps, "_r",
                   config.rows_per_warp, "_u", config.unroll),
      std::move(spec),
      se::BlockDim(CeilOfRatio<int64_t>(n, rows_per_block), 1, 1),
      se::ThreadDim(config.num_warps * 32, 1, 1),
      /*shared_memory_bytes=*/k * sizeof(__nv_bfloat16));
}

}  // namespace xla::gpu::kernel
