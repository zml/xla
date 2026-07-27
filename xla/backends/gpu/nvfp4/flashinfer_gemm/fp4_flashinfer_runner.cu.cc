// FlashInfer's autotuned SM120 NVFP4 W4A4 GEMM, vendored into XLA. This is the
// kernel vLLM actually selects at runtime to hit ~126 tok/s on gemma-4-12B-NVFP4
// bs=1 decode. Unlike our single 128x128x128 cutlass config (64.8 tok/s), this
// template compiles the full SM120 tile ladder (thin-N tiles for the decode
// GEMV) and picks the fastest per shape.
//
// Provenance: flashinfer include/flashinfer/gemm/{fp4_gemm_template_sm120.h,
// fp4_gemm_cutlass_template_sm120.h, cutlass_gemm_configs.h, fp4_gemm_cutlass.h}
// + arch_condition.h, slimmed cutlass_utils->cutlass_dtype.h. StreamK codegen is
// aliased to the DP scheduler via FI_FP4_NO_STREAMK (halves kernel count).
//
// Scales arrive ALREADY SWIZZLED (cutlass blockscaled layout) — same layout
// cutlass_fp4_mm.cu.cc consumes, produced by the framework at load time.
// alpha == global_sf == 1/(igs*wgs).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "fp4_gemm_cutlass_template_sm120.h"

namespace flashinfer {
namespace gemm {

// ---- Explicit kernel instantiations (DP scheduler; StreamK aliased to DP). ----
// (CTA_M, CTA_N, CTA_K) per the dispatcher's tile->shape mapping.
#ifdef FI_FP4_FULL_TILESET
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 32, 128, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 32, 128, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 32, 256, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 32, 256, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 256, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 256, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 128, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 128, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 256, 128, 128, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 256, 128, 128, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 1, 1, 1, _1SM, false)
#else
// Shakeout set: (128,64,256) is the thin-N K=256 tile that failed the SF-TMA
// size-equivalence static_assert on cutlass 4.4.2. If it compiles after the
// bump to 4.6.1, the full ladder is unlocked. (128,128,256) is the known-good
// control. CtaShape128x64x128B=(128,64,256), CtaShape128x128x128B=(128,128,256).
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 256, 1, 1, 1, _1SM, false)
#endif

}  // namespace gemm
}  // namespace flashinfer

using flashinfer::gemm::CutlassFp4GemmRunner;
using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM120;
using flashinfer::gemm::FP4GemmType;

// The tile shapes we actually instantiated above (must stay in sync). Autotuning
// only tries these; anything else in getConfigs() is filtered out so we never
// dispatch a non-compiled tile.
static bool tile_is_available(CutlassTileConfigSM120 t) {
#ifdef FI_FP4_FULL_TILESET
  switch (t) {
    case CutlassTileConfigSM120::CtaShape128x32x64B:
    case CutlassTileConfigSM120::CtaShape128x32x128B:
    case CutlassTileConfigSM120::CtaShape128x64x64B:
    case CutlassTileConfigSM120::CtaShape128x64x128B:
    case CutlassTileConfigSM120::CtaShape128x128x64B:
    case CutlassTileConfigSM120::CtaShape128x128x128B:
    case CutlassTileConfigSM120::CtaShape256x128x64B:
    case CutlassTileConfigSM120::CtaShape128x256x64B:
      return true;
    default:
      return false;
  }
#else
  return t == CutlassTileConfigSM120::CtaShape128x128x128B ||  // (128,128,256)
         t == CutlassTileConfigSM120::CtaShape128x64x128B;     // (128,64,256)
#endif
}

// DP configs (use_stream_k==false) restricted to compiled tiles. StreamK was
// tried (split-K for small-N/large-K GEMVs) but never won the autotune — these
// M=1 GEMVs are memory-bound and DP already saturates the bus; split-K's
// reduction overhead only hurts. Kept DP-only (halves the kernel count).
static const std::vector<CutlassGemmConfig>& dp_configs() {
  static const std::vector<CutlassGemmConfig> cfgs = [] {
    CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4> r;
    std::vector<CutlassGemmConfig> out;
    for (auto& c : r.getConfigs())
      if (!c.use_stream_k && tile_is_available(c.tile_config_sm120)) out.push_back(c);
    return out;
  }();
  return cfgs;
}

extern "C" int flashinfer_fp4_num_tactics() { return (int)dp_configs().size(); }

extern "C" size_t flashinfer_fp4_workspace(int M, int N, int K) {
  // Max workspace over all DP configs for this shape (getWorkspaceSize iterates
  // configs internally, dispatching each via dispatchToArch with null ptrs).
  CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4> r;
  return r.getWorkspaceSize(M, N, K, 1);
}

// Run one tactic. Scales must be pre-swizzled. Returns false if the config can't
// implement this shape (e.g. smem exceeds device max) so the caller can skip it.
extern "C" bool flashinfer_fp4_gemm_bf16(void* D, const void* A_fp4, const void* B_fp4,
                                         const void* Asf_sw, const void* Bsf_sw,
                                         const float* alpha, int M, int N, int K,
                                         cudaStream_t stream, void* ws, size_t ws_bytes,
                                         int tactic) {
  const auto& cfgs = dp_configs();
  if (tactic < 0 || tactic >= (int)cfgs.size()) return false;
  CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4> r;
  try {
    r.gemm(D, A_fp4, B_fp4, Asf_sw, Bsf_sw, alpha, M, N, K, 1, cfgs[tactic],
           static_cast<char*>(ws), ws_bytes, stream);
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

// Time every DP tactic (NOT graph-capture safe: it syncs). Returns best tactic
// index and, if out_best_us != nullptr, its per-call latency in microseconds.
extern "C" int flashinfer_fp4_autotune(void* D, const void* A_fp4, const void* B_fp4,
                                       const void* Asf_sw, const void* Bsf_sw, const float* alpha,
                                       int M, int N, int K, cudaStream_t stream, void* ws,
                                       size_t ws_bytes, int iters, float* out_best_us) {
  int nt = flashinfer_fp4_num_tactics();
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  int best = -1;
  float best_ms = 1e30f;
  for (int t = 0; t < nt; t++) {
    if (!flashinfer_fp4_gemm_bf16(D, A_fp4, B_fp4, Asf_sw, Bsf_sw, alpha, M, N, K, stream, ws,
                                  ws_bytes, t))
      continue;
    cudaStreamSynchronize(stream);
    cudaEventRecord(beg, stream);
    for (int i = 0; i < iters; i++)
      flashinfer_fp4_gemm_bf16(D, A_fp4, B_fp4, Asf_sw, Bsf_sw, alpha, M, N, K, stream, ws, ws_bytes,
                               t);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, beg, end);
    if (ms < best_ms) {
      best_ms = ms;
      best = t;
    }
  }
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  if (out_best_us) *out_best_us = best >= 0 ? best_ms / iters * 1000.f : -1.f;
  return best;
}

// ------------------------------- bench main -------------------------------
#ifndef FI_FP4_BENCH_NO_MAIN
#include "cute/tensor.hpp"
using namespace cute;

// Reference kernel (CPU-verified) from cutlass_fp4_mm.cu.cc: pre-swizzled scales
// in, applies alpha. We compare FlashInfer's output to it.
using AllocFn = void* (*)(void* ctx, size_t size);
extern "C" void cutlass_fp4_mm_sm120a(void* D, const void* A, const void* B, const void* Asf_sw,
                                      const void* Bsf_sw, const float* alpha, int M, int N, int K,
                                      cudaStream_t stream, AllocFn alloc, void* actx);

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e = (x);                                                                      \
    if (e != cudaSuccess) {                                                                   \
      printf("CUDA %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);                 \
      abort();                                                                                \
    }                                                                                         \
  } while (0)

// Layout ORACLE: scatters a natural [rows,kg] ue4m3 scale into the SF layout by
// indexing cutlass's own cute layout object, deriving nothing by hand. Nobody
// ships this (it is one write per element) — it exists so the bench can prove the
// closed-form layout the framework implements (swizzle_sf_ref below, mirrored by
// ZML ops.swizzleNvfp4Scale) is byte-exactly what cutlass indexes.
template <class LayoutSF>
__global__ void fi_swizzle(cutlass::float_ue4m3_t* dst, const cutlass::float_ue4m3_t* src,
                           LayoutSF layout, int rows, int kg) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kg) return;
  int r = i / kg, g = i % kg;
  auto t = make_tensor(make_gmem_ptr(dst), layout);
  t(r, g * 16, 0) = src[r * kg + g];
}

// Closed form of that same layout, on the host: ceil(rows/128) x ceil(kg/4)
// blocks of 512 bytes, MN-major, sf[bn][bk][m0][m1][j] = nat[bn*128+m1*32+m0][bk*4+j],
// padding zero. This is the statement the framework-side swizzle implements;
// benchShape() below checks it against fi_swizzle byte for byte.
static std::vector<uint8_t> swizzle_sf_ref(const std::vector<uint8_t>& nat, int rows, int kg) {
  int bn = (rows + 127) / 128, bk = (kg + 3) / 4;
  std::vector<uint8_t> out((size_t)bn * bk * 512, 0);
  for (int block_n = 0; block_n < bn; block_n++)
    for (int block_k = 0; block_k < bk; block_k++)
      for (int m0 = 0; m0 < 32; m0++)
        for (int m1 = 0; m1 < 4; m1++)
          for (int j = 0; j < 4; j++) {
            int n = block_n * 128 + m1 * 32 + m0, g = block_k * 4 + j;
            if (n >= rows || g >= kg) continue;  // padding stays 0
            out[(size_t)(block_n * bk + block_k) * 512 + (m0 * 4 + m1) * 4 + j] =
                nat[(size_t)n * kg + g];
          }
  return out;
}

// Grab the (tile-independent) blockscaled SF layout from an always-instantiated Gemm.
using RefGemm = flashinfer::gemm::Fp4Gemm___nv_bfloat16_128_128_256false;
using RefSFCfg = typename RefGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

static float bf16_to_f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}
static inline uint8_t fp4code(int idx) { return (uint8_t)(idx & 15); }
static inline uint8_t ue4m3enc(float v) {
  int e = 0;
  float t = v;
  while (t >= 2.f) { t /= 2.f; e++; }
  while (t < 1.f) { t *= 2.f; e--; }
  int mant = (int)((t - 1.f) * 8.f + 0.5f);
  return (uint8_t)(((e + 7) << 3) | (mant & 7));
}

static void* mallocAlloc(void*, size_t sz) {
  void* p;
  cudaMalloc(&p, sz);
  return p;
}

static void benchShape(int M, int N, int K) {
  const int kg = K / 16;
  auto aIdx = [](int m, int k) { return (m * 7 + k * 3 + 1) & 15; };
  auto bIdx = [](int n, int k) { return (n * 5 + k * 2 + 2) & 15; };
  std::vector<uint8_t> hA(M * K / 2), hB(N * K / 2), hAsf(M * kg), hBsf(N * kg);
  for (int m = 0; m < M; m++)
    for (int j = 0; j < K / 2; j++)
      hA[m * (K / 2) + j] = (uint8_t)(fp4code(aIdx(m, 2 * j)) | (fp4code(aIdx(m, 2 * j + 1)) << 4));
  for (int n = 0; n < N; n++)
    for (int j = 0; j < K / 2; j++)
      hB[n * (K / 2) + j] = (uint8_t)(fp4code(bIdx(n, 2 * j)) | (fp4code(bIdx(n, 2 * j + 1)) << 4));
  for (int m = 0; m < M; m++)
    for (int g = 0; g < kg; g++)
      hAsf[m * kg + g] = ue4m3enc(__builtin_exp2f((float)(((m * 3 + g * 7) % 14) - 6)));
  for (int n = 0; n < N; n++)
    for (int g = 0; g < kg; g++)
      hBsf[n * kg + g] = ue4m3enc(__builtin_exp2f((float)(((n * 5 + g * 3) % 14) - 6)));
  const float alpha_h = 0.5f;

  void *dA, *dB, *dBsf_nat, *dRef, *dD, *dAlpha;
  CK(cudaMalloc(&dA, hA.size()));
  CK(cudaMalloc(&dB, hB.size()));
  CK(cudaMalloc(&dBsf_nat, hBsf.size()));
  CK(cudaMalloc(&dRef, (size_t)M * N * 2));
  CK(cudaMalloc(&dD, (size_t)M * N * 2));
  CK(cudaMalloc(&dAlpha, 4));
  CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBsf_nat, hBsf.data(), hBsf.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dAlpha, &alpha_h, 4, cudaMemcpyHostToDevice));

  // Both kernels take pre-swizzled scales; the framework does this at load time.
  auto layout_SFA = RefSFCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = RefSFCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  size_t szA = size(filter_zeros(layout_SFA)), szB = size(filter_zeros(layout_SFB));
  std::vector<uint8_t> hAsf_sw = swizzle_sf_ref(hAsf, M, kg);
  std::vector<uint8_t> hBsf_sw = swizzle_sf_ref(hBsf, N, kg);
  cutlass::float_ue4m3_t *dAsf_sw, *dBsf_sw;
  CK(cudaMalloc(&dAsf_sw, szA));
  CK(cudaMalloc(&dBsf_sw, szB));
  CK(cudaMemcpy(dAsf_sw, hAsf_sw.data(), szA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBsf_sw, hBsf_sw.data(), szB, cudaMemcpyHostToDevice));

  // ---- Prove the closed-form layout == the layout cutlass itself indexes ----
  // If this ever mismatches, the framework-side swizzle (ZML ops.swizzleNvfp4Scale)
  // is wrong for this shape and every fp4 matmul silently returns garbage.
  {
    if (hAsf_sw.size() != szA || hBsf_sw.size() != szB) {
      printf("  [swizzle] SIZE MISMATCH: host A=%zu/%zu B=%zu/%zu\n", hAsf_sw.size(), szA,
             hBsf_sw.size(), szB);
      abort();
    }
    cutlass::float_ue4m3_t* dBsf_cute;
    CK(cudaMalloc(&dBsf_cute, szB));
    CK(cudaMemset(dBsf_cute, 0, szB));
    fi_swizzle<<<(N * kg + 255) / 256, 256>>>(dBsf_cute, (cutlass::float_ue4m3_t*)dBsf_nat,
                                              layout_SFB, N, kg);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> cute_out(szB);
    CK(cudaMemcpy(cute_out.data(), dBsf_cute, szB, cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < szB; i++)
      if (cute_out[i] != hBsf_sw[i]) diff++;
    printf("  [swizzle] szB=%zu closed-form vs cute: diff=%zu %s\n", szB, diff,
           diff ? "!!MISMATCH" : "OK");
    cudaFree(dBsf_cute);
  }

  // Reference via our CPU-verified kernel, same swizzled scales FlashInfer gets.
  cutlass_fp4_mm_sm120a(dRef, dA, dB, dAsf_sw, dBsf_sw, (const float*)dAlpha, M, N, K, 0,
                        mallocAlloc, nullptr);
  CK(cudaDeviceSynchronize());
  std::vector<uint16_t> hRef(M * N);
  CK(cudaMemcpy(hRef.data(), dRef, (size_t)M * N * 2, cudaMemcpyDeviceToHost));

  size_t ws_bytes = flashinfer_fp4_workspace(M, N, K);
  void* ws = nullptr;
  if (ws_bytes) CK(cudaMalloc(&ws, ws_bytes));

  int nt = flashinfer_fp4_num_tactics();
  printf("== M=%d N=%d K=%d  (ws=%zuB, %d tactics) ==\n", M, N, K, ws_bytes, nt);
  int best = -1;
  float best_us = 1e30f;
  for (int t = 0; t < nt; t++) {
    CK(cudaMemset(dD, 0, (size_t)M * N * 2));
    bool ok = flashinfer_fp4_gemm_bf16(dD, dA, dB, dAsf_sw, dBsf_sw, (const float*)dAlpha, M, N, K,
                                       0, ws, ws_bytes, t);
    if (!ok) {
      printf("  tactic %2d: SKIP (can't implement)\n", t);
      continue;
    }
    CK(cudaDeviceSynchronize());
    // Correctness vs reference (sample).
    std::vector<uint16_t> hD(M * N);
    CK(cudaMemcpy(hD.data(), dD, (size_t)M * N * 2, cudaMemcpyDeviceToHost));
    float maxrel = 0.f;
    int nstep = (N / 64 > 0) ? N / 64 : 1;
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n += nstep) {
        float g = bf16_to_f(hD[m * N + n]), e = bf16_to_f(hRef[m * N + n]);
        float rel = __builtin_fabsf(g - e) / (__builtin_fabsf(e) + 1e-3f);
        if (rel > maxrel) maxrel = rel;
      }
    // Time.
    const int iters = 100;
    cudaEvent_t b, e;
    cudaEventCreate(&b);
    cudaEventCreate(&e);
    for (int w = 0; w < 5; w++)
      flashinfer_fp4_gemm_bf16(dD, dA, dB, dAsf_sw, dBsf_sw, (const float*)dAlpha, M, N, K, 0, ws,
                               ws_bytes, t);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(b);
    for (int i = 0; i < iters; i++)
      flashinfer_fp4_gemm_bf16(dD, dA, dB, dAsf_sw, dBsf_sw, (const float*)dAlpha, M, N, K, 0, ws,
                               ws_bytes, t);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, b, e);
    float us = ms / iters * 1000.f;
    cudaEventDestroy(b);
    cudaEventDestroy(e);
    printf("  tactic %2d: %7.2f us  maxrel=%.4f %s\n", t, us, maxrel,
           maxrel > 0.05f ? "!!MISMATCH" : "");
    if (us < best_us && maxrel < 0.05f) {
      best_us = us;
      best = t;
    }
  }
  printf("  >> BEST tactic %d @ %.2f us\n", best, best_us);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dBsf_nat);
  cudaFree(dAsf_sw);
  cudaFree(dBsf_sw);
  cudaFree(dRef);
  cudaFree(dD);
  cudaFree(dAlpha);
  if (ws) cudaFree(ws);
}

int main() {
  // Real gemma-4-12B-NVFP4 decode GEMV shapes (M=1): qkv, o, gate/up, down.
  benchShape(1, 2048, 3840);
  benchShape(1, 4096, 3840);
  benchShape(1, 3840, 4096);
  benchShape(1, 15360, 3840);
  benchShape(1, 3840, 15360);
  return 0;
}
#endif  // FI_FP4_BENCH_NO_MAIN
