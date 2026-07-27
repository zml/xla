// Real cutlass sm120a NVFP4 W4A4 GEMM entry + scale swizzle, validated with a
// distinct-value CPU reference (row-dependent scales exercise the 128x4 row
// interleave that all-ones cannot). This is the kernel the XLA FFI handler will
// call. Scales arrive in NATURAL row-major [rows, K/16] ue4m3 layout; we swizzle
// to cutlass's blockscaled layout via the kernel's own LayoutSFA/SFB.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

#define CUDA_CHECK(x)                                                     \
  do {                                                                    \
    cudaError_t e = (x);                                                  \
    if (e != cudaSuccess) {                                               \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
             __LINE__);                                                   \
      abort();                                                            \
    }                                                                     \
  } while (0)

#define CUTLASS_CHECK(status)                                        \
  do {                                                               \
    cutlass::Status s = (status);                                    \
    if (s != cutlass::Status::kSuccess) {                            \
      printf("CUTLASS error %d at %s:%d\n", int(s), __FILE__,        \
             __LINE__);                                              \
      abort();                                                       \
    }                                                                \
  } while (0)

struct sm120_fp4_config_M256 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = void;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct sm120_fp4_config_default {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = cutlass::gemm::PersistentScheduler;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename Config, typename OutType>
struct Fp4GemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;
  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          typename Config::EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Config::KernelSchedule>::CollectiveOp;
  using TileScheduler = typename Config::TileScheduler;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                           CollectiveMainloop, CollectiveEpilogue,
                                           TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ---- Block-scale (SF) layout contract ------------------------------------
// Both scale operands arrive ALREADY in cutlass's K-major block-scaled SF layout
// (Sm1xxBlockScaledConfig, SFVecSize=16, atom Shape<(32,4),(16,4)>): a physical
// block of 512 bytes = 128 MN-rows x 4 k-groups, blocks MN-major, i.e.
//     sf[n_blk][k_blk][m0][m1][j] = natural[n_blk*128 + m1*32 + m0][k_blk*4 + j]
// with total size ceil(rows/128)*ceil(kg/4)*512 bytes (== the cute layout's
// size(filter_zeros)). Nothing here swizzles:
//   - the WEIGHT scale is transformed once at load time, in the framework
//     (ZML ops.swizzleNvfp4Scale), so it is device- and backend-agnostic;
//   - the ACTIVATION scale is written straight into this layout by
//     quant_act_kernel (quant + swizzle fused, like vLLM's cvt_fp16_to_fp4).
// A caller that only has a natural [rows, kg] scale must convert it itself.
// -------------------------------------------------------------------------

// M256 config (covers decode M<=256). D bf16 [M,N] row-major. A fp4 [M,K/2]
// row-major, B fp4 [N,K/2] col-major (== weight [N,K]). Asf/Bsf pre-swizzled
// (see the SF layout contract above).
// Allocation callback: returns device scratch (from XLA's ScratchAllocator when
// available, else cudaMallocAsync). Frees are no-ops when scratch-backed.
using AllocFn = void* (*)(void* ctx, size_t size);

// ---- FlashInfer autotuned SM120 NVFP4 GEMM (vendored, cutlass 4.6.1, sm_120a).
// 3-4x faster than the local kernel at M=1 decode. Scales must be pre-swizzled
// (same layout run_fp4_mm already produces); alpha == 1/(igs*wgs). ----
extern "C" int flashinfer_fp4_num_tactics();
extern "C" size_t flashinfer_fp4_workspace(int M, int N, int K);
extern "C" bool flashinfer_fp4_gemm_bf16(void* D, const void* A_fp4, const void* B_fp4,
                                         const void* Asf_sw, const void* Bsf_sw,
                                         const float* alpha, int M, int N, int K,
                                         cudaStream_t stream, void* ws, size_t ws_bytes,
                                         int tactic);
extern "C" int flashinfer_fp4_autotune(void* D, const void* A_fp4, const void* B_fp4,
                                       const void* Asf_sw, const void* Bsf_sw, const float* alpha,
                                       int M, int N, int K, cudaStream_t stream, void* ws,
                                       size_t ws_bytes, int iters, float* out_best_us);

// Routing is ON by default; ZML_A4W4_FI=0 forces the local kernel (the
// standalone bench sets it so its correctness oracle stays independent).
static bool fi_enabled() {
  static const bool on = [] {
    const char* e = getenv("ZML_A4W4_FI");
    return !(e && e[0] == '0');
  }();
  return on;
}

// Universal near-optimal M=1 default: tactic 2 = (128,32,256) tile, swap_ab
// (dp_configs() order: tile_idx*2 + (swap?0:1); tile 1 = 128x32x128B = CTA_K256).
static int fi_default_tactic() { return 2; }

// Per-shape best tactic. Autotunes on the first NON-capturing call for a shape
// (llmd warmup), caches it, and reuses the cached tactic inside the captured
// CUDA graph. During capture with no cached entry we cannot time (no sync), so we
// fall back to fi_default_tactic() without caching.
static int fi_pick_tactic(int M, int N, int K, void* D, const void* A, const void* B,
                          const void* dAsf, const void* dBsf, const float* alpha,
                          cudaStream_t stream, void* ws, size_t ws_bytes) {
  static std::mutex mtx;
  static std::unordered_map<uint64_t, int> cache;
  uint64_t key = ((uint64_t)(uint32_t)M << 40) ^ ((uint64_t)(uint32_t)N << 20) ^ (uint32_t)K;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(stream, &st);
  if (st != cudaStreamCaptureStatusNone) return fi_default_tactic();  // capturing: cannot tune
  float us = -1.f;
  int t = flashinfer_fp4_autotune(D, A, B, dAsf, dBsf, alpha, M, N, K, stream, ws, ws_bytes, 50, &us);
  if (t < 0) t = fi_default_tactic();
  cache[key] = t;
  fprintf(stderr, "[fi autotune] M=%d N=%d K=%d -> tactic %d @ %.2f us\n", M, N, K, t, us);
  return t;
}

template <typename OutType>
static void run_fp4_mm(void* D, const void* A, const void* B,
                       const void* Asf_sw, const void* Bsf_sw,
                       const float* alpha, int M, int N, int K,
                       cudaStream_t stream, AllocFn alloc, void* actx) {
  using GemmT = Fp4GemmSm120<sm120_fp4_config_M256, OutType>;
  using Gemm = typename GemmT::Gemm;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSF = cutlass::float_ue4m3_t;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  auto layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  // Both scales are consumed as-is — see the SF layout contract above.
  auto* dAsf = const_cast<ElementSF*>(static_cast<const ElementSF*>(Asf_sw));
  auto* dBsf = const_cast<ElementSF*>(static_cast<const ElementSF*>(Bsf_sw));

  // FlashInfer autotuned GEMM path (3-4x faster than the local kernel at M=1).
  // Same SF layout as the local kernel; falls through to it on any failure.
  // Only valid for bf16 output (the only OutType this is instantiated with).
  if (fi_enabled() && std::is_same<OutType, cutlass::bfloat16_t>::value) {
    size_t fiws = flashinfer_fp4_workspace(M, N, K);
    void* dfiws = fiws ? alloc(actx, fiws) : nullptr;
    int t = fi_pick_tactic(M, N, K, D, A, B, dAsf, dBsf, alpha, stream, dfiws, fiws);
    if (t >= 0 &&
        flashinfer_fp4_gemm_bf16(D, A, B, dAsf, dBsf, alpha, M, N, K, stream, dfiws, fiws, t))
      return;
  }

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A), stride_A,
       static_cast<ElementB const*>(B), stride_B, dAsf, layout_SFA, dBsf,
       layout_SFB},
      {{},
       static_cast<ElementD const*>(D), stride_D,
       static_cast<ElementD*>(D), stride_D}};
  arguments.epilogue.thread.alpha_ptr = alpha;

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(arguments);
  void* dWs = ws ? alloc(actx, ws) : nullptr;
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, dWs, stream));
  CUTLASS_CHECK(gemm.run(arguments, dWs, stream));
}

extern "C" void cutlass_fp4_mm_sm120a(void* D, const void* A, const void* B,
                                      const void* Asf_sw, const void* Bsf_sw,
                                      const float* alpha, int M, int N, int K,
                                      cudaStream_t stream, AllocFn alloc,
                                      void* actx) {
  run_fp4_mm<cutlass::bfloat16_t>(D, A, B, Asf_sw, Bsf_sw, alpha, M, N, K,
                                  stream, alloc, actx);
}

// Dynamic NVFP4 activation quantization: x[M,K] bf16 -> fp4 packed [M,K/2]
// (cutlass e2m1, low=even) + ue4m3 block scale [M,K/16] (natural). One block
// per (m, group of 16); packs via cutlass::float_e2m1_t to GUARANTEE the byte
// format cutlass consumes. Matches vLLM scaled_fp4_quant: scale=e4m3(amax/6*igs),
// q=fp4(x*igs/scale).
// Launch: grid(M, ceil(kg/GROUPS_PER_BLOCK)), block = GROUPS_PER_BLOCK*16 = 256.
// One block handles one row m and a contiguous chunk of 16 groups (256 K-elems).
// Thread mapping gives k = blockIdx.y*256 + threadIdx.x, so every block reads a
// fully-coalesced contiguous 512-byte span of x — vs the old <<<M*kg,16>>> which
// launched ~15k half-warp blocks each reading 32 bytes (~15x off the BW floor at
// M=16). All 32 lanes of every warp stay alive through both __shfl calls (mask
// 0xffffffff, only the memory writes are guarded by `active`); the amax XOR by
// <=8 keeps each aligned 16-lane group's reduction independent, exactly as the
// old 16-thread block did. Per-element math, swizzle write, and packing are
// bit-identical to the previous kernel.
__global__ void quant_act_kernel(const __nv_bfloat16* __restrict__ x,
                                 const float* __restrict__ igs_p,
                                 uint8_t* __restrict__ q,
                                 cutlass::float_ue4m3_t* __restrict__ sf,
                                 int M, int K, int blocks_k,
                                 const float* __restrict__ wgs_p,
                                 float* __restrict__ alpha_out) {
  const int kg = K / 16;
  const int m = blockIdx.x;
  const int groups_per_block = blockDim.x >> 4;      // = 16 for 256 threads
  const int local_g = threadIdx.x >> 4;              // 0..groups_per_block-1
  const int t = threadIdx.x & 15;                    // 0..15 (position in group)
  const int g = blockIdx.y * groups_per_block + local_g;
  const bool active = (m < M) && (g < kg);
  const int k = g * 16 + t;
  const float igs = *igs_p;
  // Fold the per-token alpha = 1/(igs*wgs) here (one thread) instead of a
  // standalone ZML scalar op per matmul — the epilogue scale for the gemm.
  if (alpha_out && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    *alpha_out = 1.0f / (igs * *wgs_p);
  const float xv = active ? __bfloat162float(x[(size_t)m * K + k]) : 0.f;
  float a = fabsf(xv);
  // Full 32-lane mask: XOR by <=8 confines each reduction to its aligned 16-lane
  // group, so the two groups sharing a warp reduce independently (same result as
  // the old mask-0xffff, 16-thread block). Inactive lanes contribute a=0.
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
  cutlass::float_ue4m3_t sfq(a / 6.0f * igs);
  const float scale = float(sfq);
  // Write the block scale DIRECTLY into cutlass's swizzled SF layout (the SF
  // layout contract at the top of this file), fusing quant+swizzle into one
  // kernel (vLLM does the same in cvt_fp16_to_fp4). blocks_k = ceil(kg/4).
  if (active && t == 0) {
    const int block_n = m >> 7, m_in = m & 127, m0 = m_in & 31, m1 = m_in >> 5;
    const int mrow = m0 * 4 + m1, block_k = g >> 2, k1 = g & 3;
    sf[(size_t)(block_n * blocks_k + block_k) * 512 + mrow * 4 + k1] = sfq;
  }
  const float qv = (scale > 0.f) ? (xv * igs / scale) : 0.f;
  const uint8_t nib = cutlass::float_e2m1_t(qv).storage & 0xF;
  const uint8_t partner = __shfl_xor_sync(0xffffffff, nib, 1);
  if (active && (t & 1) == 0)
    q[(size_t)m * (K / 2) + k / 2] = nib | (partner << 4);  // low=even, high=odd
}

// D=bf16[M,N]; x=bf16[M,K]; B=fp4[N,K/2]; wgs,igs f32*. Bsf_sw is the weight
// block scale already in the swizzled SF layout (transformed at load time by the
// framework — see the SF layout contract above). wgs = weight_global_scale (raw);
// the per-token alpha=1/(igs*wgs) is computed inside quant_act (folded, not a
// standalone ZML op).
extern "C" void cutlass_fp4_mm_dyn_sm120a(void* D, const void* x, const void* B,
                                          const void* Bsf_sw, const float* wgs,
                                          const float* igs, int M, int N, int K,
                                          cudaStream_t stream, AllocFn alloc,
                                          void* actx) {
  int kg = K / 16;
  int blocks_k = (kg + 3) / 4;
  size_t szA = (size_t)((M + 127) / 128) * blocks_k * 512;  // swizzled A-scale size
  uint8_t* dq = static_cast<uint8_t*>(alloc(actx, (size_t)M * (K / 2)));
  auto* dAsf = static_cast<cutlass::float_ue4m3_t*>(alloc(actx, szA));
  float* dAlpha = static_cast<float*>(alloc(actx, sizeof(float)));
  // Fused quant + swizzle + alpha: quant_act writes the block scale straight into
  // cutlass's swizzled SF layout AND computes alpha=1/(igs*wgs) — no separate
  // swizzle pass and no per-token ZML alpha scalar op.
  // One block per (row, chunk-of-16-groups); 256 threads = 16 groups x 16 lanes.
  // k = blockIdx.y*256 + threadIdx.x => fully coalesced contiguous reads of x.
  constexpr int kGroupsPerBlock = 16;
  dim3 qgrid(M, (kg + kGroupsPerBlock - 1) / kGroupsPerBlock);
  quant_act_kernel<<<qgrid, kGroupsPerBlock * 16, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), igs, dq, dAsf, M, K, blocks_k, wgs, dAlpha);
  run_fp4_mm<cutlass::bfloat16_t>(D, dq, B, dAsf, Bsf_sw, dAlpha, M, N, K,
                                  stream, alloc, actx);
}

// ------------------------- distinct-value test -------------------------
#ifndef CUTLASS_FP4_MM_NO_MAIN
static float bf16_to_f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

// fp4 e2m1 ALL 16 codes: [sign(1)|exp(2)|mant(1)].
static inline uint8_t fp4code(int idx) { return (uint8_t)(idx & 15); }
static inline float fp4val(uint8_t c) {
  static const float t[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                              -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  return t[c & 15];
}
static inline uint8_t ue4m3enc(float v) {
  int e = 0; float t = v;
  while (t >= 2.f) { t /= 2.f; e++; }
  while (t < 1.f) { t *= 2.f; e--; }
  int mant = (int)((t - 1.f) * 8.f + 0.5f);
  return (uint8_t)(((e + 7) << 3) | (mant & 7));
}
static inline float ue4m3dec(uint8_t b) {
  int es = (b >> 3) & 0xF, m = b & 7;
  return __builtin_ldexpf(1.f + m / 8.f, es - 7);
}

// Host reference implementation of the SF layout contract (see top of file): the
// independent statement of the layout the kernels consume, and the oracle the
// framework-side swizzle (ZML ops.swizzleNvfp4Scale) must agree with byte for byte.
// natural[rows][kg] -> ceil(rows/128)*ceil(kg/4) blocks of 512 bytes, padding = 0.
static std::vector<uint8_t> swizzle_sf_host(const std::vector<uint8_t>& nat, int rows, int kg) {
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

static int runTest(int M, int N, int K);
int main() {
  int rc = 0;
  rc |= runTest(16, 128, 128);
  rc |= runTest(1, 256, 256);   // decode M=1 edge case
  rc |= runTest(1, 3840, 3840); // gemma q/gate shape at M=1
  // Real llmd shapes (all M=256): q,k/v,o,gate/up,down projections.
  rc |= runTest(256, 2048, 3840);
  rc |= runTest(256, 4096, 3840);
  rc |= runTest(256, 3840, 4096);
  rc |= runTest(256, 15360, 3840);
  rc |= runTest(256, 3840, 15360);
  // FULL-verified real-K moderate shapes (catch sparse swizzle errors sampling misses).
  rc |= runTest(256, 512, 3840);   // full: 503M
  rc |= runTest(64, 512, 15360);   // full: 503M, K=15360
  rc |= runTest(180, 256, 4096);   // full: non-128-multiple M, K=4096
  return rc;
}
static int runTest(int M, int N, int K) {
  const int kg = K / 16;
  // Distinct A,B fp4 values + distinct A,B block scales. Validates BOTH operands'
  // data layout, nibble pack order, and scale swizzles vs a full CPU reference.
  // nibble pack: byte j holds element 2j (low) and 2j+1 (high).
  auto aIdx = [](int m, int k) { return (m * 7 + k * 3 + 1) & 15; };
  auto bIdx = [](int n, int k) { return (n * 5 + k * 2 + 2) & 15; };
  std::vector<uint8_t> hA(M * K / 2), hB(N * K / 2), hAsf(M * kg), hBsf(N * kg);
  for (int m = 0; m < M; m++)
    for (int j = 0; j < K / 2; j++)
      hA[m * (K / 2) + j] =
          (uint8_t)(fp4code(aIdx(m, 2 * j)) | (fp4code(aIdx(m, 2 * j + 1)) << 4));
  for (int n = 0; n < N; n++)
    for (int j = 0; j < K / 2; j++)
      hB[n * (K / 2) + j] =
          (uint8_t)(fp4code(bIdx(n, 2 * j)) | (fp4code(bIdx(n, 2 * j + 1)) << 4));
  // WIDE dynamic range scales (2^-6 .. 2^7), like real e4m3(amax*igs/6).
  for (int m = 0; m < M; m++)
    for (int g = 0; g < kg; g++) hAsf[m * kg + g] = ue4m3enc(__builtin_exp2f((float)(((m * 3 + g * 7) % 14) - 6)));
  for (int n = 0; n < N; n++)
    for (int g = 0; g < kg; g++) hBsf[n * kg + g] = ue4m3enc(__builtin_exp2f((float)(((n * 5 + g * 3) % 14) - 6)));
  const float alpha_h = 0.5f;

  // The kernel takes both scales pre-swizzled; do it on the host so the reference
  // below stays independent of any device code.
  std::vector<uint8_t> hAsf_sw = swizzle_sf_host(hAsf, M, kg);
  std::vector<uint8_t> hBsf_sw = swizzle_sf_host(hBsf, N, kg);

  void *dA, *dB, *dAsf, *dBsf, *dD, *dAlpha;
  CUDA_CHECK(cudaMalloc(&dA, hA.size()));
  CUDA_CHECK(cudaMalloc(&dB, hB.size()));
  CUDA_CHECK(cudaMalloc(&dAsf, hAsf_sw.size()));
  CUDA_CHECK(cudaMalloc(&dBsf, hBsf_sw.size()));
  CUDA_CHECK(cudaMalloc(&dD, (size_t)M * N * 2));
  CUDA_CHECK(cudaMalloc(&dAlpha, 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dAsf, hAsf_sw.data(), hAsf_sw.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBsf, hBsf_sw.data(), hBsf_sw.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dAlpha, &alpha_h, 4, cudaMemcpyHostToDevice));

  cutlass_fp4_mm_sm120a(dD, dA, dB, dAsf, dBsf, (const float*)dAlpha, M, N, K, 0,
                        [](void*, size_t sz) -> void* { void* p; cudaMalloc(&p, sz); return p; }, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint16_t> hD(M * N);
  CUDA_CHECK(cudaMemcpy(hD.data(), dD, (size_t)M * N * 2, cudaMemcpyDeviceToHost));
  // Full check when cheap enough, else sample (full CPU ref is O(M*N*K)).
  const bool full = (long long)M * N * K <= 700000000LL;
  int ms_s[4] = {0, M > 1 ? 1 : 0, M / 2, M - 1};
  int nstep = full ? 1 : ((N / 64 > 0) ? N / 64 : 1);
  int mcount = full ? M : 4;
  int bad = 0, checked = 0; float maxrel = 0.f;
  for (int mi = 0; mi < mcount; mi++) {
    int m = full ? mi : ms_s[mi];
    for (int n = 0; n < N; n += nstep) {
      double acc = 0;
      for (int k = 0; k < K; k++) {
        float a = fp4val(fp4code(aIdx(m, k))) * ue4m3dec(hAsf[m * kg + k / 16]);
        float b = fp4val(fp4code(bIdx(n, k))) * ue4m3dec(hBsf[n * kg + k / 16]);
        acc += (double)a * b;
      }
      float expect = alpha_h * (float)acc;
      float got = bf16_to_f(hD[m * N + n]);
      float rel = __builtin_fabsf(got - expect) / (__builtin_fabsf(expect) + 1e-3f);
      if (rel > 0.03f) { if (bad < 8) printf("m%d n%d got %.2f expect %.2f\n", m, n, got, expect); bad++; }
      if (rel > maxrel) maxrel = rel;
      checked++;
    }
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dAsf); cudaFree(dBsf); cudaFree(dD); cudaFree(dAlpha);
  printf("M=%d N=%d K=%d maxrel=%.4f bad=%d/%d %s\n", M, N, K, maxrel, bad,
         checked, bad == 0 ? "GEMM-PASS" : "GEMM-FAIL");
  return bad == 0 ? 0 : 4;
}
#endif  // CUTLASS_FP4_MM_NO_MAIN
