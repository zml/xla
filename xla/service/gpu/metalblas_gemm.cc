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

#include "xla/service/gpu/metalblas_gemm.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/gpu/metal_kernels/metalblas_shaders.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metalblas_dispatch.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

// Assemble + compile one metalBLAS MSL kernel family to a metallib. Each family
// header is `#ifdef MB_BUILD_<NAME>`-guarded and the only always-active header is
// the shared epilogue mb_epi.h, so the compile unit is exactly
// `get_mb_epi() + family_source` with `#define <build_flag> 1` — byte-identical
// to compiling the upstream binder with that one flag (the binder just #includes
// every family and lets the #ifdefs select one). `__TOKEN__` tile/dtype
// placeholders are substituted, then compiled via the Metal compiler
// (`metal -std=metal4.0`, required for the cooperative-tensor GEMM families)
// → metallib.
absl::StatusOr<std::vector<uint8_t>> CompileMetalblasKernelToMetallib(
    absl::string_view build_flag, absl::string_view family_source,
    absl::Span<const std::pair<absl::string_view, std::string>> subs,
    absl::string_view extra_defines) {
  TF_ASSIGN_OR_RETURN(std::string metal_c, FindMetalTool("metal"));
  TF_ASSIGN_OR_RETURN(std::string metallib, FindMetalTool("metallib"));

  std::string src = absl::StrCat("#define ", build_flag, " 1\n", extra_defines,
                                 get_mb_epi(), "\n", family_source);
  for (const auto& [tok, val] : subs) {
    src = absl::StrReplaceAll(src, {{absl::StrCat("__", tok, "__"), val}});
  }

  tsl::Env* env = tsl::Env::Default();
  std::string base;
  if (!env->LocalTempFilename(&base)) {
    return absl::InternalError("Could not create metalBLAS temp filename.");
  }
  std::string metal_path = absl::StrCat(base, ".metal");
  std::string air_path = absl::StrCat(base, ".air");
  std::string metallib_path = absl::StrCat(base, ".metallib");
  absl::Cleanup cleanup = [&] {
    env->DeleteFile(metal_path).IgnoreError();
    env->DeleteFile(air_path).IgnoreError();
    env->DeleteFile(metallib_path).IgnoreError();
  };

  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, metal_path, src));
  TF_RETURN_IF_ERROR(
      RunCommand({metal_c, "-std=metal4.0", "-c", metal_path, "-o", air_path},
                 false)
          .status());
  TF_RETURN_IF_ERROR(
      RunCommand({metallib, air_path, "-o", metallib_path}, false).status());

  std::string bytes;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, metallib_path, &bytes));
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

absl::StatusOr<MetalGemmLaunch> CompileMetalblasGemm(int64_t M, int64_t N,
                                                     int64_t K, bool trans_a,
                                                     bool trans_b,
                                                     PrimitiveType dtype,
                                                     bool prefill_token_axis) {
  if (dtype != F32 && dtype != F16 && dtype != BF16) {
    return absl::UnimplementedError(
        absl::StrCat("metalBLAS GEMM supports only f32/f16/bf16; got ",
                     PrimitiveType_Name(dtype), "."));
  }

  const metalblas_port::Dtype mbdt =
      dtype == F32   ? metalblas_port::Dtype::kFloat32
      : dtype == F16 ? metalblas_port::Dtype::kFloat16
                     : metalblas_port::Dtype::kBfloat16;

  // The mpp_tensor cooperative-tensor kernel (metalBLAS's primary GEMM backend)
  // handles NN, NT, TN and TT: it builds matmul2d tensor views whose extents and
  // tile slices are selected per TRANS_A/TRANS_B (STATIC_SLICE=1 takes the
  // non-transposed static-extent fast path). Verified vs CPU for NN/NT/TN on
  // non-square shapes. We route ALL transposes here rather than the dispatcher's
  // choice — decide() sends transposed GEMMs to the 6-tuple mpp_gemm, whose
  // hand-rolled fragment indexing is wrong on this toolchain (an 8-row-periodic
  // corruption, reproduced standalone), whereas mpp_tensor uses the matmul2d
  // library's own (correct) fragment store.
  std::array<int64_t, 3> tile =
      metalblas_port::pick_m5_tensor_tile(M, N, K, mbdt);
  // PREFILL token-axis GEMMs are compiled at M=seqlen but clamped to ~num_tokens
  // rows at execute (MetalGemmThunk grid clamp). pick_m5_tensor_tile sees the
  // padded M=seqlen (e.g. 128) and returns a large-BM tile ({64,64,2} for bf16)
  // that launches FEW threadgroups once the grid is clamped to a short prompt →
  // ~20% GPU occupancy, ~12-23% of peak BW. A small-BM, BN=32 tile launches many
  // more threadgroups (tiles_m up from 1 to ceil(nt/16); tiles_n doubled) → the
  // clamped grid actually fills the GPU. Measured on Llama-3.2-3B (17-tok prompt,
  // seqlen 128): 98ms→63ms warm, +1.73x, OUTPUT BIT-IDENTICAL to the {64,64,2}
  // golden (fp32 accumulate; valid BM>=16 only — BM<16 is out-of-spec for the
  // cooperative tensor and produces WRONG results). BM stays >=16.
  //
  // BUT only for SMALL prefill buckets (M<=256). The grid-filling win above
  // assumes the clamped grid is starved — true at seqlen 128 / a short prompt,
  // false for the large prefill buckets llmd emits when max_token_count>256
  // (M up to 2048). There BM=16 re-reads the weights 4-8x vs pick's BM=64
  // ({64,64,2}) and goes memory-bound (~6 TFLOP/s measured on Qwen3.5-2B's
  // 970-token chunks). Large buckets only ever serve large requests (the chunk
  // ladder picks the smallest bucket >= the prompt), so their clamped grid is
  // already full — keep pick_m5_tensor_tile's large-BM tile for weight reuse.
  // kRowGranularity (256) covers pick's max BM (128), so the prefill row clamp
  // still holds.
  if (prefill_token_axis && M > 1 && N > 1 && M <= 256) {
    tile = {16, 32, 2};
  }
  // Batched-decode thin-M GEMMs — EVERY shape the gemv_bt route declines
  // (wide-N: llmd's gate/up [16,3072]→8192 and the [16,3072]→128256 lm_head;
  // deep-K: an 8B-class down proj [16,14336]→4096). Same starved-grid disease
  // as the clamped prefill above, same cure: BN=32 doubles tiles_n (gate/up
  // 128→256 threadgroups), and at M<=16 a BM=64 tile wastes 48/64 rows by
  // construction, so {16,32,2} strictly dominates. Measured (llmd 2026-06-11):
  // gate/up 143→133us (378 GB/s), lm_head 1903→1853us (425 GB/s ≈ peak).
  // Deliberately NOT
  // N-gated: the 2026-06-10 harness cross-model sweep shows gemv_bt sags to
  // ~300 GB/s at K>8192 (X re-read scales with K) while this tile holds ~378,
  // so the K>8192 deep-K shapes belong here, and anything else that slips
  // both routes still gets a filled grid instead of tiles_n<=64. BM=16 is the
  // cooperative-tensor validity floor (BM<16 produces WRONG results).
  // This mirrors the dispatcher's PORT POLICY #3 (metalblas_dispatch.cc
  // decide_gemm_float / gen_golden.py) — applied here too because this
  // function takes the tile from pick_m5_tensor_tile directly, which the
  // policy deliberately leaves untouched (candidates-record parity).
  if (!prefill_token_axis && M > 1 && M <= 16) {
    tile = {16, 32, 2};
  }
  const int64_t BM = tile[0], BN = tile[1], NSG = tile[2];
  const int64_t tiles_m = (M + BM - 1) / BM;
  const int64_t tiles_n = (N + BN - 1) / BN;
  int64_t swz = metalblas_port::round_swizzle_log(tiles_m, tiles_n);
  // Skinny-M (tiles_m < 2^swz) makes the swizzled grid over-dispatch: gx =
  // tiles_n<<swz, gy = ceil(tiles_m/2^swz) rounds up, so up to 2^swz x the real
  // tiles launch (the excess early-exit on the tgid bound, but still occupy launch
  // slots). Cap swz so 2^swz <= tiles_m, i.e. never dispatch empty m-blocks.
  while (swz > 0 && (int64_t{1} << swz) > tiles_m) --swz;
  // PREFILL token-axis GEMMs: force swz=0. The swizzle is BAKED from the compiled
  // M=seqlen tiles_m, but at execute the M-row grid is CLAMPED to ~num_tokens. A
  // nonzero swz packs M-tiles into gx's low bits (tgy = tgid.y<<swz | tgid.x&mask),
  // so clamping gy does NOT shrink the M-tiles actually dispatched — up to 2^swz
  // M-tiles still launch, and EACH re-reads the full B (weight) slice for its N-tile.
  // For a 17-token prompt that is ~4x redundant weight traffic. swz=0 makes the
  // clamped gy directly bound the M-tiles → ~2x fewer weight reads. Measured: warm
  // prefill 50.9ms→41.2ms, bit-identical. (The swizzle's L2 win is moot here — only
  // 1-2 M-tiles.)
  if (prefill_token_axis && M > 1 && N > 1) {
    swz = 0;
  }

  const char* t = dtype == F32 ? "float" : dtype == F16 ? "half" : "bfloat";
  const std::vector<std::pair<absl::string_view, std::string>> subs = {
      {"IN_T", t},
      {"OUT_T", t},
      {"BM", absl::StrCat(BM)},
      {"BN", absl::StrCat(BN)},
      {"NSG", absl::StrCat(NSG)},
      {"TRANS_A", trans_a ? "true" : "false"},
      {"TRANS_B", trans_b ? "true" : "false"},
      {"RELAXED", "true"},
      {"SWIZZLE_LOG", absl::StrCat(swz)},
      {"MN_ALIGNED", (M % BM == 0 && N % BN == 0) ? "1" : "0"},
      {"STATIC_SLICE", (!trans_a && !trans_b) ? "1" : "0"},
      // Prefill token-axis GEMMs take a device num_tokens arg (buffer 4) and
      // clamp the M grid ON-GPU (the host dispatches the full static grid), so
      // there is no host-side encode-time read of GPU-produced metadata to race.
      {"MB_TOKCLAMP", prefill_token_axis ? "1" : "0"}};
  TF_ASSIGN_OR_RETURN(
      std::vector<uint8_t> metallib,
      CompileMetalblasKernelToMetallib("MB_BUILD_MPP_TENSOR", get_mpp_tensor(),
                                       subs));

  // Row-major leading dims for MBTensorDims{M,N,K,lda,ldb,ldc}: the contiguous
  // stride of each operand's physical layout. lda = trans_a ? M : K (A is
  // [K,M] or [M,K]); ldb = trans_b ? K : N (B is [N,K] or [K,N]); ldc = N.
  const uint32_t lda = static_cast<uint32_t>(trans_a ? M : K);
  const uint32_t ldb = static_cast<uint32_t>(trans_b ? K : N);
  const uint32_t ldc = static_cast<uint32_t>(N);

  // Swizzled grid (the kernel un-swizzles tgid via SWIZZLE_LOG): grid.x =
  // tiles_n << swz, grid.y = ceil(tiles_m / (1<<swz)); threadgroup = NSG warps.
  const int64_t gx = tiles_n << swz;
  const int64_t gy = (tiles_m + (int64_t{1} << swz) - 1) >> swz;
  MetalGemmLaunch launch;
  launch.metallib = std::move(metallib);
  launch.kernel_name = "mpp_tensor_gemm";
  launch.thread_dim = se::ThreadDim(static_cast<uint64_t>(NSG * 32), 1, 1);
  launch.block_dim = se::BlockDim(static_cast<uint64_t>(gx),
                                  static_cast<uint64_t>(gy), 1);
  launch.params = {static_cast<uint32_t>(M), static_cast<uint32_t>(N),
                   static_cast<uint32_t>(K), lda, ldb, ldc};
  launch.bm = static_cast<uint32_t>(BM);
  launch.swizzle_log = static_cast<uint32_t>(swz);
  return launch;
}

absl::StatusOr<MetalGemmLaunch> CompileMetalblasSplitk(int64_t M, int64_t N,
                                                       int64_t K, bool trans_a,
                                                       bool trans_b,
                                                       PrimitiveType dtype) {
  // Regime: thin-M batched-decode x[M,K] @ W[N,K]ᵀ, f16/bf16. Bounds are
  // harness data (see the .h): N <= 8192 (lm_head-scale N is a wash on the
  // {16,32,2} tensor tile, and skipping it avoids a multi-MB staging buffer);
  // N >= 512 / K >= 2048 keep the small shapes on gemv_bt (unmeasured here,
  // staging+accum overhead is a larger fraction); K%16==0 is the kernel's
  // K_aligned contract combined with the partition math below (a K%16 tail
  // inside every partition is not what MLX's host code ever dispatches).
  if (trans_a || !trans_b || (dtype != F16 && dtype != BF16) || M < 2 ||
      M > 16 || N < 512 || N > 8192 || K < 2048 || (K % 16) != 0) {
    return absl::UnimplementedError("metalBLAS splitk: outside the regime.");
  }
  constexpr int64_t BM = 16, BN = 32, BK = 16, WM = 2, WN = 2;
  const int64_t tiles_m = (M + BM - 1) / BM;
  const int64_t tiles_n = (N + BN - 1) / BN;
  // Partition count: target ~512 total threadgroups, scaled by K depth —
  // clamp(next_pow2(512*K / (tiles*4096)), 2, 32) reproduces the per-shape
  // harness winners within ~5% (kv 16, qo 4, down 8→16, gateup 2, 8B-down 16+).
  // Then shrink until it divides the K-tile count exactly (uniform partitions).
  int64_t parts = (512 * K) / (tiles_n * tiles_m * 4096);
  int64_t p2 = 2;
  while (p2 < parts && p2 < 32) p2 <<= 1;
  parts = p2;
  while (parts > 2 && ((K / BK) % parts) != 0) parts >>= 1;
  if (((K / BK) % parts) != 0) {
    return absl::UnimplementedError("metalBLAS splitk: K does not partition.");
  }
  const int64_t k_iters = (K / BK) / parts;

  const char* t = dtype == F16 ? "half" : "bfloat16_t";
  const bool mn_aligned = (M % BM == 0) && (N % BN == 0);
  const std::vector<std::pair<absl::string_view, std::string>> subs = {
      {"IN_T", t},
      {"BM", absl::StrCat(BM)},
      {"BN", absl::StrCat(BN)},
      {"TRANS_A", "false"},
      {"TRANS_B", "true"},
      {"MN_ALIGNED", mn_aligned ? "true" : "false"},
      {"K_ALIGNED", "true"}};
  TF_ASSIGN_OR_RETURN(
      std::vector<uint8_t> metallib,
      CompileMetalblasKernelToMetallib("MB_BUILD_MLX_SPLITK",
                                       get_mlx_steel_splitk(), subs));

  MetalGemmLaunch launch;
  launch.metallib = std::move(metallib);
  launch.kernel_name = "gemm_splitk";
  launch.thread_dim = se::ThreadDim(WM * WN * 32, 1, 1);
  launch.block_dim = se::BlockDim(static_cast<uint64_t>(tiles_n),
                                  static_cast<uint64_t>(tiles_m),
                                  static_cast<uint64_t>(parts));
  // GEMMSpiltKParams (mlx steel params.h:34-52): nt → lda = ldb = K (x rows and
  // W columns both K-contiguous), ldc = N over the f32 staging planes.
  launch.splitk_params = {
      static_cast<uint32_t>(M),       static_cast<uint32_t>(N),
      static_cast<uint32_t>(K),       static_cast<uint32_t>(K),
      static_cast<uint32_t>(K),       static_cast<uint32_t>(N),
      static_cast<uint32_t>(tiles_n), static_cast<uint32_t>(tiles_m),
      static_cast<uint32_t>(parts),   static_cast<uint32_t>(M * N),
      static_cast<uint32_t>(k_iters * BK),
      /*swizzle_log=*/0,              static_cast<uint32_t>(k_iters)};
  launch.splitk_partitions = static_cast<uint32_t>(parts);
  launch.staging_bytes = static_cast<uint64_t>(parts) * M * N * 4;
  launch.accum_kernel_name = "gemm_splitk_accum";
  launch.accum_params = {static_cast<uint32_t>(parts),
                         static_cast<uint32_t>(M * N),
                         static_cast<uint32_t>(N)};
  // Accum grid: gid.x over N, gid.y over M (exact). Our launch path dispatches
  // threadgroups, so pick the largest power-of-two x-group that divides N.
  int64_t tgx = 256;
  while (tgx > 1 && (N % tgx) != 0) tgx >>= 1;
  launch.accum_thread_dim = se::ThreadDim(static_cast<uint64_t>(tgx), 1, 1);
  launch.accum_block_dim = se::BlockDim(static_cast<uint64_t>(N / tgx),
                                        static_cast<uint64_t>(M), 1);
  // params.M is splitk_params[0]; bm stays 0 -> no prefill M-clamp (decode-only
  // regime, M is compile-time).
  return launch;
}

absl::StatusOr<MetalGemmLaunch> CompileMetalblasGemv(int64_t M, int64_t N,
                                                     int64_t K, bool trans_a,
                                                     bool trans_b,
                                                     PrimitiveType dtype,
                                                     int64_t b_byte_offset) {
  if (dtype != F32 && dtype != F16 && dtype != BF16) {
    return absl::UnimplementedError("metalBLAS GEMV: only f32/f16/bf16.");
  }
  const int64_t elem = (dtype == F32) ? 4 : 2;

  // Ask the imported (golden-verified) dispatcher what to do. Row-major terms:
  // op(A)=M×K is the lhs (vector x / thin X), op(B)=K×N is the rhs (matrix B);
  // lda/ldb are the contiguous leading dims, and offset is B's element offset,
  // which decide() ORs into the VEC-load alignment clamp when it picks vec/nwarps.
  metalblas_port::Inputs in;
  in.m = M;
  in.n = N;
  in.k = K;
  in.dtype = dtype == F32   ? metalblas_port::Dtype::kFloat32
             : dtype == F16 ? metalblas_port::Dtype::kFloat16
                            : metalblas_port::Dtype::kBfloat16;
  in.trans_a = trans_a;
  in.trans_b = trans_b;
  in.lda = trans_a ? M : K;
  in.ldb = trans_b ? K : N;
  in.offset = b_byte_offset / elem;
  const metalblas_port::Decision d = metalblas_port::decide(in);

  // GEMV kernels whose matrix is the rhs B over a contiguous operand, all of
  // which put the matrix (rhs) at buffer(0) and the vector/X (lhs) at buffer(1)
  // (swap_ab). decide() chooses among:
  //   gemv_t  -- M==1, !trans_b: y = x @ B, B stored [K×N] (pre-transposed weight)
  //   gemv_nt -- M==1,  trans_b: y = x @ Wᵀ, W stored [out,in]=[N×K] -- the
  //              STANDARD decode/lm_head linear (x.dot(weight,.d) on a [.dout,.d]
  //              weight); one simdgroup per output row, simd_sum over K
  //   gemv_bt -- thin-M (2..16) low-precision X @ B
  // The N==1 gemv_nt/gemv_t variants (matrix = lhs A, would need A's offset for the
  // VEC clamp) and every non-GEMV decision -> Unimplemented -> mpp_tensor fallback.
  const bool bt = (d.path == "gemv_bt" && !trans_b);
  const bool gt = (d.path == "gemv_t" && M == 1);   // x @ B  (B is K×N)
  const bool nt = (d.path == "gemv_nt" && M == 1);  // x @ Wᵀ (W is N×K, the rhs)

  // gemv_bt TRANS_B=1: transposed thin-M decode batches x[M,K] @ W[N,K]ᵀ —
  // llmd's batch_size=16 q/k/v/o + MLP-down projections. The ROUTING + SPEC
  // (vec, nwarps, ncols, launch dims) now come from the dispatcher port's
  // decide() (PORT POLICY #2, metalblas_dispatch.cc / gen_golden.py), measured
  // on the 2026-06-10 M4 Max sweep: kv [16,3072]→1024 78→25us (×3.1), qo
  // [16,3072]→3072 133avg→51us, down [16,8192]→3072 228→154us; llmd 29.7→36.5
  // tok/s e2e. This block only TRANSLATES the decision into a TRANS_B kernel
  // build.
  const bool bt_trans = (d.path == "gemv_bt" && trans_b);
  const int64_t btv = d.vec, btw = d.nwarps, btc = d.ncols;
  if (bt_trans) {
    const char* bt_t = dtype == F16 ? "half" : "bfloat";
    const std::vector<std::pair<absl::string_view, std::string>> bsubs = {
        {"IN_T", bt_t},
        {"ACC_T", "float"},
        {"OUT_T", bt_t},
        {"NWARPS", absl::StrCat(btw)},
        {"VEC", absl::StrCat(btv)},
        {"BLOCK_N", absl::StrCat(32 * btv)},
        {"MROWS", absl::StrCat(M)},
        {"NCOLS", absl::StrCat(btc)}};
    // TRANS_B is an #ifndef default in gemv_bt.h (not a __TOKEN__), so it goes
    // in as an extra define ahead of the binder.
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> blib,
        CompileMetalblasKernelToMetallib("MB_BUILD_GEMV_BT", get_gemv_bt(),
                                         bsubs, "#define TRANS_B 1\n"));
    // Kernel column ownership: n(base) = (tgid.x * NWARPS + sgid) * NCOLS.
    const int64_t span = btw * btc;
    MetalGemmLaunch launch;
    launch.metallib = std::move(blib);
    launch.kernel_name = "gemv_bt";
    launch.thread_dim = se::ThreadDim(static_cast<uint64_t>(32 * btw), 1, 1);
    launch.block_dim =
        se::BlockDim(static_cast<uint64_t>((N + span - 1) / span), 1, 1);
    launch.swap_ab = true;  // matrix W (rhs) -> buffer(0), X (lhs) -> buffer(1)
    // MBGemvBtDims{N, K, ldb, ldx, ldy}: TRANS_B → ldb = K (W[N,K] row-major,
    // columns K-contiguous); X row-major ldx = K; Y row-major ldy = N.
    launch.params = {static_cast<uint32_t>(N), static_cast<uint32_t>(K),
                     static_cast<uint32_t>(K), static_cast<uint32_t>(K),
                     static_cast<uint32_t>(N), 0};
    // bm stays 0 -> the thunk's prefill M-clamp is skipped (all MROWS rows are
    // compile-time; the kernel is N-parallel, M never appears in the grid).
    return launch;
  }

  if (!gt && !bt && !nt) {
    return absl::UnimplementedError(absl::StrCat(
        "metalBLAS GEMV: decide() chose ", d.path,
        M == 1 ? "." : " (N==1 matrix=A path not wired)."));
  }
  const int64_t vec = d.vec, nwarps = d.nwarps;
  if (vec <= 0 || nwarps <= 0 || d.group[0] <= 0) {
    return absl::UnimplementedError("metalBLAS GEMV: degenerate launch.");
  }

  // dtype profile (metalblas/dispatch.py::_PROFILE): (IN_T, ACC_T, OUT_T).
  const char* in_t = dtype == F32 ? "float" : dtype == F16 ? "half" : "bfloat";
  std::vector<std::pair<absl::string_view, std::string>> subs = {
      {"IN_T", in_t},
      {"ACC_T", "float"},
      {"OUT_T", in_t},
      {"NWARPS", absl::StrCat(nwarps)},
      {"VEC", absl::StrCat(vec)}};
  absl::string_view build_flag;
  absl::string_view family_src;
  if (nt) {
    // gemv_nt: ROWS_PER_SG=1 row/warp, simd_sum over K (RED_TG=0 -- floats have
    // simd_sum; no BLOCK_N).
    build_flag = "MB_BUILD_GEMV_NT";
    family_src = get_gemv_nt();
    subs.push_back({"ROWS_PER_SG", "1"});
    subs.push_back({"RED_TG", "0"});
  } else {
    build_flag = bt ? "MB_BUILD_GEMV_BT" : "MB_BUILD_GEMV_T";
    family_src = bt ? get_gemv_bt() : get_gemv_t();
    subs.push_back({"BLOCK_N", absl::StrCat(32 * vec)});
    if (bt) {
      subs.push_back({"MROWS", absl::StrCat(M)});
      subs.push_back({"NCOLS", "1"});
    }
  }
  TF_ASSIGN_OR_RETURN(
      std::vector<uint8_t> metallib,
      CompileMetalblasKernelToMetallib(build_flag, family_src, subs));

  // Launch dims straight from the dispatcher: thread_dim = the threadgroup
  // (group), block_dim.x = total threads / threadgroup.
  const int64_t tg = d.group[0];
  const int64_t ng = d.threads[0] / tg;
  MetalGemmLaunch launch;
  launch.metallib = std::move(metallib);
  launch.kernel_name = nt ? "gemv_nt" : bt ? "gemv_bt" : "gemv_t";
  launch.thread_dim = se::ThreadDim(static_cast<uint64_t>(tg), 1, 1);
  launch.block_dim = se::BlockDim(static_cast<uint64_t>(ng), 1, 1);
  launch.swap_ab = true;  // matrix (rhs) -> buffer(0), vector/X (lhs) -> buffer(1)
  const uint32_t n = static_cast<uint32_t>(N), k = static_cast<uint32_t>(K);
  if (bt) {
    // MBGemvBtDims{N, K, ldb, ldx, ldy} (the kernel reads the first 5 words).
    launch.params = {n, k, /*ldb=*/n, /*ldx=*/k, /*ldy=*/n, 0};
  } else if (nt) {
    // int4 gP{gM, gK, gLda}: the matrix is W[N×K] (rhs), so gM=N rows over the
    // output, gLda = ldb (= K for trans_b). gXs is implicitly 1 (contiguous x).
    launch.params = {n, k, static_cast<uint32_t>(in.ldb), 0, 0, 0};
  } else {
    // int4 gP{gN, gK, gLdb, gXs} (the kernel reads the first 4 words).
    launch.params = {n, k, /*ldb=*/n, /*xs=*/1, 0, 0};
  }
  // bm stays 0 -> the thunk's prefill M-clamp is skipped.
  return launch;
}

}  // namespace gpu
}  // namespace xla
