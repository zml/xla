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
#include "xla/service/gpu/metal_kernels/mlx_kernels.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_include_root.h"
#include "xla/service/gpu/metalblas_dispatch.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

absl::StatusOr<std::vector<uint8_t>> CompileMetalblasKernelToMetallib(
    absl::string_view build_flag, absl::string_view family_source,
    absl::Span<const std::pair<absl::string_view, std::string>> subs,
    absl::string_view extra_defines) {
  TF_ASSIGN_OR_RETURN(std::string metal_c, FindMetalTool("metal"));
  TF_ASSIGN_OR_RETURN(std::string metallib, FindMetalTool("metallib"));
  TF_ASSIGN_OR_RETURN(std::string include_root, MetalIncludeRoot());

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
  TF_RETURN_IF_ERROR(RunCommand({metal_c, "-std=metal4.0", "-I", include_root,
                                 "-c", metal_path, "-o", air_path},
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

  // Always mpp_tensor, even for transposed operands: the mpp_gemm kernel
  // decide() would pick for them produces corrupt output on this toolchain.
  std::array<int64_t, 3> tile =
      metalblas_port::pick_m5_tensor_tile(M, N, K, mbdt);
  // Small-BM tile so the clamped prefill grid still fills the GPU. BM must stay
  // >= 16: smaller is out of spec for the cooperative tensor and silently wrong.
  if (prefill_token_axis && M > 1 && N > 1 && M <= 256) {
    tile = {16, 32, 2};
  }
  if (!prefill_token_axis && M > 1 && M <= 16) {
    tile = {16, 32, 2};
  }
  const int64_t BM = tile[0], BN = tile[1], NSG = tile[2];
  const int64_t tiles_m = (M + BM - 1) / BM;
  const int64_t tiles_n = (N + BN - 1) / BN;
  int64_t swz = metalblas_port::round_swizzle_log(tiles_m, tiles_n);
  while (swz > 0 && (int64_t{1} << swz) > tiles_m) --swz;
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
      {"MB_TOKCLAMP", prefill_token_axis ? "1" : "0"}};
  TF_ASSIGN_OR_RETURN(
      std::vector<uint8_t> metallib,
      CompileMetalblasKernelToMetallib("MB_BUILD_MPP_TENSOR", get_mpp_tensor(),
                                       subs));

  const uint32_t lda = static_cast<uint32_t>(trans_a ? M : K);
  const uint32_t ldb = static_cast<uint32_t>(trans_b ? K : N);
  const uint32_t ldc = static_cast<uint32_t>(N);

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
  if (trans_a || !trans_b || (dtype != F16 && dtype != BF16) || M < 2 ||
      M > 16 || N < 512 || N > 8192 || K < 2048 || (K % 16) != 0) {
    return absl::UnimplementedError("metalBLAS splitk: outside the regime.");
  }
  constexpr int64_t BM = 16, BN = 32, BK = 16, WM = 2, WN = 2;
  const int64_t tiles_m = (M + BM - 1) / BM;
  const int64_t tiles_n = (N + BN - 1) / BN;
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
  int64_t tgx = 256;
  while (tgx > 1 && (N % tgx) != 0) tgx >>= 1;
  launch.accum_thread_dim = se::ThreadDim(static_cast<uint64_t>(tgx), 1, 1);
  launch.accum_block_dim = se::BlockDim(static_cast<uint64_t>(N / tgx),
                                        static_cast<uint64_t>(M), 1);
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

  const bool bt = (d.path == "gemv_bt" && !trans_b);
  const bool gt = (d.path == "gemv_t" && M == 1);   // x @ B  (B is K×N)
  const bool nt = (d.path == "gemv_nt" && M == 1);  // x @ Wᵀ (W is N×K, the rhs)

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
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> blib,
        CompileMetalblasKernelToMetallib("MB_BUILD_GEMV_BT", get_gemv_bt(),
                                         bsubs, "#define TRANS_B 1\n"));
    const int64_t span = btw * btc;
    MetalGemmLaunch launch;
    launch.metallib = std::move(blib);
    launch.kernel_name = "gemv_bt";
    launch.thread_dim = se::ThreadDim(static_cast<uint64_t>(32 * btw), 1, 1);
    launch.block_dim =
        se::BlockDim(static_cast<uint64_t>((N + span - 1) / span), 1, 1);
    launch.swap_ab = true;  // matrix W (rhs) -> buffer(0), X (lhs) -> buffer(1)
    launch.params = {static_cast<uint32_t>(N), static_cast<uint32_t>(K),
                     static_cast<uint32_t>(K), static_cast<uint32_t>(K),
                     static_cast<uint32_t>(N), 0};
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
    launch.params = {n, k, /*ldb=*/n, /*ldx=*/k, /*ldy=*/n, 0};
  } else if (nt) {
    launch.params = {n, k, static_cast<uint32_t>(in.ldb), 0, 0, 0};
  } else {
    launch.params = {n, k, /*ldb=*/n, /*xs=*/1, 0, 0};
  }
  return launch;
}

}  // namespace gpu
}  // namespace xla
