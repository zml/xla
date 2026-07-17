// MLX steel split-K GEMM (MIT License, Copyright (c) 2023 Apple Inc.,
// https://github.com/ml-explore/mlx). Entry source for the mlx_steel_splitk
// bundle.
//
// The upstream text is NOT checked in. The three #includes below resolve
// against the @mlx archive pinned in //third_party/mlx:workspace.bzl, which
// MetalIncludeRoot() writes out once per process and passes to the Metal
// compiler as -I. So the compiler reads upstream's bytes and "verbatim" is true
// by construction -- there is no generated copy in this repo for anyone to
// hand-edit, and an edit under bazel-out/ dies with the next build. To change
// upstream's bytes, add a patch to //third_party/mlx:series.bzl, where a bump
// re-verifies it at fetch time; a fork you cannot express as a patch is a fork
// nothing re-verifies.
//
// The includes are upstream's own prologue, copied from
// mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.metal:4-6.
// They are the include recipe, and the order is load-bearing: MLX's leaf kernel
// headers carry no #pragma once and no self-includes (steel_gemm_splitk.h opens
// on a bare `using namespace mlx::steel;`), so they only compile behind the
// prologue their own .metal file establishes. Every header reachable more than
// once in the DAG does have #pragma once, which is what makes plain -I dedup
// correctly here -- exactly as it does for upstream's own AOT build.
//
// Two kernels: gemm_splitk (grid-level K-split, tgid.z = K-partition, writes
// f32 partial planes) + gemm_splitk_accum (sums planes into the output dtype).
// Stamped per shape via the __TOKEN__s below, like the other bundle families.
//
// NOTE: this bundle is a FRAGMENT, not a standalone TU, which is why it is a .h
// and not a .metal. CompileMetalblasKernelToMetallib concatenates it after
// metalBLAS's mb_epi.h and token-substitutes globally across both, so
// `metal -I <mlx> <this file>` does not type-check it. Compile it through
// metalblas_gemm.cc.
#ifdef MB_BUILD_MLX_SPLITK
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h"
// ---- xla bundle instantiations (token-substituted per compile) ----
template [[host_name("gemm_splitk")]] [[kernel]] void
gemm_splitk<__IN_T__, float, __BM__, __BN__, 16, 2, 2, __TRANS_A__, __TRANS_B__,
            __MN_ALIGNED__, __K_ALIGNED__>(
    const device __IN_T__*, const device __IN_T__*, device float*,
    const constant GEMMSpiltKParams*, uint, uint, uint3, uint3);

template [[host_name("gemm_splitk_accum")]] [[kernel]] void
gemm_splitk_accum<float, __IN_T__>(
    const device float*, device __IN_T__*, const constant int&,
    const constant int&, const constant int&, uint2);
#endif  // MB_BUILD_MLX_SPLITK
