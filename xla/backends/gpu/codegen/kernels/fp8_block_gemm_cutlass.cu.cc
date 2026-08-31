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

// The GEMM configuration here is vLLM's, from
// csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8_dispatch.cuh
// in vllm-project/vllm at v0.28.1rc0 (Apache 2.0), which is the kernel that
// serves block-128 FP8 checkpoints in production. Everything from ElementAB
// down to the two CollectiveBuilder instantiations is theirs; what changed is
// the boundary. Theirs takes torch tensors, allocates its own workspace and
// picks a tile from a hand-written heuristic. Ours takes device pointers and a
// stream, is handed a workspace, and exposes the tile space so the autotuner
// picks per fusion -- which is the point, because the heuristic has a cliff:
// `swap_ab = (m < 16) || (m % 4 != 0)` leaves m == 16 on the plain path, where
// the occupancy test then selects a 256-row tile for a 16-row problem. On
// gate_up that measured 69.1us against 33.4us for a swapped 128x32 tile.

#include "xla/backends/gpu/codegen/kernels/fp8_block_gemm_cutlass.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/packed_stride.hpp"

namespace xla::gpu::kernel {
namespace {

using namespace cute;  // NOLINT(build/namespaces) -- CUTLASS's own convention.

// A kernel built for one architecture range traps rather than misbehaves if it
// is ever launched outside it. vLLM's cutlass_extensions/common.hpp.
template <typename Kernel>
struct EnableSm100ToSm120 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)
    Kernel::operator()(std::forward<Args>(args)...);
#else
    asm("trap;");
#endif
#endif
  }
};

// ScaleGranularity{M,N,K} describe the checkpoint's scale blocking, not the
// tile: an activation carries one scale per row per 128 of K, a weight one per
// 128x128. Under swap A/B the operands change places, so the granularities do
// too -- the swapped M axis is the weight's N (128) and the swapped N axis is
// the activation's M (1). They are therefore fixed by the data, and only
// MmaTileShape and ClusterShape are free.
template <class OutType, int ScaleGranularityM, int ScaleGranularityN,
          int ScaleGranularityK, class MmaTileShape, class ClusterShape,
          class EpilogueScheduler, class MainloopScheduler,
          bool swap_ab_ = false>
struct Fp8BlockwiseGemm {
  static constexpr bool swap_ab = swap_ab_;
  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  // Both scales K-major, in either orientation: that is the layout the
  // frontend already produces for each of them, so neither is transposed on
  // the way in. Major::K is not a different tiling of the same buffer, it is
  // the row-major stride -- blockwise_scale_layout.hpp:222 makes SFA
  // ((1,M),(128,K/128)):((0,K/128),(0,1)) -- and the mainloop deduces which it
  // was handed from the stride
  // (sm100_mma_warpspecialized_blockwise_scaling.hpp:150), so it serves both.
  //
  // Measured on sm_103a against a CPU reference at m in
  // {1,3,16,17,32,33,64,129,512,2048}: every tile below is correct K-major,
  // and, unlike MN-major, none of them declines an m that is not a multiple of
  // 4 -- that alignment rule was the MN-major scale copy's, not the GEMM's.
  // The one width it cannot serve is a swapped tile 16 wide, whose scale tile
  // is 16x1 and whose copy vectorizes along the axis holding one element: it
  // reads neighbouring rows (wrong at 128x16) or off the end (an illegal
  // address at 256x16). Those two entries are therefore not in the table.
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      cute::UMMA::Major::K, cute::UMMA::Major::K>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using ElementScalar = float;
  using DefaultOperation =
      cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute,
                                                   ElementC, ElementScalar,
                                                   RoundStyle>;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC,
          conditional_t<swap_ab, LayoutC_Transpose, LayoutC>, AlignmentC,
          ElementD, conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
          AlignmentD, EpilogueScheduler, DefaultOperation>::CollectiveOp;

  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementB,
          cute::tuple<LayoutB_Transpose, LayoutSFA>, AlignmentB, ElementA,
          cute::tuple<LayoutA_Transpose, LayoutSFB>, AlignmentA,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, cute::tuple<LayoutA, LayoutSFA>,
          AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler>::CollectiveOp>;

  using KernelType = EnableSm100ToSm120<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

// The argument assembly is vLLM's cutlass_gemm_caller_blockwise, with the
// tensors replaced by pointers. The swap A/B branch is load bearing and stays
// exactly as written: under a swap the problem is transposed, the operands and
// their scale pointers trade places, and the C/D strides describe [N, M].
template <typename Gemm>
struct Runner {
  using GemmKernel = typename Gemm::GemmKernel;
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using LayoutSFA = typename Gemm::LayoutSFA;
  using LayoutSFB = typename Gemm::LayoutSFB;
  using ScaleConfig = typename Gemm::ScaleConfig;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using ElementBlockScale = typename Gemm::ElementBlockScale;
  static constexpr bool swap_ab = Gemm::swap_ab;

  static typename GemmKernel::Arguments MakeArguments(
      const Fp8BlockGemmCutlassParams& p) {
    const int m = static_cast<int>(p.m);
    const int n = static_cast<int>(p.n);
    const int k = static_cast<int>(p.k);

    StrideA a_stride =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    StrideB b_stride =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    StrideC c_stride = cutlass::make_cute_packed_stride(
        StrideC{}, swap_ab ? cute::make_shape(n, m, 1)
                           : cute::make_shape(m, n, 1));

    LayoutSFA layout_SFA =
        swap_ab ? ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1))
                : ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    LayoutSFB layout_SFB =
        swap_ab ? ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1))
                : ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

    auto a_ptr = static_cast<const ElementAB*>(p.a);
    auto b_ptr = static_cast<const ElementAB*>(p.b);
    auto a_scales_ptr = static_cast<const ElementBlockScale*>(p.a_scales);
    auto b_scales_ptr = static_cast<const ElementBlockScale*>(p.b_scales);

    typename GemmKernel::MainloopArguments mainloop_args{};
    mainloop_args.layout_SFA = layout_SFA;
    mainloop_args.layout_SFB = layout_SFB;
    if (swap_ab) {
      mainloop_args.ptr_A = b_ptr;
      mainloop_args.dA = b_stride;
      mainloop_args.ptr_B = a_ptr;
      mainloop_args.dB = a_stride;
      mainloop_args.ptr_SFA = b_scales_ptr;
      mainloop_args.ptr_SFB = a_scales_ptr;
    } else {
      mainloop_args.ptr_A = a_ptr;
      mainloop_args.dA = a_stride;
      mainloop_args.ptr_B = b_ptr;
      mainloop_args.dB = b_stride;
      mainloop_args.ptr_SFA = a_scales_ptr;
      mainloop_args.ptr_SFB = b_scales_ptr;
    }

    auto prob_shape =
        swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);
    auto c_ptr = static_cast<ElementD*>(p.d);
    typename GemmKernel::EpilogueArguments epilogue_args{
        {}, c_ptr, c_stride, c_ptr, c_stride};

    cutlass::KernelHardwareInfo hw_info;
    return typename GemmKernel::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args,
        epilogue_args, hw_info};
  }

  static size_t WorkspaceSize(int64_t m, int64_t n, int64_t k) {
    Fp8BlockGemmCutlassParams p;
    p.m = m;
    p.n = n;
    p.k = k;
    GemmOp gemm_op;
    return gemm_op.get_workspace_size(MakeArguments(p));
  }

  static int Run(const Fp8BlockGemmCutlassParams& p, void* stream,
                 const char** error) {
    typename GemmKernel::Arguments args = MakeArguments(p);
    GemmOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      if (error != nullptr) *error = cutlass::cutlassGetStatusString(status);
      return 1;
    }
    status = gemm_op.run(args, p.workspace,
                         static_cast<cudaStream_t>(stream));
    if (status != cutlass::Status::kSuccess) {
      if (error != nullptr) *error = cutlass::cutlassGetStatusString(status);
      return 2;
    }
    return 0;
  }
};

// A config is a compiled instantiation, so the table is fixed at build time and
// the autotuner picks by index. Nothing but the launcher and its workspace
// question survives here: with K-major scales the tile shape no longer implies
// a constraint on the problem, so there is nothing about it left to record.
struct ConfigEntry {
  const char* name;
  size_t (*workspace_size)(int64_t, int64_t, int64_t);
  int (*run)(const Fp8BlockGemmCutlassParams&, void*, const char**);
};

using OutType = cutlass::bfloat16_t;

// Plain orientation: the activation is on M, one scale per row (granularity 1)
// against the weight's 128x128 grid.
//
// vLLM passes its TILE_N as the N granularity as well, which pins the tile to
// 128 -- a wider tile would claim a wider scale block and read the grid wrong,
// and a narrower one would claim a narrower block that does not exist. The two
// are separable, though: the granularity describes the checkpoint (always 128)
// while the tile only has to avoid straddling a block boundary, so any TileN
// that divides 128 is expressible. That is worth having, because a starved
// projection is bound by what one CTA reads -- TileN x k -- rather than by the
// output it produces, so narrowing N is the lever on shapes that do not fill
// the machine. TILE_K stays 128 for the reason vLLM's does: a deeper tile spans
// two K scale groups and the scale copy cannot vectorize across them.
template <int TileM, int TileN, class ClusterShape, class EpilogueSchedule,
          class MainloopSchedule>
using PlainGemm =
    Fp8BlockwiseGemm<OutType, 1, 128, 128,
                     Shape<Int<TileM>, Int<TileN>, _128>, ClusterShape,
                     EpilogueSchedule, MainloopSchedule, false>;

// Swapped: the weight goes on the MMA's M axis (granularity 128, its own scale
// block) and the activation on N (granularity 1, per row), so a decode-shaped
// batch tiles exactly instead of padding out to 64 or 128 rows of MMA.
template <int TileM, int TileN, class ClusterShape, class EpilogueSchedule,
          class MainloopSchedule>
using SwapGemm = Fp8BlockwiseGemm<OutType, 128, 1, 128,
                                  Shape<Int<TileM>, Int<TileN>, _128>,
                                  ClusterShape, EpilogueSchedule,
                                  MainloopSchedule, true>;

using Cluster1 = Shape<_1, _1, _1>;
using Cluster2 = Shape<_2, _1, _1>;
using Mainloop1Sm = cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100;
using Mainloop2Sm = cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100;
using EpiTma1Sm = cutlass::epilogue::TmaWarpSpecialized1Sm;
using EpiTma2Sm = cutlass::epilogue::TmaWarpSpecialized2Sm;
using EpiNoSmem1Sm = cutlass::epilogue::BlockwiseNoSmemWarpSpecialized1Sm;
using EpiNoSmem2Sm = cutlass::epilogue::BlockwiseNoSmemWarpSpecialized2Sm;

#define XLA_ENTRY(kind, name, tile_m, tile_n, cluster, epi, mainloop)    \
  ConfigEntry {                                                          \
    name,                                                                \
        &Runner<kind<tile_m, tile_n, cluster, epi,                       \
                     mainloop>>::WorkspaceSize,                          \
        &Runner<kind<tile_m, tile_n, cluster, epi, mainloop>>::Run       \
  }

// Timed on sm_103a at n = k = 5120, the projection shape of a 27B block-128
// checkpoint, over m from 1 to 4096. Two regions, and the table is the union
// of what wins in each:
//
//   m <= 64      swapab_256x32 at 10.2us, flat across the whole range -- the
//                batch is on N, so what a CTA reads is the weight and the time
//                is the weight read. A 16-wide tile would not go faster (there
//                is nothing left to save) and K-major scales cannot serve it.
//   m >= 128     plain, and only with the TMA epilogue: 56us against 201us for
//                the same 128x128 tile writing D without it at m = 2048. That
//                3.6x is why no non-TMA plain config is here -- it was never
//                the winner at any m, and the epilogue only declines when N is
//                not 8-aligned, which CanRun already forbids.
//
// Tiles 256 wide in N were measured and dropped: 60us at m = 64 against 10.2,
// and 172us at m = 2048 against 56.
const ConfigEntry kConfigs[] = {
    XLA_ENTRY(PlainGemm, "plain_64x128x128_c1x1_tma", 64, 128, Cluster1,
              EpiTma1Sm, Mainloop1Sm),
    XLA_ENTRY(PlainGemm, "plain_128x128x128_c1x1_tma", 128, 128, Cluster1,
              EpiTma1Sm, Mainloop1Sm),
    XLA_ENTRY(PlainGemm, "plain_256x128x128_c2x1_tma", 256, 128, Cluster2,
              EpiTma2Sm, Mainloop2Sm),
    // No narrow-N plain config: a CTA reads TileN x k of the weight, so a 32-
    // or 64-wide tile would be the lever on a projection that cannot fill the
    // machine, but the collective refuses it --
    //   sm100_mma_warpspecialized_blockwise_scaling.hpp:133
    //     "Scale Granularity N must divide Tile Shape"
    //   :156 "Scale Granularity must be smaller than or equal to the tile shape"
    // The granularity is the checkpoint's 128, so TileN can only be a multiple
    // of it. The swap-A/B configs below narrow the other axis instead, which is
    // the activation's (granularity 1) and therefore free.
    XLA_ENTRY(SwapGemm, "swapab_128x32x128_c1x1", 128, 32, Cluster1,
              EpiNoSmem1Sm, Mainloop1Sm),
    XLA_ENTRY(SwapGemm, "swapab_128x64x128_c1x1", 128, 64, Cluster1,
              EpiNoSmem1Sm, Mainloop1Sm),
    XLA_ENTRY(SwapGemm, "swapab_128x128x128_c1x1", 128, 128, Cluster1,
              EpiNoSmem1Sm, Mainloop1Sm),
    XLA_ENTRY(SwapGemm, "swapab_256x32x128_c2x1", 256, 32, Cluster2,
              EpiNoSmem2Sm, Mainloop2Sm),
    XLA_ENTRY(SwapGemm, "swapab_256x64x128_c2x1", 256, 64, Cluster2,
              EpiNoSmem2Sm, Mainloop2Sm),
    XLA_ENTRY(SwapGemm, "swapab_256x128x128_c2x1", 256, 128, Cluster2,
              EpiNoSmem2Sm, Mainloop2Sm),
};

#undef XLA_ENTRY

constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);
constexpr int kScaleBlock = 128;

}  // namespace

int Fp8BlockGemmCutlassNumConfigs() { return kNumConfigs; }

const char* Fp8BlockGemmCutlassConfigName(int config) {
  if (config < 0 || config >= kNumConfigs) return "<out of range>";
  return kConfigs[config].name;
}

bool Fp8BlockGemmCutlassCanRun(int config, int64_t m, int64_t n, int64_t k) {
  if (config < 0 || config >= kNumConfigs) return false;
  const ConfigEntry& c = kConfigs[config];
  // The scale grids have to be whole. Nothing constrains m: vLLM needs m % 4
  // for its plain configs because an MN-major activation scale is copied in
  // M-vectors, and K-major moves that vector onto an axis whose extent is the
  // whole scale row.
  if (k % kScaleBlock != 0 || n % kScaleBlock != 0) return false;
  // The thunk has no scratch to hand the kernel, so a config that wants a
  // workspace is declined here rather than at execution: this is the predicate
  // the autotuner filters on, and a config it never offers cannot be chosen.
  if (c.workspace_size(m, n, k) != 0) return false;
  return true;
}

size_t Fp8BlockGemmCutlassWorkspaceSize(int config, int64_t m, int64_t n,
                                        int64_t k) {
  if (config < 0 || config >= kNumConfigs) return 0;
  return kConfigs[config].workspace_size(m, n, k);
}

int Fp8BlockGemmCutlassRun(int config, const Fp8BlockGemmCutlassParams& params,
                           void* stream, const char** error) {
  if (config < 0 || config >= kNumConfigs) {
    if (error != nullptr) *error = "config index out of range";
    return -1;
  }
  return kConfigs[config].run(params, stream, error);
}

}  // namespace xla::gpu::kernel
