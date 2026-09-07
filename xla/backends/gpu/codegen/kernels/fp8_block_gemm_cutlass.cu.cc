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

// The GEMM configuration is vLLM's (csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
// scaled_mm_blockwise_sm100_fp8_dispatch.cuh at v0.28.1rc0, Apache 2.0). The collectives are
// theirs; the boundary -- device pointers, a stream, an autotuned tile index -- is ours.

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

// vLLM's cutlass_extensions/common.hpp: traps rather than misbehaves off its architecture.
template <typename Kernel, int ArchLo, int ArchHi>
struct EnableArch : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__)
    if constexpr (__CUDA_ARCH__ >= ArchLo && __CUDA_ARCH__ < ArchHi) {
      Kernel::operator()(std::forward<Args>(args)...);
    } else {
      asm("trap;");
    }
#endif
  }
};

// SM100 (tcgen05) and SM120 (warp mma + TMA) differ only in their collectives, so the family is
// a template parameter. `cc_minor` -1 means any minor; SM120 is 12.0 only, a plain 12.1 build
// compiles TMA out.
struct Sm100Family {
  static constexpr int cc_major = 10;
  static constexpr int cc_minor = -1;
  using ArchTag = cutlass::arch::Sm100;
  template <class Kernel>
  using Guard = EnableArch<Kernel, 1000, 1200>;
};

struct Sm120Family {
  static constexpr int cc_major = 12;
  static constexpr int cc_minor = 0;
  using ArchTag = cutlass::arch::Sm120;
  template <class Kernel>
  using Guard = EnableArch<Kernel, 1200, 1210>;
};

// The granularities are the checkpoint's, not the tile's; under swap A/B they trade places.
template <class Family, class OutType, int ScaleGranularityM,
          int ScaleGranularityN, int ScaleGranularityK, class MmaTileShape,
          class ClusterShape, class EpilogueScheduler, class MainloopScheduler,
          bool swap_ab_ = false>
struct Fp8BlockwiseGemm {
  static constexpr bool swap_ab = swap_ab_;
  static constexpr int cc_major = Family::cc_major;
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

  // Both scales K-major, the layout the frontend already produces. A swapped tile 16 wide cannot be
  // served: its scale copy vectorizes along a one-element axis.
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      cute::UMMA::Major::K, cute::UMMA::Major::K>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = typename Family::ArchTag;
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

  using KernelType =
      typename Family::template Guard<cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

// vLLM's cutlass_gemm_caller_blockwise with pointers; under a swap the strides describe [N, M].
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

// One compiled instantiation per entry; the autotuner picks by index.
struct ConfigEntry {
  const char* name;
  int cc_major;
  int cc_minor;
  size_t (*workspace_size)(int64_t, int64_t, int64_t);
  int (*run)(const Fp8BlockGemmCutlassParams&, void*, const char**);
};

using OutType = cutlass::bfloat16_t;

// Plain: activation on M. TileN may be any divisor of 128; TILE_K stays 128 so a tile never
// spans two K scale groups.
template <class Family, int TileM, int TileN, class ClusterShape,
          class EpilogueSchedule, class MainloopSchedule>
using PlainGemm =
    Fp8BlockwiseGemm<Family, OutType, 1, 128, 128,
                     Shape<Int<TileM>, Int<TileN>, _128>, ClusterShape,
                     EpilogueSchedule, MainloopSchedule, false>;

// Swapped: weight on M (so TileM >= 128), activation on N, so a decode batch tiles exactly.
template <class Family, int TileM, int TileN, class ClusterShape,
          class EpilogueSchedule, class MainloopSchedule>
using SwapGemm = Fp8BlockwiseGemm<Family, OutType, 128, 1, 128,
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

// SM120 has no blockwise epilogue tag; Cooperative needs 128 rows, so the 64-row tile is Pingpong.
using MainloopSm120Coop = cutlass::gemm::KernelScheduleSm120Blockwise;
using MainloopSm120Pingpong =
    cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
using EpiSm120 = cutlass::epilogue::collective::EpilogueScheduleAuto;

#define XLA_ENTRY(family, kind, name, tile_m, tile_n, cluster, epi, mainloop) \
  ConfigEntry {                                                               \
    name, family::cc_major, family::cc_minor,                                 \
        &Runner<kind<family, tile_m, tile_n, cluster, epi,                    \
                     mainloop>>::WorkspaceSize,                               \
        &Runner<kind<family, tile_m, tile_n, cluster, epi, mainloop>>::Run    \
  }

const ConfigEntry kConfigs[] = {
    XLA_ENTRY(Sm100Family, PlainGemm, "plain_64x128x128_c1x1_tma", 64, 128,
              Cluster1, EpiTma1Sm, Mainloop1Sm),
    XLA_ENTRY(Sm100Family, PlainGemm, "plain_128x128x128_c1x1_tma", 128, 128,
              Cluster1, EpiTma1Sm, Mainloop1Sm),
    XLA_ENTRY(Sm100Family, PlainGemm, "plain_256x128x128_c2x1_tma", 256, 128,
              Cluster2, EpiTma2Sm, Mainloop2Sm),
    // No narrow-N plain config: the collective needs TileN to be a multiple of the scale block.
    XLA_ENTRY(Sm100Family, SwapGemm, "swapab_128x32x128_c1x1", 128, 32,
              Cluster1, EpiNoSmem1Sm, Mainloop1Sm),
    XLA_ENTRY(Sm100Family, SwapGemm, "swapab_128x64x128_c1x1", 128, 64,
              Cluster1, EpiNoSmem1Sm, Mainloop1Sm),
    XLA_ENTRY(Sm100Family, SwapGemm, "swapab_128x128x128_c1x1", 128, 128,
              Cluster1, EpiNoSmem1Sm, Mainloop1Sm),
    XLA_ENTRY(Sm100Family, SwapGemm, "swapab_256x32x128_c2x1", 256, 32,
              Cluster2, EpiNoSmem2Sm, Mainloop2Sm),
    XLA_ENTRY(Sm100Family, SwapGemm, "swapab_256x64x128_c2x1", 256, 64,
              Cluster2, EpiNoSmem2Sm, Mainloop2Sm),
    XLA_ENTRY(Sm100Family, SwapGemm, "swapab_256x128x128_c2x1", 256, 128,
              Cluster2, EpiNoSmem2Sm, Mainloop2Sm),

    XLA_ENTRY(Sm120Family, PlainGemm, "sm120_plain_64x128x128_pp", 64, 128,
              Cluster1, EpiSm120, MainloopSm120Pingpong),
    XLA_ENTRY(Sm120Family, PlainGemm, "sm120_plain_128x128x128", 128, 128,
              Cluster1, EpiSm120, MainloopSm120Coop),
    XLA_ENTRY(Sm120Family, SwapGemm, "sm120_swapab_128x32x128", 128, 32,
              Cluster1, EpiSm120, MainloopSm120Coop),
    XLA_ENTRY(Sm120Family, SwapGemm, "sm120_swapab_128x64x128", 128, 64,
              Cluster1, EpiSm120, MainloopSm120Coop),
    XLA_ENTRY(Sm120Family, SwapGemm, "sm120_swapab_128x128x128", 128, 128,
              Cluster1, EpiSm120, MainloopSm120Coop),
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

bool Fp8BlockGemmCutlassCanRun(int config, int cc_major, int cc_minor,
                               int64_t m, int64_t n, int64_t k) {
  if (config < 0 || config >= kNumConfigs) return false;
  const ConfigEntry& c = kConfigs[config];
  if (c.cc_major != cc_major) return false;
  if (c.cc_minor >= 0 && c.cc_minor != cc_minor) return false;
  if (k % kScaleBlock != 0 || n % kScaleBlock != 0) return false;
  // The thunk has no scratch, so a config wanting a workspace is declined here.
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
