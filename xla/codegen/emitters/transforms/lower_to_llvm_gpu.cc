/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"   // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"   // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "google/protobuf/text_format.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/transforms/lower_to_llvm_common.h"
#include "xla/codegen/emitters/transforms/lowering_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_LOWERTOLLVMGPUPASS
#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h.inc"

namespace {

namespace se = ::stream_executor;

// log2(e), used to express exp(x) = exp2(x * log2(e)).
constexpr double kLog2E = 1.4426950408889634;

template <typename Op>
struct VulkanLaunchConfigConversion : public mlir::ConvertOpToLLVMPattern<Op> {
  VulkanLaunchConfigConversion(const mlir::LLVMTypeConverter& converter,
                               llvm::StringRef intrinsic)
      : mlir::ConvertOpToLLVMPattern<Op>(converter,
                                         /*benefit=*/mlir::PatternBenefit(2)),
        intrinsic_(intrinsic) {}

  mlir::LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    mlir::Location loc = op.getLoc();
    mlir::Value dimension =
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       static_cast<int64_t>(op.getDimension()));
    mlir::Type index_type = this->getTypeConverter()->getIndexType();
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, index_type, rewriter.getStringAttr(intrinsic_),
        mlir::ValueRange{dimension});
    return mlir::success();
  }

 private:
  std::string intrinsic_;
};

template <typename Op>
struct VulkanSubgroupQueryConversion : public mlir::ConvertOpToLLVMPattern<Op> {
  VulkanSubgroupQueryConversion(const mlir::LLVMTypeConverter& converter,
                                llvm::StringRef intrinsic)
      : mlir::ConvertOpToLLVMPattern<Op>(converter,
                                         /*benefit=*/mlir::PatternBenefit(2)),
        intrinsic_(intrinsic) {}

  mlir::LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    mlir::Location loc = op.getLoc();
    mlir::Value result =
        mlir::LLVM::CallIntrinsicOp::create(
            rewriter, loc, rewriter.getI32Type(),
            rewriter.getStringAttr(intrinsic_), mlir::ValueRange{})
            .getResult(0);
    mlir::Type index_type = this->getTypeConverter()->getIndexType();
    if (result.getType() != index_type) {
      result = mlir::LLVM::ZExtOp::create(rewriter, loc, index_type, result);
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
  }

 private:
  std::string intrinsic_;
};

struct VulkanBarrierConversion
    : public mlir::ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::gpu::BarrierOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<mlir::ArrayAttr> address_spaces = adaptor.getAddressSpaces();
    bool workgroup_memory_only = address_spaces.has_value();
    if (workgroup_memory_only) {
      for (mlir::Attribute attribute : *address_spaces) {
        auto address_space =
            mlir::cast<mlir::gpu::AddressSpaceAttr>(attribute).getValue();
        if (address_space != mlir::gpu::AddressSpace::Workgroup) {
          workgroup_memory_only = false;
          break;
        }
      }
    }
    llvm::StringRef intrinsic =
        workgroup_memory_only ? "llvm.spv.group.memory.barrier.with.group.sync"
                              : "llvm.spv.all.memory.barrier.with.group.sync";
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, rewriter.getStringAttr(intrinsic), mlir::ValueRange{});
    return mlir::success();
  }
};

struct VulkanShuffleConversion
    : public mlir::ConvertOpToLLVMPattern<mlir::gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::gpu::ShuffleOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Type i32 = rewriter.getI32Type();
    mlir::Value lane =
        mlir::LLVM::CallIntrinsicOp::create(
            rewriter, loc, i32,
            rewriter.getStringAttr("llvm.spv.subgroup.local.invocation.id"),
            mlir::ValueRange{})
            .getResult(0);
    mlir::Value source_lane;
    switch (op.getMode()) {
      case mlir::gpu::ShuffleMode::IDX:
        source_lane = adaptor.getOffset();
        break;
      case mlir::gpu::ShuffleMode::XOR:
        source_lane = mlir::LLVM::XOrOp::create(rewriter, loc, i32, lane,
                                                adaptor.getOffset());
        break;
      case mlir::gpu::ShuffleMode::UP:
        source_lane = mlir::LLVM::SubOp::create(rewriter, loc, i32, lane,
                                                adaptor.getOffset());
        break;
      case mlir::gpu::ShuffleMode::DOWN:
        source_lane = mlir::LLVM::AddOp::create(rewriter, loc, i32, lane,
                                                adaptor.getOffset());
        break;
    }

    mlir::Value shuffled =
        mlir::LLVM::CallIntrinsicOp::create(
            rewriter, loc, adaptor.getValue().getType(),
            rewriter.getStringAttr("llvm.spv.wave.readlane"),
            mlir::ValueRange{adaptor.getValue(), source_lane})
            .getResult(0);
    mlir::Value lane_is_valid = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::ult, lane,
        adaptor.getWidth());
    mlir::Value source_is_valid = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::ult, source_lane,
        adaptor.getWidth());
    mlir::Value valid = mlir::LLVM::AndOp::create(
        rewriter, loc, rewriter.getI1Type(), lane_is_valid, source_is_valid);
    rewriter.replaceOp(op, {shuffled, valid});
    return mlir::success();
  }
};

unsigned VulkanAddressSpace(mlir::gpu::AddressSpace address_space) {
  switch (address_space) {
    case mlir::gpu::AddressSpace::Global:
      return 11;  // SPIR-V StorageBuffer.
    case mlir::gpu::AddressSpace::Workgroup:
      return 3;
    case mlir::gpu::AddressSpace::Private:
      return 10;
    case mlir::gpu::AddressSpace::Constant:
      return 2;  // SPIR-V UniformConstant.
  }
  llvm_unreachable("unknown GPU address space");
}

void populateGpuToLLVMVulkanConversionPatterns(
    mlir::LLVMTypeConverter& converter, mlir::RewritePatternSet& patterns) {
  // Retain the generic GPU function/return conversions, but give every GPU
  // concept with an OpenCL-oriented implementation a higher-benefit Vulkan
  // lowering below.
  mlir::populateGpuToLLVMSPVConversionPatterns(converter, patterns);
  patterns.add<VulkanLaunchConfigConversion<mlir::gpu::ThreadIdOp>>(
      converter, "llvm.spv.thread.id.in.group");
  patterns.add<VulkanLaunchConfigConversion<mlir::gpu::BlockIdOp>>(
      converter, "llvm.spv.group.id");
  patterns.add<VulkanLaunchConfigConversion<mlir::gpu::BlockDimOp>>(
      converter, "llvm.spv.workgroup.size");
  patterns.add<VulkanLaunchConfigConversion<mlir::gpu::GridDimOp>>(
      converter, "llvm.spv.num.workgroups");
  patterns.add<VulkanLaunchConfigConversion<mlir::gpu::GlobalIdOp>>(
      converter, "llvm.spv.thread.id");
  patterns.add<VulkanSubgroupQueryConversion<mlir::gpu::LaneIdOp>>(
      converter, "llvm.spv.subgroup.local.invocation.id");
  patterns.add<VulkanSubgroupQueryConversion<mlir::gpu::SubgroupIdOp>>(
      converter, "llvm.spv.subgroup.id");
  patterns.add<VulkanSubgroupQueryConversion<mlir::gpu::NumSubgroupsOp>>(
      converter, "llvm.spv.num.subgroups");
  patterns.add<VulkanSubgroupQueryConversion<mlir::gpu::SubgroupSizeOp>>(
      converter, "llvm.spv.subgroup.size");
  patterns.add<VulkanBarrierConversion, VulkanShuffleConversion>(
      converter, /*benefit=*/mlir::PatternBenefit(2));
  mlir::populateGpuMemorySpaceAttributeConversions(converter,
                                                   VulkanAddressSpace);
}

// Lowers a scalar bf16 `math.exp2` to the native gfx1250 `v_exp_bf16`
// instruction via the `llvm.amdgcn.exp2` intrinsic. Without this, the default
// MathToROCDL lowering upcasts bf16 to f32 and calls `__ocml_exp2_f32`, never
// using the hardware bf16 transcendental unit. Vector ops are scalarized first
// by MathToROCDL's ScalarizeVectorOpLowering (lower benefit), so this pattern
// only needs to handle the scalar case.
struct Exp2BF16ToAMDGPU
    : public mlir::ConvertOpToLLVMPattern<mlir::math::Exp2Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::Exp2Op op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 exp2");
    }
    mlir::Value operand = adaptor.getOperands().front();
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, /*resultType=*/operand.getType(),
        rewriter.getStringAttr("llvm.amdgcn.exp2"), mlir::ValueRange{operand});
    return mlir::success();
  }
};

// Lowers a scalar bf16 `math.exp` to the native gfx1250 `v_exp_bf16`
// instruction by rewriting exp(x) = exp2(x * log2(e)) and emitting the
// `llvm.amdgcn.exp2` intrinsic. See Exp2BF16ToAMDGPU for the rationale.
struct ExpBF16ToAMDGPU
    : public mlir::ConvertOpToLLVMPattern<mlir::math::ExpOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 exp");
    }
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperands().front();
    mlir::Type bf16 = operand.getType();
    mlir::Value log2e = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, bf16, rewriter.getFloatAttr(bf16, kLog2E));
    mlir::Value scaled =
        rewriter.create<mlir::LLVM::FMulOp>(loc, operand, log2e);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, /*resultType=*/bf16, rewriter.getStringAttr("llvm.amdgcn.exp2"),
        mlir::ValueRange{scaled});
    return mlir::success();
  }
};

class LowerToLLVMGPUPass
    : public impl::LowerToLLVMGPUPassBase<LowerToLLVMGPUPass> {
 public:
  LowerToLLVMGPUPass() = default;

  explicit LowerToLLVMGPUPass(const LowerToLLVMGPUPassOptions& options)
      : LowerToLLVMGPUPassBase(options) {}

  explicit LowerToLLVMGPUPass(const se::DeviceDescription& device_description)
      : device_spec_(device_description) {}

  void runOnOperation() override {
    if (!gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      absl::StatusOr<se::DeviceDescription> device_description =
          se::DeviceDescription::FromProto(device_info);
      CHECK_OK(device_description.status());
      *device_spec_.mutable_type() = *device_description;
    }

    auto populate_patterns =
        [&](mlir::LLVMTypeConverter& converter,
            mlir::RewritePatternSet& patterns,
            mlir::ConversionTarget& target) -> mlir::LogicalResult {
      if (device_spec_.IsAmdGpu()) {
        std::string chipset =
            device_spec_.gpu().rocm_compute_capability().gfx_version();
        llvm::FailureOr<mlir::amdgpu::Chipset> maybeChipset =
            mlir::amdgpu::Chipset::parse(chipset);
        if (mlir::failed(maybeChipset)) {
          mlir::emitError(mlir::UnknownLoc::get(&getContext()),
                          "Invalid chipset name: " + chipset);
          return mlir::failure();
        }
        mlir::populateGpuToROCDLConversionPatterns(
            converter, patterns, mlir::gpu::amd::Runtime::Unknown,
            *maybeChipset);
        mlir::configureGpuToROCDLConversionLegality(target);
        mlir::populateAMDGPUToROCDLConversionPatterns(converter, patterns,
                                                      *maybeChipset);
        // On gfx1250 emit native bf16 exp via v_exp_bf16 instead of upcasting
        // to f32 and calling __ocml_exp(2)_f32. Higher benefit than the default
        // MathToROCDL patterns so it wins for scalar bf16 ops.
        if (device_spec_.gpu()
                .rocm_compute_capability()
                .has_bf16_transcendental_support()) {
          patterns.add<ExpBF16ToAMDGPU, Exp2BF16ToAMDGPU>(
              converter, /*benefit=*/mlir::PatternBenefit(2));
        }
        target.addIllegalDialect<mlir::amdgpu::AMDGPUDialect>();
      } else if (device_spec_.IsIntelGpu()) {
        // Add sub-group-size attribute to Intel functions.
        int32_t sub_group_size = device_spec_.gpu().threads_per_warp();
        if (auto module_op = mlir::dyn_cast<mlir::ModuleOp>(getOperation())) {
          module_op.walk([sub_group_size](mlir::func::FuncOp func) {
            if (!func.getBody().empty()) {
              mlir::OpBuilder b(func.getContext());
              auto sub_group_attr = b.getI32IntegerAttr(sub_group_size);
              func->setAttr("intel_reqd_sub_group_size", sub_group_attr);
            }
          });
        }
        populateGpuToLLVMSPVConversionPatterns(converter, patterns);
        spirv::populateMathToLLVMSPVConversionPatterns(spirv::getSPIRVMathOps(),
                                                       converter, patterns);
        populateGpuMemorySpaceAttributeConversions(converter);
      } else if (device_spec_.IsVulkan()) {
        populateGpuToLLVMVulkanConversionPatterns(converter, patterns);
      } else {
        mlir::populateGpuToNVVMConversionPatterns(converter, patterns);
        mlir::configureGpuToNVVMConversionLegality(target);
      }
      return mlir::success();
    };

    if (mlir::failed(LowerToLLVM(getOperation(), populate_patterns))) {
      signalPassFailure();
      return;
    }

    if (device_spec_.IsAmdGpu()) {
      EnsureAMDGPUAllocasUseAS5(getOperation());
    }
  }

 private:
  DeviceSpec device_spec_;
};

}  // namespace

std::unique_ptr<::mlir::Pass> createLowerToLLVMGPUPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<LowerToLLVMGPUPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
