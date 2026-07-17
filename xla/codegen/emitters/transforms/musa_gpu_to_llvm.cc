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

#include "xla/codegen/emitters/transforms/musa_gpu_to_llvm.h"

#include <array>
#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace xla::emitters {
namespace {

namespace musa = ::xla::gpu::musa;

mlir::LogicalResult ValidateMusaTargetAttribute(mlir::ModuleOp module,
                                                llvm::StringRef attribute_name,
                                                llvm::StringRef expected_value,
                                                llvm::StringRef contract_name) {
  mlir::Attribute existing = module->getAttr(attribute_name);
  if (!existing) return mlir::success();

  auto existing_string = mlir::dyn_cast<mlir::StringAttr>(existing);
  if (!existing_string || existing_string.getValue() != expected_value) {
    module.emitError() << "conflicting MUSA " << contract_name
                       << ": expected \"" << expected_value << "\"";
    return mlir::failure();
  }
  return mlir::success();
}

const musa::MusaShimSpec* FindShimById(musa::MusaShimId id) {
  for (const musa::MusaShimSpec& spec : musa::MusaShimSpecs()) {
    if (spec.id == id) return &spec;
  }
  return nullptr;
}

mlir::Type ShimResultType(mlir::MLIRContext* context,
                          musa::MusaShimSignature signature) {
  switch (signature) {
    case musa::MusaShimSignature::kVoidVoid:
      return mlir::LLVM::LLVMVoidType::get(context);
    case musa::MusaShimSignature::kI32Void:
      return mlir::IntegerType::get(context, 32);
    case musa::MusaShimSignature::kI64Void:
      return mlir::IntegerType::get(context, 64);
  }
  llvm_unreachable("unknown MUSA shim signature");
}

mlir::LLVM::MemoryEffectsAttr ShimMemoryEffects(
    mlir::OpBuilder& builder, musa::MusaMemoryEffects effects) {
  using ModRefInfo = mlir::LLVM::ModRefInfo;
  constexpr ModRefInfo kNone = ModRefInfo::NoModRef;
  switch (effects) {
    case musa::MusaMemoryEffects::kNone:
      return builder.getAttr<mlir::LLVM::MemoryEffectsAttr>(
          /*other=*/kNone, /*argMem=*/kNone, /*inaccessibleMem=*/kNone,
          /*errnoMem=*/kNone, /*targetMem0=*/kNone, /*targetMem1=*/kNone);
    case musa::MusaMemoryEffects::kReadWrite:
      // An absent memory-effects attribute translates to unknown read/write
      // effects, which is the exact mapping-v1 contract for a full barrier.
      return {};
    case musa::MusaMemoryEffects::kInaccessibleRead:
      return builder.getAttr<mlir::LLVM::MemoryEffectsAttr>(
          /*other=*/kNone, /*argMem=*/kNone,
          /*inaccessibleMem=*/ModRefInfo::Ref, /*errnoMem=*/kNone,
          /*targetMem0=*/kNone, /*targetMem1=*/kNone);
    case musa::MusaMemoryEffects::kInaccessibleReadWrite:
      return builder.getAttr<mlir::LLVM::MemoryEffectsAttr>(
          /*other=*/kNone, /*argMem=*/kNone,
          /*inaccessibleMem=*/ModRefInfo::ModRef, /*errnoMem=*/kNone,
          /*targetMem0=*/kNone, /*targetMem1=*/kNone);
  }
  llvm_unreachable("unknown MUSA shim memory effects");
}

mlir::LLVM::LLVMFuncOp LookupOrCreateShim(
    mlir::Operation* source, const musa::MusaShimSpec& spec,
    mlir::ConversionPatternRewriter& rewriter) {
  mlir::ModuleOp module = source->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    source->emitError("MUSA shim call must be nested in a module");
    return {};
  }

  mlir::Type result_type =
      ShimResultType(rewriter.getContext(), spec.signature);
  auto function_type = mlir::LLVM::LLVMFunctionType::get(result_type, {});
  auto memory_effects = ShimMemoryEffects(rewriter, spec.memory_effects);

  if (mlir::Operation* symbol =
          mlir::SymbolTable::lookupSymbolIn(module, spec.xla_symbol)) {
    auto function = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(symbol);
    const bool valid =
        function && function.isExternal() &&
        function.getFunctionType() == function_type &&
        function.getLinkage() == mlir::LLVM::Linkage::External &&
        function.getCConv() == mlir::LLVM::cconv::CConv::C &&
        static_cast<bool>(function.getConvergentAttr()) == spec.convergent &&
        static_cast<bool>(function.getNoUnwindAttr()) ==
            ((spec.required_attributes & musa::kNoUnwind) != 0) &&
        static_cast<bool>(function.getWillReturnAttr()) ==
            ((spec.required_attributes & musa::kWillReturn) != 0) &&
        function.getMemoryEffectsAttr() == memory_effects;
    if (!valid) {
      source->emitError("conflicting declaration for versioned MUSA shim ")
          << spec.xla_symbol;
      return {};
    }
    return function;
  }

  mlir::OpBuilder builder(module.getBodyRegion());
  auto function = mlir::LLVM::LLVMFuncOp::create(
      builder, source->getLoc(), spec.xla_symbol, function_type);
  function.setCConv(mlir::LLVM::cconv::CConv::C);
  function.setConvergent(spec.convergent);
  function.setNoUnwind((spec.required_attributes & musa::kNoUnwind) != 0);
  function.setWillReturn((spec.required_attributes & musa::kWillReturn) != 0);
  if (memory_effects) function.setMemoryEffectsAttr(memory_effects);
  return function;
}

mlir::FailureOr<mlir::LLVM::CallOp> CreateShimCall(
    mlir::Operation* source, musa::MusaShimId id,
    mlir::ConversionPatternRewriter& rewriter) {
  const musa::MusaShimSpec* spec = FindShimById(id);
  if (spec == nullptr ||
      spec->minimum_mapping_version > musa::kMusaShimMappingVersion) {
    source->emitError("missing qualified MUSA shim mapping");
    return mlir::failure();
  }
  mlir::LLVM::LLVMFuncOp function = LookupOrCreateShim(source, *spec, rewriter);
  if (!function) return mlir::failure();

  // Do not copy declaration attributes to the call. Mapping version 1 has an
  // intentionally attribute-free call-site ABI.
  return mlir::LLVM::CallOp::create(rewriter, source->getLoc(), function,
                                    mlir::ValueRange{});
}

template <typename Op>
class MusaIndexOpLowering : public mlir::ConvertOpToLLVMPattern<Op> {
 public:
  MusaIndexOpLowering(const mlir::LLVMTypeConverter& converter,
                      std::array<musa::MusaShimId, 3> dimension_shims)
      : mlir::ConvertOpToLLVMPattern<Op>(converter),
        dimension_shims_(dimension_shims) {}

  mlir::LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    int dimension = 0;
    switch (op.getDimension()) {
      case mlir::gpu::Dimension::x:
        dimension = 0;
        break;
      case mlir::gpu::Dimension::y:
        dimension = 1;
        break;
      case mlir::gpu::Dimension::z:
        dimension = 2;
        break;
    }

    mlir::FailureOr<mlir::LLVM::CallOp> call =
        CreateShimCall(op, dimension_shims_[dimension], rewriter);
    if (mlir::failed(call)) return mlir::failure();

    auto converted_type = mlir::dyn_cast<mlir::IntegerType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op,
                                         "expected an integer index lowering");
    }

    mlir::Value result = call->getResult();
    if (converted_type.getWidth() > 32) {
      result = mlir::LLVM::ZExtOp::create(rewriter, op.getLoc(), converted_type,
                                          result);
    } else if (converted_type.getWidth() < 32) {
      result = mlir::LLVM::TruncOp::create(rewriter, op.getLoc(),
                                           converted_type, result);
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
  }

 private:
  std::array<musa::MusaShimId, 3> dimension_shims_;
};

class MusaBarrierOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
 public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::gpu::BarrierOp op, OpAdaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (op.getNamedBarrier() || op.getAddressSpacesAttr() ||
        op.getScope() != mlir::gpu::BarrierScope::Workgroup) {
      return rewriter.notifyMatchFailure(
          op,
          "mapping version 1 supports only an unnamed full workgroup "
          "barrier");
    }
    if (mlir::failed(CreateShimCall(op, musa::MusaShimId::kWorkgroupBarrier,
                                    rewriter))) {
      return mlir::failure();
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

}  // namespace

mlir::LogicalResult ConfigureMusaLLVMModule(mlir::ModuleOp module) {
  llvm::StringRef target_triple_attr =
      mlir::LLVM::LLVMDialect::getTargetTripleAttrName();
  llvm::StringRef data_layout_attr =
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName();
  if (mlir::failed(ValidateMusaTargetAttribute(
          module, target_triple_attr, stream_executor::musa::kMusaTargetTriple,
          "target triple")) ||
      mlir::failed(ValidateMusaTargetAttribute(
          module, data_layout_attr,
          stream_executor::musa::kMusaTargetDataLayout, "data layout"))) {
    return mlir::failure();
  }

  mlir::MLIRContext* context = module.getContext();
  module->setAttr(
      target_triple_attr,
      mlir::StringAttr::get(context, stream_executor::musa::kMusaTargetTriple));
  module->setAttr(data_layout_attr,
                  mlir::StringAttr::get(
                      context, stream_executor::musa::kMusaTargetDataLayout));
  return mlir::success();
}

void PopulateMusaGpuToLLVMConversionPatterns(
    const mlir::LLVMTypeConverter& converter, mlir::RewritePatternSet& patterns,
    mlir::ConversionTarget& target) {
  patterns.add<MusaIndexOpLowering<mlir::gpu::ThreadIdOp>>(
      converter,
      std::array{musa::MusaShimId::kThreadIdX, musa::MusaShimId::kThreadIdY,
                 musa::MusaShimId::kThreadIdZ});
  patterns.add<MusaIndexOpLowering<mlir::gpu::BlockIdOp>>(
      converter,
      std::array{musa::MusaShimId::kBlockIdX, musa::MusaShimId::kBlockIdY,
                 musa::MusaShimId::kBlockIdZ});
  patterns.add<MusaIndexOpLowering<mlir::gpu::BlockDimOp>>(
      converter,
      std::array{musa::MusaShimId::kBlockDimX, musa::MusaShimId::kBlockDimY,
                 musa::MusaShimId::kBlockDimZ});
  patterns.add<MusaIndexOpLowering<mlir::gpu::GridDimOp>>(
      converter,
      std::array{musa::MusaShimId::kGridDimX, musa::MusaShimId::kGridDimY,
                 musa::MusaShimId::kGridDimZ});
  patterns.add<MusaBarrierOpLowering>(converter);

  target.addIllegalDialect<mlir::gpu::GPUDialect>();
}

}  // namespace xla::emitters
