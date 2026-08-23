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
#include "xla/backends/gpu/codegen/flydsl/compiler.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

namespace xla::gpu::flydsl {
namespace {

constexpr llvm::StringLiteral kGenericFusionAttr = "xla.fly.generic_fusion";
constexpr llvm::StringLiteral kOriginalInvariantAttr =
    "xla.fly.original_invariant";
constexpr llvm::StringLiteral kOriginalAlignmentAttr =
    "xla.fly.original_alignment";

std::optional<mlir::BlockArgument> GetKernelArgument(mlir::Value value) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto function =
        mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(argument.getOwner()->getParentOp());
    if (function && argument.getOwner() == &function.getBody().front()) {
      return argument;
    }
    return std::nullopt;
  }
  mlir::Operation* defining_op = value.getDefiningOp();
  if (defining_op == nullptr || defining_op->getNumOperands() == 0) {
    return std::nullopt;
  }
  if (mlir::isa<mlir::LLVM::GEPOp, mlir::LLVM::BitcastOp,
                mlir::LLVM::AddrSpaceCastOp>(defining_op)) {
    return GetKernelArgument(defining_op->getOperand(0));
  }
  return std::nullopt;
}

mlir::fly::AddressSpace AddressSpaceFor(unsigned address_space) {
  switch (address_space) {
    case mlir::ROCDL::ROCDLDialect::kGlobalMemoryAddressSpace:
      return mlir::fly::AddressSpace::Global;
    case mlir::ROCDL::ROCDLDialect::kSharedMemoryAddressSpace:
      return mlir::fly::AddressSpace::Shared;
    case mlir::ROCDL::ROCDLDialect::kPrivateMemoryAddressSpace:
      return mlir::fly::AddressSpace::Register;
    default:
      return mlir::fly::AddressSpace::Generic;
  }
}

mlir::Type PointerElementType(mlir::Type value_type) {
  if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(value_type)) {
    return vector_type.getElementType();
  }
  return value_type;
}

// Builds a Fly buffer-descriptor pointer when XLA's kernel ABI proves that
// the complete allocation fits in AMD's 32-bit raw-buffer extent. The current
// LLVM address may be a nested GEP/bitcast of the kernel argument, so recover
// its byte offset from the address difference rather than duplicating XLA's
// indexing analysis here. The Fly pointer is deliberately byte-addressed: the
// raw-buffer operation carries the actual loaded/stored value type, while an
// i8 pointer preserves arbitrary LLVM bitcast and unaligned byte offsets.
std::optional<mlir::Value> CreateBufferDescriptorPointer(
    mlir::OpBuilder& builder, mlir::Location location, mlir::Value llvm_address,
    mlir::Type value_type) {
  std::optional<mlir::BlockArgument> argument =
      GetKernelArgument(llvm_address);
  if (!argument.has_value()) {
    return std::nullopt;
  }

  auto function = mlir::cast<mlir::LLVM::LLVMFuncOp>(
      argument->getOwner()->getParentOp());
  auto dereferenceable = function.getArgAttrOfType<mlir::IntegerAttr>(
      argument->getArgNumber(),
      mlir::LLVM::LLVMDialect::getDereferenceableAttrName());
  if (!dereferenceable) {
    return std::nullopt;
  }
  const int64_t allocation_bytes = dereferenceable.getInt();
  if (allocation_bytes <= 0 ||
      static_cast<uint64_t>(allocation_bytes) >
          std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }

  mlir::Type element_type = PointerElementType(value_type);
  if (!element_type.isIntOrFloat()) {
    return std::nullopt;
  }
  const unsigned element_bits = element_type.getIntOrFloatBitWidth();
  if (element_bits < 8 || element_bits % 8 != 0) {
    return std::nullopt;
  }
  uint64_t access_bits = element_bits;
  if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(value_type)) {
    access_bits *= vector_type.getNumElements();
  }
  // gfx942 raw-buffer loads/stores support at most four dwords. Wider XLA
  // vectors must remain ordinary global accesses unless explicitly split.
  if (access_bits > 128) {
    return std::nullopt;
  }
  mlir::MLIRContext* context = builder.getContext();
  auto buffer_address_space =
      mlir::fly_rocdl::BufferDescAddressAttr::get(context);
  auto buffer_pointer_type = mlir::fly::PointerType::get(
      builder.getI8Type(), buffer_address_space);

  mlir::Value stride = mlir::LLVM::ConstantOp::create(
      builder, location, builder.getI16Type(),
      builder.getI16IntegerAttr(0));
  mlir::Value extent = mlir::LLVM::ConstantOp::create(
      builder, location, builder.getI64Type(),
      builder.getI64IntegerAttr(allocation_bytes));
  // CDNA buffer descriptor: DATA_FORMAT=7 and NUM_FORMAT=4.
  mlir::Value flags = mlir::LLVM::ConstantOp::create(
      builder, location, builder.getI32Type(),
      builder.getI32IntegerAttr(0x27000));
  mlir::Value buffer_pointer = mlir::fly::MakePtrOp::create(
      builder, location, buffer_pointer_type,
      mlir::ValueRange{*argument, stride, extent, flags},
      /*dictAttrs=*/nullptr);

  mlir::Value current_address = mlir::LLVM::PtrToIntOp::create(
      builder, location, builder.getI64Type(), llvm_address);
  mlir::Value base_address = mlir::LLVM::PtrToIntOp::create(
      builder, location, builder.getI64Type(), *argument);
  mlir::Value byte_offset = mlir::LLVM::SubOp::create(
      builder, location, current_address, base_address);
  mlir::Value byte_offset_i32 = mlir::LLVM::TruncOp::create(
      builder, location, builder.getI32Type(), byte_offset);

  auto offset_attr =
      mlir::fly::IntTupleAttr::getLeafDynamic(context, /*width=*/32,
                                              /*divisibility=*/1);
  auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
  mlir::Value offset = mlir::fly::MakeIntTupleOp::create(
      builder, location, offset_type, mlir::ValueRange{byte_offset_i32});
  return mlir::fly::AddOffsetOp::create(
      builder, location, buffer_pointer_type, buffer_pointer, offset);
}

// XLA's generic emitter already has comprehensive HLO indexing and scalar
// lowering. Preserve that target-independent work, then make the target memory
// boundary explicitly Fly. The ptrtoint/inttoptr bridge is eliminated by LLVM
// after FlyToROCDL and preserves XLA's pointer-based kernel ABI exactly.
class WrapLlvmMemoryInFlyPass
    : public mlir::PassWrapper<WrapLlvmMemoryInFlyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapLlvmMemoryInFlyPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    module.getContext()->getOrLoadDialect<mlir::fly::FlyDialect>();
    module.getContext()->getOrLoadDialect<mlir::fly_rocdl::FlyROCDLDialect>();
    llvm::SmallVector<mlir::LLVM::LoadOp> loads;
    llvm::SmallVector<mlir::LLVM::StoreOp> stores;
    module.walk([&](mlir::LLVM::LoadOp op) {
      if (GetKernelArgument(op.getAddr()).has_value() &&
          op.getOrdering() == mlir::LLVM::AtomicOrdering::not_atomic &&
          !op->hasAttr("volatile_")) {
        loads.push_back(op);
      }
    });
    module.walk([&](mlir::LLVM::StoreOp op) {
      if (GetKernelArgument(op.getAddr()).has_value() &&
          op.getOrdering() == mlir::LLVM::AtomicOrdering::not_atomic &&
          !op->hasAttr("volatile_")) {
        stores.push_back(op);
      }
    });

    for (mlir::LLVM::LoadOp op : loads) {
      mlir::OpBuilder builder(op);
      std::optional<mlir::Value> buffer_pointer =
          CreateBufferDescriptorPointer(builder, op.getLoc(), op.getAddr(),
                                        op.getType());
      mlir::Value fly_ptr;
      if (buffer_pointer.has_value()) {
        fly_ptr = *buffer_pointer;
      } else {
        auto llvm_ptr_type =
            mlir::cast<mlir::LLVM::LLVMPointerType>(op.getAddr().getType());
        auto address_space = mlir::fly::AddressSpaceAttr::get(
            module.getContext(),
            AddressSpaceFor(llvm_ptr_type.getAddressSpace()));
        auto fly_ptr_type = mlir::fly::PointerType::get(
            PointerElementType(op.getType()), address_space);
        mlir::Value address = mlir::LLVM::PtrToIntOp::create(
            builder, op.getLoc(), builder.getI64Type(), op.getAddr());
        mlir::Operation* address_bridge = address.getDefiningOp();
        if (op.getInvariant()) {
          address_bridge->setAttr(kOriginalInvariantAttr,
                                  builder.getUnitAttr());
        }
        if (std::optional<uint64_t> alignment = op.getAlignment()) {
          address_bridge->setAttr(kOriginalAlignmentAttr,
                                  builder.getI64IntegerAttr(*alignment));
        }
        fly_ptr = mlir::fly::IntToPtrOp::create(
            builder, op.getLoc(), fly_ptr_type, address);
      }
      mlir::Value loaded = mlir::fly::PtrLoadOp::create(builder, op.getLoc(),
                                                        op.getType(), fly_ptr);
      op.replaceAllUsesWith(loaded);
      op.erase();
    }

    for (mlir::LLVM::StoreOp op : stores) {
      mlir::OpBuilder builder(op);
      std::optional<mlir::Value> buffer_pointer =
          CreateBufferDescriptorPointer(builder, op.getLoc(), op.getAddr(),
                                        op.getValue().getType());
      mlir::Value fly_ptr;
      if (buffer_pointer.has_value()) {
        fly_ptr = *buffer_pointer;
      } else {
        auto llvm_ptr_type =
            mlir::cast<mlir::LLVM::LLVMPointerType>(op.getAddr().getType());
        auto address_space = mlir::fly::AddressSpaceAttr::get(
            module.getContext(),
            AddressSpaceFor(llvm_ptr_type.getAddressSpace()));
        auto fly_ptr_type = mlir::fly::PointerType::get(
            PointerElementType(op.getValue().getType()), address_space);
        mlir::Value address = mlir::LLVM::PtrToIntOp::create(
            builder, op.getLoc(), builder.getI64Type(), op.getAddr());
        mlir::Operation* address_bridge = address.getDefiningOp();
        if (std::optional<uint64_t> alignment = op.getAlignment()) {
          address_bridge->setAttr(kOriginalAlignmentAttr,
                                  builder.getI64IntegerAttr(*alignment));
        }
        fly_ptr = mlir::fly::IntToPtrOp::create(
            builder, op.getLoc(), fly_ptr_type, address);
      }
      mlir::fly::PtrStoreOp::create(builder, op.getLoc(), op.getValue(),
                                    fly_ptr);
      op.erase();
    }
  }
};

// Fly pointer operations intentionally have a compact target-neutral surface
// and currently do not model LLVM load metadata. Carry XLA's proven invariant
// and alignment facts on the ptrtoint bridge and restore them immediately
// after FlyToROCDL, before canonicalization removes that bridge.
class RestoreGenericMemoryMetadataPass
    : public mlir::PassWrapper<RestoreGenericMemoryMetadataPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RestoreGenericMemoryMetadataPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    module.walk([&](mlir::LLVM::LoadOp load) {
      auto int_to_ptr = load.getAddr().getDefiningOp<mlir::LLVM::IntToPtrOp>();
      if (!int_to_ptr) {
        return;
      }
      auto ptr_to_int =
          int_to_ptr.getArg().getDefiningOp<mlir::LLVM::PtrToIntOp>();
      if (!ptr_to_int) {
        return;
      }
      if (ptr_to_int->hasAttr(kOriginalInvariantAttr)) {
        load.setInvariant(true);
      }
      if (auto alignment = ptr_to_int->getAttrOfType<mlir::IntegerAttr>(
              kOriginalAlignmentAttr)) {
        load.setAlignment(alignment.getInt());
      }
    });
    module.walk([&](mlir::LLVM::StoreOp store) {
      auto int_to_ptr = store.getAddr().getDefiningOp<mlir::LLVM::IntToPtrOp>();
      if (!int_to_ptr) {
        return;
      }
      auto ptr_to_int =
          int_to_ptr.getArg().getDefiningOp<mlir::LLVM::PtrToIntOp>();
      if (!ptr_to_int) {
        return;
      }
      if (auto alignment = ptr_to_int->getAttrOfType<mlir::IntegerAttr>(
              kOriginalAlignmentAttr)) {
        store.setAlignment(alignment.getInt());
      }
    });
    module.walk([&](mlir::LLVM::PtrToIntOp bridge) {
      bridge->removeAttr(kOriginalInvariantAttr);
      bridge->removeAttr(kOriginalAlignmentAttr);
    });
  }
};

}  // namespace

void RegisterDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::fly::FlyDialect, mlir::fly_rocdl::FlyROCDLDialect,
                  mlir::ub::UBDialect>();
}

bool HasOperations(mlir::ModuleOp module) {
  bool found = false;
  module->walk([&](mlir::Operation* op) {
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "fly" || dialect == "fly_rocdl") {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

void MarkGenericFusion(mlir::ModuleOp module) {
  module->setAttr(kGenericFusionAttr, mlir::UnitAttr::get(module.getContext()));
}

bool IsGenericFusion(mlir::ModuleOp module) {
  return module->hasAttr(kGenericFusionAttr);
}

void AddGenericMemoryPasses(mlir::OpPassManager& pm) {
  pm.addPass(std::make_unique<WrapLlvmMemoryInFlyPass>());
}

void AddLoweringPasses(mlir::OpPassManager& pm,
                       bool restore_generic_memory_metadata) {
  pm.addPass(mlir::fly::createFlyRewriteFuncSignaturePass());
  pm.addPass(mlir::fly::createFlyCanonicalizePass());
  pm.addPass(mlir::fly::createFlyLayoutLoweringPass());
  pm.addPass(mlir::fly::createFlyIntSwizzleSimplifyPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::fly::createFlyConvertAtomCallToSSAFormPass());
  pm.addPass(mlir::fly::createFlyPromoteRegMemToVectorSSAPass());
  pm.addPass(mlir::createFlyToROCDLConversionPass());
  if (restore_generic_memory_metadata) {
    pm.addPass(std::make_unique<RestoreGenericMemoryMetadataPass>());
  }
  pm.addPass(mlir::createCanonicalizerPass());
  // Generic Fly memory wrapping runs after XLA's ordinary LLVM lowering.
  // FlyToROCDL materializes a few arith operations for buffer-fat-pointer
  // offsets, so finish lowering those operations before LLVM IR translation.
  if (restore_generic_memory_metadata) {
    pm.addPass(mlir::createArithToLLVMConversionPass());
  }
}

}  // namespace xla::gpu::flydsl
