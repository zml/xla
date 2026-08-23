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
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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

bool IsKernelArgumentDerived(mlir::Value value) {
  if (mlir::isa<mlir::BlockArgument>(value)) {
    return true;
  }
  mlir::Operation* defining_op = value.getDefiningOp();
  if (defining_op == nullptr || defining_op->getNumOperands() == 0) {
    return false;
  }
  if (mlir::isa<mlir::LLVM::GEPOp, mlir::LLVM::BitcastOp,
                mlir::LLVM::AddrSpaceCastOp>(defining_op)) {
    return IsKernelArgumentDerived(defining_op->getOperand(0));
  }
  return false;
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
    llvm::SmallVector<mlir::LLVM::LoadOp> loads;
    llvm::SmallVector<mlir::LLVM::StoreOp> stores;
    module.walk([&](mlir::LLVM::LoadOp op) {
      if (IsKernelArgumentDerived(op.getAddr()) &&
          op.getOrdering() == mlir::LLVM::AtomicOrdering::not_atomic &&
          !op->hasAttr("volatile_")) {
        loads.push_back(op);
      }
    });
    module.walk([&](mlir::LLVM::StoreOp op) {
      if (IsKernelArgumentDerived(op.getAddr()) &&
          op.getOrdering() == mlir::LLVM::AtomicOrdering::not_atomic &&
          !op->hasAttr("volatile_")) {
        stores.push_back(op);
      }
    });

    for (mlir::LLVM::LoadOp op : loads) {
      mlir::OpBuilder builder(op);
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
        address_bridge->setAttr(kOriginalInvariantAttr, builder.getUnitAttr());
      }
      if (std::optional<uint64_t> alignment = op.getAlignment()) {
        address_bridge->setAttr(kOriginalAlignmentAttr,
                                builder.getI64IntegerAttr(*alignment));
      }
      mlir::Value fly_ptr = mlir::fly::IntToPtrOp::create(
          builder, op.getLoc(), fly_ptr_type, address);
      mlir::Value loaded = mlir::fly::PtrLoadOp::create(builder, op.getLoc(),
                                                        op.getType(), fly_ptr);
      op.replaceAllUsesWith(loaded);
      op.erase();
    }

    for (mlir::LLVM::StoreOp op : stores) {
      mlir::OpBuilder builder(op);
      auto llvm_ptr_type =
          mlir::cast<mlir::LLVM::LLVMPointerType>(op.getAddr().getType());
      auto address_space = mlir::fly::AddressSpaceAttr::get(
          module.getContext(),
          AddressSpaceFor(llvm_ptr_type.getAddressSpace()));
      auto fly_ptr_type = mlir::fly::PointerType::get(
          PointerElementType(op.getValue().getType()), address_space);
      mlir::Value address = mlir::LLVM::PtrToIntOp::create(
          builder, op.getLoc(), builder.getI64Type(), op.getAddr());
      mlir::Value fly_ptr = mlir::fly::IntToPtrOp::create(
          builder, op.getLoc(), fly_ptr_type, address);
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
}

}  // namespace xla::gpu::flydsl
