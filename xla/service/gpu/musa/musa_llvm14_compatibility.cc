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

#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "xla/service/gpu/musa/musa_bridge_ir_validator.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"

namespace xla::gpu::musa {
namespace {

bool IsSafeModuleName(absl::string_view value) {
  if (value.empty() || value.size() > kMusaBridgeMaxModuleNameBytes) {
    return false;
  }
  const auto valid_first = [](char c) {
    return absl::ascii_isalpha(c) || absl::ascii_isdigit(c) || c == '_';
  };
  const auto valid_body = [](char c) {
    return absl::ascii_isalpha(c) || absl::ascii_isdigit(c) || c == '_' ||
           c == '-' || c == '.' || c == '+';
  };
  return valid_first(value.front()) &&
         std::all_of(value.begin() + 1, value.end(), valid_body);
}

std::string SanitizedModuleName(absl::string_view module_name) {
  return IsSafeModuleName(module_name) ? std::string(module_name)
                                       : "<invalid-module-name>";
}

absl::Status Rejected(absl::string_view module_name,
                      absl::string_view capability, absl::string_view detail) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "MUSA LLVM 14 compatibility rejected: revision=%s module=%s "
      "capability=%s: %s",
      kMusaLlvm14CompatibilityRevision, SanitizedModuleName(module_name),
      capability, detail));
}

std::string PrintModule(const llvm::Module& module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream, nullptr, /*ShouldPreserveUseListOrder=*/false,
               /*IsForDebug=*/false);
  stream.flush();
  return text;
}

absl::Status ValidateCurrentTarget(const llvm::Module& module,
                                   absl::string_view module_name) {
  if (module.getTargetTriple().str() != kMusaTargetTriple) {
    return Rejected(module_name, "target-triple",
                    "module does not use the frozen MUSA target triple");
  }
  if (module.getDataLayoutStr() != kMusaDataLayout) {
    return Rejected(module_name, "data-layout",
                    "module does not use the frozen mp_21 data layout");
  }
  return absl::OkStatus();
}

absl::Status StripReviewedNamedMetadata(llvm::Module& module,
                                        absl::string_view module_name) {
  llvm::SmallVector<llvm::NamedMDNode*, 4> erase;
  for (llvm::NamedMDNode& named : module.named_metadata()) {
    if (named.getName() == "llvm.ident") {
      erase.push_back(&named);
      continue;
    }
    if (named.getName() != "llvm.module.flags") {
      return Rejected(module_name, "named-metadata",
                      "named metadata is outside the active compatibility "
                      "revision");
    }

    // Debug-only flags do not affect executable semantics after debug
    // metadata is stripped. Every other module flag is ABI- or optimization-
    // relevant and therefore requires a reviewed compatibility revision.
    for (const llvm::MDNode* flag : named.operands()) {
      if (flag == nullptr || flag->getNumOperands() != 3) {
        return Rejected(module_name, "module-flags",
                        "malformed module flag metadata");
      }
      const auto* key =
          llvm::dyn_cast_or_null<llvm::MDString>(flag->getOperand(1));
      if (key == nullptr || (key->getString() != "Debug Info Version" &&
                             key->getString() != "Dwarf Version")) {
        return Rejected(module_name, "module-flags",
                        "non-debug module flags are not qualified");
      }
    }
    erase.push_back(&named);
  }
  for (llvm::NamedMDNode* named : erase) named->eraseFromParent();
  return absl::OkStatus();
}

bool IsLegacyBitonicCoordinateShim(const llvm::Function& function) {
  const MusaShimSpec* spec = FindMusaShim(function.getName());
  if (spec == nullptr) return false;
  switch (spec->id) {
    case MusaShimId::kBlockIdX:
    case MusaShimId::kBlockIdY:
    case MusaShimId::kBlockIdZ:
    case MusaShimId::kThreadIdX:
    case MusaShimId::kThreadIdY:
    case MusaShimId::kThreadIdZ:
      return true;
    default:
      return false;
  }
}

absl::Status StripLegacyBitonicOptimizationHints(
    llvm::Module& module, absl::string_view module_name) {
  if (llvm::Function* assume = module.getFunction("llvm.assume")) {
    if (!assume->isDeclaration() ||
        assume->getLinkage() != llvm::GlobalValue::ExternalLinkage ||
        assume->getCallingConv() != llvm::CallingConv::C ||
        assume->getIntrinsicID() != llvm::Intrinsic::assume ||
        !assume->getReturnType()->isVoidTy() || assume->arg_size() != 1 ||
        !assume->getArg(0)->getType()->isIntegerTy(1)) {
      return Rejected(module_name, "legacy-bitonic-hints",
                      "llvm.assume is outside the generated bitonic profile");
    }
    const llvm::AttributeList canonical = llvm::Intrinsic::getAttributes(
        module.getContext(), llvm::Intrinsic::assume,
        assume->getFunctionType());
    if (assume->getAttributes() != canonical) {
      return Rejected(module_name, "legacy-bitonic-hints",
                      "llvm.assume attributes are outside the generated "
                      "bitonic profile");
    }

    std::vector<llvm::CallInst*> calls;
    calls.reserve(assume->getNumUses());
    for (llvm::User* user : assume->users()) {
      auto* call = llvm::dyn_cast<llvm::CallInst>(user);
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4> metadata;
      if (call != nullptr) call->getAllMetadata(metadata);
      if (call == nullptr || call->getCalledFunction() != assume ||
          call->getCallingConv() != llvm::CallingConv::C ||
          call->isTailCall() || call->arg_size() != 1 ||
          !call->getArgOperand(0)->getType()->isIntegerTy(1) ||
          !call->getType()->isVoidTy() || !call->getAttributes().isEmpty() ||
          call->hasOperandBundles() || !metadata.empty() ||
          call->getDebugLoc()) {
        return Rejected(
            module_name, "legacy-bitonic-hints",
            "llvm.assume use is outside the generated bitonic profile");
      }
      calls.push_back(call);
    }
    // The generated assume communicates only an optimization promise. Erasing
    // this exact plain form widens semantics and avoids exposing an unqualified
    // current-LLVM intrinsic to vendor LLVM 14.
    for (llvm::CallInst* call : calls) call->eraseFromParent();
    if (!assume->use_empty()) {
      return Rejected(module_name, "legacy-bitonic-hints",
                      "llvm.assume retains an unsupported use");
    }
    assume->eraseFromParent();
  }

  for (llvm::Function& function : module.functions()) {
    for (llvm::BasicBlock& block : function) {
      for (llvm::Instruction& instruction : block) {
        llvm::MDNode* range =
            instruction.getMetadata(llvm::LLVMContext::MD_range);
        if (range == nullptr) continue;
        auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        llvm::Function* callee =
            call == nullptr ? nullptr : call->getCalledFunction();
        const auto* lower = range->getNumOperands() == 2
                                ? llvm::mdconst::dyn_extract<llvm::ConstantInt>(
                                      range->getOperand(0).get())
                                : nullptr;
        const auto* upper = range->getNumOperands() == 2
                                ? llvm::mdconst::dyn_extract<llvm::ConstantInt>(
                                      range->getOperand(1).get())
                                : nullptr;
        if (callee == nullptr || !IsLegacyBitonicCoordinateShim(*callee) ||
            !instruction.getType()->isIntegerTy(32) || lower == nullptr ||
            upper == nullptr || lower->getBitWidth() != 32 ||
            upper->getBitWidth() != 32 || !lower->isZero() || upper->isZero()) {
          return Rejected(
              module_name, "legacy-bitonic-hints",
              "range metadata is outside the generated coordinate profile");
        }
        instruction.setMetadata(llvm::LLVMContext::MD_range, nullptr);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status NormalizeKernelArgumentAttributes(llvm::Function& function,
                                               absl::string_view module_name) {
  if (function.getAttributes().hasRetAttrs()) {
    return Rejected(module_name, "kernel-attributes",
                    "kernel return attributes are not qualified");
  }
  for (llvm::Argument& argument : function.args()) {
    for (llvm::Attribute attribute : argument.getAttributes()) {
      if (attribute.isStringAttribute()) {
        return Rejected(module_name, "kernel-attributes",
                        "kernel string argument attributes are not qualified");
      }
      switch (attribute.getKindAsEnum()) {
        case llvm::Attribute::NoAlias:
          if (!argument.getType()->isPointerTy()) {
            return Rejected(module_name, "kernel-attributes",
                            "noalias requires a pointer kernel argument");
          }
          break;
        case llvm::Attribute::Alignment:
          if (!argument.getType()->isPointerTy() ||
              !attribute.getAlignment().has_value()) {
            return Rejected(module_name, "kernel-attributes",
                            "align requires a valid pointer kernel argument");
          }
          break;
        case llvm::Attribute::Dereferenceable:
          if (!argument.getType()->isPointerTy() ||
              attribute.getDereferenceableBytes() == 0) {
            return Rejected(
                module_name, "kernel-attributes",
                "dereferenceable has an invalid kernel pointer bound");
          }
          break;
        default:
          return Rejected(module_name, "kernel-attributes",
                          "kernel argument attribute is not qualified");
      }
    }
    // These attributes are current-side alias/bounds promises already consumed
    // by optimization. A dereferenceable byte count describes the logical
    // device buffer and can legitimately exceed the serialized-IR transport
    // limit. It is not part of the vendor kernel ABI and is discarded here.
    argument.removeAttr(llvm::Attribute::NoAlias);
    argument.removeAttr(llvm::Attribute::Alignment);
    argument.removeAttr(llvm::Attribute::Dereferenceable);
  }
  return absl::OkStatus();
}

absl::Status NormalizeLocalHelperArgumentAttributes(
    llvm::Function& function, absl::string_view module_name) {
  if (function.getAttributes().hasRetAttrs() ||
      function.getAttributes().getFnAttrs().getNumAttributes() != 0) {
    return Rejected(module_name, "helper-attributes",
                    "local helper return and function attributes are not "
                    "qualified");
  }
  for (llvm::Argument& argument : function.args()) {
    for (llvm::Attribute attribute : argument.getAttributes()) {
      if (attribute.isStringAttribute() || !argument.getType()->isPointerTy() ||
          (attribute.getKindAsEnum() != llvm::Attribute::NoAlias &&
           attribute.getKindAsEnum() != llvm::Attribute::Dereferenceable)) {
        return Rejected(module_name, "helper-attributes",
                        "only noalias and bounded dereferenceable on local "
                        "helper pointer arguments are qualified");
      }
      if (attribute.getKindAsEnum() == llvm::Attribute::Dereferenceable &&
          attribute.getDereferenceableBytes() == 0) {
        return Rejected(module_name, "helper-attributes",
                        "local helper dereferenceable bytes are invalid");
      }
    }
    // These are optimization-only promises, not part of the textual vendor
    // LLVM helper ABI, and are safe to discard at this boundary.
    argument.removeAttr(llvm::Attribute::NoAlias);
    argument.removeAttr(llvm::Attribute::Dereferenceable);
  }
  if (!function.getAttributes().isEmpty()) {
    return Rejected(module_name, "helper-attributes",
                    "local helper attributes remain after normalization");
  }
  return absl::OkStatus();
}

absl::Status NormalizePureIntrinsicAttributes(llvm::Function& function,
                                              absl::string_view module_name) {
  const llvm::AttributeList current = llvm::Intrinsic::getAttributes(
      function.getContext(), function.getIntrinsicID(),
      function.getFunctionType());
  if (function.getAttributes() != current) {
    return Rejected(module_name, "intrinsic-attributes",
                    "pure intrinsic must have canonical current-LLVM attrs");
  }

  // This is the exact canonical LLVM 14 declaration observed and checked in
  // the isolated bridge. Do not infer it by loading vendor headers here.
  function.setAttributes(llvm::AttributeList());
  function.addFnAttr(llvm::Attribute::NoFree);
  function.addFnAttr(llvm::Attribute::NoSync);
  function.addFnAttr(llvm::Attribute::NoUnwind);
  function.addFnAttr(llvm::Attribute::Speculatable);
  function.addFnAttr(llvm::Attribute::WillReturn);
  function.setMemoryEffects(llvm::MemoryEffects::none());
  return absl::OkStatus();
}

bool HasOnlyNoSignedZeros(const llvm::FastMathFlags& flags) {
  return flags.noSignedZeros() && !flags.allowReassoc() && !flags.noNaNs() &&
         !flags.noInfs() && !flags.allowReciprocal() &&
         !flags.allowContract() && !flags.approxFunc();
}

bool IsFloatingMinMaxIntrinsic(llvm::Intrinsic::ID id) {
  return id == llvm::Intrinsic::maximum || id == llvm::Intrinsic::minimum ||
         id == llvm::Intrinsic::maxnum || id == llvm::Intrinsic::minnum;
}

bool IsReviewedPureIntrinsic(llvm::Intrinsic::ID id) {
  switch (id) {
    // The pinned MUSA LLVM 14 registry gives each reviewed intrinsic the same
    // pure, speculatable, will-return contract as current LLVM.
    case llvm::Intrinsic::cos:
    case llvm::Intrinsic::exp:
    case llvm::Intrinsic::maximum:
    case llvm::Intrinsic::maxnum:
    case llvm::Intrinsic::minimum:
    case llvm::Intrinsic::minnum:
    case llvm::Intrinsic::fabs:
    case llvm::Intrinsic::smax:
    case llvm::Intrinsic::smin:
    case llvm::Intrinsic::sin:
    case llvm::Intrinsic::sqrt:
    case llvm::Intrinsic::umax:
    case llvm::Intrinsic::umin:
      return true;
    default:
      return false;
  }
}

absl::Status ValidateFloatingMinMaxCall(const llvm::Function& intrinsic,
                                        const llvm::CallInst& call,
                                        absl::string_view module_name) {
  if (call.getCalledFunction() != &intrinsic ||
      call.getCallingConv() != llvm::CallingConv::C || call.isTailCall() ||
      call.arg_size() != 2 || !call.getAttributes().isEmpty() ||
      call.hasOperandBundles()) {
    return Rejected(module_name, "floating-minmax-call",
                    "floating min/max must be a plain direct two-operand call");
  }

  llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4> metadata;
  call.getAllMetadata(metadata);
  if (!metadata.empty() || call.getDebugLoc()) {
    return Rejected(module_name, "floating-minmax-call",
                    "floating min/max call metadata is not qualified");
  }

  const llvm::FastMathFlags flags = call.getFastMathFlags();
  if (flags.any() && !HasOnlyNoSignedZeros(flags)) {
    return Rejected(module_name, "fast-math-flags",
                    "floating min/max carries flags outside the nsz-only "
                    "contract");
  }

  llvm::Type* type = call.getType();
  llvm::Type* scalar_type = type->getScalarType();
  if ((!scalar_type->isHalfTy() && !scalar_type->isFloatTy() &&
       !scalar_type->isDoubleTy()) ||
      (type->isVectorTy() &&
       llvm::cast<llvm::VectorType>(type)->getElementCount().isScalable()) ||
      call.getArgOperand(0)->getType() != type ||
      call.getArgOperand(1)->getType() != type) {
    return Rejected(module_name, "floating-minmax-type",
                    "floating min/max requires a qualified scalar or fixed "
                    "vector f16, f32, or f64 overload");
  }
  return absl::OkStatus();
}

llvm::Value* LowerFloatingMinMaxCall(llvm::CallInst& call) {
  const llvm::Intrinsic::ID id = call.getCalledFunction()->getIntrinsicID();
  const bool is_max =
      id == llvm::Intrinsic::maximum || id == llvm::Intrinsic::maxnum;
  const bool propagates_nan =
      id == llvm::Intrinsic::maximum || id == llvm::Intrinsic::minimum;
  llvm::Value* lhs = call.getArgOperand(0);
  llvm::Value* rhs = call.getArgOperand(1);
  llvm::Type* type = call.getType();
  llvm::Type* scalar_type = type->getScalarType();
  llvm::Type* integer_scalar = llvm::Type::getIntNTy(
      call.getContext(), scalar_type->getPrimitiveSizeInBits());
  llvm::Type* integer_type = integer_scalar;
  if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    integer_type = llvm::FixedVectorType::get(integer_scalar,
                                              vector_type->getNumElements());
  }

  const std::string prefix =
      call.hasName() ? call.getName().str() : "musa_floating_minmax";
  llvm::IRBuilder<> builder(&call);
  llvm::Value* ordered_compare =
      is_max ? builder.CreateFCmpOGT(lhs, rhs, prefix + ".ordered_compare")
             : builder.CreateFCmpOLT(lhs, rhs, prefix + ".ordered_compare");
  llvm::Value* ordered = builder.CreateSelect(ordered_compare, lhs, rhs,
                                              prefix + ".ordered_value");

  // Equal non-NaN values have identical encodings except for signed zero.
  // IEEE sign ordering is therefore exactly integer AND for maximum and
  // integer OR for minimum, including +0/-0 in either operand order.
  llvm::Value* equal = builder.CreateFCmpOEQ(lhs, rhs, prefix + ".equal");
  llvm::Value* lhs_bits =
      builder.CreateBitCast(lhs, integer_type, prefix + ".lhs_bits");
  llvm::Value* rhs_bits =
      builder.CreateBitCast(rhs, integer_type, prefix + ".rhs_bits");
  llvm::Value* equal_bits =
      is_max ? builder.CreateAnd(lhs_bits, rhs_bits, prefix + ".equal_bits")
             : builder.CreateOr(lhs_bits, rhs_bits, prefix + ".equal_bits");
  llvm::Value* equal_value =
      builder.CreateBitCast(equal_bits, type, prefix + ".equal_value");
  ordered = builder.CreateSelect(equal, equal_value, ordered,
                                 prefix + ".zero_ordered_value");

  if (propagates_nan) {
    llvm::Constant* nan_bits = llvm::ConstantInt::get(
        integer_scalar,
        llvm::APFloat::getNaN(scalar_type->getFltSemantics()).bitcastToAPInt());
    if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
      nan_bits = llvm::ConstantVector::getSplat(vector_type->getElementCount(),
                                                nan_bits);
    }
    llvm::Value* nan =
        new llvm::BitCastInst(nan_bits, type, prefix + ".canonical_nan", &call);
    llvm::Value* unordered =
        builder.CreateFCmpUNO(lhs, rhs, prefix + ".unordered");
    return builder.CreateSelect(unordered, nan, ordered, prefix + ".result");
  }

  llvm::Value* lhs_nan = builder.CreateFCmpUNO(lhs, lhs, prefix + ".lhs_nan");
  llvm::Value* rhs_nan = builder.CreateFCmpUNO(rhs, rhs, prefix + ".rhs_nan");
  llvm::Value* rhs_checked = builder.CreateSelect(
      rhs_nan, lhs, ordered, prefix + ".rhs_checked_value");
  return builder.CreateSelect(lhs_nan, rhs, rhs_checked, prefix + ".result");
}

absl::Status LowerFloatingMinMaxIntrinsics(llvm::Module& module,
                                           absl::string_view module_name) {
  std::vector<llvm::Function*> intrinsics;
  for (llvm::Function& function : module.functions()) {
    if (function.isIntrinsic() &&
        IsFloatingMinMaxIntrinsic(function.getIntrinsicID())) {
      intrinsics.push_back(&function);
    }
  }

  for (llvm::Function* intrinsic : intrinsics) {
    if (!intrinsic->isDeclaration() ||
        intrinsic->getLinkage() != llvm::GlobalValue::ExternalLinkage ||
        intrinsic->getCallingConv() != llvm::CallingConv::C) {
      return Rejected(module_name, "floating-minmax-declaration",
                      "floating min/max must be an external C declaration");
    }
    std::vector<llvm::CallInst*> calls;
    calls.reserve(intrinsic->getNumUses());
    for (llvm::User* user : intrinsic->users()) {
      auto* call = llvm::dyn_cast<llvm::CallInst>(user);
      if (call == nullptr) {
        return Rejected(module_name, "floating-minmax-call",
                        "floating min/max has a non-call use");
      }
      if (absl::Status status =
              ValidateFloatingMinMaxCall(*intrinsic, *call, module_name);
          !status.ok()) {
        return status;
      }
      calls.push_back(call);
    }
    for (llvm::CallInst* call : calls) {
      call->replaceAllUsesWith(LowerFloatingMinMaxCall(*call));
      call->eraseFromParent();
    }
    if (!intrinsic->use_empty()) {
      return Rejected(module_name, "floating-minmax-call",
                      "floating min/max retains an unsupported use");
    }
    intrinsic->eraseFromParent();
  }
  return absl::OkStatus();
}

absl::Status NormalizeFunctions(llvm::Module& module,
                                absl::string_view module_name,
                                MusaBridgeIrMetadata& metadata) {
  for (llvm::Function& function : module.functions()) {
    if (function.isIntrinsic()) {
      if (!IsReviewedPureIntrinsic(function.getIntrinsicID())) {
        return Rejected(
            module_name, "intrinsic-compatibility",
            absl::StrCat("generic intrinsic ", function.getName().str(),
                         " lacks an LLVM 14 profile"));
      }
      if (absl::Status status =
              NormalizePureIntrinsicAttributes(function, module_name);
          !status.ok()) {
        return status;
      }
      continue;
    }

    const bool marked = function.hasFnAttribute(kMusaLlvmKernelMarker);
    if (!marked) {
      if (!function.isDeclaration() && !function.hasLocalLinkage()) {
        return Rejected(module_name, "kernel-marker",
                        "externally visible definition lacks the v1 marker");
      }
      if (!function.isDeclaration()) {
        if (absl::Status status =
                NormalizeLocalHelperArgumentAttributes(function, module_name);
            !status.ok()) {
          return status;
        }
      }
      continue;
    }

    llvm::Attribute marker = function.getFnAttribute(kMusaLlvmKernelMarker);
    if (!marker.isStringAttribute() || !marker.getValueAsString().empty() ||
        function.isDeclaration() ||
        function.getLinkage() != llvm::GlobalValue::ExternalLinkage ||
        function.getCallingConv() != llvm::CallingConv::C ||
        !function.getReturnType()->isVoidTy() || function.isVarArg()) {
      return Rejected(module_name, "kernel-marker",
                      "marked kernel violates the ordinary-C entry contract");
    }

    function.removeFnAttr(kMusaLlvmKernelMarker);
    if (function.getAttributes().getFnAttrs().getNumAttributes() != 0) {
      return Rejected(module_name, "kernel-attributes",
                      "kernel function attributes are not qualified");
    }
    if (absl::Status status =
            NormalizeKernelArgumentAttributes(function, module_name);
        !status.ok()) {
      return status;
    }
    if (!function.getAttributes().isEmpty()) {
      return Rejected(module_name, "kernel-attributes",
                      "kernel attributes remain after normalization");
    }
    metadata.kernel_entry_names.push_back(function.getName().str());
  }

  std::sort(metadata.kernel_entry_names.begin(),
            metadata.kernel_entry_names.end());
  if (std::adjacent_find(metadata.kernel_entry_names.begin(),
                         metadata.kernel_entry_names.end()) !=
      metadata.kernel_entry_names.end()) {
    return Rejected(module_name, "kernel-marker",
                    "kernel marker set must be unique");
  }
  return absl::OkStatus();
}

void StripInvariantLoadMetadata(llvm::Module& module) {
  for (llvm::Function& function : module.functions()) {
    for (llvm::BasicBlock& block : function) {
      for (llvm::Instruction& instruction : block) {
        if (llvm::isa<llvm::LoadInst>(instruction)) {
          instruction.setMetadata(llvm::LLVMContext::MD_invariant_load,
                                  nullptr);
        }
      }
    }
  }
}

bool TypeContainsBfloat(const llvm::Type* type) {
  if (type->isBFloatTy()) return true;
  if (const auto* vector = llvm::dyn_cast<llvm::VectorType>(type)) {
    return TypeContainsBfloat(vector->getElementType());
  }
  if (const auto* array = llvm::dyn_cast<llvm::ArrayType>(type)) {
    return TypeContainsBfloat(array->getElementType());
  }
  if (const auto* structure = llvm::dyn_cast<llvm::StructType>(type)) {
    return std::any_of(
        structure->element_begin(), structure->element_end(),
        [](const llvm::Type* element) { return TypeContainsBfloat(element); });
  }
  if (const auto* function = llvm::dyn_cast<llvm::FunctionType>(type)) {
    if (TypeContainsBfloat(function->getReturnType())) return true;
    return std::any_of(function->param_begin(), function->param_end(),
                       [](const llvm::Type* parameter) {
                         return TypeContainsBfloat(parameter);
                       });
  }
  return false;
}

bool IsQualifiedBfloatValueType(const llvm::Type* type) {
  if (type->isBFloatTy()) return true;
  const auto* vector = llvm::dyn_cast<llvm::FixedVectorType>(type);
  return vector != nullptr && vector->getElementType()->isBFloatTy() &&
         (vector->getNumElements() == 2 || vector->getNumElements() == 4);
}

bool IsQualifiedBfloatStorageType(const llvm::Type* type) {
  const auto* array = llvm::dyn_cast<llvm::ArrayType>(type);
  return array != nullptr && array->getNumElements() != 0 &&
         array->getElementType()->isBFloatTy();
}

absl::Status ValidateBfloatIntegerLoweringProfile(
    const llvm::Module& module, absl::string_view module_name) {
  for (const llvm::Function& function : module.functions()) {
    if (TypeContainsBfloat(function.getFunctionType())) {
      if (function.isDeclaration() || !function.hasLocalLinkage() ||
          llvm::isa<llvm::VectorType>(function.getReturnType())) {
        return Rejected(module_name, "bfloat-integer-profile",
                        "bfloat function signatures require a local helper");
      }
      unsigned bfloat_parameters = 0;
      for (const llvm::Argument& argument : function.args()) {
        if (!TypeContainsBfloat(argument.getType())) continue;
        if (!argument.getType()->isBFloatTy() || ++bfloat_parameters != 1) {
          return Rejected(module_name, "bfloat-integer-profile",
                          "local helper supports one scalar bfloat parameter");
        }
        for (const llvm::User* user : argument.users()) {
          const auto* extension = llvm::dyn_cast<llvm::FPExtInst>(user);
          if (extension == nullptr || !extension->getType()->isFloatTy()) {
            return Rejected(
                module_name, "bfloat-integer-profile",
                "local helper bfloat parameter must extend directly to f32");
          }
        }
      }
      if (function.getReturnType()->isBFloatTy()) {
        unsigned return_sites = 0;
        for (const llvm::BasicBlock& block : function) {
          const auto* ret =
              llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator());
          if (ret == nullptr) continue;
          ++return_sites;
          const auto* truncation =
              llvm::dyn_cast_or_null<llvm::FPTruncInst>(ret->getReturnValue());
          if (truncation == nullptr ||
              !truncation->getOperand(0)->getType()->isFloatTy()) {
            return Rejected(
                module_name, "bfloat-integer-profile",
                "local helper bfloat return must truncate directly from f32");
          }
        }
        if (return_sites != 1) {
          return Rejected(
              module_name, "bfloat-integer-profile",
              "local helper bfloat return requires one return site");
        }
      }
    }

    for (const llvm::BasicBlock& block : function) {
      for (const llvm::Instruction& instruction : block) {
        if (const auto* gep =
                llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction);
            gep != nullptr && TypeContainsBfloat(gep->getSourceElementType()) &&
            !IsQualifiedBfloatStorageType(gep->getSourceElementType())) {
          return Rejected(module_name, "bfloat-integer-profile",
                          "bfloat GEP requires a flat scalar storage array");
        }

        const bool has_bfloat =
            TypeContainsBfloat(instruction.getType()) ||
            std::any_of(instruction.value_op_begin(),
                        instruction.value_op_end(),
                        [](const llvm::Value* value) {
                          return TypeContainsBfloat(value->getType());
                        });
        if (!has_bfloat) continue;

        if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
          if (!load->isSimple() || load->getAlign().value() != 2 ||
              !IsQualifiedBfloatValueType(load->getType())) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "bfloat load is outside the qualified profile");
          }
          continue;
        }
        if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction)) {
          if (!store->isSimple() || store->getAlign().value() != 2 ||
              !IsQualifiedBfloatValueType(
                  store->getValueOperand()->getType())) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "bfloat store is outside the qualified profile");
          }
          continue;
        }
        if (const auto* extension =
                llvm::dyn_cast<llvm::FPExtInst>(&instruction)) {
          if (!extension->getOperand(0)->getType()->isBFloatTy() ||
              !extension->getType()->isFloatTy()) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "only scalar bfloat-to-f32 extension is allowed");
          }
          continue;
        }
        if (const auto* truncation =
                llvm::dyn_cast<llvm::FPTruncInst>(&instruction)) {
          if (!truncation->getOperand(0)->getType()->isFloatTy() ||
              !truncation->getType()->isBFloatTy()) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "only scalar f32-to-bfloat truncation is allowed");
          }
          continue;
        }
        if (const auto* comparison =
                llvm::dyn_cast<llvm::FCmpInst>(&instruction)) {
          if (!comparison->getOperand(0)->getType()->isBFloatTy() ||
              !comparison->getOperand(1)->getType()->isBFloatTy() ||
              comparison->getFastMathFlags().any()) {
            return Rejected(
                module_name, "bfloat-integer-profile",
                "only exact scalar bfloat comparisons are qualified");
          }
          continue;
        }
        if (const auto* extract =
                llvm::dyn_cast<llvm::ExtractElementInst>(&instruction)) {
          const auto* vector = llvm::dyn_cast<llvm::FixedVectorType>(
              extract->getVectorOperandType());
          const auto* index =
              llvm::dyn_cast<llvm::ConstantInt>(extract->getIndexOperand());
          if (vector == nullptr ||
              (vector->getNumElements() != 2 &&
               vector->getNumElements() != 4) ||
              !vector->getElementType()->isBFloatTy() || index == nullptr ||
              index->getZExtValue() >= vector->getNumElements()) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "bfloat extractelement requires width two or four");
          }
          continue;
        }
        if (const auto* insert =
                llvm::dyn_cast<llvm::InsertElementInst>(&instruction)) {
          const auto* vector =
              llvm::dyn_cast<llvm::FixedVectorType>(insert->getType());
          const auto* index =
              llvm::dyn_cast<llvm::ConstantInt>(insert->getOperand(2));
          if (vector == nullptr ||
              (vector->getNumElements() != 2 &&
               vector->getNumElements() != 4) ||
              !vector->getElementType()->isBFloatTy() ||
              !insert->getOperand(1)->getType()->isBFloatTy() ||
              index == nullptr ||
              index->getZExtValue() >= vector->getNumElements() ||
              (!llvm::isa<llvm::PoisonValue>(insert->getOperand(0)) &&
               !llvm::isa<llvm::InsertElementInst>(insert->getOperand(0)))) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "bfloat insertelement requires a width-two or "
                            "width-four chain");
          }
          continue;
        }
        if (const auto* phi = llvm::dyn_cast<llvm::PHINode>(&instruction)) {
          if (!phi->getType()->isBFloatTy() ||
              phi->getNumIncomingValues() != 2) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "bfloat phi requires two scalar incoming values");
          }
          for (const llvm::Value* incoming : phi->incoming_values()) {
            const auto* truncation =
                llvm::dyn_cast<llvm::FPTruncInst>(incoming);
            if (truncation == nullptr ||
                !truncation->getOperand(0)->getType()->isFloatTy()) {
              return Rejected(
                  module_name, "bfloat-integer-profile",
                  "bfloat phi incoming values must truncate directly from f32");
            }
          }
          continue;
        }
        if (const auto* select =
                llvm::dyn_cast<llvm::SelectInst>(&instruction)) {
          if (!select->getType()->isBFloatTy() ||
              !select->getCondition()->getType()->isIntegerTy(1) ||
              !select->getTrueValue()->getType()->isBFloatTy() ||
              !select->getFalseValue()->getType()->isBFloatTy()) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "only scalar bfloat selection is qualified");
          }
          continue;
        }
        if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
          const llvm::Function* callee = call->getCalledFunction();
          if (callee == nullptr || callee->isDeclaration() ||
              !callee->hasLocalLinkage() ||
              !TypeContainsBfloat(callee->getFunctionType())) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "bfloat call requires a qualified local helper");
          }
          continue;
        }
        if (const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(&instruction)) {
          const auto* truncation =
              llvm::dyn_cast_or_null<llvm::FPTruncInst>(ret->getReturnValue());
          if (!function.getReturnType()->isBFloatTy() ||
              truncation == nullptr ||
              !truncation->getOperand(0)->getType()->isFloatTy()) {
            return Rejected(module_name, "bfloat-integer-profile",
                            "bfloat return requires direct f32 truncation");
          }
          continue;
        }
        return Rejected(module_name, "bfloat-integer-profile",
                        absl::StrCat("unreviewed bfloat opcode ",
                                     instruction.getOpcodeName(), " remains"));
      }
    }
  }
  return absl::OkStatus();
}

bool HasUnsupportedInstructionMetadata(const llvm::Instruction& instruction) {
  llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4> metadata;
  instruction.getAllMetadata(metadata);
  return !metadata.empty() || instruction.getDebugLoc();
}

std::string BfloatBitsName(const llvm::Value& value) {
  return value.hasName() ? value.getName().str() + ".bf16_bits" : "bf16_bits";
}

llvm::Type* EncodedBfloatType(llvm::Type* type) {
  llvm::Type* i16 = llvm::Type::getInt16Ty(type->getContext());
  if (type->isBFloatTy()) return i16;
  auto* vector = llvm::dyn_cast<llvm::FixedVectorType>(type);
  if (vector == nullptr || !vector->getElementType()->isBFloatTy() ||
      (vector->getNumElements() != 2 && vector->getNumElements() != 4)) {
    return nullptr;
  }
  return llvm::FixedVectorType::get(i16, vector->getNumElements());
}

absl::Status InlineQualifiedBfloatHelpers(llvm::Module& module,
                                          absl::string_view module_name) {
  std::vector<llvm::Function*> helpers;
  for (llvm::Function& function : module.functions()) {
    if (!TypeContainsBfloat(function.getFunctionType())) continue;
    if (function.isDeclaration() || !function.hasLocalLinkage() ||
        function.getCallingConv() != llvm::CallingConv::C ||
        function.isVarArg() || function.hasPersonalityFn() ||
        function.hasGC() || function.hasPrefixData() ||
        function.hasPrologueData() || function.hasSection() ||
        function.hasComdat()) {
      return Rejected(module_name, "bfloat-helper-inline",
                      "bfloat helper must be a local ordinary-C definition");
    }
    for (const llvm::BasicBlock& block : function) {
      for (const llvm::Instruction& instruction : block) {
        if (llvm::isa<llvm::AllocaInst>(instruction)) {
          return Rejected(module_name, "bfloat-helper-inline",
                          "bfloat helper cannot contain stack allocations");
        }
        const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        if (call == nullptr) continue;
        const auto* plain_call = llvm::dyn_cast<llvm::CallInst>(call);
        const llvm::Function* callee = call->getCalledFunction();
        if (plain_call == nullptr || callee == nullptr ||
            !callee->isIntrinsic() ||
            !IsReviewedPureIntrinsic(callee->getIntrinsicID()) ||
            call->getCallingConv() != llvm::CallingConv::C ||
            plain_call->isTailCall() || !call->getAttributes().isEmpty() ||
            call->hasOperandBundles() || call->getFastMathFlags().any() ||
            HasUnsupportedInstructionMetadata(instruction)) {
          return Rejected(
              module_name, "bfloat-helper-inline",
              "bfloat helper calls must be plain reviewed pure intrinsics");
        }
      }
    }
    for (llvm::User* user : function.users()) {
      auto* call = llvm::dyn_cast<llvm::CallInst>(user);
      if (call == nullptr || call->getCalledFunction() != &function ||
          call->getCallingConv() != llvm::CallingConv::C ||
          call->isTailCall() || !call->getAttributes().isEmpty() ||
          call->hasOperandBundles() || call->getFastMathFlags().any() ||
          HasUnsupportedInstructionMetadata(*call)) {
        return Rejected(module_name, "bfloat-helper-inline",
                        "bfloat helper requires plain direct call users");
      }
    }
    helpers.push_back(&function);
  }

  std::vector<llvm::CallInst*> calls;
  for (llvm::Function& caller : module.functions()) {
    for (llvm::BasicBlock& block : caller) {
      for (llvm::Instruction& instruction : block) {
        auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        if (call == nullptr ||
            std::find(helpers.begin(), helpers.end(),
                      call->getCalledFunction()) == helpers.end()) {
          continue;
        }
        calls.push_back(call);
      }
    }
  }

  for (llvm::CallInst* call : calls) {
    llvm::InlineFunctionInfo info;
    llvm::InlineResult result = llvm::InlineFunction(
        *call, info, /*MergeAttributes=*/false, /*CalleeAAR=*/nullptr,
        /*InsertLifetime=*/false);
    if (!result.isSuccess()) {
      return Rejected(module_name, "bfloat-helper-inline",
                      "mechanical inlining rejected a qualified helper");
    }
  }
  for (llvm::Function* helper : helpers) {
    if (!helper->use_empty()) {
      return Rejected(module_name, "bfloat-helper-inline",
                      "qualified helper retains a non-inlined use");
    }
  }
  for (llvm::Function* helper : helpers) helper->eraseFromParent();
  return absl::OkStatus();
}

bool IsPromotableModularIndexOperation(const llvm::BinaryOperator& operation,
                                       unsigned source_width) {
  if (HasUnsupportedInstructionMetadata(operation) ||
      operation.hasPoisonGeneratingFlags()) {
    return false;
  }
  switch (operation.getOpcode()) {
    case llvm::Instruction::Add:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Mul:
    case llvm::Instruction::And:
    case llvm::Instruction::Or:
    case llvm::Instruction::Xor:
      return true;
    case llvm::Instruction::Shl: {
      const auto* distance =
          llvm::dyn_cast<llvm::ConstantInt>(operation.getOperand(1));
      return distance != nullptr && distance->getZExtValue() < source_width;
    }
    default:
      return false;
  }
}

llvm::Value* PromoteModularIndexExpression(
    llvm::Value* value, llvm::IntegerType* source_type,
    llvm::IntegerType* pointer_index_type, llvm::IRBuilder<>& builder,
    llvm::DenseMap<llvm::Value*, llvm::Value*>& promoted) {
  if (auto found = promoted.find(value); found != promoted.end()) {
    return found->second;
  }

  llvm::Value* result = nullptr;
  auto* operation = llvm::dyn_cast<llvm::BinaryOperator>(value);
  if (operation != nullptr && operation->getType() == source_type &&
      IsPromotableModularIndexOperation(*operation,
                                        source_type->getBitWidth())) {
    llvm::Value* lhs =
        PromoteModularIndexExpression(operation->getOperand(0), source_type,
                                      pointer_index_type, builder, promoted);
    llvm::Value* rhs =
        PromoteModularIndexExpression(operation->getOperand(1), source_type,
                                      pointer_index_type, builder, promoted);
    result = builder.CreateBinOp(
        static_cast<llvm::Instruction::BinaryOps>(operation->getOpcode()), lhs,
        rhs, absl::StrCat(operation->getName().str(), ".pointer_width"));
  } else if (const auto* constant = llvm::dyn_cast<llvm::ConstantInt>(value)) {
    result = llvm::ConstantInt::get(
        pointer_index_type,
        constant->getValue().zextOrTrunc(pointer_index_type->getBitWidth()));
  } else {
    result = builder.CreateZExt(
        value, pointer_index_type,
        absl::StrCat(value->getName().str(), ".pointer_width"));
  }
  promoted[value] = result;
  return result;
}

llvm::Value* MakeBfloatGepIndexPointerWidth(
    llvm::Value* index, llvm::IntegerType* pointer_index_type,
    llvm::IRBuilder<>& builder, absl::string_view name) {
  auto* source_type = llvm::dyn_cast<llvm::IntegerType>(index->getType());
  if (source_type == nullptr) return nullptr;
  const unsigned source_width = source_type->getBitWidth();
  const unsigned pointer_width = pointer_index_type->getBitWidth();
  if (source_width >= pointer_width) {
    return builder.CreateSExtOrTrunc(index, pointer_index_type, name);
  }

  llvm::DenseMap<llvm::Value*, llvm::Value*> promoted;
  llvm::Value* wide = PromoteModularIndexExpression(
      index, source_type, pointer_index_type, builder, promoted);

  // The promoted add/sub/mul/bitwise expression computes the same low N bits
  // as the original N-bit modular expression. Reconstruct the GEP's implicit
  // signed extension from those bits without sending the vendor compiler back
  // through the narrow address-expression DAG.
  llvm::APInt low_bits =
      llvm::APInt::getLowBitsSet(pointer_width, source_width);
  wide = builder.CreateAnd(wide,
                           llvm::ConstantInt::get(pointer_index_type, low_bits),
                           absl::StrCat(name, ".wrapped"));
  const unsigned shift = pointer_width - source_width;
  wide = builder.CreateShl(wide, shift, absl::StrCat(name, ".sign_bits"));
  return builder.CreateAShr(wide, shift, name);
}

absl::Status LowerBfloatStorageGeps(llvm::Module& module,
                                    absl::string_view module_name) {
  std::vector<llvm::GetElementPtrInst*> geps;
  for (llvm::Function& function : module.functions()) {
    for (llvm::BasicBlock& block : function) {
      for (llvm::Instruction& instruction : block) {
        auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction);
        if (gep != nullptr && TypeContainsBfloat(gep->getSourceElementType())) {
          geps.push_back(gep);
        }
      }
    }
  }

  const llvm::DataLayout& data_layout = module.getDataLayout();
  llvm::Type* i16 = llvm::Type::getInt16Ty(module.getContext());
  for (llvm::GetElementPtrInst* gep : geps) {
    auto* source = llvm::dyn_cast<llvm::ArrayType>(gep->getSourceElementType());
    if (source == nullptr || source->getNumElements() == 0 ||
        !source->getElementType()->isBFloatTy() ||
        HasUnsupportedInstructionMetadata(*gep)) {
      return Rejected(module_name, "bfloat-storage-gep",
                      "bfloat GEP is outside the flat array profile");
    }
    llvm::Type* encoded_source =
        llvm::ArrayType::get(i16, source->getNumElements());
    const llvm::TypeSize source_size = data_layout.getTypeAllocSize(source);
    const llvm::TypeSize encoded_size =
        data_layout.getTypeAllocSize(encoded_source);
    if (source_size.isScalable() || encoded_size.isScalable() ||
        source_size.getFixedValue() != encoded_size.getFixedValue() ||
        data_layout.getABITypeAlign(source) !=
            data_layout.getABITypeAlign(encoded_source)) {
      return Rejected(module_name, "bfloat-storage-gep",
                      "i16 storage does not preserve bfloat array layout");
    }
    if (gep->getNumOperands() != 3) {
      return Rejected(
          module_name, "bfloat-storage-gep",
          "bfloat array GEP requires exactly zero and element indices");
    }
    const auto* array_index =
        llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(1));
    if (array_index == nullptr || !array_index->isZero()) {
      return Rejected(module_name, "bfloat-storage-gep",
                      "bfloat array GEP requires a zero array index");
    }

    // The qualified [N x bfloat] GEP has a zero array index and one scalar
    // element index, so its byte offset is exactly the scalar i16 element
    // offset after integer lowering. Rebuild that address expression at the
    // pointer index width and erase the giant array source type: the qualified
    // S80 toolchain otherwise folds large array strides incorrectly.
    llvm::IRBuilder<> builder(gep);
    auto* pointer_index_type = llvm::cast<llvm::IntegerType>(
        data_layout.getIndexType(gep->getPointerOperandType()));
    llvm::SmallVector<llvm::WeakTrackingVH, 4> old_indices;
    llvm::Value* element_index = gep->getOperand(2);
    llvm::Value* pointer_width_index = MakeBfloatGepIndexPointerWidth(
        element_index, pointer_index_type, builder,
        absl::StrCat(BfloatBitsName(*gep), ".index"));
    if (pointer_width_index == nullptr) {
      return Rejected(module_name, "bfloat-storage-gep",
                      "bfloat GEP element index must be a scalar integer");
    }
    if (llvm::isa<llvm::Instruction>(element_index)) {
      old_indices.push_back(element_index);
    }
    llvm::Value* indices[] = {pointer_width_index};
    llvm::GetElementPtrInst* encoded = llvm::GetElementPtrInst::Create(
        i16, gep->getPointerOperand(), indices, gep->getNoWrapFlags(),
        BfloatBitsName(*gep), gep);
    if (encoded->getType() != gep->getType()) {
      return Rejected(module_name, "bfloat-storage-gep",
                      "encoded GEP changed the opaque-pointer result type");
    }
    gep->replaceAllUsesWith(encoded);
    gep->eraseFromParent();
    llvm::RecursivelyDeleteTriviallyDeadInstructionsPermissive(old_indices);
  }
  return absl::OkStatus();
}

class BfloatIntegerLowerer {
 public:
  BfloatIntegerLowerer(llvm::Module& module, absl::string_view module_name)
      : module_(module), module_name_(module_name) {}

  absl::Status Lower() {
    std::vector<llvm::Instruction*> bfloat_values;
    std::vector<llvm::FPExtInst*> extensions;
    std::vector<llvm::FCmpInst*> comparisons;
    std::vector<llvm::StoreInst*> stores;
    for (llvm::Function& function : module_.functions()) {
      for (llvm::BasicBlock& block : function) {
        for (llvm::Instruction& instruction : block) {
          if (TypeContainsBfloat(instruction.getType())) {
            bfloat_values.push_back(&instruction);
          }
          if (auto* extension = llvm::dyn_cast<llvm::FPExtInst>(&instruction);
              extension != nullptr &&
              TypeContainsBfloat(extension->getOperand(0)->getType())) {
            extensions.push_back(extension);
          }
          if (auto* comparison = llvm::dyn_cast<llvm::FCmpInst>(&instruction);
              comparison != nullptr &&
              TypeContainsBfloat(comparison->getOperand(0)->getType())) {
            comparisons.push_back(comparison);
          }
          if (auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction);
              store != nullptr &&
              TypeContainsBfloat(store->getValueOperand()->getType())) {
            stores.push_back(store);
          }
        }
      }
    }

    for (llvm::Instruction* value : bfloat_values) {
      absl::StatusOr<llvm::Value*> encoded = Encode(value);
      if (!encoded.ok()) return encoded.status();
    }
    for (llvm::FPExtInst* extension : extensions) {
      if (HasUnsupportedInstructionMetadata(*extension)) {
        return Rejected(module_name_, "bfloat-integer-lowering",
                        "bfloat extension carries unsupported metadata");
      }
      absl::StatusOr<llvm::Value*> encoded = Encode(extension->getOperand(0));
      if (!encoded.ok()) return encoded.status();
      llvm::Value* decoded = Decode(*encoded, extension);
      extension->replaceAllUsesWith(decoded);
    }
    for (llvm::FCmpInst* comparison : comparisons) {
      if (HasUnsupportedInstructionMetadata(*comparison) ||
          comparison->getFastMathFlags().any()) {
        return Rejected(module_name_, "bfloat-integer-lowering",
                        "bfloat comparison carries unsupported semantics");
      }
      absl::StatusOr<llvm::Value*> lhs = Encode(comparison->getOperand(0));
      if (!lhs.ok()) return lhs.status();
      absl::StatusOr<llvm::Value*> rhs = Encode(comparison->getOperand(1));
      if (!rhs.ok()) return rhs.status();
      llvm::IRBuilder<> builder(comparison);
      llvm::Value* lowered = builder.CreateFCmp(
          comparison->getPredicate(), Decode(*lhs, comparison),
          Decode(*rhs, comparison), comparison->getName());
      comparison->replaceAllUsesWith(lowered);
    }
    for (llvm::StoreInst* store : stores) {
      if (HasUnsupportedInstructionMetadata(*store)) {
        return Rejected(module_name_, "bfloat-integer-lowering",
                        "bfloat store carries unsupported metadata");
      }
      absl::StatusOr<llvm::Value*> encoded = Encode(store->getValueOperand());
      if (!encoded.ok()) return encoded.status();
      llvm::IRBuilder<> builder(store);
      llvm::StoreInst* lowered =
          builder.CreateStore(*encoded, store->getPointerOperand());
      lowered->setAlignment(store->getAlign());
    }

    for (llvm::FPExtInst* extension : extensions) extension->eraseFromParent();
    for (llvm::FCmpInst* comparison : comparisons)
      comparison->eraseFromParent();
    for (llvm::StoreInst* store : stores) store->eraseFromParent();

    for (llvm::Instruction* instruction : obsolete_) {
      for (llvm::User* user : instruction->users()) {
        if (encoded_.find(user) == encoded_.end()) {
          return Rejected(module_name_, "bfloat-integer-lowering",
                          "lowered bfloat value retains an unsupported user");
        }
      }
    }
    for (llvm::Instruction* instruction : obsolete_) {
      instruction->dropAllReferences();
    }
    for (llvm::Instruction* instruction : obsolete_) {
      instruction->eraseFromParent();
    }
    return absl::OkStatus();
  }

 private:
  llvm::Value* Decode(llvm::Value* encoded, llvm::Instruction* insertion) {
    llvm::IRBuilder<> builder(insertion);
    llvm::Value* widened =
        builder.CreateZExt(encoded, builder.getInt32Ty(), "bf16_widened_bits");
    llvm::Value* shifted = builder.CreateShl(widened, 16, "bf16_shifted_bits");
    return builder.CreateBitCast(shifted, builder.getFloatTy(), "bf16_value");
  }

  absl::StatusOr<llvm::Value*> Encode(llvm::Value* value) {
    if (auto found = encoded_.find(value); found != encoded_.end()) {
      return found->second;
    }
    llvm::Type* encoded_type = EncodedBfloatType(value->getType());
    if (encoded_type == nullptr) {
      return Rejected(module_name_, "bfloat-integer-lowering",
                      "value is outside the scalar/width-two/width-four "
                      "profile");
    }
    if (auto* constant = llvm::dyn_cast<llvm::ConstantFP>(value)) {
      llvm::APInt bits = constant->getValueAPF().bitcastToAPInt();
      if (bits.getBitWidth() != 16) {
        return Rejected(module_name_, "bfloat-integer-lowering",
                        "bfloat constant has an unexpected bit width");
      }
      llvm::Value* encoded = llvm::ConstantInt::get(module_.getContext(), bits);
      encoded_[value] = encoded;
      return encoded;
    }
    if (llvm::isa<llvm::PoisonValue>(value)) {
      llvm::Value* encoded = llvm::PoisonValue::get(encoded_type);
      encoded_[value] = encoded;
      return encoded;
    }

    auto* instruction = llvm::dyn_cast<llvm::Instruction>(value);
    if (instruction == nullptr ||
        HasUnsupportedInstructionMetadata(*instruction)) {
      return Rejected(module_name_, "bfloat-integer-lowering",
                      "bfloat value carries an unsupported form or metadata");
    }

    if (auto* phi = llvm::dyn_cast<llvm::PHINode>(instruction)) {
      auto* encoded = llvm::PHINode::Create(
          encoded_type, phi->getNumIncomingValues(), BfloatBitsName(*phi), phi);
      encoded_[value] = encoded;
      obsolete_.push_back(phi);
      for (unsigned index = 0; index < phi->getNumIncomingValues(); ++index) {
        absl::StatusOr<llvm::Value*> incoming =
            Encode(phi->getIncomingValue(index));
        if (!incoming.ok()) return incoming.status();
        encoded->addIncoming(*incoming, phi->getIncomingBlock(index));
      }
      return encoded;
    }

    llvm::IRBuilder<> builder(instruction);
    llvm::Value* encoded = nullptr;
    if (auto* load = llvm::dyn_cast<llvm::LoadInst>(instruction)) {
      llvm::LoadInst* lowered = builder.CreateLoad(
          encoded_type, load->getPointerOperand(), BfloatBitsName(*load));
      lowered->setAlignment(load->getAlign());
      encoded = lowered;
    } else if (auto* truncation =
                   llvm::dyn_cast<llvm::FPTruncInst>(instruction)) {
      encoded = EncodeFloat(*truncation);
    } else if (auto* extract =
                   llvm::dyn_cast<llvm::ExtractElementInst>(instruction)) {
      absl::StatusOr<llvm::Value*> vector = Encode(extract->getVectorOperand());
      if (!vector.ok()) return vector.status();
      encoded = builder.CreateExtractElement(
          *vector, extract->getIndexOperand(), BfloatBitsName(*extract));
    } else if (auto* insert =
                   llvm::dyn_cast<llvm::InsertElementInst>(instruction)) {
      absl::StatusOr<llvm::Value*> vector = Encode(insert->getOperand(0));
      if (!vector.ok()) return vector.status();
      absl::StatusOr<llvm::Value*> element = Encode(insert->getOperand(1));
      if (!element.ok()) return element.status();
      encoded = builder.CreateInsertElement(
          *vector, *element, insert->getOperand(2), BfloatBitsName(*insert));
    } else if (auto* select = llvm::dyn_cast<llvm::SelectInst>(instruction)) {
      absl::StatusOr<llvm::Value*> true_value = Encode(select->getTrueValue());
      if (!true_value.ok()) return true_value.status();
      absl::StatusOr<llvm::Value*> false_value =
          Encode(select->getFalseValue());
      if (!false_value.ok()) return false_value.status();
      encoded = builder.CreateSelect(select->getCondition(), *true_value,
                                     *false_value, BfloatBitsName(*select));
    } else {
      return Rejected(module_name_, "bfloat-integer-lowering",
                      "unreviewed bfloat-producing instruction remains");
    }

    encoded_[value] = encoded;
    obsolete_.push_back(instruction);
    return encoded;
  }

  llvm::Value* EncodeFloat(llvm::FPTruncInst& truncation) {
    llvm::IRBuilder<> builder(&truncation);
    llvm::Type* i16 = builder.getInt16Ty();
    llvm::Type* i32 = builder.getInt32Ty();
    llvm::Value* bits = builder.CreateBitCast(truncation.getOperand(0), i32);
    llvm::Value* upper = builder.CreateLShr(bits, 16);
    llvm::Value* least_significant = builder.CreateAnd(upper, 1);
    llvm::Value* bias = builder.CreateAdd(builder.getInt32(0x7fff),
                                          least_significant, "bf16_round_bias");
    llvm::Value* rounded = builder.CreateAdd(bits, bias, "bf16_rounded");
    llvm::Value* finite_encoded =
        builder.CreateTrunc(builder.CreateLShr(rounded, 16), i16);
    llvm::Value* magnitude =
        builder.CreateAnd(bits, builder.getInt32(0x7fffffff));
    llvm::Value* is_nan = builder.CreateICmpUGT(
        magnitude, builder.getInt32(0x7f800000), "bf16_is_nan");
    llvm::Value* nan_encoded = builder.CreateOr(builder.CreateTrunc(upper, i16),
                                                builder.getInt16(0x40));
    return builder.CreateSelect(is_nan, nan_encoded, finite_encoded,
                                BfloatBitsName(truncation));
  }

  llvm::Module& module_;
  absl::string_view module_name_;
  llvm::DenseMap<llvm::Value*, llvm::Value*> encoded_;
  std::vector<llvm::Instruction*> obsolete_;
};

bool ConstantTreeContainsBfloat(
    const llvm::Constant& constant,
    llvm::DenseSet<const llvm::Constant*>& visited) {
  if (!visited.insert(&constant).second) return false;
  if (TypeContainsBfloat(constant.getType())) return true;
  if (const auto* gep = llvm::dyn_cast<llvm::GEPOperator>(&constant);
      gep != nullptr && TypeContainsBfloat(gep->getSourceElementType())) {
    return true;
  }
  for (const llvm::Use& operand : constant.operands()) {
    if (const auto* child = llvm::dyn_cast<llvm::Constant>(operand.get());
        child != nullptr && ConstantTreeContainsBfloat(*child, visited)) {
      return true;
    }
  }
  return false;
}

absl::Status ValidateNoBfloatTypes(const llvm::Module& module,
                                   absl::string_view module_name) {
  for (const llvm::StructType* type : module.getIdentifiedStructTypes()) {
    if (TypeContainsBfloat(type)) {
      return Rejected(module_name, "bfloat-integer-lowering",
                      "identified bfloat type remains after lowering");
    }
  }
  for (const llvm::GlobalVariable& global : module.globals()) {
    llvm::DenseSet<const llvm::Constant*> visited;
    if (TypeContainsBfloat(global.getValueType()) ||
        (global.hasInitializer() &&
         ConstantTreeContainsBfloat(*global.getInitializer(), visited))) {
      return Rejected(module_name, "bfloat-integer-lowering",
                      "global bfloat type remains after lowering");
    }
  }
  for (const llvm::Function& function : module.functions()) {
    if (TypeContainsBfloat(function.getFunctionType())) {
      return Rejected(module_name, "bfloat-integer-lowering",
                      "function bfloat type remains after lowering");
    }
    for (const llvm::BasicBlock& block : function) {
      for (const llvm::Instruction& instruction : block) {
        if (TypeContainsBfloat(instruction.getType())) {
          return Rejected(module_name, "bfloat-integer-lowering",
                          "instruction bfloat result remains after lowering");
        }
        if (const auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(&instruction);
            alloca != nullptr &&
            TypeContainsBfloat(alloca->getAllocatedType())) {
          return Rejected(module_name, "bfloat-integer-lowering",
                          "bfloat alloca remains after lowering");
        }
        if (const auto* gep =
                llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction);
            gep != nullptr && TypeContainsBfloat(gep->getSourceElementType())) {
          return Rejected(module_name, "bfloat-integer-lowering",
                          "bfloat GEP remains after lowering");
        }
        for (const llvm::Use& operand : instruction.operands()) {
          if (TypeContainsBfloat(operand->getType())) {
            return Rejected(
                module_name, "bfloat-integer-lowering",
                "instruction bfloat operand remains after lowering");
          }
          if (const auto* constant =
                  llvm::dyn_cast<llvm::Constant>(operand.get())) {
            llvm::DenseSet<const llvm::Constant*> visited;
            if (ConstantTreeContainsBfloat(*constant, visited)) {
              return Rejected(module_name, "bfloat-integer-lowering",
                              "constant bfloat type remains after lowering");
            }
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status LowerBfloatMemoryOperations(llvm::Module& module,
                                         absl::string_view module_name) {
  // "Native" names the reviewed current-LLVM input profile. No native bfloat
  // type is allowed to survive the integer-only lowering below.
  if (absl::Status status =
          ValidateBfloatIntegerLoweringProfile(module, module_name);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = InlineQualifiedBfloatHelpers(module, module_name);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = LowerBfloatStorageGeps(module, module_name);
      !status.ok()) {
    return status;
  }
  BfloatIntegerLowerer lowerer(module, module_name);
  if (absl::Status status = lowerer.Lower(); !status.ok()) return status;
  return ValidateNoBfloatTypes(module, module_name);
}

absl::Status CollectExportedGlobals(const llvm::Module& module,
                                    absl::string_view module_name,
                                    MusaBridgeIrMetadata& metadata) {
  const llvm::DataLayout& data_layout = module.getDataLayout();
  for (const llvm::GlobalVariable& global : module.globals()) {
    if (global.hasLocalLinkage()) continue;
    if (global.getLinkage() != llvm::GlobalValue::ExternalLinkage ||
        global.isDeclaration()) {
      return Rejected(module_name, "exported-globals",
                      "nonlocal globals require an external definition");
    }
    const MusaAddressSpaceSpec* address_space =
        FindMusaAddressSpace(global.getAddressSpace());
    if (address_space == nullptr || !address_space->allowed_in_interchange) {
      return Rejected(module_name, "exported-globals",
                      "global uses an unqualified address space");
    }
    llvm::TypeSize size = data_layout.getTypeAllocSize(global.getValueType());
    if (size.isScalable() || size.getFixedValue() == 0) {
      return Rejected(module_name, "exported-globals",
                      "global allocation size must be fixed and nonzero");
    }
    const uint64_t alignment =
        global.getAlign()
            ? global.getAlign()->value()
            : data_layout.getABITypeAlign(global.getValueType()).value();
    metadata.exported_globals.push_back(MusaExportedGlobal{
        global.getName().str(),
        global.isConstant() ? MusaExportedGlobalKind::kConstant
                            : MusaExportedGlobalKind::kMutable,
        global.getAddressSpace(), size.getFixedValue(), alignment});
  }
  std::sort(
      metadata.exported_globals.begin(), metadata.exported_globals.end(),
      [](const MusaExportedGlobal& left, const MusaExportedGlobal& right) {
        return left.name < right.name;
      });
  return absl::OkStatus();
}

absl::StatusOr<std::string> RewriteCurrentMemoryAttributes(
    absl::string_view current_text, absl::string_view module_name) {
  std::vector<absl::string_view> lines = absl::StrSplit(current_text, '\n');
  std::string normalized;
  normalized.reserve(current_text.size());
  for (size_t index = 0; index < lines.size(); ++index) {
    absl::string_view line = lines[index];
    absl::string_view trimmed = absl::StripAsciiWhitespace(line);
    if (absl::StartsWith(trimmed, "; Function Attrs:")) continue;

    std::string rewritten(line);
    if (absl::StrContains(rewritten, "memory(")) {
      if (!absl::StartsWith(trimmed, "attributes #")) {
        return Rejected(module_name, "memory-attributes",
                        "current memory syntax appeared outside an attribute "
                        "group");
      }
      rewritten = absl::StrReplaceAll(
          rewritten,
          {{"memory(inaccessiblemem: readwrite)", "inaccessiblememonly"},
           {"memory(inaccessiblemem: read)", "inaccessiblememonly readonly"},
           {"memory(argmem: readwrite)", "argmemonly"},
           {"memory(argmem: read)", "argmemonly readonly"},
           {"memory(argmem: write)", "argmemonly writeonly"},
           {"memory(none)", "readnone"}});
      if (absl::StrContains(rewritten, "memory(")) {
        return Rejected(module_name, "memory-attributes",
                        "memory effect has no reviewed LLVM 14 spelling");
      }
    }
    normalized.append(rewritten);
    if (index + 1 < lines.size()) normalized.push_back('\n');
  }
  return normalized;
}

int HexDigitValue(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

absl::StatusOr<std::string> FormatLlvm14FloatLiteral(
    llvm::APFloat value, absl::string_view module_name) {
  bool loses_information = false;
  const llvm::APFloat::opStatus status =
      value.convert(llvm::APFloat::IEEEdouble(),
                    llvm::APFloat::rmNearestTiesToEven, &loses_information);
  if (status != llvm::APFloat::opOK || loses_information) {
    return Rejected(module_name, "float-literals",
                    "float literal cannot be represented by LLVM 14");
  }
  return absl::StrFormat(
      "0x%016X", static_cast<uint64_t>(value.bitcastToAPInt().getZExtValue()));
}

bool IsLlvmIdentifierBody(char value) {
  return absl::ascii_isalnum(static_cast<unsigned char>(value)) ||
         value == '-' || value == '$' || value == '.' || value == '_';
}

bool IsLlvmTokenTerminator(char value) {
  return absl::ascii_isspace(static_cast<unsigned char>(value)) ||
         value == ',' || value == ')' || value == ']' || value == '}' ||
         value == '>';
}

struct DecimalFloatLiteral {
  size_t end;
  // Empty when LLVM 14 accepts the exact decimal spelling unchanged.
  std::string llvm14_spelling;
};

absl::StatusOr<std::optional<DecimalFloatLiteral>> ParseSpecialFloatLiteral(
    absl::string_view current_text, size_t literal_start,
    absl::string_view module_name) {
  struct SpecialValue {
    absl::string_view spelling;
    uint32_t single_bits;
  };
  constexpr SpecialValue kSpecialValues[] = {
      {"+inf", 0x7f800000},  {"-inf", 0xff800000}, {"+qnan", 0x7fc00000},
      {"-qnan", 0xffc00000}, {"nan", 0x7fc00000},
  };
  for (const SpecialValue& special : kSpecialValues) {
    const size_t end = literal_start + special.spelling.size();
    if (end > current_text.size() ||
        current_text.substr(literal_start, special.spelling.size()) !=
            special.spelling ||
        (end < current_text.size() &&
         !IsLlvmTokenTerminator(current_text[end]))) {
      continue;
    }
    llvm::APFloat value(llvm::APFloat::IEEEsingle(),
                        llvm::APInt(32, special.single_bits));
    absl::StatusOr<std::string> rewritten =
        FormatLlvm14FloatLiteral(std::move(value), module_name);
    if (!rewritten.ok()) return rewritten.status();
    return DecimalFloatLiteral{end, std::move(*rewritten)};
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<DecimalFloatLiteral>> ParseDecimalFloatLiteral(
    absl::string_view current_text, size_t literal_start,
    absl::string_view module_name) {
  size_t cursor = literal_start;
  if (current_text[cursor] == '+' || current_text[cursor] == '-') ++cursor;
  const size_t integer_start = cursor;
  while (
      cursor < current_text.size() &&
      absl::ascii_isdigit(static_cast<unsigned char>(current_text[cursor]))) {
    ++cursor;
  }
  if (cursor == integer_start) return std::nullopt;

  bool has_fraction_or_exponent = false;
  if (cursor < current_text.size() && current_text[cursor] == '.') {
    has_fraction_or_exponent = true;
    ++cursor;
    const size_t fraction_start = cursor;
    while (
        cursor < current_text.size() &&
        absl::ascii_isdigit(static_cast<unsigned char>(current_text[cursor]))) {
      ++cursor;
    }
    if (cursor == fraction_start) return std::nullopt;
  }
  if (cursor < current_text.size() &&
      (current_text[cursor] == 'e' || current_text[cursor] == 'E')) {
    has_fraction_or_exponent = true;
    ++cursor;
    if (cursor < current_text.size() &&
        (current_text[cursor] == '+' || current_text[cursor] == '-')) {
      ++cursor;
    }
    const size_t exponent_start = cursor;
    while (
        cursor < current_text.size() &&
        absl::ascii_isdigit(static_cast<unsigned char>(current_text[cursor]))) {
      ++cursor;
    }
    if (cursor == exponent_start) return std::nullopt;
  }
  if (!has_fraction_or_exponent ||
      (cursor < current_text.size() &&
       !IsLlvmTokenTerminator(current_text[cursor]))) {
    return std::nullopt;
  }

  llvm::APFloat value(llvm::APFloat::IEEEsingle());
  llvm::Expected<llvm::APFloat::opStatus> parsed = value.convertFromString(
      llvm::StringRef(current_text.data() + literal_start,
                      cursor - literal_start),
      llvm::APFloat::rmNearestTiesToEven);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return Rejected(module_name, "float-literals",
                    "current-LLVM decimal float literal could not be parsed");
  }
  if ((*parsed & (llvm::APFloat::opInvalidOp | llvm::APFloat::opOverflow)) !=
      0) {
    return Rejected(module_name, "float-literals",
                    "current-LLVM decimal float literal is not finite");
  }
  // LLVM 14 accepts decimal f32 literals only when conversion is exact.
  // Preserve those spellings so existing interchange goldens stay stable.
  if (*parsed == llvm::APFloat::opOK) {
    return DecimalFloatLiteral{cursor, std::string()};
  }

  absl::StatusOr<std::string> rewritten =
      FormatLlvm14FloatLiteral(std::move(value), module_name);
  if (!rewritten.ok()) return rewritten.status();
  return DecimalFloatLiteral{cursor, std::move(*rewritten)};
}

// Current LLVM can print finite f32 constants in decimal forms that LLVM 14
// rejects. Recognize only numeric operands associated with an f32 `float`
// type, including every incoming value of a scalar `phi float`; declarations,
// SSA operands, other floating types, comments, and strings are left untouched.
absl::StatusOr<size_t> RewriteDecimalFloatAfterType(
    absl::string_view current_text, size_t type_start,
    absl::string_view module_name, std::string& normalized) {
  constexpr absl::string_view kFloatType = "float";
  const size_t after_type = type_start + kFloatType.size();
  if ((type_start != 0 && IsLlvmIdentifierBody(current_text[type_start - 1])) ||
      (after_type < current_text.size() &&
       IsLlvmIdentifierBody(current_text[after_type]))) {
    return type_start;
  }

  size_t operand_start = after_type;
  while (operand_start < current_text.size() &&
         (current_text[operand_start] == ' ' ||
          current_text[operand_start] == '\t')) {
    ++operand_start;
  }
  if (operand_start == after_type || operand_start >= current_text.size()) {
    return type_start;
  }

  if (current_text[operand_start] != '[') {
    auto parse_literal = [&](size_t start)
        -> absl::StatusOr<std::optional<DecimalFloatLiteral>> {
      absl::StatusOr<std::optional<DecimalFloatLiteral>> literal =
          ParseSpecialFloatLiteral(current_text, start, module_name);
      if (!literal.ok() || literal->has_value()) return literal;
      return ParseDecimalFloatLiteral(current_text, start, module_name);
    };

    // Binary floating-point instructions spell the type only once. Therefore
    // the second operand of e.g. `fcmp one float %value, +inf` must be examined
    // together with the first operand; a token-only scan would either miss it
    // or risk rewriting comments and quoted strings.
    const size_t line_end = current_text.find('\n', operand_start);
    size_t bounded_line_end =
        line_end == absl::string_view::npos ? current_text.size() : line_end;
    const size_t comment_start = current_text.find(';', operand_start);
    if (comment_start != absl::string_view::npos &&
        comment_start < bounded_line_end) {
      bounded_line_end = comment_start;
    }

    absl::StatusOr<std::optional<DecimalFloatLiteral>> first =
        parse_literal(operand_start);
    if (!first.ok()) return first.status();

    size_t second_start = absl::string_view::npos;
    absl::StatusOr<std::optional<DecimalFloatLiteral>> second = std::nullopt;
    const size_t comma = current_text.find(',', operand_start);
    if (comma != absl::string_view::npos && comma < bounded_line_end) {
      second_start = comma + 1;
      while (second_start < bounded_line_end &&
             (current_text[second_start] == ' ' ||
              current_text[second_start] == '\t')) {
        ++second_start;
      }
      if (second_start < bounded_line_end) {
        second = parse_literal(second_start);
        if (!second.ok()) return second.status();
      }
    }

    const bool rewrite_first =
        first->has_value() && !(*first)->llvm14_spelling.empty();
    const bool rewrite_second =
        second->has_value() && !(*second)->llvm14_spelling.empty();
    if (!rewrite_first && !rewrite_second) return type_start;

    size_t copy_start = type_start;
    if (rewrite_first) {
      normalized.append(
          current_text.substr(copy_start, operand_start - copy_start));
      normalized.append((*first)->llvm14_spelling);
      copy_start = (*first)->end;
    }
    if (rewrite_second) {
      normalized.append(
          current_text.substr(copy_start, second_start - copy_start));
      normalized.append((*second)->llvm14_spelling);
      copy_start = (*second)->end;
    }
    return copy_start;
  }

  const size_t line_end = current_text.find('\n', operand_start);
  const size_t bounded_line_end =
      line_end == absl::string_view::npos ? current_text.size() : line_end;
  size_t copy_start = type_start;
  size_t scan = operand_start;
  bool changed = false;
  while (scan < bounded_line_end) {
    const size_t bracket = current_text.find('[', scan);
    if (bracket == absl::string_view::npos || bracket >= bounded_line_end) {
      break;
    }
    size_t literal_start = bracket + 1;
    while (literal_start < bounded_line_end &&
           (current_text[literal_start] == ' ' ||
            current_text[literal_start] == '\t')) {
      ++literal_start;
    }
    absl::StatusOr<std::optional<DecimalFloatLiteral>> literal =
        ParseSpecialFloatLiteral(current_text, literal_start, module_name);
    if (!literal.ok()) return literal.status();
    if (!literal->has_value()) {
      literal =
          ParseDecimalFloatLiteral(current_text, literal_start, module_name);
      if (!literal.ok()) return literal.status();
    }
    if (!literal->has_value()) {
      scan = literal_start;
      continue;
    }
    if (!(*literal)->llvm14_spelling.empty()) {
      normalized.append(
          current_text.substr(copy_start, literal_start - copy_start));
      normalized.append((*literal)->llvm14_spelling);
      copy_start = (*literal)->end;
      changed = true;
    }
    scan = (*literal)->end;
  }
  if (!changed) return type_start;
  normalized.append(
      current_text.substr(copy_start, bounded_line_end - copy_start));
  return bounded_line_end;
}

absl::StatusOr<std::string> RewriteCurrentFloatLiterals(
    absl::string_view current_text, absl::string_view module_name) {
  std::string normalized;
  normalized.reserve(current_text.size());
  for (size_t index = 0; index < current_text.size();) {
    if (index + 5 <= current_text.size() &&
        current_text.substr(index, 5) == "float") {
      absl::StatusOr<size_t> after = RewriteDecimalFloatAfterType(
          current_text, index, module_name, normalized);
      if (!after.ok()) return after.status();
      if (*after != index) {
        index = *after;
        continue;
      }
    }
    if (index + 3 > current_text.size() ||
        current_text.substr(index, 3) != "f0x" ||
        (index != 0 && !absl::ascii_isspace(static_cast<unsigned char>(
                           current_text[index - 1])))) {
      normalized.push_back(current_text[index++]);
      continue;
    }
    constexpr size_t kFloatHexDigits = 8;
    if (index + 3 + kFloatHexDigits > current_text.size()) {
      return Rejected(module_name, "float-literals",
                      "truncated current-LLVM float literal");
    }
    uint32_t single_bits = 0;
    for (size_t digit = 0; digit < kFloatHexDigits; ++digit) {
      const int nibble = HexDigitValue(current_text[index + 3 + digit]);
      if (nibble < 0) {
        return Rejected(module_name, "float-literals",
                        "malformed current-LLVM float literal");
      }
      single_bits = (single_bits << 4) | static_cast<uint32_t>(nibble);
    }
    const size_t after = index + 3 + kFloatHexDigits;
    if (after < current_text.size() &&
        HexDigitValue(current_text[after]) >= 0) {
      return Rejected(module_name, "float-literals",
                      "overlong current-LLVM float literal");
    }

    llvm::APFloat value(llvm::APFloat::IEEEsingle(),
                        llvm::APInt(32, single_bits));
    absl::StatusOr<std::string> rewritten =
        FormatLlvm14FloatLiteral(std::move(value), module_name);
    if (!rewritten.ok()) return rewritten.status();
    normalized.append(*rewritten);
    index = after;
  }
  return normalized;
}

absl::Status VerifyNormalizedText(absl::string_view normalized,
                                  absl::string_view module_name,
                                  const MusaBridgeIrMetadata& metadata) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> reparsed = llvm::parseAssemblyString(
      llvm::StringRef(normalized.data(), normalized.size()), diagnostic,
      context);
  if (reparsed == nullptr) {
    return Rejected(module_name, "current-reparse",
                    "normalized LLVM failed current parser validation");
  }
  if (llvm::verifyModule(*reparsed)) {
    return Rejected(module_name, "current-verifier",
                    "normalized LLVM failed current verifier validation");
  }
  return ValidateMusaBridgeIr(normalized, metadata);
}

}  // namespace

absl::StatusOr<MusaLlvm14CompatibilityResult> NormalizeMusaLlvmForLlvm14(
    const llvm::Module& input, absl::string_view module_name) {
  if (!IsSafeModuleName(module_name)) {
    return Rejected(module_name, "module-name",
                    "module name must be a bounded safe token");
  }
  if (llvm::verifyModule(input)) {
    return Rejected(module_name, "current-verifier",
                    "input module failed current LLVM verification");
  }
  if (absl::Status status = ValidateCurrentTarget(input, module_name);
      !status.ok()) {
    return status;
  }
  const std::string input_text = PrintModule(input);
  if (input_text.empty() || input_text.size() > kMaxMusaInterchangeIrBytes) {
    return Rejected(module_name, "input-size",
                    "current LLVM input must be nonempty and bounded");
  }

  std::unique_ptr<llvm::Module> module = llvm::CloneModule(input);
  module->setModuleIdentifier(std::string(module_name));
  module->setSourceFileName(std::string(module_name));
  llvm::StripDebugInfo(*module);

  if (absl::Status status = StripReviewedNamedMetadata(*module, module_name);
      !status.ok()) {
    return status;
  }
  if (absl::Status status =
          StripLegacyBitonicOptimizationHints(*module, module_name);
      !status.ok()) {
    return status;
  }

  MusaBridgeIrMetadata metadata;
  metadata.module_name = std::string(module_name);
  if (absl::Status status = NormalizeFunctions(*module, module_name, metadata);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = LowerFloatingMinMaxIntrinsics(*module, module_name);
      !status.ok()) {
    return status;
  }
  StripInvariantLoadMetadata(*module);
  if (absl::Status status = LowerBfloatMemoryOperations(*module, module_name);
      !status.ok()) {
    return status;
  }
  if (absl::Status status =
          CollectExportedGlobals(*module, module_name, metadata);
      !status.ok()) {
    return status;
  }
  if (metadata.kernel_entry_names.empty() &&
      metadata.exported_globals.empty()) {
    return Rejected(module_name, "module-exports",
                    "module must export at least one kernel or typed global");
  }

  const std::string current_text = PrintModule(*module);
  // Validate the semantic current-LLVM form before changing only the textual
  // spelling of memory effects.
  if (absl::Status status = ValidateMusaBridgeIr(current_text, metadata);
      !status.ok()) {
    return status;
  }

  absl::StatusOr<std::string> normalized =
      RewriteCurrentMemoryAttributes(current_text, module_name);
  if (!normalized.ok()) return normalized.status();
  normalized = RewriteCurrentFloatLiterals(*normalized, module_name);
  if (!normalized.ok()) return normalized.status();
  if (normalized->empty() || normalized->size() > kMaxMusaInterchangeIrBytes) {
    return Rejected(module_name, "output-size",
                    "normalized LLVM output must be nonempty and bounded");
  }
  if (absl::Status status =
          VerifyNormalizedText(*normalized, module_name, metadata);
      !status.ok()) {
    return status;
  }

  MusaLlvm14CompatibilityResult result;
  result.normalized_llvm = std::move(*normalized);
  result.normalized_llvm_sha256 = MusaBridgeSha256Hex(result.normalized_llvm);
  result.metadata = std::move(metadata);
  return result;
}

absl::StatusOr<MusaLlvm14CompatibilityResult> NormalizeMusaLlvmTextForLlvm14(
    absl::string_view current_llvm, absl::string_view module_name) {
  if (current_llvm.empty() ||
      current_llvm.size() > kMaxMusaInterchangeIrBytes) {
    return Rejected(module_name, "input-size",
                    "textual current LLVM input must be nonempty and bounded");
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
      llvm::StringRef(current_llvm.data(), current_llvm.size()), diagnostic,
      context);
  if (module == nullptr) {
    return Rejected(module_name, "current-parse",
                    "current LLVM parser rejected the module");
  }
  return NormalizeMusaLlvmForLlvm14(*module, module_name);
}

}  // namespace xla::gpu::musa
