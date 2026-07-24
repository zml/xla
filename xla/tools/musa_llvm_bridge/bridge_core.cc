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

#include "xla/tools/musa_llvm_bridge/bridge_core.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <dlfcn.h>
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"

namespace xla::gpu::musa::bridge {
namespace {

constexpr absl::string_view kGlobalsOnlyAnchorKernel =
    "__musa_xla_globals_anchor_v1";

using ::xla::gpu::musa::FindMusaAddressSpace;
using ::xla::gpu::musa::FindMusaShim;
using ::xla::gpu::musa::MusaAddressSpaceSpec;
using ::xla::gpu::musa::MusaMemoryEffects;
using ::xla::gpu::musa::MusaShimSignature;
using ::xla::gpu::musa::MusaShimSpec;

std::string SanitizedSymbol(llvm::StringRef value) {
  if (value.empty() || value.size() > kMusaBridgeMaxSymbolNameBytes) {
    return "<invalid-symbol>";
  }
  const auto valid_first = [](char c) {
    return absl::ascii_isalpha(c) || c == '_' || c == '.' || c == '$';
  };
  const auto valid_rest = [&](char c) {
    return valid_first(c) || absl::ascii_isdigit(c) || c == '-';
  };
  if (!valid_first(value.front()) ||
      !std::all_of(value.begin() + 1, value.end(), valid_rest)) {
    return "<invalid-symbol>";
  }
  return value.str();
}

// Vendor LLVM 14 must independently enforce the mapping-v2 fast-math
// contract. `nsz` is the sole accepted relaxation.
bool HasOnlyNoSignedZeros(const llvm::FastMathFlags& flags) {
  return flags.noSignedZeros() && !flags.allowReassoc() && !flags.noNaNs() &&
         !flags.noInfs() && !flags.allowReciprocal() &&
         !flags.allowContract() && !flags.approxFunc();
}

// Revalidate mapping v3's single probed atomic primitive inside the isolated
// vendor-LLVM process; never trust the host validator as a security boundary.
bool IsQualifiedMusaAtomicCmpXchg(const llvm::AtomicCmpXchgInst& cmpxchg) {
  return cmpxchg.getPointerAddressSpace() == kMusaAtomicCmpXchgAddressSpace &&
         cmpxchg.getCompareOperand()->getType()->isIntegerTy(
             kMusaAtomicCmpXchgBitWidth) &&
         cmpxchg.getNewValOperand()->getType()->isIntegerTy(
             kMusaAtomicCmpXchgBitWidth) &&
         !cmpxchg.isWeak() && !cmpxchg.isVolatile() &&
         cmpxchg.getSyncScopeID() == llvm::SyncScope::System &&
         cmpxchg.getSuccessOrdering() == llvm::AtomicOrdering::Monotonic &&
         cmpxchg.getFailureOrdering() == llvm::AtomicOrdering::Monotonic &&
         cmpxchg.getAlign().value() == kMusaAtomicCmpXchgAlignment;
}

absl::Status Rejected(const MusaBridgeCompileRequest& request,
                      absl::string_view capability, absl::string_view detail) {
  // ValidateMusaBridgeCompileRequest has already constrained module_name and
  // every symbol before this helper is called. LLVM parser/verifier text is
  // deliberately never attached because it can contain attacker-controlled
  // IR or local paths from the vendor library.
  return absl::InvalidArgumentError(absl::StrFormat(
      "MUSA vendor bridge rejected module=%s mapping=%d capability=%s: %s",
      request.module_name(), request.mapping_version(), capability, detail));
}

absl::Status ToolchainMismatch(const MusaBridgeCompileRequest& request,
                               absl::string_view detail) {
  return absl::FailedPreconditionError(absl::StrFormat(
      "MUSA vendor LLVM mapping mismatch for module=%s mapping=%d: %s",
      request.module_name(), request.mapping_version(), detail));
}

absl::Status ValidateType(const llvm::Type* type,
                          const MusaBridgeCompileRequest& request,
                          llvm::SmallPtrSetImpl<const llvm::Type*>& visited) {
  if (!visited.insert(type).second) return absl::OkStatus();

  if (const auto* pointer = llvm::dyn_cast<llvm::PointerType>(type)) {
    if (!pointer->isOpaque()) {
      return Rejected(request, "pointer-model",
                      "the active mapping requires opaque pointers");
    }
    const MusaAddressSpaceSpec* address_space =
        FindMusaAddressSpace(pointer->getAddressSpace());
    if (address_space == nullptr || !address_space->allowed_in_interchange) {
      return Rejected(
          request, "address-space",
          absl::StrCat("interchange type uses forbidden address space ",
                       pointer->getAddressSpace()));
    }
    return absl::OkStatus();
  }

  if (const auto* function = llvm::dyn_cast<llvm::FunctionType>(type)) {
    if (function->isVarArg()) {
      return Rejected(request, "variadic-function",
                      "variadic function types are outside the active mapping");
    }
    if (absl::Status status =
            ValidateType(function->getReturnType(), request, visited);
        !status.ok()) {
      return status;
    }
    for (llvm::Type* parameter : function->params()) {
      if (absl::Status status = ValidateType(parameter, request, visited);
          !status.ok()) {
        return status;
      }
    }
  } else if (const auto* structure = llvm::dyn_cast<llvm::StructType>(type)) {
    for (llvm::Type* element : structure->elements()) {
      if (absl::Status status = ValidateType(element, request, visited);
          !status.ok()) {
        return status;
      }
    }
  } else if (const auto* array = llvm::dyn_cast<llvm::ArrayType>(type)) {
    return ValidateType(array->getElementType(), request, visited);
  } else if (const auto* vector = llvm::dyn_cast<llvm::VectorType>(type)) {
    if (vector->getElementCount().isScalable()) {
      return Rejected(request, "scalable-vector",
                      "scalable vectors are outside the active mapping");
    }
    return ValidateType(vector->getElementType(), request, visited);
  }
  return absl::OkStatus();
}

absl::Status ValidateValueType(const llvm::Value& value,
                               const MusaBridgeCompileRequest& request) {
  llvm::SmallPtrSet<const llvm::Type*, 16> visited;
  return ValidateType(value.getType(), request, visited);
}

absl::Status ValidateConstantTree(
    const llvm::Constant& constant, const MusaBridgeCompileRequest& request,
    llvm::SmallPtrSetImpl<const llvm::Constant*>& visited) {
  if (!visited.insert(&constant).second) return absl::OkStatus();
  if (absl::Status status = ValidateValueType(constant, request);
      !status.ok()) {
    return status;
  }
  if (const auto* gep = llvm::dyn_cast<llvm::GEPOperator>(&constant)) {
    llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
    if (absl::Status status =
            ValidateType(gep->getSourceElementType(), request, visited_types);
        !status.ok()) {
      return status;
    }
  }
  if (llvm::isa<llvm::GlobalValue>(constant)) return absl::OkStatus();
  for (const llvm::Use& operand : constant.operands()) {
    if (const auto* child = llvm::dyn_cast<llvm::Constant>(operand.get())) {
      if (absl::Status status = ValidateConstantTree(*child, request, visited);
          !status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

bool HasForbiddenTargetPrefix(llvm::StringRef name) {
  return name.startswith("llvm.musa.") || name.startswith("llvm.nvvm.") ||
         name.startswith("llvm.amdgcn.") || name.startswith("llvm.r600.") ||
         name.startswith("__ockl_") || name.startswith("__ocml_");
}

bool IsNameOrOverload(llvm::StringRef name, llvm::StringRef base) {
  return name == base || (name.size() > base.size() && name.startswith(base) &&
                          name[base.size()] == '.');
}

bool IsAllowedGenericIntrinsic(llvm::StringRef name) {
  // Name matching rather than enum matching keeps this allowlist explicit
  // across the current-LLVM/vendor-LLVM version boundary. isIntrinsic() and
  // the canonical attribute check below still require LLVM 14 to recognize
  // the exact overload.
  static constexpr llvm::StringLiteral kAllowed[] = {
      "llvm.abs",
      "llvm.assume",
      "llvm.bitreverse",
      "llvm.bswap",
      "llvm.canonicalize",
      "llvm.ceil",
      "llvm.copysign",
      "llvm.cos",
      "llvm.ctlz",
      "llvm.ctpop",
      "llvm.cttz",
      "llvm.exp",
      "llvm.exp2",
      "llvm.expect",
      "llvm.fabs",
      "llvm.floor",
      "llvm.fma",
      "llvm.fshl",
      "llvm.fshr",
      "llvm.lifetime.end",
      "llvm.lifetime.start",
      "llvm.log",
      "llvm.log10",
      "llvm.log2",
      "llvm.memcpy",
      "llvm.memmove",
      "llvm.memset",
      "llvm.nearbyint",
      "llvm.pow",
      "llvm.powi",
      "llvm.rint",
      "llvm.round",
      "llvm.roundeven",
      "llvm.sadd.with.overflow",
      "llvm.sin",
      "llvm.smax",
      "llvm.smin",
      "llvm.sqrt",
      "llvm.ssub.with.overflow",
      "llvm.trunc",
      "llvm.uadd.with.overflow",
      "llvm.umax",
      "llvm.umin",
      "llvm.usub.with.overflow",
  };
  return std::any_of(
      std::begin(kAllowed), std::end(kAllowed),
      [&](llvm::StringRef base) { return IsNameOrOverload(name, base); });
}

bool HasExactlyMemoryEffects(const llvm::Function& function,
                             MusaMemoryEffects effects) {
  const bool read_none = function.hasFnAttribute(llvm::Attribute::ReadNone);
  const bool read_only = function.hasFnAttribute(llvm::Attribute::ReadOnly);
  const bool write_only = function.hasFnAttribute(llvm::Attribute::WriteOnly);
  const bool arg_only = function.hasFnAttribute(llvm::Attribute::ArgMemOnly);
  const bool inaccessible_only =
      function.hasFnAttribute(llvm::Attribute::InaccessibleMemOnly);
  const bool inaccessible_or_arg =
      function.hasFnAttribute(llvm::Attribute::InaccessibleMemOrArgMemOnly);

  switch (effects) {
    case MusaMemoryEffects::kNone:
      return read_none && !read_only && !write_only && !arg_only &&
             !inaccessible_only && !inaccessible_or_arg;
    case MusaMemoryEffects::kReadWrite:
      return !read_none && !read_only && !write_only && !arg_only &&
             !inaccessible_only && !inaccessible_or_arg;
    case MusaMemoryEffects::kInaccessibleRead:
      return !read_none && read_only && !write_only && !arg_only &&
             inaccessible_only && !inaccessible_or_arg;
    case MusaMemoryEffects::kInaccessibleReadWrite:
      return !read_none && !read_only && !write_only && !arg_only &&
             inaccessible_only && !inaccessible_or_arg;
  }
  return false;
}

bool HasExpectedSignature(const llvm::Function& function,
                          MusaShimSignature signature) {
  const llvm::FunctionType* type = function.getFunctionType();
  if (type->isVarArg()) return false;
  switch (signature) {
    case MusaShimSignature::kVoidVoid:
      return type->params().empty() && type->getReturnType()->isVoidTy();
    case MusaShimSignature::kI32Void:
      return type->params().empty() && type->getReturnType()->isIntegerTy(32);
    case MusaShimSignature::kI64Void:
      return type->params().empty() && type->getReturnType()->isIntegerTy(64);
    case MusaShimSignature::kI32I32I32:
      return type->getReturnType()->isIntegerTy(32) &&
             type->params().size() == 2 &&
             type->getParamType(0)->isIntegerTy(32) &&
             type->getParamType(1)->isIntegerTy(32);
  }
  return false;
}

bool HasExpectedVendorShuffleSignature(const llvm::Function& function) {
  const llvm::FunctionType* type = function.getFunctionType();
  return !type->isVarArg() && type->getReturnType()->isIntegerTy(32) &&
         type->getNumParams() == 6 && type->getParamType(0)->isIntegerTy(32) &&
         type->getParamType(1)->isIntegerTy(32) &&
         type->getParamType(2)->isIntegerTy(32) &&
         type->getParamType(3)->isIntegerTy(32) &&
         type->getParamType(4)->isPointerTy() &&
         type->getParamType(4)->getPointerAddressSpace() == 5 &&
         type->getParamType(5)->isPointerTy() &&
         type->getParamType(5)->getPointerAddressSpace() == 3;
}

absl::Status ValidateShimAttributes(const llvm::Function& function,
                                    const MusaShimSpec& spec,
                                    const MusaBridgeCompileRequest& request) {
  const llvm::AttributeList& attributes = function.getAttributes();
  if (attributes.hasRetAttrs()) {
    return Rejected(
        request, "shim-attributes",
        absl::StrCat("shim ", spec.xla_symbol, " carries return attributes"));
  }
  for (unsigned index = 0; index < function.arg_size(); ++index) {
    if (attributes.hasParamAttrs(index)) {
      return Rejected(request, "shim-attributes",
                      absl::StrCat("shim ", spec.xla_symbol,
                                   " carries parameter attributes"));
    }
  }
  for (llvm::Attribute attribute : attributes.getFnAttrs()) {
    if (attribute.isStringAttribute()) {
      return Rejected(request, "shim-attributes",
                      absl::StrCat("shim ", spec.xla_symbol,
                                   " carries a string attribute"));
    }
    switch (attribute.getKindAsEnum()) {
      case llvm::Attribute::ArgMemOnly:
      case llvm::Attribute::Convergent:
      case llvm::Attribute::InaccessibleMemOnly:
      case llvm::Attribute::InaccessibleMemOrArgMemOnly:
      case llvm::Attribute::NoUnwind:
      case llvm::Attribute::ReadNone:
      case llvm::Attribute::ReadOnly:
      case llvm::Attribute::WillReturn:
      case llvm::Attribute::WriteOnly:
        break;
      default:
        return Rejected(request, "shim-attributes",
                        absl::StrCat("shim ", spec.xla_symbol,
                                     " carries an unversioned attribute"));
    }
  }

  const bool require_no_unwind = (spec.required_attributes & kNoUnwind) != 0;
  const bool require_will_return =
      (spec.required_attributes & kWillReturn) != 0;
  if (function.hasFnAttribute(llvm::Attribute::NoUnwind) != require_no_unwind ||
      function.hasFnAttribute(llvm::Attribute::WillReturn) !=
          require_will_return ||
      function.hasFnAttribute(llvm::Attribute::Convergent) != spec.convergent ||
      !HasExactlyMemoryEffects(function, spec.memory_effects)) {
    return Rejected(
        request, "shim-semantics",
        absl::StrCat("shim ", spec.xla_symbol,
                     " does not match the versioned attribute contract"));
  }
  return absl::OkStatus();
}

absl::Status ValidateShimDeclaration(const llvm::Function& function,
                                     const MusaShimSpec& spec,
                                     const MusaBridgeCompileRequest& request) {
  if (!function.isDeclaration() ||
      function.getLinkage() != llvm::GlobalValue::ExternalLinkage) {
    return Rejected(request, "shim-definition",
                    absl::StrCat("shim ", spec.xla_symbol,
                                 " must be an external declaration"));
  }
  if (!HasExpectedSignature(function, spec.signature)) {
    return Rejected(
        request, "shim-signature",
        absl::StrCat("shim ", spec.xla_symbol, " has the wrong signature"));
  }
  if (function.getCallingConv() != llvm::CallingConv::C) {
    return Rejected(request, "calling-convention",
                    absl::StrCat("shim ", spec.xla_symbol,
                                 " must use the C calling convention"));
  }
  if (absl::Status status = ValidateShimAttributes(function, spec, request);
      !status.ok()) {
    return status;
  }
  for (const llvm::User* user : function.users()) {
    const auto* call = llvm::dyn_cast<llvm::CallBase>(user);
    if (call == nullptr || call->getCalledFunction() != &function) {
      return Rejected(request, "shim-address-taken",
                      absl::StrCat("shim ", spec.xla_symbol,
                                   " is not used only by direct calls"));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateFunctionObjectState(
    const llvm::Function& function, const MusaBridgeCompileRequest& request,
    bool translated_kernel = false) {
  const bool visibility_ok = translated_kernel
                                 ? function.hasProtectedVisibility()
                                 : function.hasDefaultVisibility();
  if (!visibility_ok ||
      function.getDLLStorageClass() != llvm::GlobalValue::DefaultStorageClass ||
      function.hasAtLeastLocalUnnamedAddr() || function.hasSection() ||
      function.getSectionPrefix().hasValue() || function.hasComdat() ||
      function.hasPartition() || function.hasGC() ||
      function.hasPersonalityFn() || function.hasPrefixData() ||
      function.hasPrologueData()) {
    return Rejected(
        request, "function-object-state",
        absl::StrCat("function ", SanitizedSymbol(function.getName()),
                     " carries unsupported linker or ABI state"));
  }
  return absl::OkStatus();
}

absl::Status ValidateGlobalObjectState(
    const llvm::GlobalVariable& global,
    const MusaBridgeCompileRequest& request) {
  const bool visibility_ok =
      global.hasDefaultVisibility() ||
      (!global.hasLocalLinkage() && global.hasProtectedVisibility());
  if (!visibility_ok ||
      global.getDLLStorageClass() != llvm::GlobalValue::DefaultStorageClass ||
      global.hasAtLeastLocalUnnamedAddr() || global.hasSection() ||
      global.hasComdat() || global.hasPartition() ||
      global.isExternallyInitialized()) {
    return Rejected(request, "global-object-state",
                    absl::StrCat("global ", SanitizedSymbol(global.getName()),
                                 " carries unsupported linker or ABI state"));
  }
  return absl::OkStatus();
}

absl::Status ValidateGlobals(const llvm::Module& module,
                             const MusaBridgeCompileRequest& request) {
  absl::flat_hash_map<std::string, const MusaBridgeExportedGlobal*> expected;
  for (const MusaBridgeExportedGlobal& global : request.exported_globals()) {
    expected.emplace(global.name(), &global);
  }
  absl::flat_hash_set<std::string> found;
  const llvm::DataLayout& data_layout = module.getDataLayout();

  for (const llvm::GlobalVariable& global : module.globals()) {
    if (absl::Status status = ValidateValueType(global, request);
        !status.ok()) {
      return status;
    }
    if (absl::Status status = ValidateGlobalObjectState(global, request);
        !status.ok()) {
      return status;
    }
    llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
    if (absl::Status status =
            ValidateType(global.getValueType(), request, visited_types);
        !status.ok()) {
      return status;
    }
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4> metadata;
    global.getAllMetadata(metadata);
    if (!metadata.empty()) {
      return Rejected(request, "global-metadata",
                      absl::StrCat("global ", SanitizedSymbol(global.getName()),
                                   " carries metadata"));
    }
    if (global.isThreadLocal() || global.isDeclaration()) {
      return Rejected(request, "global-definition",
                      absl::StrCat("global ", SanitizedSymbol(global.getName()),
                                   " must be a non-thread-local definition"));
    }
    if (global.hasInitializer()) {
      llvm::SmallPtrSet<const llvm::Constant*, 16> visited_constants;
      if (absl::Status status = ValidateConstantTree(
              *global.getInitializer(), request, visited_constants);
          !status.ok()) {
        return status;
      }
    }

    auto expected_it = expected.find(global.getName().str());
    if (global.hasLocalLinkage()) {
      if (expected_it != expected.end()) {
        return Rejected(
            request, "exported-globals",
            absl::StrCat("listed global ", SanitizedSymbol(global.getName()),
                         " has local linkage"));
      }
      continue;
    }
    if (global.getLinkage() != llvm::GlobalValue::ExternalLinkage ||
        expected_it == expected.end()) {
      return Rejected(request, "exported-globals",
                      absl::StrCat("externally visible global ",
                                   SanitizedSymbol(global.getName()),
                                   " is not exactly described by the request"));
    }

    const MusaBridgeExportedGlobal& spec = *expected_it->second;
    const llvm::TypeSize allocation_size =
        data_layout.getTypeAllocSize(global.getValueType());
    if (allocation_size.isScalable()) {
      return Rejected(request, "exported-globals",
                      absl::StrCat("global ", spec.name(),
                                   " has scalable allocation size"));
    }
    const uint64_t alignment =
        global.getAlign()
            ? global.getAlign()->value()
            : data_layout.getABITypeAlign(global.getValueType()).value();
    const bool expected_constant =
        spec.kind() == MUSA_BRIDGE_GLOBAL_KIND_CONSTANT;
    if (global.getAddressSpace() != spec.address_space() ||
        allocation_size.getFixedSize() != spec.size_bytes() ||
        alignment != spec.alignment_bytes() ||
        global.isConstant() != expected_constant) {
      return Rejected(request, "exported-globals",
                      absl::StrCat("global ", spec.name(),
                                   " does not match its typed request record"));
    }
    found.insert(spec.name());
  }

  if (found.size() != expected.size()) {
    return Rejected(request, "exported-globals",
                    "one or more requested globals are missing");
  }
  return absl::OkStatus();
}

absl::Status ValidateInstruction(const llvm::Instruction& instruction,
                                 const MusaBridgeCompileRequest& request,
                                 absl::string_view function_name) {
  if (absl::Status status = ValidateValueType(instruction, request);
      !status.ok()) {
    return status;
  }
  if (const auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(&instruction)) {
    llvm::SmallPtrSet<const llvm::Type*, 16> visited;
    if (absl::Status status =
            ValidateType(alloca->getAllocatedType(), request, visited);
        !status.ok()) {
      return status;
    }
  }
  if (const auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction)) {
    llvm::SmallPtrSet<const llvm::Type*, 16> visited;
    if (absl::Status status =
            ValidateType(gep->getSourceElementType(), request, visited);
        !status.ok()) {
      return status;
    }
  }
  if (const auto* fp = llvm::dyn_cast<llvm::FPMathOperator>(&instruction);
      fp != nullptr && fp->getFastMathFlags().any() &&
      !HasOnlyNoSignedZeros(fp->getFastMathFlags())) {
    return Rejected(request, "fast-math-flags",
                    absl::StrCat("function ", function_name,
                                 " contains fast-math flags outside the "
                                 "active mapping's nsz-only contract"));
  }
  for (const llvm::Use& operand : instruction.operands()) {
    if (absl::Status status = ValidateValueType(*operand.get(), request);
        !status.ok()) {
      return status;
    }
    if (const auto* constant = llvm::dyn_cast<llvm::Constant>(operand.get())) {
      llvm::SmallPtrSet<const llvm::Constant*, 16> visited;
      if (absl::Status status =
              ValidateConstantTree(*constant, request, visited);
          !status.ok()) {
        return status;
      }
    }
  }

  llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4> metadata;
  instruction.getAllMetadata(metadata);
  if (!metadata.empty() || instruction.getDebugLoc()) {
    return Rejected(request, "instruction-metadata",
                    absl::StrCat("function ", function_name,
                                 " carries instruction metadata"));
  }
  if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
    if (!call->getAttributes().isEmpty()) {
      return Rejected(request, "call-site-attributes",
                      absl::StrCat("function ", function_name,
                                   " has a call with unversioned attributes"));
    }
    if (call->isInlineAsm()) {
      return Rejected(request, "inline-assembly",
                      absl::StrCat("function ", function_name,
                                   " contains inline assembly"));
    }
    if (call->hasOperandBundles()) {
      return Rejected(request, "operand-bundle",
                      absl::StrCat("function ", function_name,
                                   " contains an operand bundle"));
    }
    if (call->getCalledFunction() == nullptr) {
      return Rejected(request, "indirect-call",
                      absl::StrCat("function ", function_name,
                                   " contains an indirect call"));
    }
    if (call->getCallingConv() != llvm::CallingConv::C) {
      return Rejected(request, "calling-convention",
                      absl::StrCat("call in function ", function_name,
                                   " does not use the C convention"));
    }
  }

  if (const auto* cmpxchg =
          llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&instruction)) {
    if (!IsQualifiedMusaAtomicCmpXchg(*cmpxchg)) {
      return Rejected(request, "atomics",
                      "cmpxchg is outside the exact mapping-v3 contract");
    }
    return absl::OkStatus();
  }
  const auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction);
  const auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction);
  if ((load != nullptr && load->isAtomic()) ||
      (store != nullptr && store->isAtomic()) ||
      llvm::isa<llvm::AtomicRMWInst, llvm::FenceInst>(instruction)) {
    return Rejected(request, "atomics",
                    "atomic instruction is outside the exact mapping-v3 "
                    "contract");
  }
  return absl::OkStatus();
}

absl::Status ValidateFunctions(const llvm::Module& module,
                               const MusaBridgeCompileRequest& request) {
  absl::flat_hash_set<std::string> expected_kernels;
  for (const std::string& name : request.kernel_entry_names()) {
    expected_kernels.insert(name);
  }
  absl::flat_hash_set<std::string> found_kernels;

  for (const llvm::Function& function : module.functions()) {
    const std::string raw_function_name = function.getName().str();
    const std::string function_name = SanitizedSymbol(function.getName());
    if (function.getAddressSpace() != 0) {
      return Rejected(request, "function-address-space",
                      absl::StrCat("function ", function_name,
                                   " is not in address space 0"));
    }
    if (absl::Status status = ValidateValueType(function, request);
        !status.ok()) {
      return status;
    }
    if (absl::Status status = ValidateFunctionObjectState(function, request);
        !status.ok()) {
      return status;
    }
    llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
    if (absl::Status status =
            ValidateType(function.getFunctionType(), request, visited_types);
        !status.ok()) {
      return status;
    }
    if (function.getCallingConv() != llvm::CallingConv::C) {
      return Rejected(
          request, "calling-convention",
          absl::StrCat("function ", function_name,
                       " crosses the bridge with a non-C convention"));
    }
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4> metadata;
    function.getAllMetadata(metadata);
    if (!metadata.empty()) {
      return Rejected(
          request, "function-metadata",
          absl::StrCat("function ", function_name, " carries metadata"));
    }

    if (HasForbiddenTargetPrefix(function.getName())) {
      return Rejected(request, "foreign-target-construct",
                      absl::StrCat("raw target function ", function_name,
                                   " cannot cross the bridge"));
    }
    if (function.getName().startswith("__xla_musa_")) {
      const MusaShimSpec* spec = FindMusaShim(raw_function_name);
      if (spec == nullptr) {
        return Rejected(request, "unknown-shim",
                        absl::StrCat("unknown shim ", function_name));
      }
      if (absl::Status status =
              ValidateShimDeclaration(function, *spec, request);
          !status.ok()) {
        return status;
      }
      continue;
    }
    if (function.isIntrinsic()) {
      if (!IsAllowedGenericIntrinsic(function.getName())) {
        return Rejected(request, "llvm-intrinsic",
                        absl::StrCat("intrinsic ", function_name,
                                     " is outside the active mapping"));
      }
      const llvm::AttributeList canonical = llvm::Intrinsic::getAttributes(
          function.getContext(), function.getIntrinsicID());
      if (function.getAttributes() != canonical ||
          function.getLinkage() != llvm::GlobalValue::ExternalLinkage) {
        return Rejected(
            request, "intrinsic-contract",
            absl::StrCat("intrinsic ", function_name,
                         " is not its canonical LLVM 14 declaration"));
      }
      continue;
    }
    if (function.isDeclaration()) {
      return Rejected(request, "unresolved-call",
                      absl::StrCat("external function ", function_name,
                                   " is not a mapped shim or intrinsic"));
    }
    if (!function.getAttributes().isEmpty()) {
      return Rejected(request, "function-attributes",
                      absl::StrCat("defined function ", function_name,
                                   " carries unversioned attributes"));
    }

    const bool listed_kernel = expected_kernels.contains(raw_function_name);
    if (listed_kernel) {
      if (function.getLinkage() != llvm::GlobalValue::ExternalLinkage ||
          !function.getReturnType()->isVoidTy() || function.isVarArg()) {
        return Rejected(request, "kernel-list",
                        absl::StrCat("kernel ", function_name,
                                     " has an invalid entry ABI"));
      }
      found_kernels.insert(raw_function_name);
    } else if (!function.hasLocalLinkage()) {
      return Rejected(
          request, "kernel-list",
          absl::StrCat("externally visible function ", function_name,
                       " is absent from the kernel list"));
    }

    for (const llvm::BasicBlock& block : function) {
      for (const llvm::Instruction& instruction : block) {
        if (absl::Status status =
                ValidateInstruction(instruction, request, function_name);
            !status.ok()) {
          return status;
        }
      }
    }
  }

  if (found_kernels.size() != expected_kernels.size()) {
    return Rejected(request, "kernel-list",
                    "one or more requested kernels are missing");
  }
  return absl::OkStatus();
}

absl::Status ValidateExportSet(const MusaBridgeCompileRequest& request) {
  absl::flat_hash_set<std::string> expected;
  for (const std::string& kernel : request.kernel_entry_names()) {
    expected.insert(kernel);
  }
  for (const MusaBridgeExportedGlobal& global : request.exported_globals()) {
    expected.insert(global.name());
  }
  if (expected.size() != request.exported_symbol_names_size()) {
    return Rejected(request, "exported-symbols",
                    "exports must be exactly kernels plus typed globals");
  }
  for (const std::string& name : request.exported_symbol_names()) {
    if (!expected.contains(name)) {
      return Rejected(request, "exported-symbols",
                      "exports must be exactly kernels plus typed globals");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateInterchangeModule(
    const llvm::Module& module, const MusaBridgeCompileRequest& request) {
  if (module.getTargetTriple() != kMusaTargetTriple) {
    return Rejected(request, "target-triple",
                    "module triple does not match the frozen MUSA ABI");
  }
  if (module.getDataLayoutStr() != kMusaDataLayout) {
    return Rejected(request, "data-layout",
                    "module data layout does not match the frozen MUSA ABI");
  }
  if (!module.getModuleInlineAsm().empty()) {
    return Rejected(request, "inline-assembly",
                    "module-level inline assembly is forbidden");
  }
  if (!module.alias_empty()) {
    return Rejected(request, "global-alias",
                    "global aliases are outside the active mapping");
  }
  if (!module.ifunc_empty()) {
    return Rejected(request, "global-ifunc",
                    "indirect functions are outside the active mapping");
  }
  if (!module.getComdatSymbolTable().empty()) {
    return Rejected(request, "module-comdat",
                    "COMDAT definitions are outside the active mapping");
  }
  if (module.named_metadata_begin() != module.named_metadata_end()) {
    return Rejected(request, "named-metadata",
                    "named metadata may only be installed by the bridge");
  }
  for (llvm::StructType* type : module.getIdentifiedStructTypes()) {
    llvm::SmallPtrSet<const llvm::Type*, 16> visited;
    if (absl::Status status = ValidateType(type, request, visited);
        !status.ok()) {
      return status;
    }
  }
  if (absl::Status status = ValidateGlobals(module, request); !status.ok()) {
    return status;
  }
  return ValidateFunctions(module, request);
}

absl::StatusOr<llvm::Function*> GetVendorIntrinsic(
    llvm::Module& module, const MusaShimSpec& spec,
    const MusaBridgeCompileRequest& request) {
  const llvm::Intrinsic::ID id =
      llvm::Function::lookupIntrinsicID(spec.vendor_intrinsic);
  if (id == llvm::Intrinsic::not_intrinsic) {
    return ToolchainMismatch(
        request, absl::StrCat("SDK does not register ", spec.vendor_intrinsic));
  }
  llvm::SmallVector<llvm::Type*, 1> overload_types;
  if (llvm::Intrinsic::isOverloaded(id)) {
    if (spec.signature != MusaShimSignature::kI32I32I32) {
      return ToolchainMismatch(
          request,
          absl::StrCat("mapped intrinsic has an unqualified overload: ",
                       spec.vendor_intrinsic));
    }
    overload_types.push_back(llvm::Type::getInt32Ty(module.getContext()));
  }
  llvm::Function* function =
      llvm::Intrinsic::getDeclaration(&module, id, overload_types);
  const llvm::StringRef vendor_name(spec.vendor_intrinsic.data(),
                                    spec.vendor_intrinsic.size());
  if (spec.id == MusaShimId::kLogicalShuffleI32) {
    if (function == nullptr || function->getName() != vendor_name ||
        function->getIntrinsicID() != id ||
        !HasExpectedVendorShuffleSignature(*function) ||
        function->getCallingConv() != llvm::CallingConv::C ||
        !function->hasFnAttribute(llvm::Attribute::NoUnwind) ||
        !function->hasFnAttribute(llvm::Attribute::Convergent) ||
        !function->hasFnAttribute(llvm::Attribute::WriteOnly) ||
        function->hasFnAttribute(llvm::Attribute::ReadNone) ||
        function->hasFnAttribute(llvm::Attribute::ReadOnly) ||
        function->hasFnAttribute(llvm::Attribute::ArgMemOnly) ||
        function->hasFnAttribute(llvm::Attribute::InaccessibleMemOnly) ||
        function->hasFnAttribute(
            llvm::Attribute::InaccessibleMemOrArgMemOnly)) {
      return ToolchainMismatch(
          request,
          "SDK fake-shuffle declaration disagrees with the active "
          "logical-shuffle contract introduced in mapping v2");
    }
    return function;
  }
  if (function == nullptr || function->getName() != vendor_name ||
      function->getIntrinsicID() != id ||
      !HasExpectedSignature(*function, spec.signature) ||
      function->getCallingConv() != llvm::CallingConv::C ||
      function->hasFnAttribute(llvm::Attribute::NoUnwind) !=
          ((spec.required_attributes & kNoUnwind) != 0) ||
      function->hasFnAttribute(llvm::Attribute::WillReturn) !=
          ((spec.required_attributes & kWillReturn) != 0) ||
      function->hasFnAttribute(llvm::Attribute::Convergent) !=
          spec.convergent ||
      !HasExactlyMemoryEffects(*function, spec.memory_effects)) {
    return ToolchainMismatch(
        request, absl::StrCat("SDK declaration disagrees with mapping for ",
                              spec.vendor_intrinsic));
  }
  return function;
}

absl::StatusOr<uint32_t> TranslateShimCalls(
    llvm::Module& module, const MusaBridgeCompileRequest& request) {
  struct ShimDeclaration {
    llvm::Function* function;
    const MusaShimSpec* spec;
  };
  std::vector<ShimDeclaration> shims;
  for (llvm::Function& function : module.functions()) {
    if (!function.getName().startswith("__xla_musa_")) continue;
    const MusaShimSpec* spec = FindMusaShim(function.getName().str());
    if (spec == nullptr) {
      return Rejected(request, "unknown-shim",
                      "an unknown shim survived pretranslation validation");
    }
    shims.push_back({&function, spec});
  }

  llvm::Constant* shuffle_scratch_pointer = nullptr;

  uint32_t translated_calls = 0;
  for (const ShimDeclaration& shim : shims) {
    absl::StatusOr<llvm::Function*> vendor =
        GetVendorIntrinsic(module, *shim.spec, request);
    if (!vendor.ok()) return vendor.status();

    llvm::SmallVector<llvm::CallBase*, 8> calls;
    for (llvm::User* user : shim.function->users()) {
      auto* call = llvm::dyn_cast<llvm::CallBase>(user);
      if (call == nullptr || call->getCalledFunction() != shim.function) {
        return Rejected(request, "shim-address-taken",
                        "a shim acquired a non-call use during translation");
      }
      calls.push_back(call);
    }
    for (llvm::CallBase* call : calls) {
      if (shim.spec->id == MusaShimId::kLogicalShuffleI32) {
        auto* call_instruction = llvm::dyn_cast<llvm::CallInst>(call);
        if (call_instruction == nullptr) {
          return Rejected(request, "shuffle-adapter",
                          "logical shuffle must use an ordinary call");
        }
        if (shuffle_scratch_pointer == nullptr) {
          llvm::Type* i32 = llvm::Type::getInt32Ty(module.getContext());
          llvm::ArrayType* scratch_type = llvm::ArrayType::get(i32, 128);
          auto* scratch = new llvm::GlobalVariable(
              module, scratch_type, /*isConstant=*/false,
              llvm::GlobalValue::PrivateLinkage,
              llvm::UndefValue::get(scratch_type),
              "__musa_xla_shuffle_scratch_v2", /*InsertBefore=*/nullptr,
              llvm::GlobalValue::NotThreadLocal, /*AddressSpace=*/3);
          scratch->setAlignment(llvm::Align(4));
          llvm::Constant* zero = llvm::ConstantInt::get(i32, 0);
          llvm::Constant* indices[] = {zero, zero};
          shuffle_scratch_pointer =
              llvm::ConstantExpr::getInBoundsGetElementPtr(scratch_type,
                                                           scratch, indices);
        }

        llvm::IRBuilder<> builder(call_instruction);
        llvm::Type* i32 = llvm::Type::getInt32Ty(module.getContext());
        llvm::Value* arguments[] = {
            llvm::ConstantInt::getSigned(i32, -1),
            call_instruction->getArgOperand(0),
            call_instruction->getArgOperand(1),
            llvm::ConstantInt::get(i32, 32),
            llvm::ConstantPointerNull::get(llvm::PointerType::get(
                llvm::Type::getInt8Ty(module.getContext()), 5)),
            shuffle_scratch_pointer};
        llvm::CallInst* translated =
            builder.CreateCall(*vendor, arguments, "musa_shuffle");
        translated->setCallingConv((*vendor)->getCallingConv());
        call_instruction->replaceAllUsesWith(translated);
        call_instruction->eraseFromParent();
      } else {
        call->setCalledFunction(*vendor);
      }
      ++translated_calls;
    }
    if (!shim.function->use_empty()) {
      return absl::InternalError(
          "MUSA bridge failed to replace every validated shim use");
    }
    shim.function->eraseFromParent();
  }
  return translated_calls;
}

absl::Status LowerGenericExportedGlobalAddressSpaces(
    llvm::Module& module, const MusaBridgeCompileRequest& request) {
  for (const MusaBridgeExportedGlobal& global_spec :
       request.exported_globals()) {
    llvm::GlobalVariable* global = module.getGlobalVariable(global_spec.name());
    if (global == nullptr) {
      return absl::InternalError(
          "MUSA bridge lost a validated global before target lowering");
    }
    if (global->getAddressSpace() != 0) continue;
    if (!global->use_empty()) {
      return Rejected(request, "global-address-space",
                      absl::StrCat("generic exported global ",
                                   SanitizedSymbol(global->getName()),
                                   " has in-module uses"));
    }

    const unsigned native_address_space =
        global_spec.kind() == MUSA_BRIDGE_GLOBAL_KIND_CONSTANT ? 2 : 1;
    auto* lowered = new llvm::GlobalVariable(
        module, global->getValueType(), global->isConstant(),
        global->getLinkage(), global->getInitializer(), /*Name=*/"",
        /*InsertBefore=*/nullptr, global->getThreadLocalMode(),
        native_address_space, global->isExternallyInitialized());
    lowered->copyAttributesFrom(global);
    lowered->takeName(global);
    global->eraseFromParent();
  }
  return absl::OkStatus();
}

absl::Status InstallExportAbi(llvm::Module& module,
                              const MusaBridgeCompileRequest& request) {
  llvm::NamedMDNode* annotations =
      module.getOrInsertNamedMetadata("musa.annotations");
  llvm::Type* i32 = llvm::Type::getInt32Ty(module.getContext());
  for (const std::string& kernel_name : request.kernel_entry_names()) {
    llvm::Function* kernel = module.getFunction(kernel_name);
    if (kernel == nullptr || kernel->isDeclaration()) {
      return absl::InternalError(
          "MUSA bridge lost a validated kernel during translation");
    }
    kernel->setCallingConv(llvm::CallingConv::MTGPU_KERNEL);
    kernel->setVisibility(llvm::GlobalValue::ProtectedVisibility);
    llvm::Metadata* operands[] = {
        llvm::ValueAsMetadata::get(kernel),
        llvm::MDString::get(module.getContext(), "kernel"),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
    };
    annotations->addOperand(llvm::MDNode::get(module.getContext(), operands));
  }
  llvm::BasicBlock* globals_anchor_entry = nullptr;
  if (request.kernel_entry_names().empty()) {
    if (module.getNamedValue(
            llvm::StringRef(kGlobalsOnlyAnchorKernel.data(),
                            kGlobalsOnlyAnchorKernel.size())) != nullptr) {
      return Rejected(request, "reserved-symbol",
                      "module defines the globals-only anchor symbol");
    }
    llvm::FunctionType* type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(module.getContext()), /*isVarArg=*/false);
    llvm::Function* anchor =
        llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                               llvm::StringRef(kGlobalsOnlyAnchorKernel.data(),
                                               kGlobalsOnlyAnchorKernel.size()),
                               module);
    anchor->setCallingConv(llvm::CallingConv::MTGPU_KERNEL);
    anchor->setVisibility(llvm::GlobalValue::ProtectedVisibility);
    globals_anchor_entry =
        llvm::BasicBlock::Create(module.getContext(), "entry", anchor);
    llvm::Metadata* operands[] = {
        llvm::ValueAsMetadata::get(anchor),
        llvm::MDString::get(module.getContext(), "kernel"),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
    };
    annotations->addOperand(llvm::MDNode::get(module.getContext(), operands));
  }
  for (const MusaBridgeExportedGlobal& global_spec :
       request.exported_globals()) {
    llvm::GlobalVariable* global = module.getGlobalVariable(global_spec.name());
    if (global == nullptr || global->hasLocalLinkage()) {
      return absl::InternalError(
          "MUSA bridge lost a validated global during translation");
    }
    // The driver-visible ABI uses protected exports for both functions and
    // data. Keep that target-specific linker detail on the vendor side of the
    // bridge, just like the MTGPU kernel calling convention and annotations.
    global->setVisibility(llvm::GlobalValue::ProtectedVisibility);
    if (globals_anchor_entry != nullptr) {
      llvm::IRBuilder<> builder(globals_anchor_entry);
      llvm::LoadInst* load = builder.CreateLoad(
          llvm::Type::getInt8Ty(module.getContext()), global, "global_anchor");
      load->setVolatile(true);
    }
  }
  if (globals_anchor_entry != nullptr) {
    llvm::ReturnInst::Create(module.getContext(), globals_anchor_entry);
  }
  return absl::OkStatus();
}

absl::Status ValidateKernelAnnotations(
    const llvm::Module& module, const MusaBridgeCompileRequest& request) {
  const llvm::NamedMDNode* annotations =
      module.getNamedMetadata("musa.annotations");
  const int expected_annotation_count =
      request.kernel_entry_names_size() +
      (request.kernel_entry_names().empty() ? 1 : 0);
  if (annotations == nullptr ||
      annotations->getNumOperands() != expected_annotation_count) {
    return absl::InternalError(
        "MUSA bridge installed an incomplete kernel annotation table");
  }

  const int annotations_to_validate = request.kernel_entry_names().empty()
                                          ? 1
                                          : request.kernel_entry_names_size();
  for (int i = 0; i < annotations_to_validate; ++i) {
    const llvm::MDNode* node = annotations->getOperand(i);
    if (node == nullptr || node->getNumOperands() != 3) {
      return absl::InternalError(
          "MUSA bridge installed a malformed kernel annotation");
    }
    const auto* value =
        llvm::dyn_cast<llvm::ValueAsMetadata>(node->getOperand(0));
    const auto* tag = llvm::dyn_cast<llvm::MDString>(node->getOperand(1));
    const auto* enabled =
        llvm::dyn_cast<llvm::ConstantAsMetadata>(node->getOperand(2));
    const auto* enabled_value =
        enabled == nullptr
            ? nullptr
            : llvm::dyn_cast<llvm::ConstantInt>(enabled->getValue());
    const llvm::Function* kernel = llvm::dyn_cast_or_null<llvm::Function>(
        value == nullptr ? nullptr : value->getValue());
    const absl::string_view expected_name =
        request.kernel_entry_names().empty()
            ? kGlobalsOnlyAnchorKernel
            : absl::string_view(request.kernel_entry_names(i));
    if (kernel == nullptr ||
        kernel->getName() !=
            llvm::StringRef(expected_name.data(), expected_name.size()) ||
        tag == nullptr || tag->getString() != "kernel" ||
        enabled_value == nullptr || !enabled_value->equalsInt(1)) {
      return absl::InternalError(
          "MUSA bridge installed an incorrect kernel annotation");
    }
  }
  if (request.kernel_entry_names().empty()) {
    const llvm::MDNode* node = annotations->getOperand(0);
    const auto* value =
        llvm::dyn_cast<llvm::ValueAsMetadata>(node->getOperand(0));
    const llvm::Function* kernel = llvm::dyn_cast_or_null<llvm::Function>(
        value == nullptr ? nullptr : value->getValue());
    if (kernel == nullptr ||
        kernel->getName() != llvm::StringRef(kGlobalsOnlyAnchorKernel.data(),
                                             kGlobalsOnlyAnchorKernel.size()) ||
        kernel->getCallingConv() != llvm::CallingConv::MTGPU_KERNEL ||
        !kernel->hasProtectedVisibility()) {
      return absl::InternalError(
          "MUSA bridge installed an invalid globals-only anchor kernel");
    }
    if (absl::Status status = ValidateFunctionObjectState(
            *kernel, request, /*translated_kernel=*/true);
        !status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateTranslatedModule(const llvm::Module& module,
                                      const MusaBridgeCompileRequest& request) {
  if (llvm::verifyModule(module)) {
    return absl::InternalError(
        "MUSA vendor LLVM verifier rejected the translated module");
  }
  if (std::distance(module.named_metadata_begin(),
                    module.named_metadata_end()) != 1) {
    return absl::InternalError(
        "MUSA bridge produced unexpected native named metadata");
  }
  if (absl::Status status = ValidateKernelAnnotations(module, request);
      !status.ok()) {
    return status;
  }
  for (const std::string& kernel_name : request.kernel_entry_names()) {
    const llvm::Function* kernel = module.getFunction(kernel_name);
    if (kernel == nullptr ||
        kernel->getCallingConv() != llvm::CallingConv::MTGPU_KERNEL ||
        !kernel->hasProtectedVisibility()) {
      return absl::InternalError(
          "MUSA bridge failed to install the native kernel ABI");
    }
    if (absl::Status status =
            ValidateFunctionObjectState(*kernel, request,
                                        /*translated_kernel=*/true);
        !status.ok()) {
      return status;
    }
  }
  for (const MusaBridgeExportedGlobal& global_spec :
       request.exported_globals()) {
    const llvm::GlobalVariable* global =
        module.getGlobalVariable(global_spec.name());
    if (global == nullptr || !global->hasProtectedVisibility()) {
      return absl::InternalError(
          "MUSA bridge failed to install the native global export ABI");
    }
  }
  for (const llvm::Function& function : module.functions()) {
    if (function.getName().startswith("__xla_musa_")) {
      return absl::InternalError(
          "MUSA bridge left an interchange shim in native LLVM");
    }
    if (!function.getName().startswith("llvm.musa.")) continue;
    bool mapped = false;
    for (const MusaShimSpec& spec : MusaShimSpecs()) {
      const llvm::StringRef vendor_name(spec.vendor_intrinsic.data(),
                                        spec.vendor_intrinsic.size());
      if (function.getName() == vendor_name) {
        mapped = true;
        if (function.getIntrinsicID() == llvm::Intrinsic::not_intrinsic) {
          return absl::InternalError(
              "MUSA bridge emitted an unregistered target intrinsic");
        }
        break;
      }
    }
    if (!mapped) {
      return absl::InternalError(
          "MUSA bridge emitted a target intrinsic outside the active mapping");
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> LoadedVendorLlvmSharedObjectPath() {
  Dl_info info = {};
  auto* parser = &llvm::parseAssemblyString;
  if (dladdr(reinterpret_cast<void*>(parser), &info) == 0 ||
      info.dli_fname == nullptr) {
    return absl::InternalError(
        "failed to identify the loaded vendor LLVM shared object");
  }
  char* resolved = realpath(info.dli_fname, nullptr);
  if (resolved == nullptr) {
    return absl::InternalError(
        "failed to resolve the loaded vendor LLVM shared object");
  }
  std::string path(resolved);
  std::free(resolved);
  return path;
}

absl::StatusOr<VendorLlvmModule> TranslateMusaBridgeRequestToVendorLlvm(
    const MusaBridgeCompileRequest& request) {
  if (absl::Status status = ValidateMusaBridgeCompileRequest(request);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = ValidateMusaShimTable(); !status.ok()) {
    return status;
  }
  if (absl::Status status = ValidateExportSet(request); !status.ok()) {
    return status;
  }

  llvm::LLVMContext context;
  context.enableOpaquePointers();
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
      llvm::StringRef(request.normalized_llvm().data(),
                      request.normalized_llvm().size()),
      diagnostic, context);
  if (module == nullptr) {
    return Rejected(
        request, "llvm14-parse",
        absl::StrFormat(
            "vendor LLVM 14 parser rejected the module at line %d column %d",
            diagnostic.getLineNo(), diagnostic.getColumnNo()));
  }
  if (llvm::verifyModule(*module)) {
    return Rejected(request, "llvm14-verifier",
                    "vendor LLVM 14 verifier rejected the module");
  }
  if (absl::Status status = ValidateInterchangeModule(*module, request);
      !status.ok()) {
    return status;
  }

  absl::StatusOr<uint32_t> translated_calls =
      TranslateShimCalls(*module, request);
  if (!translated_calls.ok()) return translated_calls.status();
  if (absl::Status status =
          LowerGenericExportedGlobalAddressSpaces(*module, request);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = InstallExportAbi(*module, request); !status.ok()) {
    return status;
  }
  module->setModuleIdentifier(request.module_name());
  module->setSourceFileName(request.module_name());
  if (absl::Status status = ValidateTranslatedModule(*module, request);
      !status.ok()) {
    return status;
  }

  std::string vendor_ir;
  llvm::raw_string_ostream stream(vendor_ir);
  module->print(stream, nullptr, /*ShouldPreserveUseListOrder=*/true,
                /*IsForDebug=*/false);
  stream.flush();
  return VendorLlvmModule{
      .llvm_ir = std::move(vendor_ir),
      .translated_shim_calls = *translated_calls,
      .kernel_count = static_cast<uint32_t>(request.kernel_entry_names_size()),
  };
}

}  // namespace xla::gpu::musa::bridge
