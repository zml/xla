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

#include "xla/service/gpu/musa/musa_bridge_ir_validator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
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
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace xla::gpu::musa {
namespace {

bool IsSafeToken(absl::string_view value, bool allow_dash,
                 size_t max_bytes = kMaxMusaSymbolBytes) {
  if (value.empty() || value.size() > max_bytes) return false;
  const auto valid_first = [](char c) {
    return absl::ascii_isalpha(c) || c == '_' || c == '.' || c == '$';
  };
  const auto valid_rest = [&](char c) {
    return valid_first(c) || absl::ascii_isdigit(c) || (allow_dash && c == '-');
  };
  return valid_first(value.front()) &&
         std::all_of(value.begin() + 1, value.end(), valid_rest);
}

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

std::string SanitizedSymbol(llvm::StringRef value) {
  return IsSafeToken(value.str(), /*allow_dash=*/true) ? value.str()
                                                       : "<invalid-symbol>";
}

absl::Status Rejected(const MusaBridgeIrMetadata& metadata,
                      absl::string_view capability, absl::string_view detail) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "MUSA bridge IR rejected: protocol=%d shim_abi=%d mapping=%d "
      "module=%s capability=%s: %s",
      metadata.protocol_version, metadata.shim_abi_version,
      metadata.mapping_version,
      IsSafeModuleName(metadata.module_name) ? metadata.module_name
                                             : "<invalid-module-name>",
      capability, detail));
}

absl::Status ValidateMetadata(const MusaBridgeIrMetadata& metadata) {
  if (metadata.protocol_version != kMusaBridgeProtocolVersion ||
      metadata.shim_abi_version != kMusaShimAbiVersion ||
      metadata.mapping_version != kMusaShimMappingVersion) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "MUSA bridge contract version mismatch: got protocol=%d shim_abi=%d "
        "mapping=%d; require %d/%d/%d",
        metadata.protocol_version, metadata.shim_abi_version,
        metadata.mapping_version, kMusaBridgeProtocolVersion,
        kMusaShimAbiVersion, kMusaShimMappingVersion));
  }
  if (!IsSafeModuleName(metadata.module_name)) {
    return absl::InvalidArgumentError(
        "MUSA bridge module name must be a bounded safe token");
  }
  if (metadata.architecture != kMusaTargetArchitecture) {
    return Rejected(metadata, "architecture",
                    absl::StrCat("target does not match required architecture ",
                                 kMusaTargetArchitecture));
  }
  if (metadata.kernel_entry_names.empty() ||
      metadata.kernel_entry_names.size() > kMaxMusaKernelEntries) {
    return Rejected(metadata, "kernel-list",
                    "kernel list must be nonempty and bounded");
  }
  absl::string_view previous;
  for (const std::string& name : metadata.kernel_entry_names) {
    if (!IsSafeToken(name, /*allow_dash=*/true)) {
      return Rejected(metadata, "kernel-list",
                      "kernel entry is not a bounded LLVM symbol");
    }
    if (!previous.empty() && previous >= name) {
      return Rejected(metadata, "kernel-list",
                      "kernel entries must be sorted and unique");
    }
    previous = name;
  }
  if (metadata.exported_globals.size() > kMaxMusaExportedGlobals) {
    return Rejected(metadata, "exported-globals",
                    "exported-global list exceeds the protocol limit");
  }
  previous = {};
  for (const MusaExportedGlobal& global : metadata.exported_globals) {
    if (!IsSafeToken(global.name, /*allow_dash=*/true) ||
        (!previous.empty() && previous >= global.name)) {
      return Rejected(metadata, "exported-globals",
                      "exported globals must be sorted, unique LLVM symbols");
    }
    const MusaAddressSpaceSpec* address_space =
        FindMusaAddressSpace(global.address_space);
    if (address_space == nullptr || !address_space->allowed_in_interchange) {
      return Rejected(
          metadata, "address-space",
          absl::StrCat("global ", global.name, " uses forbidden address space ",
                       global.address_space));
    }
    if (global.size == 0 || global.alignment == 0 ||
        (global.alignment & (global.alignment - 1)) != 0) {
      return Rejected(
          metadata, "exported-globals",
          absl::StrCat("global ", global.name, " has invalid size/alignment"));
    }
    previous = global.name;
  }
  return ValidateMusaShimTable();
}

absl::Status ValidateType(const llvm::Type* type,
                          const MusaBridgeIrMetadata& metadata,
                          llvm::SmallPtrSetImpl<const llvm::Type*>& visited) {
  if (!visited.insert(type).second) return absl::OkStatus();
  if (const auto* pointer = llvm::dyn_cast<llvm::PointerType>(type)) {
    uint32_t number = pointer->getAddressSpace();
    const MusaAddressSpaceSpec* spec = FindMusaAddressSpace(number);
    if (spec == nullptr || !spec->allowed_in_interchange) {
      return Rejected(metadata, "address-space",
                      absl::StrCat("interchange type uses forbidden address "
                                   "space ",
                                   number));
    }
    return absl::OkStatus();
  }
  if (const auto* function = llvm::dyn_cast<llvm::FunctionType>(type)) {
    if (function->isVarArg()) {
      return Rejected(metadata, "variadic-function",
                      "variadic function types are not supported");
    }
    if (absl::Status status =
            ValidateType(function->getReturnType(), metadata, visited);
        !status.ok()) {
      return status;
    }
    for (llvm::Type* parameter : function->params()) {
      if (absl::Status status = ValidateType(parameter, metadata, visited);
          !status.ok()) {
        return status;
      }
    }
  } else if (const auto* structure = llvm::dyn_cast<llvm::StructType>(type)) {
    for (llvm::Type* element : structure->elements()) {
      if (absl::Status status = ValidateType(element, metadata, visited);
          !status.ok()) {
        return status;
      }
    }
  } else if (const auto* array = llvm::dyn_cast<llvm::ArrayType>(type)) {
    return ValidateType(array->getElementType(), metadata, visited);
  } else if (const auto* vector = llvm::dyn_cast<llvm::VectorType>(type)) {
    if (vector->getElementCount().isScalable()) {
      return Rejected(metadata, "scalable-vector",
                      "scalable vectors are not in mapping version 1");
    }
    return ValidateType(vector->getElementType(), metadata, visited);
  } else if (llvm::isa<llvm::TargetExtType>(type)) {
    return Rejected(metadata, "target-extension-type",
                    "target extension types cannot cross the bridge");
  }
  return absl::OkStatus();
}

absl::Status ValidateValueType(const llvm::Value& value,
                               const MusaBridgeIrMetadata& metadata) {
  llvm::SmallPtrSet<const llvm::Type*, 16> visited;
  return ValidateType(value.getType(), metadata, visited);
}

absl::Status ValidateConstantTree(
    const llvm::Constant& constant, const MusaBridgeIrMetadata& metadata,
    llvm::SmallPtrSetImpl<const llvm::Constant*>& visited) {
  if (!visited.insert(&constant).second) return absl::OkStatus();
  if (absl::Status status = ValidateValueType(constant, metadata);
      !status.ok()) {
    return status;
  }
  if (const auto* gep = llvm::dyn_cast<llvm::GEPOperator>(&constant)) {
    llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
    if (absl::Status status =
            ValidateType(gep->getSourceElementType(), metadata, visited_types);
        !status.ok()) {
      return status;
    }
  }
  if (llvm::isa<llvm::GlobalValue>(constant)) return absl::OkStatus();
  for (const llvm::Use& operand : constant.operands()) {
    if (const auto* child = llvm::dyn_cast<llvm::Constant>(operand.get())) {
      if (absl::Status status = ValidateConstantTree(*child, metadata, visited);
          !status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

bool HasForbiddenTargetPrefix(llvm::StringRef name) {
  return name.starts_with("llvm.musa.") || name.starts_with("llvm.nvvm.") ||
         name.starts_with("llvm.amdgcn.") || name.starts_with("llvm.r600.") ||
         name.starts_with("__ockl_") || name.starts_with("__ocml_");
}

bool IsAllowedGenericIntrinsic(llvm::Intrinsic::ID id) {
  switch (id) {
    case llvm::Intrinsic::abs:
    case llvm::Intrinsic::assume:
    case llvm::Intrinsic::bitreverse:
    case llvm::Intrinsic::bswap:
    case llvm::Intrinsic::canonicalize:
    case llvm::Intrinsic::ceil:
    case llvm::Intrinsic::copysign:
    case llvm::Intrinsic::cos:
    case llvm::Intrinsic::ctlz:
    case llvm::Intrinsic::ctpop:
    case llvm::Intrinsic::cttz:
    case llvm::Intrinsic::exp:
    case llvm::Intrinsic::exp2:
    case llvm::Intrinsic::expect:
    case llvm::Intrinsic::fabs:
    case llvm::Intrinsic::floor:
    case llvm::Intrinsic::fma:
    case llvm::Intrinsic::fshl:
    case llvm::Intrinsic::fshr:
    case llvm::Intrinsic::lifetime_end:
    case llvm::Intrinsic::lifetime_start:
    case llvm::Intrinsic::log:
    case llvm::Intrinsic::log10:
    case llvm::Intrinsic::log2:
    case llvm::Intrinsic::maximum:
    case llvm::Intrinsic::maxnum:
    case llvm::Intrinsic::memcpy:
    case llvm::Intrinsic::memmove:
    case llvm::Intrinsic::memset:
    case llvm::Intrinsic::minimum:
    case llvm::Intrinsic::minnum:
    case llvm::Intrinsic::nearbyint:
    case llvm::Intrinsic::pow:
    case llvm::Intrinsic::powi:
    case llvm::Intrinsic::rint:
    case llvm::Intrinsic::round:
    case llvm::Intrinsic::roundeven:
    case llvm::Intrinsic::sadd_with_overflow:
    case llvm::Intrinsic::sin:
    case llvm::Intrinsic::smax:
    case llvm::Intrinsic::smin:
    case llvm::Intrinsic::sqrt:
    case llvm::Intrinsic::ssub_with_overflow:
    case llvm::Intrinsic::trunc:
    case llvm::Intrinsic::uadd_with_overflow:
    case llvm::Intrinsic::umax:
    case llvm::Intrinsic::umin:
    case llvm::Intrinsic::usub_with_overflow:
      return true;
    default:
      return false;
  }
}

absl::Status ValidateShimAttributes(const llvm::Function& function,
                                    const MusaShimSpec& spec,
                                    const MusaBridgeIrMetadata& metadata);

absl::Status ValidateShim(const llvm::Function& function,
                          const MusaShimSpec& spec,
                          const MusaBridgeIrMetadata& metadata) {
  if (!function.isDeclaration()) {
    return Rejected(metadata, "shim-definition",
                    absl::StrCat("shim ", spec.xla_symbol,
                                 " must remain an external declaration"));
  }
  if (function.getLinkage() != llvm::GlobalValue::ExternalLinkage) {
    return Rejected(
        metadata, "shim-linkage",
        absl::StrCat("shim ", spec.xla_symbol, " must have external linkage"));
  }
  const llvm::FunctionType* type = function.getFunctionType();
  bool signature_matches = !type->isVarArg() && type->params().empty();
  switch (spec.signature) {
    case MusaShimSignature::kVoidVoid:
      signature_matches &= type->getReturnType()->isVoidTy();
      break;
    case MusaShimSignature::kI32Void:
      signature_matches &= type->getReturnType()->isIntegerTy(32);
      break;
    case MusaShimSignature::kI64Void:
      signature_matches &= type->getReturnType()->isIntegerTy(64);
      break;
  }
  if (!signature_matches) {
    std::string rendered;
    llvm::raw_string_ostream stream(rendered);
    type->print(stream);
    stream.flush();
    return Rejected(
        metadata, "shim-signature",
        absl::StrCat("shim ", spec.xla_symbol, " has ", rendered, "; require ",
                     MusaShimSignatureText(spec.signature)));
  }
  if (function.getCallingConv() != llvm::CallingConv::C) {
    return Rejected(metadata, "calling-convention",
                    absl::StrCat("shim ", spec.xla_symbol,
                                 " must use the ordinary C convention"));
  }
  if (absl::Status status = ValidateShimAttributes(function, spec, metadata);
      !status.ok()) {
    return status;
  }
  if (function.hasFnAttribute(llvm::Attribute::Convergent) != spec.convergent) {
    return Rejected(metadata, "shim-convergence",
                    absl::StrCat("shim ", spec.xla_symbol,
                                 " has incorrect convergent semantics"));
  }
  llvm::MemoryEffects expected = llvm::MemoryEffects::none();
  switch (spec.memory_effects) {
    case MusaMemoryEffects::kNone:
      break;
    case MusaMemoryEffects::kReadWrite:
      expected = llvm::MemoryEffects::unknown();
      break;
    case MusaMemoryEffects::kInaccessibleRead:
      expected =
          llvm::MemoryEffects::inaccessibleMemOnly(llvm::ModRefInfo::Ref);
      break;
    case MusaMemoryEffects::kInaccessibleReadWrite:
      expected =
          llvm::MemoryEffects::inaccessibleMemOnly(llvm::ModRefInfo::ModRef);
      break;
  }
  if (function.getMemoryEffects() != expected) {
    return Rejected(metadata, "shim-memory-effects",
                    absl::StrCat("shim ", spec.xla_symbol, " requires ",
                                 MusaMemoryEffectsText(spec.memory_effects)));
  }
  for (const llvm::User* user : function.users()) {
    const auto* call = llvm::dyn_cast<llvm::CallBase>(user);
    if (call == nullptr || call->getCalledFunction() != &function) {
      return Rejected(metadata, "shim-address-taken",
                      absl::StrCat("shim ", spec.xla_symbol,
                                   " must be used only by direct calls"));
    }
  }
  return absl::OkStatus();
}

absl::Status RejectFunctionAttributes(const llvm::Function& function,
                                      const MusaBridgeIrMetadata& metadata) {
  if (!function.getAttributes().isEmpty()) {
    return Rejected(
        metadata, "function-attributes",
        absl::StrCat("function ", SanitizedSymbol(function.getName()),
                     " carries attributes outside mapping version "
                     "1"));
  }
  return absl::OkStatus();
}

absl::Status ValidateFunctionObjectState(const llvm::Function& function,
                                         const MusaBridgeIrMetadata& metadata) {
  if (!function.hasDefaultVisibility() ||
      function.getDLLStorageClass() != llvm::GlobalValue::DefaultStorageClass ||
      function.hasAtLeastLocalUnnamedAddr() || function.hasSection() ||
      function.getSectionPrefix().has_value() || function.hasComdat() ||
      function.hasPartition() || function.hasSanitizerMetadata() ||
      function.hasGC() || function.hasPersonalityFn() ||
      function.hasPrefixData() || function.hasPrologueData()) {
    return Rejected(
        metadata, "function-object-state",
        absl::StrCat("function ", SanitizedSymbol(function.getName()),
                     " carries unsupported ABI or linker state"));
  }
  return absl::OkStatus();
}

absl::Status ValidateGlobalObjectState(const llvm::GlobalVariable& global,
                                       const MusaBridgeIrMetadata& metadata) {
  const bool allowed_visibility =
      global.hasDefaultVisibility() ||
      (!global.hasLocalLinkage() && global.hasProtectedVisibility());
  if (!allowed_visibility ||
      global.getDLLStorageClass() != llvm::GlobalValue::DefaultStorageClass ||
      global.hasAtLeastLocalUnnamedAddr() || global.hasSection() ||
      global.getSectionPrefix().has_value() || global.hasComdat() ||
      global.hasPartition() || global.hasSanitizerMetadata() ||
      global.isExternallyInitialized()) {
    return Rejected(metadata, "global-object-state",
                    absl::StrCat("global ", SanitizedSymbol(global.getName()),
                                 " carries unsupported ABI or linker state"));
  }
  return absl::OkStatus();
}

absl::Status ValidateIntrinsicAttributes(const llvm::Function& function,
                                         const MusaBridgeIrMetadata& metadata) {
  const llvm::AttributeList expected = llvm::Intrinsic::getAttributes(
      function.getContext(), function.getIntrinsicID(),
      function.getFunctionType());
  if (function.getAttributes() != expected) {
    return Rejected(
        metadata, "intrinsic-attributes",
        absl::StrCat("intrinsic ", SanitizedSymbol(function.getName()),
                     " does not carry its canonical LLVM "
                     "attributes"));
  }
  return absl::OkStatus();
}

absl::Status ValidateShimAttributes(const llvm::Function& function,
                                    const MusaShimSpec& spec,
                                    const MusaBridgeIrMetadata& metadata) {
  const llvm::AttributeList& attributes = function.getAttributes();
  if (attributes.hasRetAttrs()) {
    return Rejected(metadata, "shim-attributes",
                    absl::StrCat("shim ", spec.xla_symbol,
                                 " must not carry return attributes"));
  }
  for (unsigned index = 0; index < function.arg_size(); ++index) {
    if (attributes.hasParamAttrs(index)) {
      return Rejected(metadata, "shim-attributes",
                      absl::StrCat("shim ", spec.xla_symbol,
                                   " must not carry parameter attributes"));
    }
  }
  for (llvm::Attribute attribute : attributes.getFnAttrs()) {
    if (attribute.isStringAttribute()) {
      return Rejected(metadata, "shim-attributes",
                      absl::StrCat("shim ", spec.xla_symbol,
                                   " carries an unsupported string attribute"));
    }
    switch (attribute.getKindAsEnum()) {
      case llvm::Attribute::Convergent:
      case llvm::Attribute::Memory:
      case llvm::Attribute::NoUnwind:
      case llvm::Attribute::WillReturn:
        break;
      default:
        return Rejected(
            metadata, "shim-attributes",
            absl::StrCat("shim ", spec.xla_symbol,
                         " carries an attribute outside the versioned ABI"));
    }
  }

  const bool require_no_unwind = (spec.required_attributes & kNoUnwind) != 0;
  const bool require_will_return =
      (spec.required_attributes & kWillReturn) != 0;
  if (function.hasFnAttribute(llvm::Attribute::NoUnwind) != require_no_unwind ||
      function.hasFnAttribute(llvm::Attribute::WillReturn) !=
          require_will_return) {
    return Rejected(metadata, "shim-attributes",
                    absl::StrCat("shim ", spec.xla_symbol,
                                 " does not carry exactly its versioned "
                                 "attributes"));
  }
  return absl::OkStatus();
}

absl::Status ValidateGlobals(const llvm::Module& module,
                             const MusaBridgeIrMetadata& metadata) {
  absl::flat_hash_map<absl::string_view, const MusaExportedGlobal*> expected;
  for (const MusaExportedGlobal& global : metadata.exported_globals) {
    expected.emplace(global.name, &global);
  }
  absl::flat_hash_set<absl::string_view> found;
  const llvm::DataLayout& data_layout = module.getDataLayout();
  for (const llvm::GlobalVariable& global : module.globals()) {
    if (absl::Status status = ValidateValueType(global, metadata);
        !status.ok()) {
      return status;
    }
    if (absl::Status status = ValidateGlobalObjectState(global, metadata);
        !status.ok()) {
      return status;
    }
    llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
    if (absl::Status status =
            ValidateType(global.getValueType(), metadata, visited_types);
        !status.ok()) {
      return status;
    }
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4> all_metadata;
    global.getAllMetadata(all_metadata);
    if (!all_metadata.empty()) {
      return Rejected(metadata, "global-metadata",
                      absl::StrCat("global ", SanitizedSymbol(global.getName()),
                                   " carries unsupported metadata"));
    }
    if (global.isThreadLocal()) {
      return Rejected(metadata, "thread-local-global",
                      absl::StrCat("global ", SanitizedSymbol(global.getName()),
                                   " is thread-local"));
    }
    if (global.isDeclaration()) {
      return Rejected(metadata, "unresolved-global",
                      absl::StrCat("global ", SanitizedSymbol(global.getName()),
                                   " is unresolved"));
    }
    if (global.hasInitializer()) {
      llvm::SmallPtrSet<const llvm::Constant*, 16> visited_constants;
      if (absl::Status status = ValidateConstantTree(
              *global.getInitializer(), metadata, visited_constants);
          !status.ok()) {
        return status;
      }
    }
    auto it = expected.find(global.getName());
    if (global.hasLocalLinkage()) {
      if (it != expected.end()) {
        return Rejected(
            metadata, "exported-globals",
            absl::StrCat("listed global ", SanitizedSymbol(global.getName()),
                         " has local linkage"));
      }
      continue;
    }
    if (global.getLinkage() != llvm::GlobalValue::ExternalLinkage) {
      return Rejected(
          metadata, "global-linkage",
          absl::StrCat("exported global ", SanitizedSymbol(global.getName()),
                       " must have external linkage"));
    }
    if (it == expected.end()) {
      return Rejected(metadata, "exported-globals",
                      absl::StrCat("externally visible global ",
                                   SanitizedSymbol(global.getName()),
                                   " is absent from the request"));
    }
    const MusaExportedGlobal& spec = *it->second;
    llvm::TypeSize allocation_size =
        data_layout.getTypeAllocSize(global.getValueType());
    if (allocation_size.isScalable()) {
      return Rejected(
          metadata, "exported-globals",
          absl::StrCat("global ", spec.name, " has scalable allocation size"));
    }
    uint64_t alignment =
        global.getAlign()
            ? global.getAlign()->value()
            : data_layout.getABITypeAlign(global.getValueType()).value();
    if (global.getAddressSpace() != spec.address_space ||
        allocation_size.getFixedValue() != spec.size ||
        alignment != spec.alignment ||
        global.isConstant() !=
            (spec.kind == MusaExportedGlobalKind::kConstant)) {
      return Rejected(metadata, "exported-globals",
                      absl::StrCat("global ", spec.name,
                                   " does not match kind/address-space/size/"
                                   "alignment metadata"));
    }
    found.insert(spec.name);
  }
  if (found.size() != expected.size()) {
    return Rejected(metadata, "exported-globals",
                    "one or more listed globals are missing from the module");
  }
  return absl::OkStatus();
}

absl::Status ValidateFunctions(const llvm::Module& module,
                               const MusaBridgeIrMetadata& metadata) {
  absl::flat_hash_set<absl::string_view> expected_kernels;
  for (const std::string& name : metadata.kernel_entry_names) {
    expected_kernels.insert(name);
  }
  absl::flat_hash_set<absl::string_view> found_kernels;

  for (const llvm::Function& function : module.functions()) {
    const llvm::StringRef name = function.getName();
    const std::string function_name = SanitizedSymbol(name);
    if (function.getAddressSpace() != 0) {
      return Rejected(metadata, "function-address-space",
                      absl::StrCat("function ", function_name,
                                   " must use generic address space 0"));
    }
    if (absl::Status status = ValidateValueType(function, metadata);
        !status.ok()) {
      return status;
    }
    if (absl::Status status = ValidateFunctionObjectState(function, metadata);
        !status.ok()) {
      return status;
    }
    llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
    if (absl::Status status =
            ValidateType(function.getFunctionType(), metadata, visited_types);
        !status.ok()) {
      return status;
    }
    if (function.getCallingConv() != llvm::CallingConv::C) {
      return Rejected(
          metadata, "calling-convention",
          absl::StrCat("function ", function_name, " uses calling convention ",
                       function.getCallingConv(), "; native convention ",
                       stream_executor::musa::kMusaKernelCallingConvention,
                       " is bridge-only"));
    }
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4> function_metadata;
    function.getAllMetadata(function_metadata);
    if (!function_metadata.empty()) {
      return Rejected(metadata, "function-metadata",
                      absl::StrCat("function ", function_name,
                                   " carries unsupported metadata"));
    }

    if (HasForbiddenTargetPrefix(name)) {
      return Rejected(metadata, "foreign-target-construct",
                      absl::StrCat("forbidden function ", function_name));
    }
    if (name.starts_with("__xla_musa_")) {
      const MusaShimSpec* spec = FindMusaShim(name.str());
      if (spec == nullptr) {
        return Rejected(metadata, "unknown-shim",
                        absl::StrCat("unknown shim ", function_name));
      }
      if (absl::Status status = ValidateShim(function, *spec, metadata);
          !status.ok()) {
        return status;
      }
      continue;
    }
    if (function.isIntrinsic()) {
      if (!IsAllowedGenericIntrinsic(function.getIntrinsicID())) {
        return Rejected(metadata, "llvm-intrinsic",
                        absl::StrCat("intrinsic ", function_name,
                                     " is not in mapping version 1"));
      }
      if (absl::Status status = ValidateIntrinsicAttributes(function, metadata);
          !status.ok()) {
        return status;
      }
      if (function.getLinkage() != llvm::GlobalValue::ExternalLinkage) {
        return Rejected(metadata, "intrinsic-linkage",
                        absl::StrCat("intrinsic ", function_name,
                                     " must have external linkage"));
      }
      continue;
    }
    if (function.isDeclaration()) {
      return Rejected(metadata, "unresolved-call",
                      absl::StrCat("external function ", function_name,
                                   " is not an allowed shim or intrinsic"));
    }
    if (absl::Status status = RejectFunctionAttributes(function, metadata);
        !status.ok()) {
      return status;
    }

    bool listed_kernel = expected_kernels.contains(name.str());
    if (listed_kernel) {
      if (function.getLinkage() != llvm::GlobalValue::ExternalLinkage ||
          !function.getReturnType()->isVoidTy() || function.isVarArg()) {
        return Rejected(
            metadata, "kernel-list",
            absl::StrCat("kernel ", function_name,
                         " must have external linkage, return void, and be "
                         "nonvariadic"));
      }
      found_kernels.insert(name.str());
    } else if (!function.hasLocalLinkage()) {
      return Rejected(
          metadata, "kernel-list",
          absl::StrCat("externally visible function ", function_name,
                       " is not listed as a kernel"));
    }

    for (const llvm::BasicBlock& block : function) {
      for (const llvm::Instruction& instruction : block) {
        if (absl::Status status = ValidateValueType(instruction, metadata);
            !status.ok()) {
          return status;
        }
        if (const auto* alloca =
                llvm::dyn_cast<llvm::AllocaInst>(&instruction)) {
          llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
          if (absl::Status status = ValidateType(alloca->getAllocatedType(),
                                                 metadata, visited_types);
              !status.ok()) {
            return status;
          }
        }
        if (const auto* gep =
                llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction)) {
          llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
          if (absl::Status status = ValidateType(gep->getSourceElementType(),
                                                 metadata, visited_types);
              !status.ok()) {
            return status;
          }
        }
        if (const auto* fp = llvm::dyn_cast<llvm::FPMathOperator>(&instruction);
            fp != nullptr && fp->getFastMathFlags().any()) {
          return Rejected(metadata, "fast-math-flags",
                          absl::StrCat("function ", function_name,
                                       " contains unversioned fast-math "
                                       "flags"));
        }
        for (const llvm::Use& operand : instruction.operands()) {
          if (absl::Status status = ValidateValueType(*operand.get(), metadata);
              !status.ok()) {
            return status;
          }
          if (const auto* constant =
                  llvm::dyn_cast<llvm::Constant>(operand.get())) {
            llvm::SmallPtrSet<const llvm::Constant*, 16> visited_constants;
            if (absl::Status status = ValidateConstantTree(*constant, metadata,
                                                           visited_constants);
                !status.ok()) {
              return status;
            }
          }
        }
        llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 4>
            instruction_metadata;
        instruction.getAllMetadata(instruction_metadata);
        if (!instruction_metadata.empty() || instruction.getDebugLoc()) {
          return Rejected(metadata, "instruction-metadata",
                          absl::StrCat("function ", function_name,
                                       " carries unsupported metadata"));
        }
        if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
          if (!call->getAttributes().isEmpty()) {
            return Rejected(metadata, "call-site-attributes",
                            absl::StrCat("call in function ", function_name,
                                         " carries attributes outside mapping "
                                         "version 1"));
          }
          if (call->isInlineAsm()) {
            return Rejected(metadata, "inline-assembly",
                            absl::StrCat("function ", function_name,
                                         " contains inline assembly"));
          }
          if (call->hasOperandBundles()) {
            return Rejected(metadata, "operand-bundle",
                            absl::StrCat("function ", function_name,
                                         " contains an operand bundle"));
          }
          if (call->getCalledFunction() == nullptr) {
            return Rejected(metadata, "indirect-call",
                            absl::StrCat("function ", function_name,
                                         " contains an indirect call"));
          }
          if (call->getCallingConv() != llvm::CallingConv::C) {
            return Rejected(metadata, "calling-convention",
                            absl::StrCat("call in function ", function_name,
                                         " uses a non-C calling convention"));
          }
        }
        if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction);
            (load != nullptr && load->isAtomic()) ||
            llvm::isa<llvm::AtomicRMWInst, llvm::AtomicCmpXchgInst,
                      llvm::FenceInst>(instruction)) {
          return Rejected(
              metadata, "atomics",
              "atomics are reserved pending the C06 mapping probes");
        }
        if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction);
            store != nullptr && store->isAtomic()) {
          return Rejected(
              metadata, "atomics",
              "atomics are reserved pending the C06 mapping probes");
        }
      }
    }
  }
  if (found_kernels.size() != expected_kernels.size()) {
    return Rejected(metadata, "kernel-list",
                    "one or more listed kernels are missing from the module");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateMusaBridgeIr(absl::string_view llvm_ir,
                                  const MusaBridgeIrMetadata& metadata) {
  if (absl::Status status = ValidateMetadata(metadata); !status.ok()) {
    return status;
  }
  if (llvm_ir.empty() || llvm_ir.size() > kMaxMusaInterchangeIrBytes) {
    return Rejected(metadata, "input-size",
                    "LLVM interchange input must be nonempty and bounded");
  }

  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
      llvm::StringRef(llvm_ir.data(), llvm_ir.size()), diagnostic, context);
  if (module == nullptr) {
    return Rejected(metadata, "llvm-parse",
                    "current LLVM parser rejected the module");
  }
  // Parsing does not imply verification. Verify before querying types, data
  // layout, operands, attributes, or metadata so malformed interchange cannot
  // reach semantic traversal. Do not attach LLVM's verifier output: it embeds
  // attacker-controlled instructions despite this API's sanitized-diagnostic
  // contract.
  if (llvm::verifyModule(*module)) {
    return Rejected(metadata, "llvm-verifier",
                    "current LLVM verifier rejected the module");
  }
  if (module->getTargetTriple().str() != kMusaTargetTriple) {
    return Rejected(
        metadata, "target-triple",
        absl::StrCat("module triple does not match required target ",
                     kMusaTargetTriple));
  }
  if (module->getDataLayoutStr() != kMusaDataLayout) {
    return Rejected(metadata, "data-layout",
                    "module data layout does not match the frozen mp_21 ABI");
  }
  if (!module->getModuleInlineAsm().empty()) {
    return Rejected(metadata, "inline-assembly",
                    "module-level inline assembly is forbidden");
  }
  if (!module->aliases().empty()) {
    return Rejected(metadata, "global-alias",
                    "global aliases are not in mapping version 1");
  }
  if (!module->ifuncs().empty()) {
    return Rejected(metadata, "global-ifunc",
                    "indirect functions are not in mapping version 1");
  }
  if (!module->getComdatSymbolTable().empty()) {
    return Rejected(metadata, "module-comdat",
                    "COMDAT definitions are not in mapping version 1");
  }
  if (module->named_metadata_begin() != module->named_metadata_end()) {
    return Rejected(metadata, "named-metadata",
                    "named metadata is not in mapping version 1");
  }
  for (llvm::StructType* type : module->getIdentifiedStructTypes()) {
    llvm::SmallPtrSet<const llvm::Type*, 16> visited_types;
    if (absl::Status status = ValidateType(type, metadata, visited_types);
        !status.ok()) {
      return status;
    }
  }

  if (absl::Status status = ValidateGlobals(*module, metadata); !status.ok()) {
    return status;
  }
  if (absl::Status status = ValidateFunctions(*module, metadata);
      !status.ok()) {
    return status;
  }

  return absl::OkStatus();
}

absl::Status ValidateMusaBridgeCompileRequestIr(
    const MusaBridgeCompileRequest& request) {
  if (absl::Status status = ValidateMusaBridgeCompileRequest(request);
      !status.ok()) {
    return status;
  }

  MusaBridgeIrMetadata metadata;
  metadata.protocol_version = request.protocol_version();
  metadata.shim_abi_version = request.shim_abi_version();
  metadata.mapping_version = request.mapping_version();
  metadata.module_name = request.module_name();
  metadata.architecture = request.architecture();
  metadata.kernel_entry_names.assign(request.kernel_entry_names().begin(),
                                     request.kernel_entry_names().end());
  metadata.exported_globals.reserve(request.exported_globals_size());
  for (const MusaBridgeExportedGlobal& global : request.exported_globals()) {
    metadata.exported_globals.push_back(MusaExportedGlobal{
        global.name(),
        global.kind() == MUSA_BRIDGE_GLOBAL_KIND_CONSTANT
            ? MusaExportedGlobalKind::kConstant
            : MusaExportedGlobalKind::kMutable,
        global.address_space(), global.size_bytes(), global.alignment_bytes()});
  }

  // Version 1 exports only kernel entries and the typed globals described in
  // the request. Extra preservation roots would be an unversioned ABI surface.
  if (request.exported_symbol_names_size() !=
      request.kernel_entry_names_size() + request.exported_globals_size()) {
    return Rejected(metadata, "exported-symbols",
                    "exported symbols must be exactly kernels plus typed "
                    "globals in protocol version 1");
  }
  absl::flat_hash_set<absl::string_view> preservation_roots;
  for (const std::string& kernel : request.kernel_entry_names()) {
    preservation_roots.insert(kernel);
  }
  for (const MusaExportedGlobal& global : metadata.exported_globals) {
    preservation_roots.insert(global.name);
  }
  for (const std::string& symbol : request.exported_symbol_names()) {
    if (!preservation_roots.contains(symbol)) {
      return Rejected(metadata, "exported-symbols",
                      absl::StrCat("unexpected preservation root ", symbol));
    }
  }
  return ValidateMusaBridgeIr(request.normalized_llvm(), metadata);
}

}  // namespace xla::gpu::musa
