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
#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
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
                      "named metadata is outside compatibility revision 1");
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
              attribute.getDereferenceableBytes() == 0 ||
              attribute.getDereferenceableBytes() >
                  kMaxMusaInterchangeIrBytes) {
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
    // by optimization. They are not part of the vendor kernel ABI.
    argument.removeAttr(llvm::Attribute::NoAlias);
    argument.removeAttr(llvm::Attribute::Alignment);
    argument.removeAttr(llvm::Attribute::Dereferenceable);
  }
  return absl::OkStatus();
}

absl::Status NormalizeSqrtIntrinsicAttributes(llvm::Function& function,
                                              absl::string_view module_name) {
  const llvm::AttributeList current = llvm::Intrinsic::getAttributes(
      function.getContext(), function.getIntrinsicID(),
      function.getFunctionType());
  if (function.getAttributes() != current) {
    return Rejected(module_name, "intrinsic-attributes",
                    "sqrt must begin with its canonical current-LLVM attrs");
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

absl::Status NormalizeFunctions(llvm::Module& module,
                                absl::string_view module_name,
                                MusaBridgeIrMetadata& metadata) {
  for (llvm::Function& function : module.functions()) {
    if (function.isIntrinsic()) {
      if (function.getIntrinsicID() != llvm::Intrinsic::sqrt) {
        return Rejected(module_name, "intrinsic-compatibility",
                        "generic intrinsic lacks an LLVM 14 profile");
      }
      if (absl::Status status =
              NormalizeSqrtIntrinsicAttributes(function, module_name);
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
  if (metadata.kernel_entry_names.empty() ||
      std::adjacent_find(metadata.kernel_entry_names.begin(),
                         metadata.kernel_entry_names.end()) !=
          metadata.kernel_entry_names.end()) {
    return Rejected(module_name, "kernel-marker",
                    "kernel marker set must be nonempty and unique");
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

  MusaBridgeIrMetadata metadata;
  metadata.module_name = std::string(module_name);
  if (absl::Status status = NormalizeFunctions(*module, module_name, metadata);
      !status.ok()) {
    return status;
  }
  StripInvariantLoadMetadata(*module);
  if (absl::Status status =
          CollectExportedGlobals(*module, module_name, metadata);
      !status.ok()) {
    return status;
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
