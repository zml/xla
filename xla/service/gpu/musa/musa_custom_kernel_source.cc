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

#include "xla/service/gpu/musa/musa_custom_kernel_source.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"

namespace xla::gpu::musa {

absl::StatusOr<LlvmKernelSource> ParseMusaCustomKernelSource(
    absl::string_view kernel_name, absl::string_view llvm_ir) {
  if (kernel_name.empty() || kernel_name.size() > 256) {
    return absl::InvalidArgumentError(
        "MUSA custom kernel name must be nonempty and at most 256 bytes");
  }

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
      llvm::StringRef(llvm_ir.data(), llvm_ir.size()), diagnostic,
      *llvm_context);
  if (module == nullptr) {
    return absl::InvalidArgumentError(
        "MUSA custom kernel current-LLVM parser rejected the source");
  }

  absl::StatusOr<MusaLlvm14CompatibilityResult> compatible =
      NormalizeMusaLlvmForLlvm14(*module, "musa_custom_kernel");
  if (!compatible.ok()) return compatible.status();
  if (compatible->metadata.kernel_entry_names.size() != 1) {
    return absl::InvalidArgumentError(
        "MUSA custom kernel source must define exactly one marked kernel");
  }
  if (compatible->metadata.kernel_entry_names.front() != kernel_name) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MUSA custom kernel entry does not match requested name '",
        kernel_name, "'"));
  }
  if (!compatible->metadata.exported_globals.empty()) {
    return absl::UnimplementedError(
        "MUSA custom kernel source does not yet support exported globals");
  }

  return LlvmKernelSource(std::move(llvm_context), std::move(module));
}

}  // namespace xla::gpu::musa
