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

#include "xla/tools/musa_llvm_bridge/mubin_validator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"
#include "xla/stream_executor/musa/musa_mubin.h"

namespace xla::gpu::musa::bridge {
namespace {

struct DynamicSymbol {
  uint64_t size;
  uint8_t binding;
  uint8_t type;
  uint8_t visibility;
  bool defined;
};

absl::Status InvalidOutput(absl::string_view detail) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid MUSA compiler output: ", detail));
}

template <typename T>
absl::StatusOr<T> TakeExpected(llvm::Expected<T> expected,
                               absl::string_view detail) {
  if (!expected) {
    llvm::consumeError(expected.takeError());
    return InvalidOutput(detail);
  }
  return std::move(*expected);
}

}  // namespace

absl::Status ValidateMubinOutput(absl::Span<const uint8_t> mubin,
                                 const MusaBridgeCompileRequest& request) {
  if (absl::Status status = ValidateMusaBridgeCompileRequest(request);
      !status.ok()) {
    return status;
  }
  if (mubin.size() > kMusaBridgeMaxMubinBytes) {
    return InvalidOutput("MUBIN exceeds the protocol size limit");
  }
  if (auto metadata = stream_executor::musa::ValidateMusaMubin(mubin);
      !metadata.ok()) {
    return metadata.status();
  }

  const llvm::StringRef bytes(reinterpret_cast<const char*>(mubin.data()),
                              mubin.size());
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> object_or =
      llvm::object::ObjectFile::createObjectFile(
          llvm::MemoryBufferRef(bytes, "mubin-memory"));
  if (!object_or) {
    llvm::consumeError(object_or.takeError());
    return InvalidOutput("vendor LLVM could not parse the MUBIN object");
  }
  std::unique_ptr<llvm::object::ObjectFile> object = std::move(*object_or);
  const auto* elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(object.get());
  if (elf == nullptr) {
    return InvalidOutput("MUBIN is not an ELF object");
  }

  absl::flat_hash_map<std::string, DynamicSymbol> symbols;
  for (const llvm::object::ELFSymbolRef& symbol :
       elf->getDynamicSymbolIterators()) {
    absl::StatusOr<llvm::StringRef> name =
        TakeExpected(symbol.getName(), "dynamic symbol has no valid name");
    if (!name.ok()) return name.status();
    if (name->empty()) continue;

    absl::StatusOr<llvm::object::section_iterator> section = TakeExpected(
        symbol.getSection(), "dynamic symbol has an invalid section");
    if (!section.ok()) return section.status();
    const DynamicSymbol record = {
        .size = symbol.getSize(),
        .binding = symbol.getBinding(),
        .type = symbol.getELFType(),
        .visibility = static_cast<uint8_t>(symbol.getOther() & 0x3),
        .defined = *section != object->section_end(),
    };
    if (!symbols.emplace(name->str(), record).second) {
      // Do not echo an untrusted compiler-output symbol into diagnostics.
      return InvalidOutput(".dynsym contains a duplicate symbol name");
    }
  }

  const auto validate_common = [&](absl::string_view name,
                                   uint8_t expected_type,
                                   uint64_t expected_size) -> absl::Status {
    auto it = symbols.find(name);
    if (it == symbols.end()) {
      return InvalidOutput(
          absl::StrCat("requested symbol ", name, " is absent from .dynsym"));
    }
    const DynamicSymbol& symbol = it->second;
    if (!symbol.defined) {
      return InvalidOutput(
          absl::StrCat("requested symbol ", name, " is undefined"));
    }
    if (symbol.binding != llvm::ELF::STB_GLOBAL) {
      return InvalidOutput(
          absl::StrCat("requested symbol ", name, " is not strongly exported"));
    }
    if (symbol.visibility != llvm::ELF::STV_PROTECTED) {
      return InvalidOutput(absl::StrCat("requested symbol ", name,
                                        " does not have protected visibility"));
    }
    if (symbol.type != expected_type) {
      return InvalidOutput(absl::StrCat("requested symbol ", name,
                                        " has the wrong ELF symbol type"));
    }
    if (expected_type == llvm::ELF::STT_OBJECT &&
        symbol.size != expected_size) {
      return InvalidOutput(
          absl::StrCat("requested global ", name, " has the wrong ELF size"));
    }
    return absl::OkStatus();
  };

  for (const std::string& kernel : request.kernel_entry_names()) {
    if (absl::Status status =
            validate_common(kernel, llvm::ELF::STT_FUNC, /*expected_size=*/0);
        !status.ok()) {
      return status;
    }
  }
  for (const MusaBridgeExportedGlobal& global : request.exported_globals()) {
    if (absl::Status status = validate_common(
            global.name(), llvm::ELF::STT_OBJECT, global.size_bytes());
        !status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu::musa::bridge
