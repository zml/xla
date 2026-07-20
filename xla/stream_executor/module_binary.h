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

#ifndef XLA_STREAM_EXECUTOR_MODULE_BINARY_H_
#define XLA_STREAM_EXECUTOR_MODULE_BINARY_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/module_binary.pb.h"

namespace stream_executor {

enum class ModuleFormat : uint8_t {
  kUnspecified = 0,
  kCudaCubin,
  kRocmHsaco,
  kSpirv,
  kLevelZeroNative,
};

// A non-owning view of a target module.
struct ModuleBinaryView {
  ModuleBinaryView() = default;
  // Backward-compatible view for callers that have not yet attached a format.
  ModuleBinaryView(absl::Span<const uint8_t> bytes) : bytes(bytes) {}
  ModuleBinaryView(absl::Span<const uint8_t> bytes, ModuleFormat format,
                   absl::string_view compatibility_key = {})
      : bytes(bytes), format(format), compatibility_key(compatibility_key) {}

  bool empty() const { return bytes.empty(); }
  size_t size() const { return bytes.size(); }

  absl::Span<const uint8_t> bytes;
  ModuleFormat format = ModuleFormat::kUnspecified;
  absl::string_view compatibility_key;
};

// An owning target module. Native modules carry a compatibility key that is
// validated by the executor before they are loaded.
struct ModuleBinary {
  ModuleBinary() = default;
  ModuleBinary(std::initializer_list<uint8_t> bytes) : bytes(bytes) {}
  ModuleBinary(std::vector<uint8_t> bytes, ModuleFormat format,
               std::string compatibility_key = {})
      : bytes(std::move(bytes)),
        format(format),
        compatibility_key(std::move(compatibility_key)) {}

  bool empty() const { return bytes.empty(); }
  size_t size() const { return bytes.size(); }
  ModuleBinaryView view() const {
    return ModuleBinaryView(bytes, format, compatibility_key);
  }

  ModuleBinaryProto ToProto() const;
  static absl::StatusOr<ModuleBinary> FromProto(const ModuleBinaryProto& proto);

  std::vector<uint8_t> bytes;
  ModuleFormat format = ModuleFormat::kUnspecified;
  std::string compatibility_key;
};

absl::string_view ModuleFormatName(ModuleFormat format);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MODULE_BINARY_H_
