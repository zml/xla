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

#include "xla/stream_executor/module_binary.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/module_binary.pb.h"

namespace stream_executor {
namespace {

ModuleFormatProto ToProtoFormat(ModuleFormat format) {
  switch (format) {
    case ModuleFormat::kUnspecified:
      return MODULE_FORMAT_UNSPECIFIED;
    case ModuleFormat::kCudaCubin:
      return MODULE_FORMAT_CUDA_CUBIN;
    case ModuleFormat::kRocmHsaco:
      return MODULE_FORMAT_ROCM_HSACO;
    case ModuleFormat::kSpirv:
      return MODULE_FORMAT_SPIRV;
    case ModuleFormat::kLevelZeroNative:
      return MODULE_FORMAT_LEVEL_ZERO_NATIVE;
  }
}

absl::StatusOr<ModuleFormat> FromProtoFormat(ModuleFormatProto format) {
  switch (format) {
    case MODULE_FORMAT_UNSPECIFIED:
      return ModuleFormat::kUnspecified;
    case MODULE_FORMAT_CUDA_CUBIN:
      return ModuleFormat::kCudaCubin;
    case MODULE_FORMAT_ROCM_HSACO:
      return ModuleFormat::kRocmHsaco;
    case MODULE_FORMAT_SPIRV:
      return ModuleFormat::kSpirv;
    case MODULE_FORMAT_LEVEL_ZERO_NATIVE:
      return ModuleFormat::kLevelZeroNative;
    default:
      return absl::InvalidArgumentError("Unknown module binary format");
  }
}

}  // namespace

ModuleBinaryProto ModuleBinary::ToProto() const {
  ModuleBinaryProto proto;
  proto.set_format(ToProtoFormat(format));
  proto.set_data(bytes.data(), bytes.size());
  proto.set_compatibility_key(compatibility_key);
  return proto;
}

absl::StatusOr<ModuleBinary> ModuleBinary::FromProto(
    const ModuleBinaryProto& proto) {
  absl::StatusOr<ModuleFormat> format = FromProtoFormat(proto.format());
  if (!format.ok()) return format.status();
  return ModuleBinary(
      std::vector<uint8_t>(proto.data().begin(), proto.data().end()), *format,
      proto.compatibility_key());
}

absl::string_view ModuleFormatName(ModuleFormat format) {
  switch (format) {
    case ModuleFormat::kUnspecified:
      return "unspecified";
    case ModuleFormat::kCudaCubin:
      return "CUDA CUBIN";
    case ModuleFormat::kRocmHsaco:
      return "ROCm HSACO";
    case ModuleFormat::kSpirv:
      return "SPIR-V";
    case ModuleFormat::kLevelZeroNative:
      return "Level Zero native";
  }
}

}  // namespace stream_executor
