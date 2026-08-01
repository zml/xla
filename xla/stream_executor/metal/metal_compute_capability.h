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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_COMPUTE_CAPABILITY_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_COMPUTE_CAPABILITY_H_

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "xla/stream_executor/device_description.pb.h"

namespace stream_executor {

class MetalComputeCapability {
 public:
  static constexpr int kDefaultArchitectureGen = 15;
  static constexpr char kDefaultArchitectureSize = ' ';

  MetalComputeCapability() = default;
  explicit MetalComputeCapability(std::string architecture)
      : architecture_(std::move(architecture)) {}
  explicit MetalComputeCapability(const MetalComputeCapabilityProto& proto)
      : architecture_(proto.gpu_family()) {}

  const std::string& architecture() const { return architecture_; }

  int architecture_gen() const {
    if (!HasMlxArchitectureSuffix()) return kDefaultArchitectureGen;
    const size_t n = architecture_.size();
    return (architecture_[n - 3] - '0') * 10 + (architecture_[n - 2] - '0');
  }

  char architecture_size() const {
    return HasMlxArchitectureSuffix() ? architecture_.back()
                                      : kDefaultArchitectureSize;
  }

  std::string ToString() const {
    return absl::StrCat("Metal:", architecture_);
  }

  MetalComputeCapabilityProto ToProto() const {
    MetalComputeCapabilityProto proto;
    proto.set_gpu_family(architecture_);
    return proto;
  }

  static MetalComputeCapability FromProto(
      const MetalComputeCapabilityProto& proto) {
    return MetalComputeCapability(proto.gpu_family());
  }

  bool operator==(const MetalComputeCapability& other) const {
    return architecture_ == other.architecture_;
  }
  bool operator!=(const MetalComputeCapability& other) const {
    return !(*this == other);
  }

 private:
  bool HasMlxArchitectureSuffix() const {
    if (architecture_.size() < 3) return false;
    const size_t n = architecture_.size();
    return architecture_[n - 3] >= '0' && architecture_[n - 3] <= '9' &&
           architecture_[n - 2] >= '0' && architecture_[n - 2] <= '9';
  }

  std::string architecture_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_COMPUTE_CAPABILITY_H_
