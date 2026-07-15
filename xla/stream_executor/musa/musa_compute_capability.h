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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_COMPUTE_CAPABILITY_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_COMPUTE_CAPABILITY_H_

#include <string>
#include <utility>

#include "xla/stream_executor/device_description.pb.h"

namespace stream_executor {

class MusaComputeCapability {
 public:
  explicit MusaComputeCapability(std::string architecture)
      : architecture_(std::move(architecture)) {}

  explicit MusaComputeCapability(const MusaComputeCapabilityProto& proto)
      : architecture_(proto.architecture()) {}

  MusaComputeCapability() = default;

  const std::string& architecture() const { return architecture_; }

  std::string ToString() const { return architecture_; }

  MusaComputeCapabilityProto ToProto() const {
    MusaComputeCapabilityProto proto;
    proto.set_architecture(architecture_);
    return proto;
  }

  static MusaComputeCapability FromProto(
      const MusaComputeCapabilityProto& proto) {
    return MusaComputeCapability{proto.architecture()};
  }

  bool operator==(const MusaComputeCapability& other) const {
    return architecture_ == other.architecture_;
  }

  bool operator!=(const MusaComputeCapability& other) const {
    return !this->operator==(other);
  }

 private:
  std::string architecture_ = "unknown";
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_COMPUTE_CAPABILITY_H_
