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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.pb.h"

namespace stream_executor {

class MusaComputeCapability {
 public:
  explicit MusaComputeCapability(std::string architecture)
      : architecture_(std::move(architecture)) {}

  MusaComputeCapability(std::string architecture, int major, int minor,
                        int hardware_warp_size, int logical_subgroup_size)
      : architecture_(std::move(architecture)),
        major_(major),
        minor_(minor),
        hardware_warp_size_(hardware_warp_size),
        logical_subgroup_size_(logical_subgroup_size) {}

  explicit MusaComputeCapability(const MusaComputeCapabilityProto& proto)
      : architecture_(proto.architecture()),
        major_(proto.major()),
        minor_(proto.minor()),
        hardware_warp_size_(proto.hardware_warp_size()),
        logical_subgroup_size_(proto.logical_subgroup_size()) {}

  MusaComputeCapability() = default;

  const std::string& architecture() const { return architecture_; }
  int major() const { return major_; }
  int minor() const { return minor_; }
  int hardware_warp_size() const { return hardware_warp_size_; }
  int logical_subgroup_size() const { return logical_subgroup_size_; }

  std::string ToString() const { return architecture_; }

  MusaComputeCapabilityProto ToProto() const {
    MusaComputeCapabilityProto proto;
    proto.set_architecture(architecture_);
    proto.set_major(major_);
    proto.set_minor(minor_);
    proto.set_hardware_warp_size(hardware_warp_size_);
    proto.set_logical_subgroup_size(logical_subgroup_size_);
    return proto;
  }

  static absl::StatusOr<MusaComputeCapability> FromProto(
      const MusaComputeCapabilityProto& proto) {
    const bool has_extended_metadata =
        proto.major() != 0 || proto.minor() != 0 ||
        proto.hardware_warp_size() != 0 || proto.logical_subgroup_size() != 0;

    // Protos emitted before the numeric capability and width fields were
    // introduced contain only the architecture string. Keep accepting those
    // payloads; a zero-valued extension tuple is the unambiguous legacy form.
    if (!has_extended_metadata) {
      return MusaComputeCapability{proto};
    }

    if (proto.major() <= 0 || proto.minor() < 0 ||
        proto.hardware_warp_size() <= 0 || proto.logical_subgroup_size() <= 0) {
      return absl::InvalidArgumentError(
          "An extended MUSA compute capability must contain a positive major "
          "version, a non-negative minor version, and positive hardware and "
          "logical widths.");
    }

    const std::string expected_architecture =
        "mp_" + std::to_string(proto.major()) + std::to_string(proto.minor());
    if (proto.architecture() != expected_architecture) {
      return absl::InvalidArgumentError(
          "MUSA architecture does not match its numeric compute capability: "
          "expected " +
          expected_architecture + ", got " + proto.architecture());
    }

    if (proto.hardware_warp_size() % proto.logical_subgroup_size() != 0) {
      return absl::InvalidArgumentError(
          "MUSA logical subgroup size must divide the hardware warp size.");
    }

    return MusaComputeCapability{proto};
  }

  bool operator==(const MusaComputeCapability& other) const {
    return architecture_ == other.architecture_ && major_ == other.major_ &&
           minor_ == other.minor_ &&
           hardware_warp_size_ == other.hardware_warp_size_ &&
           logical_subgroup_size_ == other.logical_subgroup_size_;
  }

  bool operator!=(const MusaComputeCapability& other) const {
    return !this->operator==(other);
  }

 private:
  std::string architecture_ = "unknown";
  int major_ = 0;
  int minor_ = 0;
  int hardware_warp_size_ = 0;
  int logical_subgroup_size_ = 0;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_COMPUTE_CAPABILITY_H_
