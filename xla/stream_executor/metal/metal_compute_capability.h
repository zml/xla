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

// A first-class GpuComputeCapability alternative for the Apple Metal/AIR backend
// (peer to CudaComputeCapability / RocmComputeCapability / OneAPIComputeCapability).
// `gpu_family` is the Apple GPU family string (e.g. "Apple9" for M3/M4-class
// devices); it is descriptive — nothing branches on it yet, but it gives Metal a
// real identity so the backend no longer has to masquerade as a CUDA device.
class MetalComputeCapability {
 public:
  MetalComputeCapability() = default;
  explicit MetalComputeCapability(std::string gpu_family)
      : gpu_family_(std::move(gpu_family)) {}
  explicit MetalComputeCapability(const MetalComputeCapabilityProto& proto)
      : gpu_family_(proto.gpu_family()) {}

  const std::string& gpu_family() const { return gpu_family_; }

  std::string ToString() const { return absl::StrCat("Metal:", gpu_family_); }

  MetalComputeCapabilityProto ToProto() const {
    MetalComputeCapabilityProto proto;
    proto.set_gpu_family(gpu_family_);
    return proto;
  }

  static MetalComputeCapability FromProto(
      const MetalComputeCapabilityProto& proto) {
    return MetalComputeCapability(proto.gpu_family());
  }

  bool operator==(const MetalComputeCapability& other) const {
    return gpu_family_ == other.gpu_family_;
  }
  bool operator!=(const MetalComputeCapability& other) const {
    return !(*this == other);
  }

 private:
  std::string gpu_family_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_COMPUTE_CAPABILITY_H_
