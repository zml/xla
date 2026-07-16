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

#include "xla/stream_executor/musa/musa_kernel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/musa/musa_module.h"

namespace stream_executor::musa {

MusaKernel::MusaKernel(StreamExecutor* executor, MUfunction function,
                       std::shared_ptr<MusaModule> module, unsigned arity)
    : executor_(executor),
      function_(function),
      module_(std::move(module)),
      arity_(arity) {
  CHECK(executor_ != nullptr);
  CHECK(function_ != nullptr);
  CHECK(module_ != nullptr);
}

absl::StatusOr<int32_t> MusaKernel::GetMaxOccupiedBlocksPerCore(ThreadDim,
                                                                size_t) const {
  return absl::UnimplementedError(
      "MUSA kernel occupancy queries are implemented in C04");
}

absl::Status MusaKernel::Launch(const ThreadDim&, const BlockDim&,
                                const std::optional<ClusterDim>&, Stream*,
                                const KernelArgs&) {
  return absl::UnimplementedError("MUSA kernel launch is implemented in C04");
}

}  // namespace stream_executor::musa
