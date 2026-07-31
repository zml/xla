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

#include "xla/stream_executor/musa/musa_graph_arguments.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"

namespace stream_executor::musa {

absl::StatusOr<std::unique_ptr<KernelArgsPackedVector>>
CloneMusaGraphKernelArguments(const KernelArgsPackedArrayBase& arguments) {
  if (const auto* packed_vector =
          dynamic_cast<const KernelArgsPackedVector*>(&arguments)) {
    return std::make_unique<KernelArgsPackedVector>(
        packed_vector->argument_storage(), arguments.number_of_shared_bytes());
  }
  const auto* packable = dynamic_cast<const PackableKernelArgs*>(&arguments);
  if (packable == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require packed arguments with discoverable byte sizes");
  }

  absl::Span<const void* const> addresses = arguments.argument_addresses();
  absl::Span<const std::unique_ptr<PackedArgBase>> packed =
      packable->packed_args();
  if (addresses.size() != packed.size()) {
    return absl::InvalidArgumentError(
        "MUSA graph packed argument addresses and storage sizes differ");
  }

  std::vector<std::vector<char>> storage;
  storage.reserve(addresses.size());
  for (size_t index = 0; index < addresses.size(); ++index) {
    if (addresses[index] == nullptr || packed[index] == nullptr ||
        packed[index]->size() <= 0) {
      return absl::InvalidArgumentError(
          "MUSA graph kernel arguments must have non-empty storage");
    }
    std::vector<char> bytes(static_cast<size_t>(packed[index]->size()));
    std::memcpy(bytes.data(), addresses[index], bytes.size());
    storage.push_back(std::move(bytes));
  }
  return std::make_unique<KernelArgsPackedVector>(
      std::move(storage), arguments.number_of_shared_bytes());
}

}  // namespace stream_executor::musa
