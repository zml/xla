/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_ARGS_PACKED_VECTOR_H_
#define XLA_STREAM_EXECUTOR_KERNEL_ARGS_PACKED_VECTOR_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/stream_executor/kernel_args.h"

namespace stream_executor {

// An array of arguments for a kernel call, where each argument is stored in a
// separate vector. The storage is owned by the KernelArgsPackedVector.
class KernelArgsPackedVector : public KernelArgsPackedArrayBase {
 public:
  KernelArgsPackedVector(std::vector<std::vector<char>> arguments,
                         size_t shared_memory_bytes)
      : argument_storage_(std::move(arguments)),
        shared_memory_bytes_(shared_memory_bytes) {
    InitializeArgumentAddresses();
  }

  KernelArgsPackedVector(
      std::vector<std::vector<char>> arguments,
      std::vector<KernelArgumentMetadata> argument_metadata,
      size_t shared_memory_bytes)
      : argument_storage_(std::move(arguments)),
        argument_metadata_(std::move(argument_metadata)),
        shared_memory_bytes_(shared_memory_bytes) {
    if (argument_metadata_.size() == argument_storage_.size()) {
      for (size_t i = 0; i < argument_metadata_.size(); ++i) {
        argument_metadata_[i].size = argument_storage_[i].size();
      }
    }
    InitializeArgumentAddresses();
  }

  size_t number_of_arguments() const final {
    // We need to add 1 to the number of arguments if the kernel is using shared
    // memory because we treat the number of shared memory bytes like an
    // additional argument in the StreamExecutor kernel launch API. Note that
    // shared memory doesn't appear in
    // KernelArgsPackedVector::argument_addresses().
    return argument_storage_.size() + (shared_memory_bytes_ > 0);
  }

  // Returns the number of bytes that need to be allocated in shared memory for
  // this kernel. This value is treated like an additional argument in the CUDA
  // kernel launch API, therefore we pass it alongside the real kernel arguments
  // here.
  uint64_t number_of_shared_bytes() const final { return shared_memory_bytes_; }

  absl::Span<const void* const> argument_addresses() const final {
    return argument_addresses_;
  }

  absl::Span<const KernelArgumentMetadata> argument_metadata() const final {
    return argument_metadata_;
  }

 private:
  void InitializeArgumentAddresses() {
    argument_addresses_.reserve(argument_storage_.size());
    for (std::vector<char>& argument : argument_storage_) {
      argument_addresses_.push_back(argument.data());
    }
  }

  std::vector<std::vector<char>> argument_storage_;
  std::vector<KernelArgumentMetadata> argument_metadata_;
  size_t shared_memory_bytes_ = 0;
  std::vector<const void*> argument_addresses_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_ARGS_PACKED_VECTOR_H_
