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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "musa.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor {

class Stream;
class StreamExecutor;

namespace musa {

class MusaModule;

// Non-owning native function paired with strong ownership of its parent
// module. Function lookup and launch are intentionally added in C04; this C03
// shell freezes the lifetime relationship without claiming execution support.
class MusaKernel final : public Kernel {
 public:
  MusaKernel(StreamExecutor* executor, MUfunction function,
             std::shared_ptr<MusaModule> module, unsigned arity);

  unsigned Arity() const override { return arity_; }
  MUfunction function() const { return function_; }
  const std::shared_ptr<MusaModule>& module() const { return module_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      const std::optional<ClusterDim>& cluster_dims,
                      Stream* stream, const KernelArgs& args) override;

 private:
  StreamExecutor* const executor_;
  const MUfunction function_;
  const std::shared_ptr<MusaModule> module_;
  const unsigned arity_;
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_
