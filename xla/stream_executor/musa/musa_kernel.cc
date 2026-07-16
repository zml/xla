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
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/musa/musa_module.h"
#include "xla/stream_executor/musa/musa_module_reaper.h"
#include "xla/stream_executor/musa/musa_module_use_tracker.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

MusaKernel::MusaKernel(StreamExecutor* executor, MUfunction function,
                       std::shared_ptr<MusaModule> module,
                       MusaModuleReaper* module_reaper, unsigned arity)
    : executor_(executor),
      function_(function),
      module_(std::move(module)),
      module_reaper_(module_reaper),
      arity_(arity) {
  CHECK(executor_ != nullptr);
  CHECK(function_ != nullptr);
  CHECK(module_ != nullptr);
  CHECK(module_reaper_ != nullptr);
}

MusaKernel::~MusaKernel() { module_reaper_->RetireModule(std::move(module_)); }

absl::StatusOr<int32_t> MusaKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  if (threads.x == 0 || threads.y == 0 || threads.z == 0) {
    return absl::InvalidArgumentError(
        "MUSA occupancy thread dimensions must be nonzero");
  }
  if (threads.x > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) /
                      threads.y ||
      threads.x * threads.y >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) /
              threads.z) {
    return absl::InvalidArgumentError(
        "MUSA occupancy thread dimensions overflow int32");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return module_->driver()->MaxActiveBlocksPerMultiprocessor(
      function_, static_cast<int>(threads.x * threads.y * threads.z),
      dynamic_shared_memory_bytes);
}

absl::StatusOr<KernelMetadata> MusaKernel::GetKernelMetadata() const {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  KernelMetadata metadata;
  TF_ASSIGN_OR_RETURN(int registers,
                      module_->driver()->FunctionAttribute(
                          function_, MU_FUNC_ATTRIBUTE_NUM_REGS));
  metadata.set_registers_per_thread(registers);
  TF_ASSIGN_OR_RETURN(int shared_memory,
                      module_->driver()->FunctionAttribute(
                          function_, MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
  metadata.set_shared_memory_bytes(shared_memory);
  return metadata;
}

absl::Status MusaKernel::Launch(const ThreadDim& thread_dims,
                                const BlockDim& block_dims,
                                const std::optional<ClusterDim>& cluster_dims,
                                Stream* stream, const KernelArgs& args) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MUSA kernel %s stream must not be null", name()));
  }
  if (stream->parent() != executor_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MUSA kernel %s cannot launch on a stream from another executor",
        name()));
  }
  auto* module_use_tracker = dynamic_cast<MusaModuleUseTracker*>(stream);
  if (module_use_tracker == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MUSA kernel %s requires a stream with module lifetime tracking",
        name()));
  }

  auto launch = [this, stream, &thread_dims, &block_dims, &cluster_dims,
                 module_use_tracker](
                    const KernelArgsPackedArrayBase& packed) -> absl::Status {
    const size_t expected_arguments =
        Arity() + (packed.number_of_shared_bytes() > 0 ? 1 : 0);
    if (packed.number_of_arguments() != expected_arguments ||
        packed.argument_addresses().size() != Arity()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MUSA kernel %s received %d arguments (%d launch parameters), "
          "expected arity %d with %d shared-memory argument",
          name(), packed.number_of_arguments(),
          packed.argument_addresses().size(), Arity(),
          packed.number_of_shared_bytes() > 0 ? 1 : 0));
    }
    if (packed.number_of_shared_bytes() >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MUSA kernel %s dynamic shared-memory size exceeds int64", name()));
    }
    const uint64_t static_shared_memory =
        metadata().shared_memory_bytes().value_or(0);
    const uint64_t shared_memory_limit = static_cast<uint64_t>(
        executor_->GetDeviceDescription().shared_memory_per_block());
    if (static_shared_memory > shared_memory_limit ||
        packed.number_of_shared_bytes() >
            shared_memory_limit - static_shared_memory) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MUSA kernel %s requires %d static + %d dynamic shared-memory "
          "bytes, exceeding the %d-byte device limit",
          name(), static_shared_memory, packed.number_of_shared_bytes(),
          shared_memory_limit));
    }

    void** parameters = const_cast<void**>(packed.argument_addresses().data());
    absl::Status launch_status = stream->LaunchKernel(
        thread_dims, block_dims, cluster_dims, function_, name(), parameters,
        packed.number_of_shared_bytes(), use_pdl());
    if (!launch_status.ok()) {
      // Driver launch errors can be asynchronous or otherwise ambiguous. Keep
      // the module until the owning context is successfully synchronized.
      module_use_tracker->OrphanModuleUse(module_);
      return launch_status;
    }
    absl::Status tracking_status = module_use_tracker->RecordModuleUse(module_);
    if (!tracking_status.ok()) {
      return absl::Status(
          tracking_status.code(),
          absl::StrFormat(
              "MUSA kernel %s launched but its completion callback could not "
              "be recorded: %s",
              name(), tracking_status.message()));
    }
    return tracking_status;
  };

  if (const auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    const auto& pack = args_packing();
    if (!pack) return launch(*packed);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<KernelArgsPackedArrayBase> repacked,
                        pack(*this, *packed));
    return launch(*repacked);
  }

  if (const auto* device_addresses =
          DynCast<KernelArgsDeviceAddressArray>(&args)) {
    const auto& pack = args_packing();
    if (!pack) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MUSA kernel %s requires a custom packing function for a device "
          "address array",
          name()));
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<KernelArgsPackedArrayBase> packed,
                        pack(*this, *device_addresses));
    return launch(*packed);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("MUSA kernel %s received unsupported arguments", name()));
}

}  // namespace stream_executor::musa
