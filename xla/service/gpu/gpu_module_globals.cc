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

#include "xla/service/gpu/gpu_module_globals.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla::gpu {

GpuExecutableProto::ConstantInfoProto GpuModuleGlobals::ConstantInfo::ToProto(
    bool skip_content_serialization) const {
  GpuExecutableProto::ConstantInfoProto proto;
  proto.set_symbol_name(symbol_name);
  if (!skip_content_serialization) {
    *proto.mutable_content() = content.ToProto();
  }
  proto.set_allocation_index(allocation_index);
  return proto;
}

absl::StatusOr<GpuModuleGlobals::ConstantInfo>
GpuModuleGlobals::ConstantInfo::FromProto(
    const GpuExecutableProto::ConstantInfoProto& proto,
    const absl::flat_hash_map<std::string, const HloInstruction*>* absl_nullable
        content_overrides) {
  if (content_overrides) {
    auto it = content_overrides->find(proto.symbol_name());
    if (it == content_overrides->end()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Instruction for ", proto.symbol_name(), " constant missing."));
    }
    const HloInstruction* instr = it->second;
    ASSIGN_OR_RETURN(DenseDataIntermediate content,
                     LiteralToXlaFormat(instr->literal()));
    return ConstantInfo{proto.symbol_name(), content,
                        static_cast<int>(proto.allocation_index())};
  }
  return ConstantInfo{proto.symbol_name(),
                      DenseDataIntermediate::FromProto(proto.content()),
                      static_cast<int>(proto.allocation_index())};
}

absl::StatusOr<const GpuModuleGlobals::BufferAllocToDeviceMemoryMap*>
GpuModuleGlobals::Resolve(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();
  const bool is_musa =
      executor->GetPlatform()->id() == se::musa::kMusaPlatformId;

  absl::MutexLock lock(mutex_);
  auto it = globals_.find(executor);
  if (it != globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary_.empty()) {
    if (is_musa) {
      module_spec.AddMusaMubinInMemory(binary_);
    } else {
      module_spec.AddCudaCubinInMemory(binary_);
    }
  }

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // An empty binary has no module to load on any GPU platform. If the module
  // isn't loaded, all symbol lookups fail, just as they should for an empty
  // module.
  if (!binary_.empty()) {
    ASSIGN_OR_RETURN(module_handle, executor->LoadModule(module_spec));
  }
  se::ScopedModuleHandle scoped_module(executor, module_handle);

  struct ResolvedConstant {
    const ConstantInfo* info;
    se::DeviceAddressBase global;
  };
  std::vector<ResolvedConstant> resolved_constants;
  resolved_constants.reserve(constants_.size());

  // Resolve and validate every symbol before submitting any asynchronous
  // initializer copy. A later missing or undersized symbol must not race an
  // early-return module unload with work already queued on the stream.
  for (const ConstantInfo& info : constants_) {
    if (!static_cast<bool>(module_handle)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Cannot resolve GPU global %s without a loaded module",
          info.symbol_name));
    }
    absl::StatusOr<se::DeviceAddressBase> global_status =
        executor->GetSymbol(info.symbol_name, module_handle);
    if (!global_status.ok()) {
      return absl::Status(
          global_status.status().code(),
          absl::StrFormat("Failed to resolve GPU global %s: %s",
                          info.symbol_name, global_status.status().message()));
    }
    // The constant was defined in the device binary and has been allocated by
    // the platform driver.
    se::DeviceAddressBase global = *global_status;
    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "Resolved global %s to %p", info.symbol_name, global.opaque());

    if (!info.content.span().empty()) {
      if (info.content.span().size() > global.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Initializer for GPU global %s has %d bytes, exceeding the "
            "%d-byte symbol",
            info.symbol_name, info.content.span().size(), global.size()));
      }
    }

    resolved_constants.push_back(ResolvedConstant{&info, global});
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  bool submitted_mem_copies = false;
  for (const ResolvedConstant& resolved : resolved_constants) {
    const ConstantInfo& info = *resolved.info;
    se::DeviceAddressBase global = resolved.global;

    if (!info.content.span().empty()) {
      // This means the constant did not have an initializer in the device
      // binary and therefore must be initialized by XLA here.
      absl::Status copy_status = stream->Memcpy(
          &global, info.content.span().data(), info.content.span().size());
      if (!copy_status.ok()) {
        absl::Status sync_status = stream->BlockHostUntilDone();
        if (!sync_status.ok()) {
          // MusaExecutor explicitly owns released handles until synchronized
          // teardown. CUDA and ROCm do not expose equivalent retention and the
          // historical contract is fail-fast rather than unloading a module
          // while an initializer copy may still be in flight.
          if (!is_musa) {
            CHECK_OK(sync_status);
            return sync_status;
          }
          scoped_module.release();
          return absl::Status(
              sync_status.code(),
              absl::StrFormat(
                  "Failed to initialize GPU global %s (%s), and stream "
                  "quiescence could not be established (%s)%s",
                  info.symbol_name, copy_status.message(),
                  sync_status.message(),
                  is_musa ? "; retaining the MUSA module until executor "
                            "teardown"
                          : ""));
        }
        return absl::Status(
            copy_status.code(),
            absl::StrFormat("Failed to initialize GPU global %s: %s",
                            info.symbol_name, copy_status.message()));
      }
      submitted_mem_copies = true;
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    absl::Status status = stream->BlockHostUntilDone();
    if (!status.ok()) {
      if (!is_musa) {
        CHECK_OK(status);
        return status;
      }
      scoped_module.release();
      return absl::Status(
          status.code(),
          absl::StrFormat(
              "Failed to initialize GPU module globals and could not prove "
              "stream quiescence: %s%s",
              status.message(),
              is_musa ? "; retaining the MUSA module until executor teardown"
                      : ""));
    }
  }

  module_handles_.emplace(executor, std::move(scoped_module));
  return globals_.emplace(executor, std::move(globals)).first->second.get();
}

}  // namespace xla::gpu
