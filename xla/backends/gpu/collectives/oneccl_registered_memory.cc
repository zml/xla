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

#include "xla/backends/gpu/collectives/oneccl_registered_memory.h"

#include <exception>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "oneapi/ccl.hpp"
#include "xla/stream_executor/device_address.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

absl::Status OnecclExceptionStatus(const char* expr,
                                   const std::exception& error) {
  return Internal("oneCCL call failed: %s: %s", expr, error.what());
}

}  // namespace

OnecclRegisteredMemory::OnecclRegisteredMemory(
    const ccl::communicator& comm, void* handle,
    stream_executor::DeviceAddressBase addr)
    : comm_(&comm), handle_(handle), addr_(addr) {}

absl::StatusOr<std::unique_ptr<OnecclRegisteredMemory>>
OnecclRegisteredMemory::Create(const ccl::communicator& comm,
                               stream_executor::DeviceAddressBase addr) {
  VLOG(3) << absl::StrFormat(
      "Create oneCCL registered memory on comm=%p from: ptr=%p; size=%ld",
      &comm, addr.opaque(), addr.size());

  void* handle = nullptr;
  try {
    ccl::comm_register(comm, addr.opaque(), addr.size(), &handle);
  } catch (const std::exception& e) {
    return OnecclExceptionStatus("ccl::comm_register", e);
  }

  return absl::WrapUnique(new OnecclRegisteredMemory(comm, handle, addr));
}

OnecclRegisteredMemory::~OnecclRegisteredMemory() {
  VLOG(3) << absl::StrFormat("Destroy %v", *this);
  try {
    ccl::comm_deregister(*comm_, handle_);
  } catch (const std::exception& e) {
    LOG(ERROR) << OnecclExceptionStatus("ccl::comm_deregister", e);
  }
}

stream_executor::DeviceAddressBase OnecclRegisteredMemory::addr() const {
  return addr_;
}

std::string OnecclRegisteredMemory::ToString() const {
  return absl::StrFormat(
      "OnecclRegisteredMemory(comm=%p, handle=%p, ptr=%p, size=%ld)", comm_,
      handle_, addr_.opaque(), addr_.size());
}

}  // namespace xla::gpu
