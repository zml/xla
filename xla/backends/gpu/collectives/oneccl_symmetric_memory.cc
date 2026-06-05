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

#include "xla/backends/gpu/collectives/oneccl_symmetric_memory.h"

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

OnecclSymmetricMemory::OnecclSymmetricMemory(
    const ccl::communicator& comm, ccl::window window,
    stream_executor::DeviceAddressBase addr)
    : comm_(&comm),
      window_(std::make_unique<ccl::window>(std::move(window))),
      addr_(addr) {}

absl::StatusOr<std::unique_ptr<OnecclSymmetricMemory>>
OnecclSymmetricMemory::Create(const ccl::communicator& comm,
                              stream_executor::DeviceAddressBase addr) {
  VLOG(3) << absl::StrFormat(
      "Create oneCCL symmetric memory on comm=%p from: ptr=%p; size=%ld",
      &comm, addr.opaque(), addr.size());

  try {
    return absl::WrapUnique(new OnecclSymmetricMemory(
        comm,
        ccl::comm_window_register(comm, addr.opaque(), addr.size(),
                                  CCL_WIN_COLL_SYMMETRIC),
        addr));
  } catch (const std::exception& e) {
    return OnecclExceptionStatus("ccl::comm_window_register", e);
  }
}

OnecclSymmetricMemory::~OnecclSymmetricMemory() {
  VLOG(3) << absl::StrFormat("Destroy %v", *this);
  try {
    ccl::comm_window_deregister(*comm_, *window_);
  } catch (const std::exception& e) {
    LOG(ERROR) << OnecclExceptionStatus("ccl::comm_window_deregister", e);
  }
}

stream_executor::DeviceAddressBase OnecclSymmetricMemory::addr() const {
  return addr_;
}

std::string OnecclSymmetricMemory::ToString() const {
  return absl::StrFormat(
      "OnecclSymmetricMemory(comm=%p, window=%p, ptr=%p, size=%ld)", comm_,
      window_.get(), addr_.opaque(), addr_.size());
}

OnecclSymmetricMemory::PackedKernelArg OnecclSymmetricMemory::PackKernelArg()
    const {
  return window_.get();
}

}  // namespace xla::gpu
