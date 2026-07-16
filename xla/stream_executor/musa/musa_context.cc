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

#include "xla/stream_executor/musa/musa_context.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/musa/musa_driver.h"

namespace stream_executor::musa {

absl::StatusOr<std::unique_ptr<MusaContext>> MusaContext::Create(
    int device_ordinal, MusaDriver* driver) {
  if (driver == nullptr) {
    return absl::InvalidArgumentError("MUSA driver must not be null");
  }

  RETURN_IF_ERROR(driver->Init());
  TF_ASSIGN_OR_RETURN(MUdevice device, driver->Device(device_ordinal));
  TF_ASSIGN_OR_RETURN(MusaPrimaryContextState state,
                      driver->PrimaryContextState(device));

  constexpr unsigned int kDesiredFlags = MU_CTX_SCHED_AUTO;
  if (state.flags != kDesiredFlags) {
    if (state.active) {
      LOG(WARNING) << "MUSA primary context for device " << device_ordinal
                   << " is already active with flags " << state.flags
                   << "; reusing it instead of applying requested flags "
                   << kDesiredFlags;
    } else {
      RETURN_IF_ERROR(driver->SetPrimaryContextFlags(device, kDesiredFlags));
    }
  }

  TF_ASSIGN_OR_RETURN(MUcontext context, driver->RetainPrimaryContext(device));
  if (context == nullptr) {
    // A broken null result provides no reliable evidence that the driver
    // actually acquired a reference, so do not risk releasing another user's
    // primary-context reference here.
    return absl::InternalError(absl::StrFormat(
        "MUSA primary-context retain returned null for device %d",
        device_ordinal));
  }

  return std::unique_ptr<MusaContext>(
      new MusaContext(device_ordinal, device, context, driver));
}

MusaContext::~MusaContext() {
  absl::StatusOr<MUcontext> current = driver_->CurrentContext();
  if (!current.ok()) {
    LOG(ERROR) << "Failed to query current MUSA context during teardown: "
               << current.status();
  } else if (*current == context_) {
    absl::Status status = driver_->SetCurrentContext(nullptr);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to clear current MUSA context during teardown: "
                 << status;
    }
  }

  absl::Status status = driver_->ReleasePrimaryContext(device_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to release MUSA primary context for device "
               << device_ordinal_ << ": " << status;
  }
}

void MusaContext::SetActive() {
  CHECK_OK(driver_->SetCurrentContext(context_));
}

bool MusaContext::IsActive() const {
  absl::StatusOr<MUcontext> current = driver_->CurrentContext();
  return current.ok() && *current == context_;
}

absl::Status MusaContext::Synchronize() {
  gpu::ScopedActivateContext activation(this);
  return driver_->SynchronizeContext();
}

}  // namespace stream_executor::musa
