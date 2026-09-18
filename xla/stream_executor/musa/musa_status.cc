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

#include "xla/stream_executor/musa/musa_status.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "driver_types.h"
#include "musa.h"

namespace stream_executor::musa {

namespace {

absl::StatusCode RuntimeStatusCode(int result) {
  switch (static_cast<musaError_t>(result)) {
    case musaErrorInvalidValue:
    case musaErrorInvalidConfiguration:
    case musaErrorInvalidPitchValue:
    case musaErrorInvalidSymbol:
    case musaErrorInvalidHostPointer:
    case musaErrorInvalidDevicePointer:
    case musaErrorInvalidMemcpyDirection:
    case musaErrorInvalidDevice:
    case musaErrorInvalidResourceHandle:
      return absl::StatusCode::kInvalidArgument;

    case musaErrorMemoryAllocation:
    case musaErrorLaunchOutOfResources:
      return absl::StatusCode::kResourceExhausted;

    case musaErrorFileNotFound:
    case musaErrorSharedObjectSymbolNotFound:
    case musaErrorSymbolNotFound:
      return absl::StatusCode::kNotFound;

    case musaErrorInitializationError:
    case musaErrorMusartUnloading:
    case musaErrorStubLibrary:
    case musaErrorInsufficientDriver:
    case musaErrorNoDevice:
    case musaErrorDeviceUninitialized:
    case musaErrorIncompatibleDriverContext:
    case musaErrorContextIsDestroyed:
    case musaErrorSystemDriverMismatch:
      return absl::StatusCode::kFailedPrecondition;

    case musaErrorNotYetImplemented:
    case musaErrorUnsupportedLimit:
    case musaErrorPeerAccessUnsupported:
    case musaErrorNotSupported:
    case musaErrorCompatNotSupportedOnDevice:
      return absl::StatusCode::kUnimplemented;

    case musaErrorNotPermitted:
      return absl::StatusCode::kPermissionDenied;

    case musaErrorLaunchTimeout:
      return absl::StatusCode::kDeadlineExceeded;

    case musaErrorDevicesUnavailable:
    case musaErrorNotReady:
    case musaErrorSystemNotReady:
      return absl::StatusCode::kUnavailable;

    case musaErrorUnknown:
      return absl::StatusCode::kUnknown;

    default:
      return absl::StatusCode::kInternal;
  }
}

}  // namespace

absl::Status ToStatus(int result, absl::string_view expr,
                      absl::string_view error_string) {
  if (result == 0) {
    return absl::OkStatus();
  }
  if (error_string.empty()) {
    return absl::Status(RuntimeStatusCode(result),
                        absl::StrCat(expr, " failed with MUSA error ", result));
  }
  return absl::Status(RuntimeStatusCode(result),
                      absl::StrCat(expr, " failed with MUSA error ", result,
                                   ": ", error_string));
}

namespace {

absl::StatusCode DriverStatusCode(MUresult result) {
  switch (result) {
    case MUSA_ERROR_INVALID_VALUE:
    case MUSA_ERROR_INVALID_DEVICE:
    case MUSA_ERROR_INVALID_IMAGE:
    case MUSA_ERROR_INVALID_SOURCE:
    case MUSA_ERROR_INVALID_HANDLE:
      return absl::StatusCode::kInvalidArgument;

    case MUSA_ERROR_OUT_OF_MEMORY:
    case MUSA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    case MUSA_ERROR_TOO_MANY_PEERS:
      return absl::StatusCode::kResourceExhausted;

    case MUSA_ERROR_NO_BINARY_FOR_GPU:
    case MUSA_ERROR_FILE_NOT_FOUND:
    case MUSA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
    case MUSA_ERROR_NOT_FOUND:
      return absl::StatusCode::kNotFound;

    case MUSA_ERROR_NOT_INITIALIZED:
    case MUSA_ERROR_DEINITIALIZED:
    case MUSA_ERROR_NO_DEVICE:
    case MUSA_ERROR_INVALID_CONTEXT:
    case MUSA_ERROR_ILLEGAL_STATE:
    case MUSA_ERROR_PRIMARY_CONTEXT_ACTIVE:
    case MUSA_ERROR_CONTEXT_IS_DESTROYED:
    case MUSA_ERROR_SYSTEM_DRIVER_MISMATCH:
    case MUSA_ERROR_STUB_LIBRARY:
    case MUSA_ERROR_PEER_ACCESS_NOT_ENABLED:
      return absl::StatusCode::kFailedPrecondition;

    case MUSA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
      return absl::StatusCode::kAlreadyExists;

    case MUSA_ERROR_PEER_ACCESS_UNSUPPORTED:
    case MUSA_ERROR_NOT_SUPPORTED:
    case MUSA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
      return absl::StatusCode::kUnimplemented;

    case MUSA_ERROR_NOT_PERMITTED:
      return absl::StatusCode::kPermissionDenied;

    case MUSA_ERROR_LAUNCH_TIMEOUT:
    case MUSA_ERROR_TIMEOUT:
      return absl::StatusCode::kDeadlineExceeded;

    case MUSA_ERROR_SYSTEM_NOT_READY:
      return absl::StatusCode::kUnavailable;

    case MUSA_ERROR_UNKNOWN:
      return absl::StatusCode::kUnknown;

    default:
      return absl::StatusCode::kInternal;
  }
}

}  // namespace

absl::Status DriverToStatus(MUresult result, absl::string_view expr,
                            MusaDriverGetErrorNameFn get_error_name,
                            MusaDriverGetErrorStringFn get_error_string) {
  if (result == MUSA_SUCCESS) return absl::OkStatus();

  const char* name = nullptr;
  const char* description = nullptr;
  if (get_error_name != nullptr &&
      get_error_name(result, &name) != MUSA_SUCCESS) {
    name = nullptr;
  }
  if (get_error_string != nullptr &&
      get_error_string(result, &description) != MUSA_SUCCESS) {
    description = nullptr;
  }

  std::string message = absl::StrCat(expr, " failed: ");
  if (name == nullptr) {
    absl::StrAppend(&message, "UNKNOWN ERROR (", static_cast<int>(result), ")");
  } else {
    absl::StrAppend(&message, name, " (", static_cast<int>(result), ")");
  }
  if (description != nullptr) {
    absl::StrAppend(&message, ": ", description);
  }
  return absl::Status(DriverStatusCode(result), message);
}

}  // namespace stream_executor::musa
