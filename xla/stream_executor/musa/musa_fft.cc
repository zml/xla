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

#include "xla/stream_executor/musa/musa_fft.h"

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/musa/mufft_shim/mufft_shim_abi.h"
#include "xla/stream_executor/musa/musa_mufft_api.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {
namespace {

absl::StatusOr<XlaMusaMuFftType> ToMuFftType(fft::Type type) {
  switch (type) {
    case fft::Type::kC2CForward:
    case fft::Type::kC2CInverse:
      return XLA_MUSA_MUFFT_TYPE_C2C;
    case fft::Type::kR2C:
      return XLA_MUSA_MUFFT_TYPE_R2C;
    case fft::Type::kC2R:
      return XLA_MUSA_MUFFT_TYPE_C2R;
    case fft::Type::kZ2ZForward:
    case fft::Type::kZ2ZInverse:
      return XLA_MUSA_MUFFT_TYPE_Z2Z;
    case fft::Type::kD2Z:
      return XLA_MUSA_MUFFT_TYPE_D2Z;
    case fft::Type::kZ2D:
      return XLA_MUSA_MUFFT_TYPE_Z2D;
    default:
      return absl::InvalidArgumentError("invalid StreamExecutor FFT type");
  }
}

absl::StatusOr<XlaMusaMuFftDirection> ToMuFftDirection(fft::Type type) {
  switch (type) {
    case fft::Type::kC2CForward:
    case fft::Type::kZ2ZForward:
      return XLA_MUSA_MUFFT_DIRECTION_FORWARD;
    case fft::Type::kC2CInverse:
    case fft::Type::kZ2ZInverse:
      return XLA_MUSA_MUFFT_DIRECTION_INVERSE;
    default:
      return absl::InvalidArgumentError(
          "FFT direction is defined only for complex transforms");
  }
}

bool IsCompatiblePlanType(fft::Type actual, fft::Type expected) {
  if (actual == expected) return true;
  const bool actual_c2c =
      actual == fft::Type::kC2CForward || actual == fft::Type::kC2CInverse;
  const bool expected_c2c =
      expected == fft::Type::kC2CForward || expected == fft::Type::kC2CInverse;
  const bool actual_z2z =
      actual == fft::Type::kZ2ZForward || actual == fft::Type::kZ2ZInverse;
  const bool expected_z2z =
      expected == fft::Type::kZ2ZForward || expected == fft::Type::kZ2ZInverse;
  return (actual_c2c && expected_c2c) || (actual_z2z && expected_z2z);
}

}  // namespace

MusaFftPlan::MusaFftPlan(StreamExecutor* parent, MusaMuFftApi* api)
    : parent_(parent), api_(api) {}

MusaFftPlan::~MusaFftPlan() {
  if (handle_ == nullptr || api_ == nullptr || parent_ == nullptr) return;
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::Status status = api_->Destroy(handle_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy muFFT plan: " << status;
  }
}

absl::Status MusaFftPlan::Initialize(
    Stream* stream, int rank, const uint64_t* element_count,
    const uint64_t* input_embed, uint64_t input_stride, uint64_t input_distance,
    const uint64_t* output_embed, uint64_t output_stride,
    uint64_t output_distance, fft::Type type, int batch_count,
    ScratchAllocator* scratch_allocator) {
  if (initialized_) {
    return status_ = absl::FailedPreconditionError(
               "muFFT plan can be initialized only once");
  }
  initialized_ = true;
  if (parent_ == nullptr || api_ == nullptr || stream == nullptr) {
    return status_ = absl::InvalidArgumentError(
               "muFFT plan requires non-null executor, API, and stream");
  }
  if (rank < 1 || rank > 3 || element_count == nullptr) {
    return status_ = absl::InvalidArgumentError(
               "muFFT plan rank must be 1, 2, or 3 with dimensions");
  }
  if (batch_count <= 0) {
    return status_ =
               absl::InvalidArgumentError("muFFT batch count must be positive");
  }
  if (scratch_allocator == nullptr) {
    return status_ = absl::InvalidArgumentError(
               "muFFT requires a caller-owned scratch allocator");
  }
  absl::StatusOr<XlaMusaMuFftType> mufft_type = ToMuFftType(type);
  if (!mufft_type.ok()) return status_ = mufft_type.status();

  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  status_ = api_->Init();
  if (!status_.ok()) return status_;
  status_ = api_->Create(&handle_);
  if (!status_.ok()) return status_;
  status_ = api_->MakePlanMany(
      handle_, rank, element_count, input_embed, input_stride, input_distance,
      output_embed, output_stride, output_distance, *mufft_type,
      static_cast<uint64_t>(batch_count), &scratch_size_bytes_);
  if (!status_.ok()) return status_;
  type_ = type;
  return status_ = UpdateScratchAllocator(stream, scratch_allocator);
}

absl::Status MusaFftPlan::UpdateScratchAllocator(
    Stream* stream, ScratchAllocator* scratch_allocator) {
  if (handle_ == nullptr || stream == nullptr || scratch_allocator == nullptr) {
    return status_ = absl::InvalidArgumentError(
               "muFFT scratch update requires a plan, stream, and allocator");
  }
  if (scratch_size_bytes_ >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return status_ = absl::OutOfRangeError(
               "muFFT workspace size exceeds ScratchAllocator range");
  }

  DeviceAddress<uint8_t> scratch;
  if (scratch_size_bytes_ != 0) {
    absl::StatusOr<DeviceAddress<uint8_t>> allocated =
        scratch_allocator->AllocateBytes(
            static_cast<int64_t>(scratch_size_bytes_));
    if (!allocated.ok()) return status_ = allocated.status();
    if (allocated->is_null()) {
      return status_ = absl::ResourceExhaustedError(
                 "muFFT scratch allocator returned a null workspace");
    }
    scratch = *allocated;
  }

  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  status_ = api_->SetWorkArea(handle_, scratch.opaque());
  if (!status_.ok()) return status_;
  scratch_ = scratch;
  scratch_allocator_ = scratch_allocator;
  return status_ = absl::OkStatus();
}

std::unique_ptr<fft::Plan> MusaFft::CreateBatchedPlanWithScratchAllocator(
    Stream* stream, int rank, uint64_t* element_count, uint64_t* input_embed,
    uint64_t input_stride, uint64_t input_distance, uint64_t* output_embed,
    uint64_t output_stride, uint64_t output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator* scratch_allocator) {
  if (in_place_fft) {
    LOG(ERROR) << "In-place muFFT plans are not qualified by the MUSA PJRT "
                  "backend";
    return nullptr;
  }
  auto plan = std::make_unique<MusaFftPlan>(parent_, api_);
  absl::Status status =
      plan->Initialize(stream, rank, element_count, input_embed, input_stride,
                       input_distance, output_embed, output_stride,
                       output_distance, type, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to create muFFT plan: " << status;
    return nullptr;
  }
  return plan;
}

void MusaFft::UpdatePlanWithScratchAllocator(
    Stream* stream, fft::Plan* plan, ScratchAllocator* scratch_allocator) {
  auto* musa_plan = dynamic_cast<MusaFftPlan*>(plan);
  if (musa_plan == nullptr) {
    LOG(ERROR) << "Cannot update scratch for a non-MUSA FFT plan";
    return;
  }
  absl::Status status =
      musa_plan->UpdateScratchAllocator(stream, scratch_allocator);
  if (!status.ok()) {
    // FftSupport's legacy interface cannot return this status. Store it on the
    // plan and let the following DoFft fail without terminating the process.
    LOG(ERROR) << "Failed to update muFFT workspace: " << status;
  }
}

bool MusaFft::PrepareExecution(Stream* stream, MusaFftPlan* plan,
                               const DeviceAddressBase& input,
                               DeviceAddressBase* output) {
  if (stream == nullptr || plan == nullptr || output == nullptr ||
      input.is_null() || output->is_null()) {
    LOG(ERROR) << "muFFT execution received a null stream, plan, or buffer";
    return false;
  }
  if (!plan->status().ok()) {
    LOG(ERROR) << "muFFT plan is not executable: " << plan->status();
    return false;
  }
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::Status status = api_->SetStream(
      plan->handle(), stream->platform_specific_handle().stream);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to bind muFFT plan to stream: " << status;
    return false;
  }
  return true;
}

template <typename InputT, typename OutputT>
bool MusaFft::Execute(Stream* stream, fft::Plan* plan,
                      const DeviceAddress<InputT>& input,
                      DeviceAddress<OutputT>* output, fft::Type expected_type,
                      bool preserve_input) {
  auto* musa_plan = dynamic_cast<MusaFftPlan*>(plan);
  if (musa_plan == nullptr) {
    LOG(ERROR) << "The supplied FFT plan is not a MUSA FFT plan";
    return false;
  }
  if (!IsCompatiblePlanType(musa_plan->type(), expected_type)) {
    LOG(ERROR) << "The MUSA FFT plan type does not match the requested "
                  "operation";
    return false;
  }
  DeviceAddressBase output_base =
      output == nullptr ? DeviceAddressBase() : DeviceAddressBase(*output);
  if (!PrepareExecution(stream, musa_plan, input, &output_base)) return false;

  DeviceAddress<InputT> execution_input = input;
  if (preserve_input && input.opaque() != output->opaque() &&
      input.size() != 0) {
    ScratchAllocator* allocator = musa_plan->scratch_allocator();
    if (allocator == nullptr ||
        input.size() >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      LOG(ERROR) << "Unable to preserve muFFT input without a valid scratch "
                    "allocator";
      return false;
    }
    absl::StatusOr<DeviceAddress<uint8_t>> copy =
        allocator->AllocateBytes(static_cast<int64_t>(input.size()));
    if (!copy.ok() || copy->is_null()) {
      LOG(ERROR) << "Failed to allocate muFFT input-preservation buffer: "
                 << (copy.ok() ? "null buffer" : copy.status().ToString());
      return false;
    }
    absl::Status copy_status = stream->Memcpy(&*copy, input, input.size());
    if (!copy_status.ok()) {
      LOG(ERROR) << "Failed to preserve muFFT input: " << copy_status;
      return false;
    }
    execution_input = DeviceAddress<InputT>(*copy);
  }

  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::Status status;
  switch (expected_type) {
    case fft::Type::kC2CForward:
    case fft::Type::kC2CInverse: {
      absl::StatusOr<XlaMusaMuFftDirection> direction =
          ToMuFftDirection(musa_plan->type());
      if (!direction.ok()) {
        status = direction.status();
      } else {
        status = api_->ExecC2C(musa_plan->handle(), execution_input.opaque(),
                               output->opaque(), *direction);
      }
      break;
    }
    case fft::Type::kR2C:
      status = api_->ExecR2C(musa_plan->handle(), execution_input.opaque(),
                             output->opaque());
      break;
    case fft::Type::kC2R:
      status = api_->ExecC2R(musa_plan->handle(), execution_input.opaque(),
                             output->opaque());
      break;
    case fft::Type::kZ2ZForward:
    case fft::Type::kZ2ZInverse: {
      absl::StatusOr<XlaMusaMuFftDirection> direction =
          ToMuFftDirection(musa_plan->type());
      if (!direction.ok()) {
        status = direction.status();
      } else {
        status = api_->ExecZ2Z(musa_plan->handle(), execution_input.opaque(),
                               output->opaque(), *direction);
      }
      break;
    }
    case fft::Type::kD2Z:
      status = api_->ExecD2Z(musa_plan->handle(), execution_input.opaque(),
                             output->opaque());
      break;
    case fft::Type::kZ2D:
      status = api_->ExecZ2D(musa_plan->handle(), execution_input.opaque(),
                             output->opaque());
      break;
    default:
      status = absl::InvalidArgumentError("unsupported MUSA FFT operation");
  }
  if (!status.ok()) {
    LOG(ERROR) << "muFFT execution failed: " << status;
    return false;
  }
  return true;
}

bool MusaFft::DoFft(Stream* stream, fft::Plan* plan,
                    const DeviceAddress<std::complex<float>>& input,
                    DeviceAddress<std::complex<float>>* output) {
  auto* musa_plan = dynamic_cast<MusaFftPlan*>(plan);
  if (musa_plan == nullptr || (musa_plan->type() != fft::Type::kC2CForward &&
                               musa_plan->type() != fft::Type::kC2CInverse)) {
    LOG(ERROR) << "A complex-f32 FFT requires a C2C MUSA plan";
    return false;
  }
  return Execute(stream, plan, input, output, musa_plan->type(),
                 /*preserve_input=*/false);
}

bool MusaFft::DoFft(Stream* stream, fft::Plan* plan,
                    const DeviceAddress<std::complex<double>>& input,
                    DeviceAddress<std::complex<double>>* output) {
  auto* musa_plan = dynamic_cast<MusaFftPlan*>(plan);
  if (musa_plan == nullptr || (musa_plan->type() != fft::Type::kZ2ZForward &&
                               musa_plan->type() != fft::Type::kZ2ZInverse)) {
    LOG(ERROR) << "A complex-f64 FFT requires a Z2Z MUSA plan";
    return false;
  }
  return Execute(stream, plan, input, output, musa_plan->type(),
                 /*preserve_input=*/false);
}

bool MusaFft::DoFft(Stream* stream, fft::Plan* plan,
                    const DeviceAddress<float>& input,
                    DeviceAddress<std::complex<float>>* output) {
  return Execute(stream, plan, input, output, fft::Type::kR2C,
                 /*preserve_input=*/true);
}

bool MusaFft::DoFft(Stream* stream, fft::Plan* plan,
                    const DeviceAddress<double>& input,
                    DeviceAddress<std::complex<double>>* output) {
  return Execute(stream, plan, input, output, fft::Type::kD2Z,
                 /*preserve_input=*/true);
}

bool MusaFft::DoFft(Stream* stream, fft::Plan* plan,
                    const DeviceAddress<std::complex<float>>& input,
                    DeviceAddress<float>* output) {
  return Execute(stream, plan, input, output, fft::Type::kC2R,
                 /*preserve_input=*/true);
}

bool MusaFft::DoFft(Stream* stream, fft::Plan* plan,
                    const DeviceAddress<std::complex<double>>& input,
                    DeviceAddress<double>* output) {
  return Execute(stream, plan, input, output, fft::Type::kZ2D,
                 /*preserve_input=*/true);
}

void InitializeMusaFft() {
  PluginRegistry* registry = PluginRegistry::Instance();
  if (registry->HasFactory(kMusaPlatformId, PluginKind::kFft)) return;
  absl::Status status = registry->RegisterFactory<PluginRegistry::FftFactory>(
      kMusaPlatformId, "muFFT", [](StreamExecutor* parent) -> fft::FftSupport* {
        return new MusaFft(parent);
      });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register muFFT factory: " << status;
  }
}

}  // namespace stream_executor::musa

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_mufft, {
  stream_executor::musa::InitializeMusaFft();
});
