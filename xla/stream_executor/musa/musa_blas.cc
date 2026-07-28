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

#include "xla/stream_executor/musa/musa_blas.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_abi.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {
namespace {

absl::StatusOr<XlaMusaMuBlasOperation> AsMuBlasOperation(
    blas::Transpose transpose) {
  switch (transpose) {
    case blas::Transpose::kNoTranspose:
      return XLA_MUSA_MUBLAS_OPERATION_NONE;
    case blas::Transpose::kTranspose:
      return XLA_MUSA_MUBLAS_OPERATION_TRANSPOSE;
    case blas::Transpose::kConjugateTranspose:
      return XLA_MUSA_MUBLAS_OPERATION_CONJUGATE_TRANSPOSE;
  }
  return absl::InvalidArgumentError("unknown BLAS transpose operation");
}

struct MuBlasGemmTypes {
  XlaMusaMuBlasDataType input_type;
  XlaMusaMuBlasDataType output_type;
  XlaMusaMuBlasComputeType compute_type;
  uint64_t capability;
};

absl::StatusOr<XlaMusaMuBlasDataType> AsMuBlasDataType(
    blas::DataType data_type) {
  switch (data_type) {
    case blas::DataType::kHalf:
      return XLA_MUSA_MUBLAS_DATA_TYPE_F16;
    case blas::DataType::kBF16:
      return XLA_MUSA_MUBLAS_DATA_TYPE_BF16;
    case blas::DataType::kFloat:
      return XLA_MUSA_MUBLAS_DATA_TYPE_F32;
    case blas::DataType::kDouble:
      return XLA_MUSA_MUBLAS_DATA_TYPE_F64;
    default:
      return absl::UnimplementedError(
          absl::StrCat("basic MUSA muBLAS GEMM does not support data type ",
                       blas::DataTypeString(data_type)));
  }
}

absl::StatusOr<XlaMusaMuBlasComputeType> AsMuBlasComputeType(
    blas::ComputationType compute_type) {
  switch (compute_type) {
    case blas::ComputationType::kF16:
      return XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16;
    case blas::ComputationType::kF32:
      return XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32;
    case blas::ComputationType::kF64:
      return XLA_MUSA_MUBLAS_COMPUTE_TYPE_F64;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "basic MUSA muBLAS GEMM does not support computation type ",
          blas::ComputationTypeString(compute_type)));
  }
}

absl::StatusOr<MuBlasGemmTypes> GetMuBlasGemmTypes(
    blas::DataType type_a, blas::DataType type_b, blas::DataType type_c,
    blas::ComputationType computation_type) {
  if (type_a != type_b) {
    return absl::UnimplementedError(
        "basic MUSA muBLAS GEMM requires identical input types");
  }
  ASSIGN_OR_RETURN(XlaMusaMuBlasDataType input_type, AsMuBlasDataType(type_a));
  ASSIGN_OR_RETURN(XlaMusaMuBlasDataType output_type, AsMuBlasDataType(type_c));
  ASSIGN_OR_RETURN(XlaMusaMuBlasComputeType compute_type,
                   AsMuBlasComputeType(computation_type));

  uint64_t capability = 0;
  bool supported = false;
  if (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F16) {
    capability = XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F16;
    supported = output_type == XLA_MUSA_MUBLAS_DATA_TYPE_F16 &&
                compute_type == XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16;
  } else if (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F32) {
    capability = XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F32;
    supported = output_type == XLA_MUSA_MUBLAS_DATA_TYPE_F32 &&
                compute_type == XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32;
  } else if (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F64) {
    capability = XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F64;
    supported = output_type == XLA_MUSA_MUBLAS_DATA_TYPE_F64 &&
                compute_type == XLA_MUSA_MUBLAS_COMPUTE_TYPE_F64;
  }
  if (!supported) {
    return absl::UnimplementedError(absl::StrCat(
        "basic MUSA muBLAS GEMM does not support ",
        blas::DataTypeString(type_a), " inputs, ", blas::DataTypeString(type_c),
        " output, and ", blas::ComputationTypeString(computation_type),
        " computation"));
  }
  return MuBlasGemmTypes{input_type, output_type, compute_type, capability};
}

absl::Status ValidateDimensions(blas::Transpose transa, blas::Transpose transb,
                                uint64_t m, uint64_t n, uint64_t k, int lda,
                                int ldb, int ldc) {
  constexpr uint64_t kMax =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (m > kMax || n > kMax || k > kMax) {
    return absl::OutOfRangeError(
        "muBLAS GEMM dimensions exceed the shim's signed 64-bit ABI");
  }
  if (lda <= 0 || ldb <= 0 || ldc <= 0) {
    return absl::InvalidArgumentError(
        "muBLAS GEMM leading dimensions must be positive");
  }
  const uint64_t min_lda = transa == blas::Transpose::kNoTranspose ? m : k;
  const uint64_t min_ldb = transb == blas::Transpose::kNoTranspose ? k : n;
  if (static_cast<uint64_t>(lda) < std::max<uint64_t>(1, min_lda) ||
      static_cast<uint64_t>(ldb) < std::max<uint64_t>(1, min_ldb) ||
      static_cast<uint64_t>(ldc) < std::max<uint64_t>(1, m)) {
    return absl::InvalidArgumentError(
        "muBLAS GEMM leading dimension is smaller than its stored row count");
  }
  return absl::OkStatus();
}

absl::Status ValidateBatch(int batch_count) {
  if (batch_count < 0) {
    return absl::InvalidArgumentError(
        "muBLAS GEMM batch count must be non-negative");
  }
  return absl::OkStatus();
}

absl::Status ValidateStrides(int64_t stride_a, int64_t stride_b,
                             int64_t stride_c, int batch_count) {
  if (stride_a < 0 || stride_b < 0 || stride_c < 0) {
    return absl::InvalidArgumentError(
        "muBLAS GEMM batch strides must be non-negative");
  }
  // Input stride zero is a useful and well-defined broadcast. Output batches
  // must not alias when more than one operation writes concurrently.
  if (batch_count > 1 && stride_c == 0) {
    return absl::InvalidArgumentError(
        "muBLAS GEMM output batch stride must be positive");
  }
  return absl::OkStatus();
}

absl::Status ValidateAlgorithm(blas::AlgorithmType algorithm,
                               const MuBlasGemmTypes& types,
                               const MusaMuBlasApi& api,
                               const EngineOptions& engine_options,
                               XlaMusaMuBlasAlgorithm* normalized) {
  if (normalized == nullptr) {
    return absl::InvalidArgumentError("normalized algorithm output is null");
  }
  if (algorithm == blas::kDefaultAlgorithm ||
      algorithm == blas::kDefaultBlasGemm ||
      algorithm == kMusaBlasDefaultAlgorithm) {
    *normalized = XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT;
    return absl::OkStatus();
  }
  if (algorithm != kMusaBlasTensorOpAlgorithm) {
    return absl::UnimplementedError(
        absl::StrCat("unknown MUSA muBLAS GEMM algorithm ", algorithm));
  }
  if (engine_options.require_determinism) {
    return absl::UnimplementedError(
        "deterministic MUSA GEMM supports only algorithm 0 (default)");
  }
  if (types.input_type != XLA_MUSA_MUBLAS_DATA_TYPE_F32 ||
      types.output_type != XLA_MUSA_MUBLAS_DATA_TYPE_F32 ||
      types.compute_type != XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32 ||
      !api.SupportsTensorOpF32()) {
    return absl::UnimplementedError(
        "MUSA muBLAS tensor-op algorithm is qualified only for homogeneous "
        "f32 GEMM with the v2 capability");
  }
  *normalized = XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP;
  return absl::OkStatus();
}

absl::Status PopulateProfileFromTimer(
    EventBasedTimer* timer, blas::AlgorithmType algorithm,
    blas::ProfileResult* output_profile_result) {
  if (output_profile_result == nullptr) return absl::OkStatus();
  if (timer == nullptr) {
    return absl::InternalError("muBLAS profiling timer is null");
  }
  ASSIGN_OR_RETURN(absl::Duration duration, timer->GetElapsedDuration());
  output_profile_result->set_is_valid(true);
  output_profile_result->set_algorithm(algorithm);
  output_profile_result->set_elapsed_time_in_ms(
      absl::ToDoubleMilliseconds(duration));
  return absl::OkStatus();
}

absl::Status ValidatePointerArraySizes(size_t a_size, size_t b_size,
                                       size_t c_size, int batch_count) {
  RETURN_IF_ERROR(ValidateBatch(batch_count));
  const size_t count = static_cast<size_t>(batch_count);
  if (a_size < count || b_size < count || c_size < count) {
    return absl::InvalidArgumentError(absl::StrCat(
        "muBLAS pointer-array slices must contain batch_count entries: a=",
        a_size, ", b=", b_size, ", c=", c_size, ", batch_count=", count));
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> RequiredMatrixElements(blas::Transpose transpose,
                                                uint64_t rows, uint64_t columns,
                                                int ld) {
  const uint64_t stored_rows =
      transpose == blas::Transpose::kNoTranspose ? rows : columns;
  const uint64_t stored_columns =
      transpose == blas::Transpose::kNoTranspose ? columns : rows;
  if (stored_rows == 0 || stored_columns == 0) return uint64_t{0};
  const uint64_t columns_minus_one = stored_columns - 1;
  if (columns_minus_one > (std::numeric_limits<uint64_t>::max() - stored_rows) /
                              static_cast<uint64_t>(ld)) {
    return absl::OutOfRangeError("muBLAS matrix extent overflows uint64");
  }
  return columns_minus_one * static_cast<uint64_t>(ld) + stored_rows;
}

absl::Status ValidateBufferExtent(const DeviceAddressBase& buffer,
                                  uint64_t matrix_elements, int64_t stride,
                                  int batch_count, size_t element_size,
                                  absl::string_view name) {
  if (buffer.is_null()) {
    return absl::InvalidArgumentError(
        absl::StrCat("muBLAS GEMM ", name, " matrix pointer is null"));
  }
  if (buffer.size() == 0) return absl::OkStatus();
  uint64_t required_elements = matrix_elements;
  if (batch_count > 1) {
    const uint64_t batches_minus_one = static_cast<uint64_t>(batch_count - 1);
    const uint64_t unsigned_stride = static_cast<uint64_t>(stride);
    if (unsigned_stride != 0 &&
        batches_minus_one >
            (std::numeric_limits<uint64_t>::max() - required_elements) /
                unsigned_stride) {
      return absl::OutOfRangeError(
          absl::StrCat("muBLAS GEMM ", name, " batch extent overflows"));
    }
    required_elements += batches_minus_one * unsigned_stride;
  }
  if (required_elements > std::numeric_limits<uint64_t>::max() / element_size) {
    return absl::OutOfRangeError(
        absl::StrCat("muBLAS GEMM ", name, " byte extent overflows"));
  }
  const uint64_t required_bytes = required_elements * element_size;
  if (buffer.size() < required_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("muBLAS GEMM ", name, " buffer has ", buffer.size(),
                     " bytes; needs at least ", required_bytes));
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> DataTypeSize(blas::DataType type) {
  switch (type) {
    case blas::DataType::kHalf:
    case blas::DataType::kBF16:
      return size_t{2};
    case blas::DataType::kFloat:
      return size_t{4};
    case blas::DataType::kDouble:
      return size_t{8};
    default:
      return absl::UnimplementedError(
          absl::StrCat("MUSA muBLAS has no qualified element size for ",
                       blas::DataTypeString(type)));
  }
}

}  // namespace

MusaBlas::MusaBlas(StreamExecutor* parent, MusaMuBlasApi* api)
    : parent_(parent), api_(api) {}

bool MusaBlas::Init() {
  if (parent_ == nullptr || api_ == nullptr) {
    LOG(ERROR) << "Cannot initialize muBLAS with a null executor or API";
    return false;
  }
  absl::Status load = api_->Init();
  if (!load.ok()) {
    LOG(ERROR) << "Unable to load optional muBLAS shim: " << load;
    return false;
  }
  absl::StatusOr<std::shared_ptr<HandleState>> handle =
      GetOrCreateHandleForNativeStream(nullptr);
  if (!handle.ok()) {
    LOG(ERROR) << "Unable to create main-stream muBLAS handle: "
               << handle.status();
    return false;
  }
  return true;
}

MusaBlas::~MusaBlas() {
  absl::flat_hash_map<void*, std::shared_ptr<HandleState>> handles;
  std::vector<std::shared_ptr<MemoryAllocation>> failed_host_staging;
  {
    absl::MutexLock lock(&handles_mu_);
    handles.swap(handles_);
    failed_host_staging.swap(failed_host_staging_);
  }
  if (parent_ == nullptr || api_ == nullptr) return;
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  for (auto& [native_stream, state] : handles) {
    absl::MutexLock state_lock(&state->mu);
    absl::Status status = api_->Destroy(state->handle);
    if (!status.ok()) {
      LOG(ERROR) << "Unable to destroy muBLAS handle for stream "
                 << native_stream << ": " << status;
    }
    state->handle = nullptr;
  }
}

void MusaBlas::NotifyStreamDestroyed(Stream* stream) {
  if (stream == nullptr || stream->parent() != parent_) return;
  void* native_stream = stream->platform_specific_handle().stream;
  std::shared_ptr<HandleState> state;
  {
    absl::MutexLock lock(&handles_mu_);
    auto handle = handles_.find(native_stream);
    if (handle == handles_.end()) return;
    state = std::move(handle->second);
    handles_.erase(handle);
    if (last_stream_ == native_stream) last_stream_ = nullptr;
  }

  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::MutexLock state_lock(&state->mu);
  if (state->handle == nullptr) return;
  absl::Status status = api_->Destroy(state->handle);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to destroy muBLAS handle for retired stream "
               << native_stream << ": " << status;
  }
  state->handle = nullptr;
}

void MusaBlas::RetainFailedHostStaging(
    std::shared_ptr<MemoryAllocation> host_staging) {
  absl::MutexLock lock(&handles_mu_);
  failed_host_staging_.push_back(std::move(host_staging));
}

absl::StatusOr<std::shared_ptr<MusaBlas::HandleState>>
MusaBlas::GetOrCreateHandle(Stream* stream) {
  if (stream == nullptr || stream->parent() != parent_) {
    return absl::InvalidArgumentError(
        "muBLAS stream is null or belongs to another executor");
  }
  return GetOrCreateHandleForNativeStream(
      stream->platform_specific_handle().stream);
}

absl::StatusOr<std::shared_ptr<MusaBlas::HandleState>>
MusaBlas::GetOrCreateHandleForNativeStream(void* native_stream) {
  absl::MutexLock lock(&handles_mu_);
  auto existing = handles_.find(native_stream);
  if (existing != handles_.end()) return existing->second;

  auto state = std::make_shared<HandleState>(native_stream);
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  void* handle = nullptr;
  RETURN_IF_ERROR(api_->Create(&handle));
  absl::Status set_stream = api_->SetStream(handle, native_stream);
  if (!set_stream.ok()) {
    absl::Status destroy = api_->Destroy(handle);
    if (!destroy.ok()) {
      LOG(ERROR) << "Unable to destroy muBLAS handle after set_stream failure: "
                 << destroy;
    }
    return set_stream;
  }
  {
    absl::MutexLock state_lock(&state->mu);
    state->handle = handle;
  }
  handles_.emplace(native_stream, state);
  return state;
}

absl::Status MusaBlas::ConfigureAtomics(HandleState* state,
                                        const EngineOptions& engine_options,
                                        XlaMusaMuBlasAlgorithm algorithm) {
  if (state == nullptr || state->handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is not initialized");
  }
  if (engine_options.require_determinism &&
      algorithm != XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT) {
    return absl::UnimplementedError(
        "deterministic MUSA GEMM supports only algorithm 0 (default)");
  }
  if (!api_->SupportsSetAtomicsMode()) {
    if (engine_options.require_determinism) {
      return absl::UnimplementedError(
          "deterministic MUSA GEMM requires the v2 atomics-mode contract");
    }
    return absl::OkStatus();
  }
  return api_->SetAtomicsMode(state->handle,
                              /*allow_atomics=*/
                              !engine_options.require_determinism);
}

absl::Status MusaBlas::Unsupported(absl::string_view operation) const {
  return absl::UnimplementedError(
      absl::StrCat("basic MUSA muBLAS adapter does not implement ", operation));
}

bool MusaBlas::UnsupportedBool(absl::string_view operation) const {
  LOG(ERROR) << Unsupported(operation);
  return false;
}

absl::Status MusaBlas::DoBlasGemmInternal(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, const DeviceAddressBase& b,
    blas::DataType type_b, int ldb, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const EngineOptions& engine_options,
    blas::ProfileResult* output_profile_result) {
  if (engine_options.require_command_buffer) {
    return Unsupported("command-buffer GEMM");
  }
  if (alpha == nullptr || beta == nullptr || c == nullptr) {
    return absl::InvalidArgumentError(
        "muBLAS GEMM scalar or output pointer is null");
  }
  absl::StatusOr<MuBlasGemmTypes> types =
      GetMuBlasGemmTypes(type_a, type_b, type_c, computation_type);
  if (!types.ok()) return types.status();
  if ((api_->capabilities() & types->capability) == 0) {
    return absl::UnimplementedError(
        "loaded muBLAS shim does not advertise the required GEMM capability");
  }
  absl::Status dimensions =
      ValidateDimensions(transa, transb, m, n, k, lda, ldb, ldc);
  if (!dimensions.ok()) return dimensions;
  XlaMusaMuBlasAlgorithm normalized_algorithm;
  RETURN_IF_ERROR(ValidateAlgorithm(algorithm, *types, *api_, engine_options,
                                    &normalized_algorithm));
  if (m == 0 || n == 0) return absl::OkStatus();
  if (a.is_null() || b.is_null() || c->is_null()) {
    return absl::InvalidArgumentError("muBLAS GEMM matrix pointer is null");
  }

  absl::StatusOr<XlaMusaMuBlasOperation> operation_a =
      AsMuBlasOperation(transa);
  if (!operation_a.ok()) return operation_a.status();
  absl::StatusOr<XlaMusaMuBlasOperation> operation_b =
      AsMuBlasOperation(transb);
  if (!operation_b.ok()) return operation_b.status();

  ASSIGN_OR_RETURN(std::shared_ptr<HandleState> state,
                   GetOrCreateHandle(stream));
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::MutexLock state_lock(&state->mu);
  RETURN_IF_ERROR(
      ConfigureAtomics(state.get(), engine_options, normalized_algorithm));

  std::unique_ptr<EventBasedTimer> timer;
  if (output_profile_result != nullptr) {
    ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                output_profile_result->warmup_run_executed()));
  }
  absl::Status status;
  if (api_->SupportsGemmWithAlgorithm()) {
    status = api_->GemmWithAlgorithm(
        state->handle, types->input_type, types->output_type,
        types->compute_type, *operation_a, *operation_b,
        static_cast<int64_t>(m), static_cast<int64_t>(n),
        static_cast<int64_t>(k), alpha, a.opaque(), lda, b.opaque(), ldb, beta,
        c->opaque(), ldc, normalized_algorithm);
  } else {
    status = api_->Gemm(state->handle, types->input_type, types->output_type,
                        types->compute_type, *operation_a, *operation_b,
                        static_cast<int64_t>(m), static_cast<int64_t>(n),
                        static_cast<int64_t>(k), alpha, a.opaque(), lda,
                        b.opaque(), ldb, beta, c->opaque(), ldc);
  }
  RETURN_IF_ERROR(status);
  RETURN_IF_ERROR(
      PopulateProfileFromTimer(timer.get(), algorithm, output_profile_result));
  {
    absl::MutexLock lock(&handles_mu_);
    last_stream_ = state->native_stream;
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MusaBlas::IsMainStreamSet() const {
  absl::MutexLock lock(&handles_mu_);
  if (handles_.empty()) {
    return absl::FailedPreconditionError("muBLAS is not initialized");
  }
  return last_stream_ == nullptr;
}

absl::Status MusaBlas::DoBlasGemm(Stream* stream, blas::Transpose transa,
                                  blas::Transpose transb, uint64_t m,
                                  uint64_t n, uint64_t k, blas::DataType dtype,
                                  const void* alpha, const DeviceAddressBase& a,
                                  int lda, const DeviceAddressBase& b, int ldb,
                                  const void* beta, DeviceAddressBase* c,
                                  int ldc, const EngineOptions& engine_options,
                                  blas::CallContext) {
  blas::ComputationType computation_type;
  switch (dtype) {
    case blas::DataType::kHalf:
      return absl::UnimplementedError(
          "plain MUSA f16 GEMM requires f32 computation, which the qualified "
          "muBLAS GemmEx does not implement");
    case blas::DataType::kBF16:
    case blas::DataType::kFloat:
      computation_type = blas::ComputationType::kF32;
      break;
    case blas::DataType::kDouble:
      computation_type = blas::ComputationType::kF64;
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("basic MUSA muBLAS GEMM does not support data type ",
                       blas::DataTypeString(dtype)));
  }
  return DoBlasGemmInternal(stream, transa, transb, m, n, k, alpha, a, dtype,
                            lda, b, dtype, ldb, beta, c, dtype, ldc,
                            computation_type, kMusaBlasDefaultAlgorithm,
                            engine_options, /*output_profile_result=*/nullptr);
}

bool MusaBlas::GetBlasGemmAlgorithms(
    Stream* stream, const gpu::MatrixDescriptor& a,
    const gpu::MatrixDescriptor& b, gpu::OutputMatrixDescriptor* c,
    const void* alpha, const void* beta,
    std::vector<blas::AlgorithmType>* out_algorithms) {
  if (stream == nullptr || stream->parent() != parent_ || c == nullptr ||
      alpha == nullptr || beta == nullptr || out_algorithms == nullptr) {
    return false;
  }
  out_algorithms->clear();
  absl::StatusOr<MuBlasGemmTypes> types =
      GetMuBlasGemmTypes(a.type, b.type, c->type, c->compute_type);
  if (!types.ok() || (api_->capabilities() & types->capability) == 0) {
    return false;
  }
  out_algorithms->push_back(kMusaBlasDefaultAlgorithm);
  if (types->input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F32 &&
      types->output_type == XLA_MUSA_MUBLAS_DATA_TYPE_F32 &&
      types->compute_type == XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32 &&
      api_->SupportsTensorOpF32()) {
    out_algorithms->push_back(kMusaBlasTensorOpAlgorithm);
  }
  return true;
}

absl::Status MusaBlas::DoBlasGemmWithAlgorithm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, const DeviceAddressBase& b,
    blas::DataType type_b, int ldb, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const EngineOptions& engine_options,
    blas::ProfileResult* output_profile_result, blas::CallContext context) {
  if (output_profile_result != nullptr) {
    output_profile_result->set_is_valid(false);
    output_profile_result->set_algorithm(algorithm);
  }
  return DoBlasGemmInternal(stream, transa, transb, m, n, k, alpha, a, type_a,
                            lda, b, type_b, ldb, beta, c, type_c, ldc,
                            computation_type, algorithm, engine_options,
                            output_profile_result);
}

absl::Status MusaBlas::GetVersion(std::string* version) {
  if (version == nullptr) {
    return absl::InvalidArgumentError("muBLAS version output is null");
  }
  ASSIGN_OR_RETURN(std::shared_ptr<HandleState> state,
                   GetOrCreateHandleForNativeStream(nullptr));
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::MutexLock state_lock(&state->mu);
  int32_t encoded_version = 0;
  absl::Status status = api_->GetVersion(state->handle, &encoded_version);
  if (!status.ok()) return status;
  *version = std::to_string(encoded_version);
  return absl::OkStatus();
}

absl::Status MusaBlas::DoBlasScalInternal(Stream* stream, uint64_t n,
                                          const void* alpha,
                                          DeviceAddressBase* x, int incx,
                                          XlaMusaMuBlasScalType scal_type,
                                          size_t element_size) {
  if (!api_->SupportsScal()) {
    return absl::UnimplementedError(
        "loaded muBLAS shim does not advertise the optional SCAL contract");
  }
  if (n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return absl::OutOfRangeError(
        "muBLAS SCAL element count exceeds the shim's signed 64-bit ABI");
  }
  if (incx <= 0) {
    return absl::InvalidArgumentError("muBLAS SCAL increment must be positive");
  }
  if (n == 0) return absl::OkStatus();
  if (alpha == nullptr || x == nullptr || x->is_null()) {
    return absl::InvalidArgumentError(
        "muBLAS SCAL scalar or vector pointer is null");
  }

  const uint64_t unsigned_incx = static_cast<uint64_t>(incx);
  if (n - 1 > (std::numeric_limits<uint64_t>::max() - 1) / unsigned_incx) {
    return absl::OutOfRangeError("muBLAS SCAL vector extent overflows");
  }
  const uint64_t required_elements = (n - 1) * unsigned_incx + 1;
  if (required_elements > std::numeric_limits<uint64_t>::max() / element_size) {
    return absl::OutOfRangeError("muBLAS SCAL byte extent overflows");
  }
  const uint64_t required_bytes = required_elements * element_size;
  if (x->size() != 0 && x->size() < required_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("muBLAS SCAL vector has ", x->size(),
                     " bytes; needs at least ", required_bytes));
  }

  ASSIGN_OR_RETURN(std::shared_ptr<HandleState> state,
                   GetOrCreateHandle(stream));
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::MutexLock state_lock(&state->mu);
  RETURN_IF_ERROR(api_->Scal(state->handle, scal_type, static_cast<int64_t>(n),
                             alpha, x->opaque(), incx));
  {
    absl::MutexLock lock(&handles_mu_);
    last_stream_ = state->native_stream;
  }
  return absl::OkStatus();
}

bool MusaBlas::DoBlasScal(Stream* stream, uint64_t n, float alpha,
                          DeviceAddress<float>* x, int incx) {
  absl::Status status = DoBlasScalInternal(
      stream, n, &alpha, x, incx, XLA_MUSA_MUBLAS_SCAL_TYPE_F32, sizeof(float));
  if (!status.ok()) LOG(ERROR) << status;
  return status.ok();
}

bool MusaBlas::DoBlasScal(Stream* stream, uint64_t n, double alpha,
                          DeviceAddress<double>* x, int incx) {
  absl::Status status =
      DoBlasScalInternal(stream, n, &alpha, x, incx,
                         XLA_MUSA_MUBLAS_SCAL_TYPE_F64, sizeof(double));
  if (!status.ok()) LOG(ERROR) << status;
  return status.ok();
}

bool MusaBlas::DoBlasScal(Stream* stream, uint64_t n, float alpha,
                          DeviceAddress<std::complex<float>>* x, int incx) {
  absl::Status status = DoBlasScalInternal(stream, n, &alpha, x, incx,
                                           XLA_MUSA_MUBLAS_SCAL_TYPE_C64_F32,
                                           sizeof(std::complex<float>));
  if (!status.ok()) LOG(ERROR) << status;
  return status.ok();
}

bool MusaBlas::DoBlasScal(Stream* stream, uint64_t n, double alpha,
                          DeviceAddress<std::complex<double>>* x, int incx) {
  absl::Status status = DoBlasScalInternal(stream, n, &alpha, x, incx,
                                           XLA_MUSA_MUBLAS_SCAL_TYPE_C128_F64,
                                           sizeof(std::complex<double>));
  if (!status.ok()) LOG(ERROR) << status;
  return status.ok();
}

bool MusaBlas::DoBlasScal(Stream* stream, uint64_t n, std::complex<float> alpha,
                          DeviceAddress<std::complex<float>>* x, int incx) {
  absl::Status status = DoBlasScalInternal(stream, n, &alpha, x, incx,
                                           XLA_MUSA_MUBLAS_SCAL_TYPE_C64,
                                           sizeof(std::complex<float>));
  if (!status.ok()) LOG(ERROR) << status;
  return status.ok();
}

bool MusaBlas::DoBlasScal(Stream* stream, uint64_t n,
                          std::complex<double> alpha,
                          DeviceAddress<std::complex<double>>* x, int incx) {
  absl::Status status = DoBlasScalInternal(stream, n, &alpha, x, incx,
                                           XLA_MUSA_MUBLAS_SCAL_TYPE_C128,
                                           sizeof(std::complex<double>));
  if (!status.ok()) LOG(ERROR) << status;
  return status.ok();
}

bool MusaBlas::DoBlasGemv(Stream*, blas::Transpose, uint64_t, uint64_t, float,
                          const DeviceAddress<float>&, int,
                          const DeviceAddress<float>&, int, float,
                          DeviceAddress<float>*, int) {
  return UnsupportedBool("GEMV f32");
}
bool MusaBlas::DoBlasGemv(Stream*, blas::Transpose, uint64_t, uint64_t, double,
                          const DeviceAddress<double>&, int,
                          const DeviceAddress<double>&, int, double,
                          DeviceAddress<double>*, int) {
  return UnsupportedBool("GEMV f64");
}
bool MusaBlas::DoBlasGemv(Stream*, blas::Transpose, uint64_t, uint64_t,
                          std::complex<float>,
                          const DeviceAddress<std::complex<float>>&, int,
                          const DeviceAddress<std::complex<float>>&, int,
                          std::complex<float>,
                          DeviceAddress<std::complex<float>>*, int) {
  return UnsupportedBool("GEMV complex-f32");
}
bool MusaBlas::DoBlasGemv(Stream*, blas::Transpose, uint64_t, uint64_t,
                          std::complex<double>,
                          const DeviceAddress<std::complex<double>>&, int,
                          const DeviceAddress<std::complex<double>>&, int,
                          std::complex<double>,
                          DeviceAddress<std::complex<double>>*, int) {
  return UnsupportedBool("GEMV complex-f64");
}

template <typename T, typename Scalar>
bool MusaBlas::DoBlasGemmBatchedTyped(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, Scalar alpha, DeviceAddressSlice<T> a, int lda,
    DeviceAddressSlice<T> b, int ldb, Scalar beta, DeviceAddressSlice<T> c,
    int ldc, int batch_count, blas::DataType data_type,
    blas::ComputationType computation_type, const EngineOptions& engine_options,
    ScratchAllocator* scratch_allocator) {
  absl::Status sizes =
      ValidatePointerArraySizes(a.size(), b.size(), c.size(), batch_count);
  if (!sizes.ok()) {
    LOG(ERROR) << sizes;
    return false;
  }
  absl::Status dimensions =
      ValidateDimensions(transa, transb, m, n, k, lda, ldb, ldc);
  if (!dimensions.ok()) {
    LOG(ERROR) << dimensions;
    return false;
  }
  if (batch_count == 0 || m == 0 || n == 0) return true;

  absl::StatusOr<uint64_t> a_elements =
      RequiredMatrixElements(transa, m, k, lda);
  absl::StatusOr<uint64_t> b_elements =
      RequiredMatrixElements(transb, k, n, ldb);
  absl::StatusOr<uint64_t> c_elements =
      RequiredMatrixElements(blas::Transpose::kNoTranspose, m, n, ldc);
  if (!a_elements.ok() || !b_elements.ok() || !c_elements.ok()) {
    LOG(ERROR) << (!a_elements.ok()   ? a_elements.status()
                   : !b_elements.ok() ? b_elements.status()
                                      : c_elements.status());
    return false;
  }

  std::vector<const void*> a_raw;
  std::vector<const void*> b_raw;
  std::vector<void*> c_raw;
  a_raw.reserve(batch_count);
  b_raw.reserve(batch_count);
  c_raw.reserve(batch_count);
  for (int i = 0; i < batch_count; ++i) {
    if (a[i] == nullptr || b[i] == nullptr || c[i] == nullptr) {
      LOG(ERROR) << "muBLAS pointer-array slice contains a null wrapper at "
                 << i;
      return false;
    }
    absl::Status a_extent = ValidateBufferExtent(
        *a[i], *a_elements, /*stride=*/0, /*batch_count=*/1, sizeof(T), "A");
    absl::Status b_extent = ValidateBufferExtent(
        *b[i], *b_elements, /*stride=*/0, /*batch_count=*/1, sizeof(T), "B");
    absl::Status c_extent = ValidateBufferExtent(
        *c[i], *c_elements, /*stride=*/0, /*batch_count=*/1, sizeof(T), "C");
    if (!a_extent.ok() || !b_extent.ok() || !c_extent.ok()) {
      LOG(ERROR) << (!a_extent.ok()   ? a_extent
                     : !b_extent.ok() ? b_extent
                                      : c_extent);
      return false;
    }
    a_raw.push_back(a[i]->opaque());
    b_raw.push_back(b[i]->opaque());
    c_raw.push_back(c[i]->opaque());
  }

  absl::Status status = DoBlasGemmBatchedInternal(
      stream, transa, transb, m, n, k, &alpha, a_raw, data_type, lda, b_raw,
      data_type, ldb, &beta, c_raw, data_type, ldc, batch_count,
      computation_type, engine_options, scratch_allocator);
  if (!status.ok()) LOG(ERROR) << status;
  return status.ok();
}

absl::Status MusaBlas::DoBlasGemmBatchedInternal(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, absl::Span<const void* const> a,
    blas::DataType type_a, int lda, absl::Span<const void* const> b,
    blas::DataType type_b, int ldb, const void* beta, absl::Span<void* const> c,
    blas::DataType type_c, int ldc, int batch_count,
    blas::ComputationType computation_type, const EngineOptions& engine_options,
    ScratchAllocator* scratch_allocator) {
  if (engine_options.require_command_buffer) {
    return Unsupported("command-buffer pointer-array batched GEMM");
  }
  if (!api_->SupportsGemmBatched() || !api_->UsesZeroExternalWorkspace()) {
    return Unsupported(
        "pointer-array batched GEMM without the v2 zero-"
        "workspace contract");
  }
  if (alpha == nullptr || beta == nullptr) {
    return absl::InvalidArgumentError(
        "muBLAS batched GEMM scalar pointer is null");
  }
  RETURN_IF_ERROR(
      ValidatePointerArraySizes(a.size(), b.size(), c.size(), batch_count));
  RETURN_IF_ERROR(ValidateDimensions(transa, transb, m, n, k, lda, ldb, ldc));
  ASSIGN_OR_RETURN(
      MuBlasGemmTypes types,
      GetMuBlasGemmTypes(type_a, type_b, type_c, computation_type));
  if ((api_->capabilities() & types.capability) == 0) {
    return Unsupported("pointer-array batched GEMM for the requested type");
  }
  if (batch_count == 0 || m == 0 || n == 0) return absl::OkStatus();
  if (scratch_allocator == nullptr) {
    return absl::InvalidArgumentError(
        "muBLAS pointer-array batched GEMM scratch allocator is null");
  }

  const int64_t pointer_bytes =
      static_cast<int64_t>(batch_count) * static_cast<int64_t>(sizeof(void*));
  const int64_t memory_limit = scratch_allocator->GetMemoryLimitInBytes();
  if (memory_limit >= 0 && memory_limit < 3 * pointer_bytes) {
    return absl::ResourceExhaustedError(
        absl::StrCat("muBLAS pointer-array staging needs ", 3 * pointer_bytes,
                     " bytes; scratch limit is ", memory_limit));
  }
  ASSIGN_OR_RETURN(DeviceAddress<uint8_t> a_device,
                   scratch_allocator->AllocateBytes(pointer_bytes));
  ASSIGN_OR_RETURN(DeviceAddress<uint8_t> b_device,
                   scratch_allocator->AllocateBytes(pointer_bytes));
  ASSIGN_OR_RETURN(DeviceAddress<uint8_t> c_device,
                   scratch_allocator->AllocateBytes(pointer_bytes));
  // Stream::Memcpy requires registered host memory and is asynchronous. Keep
  // one contiguous allocation alive until all three pointer-array copies have
  // consumed it. The ordered callback is queued before GEMM and releases only
  // host staging; device staging belongs to the caller's ScratchAllocator.
  ASSIGN_OR_RETURN(std::unique_ptr<MemoryAllocation> host_allocation,
                   parent_->HostMemoryAllocate(3 * pointer_bytes));
  if (host_allocation == nullptr || host_allocation->address().is_null() ||
      host_allocation->address().size() < 3 * pointer_bytes) {
    return absl::InternalError(
        "muBLAS pointer-array host staging allocation is invalid");
  }
  auto host = std::shared_ptr<MemoryAllocation>(std::move(host_allocation));
  auto** host_pointers = static_cast<void**>(host->address().opaque());
  for (int i = 0; i < batch_count; ++i) {
    host_pointers[i] = const_cast<void*>(a[i]);
    host_pointers[batch_count + i] = const_cast<void*>(b[i]);
    host_pointers[2 * batch_count + i] = c[i];
  }
  absl::Status copy_status =
      stream->Memcpy(&a_device, host_pointers, pointer_bytes);
  if (copy_status.ok()) {
    copy_status =
        stream->Memcpy(&b_device, host_pointers + batch_count, pointer_bytes);
  }
  if (copy_status.ok()) {
    copy_status = stream->Memcpy(&c_device, host_pointers + 2 * batch_count,
                                 pointer_bytes);
  }
  if (!copy_status.ok()) {
    // A prior asynchronous copy may still read `host`. Drain only on this
    // exceptional path before returning the original copy error.
    absl::Status drain = stream->BlockHostUntilDone();
    if (!drain.ok()) {
      LOG(ERROR) << "Unable to drain muBLAS pointer staging after copy "
                    "failure: "
                 << drain;
      RetainFailedHostStaging(host);
    }
    return copy_status;
  }
  absl::Status callback_status = stream->DoHostCallbackWithStatus(
      [host]() mutable { return absl::OkStatus(); });
  if (!callback_status.ok()) {
    // The copies are already enqueued, but a rejected callback releases its
    // capture immediately. Keep the local owner through synchronization.
    absl::Status drain = stream->BlockHostUntilDone();
    if (!drain.ok()) {
      LOG(ERROR) << "Unable to drain muBLAS pointer staging after callback "
                    "enqueue failure: "
                 << drain;
      RetainFailedHostStaging(host);
    }
    return callback_status;
  }

  ASSIGN_OR_RETURN(XlaMusaMuBlasOperation operation_a,
                   AsMuBlasOperation(transa));
  ASSIGN_OR_RETURN(XlaMusaMuBlasOperation operation_b,
                   AsMuBlasOperation(transb));
  ASSIGN_OR_RETURN(std::shared_ptr<HandleState> state,
                   GetOrCreateHandle(stream));
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::MutexLock state_lock(&state->mu);
  RETURN_IF_ERROR(ConfigureAtomics(state.get(), engine_options,
                                   XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT));
  RETURN_IF_ERROR(api_->GemmBatched(
      state->handle, types.input_type, types.output_type, types.compute_type,
      operation_a, operation_b, static_cast<int64_t>(m),
      static_cast<int64_t>(n), static_cast<int64_t>(k), alpha,
      reinterpret_cast<const void* const*>(a_device.opaque()), lda,
      reinterpret_cast<const void* const*>(b_device.opaque()), ldb, beta,
      reinterpret_cast<void* const*>(c_device.opaque()), ldc, batch_count,
      XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT));
  {
    absl::MutexLock lock(&handles_mu_);
    last_stream_ = state->native_stream;
  }
  return absl::OkStatus();
}

bool MusaBlas::DoBlasGemmBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, float alpha, DeviceAddressSlice<Eigen::half> a,
    int lda, DeviceAddressSlice<Eigen::half> b, int ldb, float beta,
    DeviceAddressSlice<Eigen::half> c, int ldc, int batch_count,
    const EngineOptions& engine_options, ScratchAllocator* scratch_allocator,
    blas::CallContext) {
  return DoBlasGemmBatchedTyped(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      batch_count, blas::DataType::kHalf, blas::ComputationType::kF16,
      engine_options, scratch_allocator);
}
bool MusaBlas::DoBlasGemmBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, float alpha, DeviceAddressSlice<Eigen::bfloat16> a,
    int lda, DeviceAddressSlice<Eigen::bfloat16> b, int ldb, float beta,
    DeviceAddressSlice<Eigen::bfloat16> c, int ldc, int batch_count,
    const EngineOptions& engine_options, ScratchAllocator* scratch_allocator,
    blas::CallContext) {
  return DoBlasGemmBatchedTyped(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      batch_count, blas::DataType::kBF16, blas::ComputationType::kF32,
      engine_options, scratch_allocator);
}
bool MusaBlas::DoBlasGemmBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, float alpha, DeviceAddressSlice<float> a, int lda,
    DeviceAddressSlice<float> b, int ldb, float beta,
    DeviceAddressSlice<float> c, int ldc, int batch_count,
    const EngineOptions& engine_options, ScratchAllocator* scratch_allocator,
    blas::CallContext) {
  return DoBlasGemmBatchedTyped(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      batch_count, blas::DataType::kFloat, blas::ComputationType::kF32,
      engine_options, scratch_allocator);
}
bool MusaBlas::DoBlasGemmBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, double alpha, DeviceAddressSlice<double> a, int lda,
    DeviceAddressSlice<double> b, int ldb, double beta,
    DeviceAddressSlice<double> c, int ldc, int batch_count,
    const EngineOptions& engine_options, ScratchAllocator* scratch_allocator,
    blas::CallContext) {
  return DoBlasGemmBatchedTyped(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      batch_count, blas::DataType::kDouble, blas::ComputationType::kF64,
      engine_options, scratch_allocator);
}
bool MusaBlas::DoBlasGemmBatched(
    Stream*, blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64_t,
    std::complex<float>, DeviceAddressSlice<std::complex<float>>, int,
    DeviceAddressSlice<std::complex<float>>, int, std::complex<float>,
    DeviceAddressSlice<std::complex<float>>, int, int, const EngineOptions&,
    ScratchAllocator*, blas::CallContext) {
  return UnsupportedBool("batched GEMM complex-f32");
}
bool MusaBlas::DoBlasGemmBatched(
    Stream*, blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64_t,
    std::complex<double>, DeviceAddressSlice<std::complex<double>>, int,
    DeviceAddressSlice<std::complex<double>>, int, std::complex<double>,
    DeviceAddressSlice<std::complex<double>>, int, int, const EngineOptions&,
    ScratchAllocator*, blas::CallContext) {
  return UnsupportedBool("batched GEMM complex-f64");
}

absl::Status MusaBlas::DoBlasGemmStridedBatchedInternal(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, blas::DataType type_b, int ldb,
    int64_t stride_b, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, int64_t stride_c, int batch_count,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    const EngineOptions& engine_options,
    blas::ProfileResult* output_profile_result) {
  if (output_profile_result != nullptr) {
    output_profile_result->set_is_valid(false);
    output_profile_result->set_algorithm(algorithm);
  }
  if (engine_options.require_command_buffer) {
    return Unsupported("command-buffer strided-batched GEMM");
  }
  if (!api_->SupportsGemmStridedBatched() ||
      !api_->UsesZeroExternalWorkspace()) {
    return Unsupported(
        "strided-batched GEMM without the v2 zero-workspace "
        "contract");
  }
  if (alpha == nullptr || beta == nullptr || c == nullptr) {
    return absl::InvalidArgumentError(
        "muBLAS strided-batched GEMM scalar or output pointer is null");
  }
  RETURN_IF_ERROR(ValidateBatch(batch_count));
  RETURN_IF_ERROR(ValidateStrides(stride_a, stride_b, stride_c, batch_count));
  RETURN_IF_ERROR(ValidateDimensions(transa, transb, m, n, k, lda, ldb, ldc));
  ASSIGN_OR_RETURN(
      MuBlasGemmTypes types,
      GetMuBlasGemmTypes(type_a, type_b, type_c, computation_type));
  if ((api_->capabilities() & types.capability) == 0) {
    return Unsupported("strided-batched GEMM for the requested type");
  }
  XlaMusaMuBlasAlgorithm normalized_algorithm;
  RETURN_IF_ERROR(ValidateAlgorithm(algorithm, types, *api_, engine_options,
                                    &normalized_algorithm));
  if (batch_count == 0 || m == 0 || n == 0) return absl::OkStatus();

  ASSIGN_OR_RETURN(uint64_t a_elements,
                   RequiredMatrixElements(transa, m, k, lda));
  ASSIGN_OR_RETURN(uint64_t b_elements,
                   RequiredMatrixElements(transb, k, n, ldb));
  ASSIGN_OR_RETURN(
      uint64_t c_elements,
      RequiredMatrixElements(blas::Transpose::kNoTranspose, m, n, ldc));
  if (batch_count > 1 && static_cast<uint64_t>(stride_c) < c_elements) {
    return absl::InvalidArgumentError(
        absl::StrCat("muBLAS GEMM output batch stride ", stride_c,
                     " overlaps a matrix extent of ", c_elements, " elements"));
  }
  ASSIGN_OR_RETURN(size_t element_size, DataTypeSize(type_a));
  RETURN_IF_ERROR(ValidateBufferExtent(a, a_elements, stride_a, batch_count,
                                       element_size, "A"));
  RETURN_IF_ERROR(ValidateBufferExtent(b, b_elements, stride_b, batch_count,
                                       element_size, "B"));
  RETURN_IF_ERROR(ValidateBufferExtent(*c, c_elements, stride_c, batch_count,
                                       element_size, "C"));
  ASSIGN_OR_RETURN(XlaMusaMuBlasOperation operation_a,
                   AsMuBlasOperation(transa));
  ASSIGN_OR_RETURN(XlaMusaMuBlasOperation operation_b,
                   AsMuBlasOperation(transb));
  ASSIGN_OR_RETURN(std::shared_ptr<HandleState> state,
                   GetOrCreateHandle(stream));
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::MutexLock state_lock(&state->mu);
  RETURN_IF_ERROR(
      ConfigureAtomics(state.get(), engine_options, normalized_algorithm));
  std::unique_ptr<EventBasedTimer> timer;
  if (output_profile_result != nullptr) {
    ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                output_profile_result->warmup_run_executed()));
  }
  RETURN_IF_ERROR(api_->GemmStridedBatched(
      state->handle, types.input_type, types.output_type, types.compute_type,
      operation_a, operation_b, static_cast<int64_t>(m),
      static_cast<int64_t>(n), static_cast<int64_t>(k), alpha, a.opaque(), lda,
      stride_a, b.opaque(), ldb, stride_b, beta, c->opaque(), ldc, stride_c,
      batch_count, normalized_algorithm));
  RETURN_IF_ERROR(
      PopulateProfileFromTimer(timer.get(), algorithm, output_profile_result));
  {
    absl::MutexLock lock(&handles_mu_);
    last_stream_ = state->native_stream;
  }
  return absl::OkStatus();
}

absl::Status MusaBlas::DoBlasGemmStridedBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, blas::DataType dtype, const void* alpha,
    const DeviceAddressBase& a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, int ldb, int64_t stride_b, const void* beta,
    DeviceAddressBase* c, int ldc, int64_t stride_c, int batch_count,
    const EngineOptions& engine_options, blas::CallContext) {
  blas::ComputationType computation_type;
  switch (dtype) {
    case blas::DataType::kFloat:
      computation_type = blas::ComputationType::kF32;
      break;
    case blas::DataType::kDouble:
      computation_type = blas::ComputationType::kF64;
      break;
    case blas::DataType::kHalf:
      return Unsupported("plain f16 strided-batched GEMM with f32 computation");
    default:
      return Unsupported(absl::StrCat("strided-batched GEMM for ",
                                      blas::DataTypeString(dtype)));
  }
  return DoBlasGemmStridedBatchedInternal(
      stream, transa, transb, m, n, k, alpha, a, dtype, lda, stride_a, b, dtype,
      ldb, stride_b, beta, c, dtype, ldc, stride_c, batch_count,
      computation_type, kMusaBlasDefaultAlgorithm, engine_options,
      /*output_profile_result=*/nullptr);
}

absl::Status MusaBlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, blas::DataType type_b, int ldb,
    int64_t stride_b, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, int64_t stride_c, int batch_count,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    const EngineOptions& engine_options,
    blas::ProfileResult* output_profile_result, blas::CallContext) {
  return DoBlasGemmStridedBatchedInternal(
      stream, transa, transb, m, n, k, alpha, a, type_a, lda, stride_a, b,
      type_b, ldb, stride_b, beta, c, type_c, ldc, stride_c, batch_count,
      computation_type, algorithm, engine_options, output_profile_result);
}

bool MusaBlas::DoBlasTrsm(Stream*, blas::Side, blas::UpperLower,
                          blas::Transpose, blas::Diagonal, uint64_t, uint64_t,
                          float, const DeviceAddress<float>&, int,
                          DeviceAddress<float>*, int) {
  return UnsupportedBool("TRSM f32");
}
bool MusaBlas::DoBlasTrsm(Stream*, blas::Side, blas::UpperLower,
                          blas::Transpose, blas::Diagonal, uint64_t, uint64_t,
                          double, const DeviceAddress<double>&, int,
                          DeviceAddress<double>*, int) {
  return UnsupportedBool("TRSM f64");
}
bool MusaBlas::DoBlasTrsm(Stream*, blas::Side, blas::UpperLower,
                          blas::Transpose, blas::Diagonal, uint64_t, uint64_t,
                          std::complex<float>,
                          const DeviceAddress<std::complex<float>>&, int,
                          DeviceAddress<std::complex<float>>*, int) {
  return UnsupportedBool("TRSM complex-f32");
}
bool MusaBlas::DoBlasTrsm(Stream*, blas::Side, blas::UpperLower,
                          blas::Transpose, blas::Diagonal, uint64_t, uint64_t,
                          std::complex<double>,
                          const DeviceAddress<std::complex<double>>&, int,
                          DeviceAddress<std::complex<double>>*, int) {
  return UnsupportedBool("TRSM complex-f64");
}

bool MusaBlas::DoBlasTrsmBatched(Stream*, blas::Side, blas::UpperLower,
                                 blas::Transpose, blas::Diagonal, uint64_t,
                                 uint64_t, float, const DeviceAddress<float*>&,
                                 int, DeviceAddress<float*>*, int, int) {
  return UnsupportedBool("batched TRSM f32");
}
bool MusaBlas::DoBlasTrsmBatched(Stream*, blas::Side, blas::UpperLower,
                                 blas::Transpose, blas::Diagonal, uint64_t,
                                 uint64_t, double,
                                 const DeviceAddress<double*>&, int,
                                 DeviceAddress<double*>*, int, int) {
  return UnsupportedBool("batched TRSM f64");
}
bool MusaBlas::DoBlasTrsmBatched(Stream*, blas::Side, blas::UpperLower,
                                 blas::Transpose, blas::Diagonal, uint64_t,
                                 uint64_t, std::complex<float>,
                                 const DeviceAddress<std::complex<float>*>&,
                                 int, DeviceAddress<std::complex<float>*>*, int,
                                 int) {
  return UnsupportedBool("batched TRSM complex-f32");
}
bool MusaBlas::DoBlasTrsmBatched(Stream*, blas::Side, blas::UpperLower,
                                 blas::Transpose, blas::Diagonal, uint64_t,
                                 uint64_t, std::complex<double>,
                                 const DeviceAddress<std::complex<double>*>&,
                                 int, DeviceAddress<std::complex<double>*>*,
                                 int, int) {
  return UnsupportedBool("batched TRSM complex-f64");
}

void InitializeMusaBlas() {
  PluginRegistry* registry = PluginRegistry::Instance();
  if (registry->HasFactory(kMusaPlatformId, PluginKind::kBlas)) return;
  absl::Status status = registry->RegisterFactory<PluginRegistry::BlasFactory>(
      kMusaPlatformId, "muBLAS",
      [](StreamExecutor* parent) -> blas::BlasSupport* {
        auto* blas = new MusaBlas(parent);
        if (!blas->Init()) {
          delete blas;
          return nullptr;
        }
        return blas;
      });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register muBLAS factory: " << status;
  }
}

}  // namespace stream_executor::musa

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_mublas, {
  stream_executor::musa::InitializeMusaBlas();
});
