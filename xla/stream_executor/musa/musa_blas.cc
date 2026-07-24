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

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/engine_options.h"
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

absl::Status ValidateDimensions(uint64_t m, uint64_t n, uint64_t k, int lda,
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
  return absl::OkStatus();
}

}  // namespace

MusaBlas::MusaBlas(StreamExecutor* parent, MusaMuBlasApi* api)
    : parent_(parent), api_(api) {}

bool MusaBlas::Init() {
  absl::MutexLock lock(&mu_);
  if (handle_ != nullptr) return true;
  if (parent_ == nullptr || api_ == nullptr) {
    LOG(ERROR) << "Cannot initialize muBLAS with a null executor or API";
    return false;
  }
  absl::Status load = api_->Init();
  if (!load.ok()) {
    LOG(ERROR) << "Unable to load optional muBLAS shim: " << load;
    return false;
  }
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::Status create = api_->Create(&handle_);
  if (!create.ok()) {
    LOG(ERROR) << "Unable to create muBLAS handle: " << create;
    handle_ = nullptr;
    return false;
  }
  current_stream_ = nullptr;
  return true;
}

MusaBlas::~MusaBlas() {
  absl::MutexLock lock(&mu_);
  if (handle_ == nullptr || parent_ == nullptr || api_ == nullptr) return;
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::Status status = api_->Destroy(handle_);
  if (!status.ok()) LOG(ERROR) << "Unable to destroy muBLAS handle: " << status;
  handle_ = nullptr;
}

absl::Status MusaBlas::SetStream(Stream* stream) {
  if (handle_ == nullptr) {
    return absl::FailedPreconditionError("muBLAS is not initialized");
  }
  if (stream == nullptr || stream->parent() != parent_) {
    return absl::InvalidArgumentError(
        "muBLAS stream is null or belongs to another executor");
  }
  void* stream_handle = stream->platform_specific_handle().stream;
  absl::Status status = api_->SetStream(handle_, stream_handle);
  if (status.ok()) current_stream_ = stream_handle;
  return status;
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
    const EngineOptions& engine_options) {
  if (engine_options.require_command_buffer) {
    return Unsupported("command-buffer GEMM");
  }
  if (engine_options.require_determinism) {
    return Unsupported("deterministic GEMM");
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
  if (m == 0 || n == 0) return absl::OkStatus();
  if (a.is_null() || b.is_null() || c->is_null()) {
    return absl::InvalidArgumentError("muBLAS GEMM matrix pointer is null");
  }
  absl::Status dimensions = ValidateDimensions(m, n, k, lda, ldb, ldc);
  if (!dimensions.ok()) return dimensions;

  absl::StatusOr<XlaMusaMuBlasOperation> operation_a =
      AsMuBlasOperation(transa);
  if (!operation_a.ok()) return operation_a.status();
  absl::StatusOr<XlaMusaMuBlasOperation> operation_b =
      AsMuBlasOperation(transb);
  if (!operation_b.ok()) return operation_b.status();

  absl::MutexLock lock(&mu_);
  if (handle_ == nullptr) {
    return absl::FailedPreconditionError("muBLAS is not initialized");
  }
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  absl::Status set_stream = SetStream(stream);
  if (!set_stream.ok()) return set_stream;
  return api_->Gemm(handle_, types->input_type, types->output_type,
                    types->compute_type, *operation_a, *operation_b,
                    static_cast<int64_t>(m), static_cast<int64_t>(n),
                    static_cast<int64_t>(k), alpha, a.opaque(), lda, b.opaque(),
                    ldb, beta, c->opaque(), ldc);
}

absl::StatusOr<bool> MusaBlas::IsMainStreamSet() const {
  absl::MutexLock lock(&mu_);
  if (handle_ == nullptr) {
    return absl::FailedPreconditionError("muBLAS is not initialized");
  }
  return current_stream_ == nullptr;
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
                            computation_type, engine_options);
}

bool MusaBlas::GetBlasGemmAlgorithms(
    Stream*, const gpu::MatrixDescriptor&, const gpu::MatrixDescriptor&,
    gpu::OutputMatrixDescriptor*, const void*, const void*,
    std::vector<blas::AlgorithmType>* out_algorithms) {
  if (out_algorithms == nullptr) return false;
  out_algorithms->clear();
  // The basic adapter executes the explicit default algorithm but does not
  // claim a descriptor-aware algorithm-enumeration contract.
  return false;
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
  if (algorithm != blas::kDefaultAlgorithm &&
      algorithm != blas::kDefaultBlasGemm) {
    return Unsupported("non-default GEMM algorithms");
  }
  absl::Status status = DoBlasGemmInternal(
      stream, transa, transb, m, n, k, alpha, a, type_a, lda, b, type_b, ldb,
      beta, c, type_c, ldc, computation_type, engine_options);
  // The basic adapter does not provide an event-based timer. Keep the profile
  // invalid rather than reporting a successful zero-duration sample.
  return status;
}

absl::Status MusaBlas::GetVersion(std::string* version) {
  if (version == nullptr) {
    return absl::InvalidArgumentError("muBLAS version output is null");
  }
  absl::MutexLock lock(&mu_);
  if (handle_ == nullptr) {
    return absl::FailedPreconditionError("muBLAS is not initialized");
  }
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  int32_t encoded_version = 0;
  absl::Status status = api_->GetVersion(handle_, &encoded_version);
  if (!status.ok()) return status;
  *version = std::to_string(encoded_version);
  return absl::OkStatus();
}

bool MusaBlas::DoBlasScal(Stream*, uint64_t, float, DeviceAddress<float>*,
                          int) {
  return UnsupportedBool("SCAL f32");
}
bool MusaBlas::DoBlasScal(Stream*, uint64_t, double, DeviceAddress<double>*,
                          int) {
  return UnsupportedBool("SCAL f64");
}
bool MusaBlas::DoBlasScal(Stream*, uint64_t, float,
                          DeviceAddress<std::complex<float>>*, int) {
  return UnsupportedBool("SCAL complex-f32 by f32");
}
bool MusaBlas::DoBlasScal(Stream*, uint64_t, double,
                          DeviceAddress<std::complex<double>>*, int) {
  return UnsupportedBool("SCAL complex-f64 by f64");
}
bool MusaBlas::DoBlasScal(Stream*, uint64_t, std::complex<float>,
                          DeviceAddress<std::complex<float>>*, int) {
  return UnsupportedBool("SCAL complex-f32");
}
bool MusaBlas::DoBlasScal(Stream*, uint64_t, std::complex<double>,
                          DeviceAddress<std::complex<double>>*, int) {
  return UnsupportedBool("SCAL complex-f64");
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

bool MusaBlas::DoBlasGemmBatched(Stream*, blas::Transpose, blas::Transpose,
                                 uint64_t, uint64_t, uint64_t, float,
                                 DeviceAddressSlice<Eigen::half>, int,
                                 DeviceAddressSlice<Eigen::half>, int, float,
                                 DeviceAddressSlice<Eigen::half>, int, int,
                                 const EngineOptions&, ScratchAllocator*,
                                 blas::CallContext) {
  return UnsupportedBool("batched GEMM f16");
}
bool MusaBlas::DoBlasGemmBatched(Stream*, blas::Transpose, blas::Transpose,
                                 uint64_t, uint64_t, uint64_t, float,
                                 DeviceAddressSlice<Eigen::bfloat16>, int,
                                 DeviceAddressSlice<Eigen::bfloat16>, int,
                                 float, DeviceAddressSlice<Eigen::bfloat16>,
                                 int, int, const EngineOptions&,
                                 ScratchAllocator*, blas::CallContext) {
  return UnsupportedBool("batched GEMM bf16");
}
bool MusaBlas::DoBlasGemmBatched(Stream*, blas::Transpose, blas::Transpose,
                                 uint64_t, uint64_t, uint64_t, float,
                                 DeviceAddressSlice<float>, int,
                                 DeviceAddressSlice<float>, int, float,
                                 DeviceAddressSlice<float>, int, int,
                                 const EngineOptions&, ScratchAllocator*,
                                 blas::CallContext) {
  return UnsupportedBool("batched GEMM f32");
}
bool MusaBlas::DoBlasGemmBatched(Stream*, blas::Transpose, blas::Transpose,
                                 uint64_t, uint64_t, uint64_t, double,
                                 DeviceAddressSlice<double>, int,
                                 DeviceAddressSlice<double>, int, double,
                                 DeviceAddressSlice<double>, int, int,
                                 const EngineOptions&, ScratchAllocator*,
                                 blas::CallContext) {
  return UnsupportedBool("batched GEMM f64");
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

absl::Status MusaBlas::DoBlasGemmStridedBatched(
    Stream*, blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64_t,
    blas::DataType, const void*, const DeviceAddressBase&, int, int64_t,
    const DeviceAddressBase&, int, int64_t, const void*, DeviceAddressBase*,
    int, int64_t, int, const EngineOptions&, blas::CallContext) {
  return Unsupported("strided-batched GEMM");
}
absl::Status MusaBlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream*, blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64_t,
    const void*, const DeviceAddressBase&, blas::DataType, int, int64_t,
    const DeviceAddressBase&, blas::DataType, int, int64_t, const void*,
    DeviceAddressBase*, blas::DataType, int, int64_t, int,
    blas::ComputationType, blas::AlgorithmType, const EngineOptions&,
    blas::ProfileResult*, blas::CallContext) {
  return Unsupported("algorithm-selected strided-batched GEMM");
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
