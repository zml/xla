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

#include "xla/stream_executor/sycl/sycl_blas_lt.h"

#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "absl/time/time.h"
#include "oneapi/mkl/blas.hpp"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace stream_executor {
namespace sycl {
namespace {

bool EnvFlagEnabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
         value[0] == 't' || value[0] == 'T';
}

bool SyclGemvFastPathEnabled() {
  static const bool enabled =
      EnvFlagEnabled("XLA_SYCL_ENABLE_GEMV_FAST_PATH", false);
  return enabled;
}

bool SyclBlasLoggingEnabled() {
  static const bool enabled = EnvFlagEnabled("XLA_SYCL_LOG_BLAS", false);
  return enabled;
}

const char* TransposeName(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return "N";
    case blas::Transpose::kTranspose:
      return "T";
    case blas::Transpose::kConjugateTranspose:
      return "C";
  }
}

const char* DataTypeName(blas::DataType type) {
  switch (type) {
    case blas::DataType::kHalf:
      return "f16";
    case blas::DataType::kBF16:
      return "bf16";
    case blas::DataType::kFloat:
      return "f32";
    case blas::DataType::kDouble:
      return "f64";
    case blas::DataType::kComplexFloat:
      return "c64";
    case blas::DataType::kComplexDouble:
      return "c128";
    case blas::DataType::kInt8:
      return "s8";
    case blas::DataType::kInt32:
      return "s32";
    default:
      return "unknown";
  }
}

bool IsComplexDataType(blas::DataType type) {
  return type == blas::DataType::kComplexFloat ||
         type == blas::DataType::kComplexDouble;
}

blas::Transpose NormalizeRealTranspose(blas::Transpose trans) {
  return trans == blas::Transpose::kConjugateTranspose
             ? blas::Transpose::kTranspose
             : trans;
}

void LogSyclBlasCall(const char* path, blas::DataType dtype,
                     blas::Transpose transa, blas::Transpose transb,
                     uint64_t m, uint64_t n, uint64_t k, int lda, int ldb,
                     int ldc) {
  if (!SyclBlasLoggingEnabled()) {
    return;
  }
  static std::atomic<int> remaining{400};
  int old_remaining = remaining.fetch_sub(1, std::memory_order_relaxed);
  if (old_remaining <= 0) {
    return;
  }
  std::fprintf(stderr,
               "xla_sycl_blas path=%s dtype=%s transa=%s transb=%s "
               "m=%llu n=%llu k=%llu lda=%d ldb=%d ldc=%d\n",
               path, DataTypeName(dtype), TransposeName(transa),
               TransposeName(transb), static_cast<unsigned long long>(m),
               static_cast<unsigned long long>(n),
               static_cast<unsigned long long>(k), lda, ldb, ldc);
}

oneapi::mkl::transpose AsOneMklTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return oneapi::mkl::transpose::N;
    case blas::Transpose::kTranspose:
      return oneapi::mkl::transpose::T;
    case blas::Transpose::kConjugateTranspose:
      return oneapi::mkl::transpose::C;
  }
}

template <typename Scale>
Scale ReadScale(const void* value) {
  return *static_cast<const Scale*>(value);
}

template <>
::sycl::half ReadScale<::sycl::half>(const void* value) {
  return ::sycl::half(*static_cast<const float*>(value));
}

template <typename T>
absl::Status DoOneMklGemv(Stream* stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, const void* alpha,
                          const DeviceAddressBase& a, int lda,
                          const DeviceAddressBase& x, int incx,
                          const void* beta, DeviceAddressBase* y, int incy) {
  ::sycl::queue* queue = static_cast<SyclStream*>(stream)->stream_handle();
  if (queue == nullptr) {
    return absl::InternalError("SYCL GEMV stream has null queue");
  }

  oneapi::mkl::blas::gemv(
      *queue, AsOneMklTranspose(trans), static_cast<std::int64_t>(m),
      static_cast<std::int64_t>(n), ReadScale<T>(alpha),
      static_cast<const T*>(a.opaque()), static_cast<std::int64_t>(lda),
      static_cast<const T*>(x.opaque()), static_cast<std::int64_t>(incx),
      ReadScale<T>(beta), static_cast<T*>(y->opaque()),
      static_cast<std::int64_t>(incy));
  return absl::OkStatus();
}

template <typename AType, typename CType, typename Scale>
absl::Status DoOneMklGemm(Stream* stream, blas::Transpose transa,
                          blas::Transpose transb, uint64_t m, uint64_t n,
                          uint64_t k, const void* alpha,
                          const DeviceAddressBase& a, int lda,
                          const DeviceAddressBase& b, int ldb,
                          const void* beta, DeviceAddressBase* c, int ldc) {
  ::sycl::queue* queue = static_cast<SyclStream*>(stream)->stream_handle();
  if (queue == nullptr) {
    return absl::InternalError("SYCL GEMM stream has null queue");
  }

  oneapi::mkl::blas::gemm(
      *queue, AsOneMklTranspose(transa), AsOneMklTranspose(transb),
      static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
      static_cast<std::int64_t>(k), ReadScale<Scale>(alpha),
      static_cast<const AType*>(a.opaque()), static_cast<std::int64_t>(lda),
      static_cast<const AType*>(b.opaque()), static_cast<std::int64_t>(ldb),
      ReadScale<Scale>(beta), static_cast<CType*>(c->opaque()),
      static_cast<std::int64_t>(ldc));
  return absl::OkStatus();
}

template <typename AType, typename CType, typename Scale>
absl::Status DoOneMklGemmStridedBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    int lda, int64_t stride_a, const DeviceAddressBase& b, int ldb,
    int64_t stride_b, const void* beta, DeviceAddressBase* c, int ldc,
    int64_t stride_c, int batch_count) {
  ::sycl::queue* queue = static_cast<SyclStream*>(stream)->stream_handle();
  if (queue == nullptr) {
    return absl::InternalError("SYCL batched GEMM stream has null queue");
  }

  oneapi::mkl::blas::gemm_batch(
      *queue, AsOneMklTranspose(transa), AsOneMklTranspose(transb),
      static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
      static_cast<std::int64_t>(k), ReadScale<Scale>(alpha),
      static_cast<const AType*>(a.opaque()), static_cast<std::int64_t>(lda),
      static_cast<std::int64_t>(stride_a),
      static_cast<const AType*>(b.opaque()), static_cast<std::int64_t>(ldb),
      static_cast<std::int64_t>(stride_b), ReadScale<Scale>(beta),
      static_cast<CType*>(c->opaque()), static_cast<std::int64_t>(ldc),
      static_cast<std::int64_t>(stride_c),
      static_cast<std::int64_t>(batch_count));
  return absl::OkStatus();
}

absl::Status DispatchOneMklGemv(
    Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n,
    const void* alpha, const DeviceAddressBase& a, int lda,
    const DeviceAddressBase& x, int incx, const void* beta,
    DeviceAddressBase* y, int incy, blas::DataType type) {
  switch (type) {
    case blas::DataType::kFloat:
      return DoOneMklGemv<float>(stream, trans, m, n, alpha, a, lda, x, incx,
                                 beta, y, incy);
    case blas::DataType::kDouble:
      return DoOneMklGemv<double>(stream, trans, m, n, alpha, a, lda, x, incx,
                                  beta, y, incy);
    case blas::DataType::kComplexFloat:
      return DoOneMklGemv<std::complex<float>>(stream, trans, m, n, alpha, a,
                                               lda, x, incx, beta, y, incy);
    case blas::DataType::kComplexDouble:
      return DoOneMklGemv<std::complex<double>>(stream, trans, m, n, alpha, a,
                                                lda, x, incx, beta, y, incy);
    default:
      return absl::InternalError("Unsupported SYCL GEMV datatype");
  }
}

template <typename T>
bool DoBlasGemvImpl(Stream* stream, blas::Transpose trans, uint64_t m,
                    uint64_t n, T alpha, const DeviceAddress<T>& a, int lda,
                    const DeviceAddress<T>& x, int incx, T beta,
                    DeviceAddress<T>* y, int incy, blas::DataType type) {
  DeviceAddressBase y_base(*y);
  LogSyclBlasCall("gemv", type, trans, blas::Transpose::kNoTranspose, m, n,
                  /*k=*/0, lda, /*ldb=*/incx, /*ldc=*/incy);
  try {
    absl::Status status =
        DispatchOneMklGemv(stream, trans, m, n, &alpha, a, lda, x, incx, &beta,
                           &y_base, incy, type);
    if (!status.ok()) {
      LOG(ERROR) << status;
      return false;
    }
  } catch (const oneapi::mkl::exception& e) {
    LOG(ERROR) << e.what();
    return false;
  } catch (const ::sycl::exception& e) {
    LOG(ERROR) << e.what();
    return false;
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return false;
  }
  return true;
}

absl::StatusOr<bool> TryDispatchOneMklGemvFastPath(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, const DeviceAddressBase& b,
    blas::DataType type_b, int ldb, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc) {
  if (!SyclGemvFastPathEnabled() || (m != 1 && n != 1) || k == 0 ||
      type_a != type_b || type_a != type_c) {
    return false;
  }

  switch (type_a) {
    case blas::DataType::kFloat:
    case blas::DataType::kDouble:
    case blas::DataType::kComplexFloat:
    case blas::DataType::kComplexDouble:
      break;
    default:
      return false;
  }

  const bool is_complex = IsComplexDataType(type_a);
  if (is_complex && (transa == blas::Transpose::kConjugateTranspose ||
                     transb == blas::Transpose::kConjugateTranspose)) {
    return false;
  }

  transa = is_complex ? transa : NormalizeRealTranspose(transa);
  transb = is_complex ? transb : NormalizeRealTranspose(transb);

  if (n == 1) {
    int incx = transb == blas::Transpose::kNoTranspose ? 1 : ldb;
    TF_RETURN_IF_ERROR(DispatchOneMklGemv(stream, transa, m, k, alpha, a, lda,
                                          b, incx, beta, c, 1, type_a));
    return true;
  }

  int incx = transa == blas::Transpose::kNoTranspose ? lda : 1;
  blas::Transpose trans = blas::Transpose::kNoTranspose;
  uint64_t rows = n;
  uint64_t cols = k;
  if (transb == blas::Transpose::kNoTranspose) {
    trans = blas::Transpose::kTranspose;
    rows = k;
    cols = n;
  }
  TF_RETURN_IF_ERROR(DispatchOneMklGemv(stream, trans, rows, cols, alpha, b,
                                        ldb, a, incx, beta, c, ldc, type_a));
  return true;
}

absl::Status DispatchOneMklGemm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, const DeviceAddressBase& b,
    blas::DataType type_b, int ldb, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type) {
  if (type_a != type_b) {
    return absl::InternalError(
        "SYCL GEMM requires matching lhs/rhs datatypes");
  }

  switch (type_a) {
    case blas::DataType::kHalf:
      if (type_c == blas::DataType::kHalf) {
        return DoOneMklGemm<::sycl::half, ::sycl::half, ::sycl::half>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      if (type_c == blas::DataType::kFloat) {
        return DoOneMklGemm<::sycl::half, float, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      break;
    case blas::DataType::kBF16:
      if (type_c == blas::DataType::kBF16) {
        return DoOneMklGemm<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                            float>(stream, transa, transb, m, n, k, alpha, a,
                                   lda, b, ldb, beta, c, ldc);
      }
      if (type_c == blas::DataType::kFloat) {
        return DoOneMklGemm<oneapi::mkl::bfloat16, float, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      break;
    case blas::DataType::kFloat:
      if (type_c == blas::DataType::kFloat) {
        return DoOneMklGemm<float, float, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      break;
    case blas::DataType::kDouble:
      if (type_c == blas::DataType::kDouble) {
        return DoOneMklGemm<double, double, double>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      break;
    case blas::DataType::kComplexFloat:
      if (type_c == blas::DataType::kComplexFloat) {
        return DoOneMklGemm<std::complex<float>, std::complex<float>,
                            std::complex<float>>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      break;
    case blas::DataType::kComplexDouble:
      if (type_c == blas::DataType::kComplexDouble) {
        return DoOneMklGemm<std::complex<double>, std::complex<double>,
                            std::complex<double>>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      break;
    case blas::DataType::kInt8:
      if (type_c == blas::DataType::kInt32) {
        return DoOneMklGemm<std::int8_t, std::int32_t, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      if (type_c == blas::DataType::kFloat) {
        return DoOneMklGemm<std::int8_t, float, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
            ldc);
      }
      break;
    default:
      break;
  }

  return absl::InternalError("Unsupported SYCL GEMM datatype combination");
}

absl::Status DispatchOneMklGemmStridedBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, blas::DataType type_b, int ldb,
    int64_t stride_b, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, int64_t stride_c, int batch_count,
    blas::ComputationType computation_type) {
  if (type_a != type_b) {
    return absl::InternalError(
        "SYCL batched GEMM requires matching lhs/rhs datatypes");
  }

  switch (type_a) {
    case blas::DataType::kHalf:
      if (type_c == blas::DataType::kHalf) {
        return DoOneMklGemmStridedBatched<::sycl::half, ::sycl::half,
                                          ::sycl::half>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      if (type_c == blas::DataType::kFloat) {
        return DoOneMklGemmStridedBatched<::sycl::half, float, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      break;
    case blas::DataType::kBF16:
      if (type_c == blas::DataType::kBF16) {
        return DoOneMklGemmStridedBatched<oneapi::mkl::bfloat16,
                                          oneapi::mkl::bfloat16, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      if (type_c == blas::DataType::kFloat) {
        return DoOneMklGemmStridedBatched<oneapi::mkl::bfloat16, float, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      break;
    case blas::DataType::kFloat:
      if (type_c == blas::DataType::kFloat) {
        return DoOneMklGemmStridedBatched<float, float, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      break;
    case blas::DataType::kDouble:
      if (type_c == blas::DataType::kDouble) {
        return DoOneMklGemmStridedBatched<double, double, double>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      break;
    case blas::DataType::kComplexFloat:
      if (type_c == blas::DataType::kComplexFloat) {
        return DoOneMklGemmStridedBatched<std::complex<float>,
                                          std::complex<float>,
                                          std::complex<float>>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      break;
    case blas::DataType::kComplexDouble:
      if (type_c == blas::DataType::kComplexDouble) {
        return DoOneMklGemmStridedBatched<std::complex<double>,
                                          std::complex<double>,
                                          std::complex<double>>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      break;
    case blas::DataType::kInt8:
      if (type_c == blas::DataType::kInt32) {
        return DoOneMklGemmStridedBatched<std::int8_t, std::int32_t, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      if (type_c == blas::DataType::kFloat) {
        return DoOneMklGemmStridedBatched<std::int8_t, float, float>(
            stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
            stride_b, beta, c, ldc, stride_c, batch_count);
      }
      break;
    default:
      break;
  }

  return absl::InternalError(
      "Unsupported SYCL batched GEMM datatype combination");
}

absl::Status PopulateProfileFromTimer(
    EventBasedTimer* timer, blas::AlgorithmType algorithm,
    blas::ProfileResult* output_profile_result) {
  if (output_profile_result == nullptr) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(absl::Duration duration, timer->GetElapsedDuration());
  output_profile_result->set_is_valid(true);
  output_profile_result->set_algorithm(algorithm);
  output_profile_result->set_elapsed_time_in_ms(
      absl::ToDoubleMilliseconds(duration));
  return absl::OkStatus();
}

}  // namespace

absl::Status BlasLt::Init() { return absl::OkStatus(); }

auto BlasLt::GetMatmulPlan(const gpu::GemmConfig& config,
                           Epilogue epilogue) const
    -> absl::StatusOr<MatmulPlanPtr> {
  absl::MutexLock lock(&mu_);
  return std::make_unique<MatmulPlan>(config, epilogue);
}

absl::StatusOr<BlasLt::MatmulPlanPtr> BlasLt::GetGroupedMatmulPlan(
    gpu::GroupedGemmConfig& config, Epilogue epilogue) const {
  return absl::UnimplementedError(
      "Grouped GEMM is not supported for Sycl BlasLt");
}

absl::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, const gpu::BlasLt::MemoryArgs& args,
    blas::ProfileResult* profile_result) const {
  return absl::UnimplementedError(
      "SyclBlasLt MatmulPlan::ExecuteOnStream not implemented");
}

auto BlasLt::MatmulPlan::GetAlgorithms(const Stream* stream,
                                       size_t max_algorithm_count,
                                       size_t max_workspace_size) const
    -> absl::StatusOr<std::vector<MatmulAlgorithm>> {
  absl::MutexLock lock(&mu_);
  std::vector<MatmulAlgorithm> algorithms;
  algorithms.push_back({/*algorithm_id*/ kOneDnnGemm, /*workspace_size*/ 0});
  return std::move(algorithms);
}

SyclBlasSupport::SyclBlasSupport(StreamExecutor* parent) : blas_lt_(parent) {}

SyclBlasSupport::~SyclBlasSupport() {}

bool SyclBlasSupport::Init() { return true; }

bool SyclBlasSupport::DoBlasGemv(Stream* stream, blas::Transpose trans,
                                 uint64_t m, uint64_t n, float alpha,
                                 const DeviceAddress<float>& a, int lda,
                                 const DeviceAddress<float>& x, int incx,
                                 float beta, DeviceAddress<float>* y,
                                 int incy) {
  return DoBlasGemvImpl(stream, trans, m, n, alpha, a, lda, x, incx, beta, y,
                        incy, blas::DataType::kFloat);
}

bool SyclBlasSupport::DoBlasGemv(Stream* stream, blas::Transpose trans,
                                 uint64_t m, uint64_t n, double alpha,
                                 const DeviceAddress<double>& a, int lda,
                                 const DeviceAddress<double>& x, int incx,
                                 double beta, DeviceAddress<double>* y,
                                 int incy) {
  return DoBlasGemvImpl(stream, trans, m, n, alpha, a, lda, x, incx, beta, y,
                        incy, blas::DataType::kDouble);
}

bool SyclBlasSupport::DoBlasGemv(
    Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n,
    std::complex<float> alpha, const DeviceAddress<std::complex<float>>& a,
    int lda, const DeviceAddress<std::complex<float>>& x, int incx,
    std::complex<float> beta, DeviceAddress<std::complex<float>>* y,
    int incy) {
  return DoBlasGemvImpl(stream, trans, m, n, alpha, a, lda, x, incx, beta, y,
                        incy, blas::DataType::kComplexFloat);
}

bool SyclBlasSupport::DoBlasGemv(
    Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n,
    std::complex<double> alpha, const DeviceAddress<std::complex<double>>& a,
    int lda, const DeviceAddress<std::complex<double>>& x, int incx,
    std::complex<double> beta, DeviceAddress<std::complex<double>>* y,
    int incy) {
  return DoBlasGemvImpl(stream, trans, m, n, alpha, a, lda, x, incx, beta, y,
                        incy, blas::DataType::kComplexDouble);
}

absl::Status SyclBlasSupport::DoBlasGemm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, blas::DataType dtype, const void* alpha,
    const DeviceAddressBase& a, int lda, const DeviceAddressBase& b, int ldb,
    const void* beta, DeviceAddressBase* c, int ldc,
    const EngineOptions& engine_options, blas::CallContext context) {
  blas::ComputationType computation_type = [&] {
    switch (dtype) {
      case blas::DataType::kDouble:
      case blas::DataType::kComplexDouble:
        return blas::ComputationType::kF64;
      case blas::DataType::kInt8:
        return blas::ComputationType::kI32;
      default:
        return blas::ComputationType::kF32;
    }
  }();
  return DoBlasGemmWithAlgorithm(
      stream, transa, transb, m, n, k, alpha, a, dtype, lda, b, dtype, ldb,
      beta, c, dtype, ldc, computation_type, blas::kDefaultAlgorithm,
      engine_options, /*output_profile_result=*/nullptr, context);
}

absl::Status SyclBlasSupport::DoBlasGemmWithAlgorithm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, const DeviceAddressBase& b,
    blas::DataType type_b, int ldb, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const EngineOptions& engine_options,
    blas::ProfileResult* output_profile_result, blas::CallContext context) {
  std::unique_ptr<EventBasedTimer> timer;
  if (output_profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   output_profile_result->warmup_run_executed()));
  }

  try {
    TF_ASSIGN_OR_RETURN(bool dispatched_gemv,
                        TryDispatchOneMklGemvFastPath(
                            stream, transa, transb, m, n, k, alpha, a, type_a,
                            lda, b, type_b, ldb, beta, c, type_c, ldc));
    if (dispatched_gemv) {
      LogSyclBlasCall("gemv-fast-path", type_a, transa, transb, m, n, k, lda,
                      ldb, ldc);
    } else {
      LogSyclBlasCall("gemm", type_a, transa, transb, m, n, k, lda, ldb, ldc);
      TF_RETURN_IF_ERROR(DispatchOneMklGemm(
          stream, transa, transb, m, n, k, alpha, a, type_a, lda, b, type_b,
          ldb, beta, c, type_c, ldc, computation_type));
    }
    TF_RETURN_IF_ERROR(
        PopulateProfileFromTimer(timer.get(), algorithm, output_profile_result));
  } catch (const oneapi::mkl::exception& e) {
    return absl::InternalError(e.what());
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(e.what());
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }

  return absl::OkStatus();
}

absl::Status SyclBlasSupport::DoBlasGemmStridedBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, blas::DataType dtype, const void* alpha,
    const DeviceAddressBase& a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, int ldb, int64_t stride_b, const void* beta,
    DeviceAddressBase* c, int ldc, int64_t stride_c, int batch_count,
    const EngineOptions& engine_options, blas::CallContext context) {
  blas::ComputationType computation_type = [&] {
    switch (dtype) {
      case blas::DataType::kDouble:
      case blas::DataType::kComplexDouble:
        return blas::ComputationType::kF64;
      case blas::DataType::kInt8:
        return blas::ComputationType::kI32;
      default:
        return blas::ComputationType::kF32;
    }
  }();
  return DoBlasGemmStridedBatchedWithAlgorithm(
      stream, transa, transb, m, n, k, alpha, a, dtype, lda, stride_a, b, dtype,
      ldb, stride_b, beta, c, dtype, ldc, stride_c, batch_count,
      computation_type, blas::kDefaultAlgorithm, engine_options,
      /*output_profile_result=*/nullptr, context);
}

absl::Status SyclBlasSupport::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void* alpha, const DeviceAddressBase& a,
    blas::DataType type_a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, blas::DataType type_b, int ldb,
    int64_t stride_b, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, int64_t stride_c, int batch_count,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    const EngineOptions& engine_options,
    blas::ProfileResult* output_profile_result, blas::CallContext context) {
  std::unique_ptr<EventBasedTimer> timer;
  if (output_profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   output_profile_result->warmup_run_executed()));
  }

  try {
    TF_RETURN_IF_ERROR(DispatchOneMklGemmStridedBatched(
        stream, transa, transb, m, n, k, alpha, a, type_a, lda, stride_a, b,
        type_b, ldb, stride_b, beta, c, type_c, ldc, stride_c, batch_count,
        computation_type));
    TF_RETURN_IF_ERROR(
        PopulateProfileFromTimer(timer.get(), algorithm, output_profile_result));
  } catch (const oneapi::mkl::exception& e) {
    return absl::InternalError(e.what());
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(e.what());
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }

  return absl::OkStatus();
}

static void RegisterSyclBlasSupport() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::BlasFactory>(
          stream_executor::sycl::kSyclPlatformId, "syclBLAS",
          [](StreamExecutor* parent) -> blas::BlasSupport* {
            auto* blas = new SyclBlasSupport(parent);
            if (!blas->Init()) {
              delete blas;
              return nullptr;
            }
            return blas;
          });
  if (!status.ok()) {
    std::cerr << "Unable to register sycl_blas factory: " << status.message()
              << std::endl;
  }
}

}  // namespace sycl
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    syclblas, stream_executor::sycl::RegisterSyclBlasSupport());
