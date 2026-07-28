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

#include <cstdint>
#include <cstring>
#include <limits>

#include "mublas.h"
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_abi.h"
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_float16.h"

// The MUSA 4.0.1 host headers expose mublasHgemm only to __MUSACC__, although
// libmublas exports the C symbol for host callers. Declare that exported ABI
// explicitly when this shim is compiled with the host C++ compiler.
#if !defined(__MUSACC__)
extern "C" MUBLAS_EXPORT mublasStatus mublasHgemm(
    mublasHandle_t handle, mublasOperation_t trans_a, mublasOperation_t trans_b,
    mublas_int m, mublas_int n, mublas_int k, const mublas_half* alpha,
    const mublas_half* a, mublas_int lda, const mublas_half* b, mublas_int ldb,
    const mublas_half* beta, mublas_half* c, mublas_int ldc);
extern "C" MUBLAS_EXPORT mublasStatus mublasHgemmBatched(
    mublasHandle_t handle, mublasOperation_t trans_a, mublasOperation_t trans_b,
    mublas_int m, mublas_int n, mublas_int k, const mublas_half* alpha,
    const mublas_half* const a[], mublas_int lda, const mublas_half* const b[],
    mublas_int ldb, const mublas_half* beta, mublas_half* const c[],
    mublas_int ldc, mublas_int batch_count);
extern "C" MUBLAS_EXPORT mublasStatus mublasHgemmStridedBatched(
    mublasHandle_t handle, mublasOperation_t trans_a, mublasOperation_t trans_b,
    mublas_int m, mublas_int n, mublas_int k, const mublas_half* alpha,
    const mublas_half* a, mublas_int lda, long long stride_a,
    const mublas_half* b, mublas_int ldb, long long stride_b,
    const mublas_half* beta, mublas_half* c, mublas_int ldc, long long stride_c,
    mublas_int batch_count);
#endif

namespace {

using ::stream_executor::musa::mublas_shim_internal::Float32ToFloat16Bits;

XlaMusaMuBlasStatus ToShimStatus(mublasStatus status) {
  switch (status) {
    case MUBLAS_STATUS_SUCCESS:
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case MUBLAS_STATUS_INVALID_HANDLE:
    case MUBLAS_STATUS_INVALID_POINTER:
    case MUBLAS_STATUS_INVALID_SIZE:
    case MUBLAS_STATUS_INVALID_VALUE:
      return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
    case MUBLAS_STATUS_NOT_IMPLEMENTED:
      return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
    case MUBLAS_STATUS_MEMORY_ERROR:
      return XLA_MUSA_MUBLAS_STATUS_RESOURCE_EXHAUSTED;
    default:
      return XLA_MUSA_MUBLAS_STATUS_VENDOR_ERROR;
  }
}

XlaMusaMuBlasStatus Create(void** handle) {
  if (handle == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  *handle = nullptr;
  mublasHandle_t native_handle = nullptr;
  mublasStatus status = mublasCreate(&native_handle);
  if (status != MUBLAS_STATUS_SUCCESS) {
    return ToShimStatus(status);
  }
  status = mublasSetPointerMode(native_handle, MUBLAS_POINTER_MODE_HOST);
  if (status != MUBLAS_STATUS_SUCCESS) {
    mublasDestroy(native_handle);
    return ToShimStatus(status);
  }
  *handle = native_handle;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus Destroy(void* handle) {
  if (handle == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mublasDestroy(static_cast<mublasHandle_t>(handle)));
}

XlaMusaMuBlasStatus SetStream(void* handle, void* stream) {
  if (handle == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mublasSetStream(static_cast<mublasHandle_t>(handle),
                                      static_cast<MUstream>(stream)));
}

XlaMusaMuBlasStatus GetVersion(void* handle, int32_t* version) {
  if (handle == nullptr || version == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  int native_version = 0;
  mublasStatus status =
      mublasGetVersion(static_cast<mublasHandle_t>(handle), &native_version);
  if (status == MUBLAS_STATUS_SUCCESS) {
    *version = static_cast<int32_t>(native_version);
  }
  return ToShimStatus(status);
}

bool FitsNonNegativeInt32(int64_t value) {
  return value >= 0 &&
         value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

bool FitsPositiveInt32(int64_t value) {
  return value > 0 &&
         value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

bool FitsNonNegativeLongLong(int64_t value) {
  return value >= 0 &&
         static_cast<uint64_t>(value) <=
             static_cast<uint64_t>(std::numeric_limits<long long>::max());
}

XlaMusaMuBlasStatus ToOperation(XlaMusaMuBlasOperation operation,
                                mublasOperation_t* native_operation) {
  if (native_operation == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  switch (operation) {
    case XLA_MUSA_MUBLAS_OPERATION_NONE:
      *native_operation = MUBLAS_OP_N;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_OPERATION_TRANSPOSE:
      *native_operation = MUBLAS_OP_T;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_OPERATION_CONJUGATE_TRANSPOSE:
      *native_operation = MUBLAS_OP_C;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
}

XlaMusaMuBlasStatus ToDataType(XlaMusaMuBlasDataType data_type,
                               musaDataType_t* native_data_type) {
  if (native_data_type == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  switch (data_type) {
    case XLA_MUSA_MUBLAS_DATA_TYPE_F16:
      *native_data_type = MUSA_R_16F;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_DATA_TYPE_F32:
      *native_data_type = MUSA_R_32F;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_DATA_TYPE_F64:
      *native_data_type = MUSA_R_64F;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  }
}

XlaMusaMuBlasStatus ToComputeType(XlaMusaMuBlasComputeType compute_type,
                                  mublasComputeType_t* native_compute_type) {
  if (native_compute_type == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  switch (compute_type) {
    case XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16:
      *native_compute_type = MUBLAS_COMPUTE_16F;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32:
      // ABI v1 deliberately uses full F32, never FAST_TF32. XLA HIGH and
      // HIGHEST requests are therefore honored with stronger precision.
      *native_compute_type = MUBLAS_COMPUTE_32F;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_COMPUTE_TYPE_F64:
      *native_compute_type = MUBLAS_COMPUTE_64F;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  }
}

bool IsSupportedTypeCombination(XlaMusaMuBlasDataType input_type,
                                XlaMusaMuBlasDataType output_type,
                                XlaMusaMuBlasComputeType compute_type) {
  return (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F16 &&
          output_type == XLA_MUSA_MUBLAS_DATA_TYPE_F16 &&
          compute_type == XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16) ||
         (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F32 &&
          output_type == XLA_MUSA_MUBLAS_DATA_TYPE_F32 &&
          compute_type == XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32) ||
         (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F64 &&
          output_type == XLA_MUSA_MUBLAS_DATA_TYPE_F64 &&
          compute_type == XLA_MUSA_MUBLAS_COMPUTE_TYPE_F64);
}

XlaMusaMuBlasStatus ToAlgorithm(XlaMusaMuBlasAlgorithm algorithm,
                                XlaMusaMuBlasDataType input_type,
                                XlaMusaMuBlasDataType output_type,
                                XlaMusaMuBlasComputeType compute_type,
                                mublasGemmAlgo_t* native_algorithm) {
  if (native_algorithm == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  switch (algorithm) {
    case XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT:
      *native_algorithm = MUBLAS_GEMM_DEFAULT;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP:
      if (input_type != XLA_MUSA_MUBLAS_DATA_TYPE_F32 ||
          output_type != XLA_MUSA_MUBLAS_DATA_TYPE_F32 ||
          compute_type != XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32) {
        return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
      }
      *native_algorithm = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
}

XlaMusaMuBlasStatus ValidateDimensions(int64_t m, int64_t n, int64_t k,
                                       int64_t lda, int64_t ldb, int64_t ldc) {
  if (!FitsNonNegativeInt32(m) || !FitsNonNegativeInt32(n) ||
      !FitsNonNegativeInt32(k)) {
    return XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE;
  }
  if (!FitsPositiveInt32(lda) || !FitsPositiveInt32(ldb) ||
      !FitsPositiveInt32(ldc)) {
    return XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE;
  }
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus SetAtomicsMode(void* handle, uint32_t allow_atomics) {
  if (handle == nullptr || allow_atomics > 1) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mublasSetAtomicsMode(static_cast<mublasHandle_t>(handle),
                                           allow_atomics != 0
                                               ? MUBLAS_ATOMICS_ALLOWED
                                               : MUBLAS_ATOMICS_NOT_ALLOWED));
}

XlaMusaMuBlasStatus GemmWithAlgorithm(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
    const void* b, int64_t ldb, const void* beta, void* c, int64_t ldc,
    XlaMusaMuBlasAlgorithm algorithm) {
  if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr ||
      beta == nullptr || c == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  if (!IsSupportedTypeCombination(input_type, output_type, compute_type)) {
    return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  }
  XlaMusaMuBlasStatus status = ValidateDimensions(m, n, k, lda, ldb, ldc);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;

  mublasGemmAlgo_t native_algorithm;
  status = ToAlgorithm(algorithm, input_type, output_type, compute_type,
                       &native_algorithm);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;

  mublasOperation_t native_trans_a;
  mublasOperation_t native_trans_b;
  status = ToOperation(trans_a, &native_trans_a);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = ToOperation(trans_b, &native_trans_b);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) {
    return status;
  }

  if (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F16) {
    // The SDK-free ABI follows StreamExecutor's F32 scalar contract for F16.
    // mublasHgemm expects F16 scalar pointers, so narrow at the vendor
    // boundary.
    float alpha_f32 = 0.0f;
    float beta_f32 = 0.0f;
    std::memcpy(&alpha_f32, alpha, sizeof(alpha_f32));
    std::memcpy(&beta_f32, beta, sizeof(beta_f32));
    const mublas_half alpha_f16 = Float32ToFloat16Bits(alpha_f32);
    const mublas_half beta_f16 = Float32ToFloat16Bits(beta_f32);
    return ToShimStatus(mublasHgemm(
        static_cast<mublasHandle_t>(handle), native_trans_a, native_trans_b,
        static_cast<mublas_int>(m), static_cast<mublas_int>(n),
        static_cast<mublas_int>(k), &alpha_f16,
        static_cast<const mublas_half*>(a), static_cast<mublas_int>(lda),
        static_cast<const mublas_half*>(b), static_cast<mublas_int>(ldb),
        &beta_f16, static_cast<mublas_half*>(c), static_cast<mublas_int>(ldc)));
  }

  musaDataType_t native_input_type;
  musaDataType_t native_output_type;
  mublasComputeType_t native_compute_type;
  status = ToDataType(input_type, &native_input_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = ToDataType(output_type, &native_output_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = ToComputeType(compute_type, &native_compute_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) {
    return status;
  }

  return ToShimStatus(mublasGemmEx(
      static_cast<mublasHandle_t>(handle), native_trans_a, native_trans_b,
      static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k),
      alpha, a, native_input_type, static_cast<int32_t>(lda), b,
      native_input_type, static_cast<int32_t>(ldb), beta, c, native_output_type,
      static_cast<int32_t>(ldc), native_compute_type, native_algorithm));
}

XlaMusaMuBlasStatus Gemm(void* handle, XlaMusaMuBlasDataType input_type,
                         XlaMusaMuBlasDataType output_type,
                         XlaMusaMuBlasComputeType compute_type,
                         XlaMusaMuBlasOperation trans_a,
                         XlaMusaMuBlasOperation trans_b, int64_t m, int64_t n,
                         int64_t k, const void* alpha, const void* a,
                         int64_t lda, const void* b, int64_t ldb,
                         const void* beta, void* c, int64_t ldc) {
  return GemmWithAlgorithm(handle, input_type, output_type, compute_type,
                           trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb,
                           beta, c, ldc, XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT);
}

XlaMusaMuBlasStatus GemmBatched(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* const* a, int64_t lda,
    const void* const* b, int64_t ldb, const void* beta, void* const* c,
    int64_t ldc, int64_t batch_count, XlaMusaMuBlasAlgorithm algorithm) {
  if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr ||
      beta == nullptr || c == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  if (!IsSupportedTypeCombination(input_type, output_type, compute_type)) {
    return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  }
  XlaMusaMuBlasStatus status = ValidateDimensions(m, n, k, lda, ldb, ldc);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  if (!FitsNonNegativeInt32(batch_count)) {
    return XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE;
  }
  mublasGemmAlgo_t native_algorithm;
  status = ToAlgorithm(algorithm, input_type, output_type, compute_type,
                       &native_algorithm);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  mublasOperation_t native_trans_a;
  mublasOperation_t native_trans_b;
  status = ToOperation(trans_a, &native_trans_a);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToOperation(trans_b, &native_trans_b);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;

  if (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F16) {
    float alpha_f32 = 0.0f;
    float beta_f32 = 0.0f;
    std::memcpy(&alpha_f32, alpha, sizeof(alpha_f32));
    std::memcpy(&beta_f32, beta, sizeof(beta_f32));
    const mublas_half alpha_f16 = Float32ToFloat16Bits(alpha_f32);
    const mublas_half beta_f16 = Float32ToFloat16Bits(beta_f32);
    return ToShimStatus(mublasHgemmBatched(
        static_cast<mublasHandle_t>(handle), native_trans_a, native_trans_b,
        static_cast<mublas_int>(m), static_cast<mublas_int>(n),
        static_cast<mublas_int>(k), &alpha_f16,
        reinterpret_cast<const mublas_half* const*>(a),
        static_cast<mublas_int>(lda),
        reinterpret_cast<const mublas_half* const*>(b),
        static_cast<mublas_int>(ldb), &beta_f16,
        reinterpret_cast<mublas_half* const*>(c), static_cast<mublas_int>(ldc),
        static_cast<mublas_int>(batch_count)));
  }

  musaDataType_t native_input_type;
  musaDataType_t native_output_type;
  mublasComputeType_t native_compute_type;
  status = ToDataType(input_type, &native_input_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToDataType(output_type, &native_output_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToComputeType(compute_type, &native_compute_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  return ToShimStatus(mublasGemmBatchedEx(
      static_cast<mublasHandle_t>(handle), native_trans_a, native_trans_b,
      static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k),
      alpha, a, native_input_type, static_cast<int32_t>(lda), b,
      native_input_type, static_cast<int32_t>(ldb), beta, c, native_output_type,
      static_cast<int32_t>(ldc), static_cast<int32_t>(batch_count),
      native_compute_type, native_algorithm));
}

XlaMusaMuBlasStatus GemmStridedBatched(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
    int64_t stride_a, const void* b, int64_t ldb, int64_t stride_b,
    const void* beta, void* c, int64_t ldc, int64_t stride_c,
    int64_t batch_count, XlaMusaMuBlasAlgorithm algorithm) {
  if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr ||
      beta == nullptr || c == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  if (!IsSupportedTypeCombination(input_type, output_type, compute_type)) {
    return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  }
  XlaMusaMuBlasStatus status = ValidateDimensions(m, n, k, lda, ldb, ldc);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  if (!FitsNonNegativeInt32(batch_count) ||
      !FitsNonNegativeLongLong(stride_a) ||
      !FitsNonNegativeLongLong(stride_b) ||
      !FitsNonNegativeLongLong(stride_c)) {
    return XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE;
  }
  mublasGemmAlgo_t native_algorithm;
  status = ToAlgorithm(algorithm, input_type, output_type, compute_type,
                       &native_algorithm);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  mublasOperation_t native_trans_a;
  mublasOperation_t native_trans_b;
  status = ToOperation(trans_a, &native_trans_a);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToOperation(trans_b, &native_trans_b);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;

  if (input_type == XLA_MUSA_MUBLAS_DATA_TYPE_F16) {
    float alpha_f32 = 0.0f;
    float beta_f32 = 0.0f;
    std::memcpy(&alpha_f32, alpha, sizeof(alpha_f32));
    std::memcpy(&beta_f32, beta, sizeof(beta_f32));
    const mublas_half alpha_f16 = Float32ToFloat16Bits(alpha_f32);
    const mublas_half beta_f16 = Float32ToFloat16Bits(beta_f32);
    return ToShimStatus(mublasHgemmStridedBatched(
        static_cast<mublasHandle_t>(handle), native_trans_a, native_trans_b,
        static_cast<mublas_int>(m), static_cast<mublas_int>(n),
        static_cast<mublas_int>(k), &alpha_f16,
        static_cast<const mublas_half*>(a), static_cast<mublas_int>(lda),
        static_cast<long long>(stride_a), static_cast<const mublas_half*>(b),
        static_cast<mublas_int>(ldb), static_cast<long long>(stride_b),
        &beta_f16, static_cast<mublas_half*>(c), static_cast<mublas_int>(ldc),
        static_cast<long long>(stride_c),
        static_cast<mublas_int>(batch_count)));
  }

  musaDataType_t native_input_type;
  musaDataType_t native_output_type;
  mublasComputeType_t native_compute_type;
  status = ToDataType(input_type, &native_input_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToDataType(output_type, &native_output_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToComputeType(compute_type, &native_compute_type);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  return ToShimStatus(mublasGemmStridedBatchedEx(
      static_cast<mublasHandle_t>(handle), native_trans_a, native_trans_b,
      static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k),
      alpha, a, native_input_type, static_cast<int32_t>(lda),
      static_cast<long long>(stride_a), b, native_input_type,
      static_cast<int32_t>(ldb), static_cast<long long>(stride_b), beta, c,
      native_output_type, static_cast<int32_t>(ldc),
      static_cast<long long>(stride_c), static_cast<int32_t>(batch_count),
      native_compute_type, native_algorithm));
}

constexpr XlaMusaMuBlasCapabilities kCapabilities =
    XLA_MUSA_MUBLAS_CAPABILITIES_V1;

const XlaMusaMuBlasApiV1 kApiV1 = {sizeof(XlaMusaMuBlasApiV1),
                                   XLA_MUSA_MUBLAS_ABI_VERSION_1,
                                   kCapabilities,
                                   Create,
                                   Destroy,
                                   SetStream,
                                   GetVersion,
                                   Gemm};

const XlaMusaMuBlasApiV2 kApiV2 = {sizeof(XlaMusaMuBlasApiV2),
                                   XLA_MUSA_MUBLAS_ABI_VERSION_2,
                                   kCapabilities,
                                   XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2,
                                   Create,
                                   Destroy,
                                   SetStream,
                                   GetVersion,
                                   Gemm,
                                   SetAtomicsMode,
                                   GemmWithAlgorithm,
                                   GemmBatched,
                                   GemmStridedBatched};

}  // namespace

extern "C" XLA_MUSA_MUBLAS_SHIM_EXPORT const XlaMusaMuBlasApiV1*
xla_musa_mublas_get_api_v1(void) {
  return &kApiV1;
}

extern "C" XLA_MUSA_MUBLAS_SHIM_EXPORT const XlaMusaMuBlasApiV2*
xla_musa_mublas_get_api_v2(void) {
  return &kApiV2;
}
