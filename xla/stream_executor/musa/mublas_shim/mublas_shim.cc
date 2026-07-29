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

XlaMusaMuBlasStatus ToSide(XlaMusaMuBlasSide side,
                           mublasSideMode_t* native_side) {
  if (native_side == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  switch (side) {
    case XLA_MUSA_MUBLAS_SIDE_LEFT:
      *native_side = MUBLAS_SIDE_LEFT;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_SIDE_RIGHT:
      *native_side = MUBLAS_SIDE_RIGHT;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
}

XlaMusaMuBlasStatus ToFill(XlaMusaMuBlasFill fill,
                           mublasFillMode_t* native_fill) {
  if (native_fill == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  switch (fill) {
    case XLA_MUSA_MUBLAS_FILL_UPPER:
      *native_fill = MUBLAS_FILL_MODE_UPPER;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_FILL_LOWER:
      *native_fill = MUBLAS_FILL_MODE_LOWER;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
}

XlaMusaMuBlasStatus ToDiagonal(XlaMusaMuBlasDiagonal diagonal,
                               mublasDiagType_t* native_diagonal) {
  if (native_diagonal == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  switch (diagonal) {
    case XLA_MUSA_MUBLAS_DIAGONAL_NON_UNIT:
      *native_diagonal = MUBLAS_DIAG_NON_UNIT;
      return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
    case XLA_MUSA_MUBLAS_DIAGONAL_UNIT:
      *native_diagonal = MUBLAS_DIAG_UNIT;
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

XlaMusaMuBlasStatus Scal(void* handle, XlaMusaMuBlasScalType scal_type,
                         int64_t n, const void* alpha, void* x, int64_t incx) {
  if (handle == nullptr || alpha == nullptr || x == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  if (!FitsNonNegativeInt32(n) || !FitsPositiveInt32(incx)) {
    return XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE;
  }

  const mublas_int native_n = static_cast<mublas_int>(n);
  const mublas_int native_incx = static_cast<mublas_int>(incx);
  mublasHandle_t native_handle = static_cast<mublasHandle_t>(handle);
  switch (scal_type) {
    case XLA_MUSA_MUBLAS_SCAL_TYPE_F32:
      return ToShimStatus(mublasSscal(native_handle, native_n,
                                      static_cast<const float*>(alpha),
                                      static_cast<float*>(x), native_incx));
    case XLA_MUSA_MUBLAS_SCAL_TYPE_F64:
      return ToShimStatus(mublasDscal(native_handle, native_n,
                                      static_cast<const double*>(alpha),
                                      static_cast<double*>(x), native_incx));
    case XLA_MUSA_MUBLAS_SCAL_TYPE_C64: {
      muComplex native_alpha;
      static_assert(sizeof(native_alpha) == 2 * sizeof(float));
      std::memcpy(&native_alpha, alpha, sizeof(native_alpha));
      return ToShimStatus(mublasCscal(native_handle, native_n, &native_alpha,
                                      static_cast<muComplex*>(x), native_incx));
    }
    case XLA_MUSA_MUBLAS_SCAL_TYPE_C128: {
      muDoubleComplex native_alpha;
      static_assert(sizeof(native_alpha) == 2 * sizeof(double));
      std::memcpy(&native_alpha, alpha, sizeof(native_alpha));
      return ToShimStatus(mublasZscal(native_handle, native_n, &native_alpha,
                                      static_cast<muDoubleComplex*>(x),
                                      native_incx));
    }
    case XLA_MUSA_MUBLAS_SCAL_TYPE_C64_F32:
      return ToShimStatus(mublasCsscal(
          native_handle, native_n, static_cast<const float*>(alpha),
          static_cast<muComplex*>(x), native_incx));
    case XLA_MUSA_MUBLAS_SCAL_TYPE_C128_F64:
      return ToShimStatus(mublasZdscal(
          native_handle, native_n, static_cast<const double*>(alpha),
          static_cast<muDoubleComplex*>(x), native_incx));
    default:
      return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  }
}

XlaMusaMuBlasStatus ValidateTrsmDimensions(XlaMusaMuBlasSide side, int64_t m,
                                           int64_t n, int64_t lda,
                                           int64_t ldb) {
  if (!FitsNonNegativeInt32(m) || !FitsNonNegativeInt32(n) ||
      !FitsPositiveInt32(lda) || !FitsPositiveInt32(ldb)) {
    return XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE;
  }
  const int64_t required_lda =
      side == XLA_MUSA_MUBLAS_SIDE_LEFT ? (m > 1 ? m : 1) : (n > 1 ? n : 1);
  const int64_t required_ldb = m > 1 ? m : 1;
  if (lda < required_lda || ldb < required_ldb) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus Trsm(void* handle, XlaMusaMuBlasTrsmType trsm_type,
                         XlaMusaMuBlasSide side, XlaMusaMuBlasFill fill,
                         XlaMusaMuBlasOperation trans_a,
                         XlaMusaMuBlasDiagonal diagonal, int64_t m, int64_t n,
                         const void* alpha, const void* a, int64_t lda, void* b,
                         int64_t ldb) {
  if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  XlaMusaMuBlasStatus status = ValidateTrsmDimensions(side, m, n, lda, ldb);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;

  mublasSideMode_t native_side;
  mublasFillMode_t native_fill;
  mublasOperation_t native_trans_a;
  mublasDiagType_t native_diagonal;
  status = ToSide(side, &native_side);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToFill(fill, &native_fill);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToOperation(trans_a, &native_trans_a);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToDiagonal(diagonal, &native_diagonal);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;

  mublasHandle_t native_handle = static_cast<mublasHandle_t>(handle);
  const mublas_int native_m = static_cast<mublas_int>(m);
  const mublas_int native_n = static_cast<mublas_int>(n);
  const mublas_int native_lda = static_cast<mublas_int>(lda);
  const mublas_int native_ldb = static_cast<mublas_int>(ldb);
  switch (trsm_type) {
    case XLA_MUSA_MUBLAS_TRSM_TYPE_F32:
      return ToShimStatus(mublasStrsm(
          native_handle, native_side, native_fill, native_trans_a,
          native_diagonal, native_m, native_n, static_cast<const float*>(alpha),
          static_cast<const float*>(a), native_lda, static_cast<float*>(b),
          native_ldb));
    case XLA_MUSA_MUBLAS_TRSM_TYPE_F64:
      return ToShimStatus(mublasDtrsm(
          native_handle, native_side, native_fill, native_trans_a,
          native_diagonal, native_m, native_n,
          static_cast<const double*>(alpha), static_cast<const double*>(a),
          native_lda, static_cast<double*>(b), native_ldb));
    case XLA_MUSA_MUBLAS_TRSM_TYPE_C64: {
      muComplex native_alpha;
      static_assert(sizeof(native_alpha) == 2 * sizeof(float));
      std::memcpy(&native_alpha, alpha, sizeof(native_alpha));
      return ToShimStatus(
          mublasCtrsm(native_handle, native_side, native_fill, native_trans_a,
                      native_diagonal, native_m, native_n, &native_alpha,
                      static_cast<const muComplex*>(a), native_lda,
                      static_cast<muComplex*>(b), native_ldb));
    }
    case XLA_MUSA_MUBLAS_TRSM_TYPE_C128: {
      muDoubleComplex native_alpha;
      static_assert(sizeof(native_alpha) == 2 * sizeof(double));
      std::memcpy(&native_alpha, alpha, sizeof(native_alpha));
      return ToShimStatus(
          mublasZtrsm(native_handle, native_side, native_fill, native_trans_a,
                      native_diagonal, native_m, native_n, &native_alpha,
                      static_cast<const muDoubleComplex*>(a), native_lda,
                      static_cast<muDoubleComplex*>(b), native_ldb));
    }
    default:
      return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  }
}

XlaMusaMuBlasStatus TrsmBatched(void* handle, XlaMusaMuBlasTrsmType trsm_type,
                                XlaMusaMuBlasSide side, XlaMusaMuBlasFill fill,
                                XlaMusaMuBlasOperation trans_a,
                                XlaMusaMuBlasDiagonal diagonal, int64_t m,
                                int64_t n, const void* alpha,
                                const void* const* a, int64_t lda,
                                void* const* b, int64_t ldb,
                                int64_t batch_count) {
  if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr) {
    return XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT;
  }
  XlaMusaMuBlasStatus status = ValidateTrsmDimensions(side, m, n, lda, ldb);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  if (!FitsNonNegativeInt32(batch_count)) {
    return XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE;
  }

  mublasSideMode_t native_side;
  mublasFillMode_t native_fill;
  mublasOperation_t native_trans_a;
  mublasDiagType_t native_diagonal;
  status = ToSide(side, &native_side);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToFill(fill, &native_fill);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToOperation(trans_a, &native_trans_a);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;
  status = ToDiagonal(diagonal, &native_diagonal);
  if (status != XLA_MUSA_MUBLAS_STATUS_SUCCESS) return status;

  mublasHandle_t native_handle = static_cast<mublasHandle_t>(handle);
  const mublas_int native_m = static_cast<mublas_int>(m);
  const mublas_int native_n = static_cast<mublas_int>(n);
  const mublas_int native_lda = static_cast<mublas_int>(lda);
  const mublas_int native_ldb = static_cast<mublas_int>(ldb);
  const mublas_int native_batch_count = static_cast<mublas_int>(batch_count);
  switch (trsm_type) {
    case XLA_MUSA_MUBLAS_TRSM_TYPE_F32:
      return ToShimStatus(mublasStrsmBatched(
          native_handle, native_side, native_fill, native_trans_a,
          native_diagonal, native_m, native_n, static_cast<const float*>(alpha),
          reinterpret_cast<const float* const*>(a), native_lda,
          reinterpret_cast<float* const*>(b), native_ldb, native_batch_count));
    case XLA_MUSA_MUBLAS_TRSM_TYPE_F64:
      return ToShimStatus(mublasDtrsmBatched(
          native_handle, native_side, native_fill, native_trans_a,
          native_diagonal, native_m, native_n,
          static_cast<const double*>(alpha),
          reinterpret_cast<const double* const*>(a), native_lda,
          reinterpret_cast<double* const*>(b), native_ldb, native_batch_count));
    case XLA_MUSA_MUBLAS_TRSM_TYPE_C64: {
      muComplex native_alpha;
      static_assert(sizeof(native_alpha) == 2 * sizeof(float));
      std::memcpy(&native_alpha, alpha, sizeof(native_alpha));
      return ToShimStatus(mublasCtrsmBatched(
          native_handle, native_side, native_fill, native_trans_a,
          native_diagonal, native_m, native_n, &native_alpha,
          reinterpret_cast<const muComplex* const*>(a), native_lda,
          reinterpret_cast<muComplex* const*>(b), native_ldb,
          native_batch_count));
    }
    case XLA_MUSA_MUBLAS_TRSM_TYPE_C128: {
      muDoubleComplex native_alpha;
      static_assert(sizeof(native_alpha) == 2 * sizeof(double));
      std::memcpy(&native_alpha, alpha, sizeof(native_alpha));
      return ToShimStatus(mublasZtrsmBatched(
          native_handle, native_side, native_fill, native_trans_a,
          native_diagonal, native_m, native_n, &native_alpha,
          reinterpret_cast<const muDoubleComplex* const*>(a), native_lda,
          reinterpret_cast<muDoubleComplex* const*>(b), native_ldb,
          native_batch_count));
    }
    default:
      return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  }
}

constexpr XlaMusaMuBlasCapabilities kCapabilities =
    XLA_MUSA_MUBLAS_CAPABILITIES_V1;
constexpr XlaMusaMuBlasAdvancedCapabilities kAdvancedCapabilities =
    XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2 |
    XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_SCAL |
    XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_TRSM |
    XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_TRSM_BATCHED;

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
                                   kAdvancedCapabilities,
                                   Create,
                                   Destroy,
                                   SetStream,
                                   GetVersion,
                                   Gemm,
                                   SetAtomicsMode,
                                   GemmWithAlgorithm,
                                   GemmBatched,
                                   GemmStridedBatched,
                                   Scal,
                                   Trsm,
                                   TrsmBatched};

}  // namespace

extern "C" XLA_MUSA_MUBLAS_SHIM_EXPORT const XlaMusaMuBlasApiV1*
xla_musa_mublas_get_api_v1(void) {
  return &kApiV1;
}

extern "C" XLA_MUSA_MUBLAS_SHIM_EXPORT const XlaMusaMuBlasApiV2*
xla_musa_mublas_get_api_v2(void) {
  return &kApiV2;
}
