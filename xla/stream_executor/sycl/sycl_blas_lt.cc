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

#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "oneapi/mkl/blas.hpp"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace stream_executor {
namespace sycl {
namespace {

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

// Adds a broadcast vector bias (`bias_len` elements of type T) onto the
// column-major output buffer of shape (m, n) per batch slice with the given
// leading_dim and batch strides (in elements). `axis_m` selects the broadcast
// axis: true means bias is length m and is added per row (out[b,j,i]+=bias[i]),
// false means bias is length n and is added per column (out[b,j,i]+=bias[j]).
template <typename T>
void LaunchBiasAddInOrder(::sycl::queue* q, T* out, int64_t m, int64_t n,
                          int64_t ldc, int64_t batch, int64_t batch_stride,
                          const T* bias, bool axis_m) {
  ::sycl::range<3> range(static_cast<size_t>(batch), static_cast<size_t>(n),
                         static_cast<size_t>(m));
  q->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(range, [=](::sycl::id<3> idx) {
      const int64_t b = static_cast<int64_t>(idx[0]);
      const int64_t j = static_cast<int64_t>(idx[1]);
      const int64_t i = static_cast<int64_t>(idx[2]);
      const int64_t off = b * batch_stride + j * ldc + i;
      out[off] = static_cast<T>(out[off] + (axis_m ? bias[i] : bias[j]));
    });
  });
}

absl::Status ApplyVectorBias(Stream* stream, const DeviceAddressBase& bias,
                             blas::DataType type, int64_t m, int64_t n,
                             int64_t ldc, int64_t batch, int64_t batch_stride,
                             DeviceAddressBase* out) {
  ::sycl::queue* queue = static_cast<SyclStream*>(stream)->stream_handle();
  if (queue == nullptr) {
    return absl::InternalError("SYCL bias add: stream has null queue");
  }
  // Determine broadcast axis from bias byte length.
  int element_size = 0;
  switch (type) {
    case blas::DataType::kHalf:
    case blas::DataType::kBF16:
      element_size = 2;
      break;
    case blas::DataType::kFloat:
      element_size = 4;
      break;
    case blas::DataType::kDouble:
      element_size = 8;
      break;
    default:
      return absl::UnimplementedError(
          "SyclBlasLt: bias add unsupported dtype");
  }
  const uint64_t bias_elems = bias.size() / element_size;
  bool axis_m;
  if (bias_elems == static_cast<uint64_t>(m)) {
    axis_m = true;
  } else if (bias_elems == static_cast<uint64_t>(n)) {
    axis_m = false;
  } else {
    return absl::UnimplementedError(absl::StrCat(
        "SyclBlasLt: bias length ", bias_elems, " does not match m=", m,
        " or n=", n));
  }

  switch (type) {
    case blas::DataType::kHalf:
      LaunchBiasAddInOrder<::sycl::half>(
          queue, static_cast<::sycl::half*>(out->opaque()), m, n, ldc, batch,
          batch_stride, static_cast<const ::sycl::half*>(bias.opaque()),
          axis_m);
      break;
    case blas::DataType::kBF16:
      LaunchBiasAddInOrder<oneapi::mkl::bfloat16>(
          queue, static_cast<oneapi::mkl::bfloat16*>(out->opaque()), m, n, ldc,
          batch, batch_stride,
          static_cast<const oneapi::mkl::bfloat16*>(bias.opaque()), axis_m);
      break;
    case blas::DataType::kFloat:
      LaunchBiasAddInOrder<float>(
          queue, static_cast<float*>(out->opaque()), m, n, ldc, batch,
          batch_stride, static_cast<const float*>(bias.opaque()), axis_m);
      break;
    case blas::DataType::kDouble:
      LaunchBiasAddInOrder<double>(
          queue, static_cast<double*>(out->opaque()), m, n, ldc, batch,
          batch_stride, static_cast<const double*>(bias.opaque()), axis_m);
      break;
    default:
      return absl::UnimplementedError(
          "SyclBlasLt: bias add unsupported dtype");
  }
  return absl::OkStatus();
}

dnnl::memory::desc MakeDnnlMatrixDesc(
    const gpu::MatrixDescriptor& matrix, int64_t rows, int64_t cols,
    dnnl::memory::data_type data_type) {
  const bool transposed = matrix.transpose == blas::Transpose::kTranspose;
  return dnnl::memory::desc(
      {rows, cols}, data_type,
      transposed ? dnnl::memory::dims{matrix.leading_dim_stride, 1}
                 : dnnl::memory::dims{1, matrix.leading_dim_stride});
}

dnnl::matmul::primitive_desc MakeTensorScaledFp8PrimitiveDesc(
    const dnnl::engine& engine, const dnnl::memory::desc& lhs,
    const dnnl::memory::desc& rhs, const dnnl::memory::desc& out,
    float beta) {
  dnnl::primitive_attr attr;
  attr.set_scales(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::f32,
                  /*is_on_host=*/false);
  attr.set_scales(DNNL_ARG_WEIGHTS, 0, {}, dnnl::memory::data_type::f32,
                  /*is_on_host=*/false);
  if (beta != 0.0f) {
    dnnl::post_ops post_ops;
    post_ops.append_sum(beta);
    attr.set_post_ops(post_ops);
  }
  return dnnl::matmul::primitive_desc(engine, lhs, rhs, out, attr);
}

absl::Status DnnlErrorToStatus(const dnnl::error& error) {
  const std::string message = absl::StrCat(
      "SyclBlasLt oneDNN FP8 matmul failed (status ",
      static_cast<int>(error.status), "): ", error.what());
  if (error.status == dnnl_unimplemented) {
    return absl::UnimplementedError(message);
  }
  if (error.status == dnnl_invalid_arguments) {
    return absl::InvalidArgumentError(message);
  }
  return absl::InternalError(message);
}
}  // namespace

absl::Status BlasLt::Init() { return absl::OkStatus(); }

struct BlasLt::MatmulPlan::OneDnnMatmul {
  OneDnnMatmul(::sycl::queue* queue, const gpu::MatrixDescriptor& lhs,
               const gpu::MatrixDescriptor& rhs,
               const gpu::OutputMatrixDescriptor& out, float beta)
      : engine(dnnl::sycl_interop::make_engine(queue->get_device(),
                                                queue->get_context())),
        lhs_desc(MakeDnnlMatrixDesc(lhs, out.m, out.k,
                                    dnnl::memory::data_type::f8_e4m3)),
        rhs_desc(MakeDnnlMatrixDesc(rhs, out.k, out.n,
                                    dnnl::memory::data_type::f8_e4m3)),
        out_desc(MakeDnnlMatrixDesc(out, out.m, out.n,
                                    dnnl::memory::data_type::bf16)),
        scale_desc({1}, dnnl::memory::data_type::f32, {1}),
        primitive_desc(
            MakeTensorScaledFp8PrimitiveDesc(engine, lhs_desc, rhs_desc,
                                              out_desc, beta)),
        primitive(primitive_desc) {}

  dnnl::engine engine;
  dnnl::memory::desc lhs_desc;
  dnnl::memory::desc rhs_desc;
  dnnl::memory::desc out_desc;
  dnnl::memory::desc scale_desc;
  dnnl::matmul::primitive_desc primitive_desc;
  dnnl::matmul primitive;
};
BlasLt::MatmulPlan::~MatmulPlan() = default;

auto BlasLt::GetMatmulPlan(const gpu::GemmConfig& config,
                           Epilogue epilogue) const
    -> absl::StatusOr<MatmulPlanPtr> {
  absl::MutexLock lock(&mu_);
  return std::make_unique<MatmulPlan>(config, epilogue);
}

absl::Status BlasLt::MatmulPlan::ExecuteTensorScaledFp8Matmul(
    Stream* stream, const gpu::BlasLt::MemoryArgs& args,
    const xla::gpu::GemmConfig::DescriptorsTuple& descs) const {
  if (config_.alpha.real() != 1.0 || config_.alpha.imag() != 0.0 ||
      !std::isfinite(config_.beta) ||
      !std::isfinite(static_cast<float>(config_.beta))) {
    return absl::UnimplementedError(absl::StrCat(
        "SyclBlasLt: tensor-scaled FP8 matmul requires alpha=1 and a finite "
        "beta; "
        "got alpha=", config_.alpha.real(), "+", config_.alpha.imag(),
        "i, beta=", config_.beta));
  }
  if (args.a.opaque() == nullptr || args.b.opaque() == nullptr ||
      args.d.opaque() == nullptr) {
    return absl::InvalidArgumentError(
        "SyclBlasLt: tensor-scaled FP8 matmul requires non-null A, B, and D");
  }
  if (config_.beta != 0.0 && args.c.opaque() == nullptr) {
    return absl::InvalidArgumentError(
        "SyclBlasLt: tensor-scaled FP8 matmul with beta requires non-null C");
  }
  if (config_.beta != 0.0 && args.c.size() != args.d.size()) {
    return absl::UnimplementedError(
        "SyclBlasLt: beta != 0 requires C and D buffers of equal size");
  }
  if (args.aux.opaque() != nullptr || args.c_scale.opaque() != nullptr ||
      args.d_scale.opaque() != nullptr || args.d_amax.opaque() != nullptr) {
    return absl::UnimplementedError(
        "SyclBlasLt: FP8 auxiliary, output-scale, and amax outputs are not "
        "supported");
  }

  const auto& lhs = descs.lhs;
  const auto& rhs = descs.rhs;
  const auto& out = descs.output;
  if (out.batch_size != 1) {
    return absl::UnimplementedError(
        "SyclBlasLt: batched tensor-scaled FP8 matmul is not supported");
  }
  if (lhs.type != blas::DataType::kF8E4M3FN ||
      rhs.type != blas::DataType::kF8E4M3FN ||
      out.type != blas::DataType::kBF16) {
    return absl::UnimplementedError(
        "SyclBlasLt: only F8E4M3FN x F8E4M3FN -> BF16 tensor scaling is "
        "supported");
  }
  if ((lhs.transpose != blas::Transpose::kNoTranspose &&
       lhs.transpose != blas::Transpose::kTranspose) ||
      (rhs.transpose != blas::Transpose::kNoTranspose &&
       rhs.transpose != blas::Transpose::kTranspose) ||
      out.transpose != blas::Transpose::kNoTranspose) {
    return absl::UnimplementedError(
        "SyclBlasLt: unsupported tensor-scaled FP8 matrix layout");
  }

  // GetMatrixDescriptors changes a row-major output into an equivalent
  // column-major GEMM by swapping its operands. Keep each device-side scale
  // bound to the logical input buffer it was supplied with.
  const DeviceAddressBase& lhs_scale =
      descs.operands_swapped ? args.b_scale : args.a_scale;
  const DeviceAddressBase& rhs_scale =
      descs.operands_swapped ? args.a_scale : args.b_scale;
  if (lhs_scale.opaque() == nullptr || rhs_scale.opaque() == nullptr ||
      lhs_scale.size() < sizeof(float) || rhs_scale.size() < sizeof(float)) {
    return absl::InvalidArgumentError(
        "SyclBlasLt: tensor-scaled FP8 matmul requires device-side f32 A and "
        "B scales");
  }

  ::sycl::queue* queue = static_cast<SyclStream*>(stream)->stream_handle();
  if (queue == nullptr) {
    return absl::InternalError(
        "SyclBlasLt: tensor-scaled FP8 matmul stream has null queue");
  }

  OneDnnMatmul* matmul;
  try {
    {
      absl::MutexLock lock(&mu_);
      if (one_dnn_matmul_ == nullptr) {
        one_dnn_matmul_ = std::make_unique<OneDnnMatmul>(
            queue, lhs, rhs, out, static_cast<float>(config_.beta));
      }
      matmul = one_dnn_matmul_.get();
    }

    if (config_.beta != 0.0 && args.c.opaque() != args.d.opaque()) {
      queue->memcpy(args.d.opaque(), args.c.opaque(), args.d.size())
          .wait_and_throw();
    }

    dnnl::memory src = dnnl::sycl_interop::make_memory(
        matmul->lhs_desc, matmul->engine, dnnl::sycl_interop::memory_kind::usm,
        lhs.data.opaque());
    dnnl::memory weights = dnnl::sycl_interop::make_memory(
        matmul->rhs_desc, matmul->engine, dnnl::sycl_interop::memory_kind::usm,
        rhs.data.opaque());
    dnnl::memory dst = dnnl::sycl_interop::make_memory(
        matmul->out_desc, matmul->engine, dnnl::sycl_interop::memory_kind::usm,
        out.data.opaque());
    dnnl::memory src_scale = dnnl::sycl_interop::make_memory(
        matmul->scale_desc, matmul->engine,
        dnnl::sycl_interop::memory_kind::usm, lhs_scale.opaque());
    dnnl::memory weights_scale = dnnl::sycl_interop::make_memory(
        matmul->scale_desc, matmul->engine,
        dnnl::sycl_interop::memory_kind::usm, rhs_scale.opaque());
    dnnl::stream dnnl_stream = dnnl::sycl_interop::make_stream(
        matmul->engine, *queue);
    dnnl::sycl_interop::execute(
        matmul->primitive, dnnl_stream,
        {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, weights}, {DNNL_ARG_DST, dst},
         {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale},
         {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, weights_scale}});
  } catch (const dnnl::error& error) {
    return DnnlErrorToStatus(error);
  } catch (const ::sycl::exception& error) {
    return absl::InternalError(absl::StrCat(
        "SyclBlasLt oneDNN FP8 matmul SYCL failure: ", error.what()));
  } catch (const std::exception& error) {
    return absl::InternalError(absl::StrCat(
        "SyclBlasLt oneDNN FP8 matmul failure: ", error.what()));
  }
  return absl::OkStatus();
}

absl::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, const gpu::BlasLt::MemoryArgs& args,
    blas::ProfileResult* profile_result) const {
  // Only the default epilogue is supported for now. Bias / GELU / ReLU /
  // auxiliary-output epilogues require either MKL post-ops or a fused SYCL
  // kernel and are deferred. Returning Unimplemented forces the HLO compiler
  // to keep the epilogue as a separate op.
  if (epilogue_ != Epilogue::kDefault && epilogue_ != Epilogue::kBias) {
    return absl::UnimplementedError(absl::StrCat(
        "SyclBlasLt: matmul epilogue ", static_cast<int>(epilogue_),
        " is not supported"));
  }
  if (epilogue_ == Epilogue::kBias && args.bias.opaque() == nullptr) {
    return absl::InternalError("SyclBlasLt: kBias epilogue with null bias");
  }

  if (config_.scale_mode == gpu::ScaleMode::kBlockScaling) {
    return absl::UnimplementedError(
        "SyclBlasLt: block-scaled matmul is not supported");
  }

  // Build matrix descriptors (handles row/column-major canonicalization and
  // operand swapping). This is the same helper the regular RunGemm path uses.
  TF_ASSIGN_OR_RETURN(
      auto descs,
      config_.GetMatrixDescriptors(
          args.a, args.b, args.d,
          stream->parent()->GetDeviceDescription().gpu_compute_capability()));

  if (config_.scale_mode == gpu::ScaleMode::kTensorScaling) {
    std::unique_ptr<EventBasedTimer> timer;
    if (profile_result != nullptr) {
      TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                     profile_result->warmup_run_executed()));
    }
    TF_RETURN_IF_ERROR(ExecuteTensorScaledFp8Matmul(stream, args, descs));
    if (epilogue_ == Epilogue::kBias) {
      DeviceAddressBase out_buf = descs.output.data;
      TF_RETURN_IF_ERROR(ApplyVectorBias(
          stream, args.bias, descs.output.type, descs.output.m,
          descs.output.n, descs.output.leading_dim_stride,
          descs.output.batch_size, descs.output.batch_stride, &out_buf));
    }
    if (profile_result != nullptr) {
      TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
          timer.get(), kOneDnnGemm, profile_result));
    }
    return absl::OkStatus();
  }

  // We do the GEMM in-place on D. If beta != 0 and C is a distinct buffer
  // from D we copy C into D first so MKL can accumulate against it.
  const double beta = config_.beta;
  const auto& lhs = descs.lhs;
  const auto& rhs = descs.rhs;
  const auto& out = descs.output;
  if (beta != 0.0 && args.c.opaque() != nullptr &&
      args.c.opaque() != args.d.opaque()) {
    if (args.c.size() != args.d.size()) {
      return absl::UnimplementedError(
          "SyclBlasLt: beta != 0 requires C and D buffers of equal size");
    }
    DeviceAddressBase d_buf = args.d;
    TF_RETURN_IF_ERROR(stream->Memcpy(&d_buf, args.c, args.c.size()));
  }

  std::unique_ptr<EventBasedTimer> timer;
  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   profile_result->warmup_run_executed()));
  }

  // Dispatch on the scalar type used for alpha/beta. Float covers the
  // FP16/BF16/FP32/Int8 paths because the existing DispatchOneMklGemm reads
  // alpha/beta through a void* with the Scale type fixed per (Atype, Ctype)
  // pair, and every non-FP64 / non-complex Scale instantiated in the helper
  // ultimately reads a `float` (sycl::half's specialization reads float too).
  auto run = [&](auto scale_v) -> absl::Status {
    using Scale = decltype(scale_v);
    Scale alpha_s{};
    Scale beta_s{};
    if constexpr (std::is_same_v<Scale, std::complex<float>> ||
                  std::is_same_v<Scale, std::complex<double>>) {
      using Real = typename Scale::value_type;
      alpha_s = Scale(static_cast<Real>(config_.alpha.real()),
                      static_cast<Real>(config_.alpha.imag()));
      beta_s = Scale(static_cast<Real>(beta), Real{0});
    } else {
      alpha_s = static_cast<Scale>(config_.alpha.real());
      beta_s = static_cast<Scale>(beta);
    }

    DeviceAddressBase out_buf = out.data;
    const blas::ComputationType ct = blas::ComputationType::kF32;
    if (out.batch_size > 1) {
      return DispatchOneMklGemmStridedBatched(
          stream, lhs.transpose, rhs.transpose, out.m, out.n, out.k, &alpha_s,
          lhs.data, lhs.type, static_cast<int>(lhs.leading_dim_stride),
          lhs.batch_stride, rhs.data, rhs.type,
          static_cast<int>(rhs.leading_dim_stride), rhs.batch_stride, &beta_s,
          &out_buf, out.type, static_cast<int>(out.leading_dim_stride),
          out.batch_stride, static_cast<int>(out.batch_size), ct);
    }
    return DispatchOneMklGemm(
        stream, lhs.transpose, rhs.transpose, out.m, out.n, out.k, &alpha_s,
        lhs.data, lhs.type, static_cast<int>(lhs.leading_dim_stride), rhs.data,
        rhs.type, static_cast<int>(rhs.leading_dim_stride), &beta_s, &out_buf,
        out.type, static_cast<int>(out.leading_dim_stride), ct);
  };

  absl::Status status;
  try {
    switch (out.type) {
      case blas::DataType::kDouble:
        status = run(double{});
        break;
      case blas::DataType::kComplexFloat:
        status = run(std::complex<float>{});
        break;
      case blas::DataType::kComplexDouble:
        status = run(std::complex<double>{});
        break;
      default:
        status = run(float{});
        break;
    }
  } catch (const oneapi::mkl::exception& e) {
    return absl::InternalError(e.what());
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(e.what());
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }
  TF_RETURN_IF_ERROR(status);

  if (epilogue_ == Epilogue::kBias) {
    DeviceAddressBase out_buf = out.data;
    TF_RETURN_IF_ERROR(ApplyVectorBias(stream, args.bias, out.type, out.m,
                                       out.n, out.leading_dim_stride,
                                       out.batch_size, out.batch_stride,
                                       &out_buf));
  }

  if (profile_result != nullptr) {
    TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
        timer.get(), blas::kDefaultAlgorithm, profile_result));
  }
  return absl::OkStatus();
}

auto BlasLt::MatmulPlan::GetAlgorithms(size_t max_algorithm_count,
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
    TF_RETURN_IF_ERROR(DispatchOneMklGemm(
        stream, transa, transb, m, n, k, alpha, a, type_a, lda, b, type_b, ldb,
        beta, c, type_c, ldc, computation_type));
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
