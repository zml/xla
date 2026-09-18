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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

// Stable algorithm identifiers serialized by XLA's autotuning cache. They are
// deliberately independent of vendor enum values.
inline constexpr blas::AlgorithmType kMusaBlasDefaultAlgorithm = 0;
inline constexpr blas::AlgorithmType kMusaBlasTensorOpAlgorithm = 1;

// muBLAS adapter for homogeneous GEMM. The vendor library lives behind the
// SDK-free shim ABI. Handles are context-bound and one handle is owned per
// native stream, allowing independent streams to enqueue work concurrently
// without racing handle-level stream or atomics state.
class MusaBlas final : public blas::BlasSupport {
 public:
  explicit MusaBlas(StreamExecutor* parent,
                    MusaMuBlasApi* api = GetMusaMuBlasApi());
  MusaBlas(const MusaBlas&) = delete;
  MusaBlas& operator=(const MusaBlas&) = delete;
  ~MusaBlas() override;

  bool Init();
  void NotifyStreamDestroyed(Stream* stream) override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

  gpu::BlasLt* GetBlasLt() override { return nullptr; }

 private:
  struct HandleState {
    explicit HandleState(void* native_stream) : native_stream(native_stream) {}

    absl::Mutex mu;
    void* handle ABSL_GUARDED_BY(mu) = nullptr;
    void* const native_stream;
  };

  absl::StatusOr<std::shared_ptr<HandleState>> GetOrCreateHandle(
      Stream* stream);
  absl::StatusOr<std::shared_ptr<HandleState>> GetOrCreateHandleForNativeStream(
      void* native_stream);
  void RetainFailedHostStaging(std::shared_ptr<MemoryAllocation> host_staging);
  absl::Status ConfigureAtomics(HandleState* state,
                                const EngineOptions& engine_options,
                                XlaMusaMuBlasAlgorithm algorithm)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state->mu);
  absl::Status DoBlasScalInternal(Stream* stream, uint64_t n, const void* alpha,
                                  DeviceAddressBase* x, int incx,
                                  XlaMusaMuBlasScalType scal_type,
                                  size_t element_size);
  absl::Status DoBlasTrsmInternal(
      Stream* stream, blas::Side side, blas::UpperLower uplo,
      blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n,
      const void* alpha, const DeviceAddressBase& a, int lda,
      DeviceAddressBase* b, int ldb, XlaMusaMuBlasTrsmType trsm_type,
      size_t element_size);
  absl::Status DoBlasTrsmBatchedInternal(
      Stream* stream, blas::Side side, blas::UpperLower uplo,
      blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n,
      const void* alpha, const DeviceAddressBase& as, int lda,
      DeviceAddressBase* bs, int ldb, int batch_count,
      XlaMusaMuBlasTrsmType trsm_type);
  absl::Status DoBlasGemmInternal(
      Stream* stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64_t k, const void* alpha,
      const DeviceAddressBase& a, blas::DataType type_a, int lda,
      const DeviceAddressBase& b, blas::DataType type_b, int ldb,
      const void* beta, DeviceAddressBase* c, blas::DataType type_c, int ldc,
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,
      const EngineOptions& engine_options,
      blas::ProfileResult* output_profile_result);
  absl::Status DoBlasGemmBatchedInternal(
      Stream* stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64_t k, const void* alpha,
      absl::Span<const void* const> a, blas::DataType type_a, int lda,
      absl::Span<const void* const> b, blas::DataType type_b, int ldb,
      const void* beta, absl::Span<void* const> c, blas::DataType type_c,
      int ldc, int batch_count, blas::ComputationType computation_type,
      const EngineOptions& engine_options, ScratchAllocator* scratch_allocator);
  absl::Status DoBlasGemmStridedBatchedInternal(
      Stream* stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64_t k, const void* alpha,
      const DeviceAddressBase& a, blas::DataType type_a, int lda,
      int64_t stride_a, const DeviceAddressBase& b, blas::DataType type_b,
      int ldb, int64_t stride_b, const void* beta, DeviceAddressBase* c,
      blas::DataType type_c, int ldc, int64_t stride_c, int batch_count,
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,
      const EngineOptions& engine_options,
      blas::ProfileResult* output_profile_result);

  template <typename T, typename Scalar>
  bool DoBlasGemmBatchedTyped(Stream* stream, blas::Transpose transa,
                              blas::Transpose transb, uint64_t m, uint64_t n,
                              uint64_t k, Scalar alpha, DeviceAddressSlice<T> a,
                              int lda, DeviceAddressSlice<T> b, int ldb,
                              Scalar beta, DeviceAddressSlice<T> c, int ldc,
                              int batch_count, blas::DataType data_type,
                              blas::ComputationType computation_type,
                              const EngineOptions& engine_options,
                              ScratchAllocator* scratch_allocator);
  absl::Status Unsupported(absl::string_view operation) const;
  bool UnsupportedBool(absl::string_view operation) const;

  StreamExecutor* const parent_;
  MusaMuBlasApi* const api_;
  mutable absl::Mutex handles_mu_;
  absl::flat_hash_map<void*, std::shared_ptr<HandleState>> handles_
      ABSL_GUARDED_BY(handles_mu_);
  // If synchronization itself fails after a partial pointer-array enqueue,
  // retaining registered host storage is safer than letting the device read
  // freed memory. Stream teardown synchronizes again before MusaBlas dies.
  std::vector<std::shared_ptr<MemoryAllocation>> failed_host_staging_
      ABSL_GUARDED_BY(handles_mu_);
  void* last_stream_ ABSL_GUARDED_BY(handles_mu_) = nullptr;
};

void InitializeMusaBlas();

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_
