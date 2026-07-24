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

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

// Basic muBLAS adapter for homogeneous, non-batched GEMM. The vendor library
// lives behind the SDK-free shim ABI, while this class implements XLA's shared
// BlasSupport contract and owns one context-bound muBLAS handle.
class MusaBlas final : public blas::BlasSupport {
 public:
  explicit MusaBlas(StreamExecutor* parent,
                    MusaMuBlasApi* api = GetMusaMuBlasApi());
  MusaBlas(const MusaBlas&) = delete;
  MusaBlas& operator=(const MusaBlas&) = delete;
  ~MusaBlas() override;

  bool Init();

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

  gpu::BlasLt* GetBlasLt() override { return nullptr; }

 private:
  absl::Status SetStream(Stream* stream) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status DoBlasGemmInternal(
      Stream* stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64_t k, const void* alpha,
      const DeviceAddressBase& a, blas::DataType type_a, int lda,
      const DeviceAddressBase& b, blas::DataType type_b, int ldb,
      const void* beta, DeviceAddressBase* c, blas::DataType type_c, int ldc,
      blas::ComputationType computation_type,
      const EngineOptions& engine_options);
  absl::Status Unsupported(absl::string_view operation) const;
  bool UnsupportedBool(absl::string_view operation) const;

  mutable absl::Mutex mu_;
  StreamExecutor* const parent_;
  MusaMuBlasApi* const api_;
  void* handle_ ABSL_GUARDED_BY(mu_) = nullptr;
  void* current_stream_ ABSL_GUARDED_BY(mu_) = nullptr;
};

void InitializeMusaBlas();

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_
