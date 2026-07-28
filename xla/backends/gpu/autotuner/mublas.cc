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

#include "xla/backends/gpu/autotuner/mublas.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"

namespace xla::gpu {
namespace {

constexpr std::array<int64_t, 2> kSupportedAlgorithms = {
    kMublasDefaultAlgorithm, kMublasTensorOpAlgorithm};

std::unique_ptr<BackendConfig> MakeConfig(int64_t algorithm) {
  auto config = std::make_unique<BackendConfig>();
  AutotuneResult::GemmKey* gemm = config->mutable_gemm();
  gemm->set_algorithm(algorithm);
  gemm->set_autotune_workspace_size(0);
  return config;
}

bool IsSupportedAlgorithm(int64_t algorithm) {
  return algorithm == kMublasDefaultAlgorithm ||
         algorithm == kMublasTensorOpAlgorithm;
}

bool IsHomogeneousF32(const HloInstruction& instr) {
  return instr.operand_count() == 2 && instr.shape().IsArray() &&
         instr.operand(0)->shape().IsArray() &&
         instr.operand(1)->shape().IsArray() &&
         instr.shape().element_type() == F32 &&
         instr.operand(0)->shape().element_type() == F32 &&
         instr.operand(1)->shape().element_type() == F32;
}

}  // namespace

bool MublasBackend::IsSupported(const HloInstruction& instr) {
  return IsMusaGemm(instr);
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
MublasBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>{};
  }
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(kSupportedAlgorithms.size());
  for (int64_t algorithm : kSupportedAlgorithms) {
    if (algorithm == kMublasTensorOpAlgorithm) {
      if (!IsHomogeneousF32(instr) ||
          debug_options().xla_gpu_deterministic_ops() ||
          debug_options().xla_gpu_exclude_nondeterministic_ops()) {
        continue;
      }
      // A real-device tune must negotiate the v2 capability. Deviceless
      // compilation never profiles candidates and uses GetDefaultConfig().
      if (stream_executor_ != nullptr &&
          !stream_executor::musa::GetMusaMuBlasApi()->SupportsTensorOpF32()) {
        continue;
      }
    }
    configs.push_back(MakeConfig(algorithm));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> MublasBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "MublasBackend requires a __mublas$gemm custom call");
  }
  return MakeConfig(kMublasDefaultAlgorithm);
}

absl::Status MublasBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "MublasBackend requires a __mublas$gemm custom call");
  }
  if (!config.has_gemm()) {
    return absl::InvalidArgumentError(
        "MublasBackend requires an AutotuneResult::GemmKey config");
  }
  const AutotuneResult::GemmKey& gemm = config.gemm();
  if (!IsSupportedAlgorithm(gemm.algorithm())) {
    return absl::InvalidArgumentError(
        absl::StrCat("unsupported normalized muBLAS algorithm ",
                     gemm.algorithm(), "; expected 0 or 1"));
  }
  if (gemm.algorithm() == kMublasTensorOpAlgorithm &&
      !IsHomogeneousF32(instr)) {
    return absl::InvalidArgumentError(
        "normalized muBLAS algorithm 1 is qualified only for homogeneous "
        "f32 GEMM");
  }
  if (gemm.algorithm() == kMublasTensorOpAlgorithm &&
      (debug_options().xla_gpu_deterministic_ops() ||
       debug_options().xla_gpu_exclude_nondeterministic_ops())) {
    return absl::InvalidArgumentError(
        "deterministic muBLAS GEMM requires normalized algorithm 0");
  }
  if (gemm.autotune_workspace_size() != 0) {
    return absl::InvalidArgumentError(
        "qualified muBLAS algorithms require zero workspace");
  }

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  GemmBackendConfig* backend_config = gpu_config.mutable_gemm_backend_config();
  backend_config->set_selected_algorithm(gemm.algorithm());
  backend_config->set_autotune_workspace_size(0);
  return instr.set_backend_config(std::move(gpu_config));
}

std::string MublasBackend::version() const {
  const stream_executor::DeviceDescription& device =
      target_config().device_description;
  const stream_executor::MusaComputeCapability* capability =
      device.gpu_compute_capability().musa_compute_capability();
  std::string version = absl::StrCat(
      "musa_arch=",
      capability == nullptr ? "unknown" : capability->architecture(),
      ";musa_runtime=", device.runtime_version().ToString(),
      ";musa_driver=", device.driver_version().ToString(),
      ";musa_kernel_driver=", device.kernel_mode_driver_version().ToString(),
      ";musa_toolkit=", device.compile_time_toolkit_version().ToString(),
      ";mublas_required_advanced_contract=",
      stream_executor::musa::kMusaMuBlasAdvancedAbiContractV2,
      ";mublas_required_advanced_fingerprint=",
      stream_executor::musa::kMusaMuBlasAdvancedAbiFingerprintV2,
      ";mublas_deterministic_ops=", debug_options().xla_gpu_deterministic_ops(),
      ";mublas_exclude_nondeterministic_ops=",
      debug_options().xla_gpu_exclude_nondeterministic_ops());

  std::string mublas_version;
  if (stream_executor_ != nullptr) {
    stream_executor::musa::MusaMuBlasApi* api =
        stream_executor::musa::GetMusaMuBlasApi();
    if (api->Init().ok()) {
      absl::StrAppend(
          &version, ";mublas_shim_abi=", api->abi_version(),
          ";mublas_base_capabilities=", api->capabilities(),
          ";mublas_advanced_capabilities=", api->advanced_capabilities(),
          ";mublas_advanced_fingerprint=", api->advanced_abi_fingerprint());
    } else {
      absl::StrAppend(&version, ";mublas_shim_abi=unavailable");
    }
    stream_executor::blas::BlasSupport* blas = stream_executor_->AsBlas();
    if (blas != nullptr && blas->GetVersion(&mublas_version).ok()) {
      absl::StrAppend(&version, ";mublas=", mublas_version);
      return version;
    }
  }
  if (stream_executor_ == nullptr) {
    absl::StrAppend(&version, ";mublas_shim_abi=deviceless");
  }
  absl::StrAppend(&version, ";mublas=unavailable");
  return version;
}

}  // namespace xla::gpu
