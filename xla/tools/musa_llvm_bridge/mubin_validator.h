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

#ifndef XLA_TOOLS_MUSA_LLVM_BRIDGE_MUBIN_VALIDATOR_H_
#define XLA_TOOLS_MUSA_LLVM_BRIDGE_MUBIN_VALIDATOR_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/gpu/musa/protocol.pb.h"

namespace xla::gpu::musa::bridge {

// Validates both the target-independent MUBIN envelope and the dynamic symbol
// contract requested by the host. This is intentionally independent from a
// compilation provider so every future in-process or subprocess provider has
// the same post-codegen gate.
absl::Status ValidateMubinOutput(absl::Span<const uint8_t> mubin,
                                 const MusaBridgeCompileRequest& request);

}  // namespace xla::gpu::musa::bridge

#endif  // XLA_TOOLS_MUSA_LLVM_BRIDGE_MUBIN_VALIDATOR_H_
