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

#ifndef XLA_SERVICE_GPU_MUSA_PROTOCOL_H_
#define XLA_SERVICE_GPU_MUSA_PROTOCOL_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.pb.h"

namespace xla::gpu::musa {

inline constexpr absl::string_view kMusaBridgeRequestMagic =
    "XLA_MUSA_BRIDGE_REQUEST_V1\n";
inline constexpr absl::string_view kMusaBridgeResponseMagic =
    "XLA_MUSA_BRIDGE_RESPONSE_V1\n";

// Wire and field bounds are part of protocol version 1. Protobuf TextFormat
// can octal-escape every input byte to four output bytes. Keep wire ceilings
// derived from the admitted payload ceilings so every valid message remains
// encodable and decodable.
inline constexpr size_t kMusaBridgeMaxLlvmBytes = 64 * 1024 * 1024;
inline constexpr size_t kMusaBridgeMaxMubinBytes = 64 * 1024 * 1024;
inline constexpr size_t kMusaBridgeMaxModuleNameBytes = 256;
inline constexpr size_t kMusaBridgeMaxSymbolNameBytes = 512;
inline constexpr size_t kMusaBridgeMaxRevisionBytes = 128;
inline constexpr size_t kMusaBridgeMaxProviderNameBytes = 128;
inline constexpr size_t kMusaBridgeMaxKernelCount = 4096;
inline constexpr size_t kMusaBridgeMaxExportedSymbolCount = 16384;
inline constexpr size_t kMusaBridgeMaxExportedGlobalCount = 4096;
inline constexpr size_t kMusaBridgeMaxDiagnosticCount = 512;
inline constexpr size_t kMusaBridgeMaxDiagnosticMessageBytes = 64 * 1024;
inline constexpr size_t kMusaBridgeMaxDiagnosticBytes = 4 * 1024 * 1024;
inline constexpr size_t kMusaBridgeMaxRequestWireBytes =
    4 * kMusaBridgeMaxLlvmBytes + 16 * 1024 * 1024;
inline constexpr size_t kMusaBridgeMaxResponseWireBytes =
    4 * kMusaBridgeMaxMubinBytes + 4 * kMusaBridgeMaxDiagnosticBytes +
    16 * 1024 * 1024;

// Returns lower-case, unprefixed SHA-256 hex.
std::string MusaBridgeSha256Hex(absl::string_view data);

absl::Status ValidateMusaBridgeCompileRequest(
    const MusaBridgeCompileRequest& request);
absl::Status ValidateMusaBridgeCompileResponse(
    const MusaBridgeCompileResponse& response);

// Validates cross-message binding: canonical request digest, selected versus
// actual provider/toolchain fingerprints, counts, sizes, and diagnostic symbol
// references.
absl::Status ValidateMusaBridgeExchange(
    const MusaBridgeCompileRequest& request,
    const MusaBridgeCompileResponse& response);

// Encoders emit the one canonical TextFormat representation after the magic
// line. Decoders reject alternate whitespace/orderings as well as unknown
// fields; this prevents two wire representations from naming one request.
absl::StatusOr<std::string> EncodeMusaBridgeCompileRequest(
    const MusaBridgeCompileRequest& request);
absl::StatusOr<MusaBridgeCompileRequest> DecodeMusaBridgeCompileRequest(
    absl::string_view wire);
absl::StatusOr<std::string> EncodeMusaBridgeCompileResponse(
    const MusaBridgeCompileResponse& response);
absl::StatusOr<MusaBridgeCompileResponse> DecodeMusaBridgeCompileResponse(
    absl::string_view wire);

absl::StatusOr<std::string> MusaBridgeCompileRequestSha256(
    const MusaBridgeCompileRequest& request);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_PROTOCOL_H_
