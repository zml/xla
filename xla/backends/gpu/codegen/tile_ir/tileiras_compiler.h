#ifndef XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TILEIRAS_COMPILER_H_
#define XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TILEIRAS_COMPILER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu::tile_ir {

namespace se = ::stream_executor;

// Assembles Tile IR bytecode into a cubin for `device_info` by running
// `tileiras`. Results are cached per (arch, bytecode) for the process.
absl::StatusOr<std::vector<uint8_t>> CompileTileIrBytecode(
    absl::string_view bytecode, const se::DeviceDescription& device_info,
    const DebugOptions& debug_options);

// Locates a `tileiras` of at least the minimum supported version, looking in
// `preferred_cuda_dir` first.
absl::StatusOr<std::string> FindTileIrAssembler(
    absl::string_view preferred_cuda_dir);

}  // namespace xla::gpu::tile_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TILEIRAS_COMPILER_H_
