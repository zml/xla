#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/tile_ir/tile_ir_module.h"
#include "xla/backends/gpu/codegen/tile_ir/tileiras_compiler.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/cuda_tile_call.h"
#include "xla/service/gpu/cuda_tile_custom_kernel_emitter.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

xla::Future<ThunkSequence> EmitCudaTileCustomKernelThunk(
    const HloCustomCallInstruction* instr, IrEmitterContext* context) {
  absl::string_view backend_config = instr->raw_backend_config_string();
  if (backend_config.empty()) {
    return absl::InvalidArgumentError(
        "CUDA Tile IR custom call backend config is empty");
  }
  ABSL_ASSIGN_OR_RETURN(
      CudaTileCall call,
      CudaTileCall::Parse(backend_config, context->mlir_context()));

  // The device's grid limits are narrower than the parser's on y and z.
  // Checked here, not in the generator, so a cache hit does not skip it.
  const stream_executor::BlockDim& limit =
      context->gpu_device_info().block_dim_limit();
  auto over = [](int32_t axis, uint64_t axis_limit) {
    return axis_limit != 0 && static_cast<uint64_t>(axis) > axis_limit;
  };
  if (over(call.grid_x, limit.x) || over(call.grid_y, limit.y) ||
      over(call.grid_z, limit.z)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CUDA Tile IR custom call grid (", call.grid_x, ", ", call.grid_y,
        ", ", call.grid_z, ") exceeds the device grid limit (", limit.x, ", ",
        limit.y, ", ", limit.z, ")"));
  }

  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                        emitters::KernelArguments::Create(
                            context->buffer_assignment(),
                            GetDefaultBufferAlignment(), instr,
                            call.output_indices));
  const int num_args = kernel_arguments.args().size();

  // `zeroed_outputs` names array result leaves; resolve them to slices here,
  // beside the grid check, so a cache hit does not skip the validation.
  ThunkSequence thunks;
  if (!call.zeroed_outputs.empty()) {
    std::vector<std::pair<Shape, ShapeIndex>> leaves;
    ShapeUtil::ForEachSubshape(
        instr->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsArray()) leaves.emplace_back(subshape, index);
        });
    for (int32_t leaf : call.zeroed_outputs) {
      if (leaf >= leaves.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "backend_config field 'zeroed_outputs' names result leaf ", leaf,
            " but the custom call has only ", leaves.size(),
            " array results"));
      }
      const auto& [subshape, index] = leaves[leaf];
      ABSL_ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          context->buffer_assignment().GetUniqueSlice(instr, index));
      // Zeroing an aliased result would destroy the kernel's input.
      int uses = 0;
      for (const auto& arg : kernel_arguments.args()) {
        if (arg.slice() == slice) ++uses;
      }
      if (uses > 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "backend_config field 'zeroed_outputs' names result leaf ", leaf,
            ", which aliases an operand; zeroing it would destroy the "
            "kernel's input"));
      }
      thunks.emplace_back(std::make_unique<MemzeroThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(instr,
                                                  context->GetNextThunkId()),
          ShapedSlice{slice, subshape}));
    }
  }

  // cuda-tile's launch ABI: the grid counts tile blocks, the block is (1,1,1)
  // and shared memory is 0. Kept here rather than read back from the cache
  // entry, which stores a 1-D block count and would flatten a 3-D grid.
  const stream_executor::BlockDim grid(call.grid_x, call.grid_y, call.grid_z);
  const stream_executor::ThreadDim block(1, 1, 1);
  constexpr int64_t kSharedMemoryBytes = 0;

  auto generate = [context, instr, call = std::move(call), num_args,
                   grid]() -> xla::Future<KernelReuseCache::Entry> {
    std::string kernel_name = context->GetSanitizedUniqueName(call.name);

    BorrowedMlirContext borrowed = context->BorrowMlirContext();
    tile_ir::LoadMlirDialectsForCudaTile(**borrowed);
    tile_ir::TileIrModuleText dump;
    ABSL_ASSIGN_OR_RETURN(
        std::string bytecode,
        tile_ir::CudaTileTextToBytecode(
            tile_ir::TileIrModuleRequest{call.ir, call.name, kernel_name,
                                         num_args, call.bytecode_major,
                                         call.bytecode_minor},
            **borrowed, &dump));

    const HloModule& hlo_module = *instr->GetModule();
    if (DumpingEnabledForHloModule(hlo_module)) {
      DumpToFileInDirOrStdout(hlo_module, "",
                              absl::StrCat(instr->name(), ".cuda_tile.mlir"),
                              dump.cuda_tile_text);
      DumpToFileInDirOrStdout(hlo_module, "",
                              absl::StrCat(instr->name(), ".tilebc"),
                              bytecode);
    }

    // Runs tileiras; captures by value to leave this frame and the cache's
    // mutex.
    auto assemble =
        [bytecode = std::move(bytecode), kernel_name = std::move(kernel_name),
         grid, &device_info = context->gpu_device_info(),
         &debug_options = context->debug_options()]() mutable
        -> absl::StatusOr<KernelReuseCache::Entry> {
      ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> cubin,
                            tile_ir::CompileTileIrBytecode(
                                bytecode, device_info, debug_options));
      return KernelReuseCache::Entry{
          std::move(kernel_name),
          LaunchDimensions(grid, stream_executor::ThreadDim(1, 1, 1)),
          /*cluster_dim=*/std::nullopt, kSharedMemoryBytes, std::move(cubin)};
    };
    if (tsl::Executor* executor = context->kernel_compiler()->executor()) {
      return xla::MakeFutureOn(*executor, std::move(assemble));
    }
    return xla::Future<KernelReuseCache::Entry>(assemble());
  };

  auto [entry_future, was_cached] = context->kernel_cache().GetWithStatus(
      CudaTileKernelFingerprint(backend_config, num_args), generate);

  Thunk::ThunkInfo info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr, context->GetNextThunkId());
  return entry_future.Map(
      [info = std::move(info), kernel_arguments = std::move(kernel_arguments),
       thunks = std::move(thunks), grid,
       block](const KernelReuseCache::Entry* entry) mutable
          -> absl::StatusOr<ThunkSequence> {
        ABSL_ASSIGN_OR_RETURN(
            CustomKernel kernel,
            kernel::CreateOwnedCubinCustomKernel(
                entry->kernel_name, entry->binary,
                kernel_arguments.args().size(), grid, block,
                kSharedMemoryBytes));
        thunks.emplace_back(std::make_unique<CustomKernelThunk>(
            std::move(info), std::move(kernel), kernel_arguments));
        return std::move(thunks);
      });
}

}  // namespace gpu
}  // namespace xla
