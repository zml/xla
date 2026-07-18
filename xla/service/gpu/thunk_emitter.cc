/*Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/thunk_emitter.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/llvm/llvm_emitter.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"
#include "xla/backends/gpu/runtime/convolution_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/device_to_host_copy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/host_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/infeed_thunk.h"
#include "xla/backends/gpu/runtime/legacy_custom_call_thunk.h"
#include "xla/backends/gpu/runtime/norm_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/rng_seed_thunk.h"
#include "xla/backends/gpu/runtime/select_k_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/topk.h"
#include "xla/backends/gpu/runtime/triangular_solve_thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/backends/gpu/transforms/dynamic_slice_copy.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/ffi/attribute_map.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/transforms/gpu_passes.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/custom_kernel_emitter.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_hlo_ordering.h"
#include "xla/service/gpu/gpu_norm_runner.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/metal_custom_calls.h"
#include "xla/backends/gpu/runtime/metal_gemm_thunk.h"
#include "xla/backends/gpu/runtime/metal_flash_attn_thunk.h"
#include "xla/backends/gpu/runtime/metal_fp8_gemv_thunk.h"
#include "xla/backends/gpu/runtime/metal_moe_gemv_thunk.h"
#include "xla/backends/gpu/runtime/metal_nvfp4_matmul_thunk.h"
#include "xla/backends/gpu/runtime/metal_workspace.h"
#include "xla/backends/gpu/runtime/metal_gdn_thunk.h"
#include "xla/backends/gpu/runtime/metal_kv_write_thunk.h"
#include "xla/backends/gpu/runtime/metal_paged_attn_thunk.h"
#include "xla/backends/gpu/runtime/metal_print_thunk.h"
#include "xla/backends/gpu/runtime/metal_sort_thunk.h"
#include "xla/backends/gpu/runtime/metal_topk_thunk.h"
#include "xla/service/gpu/metalblas_gemm.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/triton_call.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/human_readable_json.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla::gpu {
namespace {

absl::StatusOr<TritonKernelSource> EmitTritonFrom(
    const TritonCall& call, const std::string& kernel_name,
    mlir::MLIRContext& mlir_context) {
  VLOG(3) << "Generating: " << kernel_name;

  mlir::OwningOpRef<mlir::ModuleOp> triton_module;
  {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&mlir_context);
    triton_module =
        mlir::parseSourceString<mlir::ModuleOp>(call.ir, &mlir_context);
    if (!triton_module) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse Triton module: ",
                       diagnostic_handler.ConsumeStatus().message(),
                       "\ninput ir: \"", absl::CHexEscape(call.ir), "\""));
    }
  }

  auto triton_fn = triton_module->lookupSymbol<mlir::triton::FuncOp>(call.name);
  TF_RET_CHECK(triton_fn) << "Call name not found in the Triton module: "
                          << call.name;
  triton_fn.setName(kernel_name);

  return TritonKernelSource(std::move(triton_module));
}

// TODO: move into a host_execute specific file.
bool IsHostExecuteCustomCall(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() ==
             "HostExecute";  // TODO: this constant string should be shared with
                             // the TPU one
}

template <typename ThunkType>
static constexpr bool kRequiresCollectiveKernelThunk =
    std::is_constructible_v<ThunkType, Thunk::ThunkInfo,
                            const HloAllReduceInstruction*,
                            std::vector<CollectiveThunk::Buffer>,
                            std::unique_ptr<CollectiveKernelThunk>,
                            /*p2p_memcpy_enabled=*/bool>;

Shape GetOutputShape(const HloInstruction* async_start,
                     const HloInstruction* inst, Thunk::Kind kind) {
  // Order of evaluation:
  // 1. AsyncStart <- instr points to the collective directly so shape() works.
  // 2. AllGather <- tuple output (context, output)
  // 3. AllReduce <- single output
  if (async_start->opcode() == HloOpcode::kAsyncStart) {
    return inst->shape();
  }
  if (kind == Thunk::Kind::kAllGather) {
    return inst->shape().tuple_shapes(1);
  }
  return inst->shape();
}

ShapeIndex GetDstShapeIndex(const HloInstruction* async_start,
                            const HloInstruction* inst, int64_t operand_index,
                            Thunk::Kind kind) {
  Shape output_shape = GetOutputShape(async_start, inst, kind);
  bool is_multi_output = output_shape.IsTuple();
  if (async_start->opcode() == HloOpcode::kAsyncStart) {
    return is_multi_output ? ShapeIndex({1, operand_index}) : ShapeIndex({1});
  }
  if (kind == Thunk::Kind::kAllGather) {
    return is_multi_output ? ShapeIndex({1, operand_index}) : ShapeIndex({1});
  }
  return is_multi_output ? ShapeIndex({operand_index}) : ShapeIndex({});
}

}  // namespace

// The signature of this function would change to absl::Status once we lift the
// CollectiveKernelThunk out as a top level thunk. It would then become a member
// function of ThunkEmitter.
// As it stands now the collective kernel thunk is wrapped inside other
// collective thunks such as AllReduceStart. So this function is only
// responsible for emitting the collective kernel thunk and its dependencies.
xla::Future<std::unique_ptr<CollectiveKernelThunk>> EmitCollectiveKernelThunk(
    IrEmitterContext* ir_emitter_context, const CallGraph& call_graph,
    Thunk::ThunkInfo thunk_info, std::vector<CollectiveThunk::Buffer> buffers,
    const HloAllReduceInstruction* instr, const CollectiveConfig& config,
    ThunkEmitter* compiler,
    std::vector<std::unique_ptr<HloFusionAnalysis>>&
        analysis_garbage_collector) {
  std::unique_ptr<HloModule> fused_module =
      NewModuleWithFusion(instr, HloInstruction::FusionKind::kLoop);
  HloFusionInstruction* fusion_instr = Cast<HloFusionInstruction>(
      fused_module->entry_computation()->root_instruction());
  const bool has_rank_higher_than_1 =
      instr->shape().IsArray() && instr->shape().dimensions().size() > 1;
  bool is_collective_kernel_enabled =
      instr->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_unsupported_use_all_reduce_one_shot_kernel();
  static constexpr bool kMultimemDisabled = false;
  if (is_collective_kernel_enabled && has_rank_higher_than_1) {
    const int64_t size_bytes =
        ShapeUtil::ElementsIn(instr->shape()) *
        primitive_util::ByteWidth(instr->shape().element_type());
    const bool is_two_shot =
        GetAllReduceStrategy(size_bytes, kMultimemDisabled) ==
        se::gpu::AllReduceStrategy::kTwoShot;
    if (is_two_shot) {
      RETURN_IF_ERROR(FlattenCollectiveFusion(fusion_instr));
    }
  }
  const auto make_thunk =
      [thunk_info = std::move(thunk_info), buffers = std::move(buffers), config,
       is_async = !IsGPUSyncCollective(*instr), is_collective_kernel_enabled](
          absl::string_view kernel_name, int32_t shmem_bytes,
          LaunchDimensions launch_dimensions, const std::vector<uint8_t>& cubin,
          bool use_pdl) {
        return std::make_unique<CollectiveKernelThunk>(
            thunk_info, config, is_async, std::move(buffers),
            is_collective_kernel_enabled, kernel_name, launch_dimensions,
            shmem_bytes, kMultimemDisabled,
            !cubin.empty() ? std::make_optional(cubin) : std::nullopt, use_pdl);
      };
  const GpuTopology& gpu_topology = ir_emitter_context->gpu_topology();
  const DeviceAssignment* device_assignment = nullptr;
  if (ir_emitter_context->hlo_module()
          .config()
          .has_static_device_assignment()) {
    device_assignment =
        &ir_emitter_context->hlo_module().config().static_device_assignment();
  }
  ASSIGN_OR_RETURN(bool did_set_config,
                   TrySetGpuBackendConfigForCollective(
                       gpu_topology, fusion_instr, device_assignment));
  if (!did_set_config) {
    // TODO(b/522693539):
    // Because of lack of topology information in the CollectiveKernelThunk,
    // we cannot know during emission if we can use the collective kernel
    // thunk or not.
    return nullptr;
  }
  analysis_garbage_collector.push_back(
      std::make_unique<HloFusionAnalysis>(HloFusionAnalysis::Create(
          *fusion_instr, ir_emitter_context->gpu_device_info())));
  auto emitter =
      std::make_unique<TritonFusion>(*analysis_garbage_collector.back());

  ASSIGN_OR_RETURN(std::vector<Shape> unmanaged_arguments,
                   GetCollectiveUnmanagedKernelArguments(fusion_instr));
  return emitter
      ->Emit(*ir_emitter_context, *fusion_instr,
             /*instr_override=*/instr, unmanaged_arguments)
      .Map([make_thunk = std::move(make_thunk),
            fused_module =
                std::move(fused_module)](TritonFusion::EmitResult result) {
        return make_thunk(result.entry.kernel_name, result.entry.shmem_bytes,
                          result.entry.launch_dimensions,
                          std::move(result.entry.binary), result.entry.use_pdl);
      });
}

void AppendThunkSequence(ThunkSequence& thunks,
                         ThunkSequence& additional_thunks) {
  thunks.insert(thunks.end(),
                std::make_move_iterator(additional_thunks.begin()),
                std::make_move_iterator(additional_thunks.end()));
}

ThunkSequence FlattenThunkSequence(std::vector<ThunkSequence>&& sequences) {
  ThunkSequence result;

  int total = 0;
  for (const ThunkSequence& seq : sequences) {
    total += seq.size();
  }
  result.reserve(total);

  for (ThunkSequence& seq : sequences) {
    AppendThunkSequence(result, seq);
  }
  return result;
}

absl::StatusOr<std::string> CanonicalGemmHlo(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(auto gpu_config, instr->backend_config<GpuBackendConfig>());

  auto* gemm_config = gpu_config.has_grouped_gemm_backend_config()
                          ? gpu_config.mutable_grouped_gemm_backend_config()
                                ->mutable_gemm_backend_config()
                          : gpu_config.mutable_gemm_backend_config();

  // Clear algorithm-specific fields from the cache key
  gemm_config->clear_selected_algorithm();
  gemm_config->set_autotune_workspace_size(0);
  return instr->ToString(HloPrintOptions::Fingerprint()) +
         BackendConfigWrapper(gpu_config).GetRawString();
}

ThunkEmitter::ThunkEmitter(
    IrEmitterContext* absl_nonnull ir_emitter_context,
    llvm_ir::LLVMCommandLineOptionsReleasableLock* absl_nonnull
        llvm_options_lock)
    : ir_emitter_context_(ir_emitter_context),
      send_recv_events_(std::make_shared<HostSendRecvAsyncEvents>()),
      call_graph_(CallGraph::Build(&ir_emitter_context->hlo_module())),
      constants_module_context_(std::make_unique<llvm::LLVMContext>()),
      constants_module_(ir_emitter_context_->CreateLLVMModule(
          absl::StrCat(ir_emitter_context_->hlo_module().name(), "_consts"),
          *constants_module_context_)),
      llvm_options_lock_(llvm_options_lock) {}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConstant(
    const HloConstantInstruction* instr) {
  ASSIGN_OR_RETURN(DenseDataIntermediate content,
                   LiteralToXlaFormat(instr->literal()));

  int element_bytes =
      primitive_util::ByteWidth(instr->literal().shape().element_type());
  TF_RET_CHECK(content.span().size() % element_bytes == 0);
  // Treat packed constants as a byte constant.
  int num_elements = content.span().size() / element_bytes;

  std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
  ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSliceForHlo(instr, {}));

  // LLVM and PTXAS don't deal well with large constants, so we only emit very
  // small constants directly in LLVM IR.  Larger constants are emitted with
  // zero initializers in LLVM IR and are later overwritten when the PTX/CUBIN
  // is loaded.
  bool should_emit_initializer = num_elements <= 1;
  AppendGlobalConstant(constants_module_.get(), num_elements, element_bytes,
                       global_name, slice.index(), content,
                       should_emit_initializer);

  GpuExecutable::ConstantInfo info;
  info.symbol_name.assign(global_name);
  info.allocation_index = slice.index();
  // Metal does not materialize LLVM/PTX globals at runtime, so preserve even
  // small constants that are emitted with initializers in the LLVM module.
  if (!should_emit_initializer || platform_name() == "METAL") {
    info.content = content;
  }

  ir_emitter_context_->constants().push_back(std::move(info));
  return ThunkSequence{};
}

ThunkSequence GetThunkSequence(std::unique_ptr<Thunk> ir_emitter) {
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(ir_emitter));
  return thunk_sequence;
}

AsyncThunkSequence ThunkEmitter::EmitConditional(const HloInstruction* instr) {
  std::vector<AsyncThunkSequence> branch_thunks;
  branch_thunks.reserve(instr->branch_count());
  for (HloComputation* comp : instr->branch_computations()) {
    branch_thunks.emplace_back(EmitHloComputation(comp));
  }
  ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSliceForHlo(instr->operand(0), {}));

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ShapedSlice shaped_slice{slice, instr->operand(0)->shape()};
  return tsl::JoinFutures(absl::MakeSpan(branch_thunks))
      .Map([thunk_info = std::move(thunk_info),
            shaped_slice = std::move(shaped_slice)](
               std::vector<ThunkSequence> branch_thunks) {
        return GetThunkSequence(std::make_unique<ConditionalThunk>(
            std::move(thunk_info), std::move(shaped_slice),
            std::move(branch_thunks)));
      });
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
AsyncThunkSequence ThunkEmitter::EmitPadToStatic(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  ASSIGN_OR_RETURN(
      KernelDefinition<LlvmKernelSource> kernel_def,
      EmitPadToStaticLLVMIR(instr, ir_emitter_context_, kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](std::unique_ptr<Thunk> thunk) {
        return ThunkSequence::Of(std::move(thunk));
      });
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
AsyncThunkSequence ThunkEmitter::EmitSliceToDynamic(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));
  ASSIGN_OR_RETURN(
      KernelDefinition<LlvmKernelSource> kernel_def,
      EmitSliceToDynamicLLVMIR(instr, ir_emitter_context_, kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](std::unique_ptr<Thunk> thunk) {
        return ThunkSequence::Of(std::move(thunk));
      });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolutionThunk(
    const HloCustomCallInstruction* instr) {
  std::vector<ShapedSlice> operand_slices;
  operand_slices.reserve(instr->operand_count());
  for (const HloInstruction* operand : instr->operands()) {
    ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(operand, {}));
    operand_slices.push_back(slice);
  }

  // The first and the last element in the result tuple for a convolution are
  // always the result and the scratch buffer. It may have auxiliary results in
  // addition to the main result.
  std::vector<ShapedSlice> result_slices;
  for (int i = 0; i < instr->shape().tuple_shapes().size() - 1; i++) {
    ASSIGN_OR_RETURN(ShapedSlice result_slice,
                     GetShapedSliceForHlo(instr, {i}));
    result_slices.push_back(result_slice);
  }

  ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  ASSIGN_OR_RETURN(auto gpu_config, instr->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& backend_config =
      gpu_config.cudnn_conv_backend_config();
  ASSIGN_OR_RETURN(
      BufferAllocation::Slice scratch_slice,
      GetAllocationSliceForHlo(
          instr,
          {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  GpuConvDescriptor descriptor = {kind,
                                  backend_config,
                                  instr->operand(0)->shape(),
                                  instr->operand(1)->shape(),
                                  instr->shape().tuple_shapes(0),
                                  static_cast<size_t>(scratch_slice.size()),
                                  instr->window(),
                                  instr->convolution_dimension_numbers(),
                                  instr->feature_group_count()};
  ASSIGN_OR_RETURN(auto thunk,
                   ConvolutionThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           instr, ir_emitter_context_->GetNextThunkId()),
                       std::move(descriptor), std::move(operand_slices),
                       std::move(result_slices), scratch_slice));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulThunk(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));
  bool has_matrix_bias = config.beta() != 0;

  TF_RET_CHECK(instr->operand_count() ==
               2 + int{has_matrix_bias} + int{has_vector_bias});

  ASSIGN_OR_RETURN(bool has_aux_output,
                   xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));
  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ShapedSlice c;
  if (has_matrix_bias) {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(2)));
  } else {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    ASSIGN_OR_RETURN(
        bias, GetShapedSliceForHlo(instr->operand(has_matrix_bias ? 3 : 2)));
  }

  std::optional<ShapedSlice> aux;
  if (has_aux_output) {
    ASSIGN_OR_RETURN(aux, GetShapedSliceForHlo(instr, {1}));
  }

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().IsTuple() &&
      (instr->shape().tuple_shapes().size() - has_aux_output - 1)) {
    TF_RET_CHECK(
        (has_aux_output && instr->shape().tuple_shapes().size() == 3) ||
        (!has_aux_output && instr->shape().tuple_shapes().size() == 2));
    ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt, bias, aux, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalGemmThunk(
    const HloCustomCallInstruction* instr) {
  // metalBLAS works in row-major terms (op(A)=M×K, op(B)=K×N), so derive the
  // GEMM geometry from the custom call's operands + dot dimension numbers the
  // same way the legacy Metal matmul matcher (AnalyzeMatmulShapes) does. The
  // mpp_tensor kernel handles EVERY rank-2 f32/f16/bf16 dot: any dense input
  // layout, any transpose combo, and both output layouts (dim0-major direct;
  // {0,1} column-major via the transposed-swap below). So fall_back fires only
  // for dots it genuinely can't do: a non-f32/f16/bf16 dtype, or a batched /
  // >1-contracting-dim / non-rank-2 dot (those hit the checks below before any
  // kernel selection). It fails LOUD rather than producing a wrong result (D9).
  auto fall_back = [&]() -> absl::StatusOr<ThunkSequence> {
    return absl::UnimplementedError(
        "Metal GEMM: unsupported dot (non-f32/f16/bf16, batched, >1 contracting "
        "dim, or non-rank-2) — not wired to metalBLAS.");
  };

  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const DotDimensionNumbers& dnums =
      gpu_config.gemm_backend_config().dot_dimension_numbers();

  const HloInstruction* lhs = instr->operand(0);
  const HloInstruction* rhs = instr->operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& out_shape = instr->shape().IsTuple()
                               ? instr->shape().tuple_shapes(0)
                               : instr->shape();

  const PrimitiveType dtype = out_shape.element_type();
  if (dtype != F32 && dtype != F16 && dtype != BF16) return fall_back();
  if (lhs_shape.element_type() != dtype || rhs_shape.element_type() != dtype) {
    return fall_back();
  }
  // Non-batched, rank-2, one contracting dim each, row-major operands+output.
  if (dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      lhs_shape.dimensions().size() != 2 ||
      rhs_shape.dimensions().size() != 2 ||
      out_shape.dimensions().size() != 2) {
    return fall_back();
  }
  // Every operand must be a dense, untiled rank-2 buffer (both {1,0} and {0,1}
  // qualify). metalBLAS reads each as a physically contiguous 2-D buffer and
  // derives its TRANS flag / leading dim from the physical layout, so any dense
  // layout is fine (the {0,1} output is handled by the transposed-swap below). A
  // tiled/exotic layout goes to fall_back — mpp_tensor doesn't model it.
  auto dense_rank2 = [](const Shape& s) {
    return s.has_layout() && s.layout().tiles().empty() &&
           s.layout().minor_to_major().size() == 2;
  };
  if (!dense_rank2(lhs_shape) || !dense_rank2(rhs_shape) ||
      !dense_rank2(out_shape)) {
    return fall_back();
  }

  const int lc = dnums.lhs_contracting_dimensions(0);
  const int rc = dnums.rhs_contracting_dimensions(0);
  if ((lc != 0 && lc != 1) || (rc != 0 && rc != 1)) return fall_back();
  const int64_t M = lhs_shape.dimensions(lc == 0 ? 1 : 0);
  const int64_t K = lhs_shape.dimensions(lc);
  const int64_t N = rhs_shape.dimensions(rc == 0 ? 1 : 0);
  if (rhs_shape.dimensions(rc) != K || out_shape.dimensions(0) != M ||
      out_shape.dimensions(1) != N) {
    return fall_back();
  }
  // op(A) must be M×K, op(B) must be K×N. metalBLAS's TRANS flag says whether
  // the contiguous buffer is [M,K]/[K,N] (op = itself, lda/ldb = the row width)
  // or its transpose. Derive it from the PHYSICAL layout (which logical dim is
  // minor/contiguous), not the logical dim order: op(A) wants K contiguous
  // (trans_a iff K is NOT the minor dim); op(B) wants N contiguous (trans_b iff
  // the contracting dim K IS the minor dim). For the usual row-major {1,0}
  // operands this reduces to (lc==0)/(rc==1); it also accepts the {0,1} bitcast
  // XLA hands the lm_head lhs -- physically the [M,K] normed hidden -- which the
  // old dim0-major-only check rejected (LFM2 and any tied-embedding lm_head).
  const bool trans_a = (lhs_shape.layout().minor_to_major(0) != lc);
  const bool trans_b = (rhs_shape.layout().minor_to_major(0) == rc);

  // PREFILL token-axis clamp: if this computation contains a prefill (q_len>1)
  // zml$flash_attn carrying a num_tokens operand, and this GEMM's M equals that
  // prefill seqlen (= q_len), then it is one of the per-layer token-axis GEMMs
  // ([seqlen, *] = QKV / O / gate+up / down). Pass num_tokens so the thunk clamps
  // the M-row grid to the real prompt length at execute. The lm_head GEMM lives in
  // a separate Exe with no flash_attn (and is already last-token-sliced), so it is
  // never matched; decode GEMMs (flash_attn q_len==1) are not matched either.
  // Detected BEFORE compile so the tile picker can optimize for the clamped grid.
  BufferAllocation::Slice num_tokens;
  Shape num_tokens_shape;
  bool prefill_token_axis = false;
  for (const HloInstruction* i : instr->parent()->instructions()) {
    if (i->opcode() != HloOpcode::kCustomCall) continue;
    const auto* fa = Cast<HloCustomCallInstruction>(i);
    if (fa->custom_call_target() != "zml$flash_attn") continue;
    const Shape& fq = fa->operand(0)->shape();
    if (fq.dimensions().size() != 3 || fq.dimensions(1) <= 1) continue;  // decode
    if (fq.dimensions(1) != M) continue;  // M is not this layer's token axis
    const bool fc = fa->operand(1)->shape().dimensions().size() == 4;
    const int fbase = fc ? 5 : 4;  // q,k,v,tok[,layer]
    if (fa->operand_count() != fbase + 1) continue;  // no num_tokens operand
    TF_ASSIGN_OR_RETURN(num_tokens,
                        GetAllocationSliceForHlo(fa->operand(fbase), {}));
    num_tokens_shape = fa->operand(fbase)->shape();
    prefill_token_axis = true;
    break;
  }
  // Same clamp for PAGED prefill modules (llmd's chunked prefill): a prefill
  // zml$paged_attn (total_q_tokens > num_seqs) whose q token axis equals M.
  // The real token count is query_start_len[num_seqs] (operand 5, the
  // [num_seqs+1] i32 cumulative query lengths — tokens are packed from row 0,
  // so rows >= qsl[num_seqs] are padding). Point the num_tokens slice at that
  // LAST element; the thunk's host-side u32 read then works unchanged. Without
  // this, llmd prefill GEMMs ran the full padded M=seqlen with the default
  // fat tile (measured: down-proj 1088us/layer at ~50GB/s vs the CLI's clamped
  // ~126us — a ~4.5x whole-prefill gap, 140ms vs 32ms warm).
  if (!prefill_token_axis) {
    for (const HloInstruction* i : instr->parent()->instructions()) {
      if (i->opcode() != HloOpcode::kCustomCall) continue;
      const auto* pa = Cast<HloCustomCallInstruction>(i);
      if (pa->custom_call_target() != "zml$paged_attn") continue;
      if (pa->operand_count() != 6) continue;
      const Shape& pq = pa->operand(0)->shape();  // [total_q, heads, hd]
      const Shape& bt = pa->operand(3)->shape();  // [num_seqs, max_blocks]
      if (pq.dimensions().size() != 3 || bt.dimensions().size() != 2) continue;
      const int64_t total_q = pq.dimensions(0);
      const int64_t num_seqs = bt.dimensions(0);
      if (total_q <= num_seqs) continue;  // decode module
      if (total_q != M) continue;  // M is not this layer's token axis
      const Shape& qsl_shape = pa->operand(5)->shape();
      if (qsl_shape.dimensions().size() != 1 ||
          qsl_shape.dimensions(0) != num_seqs + 1 ||
          ShapeUtil::ByteSizeOfPrimitiveType(qsl_shape.element_type()) != 4) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice qsl,
                          GetAllocationSliceForHlo(pa->operand(5), {}));
      num_tokens = BufferAllocation::Slice(qsl.allocation(),
                                           qsl.offset() + num_seqs * 4, 4);
      num_tokens_shape = ShapeUtil::MakeShape(S32, {});
      prefill_token_axis = true;
      break;
    }
  }

  const xla::ShapeIndex out_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSliceForHlo(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice c,
                      GetAllocationSliceForHlo(instr, out_index));

  // COLUMN-MAJOR ({0,1}) OUTPUT: the only rank-2 shape mpp_tensor can't take (it
  // writes op(C)=M×N row-major, ldc=N). But [M,N]{0,1} is byte-identical to
  // [N,M]{1,0}, so compute the TRANSPOSE Cᵀ=[N,M] with the SAME kernel and hand
  // the bytes back as the {0,1} output — no new kernel, native bf16. Cᵀ =
  // op(B)ᵀ·op(A)ᵀ: swap the operands (new lhs = B, new rhs = A → M'=N, N'=M) and
  // re-derive the trans flags with lhs<->rhs / lc<->rc swapped (same physical-
  // layout rule as above). metalBLAS writes [N,M] row-major into c; XLA reads c
  // as {0,1} [M,N]. {1,0} and {0,1} are the only rank-2 dense output layouts, so
  // this + the direct path cover every dot that reaches here.
  //
  // NOTE: in practice XLA NORMALIZES every dot output to {1,0} (it folds a
  // transpose-of-dot into an operand-swap at the HLO level itself), so this
  // branch is not expected to fire on real graphs — verified: transpose(A·B) and
  // a (A·B)·E chain both emit {1,0}-output __metal$gemms. It is kept purely for
  // correctness/defense (a future XLA layout-pass change, or an op that pins a
  // {0,1} operand layout): the alternative was fall_back = a hard crash. Cheap
  // and provably correct, so total coverage beats a latent Unimplemented.
  int64_t gm = M, gn = N;
  bool gtrans_a = trans_a, gtrans_b = trans_b;
  BufferAllocation::Slice ga = a, gb = b;
  Shape g_lhs_shape = lhs_shape, g_rhs_shape = rhs_shape;
  if (!LayoutUtil::IsMonotonicWithDim0Major(out_shape.layout())) {
    gm = N;
    gn = M;
    ga = b;
    gb = a;
    g_lhs_shape = rhs_shape;
    g_rhs_shape = lhs_shape;
    gtrans_a = (rhs_shape.layout().minor_to_major(0) != rc);  // op(A')=[N,K], K=rc
    gtrans_b = (lhs_shape.layout().minor_to_major(0) == lc);  // op(B')=[K,M], K=lc
    // The swapped shape is not a per-layer token-axis GEMM (those are dim0-major);
    // drop any clamp match so the row-clamp arg isn't wrongly bound.
    prefill_token_axis = false;
    num_tokens = BufferAllocation::Slice();
    num_tokens_shape = Shape();
  }

  // Kernel-family ladder, most specialized first; each rung returns
  // Unimplemented outside its regime and the next one takes over.
  //   1. MLX steel split-K — thin-M batched-decode x@Wᵀ (2<=M<=16, f16/bf16,
  //      512<=N<=8192, deep K): grid-level K-split at ~bandwidth peak
  //      (16.7-103us vs gemv_bt's 25-147 on the llmd shapes).
  //   2. metalBLAS GEMV (gemv_t / gemv_bt) — M==1 decode/lm_head + the thin-M
  //      shapes split-K declines.
  //   3. mpp_tensor GEMM — everything else, and never declines for f32/f16/bf16
  //      (any transpose combo), so no rank-2 float dot falls through. gb is the
  //      rhs (matrix B); its slice offset feeds the GEMV VEC-load alignment clamp.
  absl::StatusOr<MetalGemmLaunch> launch =
      prefill_token_axis
          ? absl::StatusOr<MetalGemmLaunch>(
                absl::UnimplementedError("prefill: tensor path"))
          : CompileMetalblasSplitk(gm, gn, K, gtrans_a, gtrans_b, dtype);
  if (absl::IsUnimplemented(launch.status())) {
    launch =
        CompileMetalblasGemv(gm, gn, K, gtrans_a, gtrans_b, dtype, gb.offset());
  }
  if (absl::IsUnimplemented(launch.status())) {
    launch = CompileMetalblasGemm(gm, gn, K, gtrans_a, gtrans_b, dtype,
                                  prefill_token_axis);
  }
  if (absl::IsUnimplemented(launch.status())) {
    return fall_back();  // unreachable for f32/f16/bf16 (mpp_tensor never declines).
  }
  TF_RETURN_IF_ERROR(launch.status());

  auto thunk = std::make_unique<MetalGemmThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(*launch), ga, g_lhs_shape, gb, g_rhs_shape, c, out_shape,
      num_tokens, num_tokens_shape);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalPrintThunk(
    const HloCustomCallInstruction* instr) {
  const HloInstruction* operand = instr->operand(0);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSliceForHlo(operand, {}));
  // Label: the op_name metadata carries zml's print(name=...); fall back to the
  // instruction name. The shape (printed too) disambiguates in practice.
  std::string label = instr->metadata().op_name().empty()
                          ? std::string(instr->name())
                          : instr->metadata().op_name();
  auto thunk = std::make_unique<MetalPrintThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(label), slice, operand->shape());
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalFlashAttnThunk(
    const HloCustomCallInstruction* instr) {
  const int operand_count = instr->operand_count();
  if (operand_count < 4 || operand_count > 6) {
    return absl::InvalidArgumentError(
        "zml$flash_attn expects 4 (q,k,v,tok), 5 (+layer or +num_tokens), or 6 "
        "(+layer +num_tokens) operands.");
  }
  // WHOLE-cache feed (RelaxFlashAttnKVLayout folded away the per-token layer
  // dynamic-slice): k/v are the rank-4 [n_layer, seqlen, n_kv, hd] cache and
  // operand 4 is the layer index. Otherwise k/v are a single sliced layer
  // [n_kv, seqlen, hd] (rank 3). PREFILL also carries num_tokens (the real prompt
  // length, a u32 scalar) as the TRAILING operand so the thunk can clamp the
  // query-row grid. Detect both from rank + count (not count alone).
  const bool kv_full_cache =
      (instr->operand(1)->shape().dimensions().size() == 4);
  const int base_operands = kv_full_cache ? 5 : 4;  // q,k,v,tok[,layer]
  const bool has_num_tokens = (operand_count == base_operands + 1);
  const Shape& q_shape = instr->operand(0)->shape();
  const Shape& k_shape = instr->operand(1)->shape();
  const Shape& v_shape = instr->operand(2)->shape();
  const Shape& tok_shape = instr->operand(3)->shape();
  const Shape& out_shape = instr->shape();

  const int64_t k_rank = k_shape.dimensions().size();
  if (q_shape.dimensions().size() != 3 || k_rank != (kv_full_cache ? 4 : 3) ||
      static_cast<int64_t>(v_shape.dimensions().size()) != k_rank) {
    return absl::UnimplementedError("zml$flash_attn: unexpected q/k/v rank.");
  }
  const int64_t n_q = q_shape.dimensions(0);
  const int64_t q_len = q_shape.dimensions(1);
  const int64_t hd = q_shape.dimensions(2);
  const int64_t n_kv =
      kv_full_cache ? k_shape.dimensions(2) : k_shape.dimensions(0);
  const int64_t seqlen = k_shape.dimensions(1);
  // q_len == 1 is decode (fa_vec); q_len > 1 is prefill (the simdgroup-matrix
  // kernel_flash_attn_ext). Both route to MetalFlashAttnThunk, which branches on
  // q_len internally. q_len must be >= 1.
  if (q_len < 1) {
    return absl::UnimplementedError("zml$flash_attn: query length must be >= 1.");
  }
  if (hd % 32 != 0 || n_kv == 0 || n_q % n_kv != 0 ||
      k_shape.dimensions(k_rank - 1) != hd ||
      v_shape.dimensions(k_rank - 1) != hd ||
      v_shape.dimensions(1) != seqlen) {
    return absl::UnimplementedError(
        "zml$flash_attn unsupported shapes (need hd%32==0, n_q%n_kv==0, "
        "matching k/v).");
  }
  // Every kernel path is bf16-typed: fa_vec / prefill are bf16-gated in the
  // thunk, and the serial fallback kernel reads/writes `bfloat` directly — a
  // non-bf16 operand would be silently bit-reinterpreted, not converted.
  // Loud, never silently wrong.
  // TODO: parameterize the kernels' element type to support f16 models.
  if (q_shape.element_type() != BF16 || k_shape.element_type() != BF16 ||
      v_shape.element_type() != BF16 || out_shape.element_type() != BF16) {
    return absl::UnimplementedError(
        "zml$flash_attn: q/k/v/out must be bf16 (the Metal flash-attention "
        "kernels are bf16-typed).");
  }
  // Prefill (q_len>1) has only the simdgroup-matrix kernel (the serial
  // fallback is single-query); reject unsupported prefill geometry at compile
  // time instead of at first execute. Decode (q_len==1) keeps the serial
  // fallback for any hd%32==0, so it is not gated here.
  // TODO: port head_dim!=128 (Gemma-shaped hd=256) and unaligned-seqlen
  // variants of the prefill kernel; until then such models must use the paged
  // path, whose tiled kernel covers hd 64/96/128/256/512.
  if (q_len > 1 && ((hd != 128 && hd != 64) || seqlen % 64 != 0)) {
    return absl::UnimplementedError(
        "zml$flash_attn: prefill (q_len>1) needs head_dim 64 or 128 and "
        "seqlen%64==0.");
  }
  const int64_t n_groups = n_q / n_kv;

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice q,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k,
                      GetAllocationSliceForHlo(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v,
                      GetAllocationSliceForHlo(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice tok,
                      GetAllocationSliceForHlo(instr->operand(3), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out,
                      GetAllocationSliceForHlo(instr, {}));

  // The layer-index operand (only present in the whole-cache feed).
  BufferAllocation::Slice layer;
  Shape layer_shape;
  if (kv_full_cache) {
    TF_ASSIGN_OR_RETURN(layer, GetAllocationSliceForHlo(instr->operand(4), {}));
    layer_shape = instr->operand(4)->shape();
  }

  // The num_tokens operand (prefill only): the real prompt length, read host-side
  // at execute to clamp the query-row grid. Trailing operand after q,k,v,tok[,layer].
  BufferAllocation::Slice num_tokens;
  Shape num_tokens_shape;
  if (has_num_tokens) {
    TF_ASSIGN_OR_RETURN(
        num_tokens, GetAllocationSliceForHlo(instr->operand(base_operands), {}));
    num_tokens_shape = instr->operand(base_operands)->shape();
  }

  // The thunk reads `tok` host-side at encode ONLY to pick a decode perf variant
  // (the nsg ramp / head-contiguous gate); the result is numerically identical
  // regardless, because fa_vec's KV-loop ceiling is the static seqlen and causality
  // comes from the on-device tok[0] (see flash_attn_vec.metal). That host read is
  // safe ONLY when `tok` is a host-staged entry parameter — RelaxFlashAttnKVLayout
  // substitutes the raw token-position param for ZML's device convert(u32->s32) on
  // the whole-cache path. If `tok` is still a GPU-produced value (the layout-only
  // fallback, or no raw param was found), dereferencing it host-side would race the
  // producer on Metal (no totally-ordered host/GPU path — the GDN-prefill race
  // class), so signal the thunk to skip the read and use a static default (kv =
  // seqlen) instead. A parameter (possibly behind pure bitcast/reshape relabels) is
  // host-valid at encode; a convert/compute is not.
  const HloInstruction* tok_src = instr->operand(3);
  while (tok_src->opcode() == HloOpcode::kBitcast ||
         tok_src->opcode() == HloOpcode::kReshape) {
    tok_src = tok_src->operand(0);
  }
  const bool tok_host_coherent = tok_src->opcode() == HloOpcode::kParameter;

  // K/V fed position-major ([..,seqlen,n_kv,hd]) instead of head-major
  // ([n_kv,seqlen,hd], operand layout {2,1,0}) when RelaxFlashAttnKVLayout
  // relaxed the layout so ZML's transpose folds to a free bitcast. The whole-
  // cache feed is always position-major within a layer; a sliced operand is
  // position-major iff its layout is {2,0,1}. The thunk derives K/V strides.
  bool kv_position_major = kv_full_cache;
  if (!kv_full_cache) {
    const Layout& k_layout = k_shape.layout();
    kv_position_major = k_layout.minor_to_major().size() == 3 &&
                        k_layout.minor_to_major(0) == 2 &&
                        k_layout.minor_to_major(1) == 0 &&
                        k_layout.minor_to_major(2) == 1;
  }

  auto thunk = std::make_unique<MetalFlashAttnThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      q, q_shape, k, k_shape, v, v_shape, tok, tok_shape, out, out_shape, n_kv,
      n_groups, seqlen, hd, kv_position_major, kv_full_cache, layer,
      layer_shape, num_tokens, num_tokens_shape, tok_host_coherent);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalPagedAttnThunk(
    const HloCustomCallInstruction* instr) {
  if (instr->operand_count() != 6) {
    return absl::InvalidArgumentError(
        "zml$paged_attn expects 6 operands "
        "(q, k_cache, v_cache, block_table, seq_lens, query_start_len).");
  }
  const Shape& q_shape = instr->operand(0)->shape();
  const Shape& k_shape = instr->operand(1)->shape();
  const Shape& v_shape = instr->operand(2)->shape();
  const Shape& bt_shape = instr->operand(3)->shape();
  const Shape& sl_shape = instr->operand(4)->shape();
  const Shape& qsl_shape = instr->operand(5)->shape();
  const Shape& out_shape = instr->shape();

  // q/out: [total_q_tokens, num_heads, head_dim]; k/v cache:
  // [num_blocks, block_size, num_kv_heads, head_dim]; block_table:
  // [num_seqs, max_num_blocks_per_seq]; seq_lens: [num_seqs];
  // query_start_len: [num_seqs+1].
  if (q_shape.dimensions().size() != 3 || k_shape.dimensions().size() != 4 ||
      v_shape.dimensions().size() != 4 || bt_shape.dimensions().size() != 2) {
    return absl::UnimplementedError("zml$paged_attn: unexpected operand ranks.");
  }
  const int64_t total_q_tokens = q_shape.dimensions(0);
  const int64_t num_heads = q_shape.dimensions(1);
  const int64_t head_dim = q_shape.dimensions(2);
  const int64_t block_size = k_shape.dimensions(1);
  const int64_t num_kv_heads = k_shape.dimensions(2);
  const int64_t num_seqs = bt_shape.dimensions(0);
  const int64_t max_num_blocks_per_seq = bt_shape.dimensions(1);

  if (num_kv_heads == 0 || num_heads % num_kv_heads != 0 ||
      k_shape.dimensions(3) != head_dim || v_shape.dimensions(1) != block_size ||
      v_shape.dimensions(2) != num_kv_heads || v_shape.dimensions(3) != head_dim) {
    return absl::UnimplementedError(
        "zml$paged_attn: inconsistent q/k/v cache shapes.");
  }
  if (head_dim != 64 && head_dim != 96 && head_dim != 128 && head_dim != 256 &&
      head_dim != 512) {
    return absl::UnimplementedError(
        "zml$paged_attn: head_dim must be 64/96/128/256/512.");
  }
  if (block_size != 8 && block_size != 16 && block_size != 32) {
    return absl::UnimplementedError(
        "zml$paged_attn: block_size must be 8/16/32.");
  }
  if (q_shape.element_type() != BF16 && q_shape.element_type() != F16) {
    return absl::UnimplementedError("zml$paged_attn: q/k/v must be bf16 or f16.");
  }

  // Attention semantics (scale / softcapping / sliding_window / is_causal) are
  // carried in the custom call's typed-FFI backend-config dictionary, set by
  // ZML's metal_attention.zig from AttentionOptions. Absent keys keep the
  // defaults (scale=1/sqrt(head_dim), no softcap, no sliding window, causal),
  // so graphs predating the attributes are unchanged. The tiled kernel applies
  // all of them (is_causal via its IS_CAUSAL function constant); the vector
  // decode kernel applies scale only and is causal-only, so the thunk routes
  // softcapped/windowed/bidirectional cases to the tiled kernel.
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  float softcapping = 0.0f;
  int sliding_window = -1;
  bool is_causal = true;
  if (const std::string& cfg = instr->raw_backend_config_string();
      !cfg.empty()) {
    mlir::Attribute parsed =
        mlir::parseAttribute(cfg, ir_emitter_context_->mlir_context());
    auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(parsed);
    if (dict == nullptr) {
      return absl::InternalError(
          "zml$paged_attn: backend config is not a dictionary attribute.");
    }
    TF_ASSIGN_OR_RETURN(ffi::AttributesMap attrs,
                        xla::ffi::BuildAttributesMap(dict));
    auto scalar = [&attrs](absl::string_view name) -> const ffi::Scalar* {
      auto it = attrs.find(name);
      if (it == attrs.end()) return nullptr;
      return std::get_if<ffi::Scalar>(&it->second.AsVariant());
    };
    if (const ffi::Scalar* s = scalar("scale")) {
      if (const float* v = std::get_if<float>(&s->AsVariant())) scale = *v;
    }
    if (const ffi::Scalar* s = scalar("softcapping")) {
      if (const float* v = std::get_if<float>(&s->AsVariant())) {
        softcapping = *v;
      }
    }
    if (const ffi::Scalar* s = scalar("sliding_window")) {
      if (const int32_t* v = std::get_if<int32_t>(&s->AsVariant())) {
        sliding_window = *v;
      }
    }
    // is_causal selects causal (default) vs bidirectional attention. The tiled
    // kernel honors it through its IS_CAUSAL function constant; is_causal=false
    // (e.g. the diffusion-draft dflash model) also forces the tiled path off the
    // causal-only vector decode kernel (see MetalPagedAttnThunk).
    if (const ffi::Scalar* s = scalar("is_causal")) {
      if (const bool* v = std::get_if<bool>(&s->AsVariant())) is_causal = *v;
    }
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice q,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k_cache,
                      GetAllocationSliceForHlo(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v_cache,
                      GetAllocationSliceForHlo(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice block_table,
                      GetAllocationSliceForHlo(instr->operand(3), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice seq_lens,
                      GetAllocationSliceForHlo(instr->operand(4), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice query_start_len,
                      GetAllocationSliceForHlo(instr->operand(5), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out,
                      GetAllocationSliceForHlo(instr, {}));

  auto thunk = std::make_unique<MetalPagedAttnThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      q, q_shape, k_cache, k_shape, v_cache, v_shape, block_table, bt_shape,
      seq_lens, sl_shape, query_start_len, qsl_shape, out, out_shape, num_heads,
      num_kv_heads, head_dim, block_size, num_seqs, max_num_blocks_per_seq,
      total_q_tokens, scale, softcapping, sliding_window, is_causal,
      q_shape.element_type());
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalGdnThunk(
    const HloCustomCallInstruction* instr) {
  if (instr->operand_count() != 8) {
    return absl::InvalidArgumentError(
        "zml$gdn expects 8 operands "
        "(q, k, v, g, beta, h0, cu_seqlens, slot_mapping).");
  }
  const Shape& q_shape = instr->operand(0)->shape();
  const Shape& k_shape = instr->operand(1)->shape();
  const Shape& v_shape = instr->operand(2)->shape();
  const Shape& g_shape = instr->operand(3)->shape();
  const Shape& beta_shape = instr->operand(4)->shape();
  const Shape& h0_shape = instr->operand(5)->shape();
  const Shape& cu_seqlens_shape = instr->operand(6)->shape();
  const Shape& slot_mapping_shape = instr->operand(7)->shape();

  // Output is a (y, ht) tuple; ht aliases h0 (set by ZML's custom-call op).
  if (!instr->shape().IsTuple() || instr->shape().tuple_shapes().size() != 2) {
    return absl::UnimplementedError(
        "zml$gdn: expected a (y, ht) tuple result.");
  }
  const Shape& y_shape = instr->shape().tuple_shapes(0);
  const Shape& ht_shape = instr->shape().tuple_shapes(1);

  // q/k: [total_tokens, Hk, Dk]; v: [total_tokens, Hv, Dv]; g/beta:
  // [total_tokens, Hv]; h0/ht: [num_seqs, Hv, Dk, Dv]; cu_seqlens:[num_seqs+1];
  // slot_mapping:[num_seqs].
  if (q_shape.dimensions().size() != 3 || k_shape.dimensions().size() != 3 ||
      v_shape.dimensions().size() != 3 || g_shape.dimensions().size() != 2 ||
      beta_shape.dimensions().size() != 2 ||
      h0_shape.dimensions().size() != 4 ||
      cu_seqlens_shape.dimensions().size() != 1 ||
      slot_mapping_shape.dimensions().size() != 1) {
    return absl::UnimplementedError("zml$gdn: unexpected operand ranks.");
  }

  const int64_t total_tokens = q_shape.dimensions(0);
  const int64_t hk = q_shape.dimensions(1);
  const int64_t dk = q_shape.dimensions(2);
  const int64_t hv = v_shape.dimensions(1);
  const int64_t dv = v_shape.dimensions(2);
  const int64_t num_seqs = h0_shape.dimensions(0);

  if (hk == 0 || hv == 0 || dk == 0 || dv == 0 || num_seqs == 0) {
    return absl::UnimplementedError("zml$gdn: invalid dimension (must be > 0).");
  }
  if (hv % hk != 0) {
    return absl::UnimplementedError(
        "zml$gdn: Hv must be a multiple of Hk (GQA-style grouping).");
  }
  if (k_shape.dimensions(0) != total_tokens || k_shape.dimensions(1) != hk ||
      k_shape.dimensions(2) != dk || v_shape.dimensions(0) != total_tokens ||
      g_shape.dimensions(0) != total_tokens || g_shape.dimensions(1) != hv ||
      beta_shape.dimensions(0) != total_tokens ||
      beta_shape.dimensions(1) != hv || h0_shape.dimensions(1) != hv ||
      h0_shape.dimensions(2) != dk || h0_shape.dimensions(3) != dv ||
      cu_seqlens_shape.dimensions(0) != num_seqs + 1 ||
      slot_mapping_shape.dimensions(0) != num_seqs) {
    return absl::UnimplementedError(
        "zml$gdn: inconsistent q/k/v/g/beta/h0/cu_seqlens/slot_mapping shapes.");
  }
  const PrimitiveType et = q_shape.element_type();
  if (et != F32 && et != F16 && et != BF16) {
    return absl::UnimplementedError("zml$gdn: operands must be f32/f16/bf16.");
  }
  if (k_shape.element_type() != et || v_shape.element_type() != et ||
      g_shape.element_type() != et || beta_shape.element_type() != et ||
      h0_shape.element_type() != et || y_shape.element_type() != et ||
      ht_shape.element_type() != et) {
    return absl::UnimplementedError(
        "zml$gdn: q/k/v/g/beta/h0/y/ht must share one element type.");
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice q,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k,
                      GetAllocationSliceForHlo(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v,
                      GetAllocationSliceForHlo(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice g,
                      GetAllocationSliceForHlo(instr->operand(3), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice beta,
                      GetAllocationSliceForHlo(instr->operand(4), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice h0,
                      GetAllocationSliceForHlo(instr->operand(5), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice cu_seqlens,
                      GetAllocationSliceForHlo(instr->operand(6), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slot_mapping,
                      GetAllocationSliceForHlo(instr->operand(7), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice y,
                      GetAllocationSliceForHlo(instr, {0}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice ht,
                      GetAllocationSliceForHlo(instr, {1}));

  auto thunk = std::make_unique<MetalGdnThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      q, q_shape, k, k_shape, v, v_shape, g, g_shape, beta, beta_shape, h0,
      h0_shape, cu_seqlens, cu_seqlens_shape, slot_mapping, slot_mapping_shape,
      y, y_shape, ht, ht_shape, num_seqs, hk, hv, dk, dv, et);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalScaledMatmulThunk(
    const HloCustomCallInstruction* instr) {
  // One HLO target; dispatch by scale/weight scheme to the matching thunk.
  if (instr->operand_count() < 3) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul expects at least 3 operands (x, w, scale).");
  }
  // Classify through the same helper the rewriter matched with, so a scheme can
  // never be fused here and unimplemented there.
  const std::optional<MetalScaledMatmulScheme> scheme =
      ClassifyMetalScaledMatmul(instr->operand(1)->shape(),
                                instr->operand(2)->shape());
  if (!scheme.has_value()) {
    return absl::UnimplementedError(absl::StrCat(
        "zml$scaled_matmul: no Metal thunk implements weight ",
        instr->operand(1)->shape().ToString(true), " with scale ",
        instr->operand(2)->shape().ToString(true), "."));
  }
  if (instr->operand_count() != 3) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul expects 3 operands (x, w, scale).");
  }
  if (instr->shape().IsTuple() && instr->shape().tuple_shapes().empty()) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul output tuple must not be empty.");
  }
  const Shape& out_shape = instr->shape().IsTuple()
                               ? instr->shape().tuple_shapes(0)
                               : instr->shape();
  auto is_row_major = [](const Shape& shape) {
    return shape.IsArray() && shape.has_layout() &&
           LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
  };
  if (!is_row_major(instr->operand(0)->shape()) ||
      !is_row_major(instr->operand(1)->shape()) ||
      !is_row_major(instr->operand(2)->shape()) ||
      !is_row_major(out_shape)) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul requires row-major contiguous x, w, scale, and "
        "output buffers.");
  }
  switch (*scheme) {
    case MetalScaledMatmulScheme::kNvfp4Group16:
      return EmitMetalNvfp4MatmulThunk(instr);
    case MetalScaledMatmulScheme::kFp8Block128:
    case MetalScaledMatmulScheme::kFp8PerChannel:
      return EmitMetalFp8GemvThunk(instr);
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalNvfp4MatmulThunk(
    const HloCustomCallInstruction* instr) {
  if (instr->operand_count() != 3) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul (NVFP4) expects 3 operands (x, w, scale).");
  }
  const Shape& x_shape = instr->operand(0)->shape();
  const Shape& w_shape = instr->operand(1)->shape();
  const Shape& scale_shape = instr->operand(2)->shape();

  // MetalWorkspaceRewriter gives the split-K path a tuple(result, s8[bytes])
  // result; every other path keeps its plain array result.
  const bool is_tuple = instr->shape().IsTuple();
  if (is_tuple && instr->shape().tuple_shapes().size() != 2) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul (NVFP4) must return either a bf16 result or "
        "tuple(bf16 result, s8 workspace).");
  }
  const Shape& out_shape =
      is_tuple ? instr->shape().tuple_shapes(0) : instr->shape();
  Shape workspace_shape;
  if (is_tuple) {
    workspace_shape = instr->shape().tuple_shapes(1);
    if (!workspace_shape.IsArray() || workspace_shape.element_type() != S8 ||
        workspace_shape.dimensions().size() != 1) {
      return absl::InvalidArgumentError(
          "zml$scaled_matmul (NVFP4): workspace must be rank-1 s8.");
    }
  }

  if (x_shape.dimensions().size() != 2 || w_shape.dimensions().size() != 2 ||
      scale_shape.dimensions().size() != 2 ||
      out_shape.dimensions().size() != 2) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (NVFP4): x, w, scale, out must all be rank 2.");
  }
  const int64_t m = x_shape.dimensions(0);
  const int64_t k = x_shape.dimensions(1);
  const int64_t n = w_shape.dimensions(0);
  if (m == 0 || k == 0 || n == 0) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (NVFP4): invalid dimension.");
  }
  if (w_shape.element_type() != F4E2M1FN || w_shape.dimensions(1) != k) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (NVFP4): w must be f4e2m1[N, K] (K minor).");
  }
  if (scale_shape.element_type() != F8E4M3FN || k % 16 != 0 ||
      scale_shape.dimensions(0) != n || scale_shape.dimensions(1) != k / 16) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (NVFP4): scale must be e4m3 [N, K/16].");
  }
  if (x_shape.element_type() != BF16 || out_shape.element_type() != BF16) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (NVFP4): x and out must be bf16.");
  }
  if (out_shape.dimensions(0) != m || out_shape.dimensions(1) != n) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (NVFP4): out must be [M, N].");
  }
  const auto metal_arch =
      ir_emitter_context_->gpu_device_info().metal_compute_capability();
  TF_ASSIGN_OR_RETURN(const int64_t expected_workspace_bytes,
                      GetMetalNvfp4WorkspaceBytes(
                          m, k, n, metal_arch.architecture_size(),
                          metal_arch.architecture_gen()));
  const int64_t workspace_bytes =
      is_tuple ? workspace_shape.dimensions(0) : 0;
  if (workspace_bytes != expected_workspace_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "zml$scaled_matmul (NVFP4): workspace must be s8[",
        expected_workspace_bytes, "]; got s8[", workspace_bytes, "]."));
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice x,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice w,
                      GetAllocationSliceForHlo(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scale,
                      GetAllocationSliceForHlo(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice out,
      GetAllocationSliceForHlo(instr, is_tuple ? ShapeIndex{0} : ShapeIndex{}));
  BufferAllocation::Slice workspace;
  if (is_tuple) {
    TF_ASSIGN_OR_RETURN(workspace,
                        GetAllocationSliceForHlo(instr, ShapeIndex{1}));
  }

  auto thunk = std::make_unique<MetalNvfp4MatmulThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      x, x_shape, w, w_shape, scale, scale_shape, out, out_shape, workspace,
      workspace_shape, m, k, n, metal_arch.architecture_size(),
      metal_arch.architecture_gen());
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalFp8GemvThunk(
    const HloCustomCallInstruction* instr) {
  if (instr->operand_count() != 3) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul (FP8) expects 3 operands (x, w, scale).");
  }
  const Shape& x_shape = instr->operand(0)->shape();
  const Shape& w_shape = instr->operand(1)->shape();
  const Shape& scale_shape = instr->operand(2)->shape();
  const bool is_tuple = instr->shape().IsTuple();
  const Shape& out_shape =
      is_tuple ? instr->shape().tuple_shapes(0) : instr->shape();

  if (x_shape.dimensions().size() != 2 || w_shape.dimensions().size() != 2 ||
      scale_shape.dimensions().size() != 2 ||
      out_shape.dimensions().size() != 2) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (FP8): x, w, scale, out must all be rank 2.");
  }
  const int64_t b = x_shape.dimensions(0);
  const int64_t k = x_shape.dimensions(1);
  const int64_t n = w_shape.dimensions(0);
  if (b == 0 || k == 0 || n == 0) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (FP8): invalid dimension.");
  }
  if (w_shape.element_type() != F8E4M3FN || w_shape.dimensions(1) != k) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (FP8): w must be f8e4m3fn[N, K] (K minor).");
  }
  const bool per_channel =
      (scale_shape.dimensions(0) == n && scale_shape.dimensions(1) == 1);
  const bool block_128 = (k % 128 == 0 && n % 128 == 0 &&
                          scale_shape.dimensions(0) == n / 128 &&
                          scale_shape.dimensions(1) == k / 128);
  if (scale_shape.element_type() != BF16 || (!per_channel && !block_128)) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (FP8): scale must be bf16 [N/128,K/128] or [N,1].");
  }
  if (x_shape.element_type() != BF16 || out_shape.element_type() != BF16) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (FP8): x and out must be bf16.");
  }
  if (out_shape.dimensions(0) != b || out_shape.dimensions(1) != n) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (FP8): out must be [B, N].");
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice x,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice w,
                      GetAllocationSliceForHlo(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scale,
                      GetAllocationSliceForHlo(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice out,
      GetAllocationSliceForHlo(instr, is_tuple ? ShapeIndex{0} : ShapeIndex{}));

  auto thunk = std::make_unique<MetalFp8GemvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      x, x_shape, w, w_shape, scale, scale_shape, out, out_shape, b, k, n,
      per_channel);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMoeGemvThunk(
    const HloCustomCallInstruction* instr) {
  // Two flavors share one thunk:
  //   nvfp4: {x, w_f4, scale, expert_id, w_global_scale?}
  //   bf16:  {x, w, expert_id}
  const bool is_nvfp4 =
      instr->custom_call_target() == kMetalMoeGemmF4CallTarget;
  const bool has_scale = is_nvfp4;
  const int64_t expected_operands = has_scale ? 4 : 3;
  // nvfp4 only: an optional trailing f32[E] per-expert global scale, folded
  // into the weight group scale by the kernels.
  const bool has_global_scale =
      is_nvfp4 && instr->operand_count() == expected_operands + 1;
  if (instr->operand_count() != expected_operands && !has_global_scale) {
    return absl::InvalidArgumentError(absl::StrCat(
        "metal MoE GEMV expects ", expected_operands,
        is_nvfp4 ? " or 5 operands." : " operands."));
  }
  const Shape& x_shape = instr->operand(0)->shape();
  const Shape& w_shape = instr->operand(1)->shape();
  const int expert_id_idx = has_scale ? 3 : 2;
  const Shape& expert_id_shape = instr->operand(expert_id_idx)->shape();

  // MetalWorkspaceRewriter gives the sorted-prefill path a tuple(result,
  // s8[bytes]) result; the per-row path keeps its plain array result.
  const bool is_tuple = instr->shape().IsTuple();
  if (is_tuple && instr->shape().tuple_shapes().size() != 2) {
    return absl::InvalidArgumentError(
        "metal MoE GEMV must return either a bf16 result or "
        "tuple(bf16 result, s8 workspace).");
  }
  const Shape& out_shape =
      is_tuple ? instr->shape().tuple_shapes(0) : instr->shape();
  Shape workspace_shape;
  if (is_tuple) {
    workspace_shape = instr->shape().tuple_shapes(1);
    if (!workspace_shape.IsArray() || workspace_shape.element_type() != S8 ||
        workspace_shape.dimensions().size() != 1) {
      return absl::InvalidArgumentError(
          "metal MoE GEMV: workspace must be rank-1 s8.");
    }
  }

  if (x_shape.dimensions().size() != 2 || w_shape.dimensions().size() != 3 ||
      expert_id_shape.dimensions().size() != 1 ||
      out_shape.dimensions().size() != 2) {
    return absl::UnimplementedError(
        "metal MoE GEMV: unexpected operand ranks.");
  }
  auto is_row_major = [](const Shape& shape) {
    return shape.IsArray() && shape.has_layout() &&
           LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
  };
  if (!is_row_major(x_shape) || !is_row_major(w_shape) ||
      !is_row_major(expert_id_shape) || !is_row_major(out_shape) ||
      (has_scale && !is_row_major(instr->operand(2)->shape()))) {
    return absl::InvalidArgumentError(
        "metal MoE GEMV requires row-major contiguous x, w, scale, "
        "expert_id, and output buffers.");
  }

  const int64_t r = x_shape.dimensions(0);
  const int64_t k = x_shape.dimensions(1);
  const int64_t e = w_shape.dimensions(0);
  const int64_t n = w_shape.dimensions(1);

  if (r == 0 || k == 0 || n == 0 || e == 0) {
    return absl::UnimplementedError(
        "metal MoE GEMV: invalid dimension (must be > 0).");
  }
  constexpr int64_t kMaxKernelDim = std::numeric_limits<int32_t>::max();
  if (r > kMaxKernelDim || k > kMaxKernelDim || n > kMaxKernelDim ||
      e > kMaxKernelDim) {
    return absl::InvalidArgumentError(
        "metal MoE GEMV: R, E, K, and N must fit signed int32.");
  }
  if (w_shape.dimensions(2) != k || out_shape.dimensions(0) != r ||
      out_shape.dimensions(1) != n || expert_id_shape.dimensions(0) != r) {
    return absl::UnimplementedError(
        "metal MoE GEMV: inconsistent x/w/expert_id/out shapes.");
  }
  // bf16 uses four-wide vector loads. NVFP4 only requires group-16 K: its
  // QMV/Steel kernels and sorted row copies all have scalar-safe N tails.
  if (is_nvfp4 && k % 16 != 0) {
    return absl::UnimplementedError(
        "metal MoE GEMV: K must be multiple of 16 "
        "(nvfp4 group-16; N supports scalar tails).");
  }
  if (!is_nvfp4 && (k % 4 != 0 || n % 4 != 0)) {
    return absl::UnimplementedError(
        "metal MoE GEMV: K and N must be multiples of 4 "
        "(bf16 vectorized load).");
  }
  const PrimitiveType expected_w = is_nvfp4 ? F4E2M1FN : BF16;
  if (w_shape.element_type() != expected_w) {
    return absl::UnimplementedError(absl::StrCat(
        "metal MoE GEMV: w must be ", is_nvfp4 ? "f4e2m1" : "bf16", "."));
  }
  if (x_shape.element_type() != BF16 || out_shape.element_type() != BF16) {
    return absl::UnimplementedError(
        "metal MoE GEMV: x and out must be bf16.");
  }
  if (expert_id_shape.element_type() != S32) {
    return absl::UnimplementedError("metal MoE GEMV: expert_id must be s32.");
  }
  TF_ASSIGN_OR_RETURN(const int64_t expected_workspace_bytes,
                      GetMetalMoeWorkspaceBytes(r, e, k, n, is_nvfp4));
  const int64_t workspace_bytes =
      is_tuple ? workspace_shape.dimensions(0) : 0;
  if (workspace_bytes != expected_workspace_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "metal MoE GEMV workspace must be s8[", expected_workspace_bytes,
        "]; got s8[", workspace_bytes, "]."));
  }

  // Scale operand: nvfp4 = [E, N, K/16] f8e4m3fn.
  BufferAllocation::Slice scale;
  Shape scale_shape;
  if (is_nvfp4) {
    scale_shape = instr->operand(2)->shape();
    if (scale_shape.dimensions().size() != 3 ||
        scale_shape.dimensions(0) != e || scale_shape.dimensions(1) != n ||
        scale_shape.dimensions(2) != k / 16) {
      return absl::UnimplementedError(
          "metal MoE GEMV: nvfp4 scale must be [E, N, K/16].");
    }
    if (scale_shape.element_type() != F8E4M3FN) {
      return absl::UnimplementedError(
          "metal MoE GEMV: nvfp4 scale must be f8e4m3fn.");
    }
    TF_ASSIGN_OR_RETURN(scale,
                        GetAllocationSliceForHlo(instr->operand(2), {}));
  }

  // Optional nvfp4 per-expert global scale (compressed-tensors g_ct): f32[E].
  BufferAllocation::Slice global_scale;
  Shape global_scale_shape;
  if (has_global_scale) {
    global_scale_shape = instr->operand(4)->shape();
    if (global_scale_shape.dimensions().size() != 1 ||
        global_scale_shape.dimensions(0) != e) {
      return absl::UnimplementedError(absl::StrCat(
          "metal MoE GEMV: nvfp4 global scale must be [", e, "]; got ",
          global_scale_shape.ToString(), "."));
    }
    if (global_scale_shape.element_type() != F32) {
      return absl::UnimplementedError(
          "metal MoE GEMV: nvfp4 global scale must be f32.");
    }
    TF_ASSIGN_OR_RETURN(global_scale,
                        GetAllocationSliceForHlo(instr->operand(4), {}));
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice x,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice w,
                      GetAllocationSliceForHlo(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice expert_id,
      GetAllocationSliceForHlo(instr->operand(expert_id_idx), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice out,
      GetAllocationSliceForHlo(instr, is_tuple ? ShapeIndex{0} : ShapeIndex{}));
  BufferAllocation::Slice workspace;
  if (is_tuple) {
    TF_ASSIGN_OR_RETURN(workspace,
                        GetAllocationSliceForHlo(instr, ShapeIndex{1}));
  }

  auto thunk = std::make_unique<MetalMoeGemvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      x, x_shape, w, w_shape, scale, scale_shape, expert_id, expert_id_shape,
      out, out_shape, workspace, workspace_shape, global_scale,
      global_scale_shape, has_global_scale, r, k, n);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalKvWriteThunk(
    const HloCustomCallInstruction* instr) {
  // Operand contract fixed by RewriteKvCacheWrites (metal_gpu_compiler.cc):
  //   k_cache [P,B,H,D] bf16, k_new [H*D] bf16, v_cache [P,B,H,D] bf16,
  //   v_new [H*D] bf16, slot s32[1], pos s32[1], freq f32[1,D/2].
  // Output tuple(k_cache', v_cache') aliases operands 0 and 2.
  TF_RET_CHECK(instr->operand_count() == 7);
  const Shape& k_cache_shape = instr->operand(0)->shape();
  TF_RET_CHECK(k_cache_shape.dimensions().size() == 4);
  TF_RET_CHECK(k_cache_shape.element_type() == BF16);
  const int64_t num_slots =
      k_cache_shape.dimensions(0) * k_cache_shape.dimensions(1);
  const int64_t kv_heads = k_cache_shape.dimensions(2);
  const int64_t head_dim = k_cache_shape.dimensions(3);

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k_cache,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k_new,
                      GetAllocationSliceForHlo(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v_cache,
                      GetAllocationSliceForHlo(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v_new,
                      GetAllocationSliceForHlo(instr->operand(3), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slot,
                      GetAllocationSliceForHlo(instr->operand(4), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice pos,
                      GetAllocationSliceForHlo(instr->operand(5), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice freq,
                      GetAllocationSliceForHlo(instr->operand(6), {}));

  auto thunk = std::make_unique<MetalKvWriteThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      k_cache, instr->operand(0)->shape(), k_new, instr->operand(1)->shape(),
      v_cache, instr->operand(2)->shape(), v_new, instr->operand(3)->shape(),
      slot, instr->operand(4)->shape(), pos, instr->operand(5)->shape(), freq,
      instr->operand(6)->shape(), num_slots, kv_heads, head_dim);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalSortThunk(
    const HloCustomCallInstruction* instr) {
  // ABI (RewriteSortToMetalThunk): operand 0 = values[..., n] (bf16/f16/f32),
  // sorted along the minor-most axis; result = tuple(sorted_values[..., n],
  // sorted_indices[..., n] s32). opaque = "desc" | "asc".
  TF_RET_CHECK(instr->operand_count() == 1);
  TF_RET_CHECK(instr->shape().IsTuple() &&
               instr->shape().tuple_shapes().size() == 2);
  const Shape& vshape = instr->operand(0)->shape();
  const PrimitiveType dtype = vshape.element_type();
  TF_RET_CHECK(dtype == BF16 || dtype == F16 || dtype == F32 || dtype == S32 ||
               dtype == S16 || dtype == S8 || dtype == U32 || dtype == U16 ||
               dtype == U8);
  const int64_t rank = vshape.dimensions().size();
  TF_RET_CHECK(rank >= 1);
  const int64_t n = vshape.dimensions(rank - 1);
  int64_t rows = 1;
  for (int64_t i = 0; i + 1 < rank; ++i) rows *= vshape.dimensions(i);
  const bool descending = instr->opaque() == "desc";

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out_vals,
                      GetAllocationSliceForHlo(instr, {0}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out_idxs,
                      GetAllocationSliceForHlo(instr, {1}));

  auto thunk = std::make_unique<MetalSortThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      data, out_vals, out_idxs, dtype, rows, n, descending);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulThunkF8(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() > 3 && instr->operand_count() < 8);
  ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  ASSIGN_OR_RETURN(bool has_aux_output,
                   xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));

  ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ShapedSlice c;
  bool has_matrix_bias = config.beta() != 0;
  if (has_matrix_bias) {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(2)));
  } else {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  int a_scale_index = has_matrix_bias ? 3 : 2;
  ASSIGN_OR_RETURN(ShapedSlice a_scale,
                   GetShapedSliceForHlo(instr->operand(a_scale_index)));
  ASSIGN_OR_RETURN(ShapedSlice b_scale,
                   GetShapedSliceForHlo(instr->operand(a_scale_index + 1)));

  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();
  bool is_fp8 = instr->shape().tuple_shapes(0).element_type() == F8E4M3FN ||
                instr->shape().tuple_shapes(0).element_type() == F8E5M2;
  // cublasLT requires c_scale/d_scale to be null when C/D is not
  // FP8. Currently, C cannot be FP8.
  std::optional<ShapedSlice> d_scale;
  if (is_cuda && is_fp8) {
    ASSIGN_OR_RETURN(d_scale, GetShapedSliceForHlo(instr->operands().back()));
  }

  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    ASSIGN_OR_RETURN(bias,
                     GetShapedSliceForHlo(instr->operand(a_scale_index + 2)));
  }

  std::optional<ShapedSlice> d_amax;
  if (config.damax_output()) {
    ASSIGN_OR_RETURN(d_amax, GetShapedSliceForHlo(instr, {1}));
  }

  ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  TF_RET_CHECK(!has_aux_output);
  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().tuple_shapes().size() - config.damax_output() == 2) {
    ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt, bias, std::nullopt, a_scale, b_scale,
      std::nullopt, d_scale, d_amax, workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtGroupedMatmulThunk(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config =
      gpu_config.grouped_gemm_backend_config().gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  // Matrix bias and vector bias add extra operands
  bool has_matrix_bias = config.beta() != 0;
  ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));
  TF_RET_CHECK(instr->operand_count() ==
               3 + int{has_matrix_bias} + int{has_vector_bias});

  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ASSIGN_OR_RETURN(ShapedSlice group_sizes,
                   GetShapedSliceForHlo(instr->operand(2)));

  // Handle matrix bias if present
  ShapedSlice c;
  if (has_matrix_bias) {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(3)));
  } else {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  // Handle vector bias if present
  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    int bias_operand_index = has_matrix_bias ? 4 : 3;
    ASSIGN_OR_RETURN(bias,
                     GetShapedSliceForHlo(instr->operand(bias_operand_index)));
  }

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().IsTuple() && (instr->shape().tuple_shapes().size() - 1)) {
    ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }
  ASSIGN_OR_RETURN(
      auto gemm_config,
      GroupedGemmConfig::For(static_cast<const HloInstruction*>(instr),
                             ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  // Extract epilogue from backend config instead of hardcoding to kDefault
  ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));

  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      std::move(group_sizes), bias, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulThunkMx(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 4);
  ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ASSIGN_OR_RETURN(ShapedSlice a_scale,
                   GetShapedSliceForHlo(instr->operand(2)));
  ASSIGN_OR_RETURN(ShapedSlice b_scale,
                   GetShapedSliceForHlo(instr->operand(3)));

  ASSIGN_OR_RETURN(ShapedSlice c, GetShapedSliceForHlo(instr, output_index));
  ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().tuple_shapes().size() == 2) {
    ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt,
      /*bias=*/std::nullopt, /*aux=*/std::nullopt, a_scale, b_scale,
      /*c_scale=*/std::nullopt, /*d_scale=*/std::nullopt,
      /*d_amax=*/std::nullopt, workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolutionReorderThunk(
    const HloCustomCallInstruction* instr) {
  bool has_bias = instr->operand_count() > 1;

  ASSIGN_OR_RETURN(ShapedSlice filter_input,
                   GetShapedSliceForHlo(instr->operand(0)));

  ShapedSlice filter_output;
  std::optional<ConvolutionReorderThunk::BiasBuffers> biases;
  if (has_bias) {
    ASSIGN_OR_RETURN(filter_output, GetShapedSliceForHlo(instr, {0}));

    ASSIGN_OR_RETURN(ShapedSlice bias_input,
                     GetShapedSliceForHlo(instr->operand(1)));
    ASSIGN_OR_RETURN(ShapedSlice bias_output, GetShapedSliceForHlo(instr, {1}));
    biases = {{bias_input, bias_output}};
  } else {
    ASSIGN_OR_RETURN(filter_output, GetShapedSliceForHlo(instr));
  }

  ASSIGN_OR_RETURN(auto thunk,
                   ConvolutionReorderThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           instr, ir_emitter_context_->GetNextThunkId()),
                       filter_input, filter_output, biases));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNormThunk(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(auto const gpu_backend_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::CudnnNormBackendConfig& backend_config =
      gpu_backend_config.cudnn_norm_backend_config();

  ASSIGN_OR_RETURN(BufferAllocation::Slice x_slice,
                   GetAllocationSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(BufferAllocation::Slice scale_slice,
                   GetAllocationSliceForHlo(instr->operand(1)));
  ASSIGN_OR_RETURN(BufferAllocation::Slice y_or_dx_slice,
                   GetAllocationSliceForHlo(instr, {0}));

  std::optional<BufferAllocation::Slice> bias_slice, expectation_slice,
      norm_factor_slice, dy_slice, dscale_slice, dbias_slice;

  if (backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER ||
      backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    ASSIGN_OR_RETURN(bias_slice, GetAllocationSliceForHlo(instr->operand(2)));
  }
  if (backend_config.kind() ==
      xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    ASSIGN_OR_RETURN(expectation_slice, GetAllocationSliceForHlo(instr, {1}));
    ASSIGN_OR_RETURN(norm_factor_slice, GetAllocationSliceForHlo(instr, {2}));
  }
  if (backend_config.kind() == xla::gpu::CudnnNormBackendConfig::LAYER_BWD) {
    ASSIGN_OR_RETURN(dy_slice, GetAllocationSliceForHlo(instr->operand(2)));
    ASSIGN_OR_RETURN(expectation_slice,
                     GetAllocationSliceForHlo(instr->operand(3)));
    ASSIGN_OR_RETURN(norm_factor_slice,
                     GetAllocationSliceForHlo(instr->operand(4)));
    ASSIGN_OR_RETURN(dscale_slice, GetAllocationSliceForHlo(instr, {1}));
    ASSIGN_OR_RETURN(dbias_slice, GetAllocationSliceForHlo(instr, {2}));
  }
  ASSIGN_OR_RETURN(
      ShapedSlice scratch_slice,
      GetShapedSliceForHlo(
          instr,
          {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));

  GpuNormDescriptor descriptor;
  descriptor.backend_config = backend_config;

  descriptor.x_shape = instr->operand(0)->shape();
  descriptor.scale_shape = instr->operand(1)->shape();
  descriptor.y_or_dx_shape = ShapeUtil::GetSubshape(instr->shape(), {0});
  descriptor.scratch_shape = scratch_slice.shape;

  if (backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER ||
      backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    descriptor.bias_shape = instr->operand(2)->shape();
  }
  if (backend_config.kind() ==
      xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    descriptor.expectation_shape = ShapeUtil::GetSubshape(instr->shape(), {1});
    descriptor.norm_factor_shape = ShapeUtil::GetSubshape(instr->shape(), {2});
  }
  if (backend_config.kind() == xla::gpu::CudnnNormBackendConfig::LAYER_BWD) {
    descriptor.dy_shape = instr->operand(2)->shape();
    descriptor.expectation_shape = instr->operand(3)->shape();
    descriptor.norm_factor_shape = instr->operand(4)->shape();
    descriptor.dscale_shape = ShapeUtil::GetSubshape(instr->shape(), {1});
    descriptor.dbias_shape = ShapeUtil::GetSubshape(instr->shape(), {2});
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<NormThunk> thunk,
      NormThunk::Create(Thunk::ThunkInfo::WithProfileAnnotation(
                            instr, ir_emitter_context_->GetNextThunkId()),
                        std::move(descriptor), x_slice, scale_slice,
                        y_or_dx_slice, bias_slice, expectation_slice,
                        norm_factor_slice, dy_slice, dscale_slice, dbias_slice,
                        scratch_slice.slice));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCuDnnThunk(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));
  ASSIGN_OR_RETURN(const std::string fingerprint,
                   FingerprintWithBackendConfig<GpuBackendConfig>(*instr));
  // check if sdpa dropout is enabled
  std::optional<int64_t> dropout_seed = std::nullopt;
  if (MHACallHasDropout(instr->custom_call_target())) {
    ASSIGN_OR_RETURN(const auto gpu_config,
                     instr->backend_config<xla::gpu::GpuBackendConfig>());
    dropout_seed = gpu_config.cudnn_fmha_backend_config().seed();
  }
  return GetThunkSequence(std::make_unique<CuDnnThunk>(
      fingerprint,
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      kernel_arguments.GetArgumentShapedSlices(),
      kernel_arguments.GetArgumentOutputFlags(),
      /*should_memzero=*/IsCustomCallTofMHA(*instr), dropout_seed));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitPtxCustomCall(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(auto thunk,
                   EmitPtxCustomKernelThunk(instr, ir_emitter_context_));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSliceForHlo(
    const HloInstruction* instr, const ShapeIndex& index) const {
  if (!allocation_overrides_.empty()) {
    auto it = allocation_overrides_.find(instr);
    if (it != allocation_overrides_.end()) {
      int64_t flat_idx = index.empty() ? 0 : index[0];
      if (flat_idx < it->second.size()) {
        return it->second[flat_idx];
      }
    }
  }
  return ir_emitter_context_->buffer_assignment().GetUniqueSlice(instr, index);
}

absl::StatusOr<ShapedSlice> ThunkEmitter::GetShapedSliceForHlo(
    const HloInstruction* instr, const ShapeIndex& index) const {
  ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSliceForHlo(instr, index));
  ASSIGN_OR_RETURN(
      Shape shape,
      ir_emitter_context_->buffer_assignment().GetShapeForUniqueSlice(instr,
                                                                      index));
  return ShapedSlice{slice, shape};
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCustomCallThunk(
    const HloCustomCallInstruction* instr) {
  const std::string& call_target_name = instr->custom_call_target();

  // Typed FFI custom calls is a replacement for legacy custom calls
  // with a rich type safe API.
  bool is_ffi_custom_call =
      instr->api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  using Slices = std::vector<NullableShapedSlice>;

  Slices operands;
  for (auto* operand : instr->operands()) {
    RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            operands.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          ASSIGN_OR_RETURN(auto slice,
                           GetAllocationSliceForHlo(operand, index));
          operands.push_back(ShapedSlice{slice, subshape});
          return absl::OkStatus();
        }));
  }

  Slices results;
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          results.push_back(std::nullopt);
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        ASSIGN_OR_RETURN(auto slice, GetAllocationSliceForHlo(instr, index));
        results.push_back(ShapedSlice{slice, subshape});
        return absl::OkStatus();
      }));

  // For XLA FFI handlers we decode opaque backend config into
  // attributes map at IR emission time, so that we do not need to
  // parse MLIR at run time. For FFI handlers backend config must be
  // a compatible MLIR dictionary.
  ffi::AttributesMap attributes;

  auto backend_config = instr->backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    VLOG(3) << "Unable to parse backend config for custom call: "
            << backend_config.status().message() << "\n"
            << "Fall back to parse the raw backend config str.";
  }

  auto ffi_thunk = [&]() -> absl::StatusOr<std::unique_ptr<Thunk>> {
    auto& called_computations = instr->called_computations();
    auto& backend_config_str =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().attributes()
            : instr->raw_backend_config_string();
    if (!backend_config_str.empty()) {
      mlir::Attribute attr = mlir::parseAttribute(
          backend_config_str, ir_emitter_context_->mlir_context());
      auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
      if (dict == nullptr) {
        return absl::InternalError(
            "Unsupported backend config. Expected a string "
            "parsable into "
            "dictionary attribute");
      }
      ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    }
    auto released_lock_keeper = llvm_options_lock_->TemporarilyReleaseLock();
    return CustomCallThunk::Create(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        call_target_name, std::move(operands), std::move(results),
        std::move(attributes),
        called_computations.empty() ? nullptr : called_computations[0],
        ir_emitter_context_->platform_name(),
        ir_emitter_context_->gpu_compute_capability(),
        /*execution_state=*/nullptr,
        ir_emitter_context_->cpu_target_machine_options());
  };

  auto legacy_thunk = [&]() -> absl::StatusOr<std::unique_ptr<Thunk>> {
    std::string opaque =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().opaque()
            : instr->raw_backend_config_string();
    return LegacyCustomCallThunk::Create(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        call_target_name, std::move(operands), std::move(results),
        std::move(opaque), instr->api_version(),
        ir_emitter_context_->platform_name());
  };

  absl::StatusOr<std::unique_ptr<Thunk>> custom_call_thunk =
      is_ffi_custom_call ? ffi_thunk() : legacy_thunk();

  ThunkSequence thunks;
  if (custom_call_thunk.ok()) {
    thunks.push_back(std::move(custom_call_thunk.value()));
  }
  if (ir_emitter_context_->debug_options().xla_gpu_mock_custom_calls()) {
    // xla_gpu_mock_custom_calls=true means we won't emit thunks for all custom
    // call targets that couldn't be found.
    return thunks;
  }
  if (!custom_call_thunk.ok()) {
    return custom_call_thunk.status();
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFftThunk(
    const HloFftInstruction* instr) {
  ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                   GetAllocationSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                   GetAllocationSliceForHlo(instr));
  return GetThunkSequence(std::make_unique<FftThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr->fft_type(), instr->fft_length(),
      /*input_buffer=*/arg_slice,
      /*output_buffer=*/dest_slice,
      /*input_shape=*/instr->operand(0)->shape(),
      /*output_shape=*/instr->shape()));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitTriangularSolveCustomCall(
    const HloInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 2);
  auto operands = instr->operands();
  TF_RET_CHECK(instr->shape().IsTuple() &&
               instr->shape().tuple_shapes().size() == 2);

  // We expect Fortran layout for everything other than the temp
  // buffer (the last operand).  Fortran layout is not XLA default
  // layout with elements 0 and 1 swapped.  For example instead of
  // default layout {3,2,1,0} we'd have Fortran layout {2,3,1,0}.
  auto has_fortran_layout = [](const Layout& layout) {
    int n = layout.minor_to_major().size();
    return layout.minor_to_major(0) == n - 2 &&
           layout.minor_to_major(1) == n - 1;
  };
  TF_RET_CHECK(has_fortran_layout(operands[0]->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(operands[1]->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(instr->shape().tuple_shapes(0).layout()));

  ASSIGN_OR_RETURN(ShapedSlice a_slice, GetShapedSliceForHlo(operands[0]));
  ASSIGN_OR_RETURN(ShapedSlice b_slice, GetShapedSliceForHlo(operands[1]));
  ASSIGN_OR_RETURN(ShapedSlice result_slice, GetShapedSliceForHlo(instr, {0}));
  ASSIGN_OR_RETURN(ShapedSlice temp_slice, GetShapedSliceForHlo(instr, {1}));

  TriangularSolveOptions backend_config;
  auto& backend_config_str = instr->raw_backend_config_string();
  if (!backend_config_str.empty()) {
    RETURN_IF_ERROR(
        tsl::HumanReadableJsonToProto(backend_config_str, &backend_config));
  }

  ThunkSequence thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output
  // if they aren't the same buffer.
  if (b_slice.slice != result_slice.slice) {
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        /*source_buffer=*/b_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(b_slice.shape)));
  }

  thunks.push_back(std::make_unique<TriangularSolveThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      backend_config, a_slice, result_slice, temp_slice));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    return thunks;
  }
  auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  // Don't repeat the annotation from inside thunks
  thunk_info.profile_annotation = {};
  return GetThunkSequence(
      std::make_unique<SequentialThunk>(thunk_info, std::move(thunks)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitTopKCustomCall(
    const HloCustomCallInstruction* instr) {
  auto operands = instr->operands();
  const auto& shape = instr->shape();
  TF_RET_CHECK(operands.size() == 1)
      << "Expect only 1 operand for TopK custom call.";
  TF_RET_CHECK(shape.IsTuple())
      << "Expect TopK custom call to have tuple shape.";
  TF_RET_CHECK(shape.tuple_shapes().size() == 2)
      << "Expect TopK custom call shape to have exactly 2 "
         "sub-shapes.";

  auto data_shape = operands[0]->shape();
  auto top_elements_shape = shape.tuple_shapes()[0];
  auto indices_shape = shape.tuple_shapes()[1];

  TF_RET_CHECK(data_shape.dimensions().size() <= 2) << "Invalid input shape.";
  TF_RET_CHECK(indices_shape.element_type() == PrimitiveType::S32)
      << "Indices should be S32.";

  bool has_batch = data_shape.dimensions().size() == 2;
  auto [batch_size, n, k] =
      has_batch
          ? std::tuple<size_t, size_t, size_t>{data_shape.dimensions(0),
                                               data_shape.dimensions(1),
                                               top_elements_shape.dimensions(1)}
          : std::tuple<size_t, size_t, size_t>{
                1, data_shape.dimensions(0), top_elements_shape.dimensions(0)};

  // Prepare kernel arguments.
  ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  auto dtype = data_shape.element_type();
  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();
  if (is_cuda && instr->GetModule()
                     ->config()
                     .debug_options()
                     .xla_gpu_experimental_use_raft_select_k()) {
    // The heuristic for deciding when to use TopK Custom Kernel versus
    // Raft::matrix::select_k was developed as part of the initial research
    // in b/409009349.
    // CustomCall TopK requires k <= 16 and n >= 1024
    bool use_raft_select_k = false;
    if (dtype == PrimitiveType::F32) {
      use_raft_select_k =
          (n < 1024) || (n == 1024 && k > 12) || (n > 1024 && k >= 8);
    } else if (dtype == PrimitiveType::BF16) {
      use_raft_select_k = n < 1024 || k >= 8;
    }

    VLOG(3) << "EmitTopKCustomCall: dtype=" << dtype << ", n=" << n
            << ", k=" << k << ", use_raft_select_k=" << use_raft_select_k;

    Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
        instr, ir_emitter_context_->GetNextThunkId());
    if (use_raft_select_k) {
      return GetThunkSequence(std::make_unique<SelectKThunk>(
          std::move(thunk_info), batch_size, n, k, dtype, kernel_arguments));
    }
  }

  // The GPU bucket/radix-select MetalTopKThunk is the SOLE Metal TopK path — the
  // shared single-pass kernel (one threadgroup over the whole vocab) is gone. It
  // handles everything the TopkSpecializer emits (k<=64, n>=1, bf16/f16/f32, any
  // batch): histogram the top bits of the sortable key -> threshold -> gather ->
  // exact select on the full key, identical Descending top-K, k-independent. Fail
  // LOUDLY on anything outside that instead of silently degrading. (The shared
  // GetTopKKernel path below is reached on NON-Metal backends only.)
  if (platform_name() == "METAL") {
    TF_RET_CHECK(k <= 64)
        << "Metal TopK: k=" << k << " > 64 — the radix select kernels template k in "
           "{1,2,4,8,16,32,64}; the TopkSpecializer should have capped k at 64.";
    TF_RET_CHECK(dtype == BF16 || dtype == F16 || dtype == F32)
        << "Metal TopK: dtype " << PrimitiveType_Name(dtype)
        << " unsupported — the radix sortable-bit transform handles bf16/f16/f32.";
    const auto& args = kernel_arguments.args();
    TF_RET_CHECK(args.size() == 3)
        << "Metal TopK expects 3 buffers (data, vals, idxs).";
    Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
        instr, ir_emitter_context_->GetNextThunkId());
    return GetThunkSequence(std::make_unique<MetalTopKThunk>(
        std::move(thunk_info), args[0].slice(), args[1].slice(),
        args[2].slice(), dtype, batch_size, n, k));
  }

  auto wavefront_size =
      ir_emitter_context_->gpu_device_info().threads_per_warp();

  TF_RET_CHECK(k <= 16) << "CustomCall TopK requires k <= 16";
  // Load TopK custom kernel.
  ASSIGN_OR_RETURN(CustomKernel kernel, kernel::topk::GetTopKKernel(
                                            "topk", dtype, n, k, batch_size,
                                            platform_name(), wavefront_size));

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return GetThunkSequence(std::make_unique<CustomKernelThunk>(
      std::move(thunk_info), std::move(kernel), kernel_arguments));
}

AsyncThunkSequence ThunkEmitter::EmitTritonCustomCall(
    const HloCustomCallInstruction* instr) {
  BorrowedMlirContext borrowed_context =
      ir_emitter_context_->BorrowMlirContext();
  LoadMlirDialectsForTriton(**borrowed_context);
  TritonCall call = TritonCall::Parse(instr->raw_backend_config_string(),
                                      borrowed_context->get());
  auto call_zeroed_outputs = call.zeroed_outputs;
  auto generate =
      [this, &instr, borrowed_context = std::move(borrowed_context),
       call =
           std::move(call)]() mutable -> xla::Future<KernelReuseCache::Entry> {
    std::string kernel_name =
        ir_emitter_context_->GetSanitizedUniqueName(call.name);

    ASSIGN_OR_RETURN(TritonKernelSource triton_source,
                     EmitTritonFrom(call, kernel_name, **borrowed_context));

    HloModule* hlo_module = instr->GetModule();

    BlockLevelParameters block_level_parameters;
    block_level_parameters.num_stages = call.num_stages;
    block_level_parameters.num_warps = call.num_warps;
    block_level_parameters.num_ctas = 1;
    block_level_parameters.global_scratch_memory_size =
        call.global_scratch_memory_size;
    block_level_parameters.is_tma_allowed = call.is_tma_allowed;

    return ir_emitter_context_->kernel_compiler()
        ->CompileTritonToLlvm(
            kernel_name, *hlo_module, ir_emitter_context_->gpu_device_info(),
            block_level_parameters, ir_emitter_context_->target_triple(),
            ir_emitter_context_->data_layout(), std::move(triton_source),
            std::move(borrowed_context), /*is_xla_fusion=*/false)
        .Map([kernel_name,
              kernel_impl_name = ir_emitter_context_->GetSanitizedUniqueName(
                  kernel_name + "_impl"),
              instr, call = std::move(call),
              kernel_compiler = ir_emitter_context_->kernel_compiler(),
              buffer_assignment = &ir_emitter_context_->buffer_assignment(),
              gpu_device_info = ir_emitter_context_->gpu_device_info()](
                 TritonWrapperResult result)
                 -> xla::Future<KernelReuseCache::Entry> {
          auto local_module =
              std::move(result.kernel_source).thread_safe_module();

          ASSIGN_OR_RETURN(
              auto kernel_arguments,
              emitters::KernelArguments::Create(
                  *buffer_assignment, GetDefaultBufferAlignment(), instr));
          auto launch_dimensions = LaunchDimensions(
              se::BlockDim(call.grid_x, call.grid_y, call.grid_z),
              se::ThreadDim(call.num_warps *
                            gpu_device_info.threads_per_warp()));

          ASSIGN_OR_RETURN(
              llvm::Function * kernel,
              RemoveUnusedTritonAbiArguments(
                  local_module.getModuleUnlocked(), kernel_name,
                  kernel_impl_name, call.global_scratch_memory_size > 0));

          AnnotateAttrsIfUnset(kernel_arguments, *kernel);
          RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
              gpu_device_info, launch_dimensions, kernel,
              local_module.getModuleUnlocked()));

          return kernel_compiler
              ->CompileToTargetBinary(LlvmKernelSource{std::move(local_module)})
              .Map([use_pdl = result.use_pdl, shmem_bytes = result.shmem_bytes,
                    launch_dimensions = std::move(launch_dimensions),
                    tma_metadata = result.tma_metadata,
                    kernel_name](const std::vector<uint8_t>& cubin) {
                return KernelReuseCache::Entry{kernel_name,
                                               launch_dimensions,
                                               /*cluster_dim=*/std::nullopt,
                                               shmem_bytes,
                                               cubin,
                                               tma_metadata,
                                               use_pdl};
              });
        });
  };

  ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  auto [status_or_entry, was_cached] =
      ir_emitter_context_->kernel_cache().GetWithStatus(
          instr->raw_backend_config_string(), generate);

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return status_or_entry.Map(
      [thunk_info = std::move(thunk_info),
       kernel_arguments = std::move(kernel_arguments),
       call_zeroed_outputs =
           std::move(call_zeroed_outputs)](const KernelReuseCache::Entry* entry)
          -> absl::StatusOr<ThunkSequence> {
        ASSIGN_OR_RETURN(CustomKernel custom_kernel,
                         kernel::CreateOwnedCubinCustomKernel(
                             entry->kernel_name, entry->binary,
                             kernel_arguments.args().size(),
                             entry->launch_dimensions.block_counts(),
                             entry->launch_dimensions.thread_counts_per_block(),
                             entry->shmem_bytes));
        return ThunkSequence::Of(std::make_unique<CustomKernelThunk>(
            thunk_info, std::move(custom_kernel), kernel_arguments,
            entry->use_pdl, call_zeroed_outputs, entry->tma_metadata));
      });
}

AsyncThunkSequence ThunkEmitter::EmitAsyncComputation(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  TF_RET_CHECK(wrapped->called_computations().size() == 1);
  HloComputation* computation = wrapped->called_computations().front();

  auto* async_start = Cast<HloAsyncInstruction>(instr);
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                   stream_assignment.GetExecutionStreamId(async_start));

  AsyncThunkSequence nested_thunks = EmitHloComputation(computation);

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  std::shared_ptr<AsyncExecution> async_execution =
      std::make_shared<AsyncExecution>(thunk_info);
  auto [it, inserted] = hlo_async_executions_.emplace(wrapped, async_execution);
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    wrapped->ToString());
  }

  return std::move(nested_thunks)
      .Map([thunk_info = std::move(thunk_info),
            async_execution = std::move(async_execution),
            execution_stream_id](ThunkSequence nested_thunks) {
        return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
            std::move(thunk_info), execution_stream_id,
            std::move(nested_thunks), async_execution));
      });
}

AsyncThunkSequence ThunkEmitter::EmitDynamicSliceCopyFusion(
    const HloFusionInstruction* instr, DynamicSliceCopyFusion copy) {
  std::vector<BufferAllocation> embedded_allocations;
  embedded_allocations.reserve(copy.parameters.size() + copy.results.size());

  for (const auto& param : copy.parameters) {
    embedded_allocations.emplace_back(embedded_allocations.size(),
                                      ShapeUtil::ByteSizeOf(param.slice_shape),
                                      0);
  }

  for (const auto& res : copy.results) {
    embedded_allocations.emplace_back(embedded_allocations.size(),
                                      ShapeUtil::ByteSizeOf(res.update_shape),
                                      0);
  }

  TF_RET_CHECK(copy.parameters.size() == 1);
  TF_RET_CHECK(copy.results.size() == 1);

  const Shape& copy_shape = copy.copy_operand->shape();
  int64_t byte_size = ShapeUtil::ByteSizeOf(copy_shape);
  BufferAllocation::Slice src_slice(&embedded_allocations[0], 0, byte_size);
  BufferAllocation::Slice dst_slice(
      &embedded_allocations[copy.parameters.size()], 0, byte_size);

  ThunkSequence embedded_thunks =
      ThunkSequence::Of(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          ShapedSlice{src_slice, copy_shape},
          ShapedSlice{dst_slice, copy.results[0].update_shape}, byte_size));

  std::vector<BufferAllocation::Slice> parameter_buffers;
  parameter_buffers.reserve(instr->operand_count());
  for (const auto* operand : instr->operands()) {
    ASSIGN_OR_RETURN(parameter_buffers.emplace_back(),
                     GetAllocationSliceForHlo(operand));
  }

  std::vector<BufferAllocation::Slice> result_buffers;
  RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      instr->shape(),
      [&](const Shape&, const ShapeIndex& index) -> absl::Status {
        ASSIGN_OR_RETURN(result_buffers.emplace_back(),
                         GetAllocationSliceForHlo(instr, index));
        return absl::OkStatus();
      }));

  RETURN_IF_ERROR(DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
      copy.results, parameter_buffers, result_buffers));

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  bool verify_offsets =
      ir_emitter_context_->debug_options()
          .xla_gpu_experimental_dynamic_slice_fusion_verify_offsets();

  return ThunkSequence::Of(std::make_unique<DynamicSliceFusionV2Thunk>(
      std::move(thunk_info), std::move(copy.parameters),
      std::move(copy.results), std::move(parameter_buffers),
      std::move(result_buffers), std::move(embedded_allocations),
      std::move(embedded_thunks), verify_offsets));
}

AsyncThunkSequence ThunkEmitter::EmitStaticSliceCopyFusion(
    const HloFusionInstruction* instr, const StaticSliceCopyFusion& copy) {
  if (copy.parameter_number < 0 ||
      copy.parameter_number >= instr->operand_count()) {
    return Internal("Static slice copy parameter %d is out of range for %s",
                    copy.parameter_number, instr->ToString());
  }

  ASSIGN_OR_RETURN(
      BufferAllocation::Slice arg_slice,
      GetAllocationSliceForHlo(instr->operand(copy.parameter_number)));
  ASSIGN_OR_RETURN(BufferAllocation::Slice dst_slice,
                   GetAllocationSliceForHlo(instr));

  int64_t byte_size = ShapeUtil::ByteSizeOf(copy.slice_shape);
  BufferAllocation::Slice src_slice(
      arg_slice.allocation(), arg_slice.offset() + copy.source_byte_offset,
      byte_size, arg_slice.element_type());

  return GetThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      ShapedSlice{src_slice, copy.slice_shape},
      ShapedSlice{dst_slice, instr->shape()}, byte_size));
}

AsyncThunkSequence ThunkEmitter::EmitFusion(const HloFusionInstruction* instr) {
  ASSIGN_OR_RETURN(std::optional<StaticSliceCopyFusion> static_copy,
                   AnalyzeStaticSliceCopyFusion(instr));
  if (static_copy.has_value()) {
    return EmitStaticSliceCopyFusion(instr, *static_copy);
  }

  ASSIGN_OR_RETURN(std::optional<DynamicSliceCopyFusion> dynamic_copy,
                   AnalyzeDynamicSliceCopyFusion(instr));
  if (dynamic_copy.has_value()) {
    return EmitDynamicSliceCopyFusion(instr, std::move(*dynamic_copy));
  }

  analysis_garbage_collector_.push_back(
      std::make_unique<HloFusionAnalysis>(HloFusionAnalysis::Create(
          *instr, ir_emitter_context_->gpu_device_info())));
  const HloFusionAnalysis& fusion_analysis =
      *analysis_garbage_collector_.back();

  // Intercept DynamicSliceFusionV2 custom fusions.
  if (fusion_analysis.emitter_fusion_kind() ==
      HloFusionAnalysis::EmitterFusionKind::kCustomFusion) {
    auto custom_name = GetCustomFusionConfigName(instr);
    if (custom_name.has_value() &&
        *custom_name == kDynamicSliceFusionConfigName) {
      return EmitDynamicSliceFusionV2(instr);
    }
  }

  std::unique_ptr<FusionInterface> emitter = GetFusionEmitter(
      HloFusionInfo(fusion_analysis, instr,
                    &ir_emitter_context_->buffer_assignment(), *call_graph_));
  return emitter->Emit(*ir_emitter_context_, *instr);
}

AsyncThunkSequence ThunkEmitter::EmitDynamicSliceFusionV2(
    const HloFusionInstruction* instr) {
  const HloComputation* body = instr->fused_instructions_computation();

  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  if (hero == nullptr) {
    return Internal("DynamicSliceFusionV2: no hero operation found");
  }

  ASSIGN_OR_RETURN(std::vector<DynamicSliceFusion::Parameter> parameters,
                   DynamicSliceFusion::ResolveParameters(hero));
  ASSIGN_OR_RETURN(std::vector<DynamicSliceFusion::Result> results,
                   DynamicSliceFusion::ResolveResults(hero));

  // parameter_buffers: one slice per fusion operand, indexed by parameter
  // number.
  std::vector<BufferAllocation::Slice> parameter_buffers;
  parameter_buffers.reserve(instr->operand_count());
  for (const auto* operand : instr->operands()) {
    ASSIGN_OR_RETURN(parameter_buffers.emplace_back(),
                     GetAllocationSliceForHlo(operand));
  }

  // result_buffers: one entry per fusion output leaf in DFS order.
  std::vector<BufferAllocation::Slice> result_buffers;
  RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      instr->shape(),
      [&](const Shape&, const ShapeIndex& index) -> absl::Status {
        ASSIGN_OR_RETURN(result_buffers.emplace_back(),
                         GetAllocationSliceForHlo(instr, index));
        return absl::OkStatus();
      }));

  RETURN_IF_ERROR(DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
      results, parameter_buffers, result_buffers));

  // embedded_allocations: synthetic allocations for the embedded thunk
  // executor. First N entries are for hero operands (one per Parameter),
  // then M entries for hero results (one per Result).
  std::vector<BufferAllocation> embedded_allocations;
  embedded_allocations.reserve(parameters.size() + results.size());

  for (const auto& param : parameters) {
    embedded_allocations.emplace_back(embedded_allocations.size(),
                                      ShapeUtil::ByteSizeOf(param.slice_shape),
                                      0);
  }

  for (const auto& res : results) {
    embedded_allocations.emplace_back(embedded_allocations.size(),
                                      ShapeUtil::ByteSizeOf(res.update_shape),
                                      0);
  }

  // Map hero operands and results to embedded allocations so the embedded
  // thunk emitter resolves the right buffers.
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<BufferAllocation::Slice>>
      overrides;

  for (int64_t i = 0; i < parameters.size(); ++i) {
    int64_t byte_size = ShapeUtil::ByteSizeOf(parameters[i].slice_shape);
    overrides[hero->operand(i)] = {
        BufferAllocation::Slice(&embedded_allocations[i], 0, byte_size)};
  }

  // One override slice per hero output leaf in DFS order.
  ShapeUtil::ForEachLeafShape(
      hero->shape(), [&](const Shape& subshape, const ShapeIndex&) {
        int64_t leaf_idx = overrides[hero].size();
        int64_t byte_size = ShapeUtil::ByteSizeOf(subshape);
        overrides[hero].push_back(BufferAllocation::Slice(
            &embedded_allocations[parameters.size() + leaf_idx], 0, byte_size));
      });

  auto overrides_cleanup = InstallAllocationOverrides(std::move(overrides));

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  bool verify_offsets =
      ir_emitter_context_->debug_options()
          .xla_gpu_experimental_dynamic_slice_fusion_verify_offsets();

  return EmitHloInstruction(hero).Map(
      [thunk_info = std::move(thunk_info), results = std::move(results),
       result_buffers = std::move(result_buffers),
       parameters = std::move(parameters),
       parameter_buffers = std::move(parameter_buffers),
       embedded_allocations = std::move(embedded_allocations),
       verify_offsets](ThunkSequence embedded_thunks) mutable {
        return ThunkSequence::Of(std::make_unique<DynamicSliceFusionV2Thunk>(
            std::move(thunk_info), std::move(parameters), std::move(results),
            std::move(parameter_buffers), std::move(result_buffers),
            std::move(embedded_allocations), std::move(embedded_thunks),
            verify_offsets));
      });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopy(
    const HloInstruction* instr) {
  TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
      instr->operand(0)->shape(), instr->shape(),
      Layout::Equal().MinorToMajorOnly()));
  ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                   GetAllocationSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                   GetAllocationSliceForHlo(instr));
  return GetThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      /*source_buffer=*/ShapedSlice{src_buffer, instr->operand(0)->shape()},
      /*destination_buffer=*/ShapedSlice{dst_buffer, instr->shape()},
      /*mem_size=*/src_buffer.size()));
}

AsyncThunkSequence ThunkEmitter::EmitAsyncCustomCallStart(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  AsyncThunkSequence custom_call_thunks = EmitCustomCallSwitch(wrapped);

  auto* async_start = Cast<HloAsyncInstruction>(instr);
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                   stream_assignment.GetExecutionStreamId(async_start));

  Thunk::ThunkInfo start_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  std::shared_ptr<AsyncExecution> async_execution =
      std::make_shared<AsyncExecution>(start_thunk_info);
  auto [it, inserted] = hlo_async_executions_.emplace(wrapped, async_execution);
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    wrapped->ToString());
  }

  return std::move(custom_call_thunks)
      .Map([start_thunk_info = std::move(start_thunk_info),
            async_execution = std::move(async_execution),
            execution_stream_id](ThunkSequence custom_call_thunks) {
        return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
            std::move(start_thunk_info), execution_stream_id,
            std::move(custom_call_thunks), std::move(async_execution)));
      });
}

absl::Status ThunkEmitter::AssertNonDeterminismIsOkay(
    const std::string& op_name) {
  if (RequireDeterminism(ir_emitter_context_->hlo_module().config())) {
    return Unimplemented(
        "HLO instruction %s does not have a deterministic "
        "implementation, "
        "but run-to-run determinism is required.",
        op_name);
  }
  return absl::OkStatus();
}

AsyncThunkSequence ThunkEmitter::EmitWhile(const HloInstruction* instr) {
  ASSIGN_OR_RETURN(auto config,
                   instr->backend_config<xla::WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count = std::nullopt;
  if (config.has_known_trip_count()) {
    trip_count = config.known_trip_count().n();
  }

  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Buffer slice holding while loop predicate.
  ASSIGN_OR_RETURN(BufferAllocation::Slice pred,
                   GetAllocationSliceForHlo(condition->root_instruction(), {}));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  return std::move(tsl::JoinFutures(EmitHloComputation(condition),
                                    EmitHloComputation(body)))
      .Map([info = std::move(info), pred = pred, trip_count = trip_count](
               std::tuple<ThunkSequence, ThunkSequence> tuple) {
        auto [cond_thunks, body_thunks] = std::move(tuple);
        return GetThunkSequence(std::make_unique<WhileThunk>(
            std::move(info), std::move(pred), std::move(cond_thunks),
            std::move(body_thunks), trip_count));
      });
}

AsyncThunkSequence ThunkEmitter::EmitCallComputation(
    const HloInstruction* instr) {
  DCHECK_EQ(instr->called_computations().size(), 1);
  const HloComputation* computation = instr->called_computations().front();
  return EmitHloComputation(computation);
}

AsyncThunkSequence ThunkEmitter::EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr) {
  ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  ASSIGN_OR_RETURN(KernelDefinition<LlvmKernelSource> kernel_def,
                   EmitRngGetAndUpdateStateLLVMIR(instr, ir_emitter_context_,
                                                  kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](std::unique_ptr<Thunk> thunk) {
        return ThunkSequence::Of(std::move(thunk));
      });
}

AsyncThunkSequence ThunkEmitter::EmitSort(const HloSortInstruction* sort) {
  if (sort->is_stable()) {
    return Internal("Stable sort not supported. Did stable_sort_expander run?");
  }
  std::string op_name(sort->name());
  const Shape& keys_shape = sort->operand(0)->shape();
  ThunkSequence thunks;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    ShapeIndex shape_index =
        sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
    // We assume that the layout of all involved operands and
    // outputs is the same.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, sort->operand(i)->shape(),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));

    BufferAllocation::Slice destination_buffer;
    BufferAllocation::Slice source_address;

    // If possible, we share buffers. If that is not possible, we
    // need to copy the values, because the emitter does the sorting
    // in-place.
    ASSIGN_OR_RETURN(destination_buffer,
                     GetAllocationSliceForHlo(sort, shape_index));
    ASSIGN_OR_RETURN(source_address,
                     GetAllocationSliceForHlo(sort->operand(i), {}));

    if (destination_buffer != source_address) {
      // TODO(b/26783907): Figure out why we never seem to share
      // buffers for key/value sort.
      VLOG(2) << op_name << " requires initial D2D copy for operand " << i;
      thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              sort, ir_emitter_context_->GetNextThunkId()),
          /*source_buffer=*/
          ShapedSlice{source_address, sort->operand(i)->shape()},
          /*destination_buffer=*/
          ShapedSlice{destination_buffer, sort->operand(i)->shape()},
          ShapeUtil::ByteSizeOf(sort->operand(i)->shape())));
    }
  }

  return EmitBitonicSortLLVMIR(sort, ir_emitter_context_)
      .Map([thunks = std::move(thunks)](ThunkSequence sort_thunks) mutable {
        AppendThunkSequence(thunks, sort_thunks);
        return std::move(thunks);
      });
}

template <typename ThunkType>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReplicaOrPartitionId(
    const HloInstruction* instr) {
  ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                   GetAllocationSliceForHlo(instr, {}));
  return GetThunkSequence(std::make_unique<ThunkType>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      result_slice));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRngSeedThunk(
    const HloInstruction* instr) {
  ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                   GetAllocationSliceForHlo(instr, {}));
  return GetThunkSequence(std::make_unique<RngSeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      result_slice));
}
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectivePermute(
    const HloCollectivePermuteInstruction* instr,
    const HloInstruction* absl_nonnull async_start) {
  TF_RET_CHECK(async_start != nullptr);
  // First output is aliased.
  if (async_start->opcode() == HloOpcode::kCollectivePermuteStart) {
    TF_RET_CHECK(async_start->shape().IsTuple() &&
                 async_start->shape().tuple_shapes().size() == 2 &&
                 Shape::Equal().IgnoreMemorySpaceInLayout()(
                     async_start->shape().tuple_shapes(0),
                     async_start->shape().tuple_shapes(1)));
  } else {
    TF_RET_CHECK(async_start->shape().IsTuple() &&
                 async_start->shape().tuple_shapes().size() == 2);
  }

  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  const int64_t replica_count = hlo_config.replica_count();
  const int64_t partition_count = hlo_config.num_partitions();

  auto operands = instr->operands();
  std::vector<CollectiveThunk::Buffer> buffers;
  ThunkSequence thunks;
  for (int oprd_idx = 0; oprd_idx < operands.size(); ++oprd_idx) {
    const auto operand = operands.at(oprd_idx);

    const ShapeIndex nested_shape_idx = ShapeIndex({1, oprd_idx});
    const ShapeIndex normal_shape_idx = ShapeIndex({1});

    const Shape operand_shape = operand->shape();
    const Shape result_shape = async_start->shape().tuple_shapes(1);

    ASSIGN_OR_RETURN(
        BufferAllocation::Slice result_slice,
        GetAllocationSliceForHlo(async_start, result_shape.IsTuple()
                                                  ? nested_shape_idx
                                                  : normal_shape_idx));
    const int64_t src_memory_space = operand_shape.layout().memory_space();
    Shape result_buffer_shape = (result_shape.IsTuple())
                                    ? result_shape.tuple_shapes(oprd_idx)
                                    : result_shape;

    const int64_t dst_memory_space =
        result_buffer_shape.layout().memory_space();

    ASSIGN_OR_RETURN(BufferAllocation::Slice source_slice,
                     GetAllocationSliceForHlo(operand));
    if (CollectivePermuteThunk::IsDegenerate(instr, replica_count,
                                             partition_count)) {
      // For a degenerate collective permute, just generate a copy
      // thunk.
      thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              async_start, ir_emitter_context_->GetNextThunkId()),
          /*source_buffer=*/ShapedSlice{source_slice, operand_shape},
          /*destination_buffer=*/
          ShapedSlice{result_slice, result_buffer_shape},
          /*mem_size=*/ShapeUtil::ByteSizeOf(operand_shape)));
    } else {
      const CollectiveThunk::Buffer buffer = {
          /*element_count=*/ShapeUtil::ElementsIn(operand_shape),
          /*source_buffer=*/ShapedSlice{source_slice, operand_shape},
          /*destination_buffer=*/ShapedSlice{result_slice, result_buffer_shape},
          /*source_memory_space=*/src_memory_space,
          /*destination_memory_space=*/dst_memory_space};
      buffers.push_back(buffer);
    }
  }
  if (!CollectivePermuteThunk::IsDegenerate(instr, replica_count,
                                            partition_count)) {
    thunks.push_back(std::make_unique<CollectivePermuteThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            async_start, ir_emitter_context_->GetNextThunkId()),
        instr, replica_count, partition_count, buffers,
        ir_emitter_context_->debug_options().xla_gpu_collective_permute_mode(),
        ir_emitter_context_->debug_options()
            .xla_gpu_collective_permute_connected_components()));
  }

  // For synchronous collectives, emit thunks directly without async wrapping.
  // However, if parallel collective overlap limit is > 1, multiple collectives
  // may be in-flight on different streams. Emitting them synchronously would be
  // unsafe as they could share communicators across streams. Force async
  // emission in that case.
  if (IsGPUSyncCollective(*async_start) &&
      ir_emitter_context_->debug_options()
              .xla_gpu_experimental_parallel_collective_overlap_limit() <= 1) {
    hlo_async_executions_.try_emplace(async_start, nullptr);
    return thunks;
  }

  // Wrap in AsyncStartThunk for asynchronous execution.
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                   stream_assignment.GetExecutionStreamId(async_start));

  auto start_thunk = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          async_start, ir_emitter_context_->GetNextThunkId()),
      execution_stream_id, std::move(thunks));

  auto [it, inserted] = hlo_async_executions_.emplace(
      async_start, start_thunk->async_execution());
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    async_start->ToString());
  }

  return GetThunkSequence(std::move(start_thunk));
}

template <typename CollectiveThunkType, typename HloInstType>
AsyncThunkSequence ThunkEmitter::EmitCollectiveThunk(
    Thunk::Kind kind, const HloInstruction* async_start,
    const HloInstType* inst, std::optional<bool> use_global_device_ids) {
  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  int64_t replica_count = hlo_config.replica_count();
  int64_t partition_count = hlo_config.num_partitions();
  int64_t operand_count = inst->operand_count();
  VLOG(2) << CollectiveThunkType::GetHloOpName()
          << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << operand_count;

  // Stash relevant information in CollectiveThunk::Buffer even if
  // we may not generate an CollectiveThunk.
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(operand_count);

  // Adds a source and destination buffers pair to `buffers`.
  auto add_buffer = [&](const HloInstruction* src, const HloInstruction* dst,
                        const ShapeIndex& dst_shape_index) -> absl::Status {
    const Shape& src_shape = src->shape();
    const Shape& dst_shape =
        ShapeUtil::GetSubshape(dst->shape(), dst_shape_index);
    ASSIGN_OR_RETURN(auto src_slice, GetAllocationSliceForHlo(src));
    ASSIGN_OR_RETURN(auto dst_slice,
                     GetAllocationSliceForHlo(dst, dst_shape_index));

    buffers.push_back(CollectiveThunk::Buffer{
        /*element_count=*/ShapeUtil::ElementsIn(src_shape),
        /*source_buffer=*/{src_slice, src_shape},
        /*destination_buffer=*/{dst_slice, dst_shape},
        /*source_memory_space=*/src_shape.layout().memory_space(),
        /*destination_memory_space=*/dst_shape.layout().memory_space()});
    return absl::OkStatus();
  };

  if (kind == Thunk::Kind::kAllGather) {
    // Start operations return a tuple of (<<inputs>>, <<outputs>>)
    // where outputs can be a tuple itself (if operation has
    // multiple operands).
    for (int64_t i = 0; i < operand_count; i++) {
      ShapeIndex idx = GetDstShapeIndex(async_start, inst, i, kind);
      RETURN_IF_ERROR(add_buffer(inst->operand(i), async_start, idx));
    }
  } else if (kind == Thunk::Kind::kRaggedAllToAll) {
    // RaggedAllToAll operation has 6 operands: input, output,
    // input_offset, send_size, output_offset, recv_size. `output`
    // operand is aliased with the instruction result. All other
    // operands are not aliased.
    RETURN_IF_ERROR(
        add_buffer(inst->operand(0), inst->operand(0), ShapeIndex({})));
    RETURN_IF_ERROR(add_buffer(inst->operand(1), async_start,
                               GetDstShapeIndex(async_start, inst, 0, kind)));

    for (int64_t i = 2; i < operand_count; i++) {
      RETURN_IF_ERROR(
          add_buffer(inst->operand(i), inst->operand(i), ShapeIndex({})));
    }
  } else {
    // For other operations simply zip operands with results.
    for (int64_t i = 0; i < operand_count; i++) {
      ShapeIndex idx = GetDstShapeIndex(async_start, inst, i, kind);
      RETURN_IF_ERROR(add_buffer(inst->operand(i), async_start, idx));
    }
  }

  // A given collective op can be degenerate if across all groups
  // formed by it are singleton. In such a case, we don't need to do
  // any communication and we can just copy the input to the output.
  //
  // The only exception is RaggedAllToAll, which is not degenerate
  // even if all groups are singleton. In a singleton group case,
  // RaggedAllToAll becomes a generic equivalent of
  // DynamicUpdateSlice, except update size is not statically known.
  // This operation can not be expressed in term of standard HLO
  // instructions, so the best solution we have is to use NCCL thunk
  // even for degenerate cases.
  bool is_degenerate = kind != Thunk::Kind::kRaggedAllToAll &&
                       GetCollectiveConfig(inst, use_global_device_ids)
                           .IsDegenerate(replica_count, partition_count);

  if (is_degenerate) {
    return EmitDegeneratedCollectiveThunk(buffers, async_start, inst);
  }

  RETURN_IF_ERROR(CollectiveThunkType::CheckImplementable(inst, replica_count,
                                                          partition_count));

  auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      inst, ir_emitter_context_->GetNextThunkId());
  // The wrapper name is used when syntactic sugar is turned on.
  if (ir_emitter_context_->debug_options().xla_syntax_sugar_async_ops()) {
    thunk_info.profile_annotation = async_start->name();
  }
  AsyncThunkSequence thunks;
  // TODO(b/828435206) Remove this constexpr once collective kernel thunk is
  // lifted out of the all reduce thunk.
  if constexpr (kRequiresCollectiveKernelThunk<CollectiveThunkType>) {
    thunks =
        EmitCollectiveKernelThunk(
            ir_emitter_context_, *call_graph_, thunk_info, buffers,
            Cast<HloAllReduceInstruction>(inst),
            GetCollectiveConfig(inst, inst->use_global_device_ids()), this,
            analysis_garbage_collector_)
            .Map([thunk_info = std::move(thunk_info),
                  use_memcpy_local_p2p = ir_emitter_context_->debug_options()
                                             .xla_gpu_use_memcpy_local_p2p(),
                  buffers = std::move(buffers),
                  inst](std::unique_ptr<CollectiveKernelThunk>
                            collective_kernel_thunk) {
              return ThunkSequence::Of(std::make_unique<CollectiveThunkType>(
                  thunk_info, inst, /*buffers=*/std::move(buffers),
                  std::move(collective_kernel_thunk), use_memcpy_local_p2p));
            });
  } else if constexpr (std::is_constructible_v<
                           CollectiveThunkType, Thunk::ThunkInfo,
                           decltype(inst),
                           std::vector<CollectiveThunk::Buffer>>) {
    thunks = ThunkSequence::Of(std::make_unique<CollectiveThunkType>(
        thunk_info, inst, /*buffers=*/std::move(buffers)));
  } else {
    thunks = ThunkSequence::Of(std::make_unique<CollectiveThunkType>(
        thunk_info, inst, /*buffers=*/std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p()));
  }

  // For synchronous collectives, emit thunk directly without async wrapping.
  // However, if parallel collective overlap limit is > 1, multiple collectives
  // may be in-flight on different streams. Emitting them synchronously would be
  // unsafe as they could share communicators across streams. Force async
  // emission in that case.
  if (IsGPUSyncCollective(*async_start) &&
      ir_emitter_context_->debug_options()
              .xla_gpu_experimental_parallel_collective_overlap_limit() <= 1) {
    hlo_async_executions_.try_emplace(async_start, nullptr);
    return thunks;
  }

  // Wrap collective thunk in AsyncStartThunk for asynchronous execution.
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                   stream_assignment.GetExecutionStreamId(async_start));

  Thunk::ThunkInfo async_start_thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(
          async_start, ir_emitter_context_->GetNextThunkId());
  std::shared_ptr<AsyncExecution> async_execution =
      std::make_shared<AsyncExecution>(async_start_thunk_info);
  auto [it, inserted] =
      hlo_async_executions_.emplace(async_start, async_execution);
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    async_start->ToString());
  }

  return std::move(thunks).Map(
      [async_start_thunk_info = std::move(async_start_thunk_info),
       execution_stream_id,
       async_execution = std::move(async_execution)](ThunkSequence thunks) {
        return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
            async_start_thunk_info, execution_stream_id, std::move(thunks),
            async_execution));
      });
}

AsyncThunkSequence ThunkEmitter::EmitCollectiveGroupStartThunk(
    const HloInstruction* instr) {
  std::vector<AsyncThunkSequence> futures;
  for (const HloInstruction* nested_instruction :
       instr->async_wrapped_computation()->instructions()) {
    futures.push_back(
        EmitHloInstruction(nested_instruction, /*emit_group_thunks=*/true));
  }

  Thunk::ThunkInfo group_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  Thunk::ThunkInfo start_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  bool is_sync = IsGPUSyncCollective(*instr);

  std::shared_ptr<AsyncExecution> async_execution;
  std::optional<ExecutionStreamId> execution_stream_id;
  if (is_sync) {
    hlo_async_executions_.try_emplace(instr, nullptr);
  } else {
    async_execution = std::make_shared<AsyncExecution>(start_thunk_info);

    auto [it, inserted] = hlo_async_executions_.emplace(instr, async_execution);
    if (!inserted) {
      return Internal("Async execution already exists for instruction %s",
                      instr->ToString());
    }

    const ExecutionStreamAssignment& stream_assignment =
        ir_emitter_context_->execution_stream_assignment();
    ASSIGN_OR_RETURN(execution_stream_id,
                     stream_assignment.GetExecutionStreamId(instr));
  }

  return tsl::JoinFutures(absl::MakeSpan(futures))
      .Map([group_thunk_info = std::move(group_thunk_info),
            start_thunk_info = std::move(start_thunk_info),
            async_execution = std::move(async_execution), execution_stream_id,
            is_sync](std::vector<ThunkSequence> sequences) {
        ThunkSequence thunks = FlattenThunkSequence(std::move(sequences));
        auto group_thunk = std::make_unique<CollectiveGroupThunk>(
            std::move(group_thunk_info), Thunk::Kind::kGroup,
            std::move(thunks));

        // For synchronous collectives, emit group thunk directly without async
        // wrapping.
        if (is_sync) {
          return ThunkSequence::Of(std::move(group_thunk));
        }

        return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
            std::move(start_thunk_info), *execution_stream_id,
            ThunkSequence::Of(std::move(group_thunk)),
            std::move(async_execution)));
      });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectiveAsyncDone(
    const HloInstruction* inst) {
  // Determine if this is a send/recv done operation.
  bool is_send_recv =
      inst->opcode() == HloOpcode::kSendDone ||
      inst->opcode() == HloOpcode::kRecvDone ||
      (inst->opcode() == HloOpcode::kAsyncDone &&
       (inst->async_wrapped_instruction()->opcode() == HloOpcode::kSend ||
        inst->async_wrapped_instruction()->opcode() == HloOpcode::kRecv));
  const HloInstruction* start =
      is_send_recv ? FindCanonicalSendRecvStartOp(inst) : inst->operand(0);

  // Find the async execution for the start operation.
  auto it = hlo_async_executions_.find(start);
  TF_RET_CHECK(it != hlo_async_executions_.end())
      << "couldn't find async execution for start operation";

  // Can be null if no start thunk was created (e.g. if the start op
  // is degenerate), in which case there's nothing to do here.
  if (!it->second) {
    return ThunkSequence{};
  }

  return GetThunkSequence(std::make_unique<AsyncDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          inst, ir_emitter_context_->GetNextThunkId()),
      it->second));
}

template <typename HloInstType>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitDegeneratedCollectiveThunk(
    std::vector<CollectiveThunk::Buffer>& buffers,
    const HloInstruction* async_start, const HloInstType* inst) {
  // Signal that start thunk not created (degenerate) with nullptr.
  hlo_async_executions_.try_emplace(async_start, nullptr);

  // Degenerate collectives are simply identity function. Buffer
  // assignment expects a copy, so that's what we do.
  ThunkSequence thunks;
  for (int64_t i = 0; i < buffers.size(); i++) {
    const Shape shape = inst->operand(i)->shape();
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            inst, ir_emitter_context_->GetNextThunkId()),
        ShapedSlice{buffers[i].source_buffer.slice, shape},
        ShapedSlice{buffers[i].destination_buffer.slice, shape},
        ShapeUtil::ByteSizeOf(shape)));
  }
  if (thunks.size() == 1) {
    return thunks;
  }
  return GetThunkSequence(std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          inst, ir_emitter_context_->GetNextThunkId()),
      std::move(thunks)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitInfeed(
    const HloInfeedInstruction* instr) {
  // Infeed instruction returns a tuple containing the result data
  // and a token. We only need the result data to construct the
  // infeed thunk.
  std::vector<ShapedSlice> shaped_slices;
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple() || subshape.IsToken()) return absl::OkStatus();
        if (subshape.IsArray()) {
          ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                           GetAllocationSliceForHlo(instr, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        instr->ToString(), index.ToString());
      }));

  return GetThunkSequence(std::make_unique<InfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(shaped_slices)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitOutfeed(
    const HloOutfeedInstruction* instr) {
  // HLO outfeed instruction has 2 operands, the source and a token,
  // and a single token output.
  const HloInstruction* source = instr->operand(0);
  std::vector<ShapedSlice> shaped_slices;
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      source->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple()) return absl::OkStatus();
        if (subshape.IsArray()) {
          ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                           GetAllocationSliceForHlo(source, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        source->ToString(), index.ToString());
      }));

  return GetThunkSequence(std::make_unique<OutfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(shaped_slices)));
}

static absl::flat_hash_map<std::string, std::string> ConvertFrontendAttributes(
    const FrontendAttributes& attrs) {
  absl::flat_hash_map<std::string, std::string> result;
  // NOLINTNEXTLINE
  for (auto& [k, v] : attrs.map()) {
    result[k] = v;
  }
  return result;
}

static std::optional<GlobalDeviceId> DeviceConstraint(
    const HloInstruction* hlo) {
  if (hlo->has_sharding() && hlo->sharding().IsSingleDevice()) {
    return GlobalDeviceId(hlo->sharding().GetUniqueDevice());
  }
  return std::nullopt;
}

absl::StatusOr<bool> ShapeHasHostMemorySpace(Shape shape, int index,
                                             int host_memory_space) {
  return shape.tuple_shapes(index).has_layout() &&
         shape.tuple_shapes(index).layout().memory_space() == host_memory_space;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyStartThunk(
    const HloCopyStartInstruction* copy_start_instr) {
  // copy-start has a tuple shape: {host, device, context},
  // or {device, host, context}.
  // Only the destination shape is needed to get the output buffer.
  ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                   GetAllocationSliceForHlo(copy_start_instr,
                                            /*index=*/{0}));

  const HloInstruction* src = copy_start_instr->operand(0);
  const Shape& input_shape = src->shape();
  ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                   GetAllocationSliceForHlo(src, {}));
  const Shape& shape = copy_start_instr->shape();
  CHECK(shape.IsTuple());
  auto host_memory_space =
      static_cast<int>(stream_executor::MemorySpace::kHost);
  ASSIGN_OR_RETURN(bool is_dst_host_memory,
                   ShapeHasHostMemorySpace(shape, 0, host_memory_space));
  ASSIGN_OR_RETURN(bool is_src_host_memory,
                   ShapeHasHostMemorySpace(shape, 1, host_memory_space));
  if (is_dst_host_memory == is_src_host_memory) {
    return absl::InternalError(
        absl::StrFormat("Copy-start %s doesn't have correct host memory space "
                        "color S(%d)",
                        copy_start_instr->ToString(),
                        static_cast<int>(stream_executor::MemorySpace::kHost)));
  }

  // Create the copy thunk with ThunkInfo derived from copy-start.
  Thunk::ThunkInfo copy_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      copy_start_instr, ir_emitter_context_->GetNextThunkId());

  std::unique_ptr<CopyThunk> copy_thunk;
  if (is_dst_host_memory) {
    copy_thunk = std::make_unique<DeviceToHostCopyThunk>(
        copy_thunk_info,
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape));
  } else {
    copy_thunk = std::make_unique<HostToDeviceCopyThunk>(
        copy_thunk_info,
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape));
  }

  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  auto execution_stream_id =
      stream_assignment.GetExecutionStreamId(copy_start_instr);

  // If copy-start is not a scope-start operation, the copy is synchronous.
  if (!execution_stream_id.ok()) {
    return GetThunkSequence(std::move(copy_thunk));
  }

  // Wrap the copy thunk in an AsyncStartThunk for asynchronous execution.
  ThunkSequence nested_thunks;
  nested_thunks.push_back(std::move(copy_thunk));

  auto start_thunk = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          copy_start_instr, ir_emitter_context_->GetNextThunkId()),
      *execution_stream_id, std::move(nested_thunks));

  auto [it, inserted] = hlo_async_executions_.emplace(
      copy_start_instr, start_thunk->async_execution());
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    copy_start_instr->ToString());
  }

  return GetThunkSequence(std::move(start_thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyDoneThunk(
    const HloInstruction* instr) {
  const HloInstruction* copy_start_instr = instr->operand(0);
  CHECK(copy_start_instr->opcode() == HloOpcode::kCopyStart);

  // If the copy-start was asynchronous, emit an AsyncDoneThunk.
  auto it = hlo_async_executions_.find(copy_start_instr);
  if (it != hlo_async_executions_.end()) {
    return GetThunkSequence(std::make_unique<AsyncDoneThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        it->second));
  }

  // Synchronous copy: copy-done is a no-op.
  return ThunkSequence();
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSendThunk(
    const HloSendInstruction* instr, bool emit_group_thunks) {
  const HloInstruction* src = instr->operand(0);
  ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(src, {}));
  if (!instr->is_host_transfer()) {
    const auto& hlo_config = ir_emitter_context_->hlo_module().config();
    const int64_t replica_count = hlo_config.replica_count();
    const int64_t partition_count = hlo_config.num_partitions();
    const int64_t memory_space =
        instr->shape().IsTuple()
            ? instr->shape().tuple_shapes(0).layout().memory_space()
            : instr->shape().layout().memory_space();

    std::unique_ptr<Thunk> thunk;
    const CollectiveThunk::Buffer buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(src->shape()),
        /*source_buffer=*/slice,
        /*destination_buffer=*/slice,
        /*source_memory_space=*/memory_space,
        /*destination_memory_space=*/memory_space};
    thunk = std::make_unique<SendThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        instr, replica_count, partition_count, buffer);
    // Wrap in AsyncStartThunk if not emitted as part of a group thunk.
    if (!emit_group_thunks) {
      const HloInstruction* canonical_send_instr =
          FindCanonicalSendRecvStartOp(instr);

      const ExecutionStreamAssignment& stream_assignment =
          ir_emitter_context_->execution_stream_assignment();
      ASSIGN_OR_RETURN(
          ExecutionStreamId execution_stream_id,
          stream_assignment.GetExecutionStreamId(canonical_send_instr));

      // Check if an async execution already exists for this canonical
      // send/recv pair (pipelined send/recv share the same async stream).
      auto existing_it = hlo_async_executions_.find(canonical_send_instr);
      if (existing_it != hlo_async_executions_.end()) {
        auto start_thunk = std::make_unique<AsyncStartThunk>(
            Thunk::ThunkInfo::WithProfileAnnotation(
                instr, ir_emitter_context_->GetNextThunkId()),
            execution_stream_id, ThunkSequence::Of(std::move(thunk)),
            existing_it->second);
        return GetThunkSequence(std::move(start_thunk));
      }

      auto start_thunk = std::make_unique<AsyncStartThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          execution_stream_id, ThunkSequence::Of(std::move(thunk)));

      hlo_async_executions_.try_emplace(canonical_send_instr,
                                        start_thunk->async_execution());
      return GetThunkSequence(std::move(start_thunk));
    }
    return GetThunkSequence(std::move(thunk));
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send instruction");
  }

  return GetThunkSequence(std::make_unique<HostSendThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      src->shape(), slice.slice, *instr->channel_id(), send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSendDoneThunk(
    const HloSendDoneInstruction* instr) {
  if (!instr->is_host_transfer()) {
    return EmitCollectiveAsyncDone(instr);
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send done "
        "instruction");
  }

  return GetThunkSequence(std::make_unique<HostSendDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      *instr->channel_id(), send_recv_events_, DeviceConstraint(instr)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRecvThunk(
    const HloRecvInstruction* instr, bool emit_group_thunks) {
  TF_RET_CHECK(instr->shape().IsTuple());
  ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(instr, {0}));

  if (!instr->is_host_transfer()) {
    const auto& hlo_config = ir_emitter_context_->hlo_module().config();
    const int64_t replica_count = hlo_config.replica_count();
    const int64_t partition_count = hlo_config.num_partitions();

    const int64_t memory_space =
        instr->shape().IsTuple()
            ? instr->shape().tuple_shapes(0).layout().memory_space()
            : instr->shape().layout().memory_space();

    std::unique_ptr<Thunk> thunk;
    const CollectiveThunk::Buffer buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(instr->shape().tuple_shapes(0)),
        /*source_buffer=*/slice,
        /*destination_buffer=*/slice,
        /*source_memory_space=*/memory_space,
        /*destination_memory_space=*/memory_space};
    thunk = std::make_unique<RecvThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        instr, replica_count, partition_count, buffer);
    // Wrap in AsyncStartThunk if not emitted as part of a group thunk.
    if (!emit_group_thunks) {
      const HloInstruction* canonical_recv_instr =
          FindCanonicalSendRecvStartOp(instr);

      const ExecutionStreamAssignment& stream_assignment =
          ir_emitter_context_->execution_stream_assignment();
      ASSIGN_OR_RETURN(
          ExecutionStreamId execution_stream_id,
          stream_assignment.GetExecutionStreamId(canonical_recv_instr));

      // Check if an async execution already exists for this canonical
      // send/recv pair (pipelined send/recv share the same async stream).
      auto existing_it = hlo_async_executions_.find(canonical_recv_instr);
      if (existing_it != hlo_async_executions_.end()) {
        auto start_thunk = std::make_unique<AsyncStartThunk>(
            Thunk::ThunkInfo::WithProfileAnnotation(
                instr, ir_emitter_context_->GetNextThunkId()),
            execution_stream_id, ThunkSequence::Of(std::move(thunk)),
            existing_it->second);
        return GetThunkSequence(std::move(start_thunk));
      }

      auto start_thunk = std::make_unique<AsyncStartThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          execution_stream_id, ThunkSequence::Of(std::move(thunk)));

      hlo_async_executions_.try_emplace(canonical_recv_instr,
                                        start_thunk->async_execution());
      return GetThunkSequence(std::move(start_thunk));
    }
    return GetThunkSequence(std::move(thunk));
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv instruction");
  }

  return GetThunkSequence(std::make_unique<HostRecvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr->shape().tuple_shapes()[0], slice.slice, *instr->channel_id(),
      send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRecvDoneThunk(
    const HloRecvDoneInstruction* instr) {
  if (!instr->is_host_transfer()) {
    return EmitCollectiveAsyncDone(instr);
  }
  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv done "
        "instruction");
  }
  return GetThunkSequence(std::make_unique<HostRecvDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      *instr->channel_id(), send_recv_events_, DeviceConstraint(instr)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncDone(
    const HloInstruction* instr) {
  if (!instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    return EmitCollectiveAsyncDone(instr);
  }
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  ThunkSequence thunks;
  switch (wrapped->opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllGather:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllToAll:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kRecv:
    case HloOpcode::kSend:
      return EmitCollectiveAsyncDone(instr);
    case HloOpcode::kFusion:
    case HloOpcode::kCall:
    case HloOpcode::kCustomCall: {
      if (IsHostExecuteCustomCall(*wrapped)) {
        auto custom_call = Cast<HloCustomCallInstruction>(wrapped);

        auto async_events =
            GetInstructionToHostExecuteAsyncEvents().at(custom_call);

        absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4>
            result_slices;
        for (auto& indexed : ShapeUtil::GetLeafShapes(wrapped->shape())) {
          TF_ASSIGN_OR_RETURN(
              auto slice,
              ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                  wrapped, indexed.index));
          result_slices.push_back({slice, indexed.shape});
        }

        thunks.push_back(std::make_unique<HostExecuteDoneThunk>(
            Thunk::ThunkInfo::WithProfileAnnotation(
                instr, ir_emitter_context_->GetNextThunkId()),
            std::move(result_slices), async_events));
        return thunks;
      }
      auto it = hlo_async_executions_.find(wrapped);
      if (it == hlo_async_executions_.end()) {
        return Internal(
            "Async execution not found for instruction %s. "
            "EmitAsyncComputation must be called before EmitAsyncDone.",
            wrapped->ToString());
      }
      thunks.push_back(std::make_unique<AsyncDoneThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          it->second));
      return thunks;
    }
    default:
      return Internal("Unsupported async done wrapped instruction: %s",
                      HloOpcodeString(wrapped->opcode()));
  }
}

AsyncThunkSequence ThunkEmitter::EmitAsyncStart(const HloInstruction* instr) {
  // Multi-op async start will emit a NCCL group thunk.
  if (!instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    return EmitCollectiveGroupStartThunk(instr);
  }
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  switch (wrapped->opcode()) {
    case HloOpcode::kAllReduce: {
      auto* all_reduce = Cast<HloAllReduceInstruction>(wrapped);
      return EmitCollectiveThunk<AllReduceThunk, HloAllReduceInstruction>(
          Thunk::kAllReduce, instr, all_reduce, std::nullopt);
    }
    case HloOpcode::kAllGather: {
      auto* all_gather = Cast<HloAllGatherInstruction>(wrapped);
      return EmitCollectiveThunk<AllGatherThunk, HloAllGatherInstruction>(
          Thunk::kAllGather, instr, all_gather, std::nullopt);
    }
    case HloOpcode::kCollectivePermute: {
      auto* collective_permute = Cast<HloCollectivePermuteInstruction>(wrapped);
      return EmitCollectivePermute(collective_permute, instr);
    }
    case HloOpcode::kReduceScatter: {
      auto* reduce_scatter = Cast<HloReduceScatterInstruction>(wrapped);
      return EmitCollectiveThunk<ReduceScatterThunk,
                                 HloReduceScatterInstruction>(
          Thunk::kReduceScatter, instr, reduce_scatter,
          reduce_scatter->use_global_device_ids());
    }
    case HloOpcode::kAllToAll: {
      auto* all_to_all = Cast<HloAllToAllInstruction>(wrapped);
      return EmitCollectiveThunk<AllToAllThunk, HloAllToAllInstruction>(
          Thunk::kAllToAll, instr, all_to_all, std::nullopt);
    }
    case HloOpcode::kRaggedAllToAll: {
      auto* ragged_all_to_all = Cast<HloRaggedAllToAllInstruction>(wrapped);
      return EmitCollectiveThunk<RaggedAllToAllThunk,
                                 HloRaggedAllToAllInstruction>(
          Thunk::kRaggedAllToAll, instr, ragged_all_to_all, std::nullopt);
    }
    case HloOpcode::kCollectiveBroadcast: {
      auto* collective_broadcast =
          Cast<HloCollectiveBroadcastInstruction>(wrapped);
      return EmitCollectiveThunk<CollectiveBroadcastThunk,
                                 HloCollectiveBroadcastInstruction>(
          Thunk::kCollectiveBroadcast, instr, collective_broadcast,
          std::nullopt);
    }
    case HloOpcode::kFusion: {
      AsyncThunkSequence fusion_thunks =
          EmitFusion(Cast<HloFusionInstruction>(wrapped));

      Thunk::ThunkInfo start_thunk_info =
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId());
      std::shared_ptr<AsyncExecution> async_execution =
          std::make_shared<AsyncExecution>(start_thunk_info);
      auto [it, inserted] =
          hlo_async_executions_.emplace(wrapped, async_execution);
      if (!inserted) {
        return Internal("Async execution already exists for instruction %s",
                        wrapped->ToString());
      }

      auto* async_start = Cast<HloAsyncInstruction>(instr);
      const ExecutionStreamAssignment& stream_assignment =
          ir_emitter_context_->execution_stream_assignment();
      ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                       stream_assignment.GetExecutionStreamId(async_start));

      return std::move(fusion_thunks)
          .Map([start_thunk_info = std::move(start_thunk_info),
                async_execution = std::move(async_execution),
                execution_stream_id](ThunkSequence fusion_thunks) {
            return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
                std::move(start_thunk_info), execution_stream_id,
                std::move(fusion_thunks), std::move(async_execution)));
          });
    }
    case HloOpcode::kCall: {
      return EmitAsyncComputation(instr);
    }
    case HloOpcode::kCustomCall: {
      if (IsHostExecuteCustomCall(*wrapped)) {
        auto custom_call = Cast<HloCustomCallInstruction>(wrapped);

        std::unique_ptr<HloModule> hlo_module =
            ExtractComputationIntoNewModule(*custom_call->called_computation());

        // All offloaded computations are marked as host computations from
        // the perspective of the GPU backend. Since these will execute on
        // the main thread from the CPU backend perspective, we need to mark
        // them as such.
        for (auto* computation : hlo_module->computations()) {
          computation->SetExecutionThread(HloInstruction::kMainExecutionThread);
        }

        absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4>
            operand_slices;
        for (HloInstruction* operand : wrapped->operands()) {
          for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
            ASSIGN_OR_RETURN(
                auto slice,
                ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                    operand, indexed.index));
            operand_slices.push_back({slice, indexed.shape});
          }
        }

        // Collect buffer slices for all results.
        absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4>
            result_slices;
        for (auto& indexed : ShapeUtil::GetLeafShapes(wrapped->shape())) {
          ASSIGN_OR_RETURN(
              auto slice,
              ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                  wrapped, indexed.index));
          result_slices.push_back({slice, indexed.shape});
        }

        HostOffloadingExecutableProto host_offloading_executable_proto;
        *host_offloading_executable_proto.mutable_hlo_module() =
            hlo_module->ToProto();
        host_offloading_executable_proto.set_executable_type(
            HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT);

        ASSIGN_OR_RETURN(
            auto thunk,
            HostExecuteStartThunk::Create(
                Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(host_offloading_executable_proto),
                std::move(operand_slices), std::move(result_slices)));

        auto async_events = thunk->async_events();

        auto [it, inserted] = GetInstructionToHostExecuteAsyncEvents().emplace(
            custom_call, async_events);
        if (!inserted) {
          return Internal(
              "Async events already exist for host offloading custom call "
              "%s.",
              custom_call->ToString());
        }
        return GetThunkSequence(std::move(thunk));
      }
      return EmitAsyncCustomCallStart(instr);
    }
    default:
      return Internal("Unsupported async start wrapped instruction: %s",
                      HloOpcodeString(wrapped->opcode()));
  }
}

AsyncThunkSequence ThunkEmitter::EmitCustomCallSwitch(
    const HloInstruction* hlo) {
  auto* custom_call = Cast<HloCustomCallInstruction>(hlo);
  // zml.Tensor.print -> "zml$print": a side-effect-only debugging print. Handle it
  // with a host-readback thunk (Metal has no FFI print handler wired).
  if (custom_call->custom_call_target() == "zml$print") {
    return EmitMetalPrintThunk(custom_call);
  }
  // zml's `.metal_fa` attention backend -> "zml$flash_attn": a fused GQA flash-
  // attention DECODE kernel replacing sdpa's 2 MPSGraph dots + softmax.
  if (custom_call->custom_call_target() == "zml$flash_attn") {
    return EmitMetalFlashAttnThunk(custom_call);
  }
  // zml's metal paged-attention backend -> "zml$paged_attn": paged attention
  // over a paged KV cache + block tables. The thunk picks the kernel like
  // vllm's paged_ops.cpp does: tiled FA-2 MMA for prefill, fa_vec_paged
  // (matrix-vector) for decode (total_q_tokens == num_seqs).
  if (custom_call->custom_call_target() == "zml$paged_attn") {
    return EmitMetalPagedAttnThunk(custom_call);
  }
  // Backend-generated (RewriteKvCacheWrites in metal_gpu_compiler.cc): the
  // decode-step paged KV-cache write as ONE predicated-store kernel, replacing
  // the 6-kernel slice/pred/select/DUS cluster.
  if (custom_call->custom_call_target() == "metal$kv_write") {
    return EmitMetalKvWriteThunk(custom_call);
  }
  // Backend-generated (RewriteSortToMetalThunk in metal_gpu_compiler.cc): a
  // generic Sort on Metal, routed to the native MLX merge sort (the legacy LLVM
  // bitonic emitter can't lower to valid AIR).
  if (IsMetalSort(*hlo)) {
    return EmitMetalSortThunk(custom_call);
  }
  // zml's Gated DeltaNet recurrence -> "zml$gdn": the recurrent delta-rule
  // linear-attention kernel for Qwen3-Next style hybrid models (the Triton GDN
  // recurrence has no Metal compiler path).
  if (custom_call->custom_call_target() == "zml$gdn") {
    return EmitMetalGdnThunk(custom_call);
  }
  // Weight-only scaled matmul: FusedScaledDotRewriter's Metal arm rewrites a
  // weight-only kScaledDot into zml$scaled_matmul; dispatch NVFP4 group-16 or
  // 128-block / per-channel FP8 via ClassifyMetalScaledMatmul (MX is not a
  // fused scheme -- it never reaches this call).
  if (IsMetalScaledMatmul(*hlo)) {
    return EmitMetalScaledMatmulThunk(custom_call);
  }
  // Model-emitted grouped MoE GEMV (bf16 / fp8 / nvfp4): the model does top-k
  // routing and emits __metal$moe_gemm{,$f8,$f4}; route to MetalMoeGemvThunk.
  if (IsMetalMoeGemmAny(*hlo)) {
    return EmitMoeGemvThunk(custom_call);
  }
  // Apple Metal has no cuBLAS/cuBLAS-LT. GemmRewriter emits an honest
  // __metal$gemm target (GetNonFp8GemmCustomCallTarget, gated on the first-class
  // MetalComputeCapability) which runs via the in-tree metalBLAS kernels.
  if (IsMetalGemm(*hlo)) {
    return EmitMetalGemmThunk(custom_call);
  }
  if (IsCublasLtMatmul(*hlo)) {
    return EmitCublasLtMatmulThunk(custom_call);
  }
  if (IsCublasLtMatmulF8(*hlo)) {
    return EmitCublasLtMatmulThunkF8(custom_call);
  }
  if (IsCublasLtGroupedMatmul(*hlo)) {
    return EmitCublasLtGroupedMatmulThunk(custom_call);
  }
  if (IsCublasLtMatmulMx(*hlo)) {
    return EmitCublasLtMatmulThunkMx(custom_call);
  }
  if (IsCudnnConvolutionReorder(*hlo)) {
    return EmitConvolutionReorderThunk(custom_call);
  }
  if (IsCustomCallToDnnNorm(*hlo)) {
    return EmitNormThunk(custom_call);
  }
  if (IsCustomCallTofMHA(*hlo) || IsCustomCallTofMHAF8(*hlo) ||
      IsCustomCallToBlockScaledDot(*hlo)) {
    return EmitCuDnnThunk(custom_call);
  }
  if (IsCustomCallToPtxKernel(*hlo)) {
    return EmitPtxCustomCall(custom_call);
  }
  if (IsCustomCallToTopK(*hlo)) {
    return EmitTopKCustomCall(custom_call);
  }
  if (IsCustomCallToDnnConvolution(*hlo)) {
    return EmitConvolutionThunk(custom_call);
  }
  if (IsTriangularSolve(*hlo)) {
    return EmitTriangularSolveCustomCall(hlo);
  }
  // CUB sort is handled as a generic FFI custom call via CustomCallThunk.
  // See xla.gpu.ext.cub_sort_keys and xla.gpu.ext.cub_sort_pairs handlers.
  if (hlo->custom_call_target() == "PadToStatic") {
    return EmitPadToStatic(custom_call);
  }
  if (hlo->custom_call_target() == "SliceToDynamic") {
    return EmitSliceToDynamic(custom_call);
  }
  if (hlo->custom_call_target() == "__gpu$xla.gpu.triton") {
    // TODO(slebedev): Remove this after June 15th 2025.
    return EmitTritonCustomCall(custom_call);
  }
  if (hlo->custom_call_target() == kNopCustomCallTarget) {
    return ThunkSequence{};
  }
  if (hlo->custom_call_target() == kPinCustomCallTarget ||
      hlo->custom_call_target() == kUnpinCustomCallTarget ||
      hlo->custom_call_target() == kCreateBufferCustomCallTarget) {
    return ThunkSequence{};
  }
  if (hlo->custom_call_target() == "GetRngSeed") {
    return EmitRngSeedThunk(hlo);
  }
  return EmitCustomCallThunk(custom_call);
}

AsyncThunkSequence ThunkEmitter::EmitHloInstruction(const HloInstruction* hlo,
                                                    bool emit_group_thunks) {
  switch (hlo->opcode()) {
    case HloOpcode::kAllGatherDone:
      return EmitCollectiveAsyncDone(hlo);
    case HloOpcode::kAllGatherStart: {
      auto* all_gather = Cast<HloAllGatherInstruction>(hlo);
      return EmitCollectiveThunk<AllGatherThunk, HloAllGatherInstruction>(
          Thunk::kAllGather, all_gather, all_gather,
          all_gather->use_global_device_ids());
    }
    case HloOpcode::kAllReduceDone:
      return EmitCollectiveAsyncDone(hlo);
    case HloOpcode::kAllReduceStart: {
      auto* all_reduce = Cast<HloAllReduceInstruction>(hlo);
      return EmitCollectiveThunk<AllReduceThunk, HloAllReduceInstruction>(
          Thunk::kAllReduce, all_reduce, all_reduce,
          all_reduce->use_global_device_ids());
    }
    case HloOpcode::kAsyncDone:
      return EmitAsyncDone(hlo);
    case HloOpcode::kAsyncStart:
      return EmitAsyncStart(hlo);
    case HloOpcode::kCall:
      return EmitCallComputation(hlo);
    case HloOpcode::kCollectivePermuteDone:
      return EmitCollectiveAsyncDone(hlo);
    case HloOpcode::kCollectivePermuteStart:
      // In the legacy scenario of using a collective permute start instead of
      // wrapping it in async-start the start instruction is also the collective
      // instruction.
      return EmitCollectivePermute(Cast<HloCollectivePermuteInstruction>(hlo),
                                   hlo);
    case HloOpcode::kConditional:
      return EmitConditional(hlo);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(hlo));
    case HloOpcode::kCustomCall:
      return EmitCustomCallSwitch(hlo);
    case HloOpcode::kFusion:
      return EmitFusion(Cast<HloFusionInstruction>(hlo));
    case HloOpcode::kCopy:
      return EmitCopy(hlo);
    case HloOpcode::kInfeed:
      return EmitInfeed(Cast<HloInfeedInstruction>(hlo));
    case HloOpcode::kOutfeed:
      return EmitOutfeed(Cast<HloOutfeedInstruction>(hlo));
    case HloOpcode::kPartitionId:
      return EmitReplicaOrPartitionId<PartitionIdThunk>(hlo);
    case HloOpcode::kFft:
      return EmitFftThunk(Cast<HloFftInstruction>(hlo));

    case HloOpcode::kRecv:
      return EmitRecvThunk(Cast<HloRecvInstruction>(hlo), emit_group_thunks);
    case HloOpcode::kRecvDone:
      return EmitRecvDoneThunk(Cast<HloRecvDoneInstruction>(hlo));

    case HloOpcode::kReplicaId:
      return EmitReplicaOrPartitionId<ReplicaIdThunk>(hlo);
    case HloOpcode::kRngGetAndUpdateState:
      return EmitRngGetAndUpdateState(
          Cast<HloRngGetAndUpdateStateInstruction>(hlo));

    case HloOpcode::kSend:
      return EmitSendThunk(Cast<HloSendInstruction>(hlo), emit_group_thunks);
    case HloOpcode::kSendDone:
      return EmitSendDoneThunk(Cast<HloSendDoneInstruction>(hlo));

    case HloOpcode::kSort:
      return EmitSort(Cast<HloSortInstruction>(hlo));
    case HloOpcode::kWhile:
      return EmitWhile(hlo);
    case HloOpcode::kCopyStart:
      return EmitCopyStartThunk(Cast<HloCopyStartInstruction>(hlo));
    case HloOpcode::kCopyDone:
      return EmitCopyDoneThunk(hlo);

    // HLO module is already scheduled, so instructions for ordering
    // are noops.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    // We don't need to emit thunks for these operations because
    // their semantics are encoded by buffers.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return ThunkSequence{};
    default:
      return Internal("Unsupported instruction opcode: %s",
                      HloOpcodeString(hlo->opcode()));
  }
  return Internal("Unhandled HLO instruction");
}

xla::Future<std::unique_ptr<SequentialThunk>>
ThunkEmitter::EmitHloEntryComputation(const HloModule* module) {
  return EmitHloComputation(module->entry_computation())
      .Map([](ThunkSequence thunks) {
        return std::make_unique<SequentialThunk>(Thunk::ThunkInfo{},
                                                 std::move(thunks));
      });
}

AsyncThunkSequence ThunkEmitter::EmitHloComputation(
    const HloComputation* computation) {
  const HloSchedule& schedule = computation->parent()->schedule();
  const HloModule* hlo_module = schedule.module();
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_command_buffer_scheduling_mode() ==
      DebugOptions::CONCURRENT_REGIONS) {
    if (concurrent_regions_ordering_.count(hlo_module) == 0) {
      concurrent_regions_ordering_[hlo_module] =
          std::make_unique<ConcurrentRegionsHloOrdering>(schedule);
    }
  }
  if (!schedule.is_computation_scheduled(computation)) {
    return Internal("Sequence not found for computation: %s",
                    computation->name());
  }
  const std::vector<HloInstruction*>& instructions =
      schedule.sequence(computation).instructions();
  std::vector<AsyncThunkSequence> futures(instructions.size());
  for (int i = 0; i < instructions.size(); i++) {
    futures[i] = EmitHloInstruction(instructions[i]);
  }

  return tsl::JoinFutures(absl::MakeSpan(futures))
      .Map([&instructions,
            &concurrent_regions_ordering = concurrent_regions_ordering_,
            hlo_module](std::vector<ThunkSequence> sequences) {
        absl::flat_hash_map<const HloInstruction*, Thunk*> instr_to_thunk;
        for (int i = 0; i < instructions.size(); i++) {
          const HloInstruction* instr = instructions[i];
          ThunkSequence& thunks = sequences[i];
          if (!thunks.empty()) {
            instr_to_thunk[instr] = thunks.back().get();
          }
          // Set the concurrent region id for the thunks, if it exists.
          if (concurrent_regions_ordering.count(hlo_module)) {
            auto concurrent_region_id =
                concurrent_regions_ordering.at(hlo_module)
                    ->GetConcurrentRegionId(instr);
            for (auto& thunk : thunks) {
              if (concurrent_region_id.has_value()) {
                thunk->set_concurrent_region_id(concurrent_region_id.value());
              }
            }
          }
        }

        return FlattenThunkSequence(std::move(sequences));
      });
}

}  // namespace xla::gpu
