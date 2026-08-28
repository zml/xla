/*Copyright 2026 The OpenXLA Authors.

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

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"
#include "xla/backends/gpu/runtime/convolution_thunk.h"
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
#include "xla/backends/gpu/runtime/memset_thunk.h"
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
#include "xla/backends/gpu/transforms/dynamic_slice_copy.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/ffi/attribute_map.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
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
#include "xla/backends/gpu/runtime/metal_flash_attn_thunk.h"
#include "xla/backends/gpu/runtime/metal_fp8_gemv_thunk.h"
#include "xla/backends/gpu/runtime/metal_gdn_thunk.h"
#include "xla/backends/gpu/runtime/metal_gemm_thunk.h"
#include "xla/backends/gpu/runtime/metal_kv_write_thunk.h"
#include "xla/backends/gpu/runtime/metal_moe_gemv_thunk.h"
#include "xla/backends/gpu/runtime/metal_mx_matmul_thunk.h"
#include "xla/backends/gpu/runtime/metal_nvfp4_matmul_thunk.h"
#include "xla/backends/gpu/runtime/metal_paged_attn_thunk.h"
#include "xla/backends/gpu/runtime/metal_print_thunk.h"
#include "xla/backends/gpu/runtime/metal_sort_thunk.h"
#include "xla/backends/gpu/runtime/metal_topk_thunk.h"
#include "xla/backends/gpu/runtime/metal_workspace.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/metal_custom_calls.h"
#include "xla/service/gpu/metalblas_gemm.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/triton_call.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
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

using ::xla::xtile::BlockLevelParameters;

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

bool IsImplicitAsyncSendRecvStart(const HloInstruction* instr) {
  // A device send/recv outside an async computation implicitly acts as an
  // async-start even though its HLO opcode does not spell out "start". Inside
  // an async computation it is emitted as a synchronous operation; the
  // enclosing generic async-start/done pair owns asynchronous execution and
  // completion.
  return !instr->parent()->IsAsyncComputation();
}

bool HasCollectivesGroupAttribute(const HloInstruction* instr) {
  return instr->frontend_attributes().map().contains(
      kCollectiveGroupMarkerAttr);
}

// FFI custom-call targets that XLA:GPU itself introduces during lowering
// (e.g. CUB radix sort), rather than the model author. They are always
// AOT-safe and therefore bypass the opt-in, user-facing AOT allowlist
// (--xla_gpu_hlo_custom_call_allowlist).
bool IsInternalAotAllowlistedCustomCall(absl::string_view target_name) {
  static constexpr absl::string_view kInternalAotAllowlist[] = {
      kCubDeviceRadixSortPairsTarget,
      kCubDeviceRadixSortKeysTarget,
  };
  return absl::c_linear_search(kInternalAotAllowlist, target_name);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Context-dependent HLO dispatch
//===----------------------------------------------------------------------===//

Future<ThunkSequence> ThunkEmitter::DispatchAsyncStart(
    const HloInstruction* instr) {
  if (instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    const HloInstruction* wrapped = instr->async_wrapped_instruction();

    // Host send/recv are selected by `is_host_transfer()` and use handler
    // completion events. HostExecute also has its own completion semantics.
    // Neither is nested in generic AsyncStartThunk/AsyncDoneThunk.
    if (auto* send = DynCast<HloSendInstruction>(wrapped);
        send != nullptr && send->is_host_transfer()) {
      return EmitHostSend(send);
    }
    if (auto* recv = DynCast<HloRecvInstruction>(wrapped);
        recv != nullptr && recv->is_host_transfer()) {
      return EmitHostRecv(recv);
    }
    if (auto* call = DynCast<HloCustomCallInstruction>(wrapped);
        call != nullptr && IsHostExecuteCustomCall(*call)) {
      return EmitHostExecuteStart(instr, call);
    }
  }
  return EmitAsyncStart(instr);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchAsyncDone(
    const HloInstruction* instr) {
  // Dispatch legacy typed done instructions first. Generic kAsyncDone
  // instructions are dispatched below according to the wrapped instruction.
  switch (instr->opcode()) {
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectivePermuteDone:
      return EmitAsyncDone(instr, instr->operand(0));
    case HloOpcode::kRecvDone:
      return DispatchRecvDone(Cast<HloRecvDoneInstruction>(instr));
    case HloOpcode::kSendDone:
      return DispatchSendDone(Cast<HloSendDoneInstruction>(instr));
    case HloOpcode::kAsyncDone:
      break;
    default:
      return Internal("Unsupported async done instruction: %s",
                      instr->ToString());
  }

  if (!instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    return EmitAsyncDone(instr, instr->operand(0));
  }

  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  switch (wrapped->opcode()) {
    // Complete a collective wrapped in generic async start/done.
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllGather:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllToAll:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
      return EmitAsyncDone(instr, instr->operand(0));

    // Complete a fusion or call wrapped in generic async start/done.
    case HloOpcode::kFusion:
    case HloOpcode::kCall:
      return EmitAsyncDone(instr, instr->operand(0));

    // Select host or device completion for a wrapped recv.
    case HloOpcode::kRecv: {
      auto* recv = Cast<HloRecvInstruction>(wrapped);
      if (recv->is_host_transfer()) {
        return EmitHostRecvDone(instr, recv);
      }
      return EmitAsyncDone(instr, instr->operand(0));
    }

    // Select host or device completion for a wrapped send.
    case HloOpcode::kSend: {
      auto* send = Cast<HloSendInstruction>(wrapped);
      if (send->is_host_transfer()) {
        return EmitHostSendDone(instr, send);
      }
      return EmitAsyncDone(instr, instr->operand(0));
    }

    // Select host-execute or generic async completion for a wrapped custom
    // call.
    case HloOpcode::kCustomCall: {
      auto* custom_call = Cast<HloCustomCallInstruction>(wrapped);
      if (IsHostExecuteCustomCall(*custom_call)) {
        return EmitHostExecuteDone(instr, custom_call);
      }
      return EmitAsyncDone(instr, instr->operand(0));
    }

    default:
      return Internal("Unsupported async done wrapped instruction: %s",
                      HloOpcodeString(wrapped->opcode()));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchSend(
    const HloSendInstruction* instr) {
  if (instr->is_host_transfer()) {
    return EmitHostSend(instr);
  }

  ABSL_ASSIGN_OR_RETURN(ThunkSequence thunks, EmitSend(instr));
  if (IsImplicitAsyncSendRecvStart(instr)) {
    return EmitAsyncSendRecvStart(instr, std::move(thunks));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchSendDone(
    const HloSendDoneInstruction* instr) {
  return instr->is_host_transfer() ? EmitHostSendDone(instr, instr)
                                   : EmitSendDone(instr);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchRecv(
    const HloRecvInstruction* instr) {
  if (instr->is_host_transfer()) {
    return EmitHostRecv(instr);
  }

  ABSL_ASSIGN_OR_RETURN(ThunkSequence thunks, EmitRecv(instr));
  if (IsImplicitAsyncSendRecvStart(instr)) {
    return EmitAsyncSendRecvStart(instr, std::move(thunks));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchRecvDone(
    const HloRecvDoneInstruction* instr) {
  return instr->is_host_transfer() ? EmitHostRecvDone(instr, instr)
                                   : EmitRecvDone(instr);
}

Future<ThunkSequence> ThunkEmitter::DispatchCustomCall(
    const HloInstruction* hlo) {
  auto* custom_call = Cast<HloCustomCallInstruction>(hlo);

  if (custom_call->custom_call_target() == "zml$print") {
    return EmitMetalPrintThunk(custom_call);
  }
  if (custom_call->custom_call_target() == "zml$flash_attn") {
    return EmitMetalFlashAttnThunk(custom_call);
  }
  if (custom_call->custom_call_target() == "zml$paged_attn") {
    return EmitMetalPagedAttnThunk(custom_call);
  }
  if (custom_call->custom_call_target() == "metal$kv_write") {
    return EmitMetalKvWriteThunk(custom_call);
  }
  if (IsMetalSort(*hlo)) {
    return EmitMetalSortThunk(custom_call);
  }
  if (custom_call->custom_call_target() == "zml$gdn") {
    return EmitMetalGdnThunk(custom_call);
  }
  if (IsMetalScaledMatmul(*hlo)) {
    return EmitMetalScaledMatmulThunk(custom_call);
  }
  if (IsMetalMoeGemmAny(*hlo)) {
    return EmitMoeGemvThunk(custom_call);
  }
  if (IsMetalGemm(*hlo)) {
    return EmitMetalGemmThunk(custom_call);
  }
  if (IsCublasLtMatmul(*hlo)) {
    return EmitCublasLtMatmul(custom_call);
  }
  if (IsCublasLtMatmulF8(*hlo)) {
    return EmitCublasLtMatmulF8(custom_call);
  }
  if (IsCublasLtGroupedMatmul(*hlo)) {
    return EmitCublasLtGroupedMatmul(custom_call);
  }
  if (IsCublasLtMatmulMx(*hlo)) {
    return EmitCublasLtMatmulMx(custom_call);
  }
  if (IsCudnnConvolutionReorder(*hlo)) {
    return EmitConvolutionReorder(custom_call);
  }
  if (IsCustomCallToDnnNorm(*hlo)) {
    return EmitNorm(custom_call);
  }
  if (IsCustomCallTofMHA(*hlo) || IsCustomCallTofMHAF8(*hlo) ||
      IsCustomCallToBlockScaledDot(*hlo)) {
    return EmitCuDnn(custom_call);
  }
  if (IsCustomCallToPtxKernel(*hlo)) {
    return EmitPtxCustomCall(custom_call);
  }
  if (IsCustomCallToTopK(*hlo)) {
    return EmitTopKCustomCall(custom_call);
  }
  if (IsCustomCallToDnnConvolution(*hlo)) {
    return EmitConvolution(custom_call);
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
    return ThunkSequence::Empty();
  }
  if (hlo->custom_call_target() == kPinCustomCallTarget ||
      hlo->custom_call_target() == kUnpinCustomCallTarget ||
      hlo->custom_call_target() == kCreateBufferCustomCallTarget) {
    return ThunkSequence::Empty();
  }
  if (hlo->custom_call_target() == "GetRngSeed") {
    return EmitRngSeed(hlo);
  }
  // Custom calls that have registered a thunk-folding handler are lowered
  // directly to a native ThunkSequence instead of a CustomCallThunk. This is
  // checked last, so the built-in specialized emitters above always take
  // precedence.
  if (std::optional<NativeCustomCallHandlerRef> handler =
          NativeCustomCallHandlerRegistry::GetGlobal().Lookup(
              hlo->custom_call_target());
      handler.has_value()) {
    return EmitNativeCustomCallThunks(custom_call, *handler);
  }
  return EmitGenericCustomCall(custom_call);
}

Future<ThunkSequence> ThunkEmitter::DispatchLegacyCollectiveStart(
    const HloInstruction* instr) {
  const bool is_legacy_collective_start =
      HloPredicateIsOp<HloOpcode::kAllGatherStart, HloOpcode::kAllReduceStart,
                       HloOpcode::kCollectivePermuteStart>(instr);
  TF_RET_CHECK(is_legacy_collective_start);
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<AsyncExecution> execution,
                   RegisterAsyncExecution(instr));
  return EmitCollective(instr).Map(
      [this, instr,
       execution = std::move(execution)](ThunkSequence thunks) mutable {
        return EmitAsyncStart(std::move(execution), instr, std::move(thunks));
      });
}

//===----------------------------------------------------------------------===//
// HLO-specific thunk emission
//===----------------------------------------------------------------------===//

absl::StatusOr<std::shared_ptr<AsyncExecution>>
ThunkEmitter::RegisterAsyncExecution(const HloInstruction* async_start) {
  // Register before starting nested thunk emission: instruction futures are
  // resolved concurrently, so the corresponding async-done may be emitted
  // while the nested-emission future is still pending.
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      async_start, ir_emitter_context_->GetNextThunkId());
  auto execution = std::make_shared<AsyncExecution>(std::move(info));
  auto [_, inserted] = hlo_async_executions_.emplace(async_start, execution);
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    async_start->ToString());
  }
  return execution;
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
  ABSL_ASSIGN_OR_RETURN(auto gpu_config, instr->backend_config<GpuBackendConfig>());

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
  ABSL_ASSIGN_OR_RETURN(DenseDataIntermediate content,
                   LiteralToXlaFormat(instr->literal()));

  int element_bytes =
      primitive_util::ByteWidth(instr->literal().shape().element_type());
  TF_RET_CHECK(content.span().size() % element_bytes == 0);
  // Treat packed constants as a byte constant.
  int num_elements = content.span().size() / element_bytes;

  std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSlice(instr, {}));

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
  if (!should_emit_initializer || platform_name() == "METAL") {
    info.content = content;
  }

  ir_emitter_context_->constants().push_back(std::move(info));
  return ThunkSequence::Empty();
}

Future<ThunkSequence> ThunkEmitter::EmitConditional(
    const HloInstruction* instr) {
  std::vector<Future<ThunkSequence>> branch_thunks;
  branch_thunks.reserve(instr->branch_count());
  for (HloComputation* comp : instr->branch_computations()) {
    branch_thunks.emplace_back(EmitHloComputation(comp));
  }
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSlice(instr->operand(0), {}));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ShapedSlice shaped_slice{slice, instr->operand(0)->shape()};
  return tsl::JoinFutures(absl::MakeSpan(branch_thunks))
      .Map([info = std::move(info), shaped_slice = std::move(shaped_slice)](
               std::vector<ThunkSequence> branch_thunks) mutable {
        return ThunkSequence::Of<ConditionalThunk>(
            std::move(info), std::move(shaped_slice), std::move(branch_thunks));
      });
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
Future<ThunkSequence> ThunkEmitter::EmitPadToStatic(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  ABSL_ASSIGN_OR_RETURN(
      KernelDefinition<LlvmKernelSource> kernel_def,
      EmitPadToStaticLLVMIR(instr, ir_emitter_context_, kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ABSL_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](auto thunk) { return ThunkSequence::Of(std::move(thunk)); });
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
Future<ThunkSequence> ThunkEmitter::EmitSliceToDynamic(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));
  ABSL_ASSIGN_OR_RETURN(
      KernelDefinition<LlvmKernelSource> kernel_def,
      EmitSliceToDynamicLLVMIR(instr, ir_emitter_context_, kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ABSL_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](auto thunk) { return ThunkSequence::Of(std::move(thunk)); });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolution(
    const HloCustomCallInstruction* instr) {
  std::vector<ShapedSlice> operand_slices;
  operand_slices.reserve(instr->operand_count());
  for (const HloInstruction* operand : instr->operands()) {
    ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(operand, {}));
    operand_slices.push_back(slice);
  }

  // The first and the last element in the result tuple for a convolution are
  // always the result and the scratch buffer. It may have auxiliary results in
  // addition to the main result.
  std::vector<ShapedSlice> result_slices;
  for (int i = 0; i < instr->shape().tuple_shapes().size() - 1; i++) {
    ABSL_ASSIGN_OR_RETURN(ShapedSlice result_slice,
                     GetShapedSliceForHlo(instr, {i}));
    result_slices.push_back(result_slice);
  }

  ABSL_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  ABSL_ASSIGN_OR_RETURN(auto gpu_config, instr->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& backend_config =
      gpu_config.cudnn_conv_backend_config();
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice scratch_slice,
      GetAllocationSlice(
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
  ABSL_ASSIGN_OR_RETURN(auto thunk,
                   ConvolutionThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           instr, ir_emitter_context_->GetNextThunkId()),
                       std::move(descriptor), std::move(operand_slices),
                       std::move(result_slices), scratch_slice));
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmul(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  ABSL_ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));
  bool has_matrix_bias = config.beta() != 0;

  TF_RET_CHECK(instr->operand_count() ==
               2 + int{has_matrix_bias} + int{has_vector_bias});

  ABSL_ASSIGN_OR_RETURN(bool has_aux_output,
                   xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));
  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ShapedSlice c;
  if (has_matrix_bias) {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(2)));
  } else {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ABSL_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    ABSL_ASSIGN_OR_RETURN(
        bias, GetShapedSliceForHlo(instr->operand(has_matrix_bias ? 3 : 2)));
  }

  std::optional<ShapedSlice> aux;
  if (has_aux_output) {
    ABSL_ASSIGN_OR_RETURN(aux, GetShapedSliceForHlo(instr, {1}));
  }

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().IsTuple() &&
      (instr->shape().tuple_shapes().size() - has_aux_output - 1)) {
    TF_RET_CHECK(
        (has_aux_output && instr->shape().tuple_shapes().size() == 3) ||
        (!has_aux_output && instr->shape().tuple_shapes().size() == 2));
    ABSL_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  ABSL_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  ABSL_ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  return ThunkSequence::Of<CublasLtMatmulThunk>(
      std::move(info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt, bias, aux, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, workspace_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalGemmThunk(
    const HloCustomCallInstruction* instr) {
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
  if (dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      lhs_shape.dimensions().size() != 2 ||
      rhs_shape.dimensions().size() != 2 ||
      out_shape.dimensions().size() != 2) {
    return fall_back();
  }
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
  const bool trans_a = (lhs_shape.layout().minor_to_major(0) != lc);
  const bool trans_b = (rhs_shape.layout().minor_to_major(0) == rc);

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
                        GetAllocationSlice(fa->operand(fbase), {}));
    num_tokens_shape = fa->operand(fbase)->shape();
    prefill_token_axis = true;
    break;
  }
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
                          GetAllocationSlice(pa->operand(5), {}));
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
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice c,
                      GetAllocationSlice(instr, out_index));

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
    prefill_token_axis = false;
    num_tokens = BufferAllocation::Slice();
    num_tokens_shape = Shape();
  }

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
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalPrintThunk(
    const HloCustomCallInstruction* instr) {
  const HloInstruction* operand = instr->operand(0);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSlice(operand, {}));
  std::string label = instr->metadata().op_name().empty()
                          ? std::string(instr->name())
                          : instr->metadata().op_name();
  auto thunk = std::make_unique<MetalPrintThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(label), slice, operand->shape());
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalFlashAttnThunk(
    const HloCustomCallInstruction* instr) {
  const int operand_count = instr->operand_count();
  if (operand_count < 4 || operand_count > 6) {
    return absl::InvalidArgumentError(
        "zml$flash_attn expects 4 (q,k,v,tok), 5 (+layer or +num_tokens), or 6 "
        "(+layer +num_tokens) operands.");
  }
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
  if (q_shape.element_type() != BF16 || k_shape.element_type() != BF16 ||
      v_shape.element_type() != BF16 || out_shape.element_type() != BF16) {
    return absl::UnimplementedError(
        "zml$flash_attn: q/k/v/out must be bf16 (the Metal flash-attention "
        "kernels are bf16-typed).");
  }
  if (q_len > 1 && ((hd != 128 && hd != 64) || seqlen % 64 != 0)) {
    return absl::UnimplementedError(
        "zml$flash_attn: prefill (q_len>1) needs head_dim 64 or 128 and "
        "seqlen%64==0.");
  }
  const int64_t n_groups = n_q / n_kv;

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice q,
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v,
                      GetAllocationSlice(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice tok,
                      GetAllocationSlice(instr->operand(3), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out,
                      GetAllocationSlice(instr, {}));

  BufferAllocation::Slice layer;
  Shape layer_shape;
  if (kv_full_cache) {
    TF_ASSIGN_OR_RETURN(layer, GetAllocationSlice(instr->operand(4), {}));
    layer_shape = instr->operand(4)->shape();
  }

  BufferAllocation::Slice num_tokens;
  Shape num_tokens_shape;
  if (has_num_tokens) {
    TF_ASSIGN_OR_RETURN(
        num_tokens, GetAllocationSlice(instr->operand(base_operands), {}));
    num_tokens_shape = instr->operand(base_operands)->shape();
  }

  // The thunk may read tok host-side only when it is an entry parameter; a
  // GPU-produced value would race its producer.
  const HloInstruction* tok_src = instr->operand(3);
  while (tok_src->opcode() == HloOpcode::kBitcast ||
         tok_src->opcode() == HloOpcode::kReshape) {
    tok_src = tok_src->operand(0);
  }
  const bool tok_host_coherent = tok_src->opcode() == HloOpcode::kParameter;

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
  return ThunkSequence::Of(std::move(thunk));
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
    if (const ffi::Scalar* s = scalar("is_causal")) {
      if (const bool* v = std::get_if<bool>(&s->AsVariant())) is_causal = *v;
    }
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice q,
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k_cache,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v_cache,
                      GetAllocationSlice(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice block_table,
                      GetAllocationSlice(instr->operand(3), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice seq_lens,
                      GetAllocationSlice(instr->operand(4), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice query_start_len,
                      GetAllocationSlice(instr->operand(5), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out,
                      GetAllocationSlice(instr, {}));

  auto thunk = std::make_unique<MetalPagedAttnThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      q, q_shape, k_cache, k_shape, v_cache, v_shape, block_table, bt_shape,
      seq_lens, sl_shape, query_start_len, qsl_shape, out, out_shape, num_heads,
      num_kv_heads, head_dim, block_size, num_seqs, max_num_blocks_per_seq,
      total_q_tokens, scale, softcapping, sliding_window, is_causal,
      q_shape.element_type());
  return ThunkSequence::Of(std::move(thunk));
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

  if (!instr->shape().IsTuple() || instr->shape().tuple_shapes().size() != 2) {
    return absl::UnimplementedError(
        "zml$gdn: expected a (y, ht) tuple result.");
  }
  const Shape& y_shape = instr->shape().tuple_shapes(0);
  const Shape& ht_shape = instr->shape().tuple_shapes(1);

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
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v,
                      GetAllocationSlice(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice g,
                      GetAllocationSlice(instr->operand(3), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice beta,
                      GetAllocationSlice(instr->operand(4), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice h0,
                      GetAllocationSlice(instr->operand(5), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice cu_seqlens,
                      GetAllocationSlice(instr->operand(6), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slot_mapping,
                      GetAllocationSlice(instr->operand(7), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice y,
                      GetAllocationSlice(instr, {0}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice ht,
                      GetAllocationSlice(instr, {1}));

  auto thunk = std::make_unique<MetalGdnThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      q, q_shape, k, k_shape, v, v_shape, g, g_shape, beta, beta_shape, h0,
      h0_shape, cu_seqlens, cu_seqlens_shape, slot_mapping, slot_mapping_shape,
      y, y_shape, ht, ht_shape, num_seqs, hk, hv, dk, dv, et);
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalScaledMatmulThunk(
    const HloCustomCallInstruction* instr) {
  if (instr->operand_count() < 3) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul expects at least 3 operands (x, w, scale).");
  }
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
    case MetalScaledMatmulScheme::kMxfp8Group32:
    case MetalScaledMatmulScheme::kMxfp4Group32:
      return EmitMetalMxMatmulThunk(instr);
    case MetalScaledMatmulScheme::kFp8Block128:
    case MetalScaledMatmulScheme::kFp8PerChannel:
      return EmitMetalFp8GemvThunk(instr, *scheme);
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalMxMatmulThunk(
    const HloCustomCallInstruction* instr) {
  if (instr->operand_count() != 3) {
    return absl::InvalidArgumentError(
        "zml$scaled_matmul (MX) expects 3 operands (x, w, scales).");
  }
  const Shape& x_shape = instr->operand(0)->shape();
  const Shape& w_shape = instr->operand(1)->shape();
  const Shape& scales_shape = instr->operand(2)->shape();

  const bool is_tuple = instr->shape().IsTuple();
  const Shape& out_shape =
      is_tuple ? instr->shape().tuple_shapes(0) : instr->shape();

  if (x_shape.dimensions().size() != 2 || w_shape.dimensions().size() != 2 ||
      scales_shape.dimensions().size() != 2 ||
      out_shape.dimensions().size() != 2) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (MX): x, w, scales, out must all be rank 2.");
  }

  const int64_t m = x_shape.dimensions(0);
  const int64_t k = x_shape.dimensions(1);
  const int64_t n = w_shape.dimensions(0);
  const int64_t w_cols = w_shape.dimensions(1);
  const int64_t scale_cols = scales_shape.dimensions(1);

  if (m == 0 || k == 0 || n == 0 || w_cols == 0 || scale_cols == 0) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (MX): invalid dimension (must be > 0).");
  }
  int64_t bits;
  const PrimitiveType wt = w_shape.element_type();
  const PrimitiveType st = scales_shape.element_type();
  if (wt == F8E4M3FN || wt == F4E2M1FN) {
    bits = (wt == F8E4M3FN) ? 8 : 4;
    if (w_cols != k) {
      return absl::UnimplementedError(
          "zml$scaled_matmul (MX): native-f8 weight must be [N, K] (K minor).");
    }
    if (st != F8E8M0FNU) {
      return absl::UnimplementedError(
          "zml$scaled_matmul (MX): native-f8 weight needs f8e8m0 scales.");
    }
  } else if (wt == U32) {
    if (st != U8) {
      return absl::UnimplementedError(
          "zml$scaled_matmul (MX): u32-packed weight needs u8 scales.");
    }
    if (k % w_cols != 0) {
      return absl::UnimplementedError(
          "zml$scaled_matmul (MX): K must be a multiple of the packed w minor.");
    }
    const int64_t pack = k / w_cols;  // values per uint32 word
    bits = 32 / pack;
    if ((bits != 8 && bits != 4) || pack * bits != 32) {
      return absl::UnimplementedError(absl::StrCat(
          "zml$scaled_matmul (MX): unsupported packing (K=", k,
          ", w_cols=", w_cols, " -> bits=", bits, ")."));
    }
  } else {
    return absl::UnimplementedError(
        "zml$scaled_matmul (MX): w must be f8e4m3fn / f4e2m1 / packed u32.");
  }
  if (k % scale_cols != 0) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (MX): K must be a multiple of the scale minor dim.");
  }
  const int64_t group_size = k / scale_cols;  // 32 for MX
  if (group_size != 32) {
    return absl::UnimplementedError(absl::StrCat(
        "zml$scaled_matmul (MX): only 32-element MX groups supported (got ",
        group_size, ")."));
  }
  if (out_shape.dimensions(0) != m || out_shape.dimensions(1) != n) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (MX): out shape must be [M, N].");
  }
  if (x_shape.element_type() != BF16 || out_shape.element_type() != BF16) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (MX): x and out must be bf16.");
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice x,
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice w,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scales,
                      GetAllocationSlice(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice out,
      GetAllocationSlice(instr, is_tuple ? ShapeIndex{0} : ShapeIndex{}));

  auto thunk = std::make_unique<MetalMxMatmulThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      x, x_shape, w, w_shape, scales, scales_shape, out, out_shape, m, k, n,
      bits, group_size);
  return ThunkSequence::Of(std::move(thunk));
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
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice w,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scale,
                      GetAllocationSlice(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice out,
      GetAllocationSlice(instr, is_tuple ? ShapeIndex{0} : ShapeIndex{}));
  BufferAllocation::Slice workspace;
  if (is_tuple) {
    TF_ASSIGN_OR_RETURN(workspace,
                        GetAllocationSlice(instr, ShapeIndex{1}));
  }

  auto thunk = std::make_unique<MetalNvfp4MatmulThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      x, x_shape, w, w_shape, scale, scale_shape, out, out_shape, workspace,
      workspace_shape, m, k, n, metal_arch.architecture_size(),
      metal_arch.architecture_gen());
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalFp8GemvThunk(
    const HloCustomCallInstruction* instr, MetalScaledMatmulScheme scheme) {
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
  const bool per_channel = scheme == MetalScaledMatmulScheme::kFp8PerChannel;
  if (x_shape.element_type() != BF16 || out_shape.element_type() != BF16) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (FP8): x and out must be bf16.");
  }
  if (out_shape.dimensions(0) != b || out_shape.dimensions(1) != n) {
    return absl::UnimplementedError(
        "zml$scaled_matmul (FP8): out must be [B, N].");
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice x,
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice w,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scale,
                      GetAllocationSlice(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice out,
      GetAllocationSlice(instr, is_tuple ? ShapeIndex{0} : ShapeIndex{}));

  auto thunk = std::make_unique<MetalFp8GemvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      x, x_shape, w, w_shape, scale, scale_shape, out, out_shape, b, k, n,
      per_channel);
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMoeGemvThunk(
    const HloCustomCallInstruction* instr) {
  const bool is_fp8 =
      instr->custom_call_target() == kMetalMoeGemmF8CallTarget;
  const bool is_nvfp4 =
      instr->custom_call_target() == kMetalMoeGemmF4CallTarget;
  const bool has_scale = is_fp8 || is_nvfp4;
  const int64_t expected_operands = has_scale ? 4 : 3;
  if (instr->operand_count() != expected_operands) {
    return absl::InvalidArgumentError(absl::StrCat(
        "metal MoE GEMV expects ", expected_operands, " operands."));
  }
  const Shape& x_shape = instr->operand(0)->shape();
  const Shape& w_shape = instr->operand(1)->shape();
  const int expert_id_idx = has_scale ? 3 : 2;
  const Shape& expert_id_shape = instr->operand(expert_id_idx)->shape();

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
  if (is_nvfp4 && k % 16 != 0) {
    return absl::UnimplementedError(
        "metal MoE GEMV: K must be multiple of 16 "
        "(nvfp4 group-16; N supports scalar tails).");
  }
  const int64_t k_block = is_fp8 ? 128 : 4;
  const int64_t n_block = is_fp8 ? 128 : 4;
  if (!is_nvfp4 && (k % k_block != 0 || n % n_block != 0)) {
    return absl::UnimplementedError(absl::StrCat(
        "metal MoE GEMV: K must be multiple of ", k_block, ", N of ", n_block,
        is_fp8 ? " (fp8 block-scale)." : " (bf16 vectorized load)."));
  }
  const PrimitiveType expected_w =
      is_fp8 ? F8E4M3FN : (is_nvfp4 ? F4E2M1FN : BF16);
  if (w_shape.element_type() != expected_w) {
    return absl::UnimplementedError(absl::StrCat(
        "metal MoE GEMV: w must be ",
        is_fp8     ? "f8e4m3fn"
        : is_nvfp4 ? "f4e2m1"
                   : "bf16",
        "."));
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

  BufferAllocation::Slice scale;
  Shape scale_shape;
  if (is_fp8) {
    scale_shape = instr->operand(2)->shape();
    if (scale_shape.dimensions().size() != 3 ||
        scale_shape.dimensions(0) != e ||
        scale_shape.dimensions(1) != n / 128 ||
        scale_shape.dimensions(2) != k / 128) {
      return absl::UnimplementedError(
          "metal MoE GEMV: fp8 scale must be [E, N/128, K/128].");
    }
    if (scale_shape.element_type() != BF16) {
      return absl::UnimplementedError(
          "metal MoE GEMV: fp8 scale must be bf16.");
    }
    TF_ASSIGN_OR_RETURN(scale,
                        GetAllocationSlice(instr->operand(2), {}));
  } else if (is_nvfp4) {
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
                        GetAllocationSlice(instr->operand(2), {}));
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice x,
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice w,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice expert_id,
      GetAllocationSlice(instr->operand(expert_id_idx), {}));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice out,
      GetAllocationSlice(instr, is_tuple ? ShapeIndex{0} : ShapeIndex{}));
  BufferAllocation::Slice workspace;
  if (is_tuple) {
    TF_ASSIGN_OR_RETURN(workspace,
                        GetAllocationSlice(instr, ShapeIndex{1}));
  }

  auto thunk = std::make_unique<MetalMoeGemvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      x, x_shape, w, w_shape, scale, scale_shape, expert_id, expert_id_shape,
      out, out_shape, workspace, workspace_shape, r, k, n);
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalKvWriteThunk(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 7);
  const Shape& k_cache_shape = instr->operand(0)->shape();
  TF_RET_CHECK(k_cache_shape.dimensions().size() == 4);
  TF_RET_CHECK(k_cache_shape.element_type() == BF16);
  const int64_t num_slots =
      k_cache_shape.dimensions(0) * k_cache_shape.dimensions(1);
  const int64_t kv_heads = k_cache_shape.dimensions(2);
  const int64_t head_dim = k_cache_shape.dimensions(3);

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k_cache,
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice k_new,
                      GetAllocationSlice(instr->operand(1), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v_cache,
                      GetAllocationSlice(instr->operand(2), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice v_new,
                      GetAllocationSlice(instr->operand(3), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slot,
                      GetAllocationSlice(instr->operand(4), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice pos,
                      GetAllocationSlice(instr->operand(5), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice freq,
                      GetAllocationSlice(instr->operand(6), {}));

  auto thunk = std::make_unique<MetalKvWriteThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      k_cache, instr->operand(0)->shape(), k_new, instr->operand(1)->shape(),
      v_cache, instr->operand(2)->shape(), v_new, instr->operand(3)->shape(),
      slot, instr->operand(4)->shape(), pos, instr->operand(5)->shape(), freq,
      instr->operand(6)->shape(), num_slots, kv_heads, head_dim);
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitMetalSortThunk(
    const HloCustomCallInstruction* instr) {
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
                      GetAllocationSlice(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out_vals,
                      GetAllocationSlice(instr, {0}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out_idxs,
                      GetAllocationSlice(instr, {1}));

  auto thunk = std::make_unique<MetalSortThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      data, out_vals, out_idxs, dtype, rows, n, descending);
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulF8(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() > 3 && instr->operand_count() < 8);
  ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  ABSL_ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  ABSL_ASSIGN_OR_RETURN(bool has_aux_output,
                   xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ShapedSlice c;
  bool has_matrix_bias = config.beta() != 0;
  if (has_matrix_bias) {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(2)));
  } else {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ABSL_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  int a_scale_index = has_matrix_bias ? 3 : 2;
  ABSL_ASSIGN_OR_RETURN(ShapedSlice a_scale,
                   GetShapedSliceForHlo(instr->operand(a_scale_index)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b_scale,
                   GetShapedSliceForHlo(instr->operand(a_scale_index + 1)));

  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();
  bool is_fp8 = instr->shape().tuple_shapes(0).element_type() == F8E4M3FN ||
                instr->shape().tuple_shapes(0).element_type() == F8E5M2;
  // cublasLT requires c_scale/d_scale to be null when C/D is not
  // FP8. Currently, C cannot be FP8.
  std::optional<ShapedSlice> d_scale;
  if (is_cuda && is_fp8) {
    ABSL_ASSIGN_OR_RETURN(d_scale, GetShapedSliceForHlo(instr->operands().back()));
  }

  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    ABSL_ASSIGN_OR_RETURN(bias,
                     GetShapedSliceForHlo(instr->operand(a_scale_index + 2)));
  }

  std::optional<ShapedSlice> d_amax;
  if (config.damax_output()) {
    ABSL_ASSIGN_OR_RETURN(d_amax, GetShapedSliceForHlo(instr, {1}));
  }

  ABSL_ASSIGN_OR_RETURN(
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
    ABSL_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ABSL_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ABSL_ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  return ThunkSequence::Of<CublasLtMatmulThunk>(
      std::move(info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt, bias, std::nullopt, a_scale, b_scale,
      std::nullopt, d_scale, d_amax, workspace_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtGroupedMatmul(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config =
      gpu_config.grouped_gemm_backend_config().gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  // Matrix bias and vector bias add extra operands
  bool has_matrix_bias = config.beta() != 0;
  ABSL_ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));
  TF_RET_CHECK(instr->operand_count() ==
               3 + int{has_matrix_bias} + int{has_vector_bias});

  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice group_sizes,
                   GetShapedSliceForHlo(instr->operand(2)));

  // Handle matrix bias if present
  ShapedSlice c;
  if (has_matrix_bias) {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(3)));
  } else {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ABSL_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  // Handle vector bias if present
  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    int bias_operand_index = has_matrix_bias ? 4 : 3;
    ABSL_ASSIGN_OR_RETURN(bias,
                     GetShapedSliceForHlo(instr->operand(bias_operand_index)));
  }

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().IsTuple() && (instr->shape().tuple_shapes().size() - 1)) {
    ABSL_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }
  ABSL_ASSIGN_OR_RETURN(
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
  ABSL_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ABSL_ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));

  return ThunkSequence::Of<CublasLtMatmulThunk>(
      std::move(info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      std::move(group_sizes), bias, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, workspace_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulMx(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 4);
  ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice a_scale,
                   GetShapedSliceForHlo(instr->operand(2)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b_scale,
                   GetShapedSliceForHlo(instr->operand(3)));

  ABSL_ASSIGN_OR_RETURN(ShapedSlice c, GetShapedSliceForHlo(instr, output_index));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  ABSL_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().tuple_shapes().size() == 2) {
    ABSL_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ABSL_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ABSL_ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  return ThunkSequence::Of<CublasLtMatmulThunk>(
      std::move(info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt,
      /*bias=*/std::nullopt, /*aux=*/std::nullopt, a_scale, b_scale,
      /*c_scale=*/std::nullopt, /*d_scale=*/std::nullopt,
      /*d_amax=*/std::nullopt, workspace_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolutionReorder(
    const HloCustomCallInstruction* instr) {
  bool has_bias = instr->operand_count() > 1;

  ABSL_ASSIGN_OR_RETURN(ShapedSlice filter_input,
                   GetShapedSliceForHlo(instr->operand(0)));

  ShapedSlice filter_output;
  std::optional<ConvolutionReorderThunk::BiasBuffers> biases;
  if (has_bias) {
    ABSL_ASSIGN_OR_RETURN(filter_output, GetShapedSliceForHlo(instr, {0}));

    ABSL_ASSIGN_OR_RETURN(ShapedSlice bias_input,
                     GetShapedSliceForHlo(instr->operand(1)));
    ABSL_ASSIGN_OR_RETURN(ShapedSlice bias_output, GetShapedSliceForHlo(instr, {1}));
    biases = {{bias_input, bias_output}};
  } else {
    ABSL_ASSIGN_OR_RETURN(filter_output, GetShapedSliceForHlo(instr));
  }

  ABSL_ASSIGN_OR_RETURN(auto thunk,
                   ConvolutionReorderThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           instr, ir_emitter_context_->GetNextThunkId()),
                       filter_input, filter_output, biases));
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNorm(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto const gpu_backend_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::CudnnNormBackendConfig& backend_config =
      gpu_backend_config.cudnn_norm_backend_config();

  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice x_slice,
                   GetAllocationSlice(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice scale_slice,
                   GetAllocationSlice(instr->operand(1)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice y_or_dx_slice,
                   GetAllocationSlice(instr, {0}));

  std::optional<BufferAllocation::Slice> bias_slice, expectation_slice,
      norm_factor_slice, dy_slice, dscale_slice, dbias_slice;

  if (backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER ||
      backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    ABSL_ASSIGN_OR_RETURN(bias_slice, GetAllocationSlice(instr->operand(2)));
  }
  if (backend_config.kind() ==
      xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    ABSL_ASSIGN_OR_RETURN(expectation_slice, GetAllocationSlice(instr, {1}));
    ABSL_ASSIGN_OR_RETURN(norm_factor_slice, GetAllocationSlice(instr, {2}));
  }
  if (backend_config.kind() == xla::gpu::CudnnNormBackendConfig::LAYER_BWD) {
    ABSL_ASSIGN_OR_RETURN(dy_slice, GetAllocationSlice(instr->operand(2)));
    ABSL_ASSIGN_OR_RETURN(expectation_slice, GetAllocationSlice(instr->operand(3)));
    ABSL_ASSIGN_OR_RETURN(norm_factor_slice, GetAllocationSlice(instr->operand(4)));
    ABSL_ASSIGN_OR_RETURN(dscale_slice, GetAllocationSlice(instr, {1}));
    ABSL_ASSIGN_OR_RETURN(dbias_slice, GetAllocationSlice(instr, {2}));
  }
  ABSL_ASSIGN_OR_RETURN(
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

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<NormThunk> thunk,
      NormThunk::Create(Thunk::ThunkInfo::WithProfileAnnotation(
                            instr, ir_emitter_context_->GetNextThunkId()),
                        std::move(descriptor), x_slice, scale_slice,
                        y_or_dx_slice, bias_slice, expectation_slice,
                        norm_factor_slice, dy_slice, dscale_slice, dbias_slice,
                        scratch_slice.slice));
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCuDnn(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));
  ABSL_ASSIGN_OR_RETURN(const std::string fingerprint,
                   FingerprintWithBackendConfig<GpuBackendConfig>(*instr));
  // check if sdpa dropout is enabled
  std::optional<int64_t> dropout_seed = std::nullopt;
  if (MHACallHasDropout(instr->custom_call_target())) {
    ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                     instr->backend_config<xla::gpu::GpuBackendConfig>());
    dropout_seed = gpu_config.cudnn_fmha_backend_config().seed();
  }
  return ThunkSequence::Of<CuDnnThunk>(
      fingerprint,
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      kernel_arguments.GetArgumentShapedSlices(),
      kernel_arguments.GetArgumentOutputFlags(),
      /*should_memzero=*/IsCustomCallTofMHA(*instr), dropout_seed);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitPtxCustomCall(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto thunk,
                   EmitPtxCustomKernelThunk(instr, ir_emitter_context_));
  return ThunkSequence::Of(std::move(thunk));
}

std::optional<BufferAllocation::Slice> ThunkEmitter::GetAllocationOverride(
    const HloInstruction* instr, const ShapeIndex& index) const {
  auto it = allocation_overrides_.find(instr);
  if (it == allocation_overrides_.end()) {
    return std::nullopt;
  }

  int64_t flat_idx = index.empty() ? 0 : index[0];
  if (flat_idx >= 0 && static_cast<size_t>(flat_idx) < it->second.size()) {
    return it->second[static_cast<size_t>(flat_idx)];
  }

  return std::nullopt;
}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSlice(
    const HloInstruction* instr, const ShapeIndex& index) const {
  if (std::optional<BufferAllocation::Slice> slice =
          GetAllocationOverride(instr, index)) {
    return *slice;
  }

  return ir_emitter_context_->buffer_assignment().GetUniqueSlice(instr, index);
}

absl::StatusOr<ShapedSlice> ThunkEmitter::GetShapedSliceForHlo(
    const HloInstruction* instr, const ShapeIndex& index) const {
  if (std::optional<BufferAllocation::Slice> slice =
          GetAllocationOverride(instr, index)) {
    return ShapedSlice{*slice, ShapeUtil::GetSubshape(instr->shape(), index)};
  }

  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSlice(instr, index));
  ABSL_ASSIGN_OR_RETURN(
      Shape shape,
      ir_emitter_context_->buffer_assignment().GetShapeForUniqueSlice(instr,
                                                                      index));
  return ShapedSlice{slice, shape};
}

class NativeCustomCallEmitterContextImpl
    : public NativeCustomCallEmitterContext {
 public:
  NativeCustomCallEmitterContextImpl(const ThunkEmitter* emitter,
                                     const HloCustomCallInstruction* instr)
      : emitter_(*emitter), instr_(*instr) {}

  const GpuTopology& GetTargetTopology() const override {
    return emitter_.ir_emitter_context_->gpu_topology();
  }

  const DebugOptions& GetDebugOptions() const override {
    return emitter_.ir_emitter_context_->debug_options();
  }

  Thunk::ThunkInfo GenerateThunkInfo() const override {
    return Thunk::ThunkInfo::WithProfileAnnotation(
        &instr_, emitter_.ir_emitter_context_->GetNextThunkId());
  }

  absl::StatusOr<BufferAllocation::Slice> GetResultAllocationSlice(
      const ShapeIndex& index) const override {
    return emitter_.GetAllocationSlice(&instr_, index);
  }

  absl::StatusOr<BufferAllocation::Slice> GetOperandAllocationSlice(
      int64_t operand_index, const ShapeIndex& index) const override {
    TF_RET_CHECK(operand_index >= 0 && operand_index < instr_.operand_count());
    return emitter_.GetAllocationSlice(instr_.operand(operand_index), index);
  }

  absl::StatusOr<xla::ffi::AttributesMap> GetFfiAttributes() const override {
    // Decode the opaque backend config into an FFI attributes map, mirroring
    // EmitGenericCustomCall. For FFI handlers the backend config must be a
    // string parsable into an MLIR dictionary attribute.
    absl::StatusOr<GpuBackendConfig> backend_config =
        instr_.backend_config<GpuBackendConfig>();
    const std::string& backend_config_str =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().attributes()
            : instr_.raw_backend_config_string();
    if (backend_config_str.empty()) {
      return xla::ffi::AttributesMap();
    }
    mlir::Attribute attr = mlir::parseAttribute(
        backend_config_str, emitter_.ir_emitter_context_->mlir_context());
    auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
    TF_RET_CHECK(dict != nullptr)
        << "Unsupported backend config. Expected a string parsable into a "
           "dictionary attribute.";
    return xla::ffi::BuildAttributesMap(dict);
  }

 private:
  const ThunkEmitter& emitter_;
  const HloCustomCallInstruction& instr_;
};

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNativeCustomCallThunks(
    const HloCustomCallInstruction* instr, NativeCustomCallHandlerRef handler) {
  NativeCustomCallEmitterContextImpl ctx(this, instr);
  return handler(*instr, ctx);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitGenericCustomCall(
    const HloCustomCallInstruction* instr) {
  const std::string& call_target_name = instr->custom_call_target();

  // Typed FFI custom calls is a replacement for legacy custom calls
  // with a rich type safe API.
  bool is_ffi_custom_call =
      instr->api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  using Slices = std::vector<NullableShapedSlice>;

  Slices operands;
  for (auto* operand : instr->operands()) {
    ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            operands.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          ABSL_ASSIGN_OR_RETURN(auto slice, GetAllocationSlice(operand, index));
          operands.push_back(ShapedSlice{slice, subshape});
          return absl::OkStatus();
        }));
  }

  Slices results;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          results.push_back(std::nullopt);
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        ABSL_ASSIGN_OR_RETURN(auto slice, GetAllocationSlice(instr, index));
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
    // Enforce the opt-in AOT custom-call allowlist. An empty allowlist
    // disables the check. This only gates FFI custom calls that lower to a
    // CustomCallThunk; legacy custom calls and custom kernels are unaffected.
    // XLA-internal FFI targets are always permitted.
    const auto& custom_call_allowlist =
        ir_emitter_context_->debug_options()
            .xla_gpu_hlo_custom_call_allowlist();
    if (!custom_call_allowlist.empty() &&
        !IsInternalAotAllowlistedCustomCall(call_target_name) &&
        !absl::c_linear_search(custom_call_allowlist, call_target_name)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Custom call target '", call_target_name,
                       "' is not in the allowlist "
                       "(--xla_gpu_hlo_custom_call_allowlist). "));
    }
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
      ABSL_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    }
    const bool enable_pdl =
        IsPdlEnabled(ir_emitter_context_->debug_options(),
                     ir_emitter_context_->gpu_compute_capability());
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
        ir_emitter_context_->cpu_target_machine_options(), enable_pdl);
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

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFft(
    const HloFftInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                   GetAllocationSlice(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                   GetAllocationSlice(instr));
  return ThunkSequence::Of<FftThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr->fft_type(), instr->fft_length(),
      /*input_buffer=*/arg_slice,
      /*output_buffer=*/dest_slice,
      /*input_shape=*/instr->operand(0)->shape(),
      /*output_shape=*/instr->shape());
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

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a_slice, GetShapedSliceForHlo(operands[0]));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b_slice, GetShapedSliceForHlo(operands[1]));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice result_slice, GetShapedSliceForHlo(instr, {0}));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice temp_slice, GetShapedSliceForHlo(instr, {1}));

  TriangularSolveOptions backend_config;
  auto& backend_config_str = instr->raw_backend_config_string();
  if (!backend_config_str.empty()) {
    ABSL_RETURN_IF_ERROR(
        tsl::HumanReadableJsonToProto(backend_config_str, &backend_config));
  }

  ThunkSequence thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output
  // if they aren't the same buffer.
  if (b_slice.slice != result_slice.slice) {
    thunks.Emplace<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        /*source_buffer=*/b_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(b_slice.shape));
  }

  thunks.Emplace<TriangularSolveThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      backend_config, a_slice, result_slice, temp_slice);

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    return thunks;
  }
  auto info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  // Don't repeat the annotation from inside thunks
  info.profile_annotation = {};
  return ThunkSequence::Of<SequentialThunk>(info, std::move(thunks));
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
  ABSL_ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  auto dtype = data_shape.element_type();
  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();

  // Enable RAFT if TopK is_stable = false.
  bool use_raft = !hlo_instruction_utils::IsTopKStable(instr);

  if (is_cuda && use_raft) {
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
    } else if (dtype == PrimitiveType::U64) {
      use_raft_select_k = true;
    }

    VLOG(3) << "EmitTopKCustomCall: dtype=" << dtype << ", n=" << n
            << ", k=" << k << ", use_raft_select_k=" << use_raft_select_k;

    Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
        instr, ir_emitter_context_->GetNextThunkId());
    if (use_raft_select_k) {
      return ThunkSequence::Of<SelectKThunk>(std::move(info), batch_size, n, k,
                                             dtype, kernel_arguments);
    }
  }

  if (platform_name() == "METAL") {
    TF_RET_CHECK(k <= 64)
        << "Metal TopK: k=" << k
        << " > 64 — the radix select kernels template k in "
           "{1,2,4,8,16,32,64}; the TopkSpecializer should have capped k at 64.";
    TF_RET_CHECK(dtype == BF16 || dtype == F16 || dtype == F32)
        << "Metal TopK: dtype " << PrimitiveType_Name(dtype)
        << " unsupported — the radix sortable-bit transform handles bf16/f16/f32.";
    const auto& args = kernel_arguments.args();
    TF_RET_CHECK(args.size() == 3)
        << "Metal TopK expects 3 buffers (data, vals, idxs).";
    return ThunkSequence::Of<MetalTopKThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        args[0].slice(), args[1].slice(), args[2].slice(), dtype, batch_size, n,
        k);
  }

  auto wavefront_size =
      ir_emitter_context_->gpu_device_info().threads_per_warp();

  TF_RET_CHECK(k <= 16) << "CustomCall TopK requires k <= 16";
  // Load TopK custom kernel.
  ABSL_ASSIGN_OR_RETURN(CustomKernel kernel, kernel::topk::GetTopKKernel(
                                            "topk", dtype, n, k, batch_size,
                                            platform_name(), wavefront_size));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return ThunkSequence::Of<CustomKernelThunk>(
      std::move(info), std::move(kernel), kernel_arguments);
}

Future<ThunkSequence> ThunkEmitter::EmitTritonCustomCall(
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

    ABSL_ASSIGN_OR_RETURN(TritonKernelSource triton_source,
                     EmitTritonFrom(call, kernel_name, **borrowed_context));

    HloModule* hlo_module = instr->GetModule();

    BlockLevelParameters block_level_parameters;
    block_level_parameters.num_stages = call.num_stages;
    block_level_parameters.num_warps = call.num_warps;
    block_level_parameters.num_ctas = 1;
    block_level_parameters.global_scratch_memory_size =
        call.global_scratch_memory_size;
    block_level_parameters.is_tma_allowed = call.is_tma_allowed;
    block_level_parameters.waves_per_eu = call.waves_per_eu;

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
              &gpu_device_info = ir_emitter_context_->gpu_device_info()](
                 TritonWrapperResult result) mutable
                 -> xla::Future<KernelReuseCache::Entry> {
          auto local_module =
              std::move(result.kernel_source).thread_safe_module();

          ABSL_ASSIGN_OR_RETURN(
              auto kernel_arguments,
              emitters::KernelArguments::Create(
                  *buffer_assignment, GetDefaultBufferAlignment(), instr));
          auto launch_dimensions = LaunchDimensions(
              se::BlockDim(call.grid_x, call.grid_y, call.grid_z),
              se::ThreadDim(call.num_warps *
                            gpu_device_info.threads_per_warp()));

          ABSL_ASSIGN_OR_RETURN(
              llvm::Function * kernel,
              RemoveUnusedTritonAbiArguments(
                  local_module.getModuleUnlocked(), kernel_name,
                  kernel_impl_name, call.global_scratch_memory_size > 0));

          AnnotateAttrsIfUnset(kernel_arguments, *kernel);
          ABSL_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
              gpu_device_info, launch_dimensions, kernel,
              local_module.getModuleUnlocked()));

          return kernel_compiler
              ->CompileToTargetBinary(LlvmKernelSource{std::move(local_module)})
              .Map([use_pdl = result.use_pdl, shmem_bytes = result.shmem_bytes,
                    launch_dimensions = std::move(launch_dimensions),
                    tma_metadata = result.tma_metadata,
                    kernel_name = std::move(kernel_name)](
                       const std::vector<uint8_t>& cubin) mutable {
                return KernelReuseCache::Entry{std::move(kernel_name),
                                               launch_dimensions,
                                               /*cluster_dim=*/std::nullopt,
                                               shmem_bytes,
                                               cubin,
                                               tma_metadata,
                                               use_pdl};
              });
        });
  };

  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  auto [status_or_entry, was_cached] =
      ir_emitter_context_->kernel_cache().GetWithStatus(
          instr->raw_backend_config_string(), generate);

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return status_or_entry.Map(
      [info = std::move(info), kernel_arguments = std::move(kernel_arguments),
       call_zeroed_outputs = std::move(call_zeroed_outputs)](
          const KernelReuseCache::Entry* entry) mutable
          -> absl::StatusOr<ThunkSequence> {
        ABSL_ASSIGN_OR_RETURN(CustomKernel custom_kernel,
                         kernel::CreateOwnedCubinCustomKernel(
                             entry->kernel_name, entry->binary,
                             kernel_arguments.args().size(),
                             entry->launch_dimensions.block_counts(),
                             entry->launch_dimensions.thread_counts_per_block(),
                             entry->shmem_bytes));
        return ThunkSequence::Of<CustomKernelThunk>(
            std::move(info), std::move(custom_kernel),
            std::move(kernel_arguments), entry->use_pdl, call_zeroed_outputs,
            entry->tma_metadata);
      });
}

Future<ThunkSequence> ThunkEmitter::EmitDynamicSliceCopyFusion(
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

  ThunkSequence embedded_thunks = ThunkSequence::Of<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      ShapedSlice{src_slice, copy_shape},
      ShapedSlice{dst_slice, copy.results[0].update_shape}, byte_size);

  std::vector<BufferAllocation::Slice> parameter_buffers;
  parameter_buffers.reserve(instr->operand_count());
  for (const auto* operand : instr->operands()) {
    ABSL_ASSIGN_OR_RETURN(parameter_buffers.emplace_back(),
                     GetAllocationSlice(operand));
  }

  std::vector<BufferAllocation::Slice> result_buffers;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      instr->shape(),
      [&](const Shape&, const ShapeIndex& index) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(result_buffers.emplace_back(),
                         GetAllocationSlice(instr, index));
        return absl::OkStatus();
      }));

  ABSL_RETURN_IF_ERROR(DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
      copy.results, parameter_buffers, result_buffers));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  bool verify_offsets =
      ir_emitter_context_->debug_options()
          .xla_gpu_experimental_dynamic_slice_fusion_verify_offsets();

  return ThunkSequence::Of<DynamicSliceFusionV2Thunk>(
      std::move(info), std::move(copy.parameters), std::move(copy.results),
      std::move(parameter_buffers), std::move(result_buffers),
      std::move(embedded_allocations), std::move(embedded_thunks),
      verify_offsets);
}

Future<ThunkSequence> ThunkEmitter::EmitStaticSliceCopyFusion(
    const HloFusionInstruction* instr, const StaticSliceCopyFusion& copy) {
  if (copy.parameter_number < 0 ||
      copy.parameter_number >= instr->operand_count()) {
    return Internal("Static slice copy parameter %d is out of range for %s",
                    copy.parameter_number, instr->ToString());
  }

  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                   GetAllocationSlice(instr->operand(copy.parameter_number)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_slice,
                   GetAllocationSlice(instr));

  int64_t byte_size = ShapeUtil::ByteSizeOf(copy.slice_shape);
  BufferAllocation::Slice src_slice(
      arg_slice.allocation(), arg_slice.offset() + copy.source_byte_offset,
      byte_size, arg_slice.element_type());

  return ThunkSequence::Of<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      ShapedSlice{src_slice, copy.slice_shape},
      ShapedSlice{dst_slice, instr->shape()}, byte_size);
}

Future<ThunkSequence> ThunkEmitter::EmitFusion(
    const HloFusionInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(std::optional<StaticSliceCopyFusion> static_copy,
                   AnalyzeStaticSliceCopyFusion(instr));
  if (static_copy.has_value()) {
    return EmitStaticSliceCopyFusion(instr, *static_copy);
  }

  ABSL_ASSIGN_OR_RETURN(std::optional<DynamicSliceCopyFusion> dynamic_copy,
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

Future<ThunkSequence> ThunkEmitter::EmitDynamicSliceFusionV2(
    const HloFusionInstruction* instr) {
  const HloComputation* body = instr->fused_instructions_computation();

  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  if (hero == nullptr) {
    return Internal("DynamicSliceFusionV2: no hero operation found");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<DynamicSliceFusion::Parameter> parameters,
                   DynamicSliceFusion::ResolveParameters(hero));
  ABSL_ASSIGN_OR_RETURN(std::vector<DynamicSliceFusion::Result> results,
                   DynamicSliceFusion::ResolveResults(hero));

  // parameter_buffers: one slice per fusion operand, indexed by parameter
  // number.
  std::vector<BufferAllocation::Slice> parameter_buffers;
  parameter_buffers.reserve(instr->operand_count());
  for (const auto* operand : instr->operands()) {
    ABSL_ASSIGN_OR_RETURN(parameter_buffers.emplace_back(),
                     GetAllocationSlice(operand));
  }

  // result_buffers: one entry per fusion output leaf in DFS order.
  std::vector<BufferAllocation::Slice> result_buffers;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      instr->shape(),
      [&](const Shape&, const ShapeIndex& index) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(result_buffers.emplace_back(),
                         GetAllocationSlice(instr, index));
        return absl::OkStatus();
      }));

  ABSL_RETURN_IF_ERROR(DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
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

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  bool verify_offsets =
      ir_emitter_context_->debug_options()
          .xla_gpu_experimental_dynamic_slice_fusion_verify_offsets();

  return EmitHloInstruction(hero).Map(
      [info = std::move(info), results = std::move(results),
       result_buffers = std::move(result_buffers),
       parameters = std::move(parameters),
       parameter_buffers = std::move(parameter_buffers),
       embedded_allocations = std::move(embedded_allocations),
       verify_offsets](ThunkSequence embedded_thunks) mutable {
        return ThunkSequence::Of<DynamicSliceFusionV2Thunk>(
            std::move(info), std::move(parameters), std::move(results),
            std::move(parameter_buffers), std::move(result_buffers),
            std::move(embedded_allocations), std::move(embedded_thunks),
            verify_offsets);
      });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopy(
    const HloInstruction* instr) {
  TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
      instr->operand(0)->shape(), instr->shape(),
      Layout::Equal().MinorToMajorOnly()));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                   GetAllocationSlice(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                   GetAllocationSlice(instr));
  return ThunkSequence::Of<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      /*source_buffer=*/ShapedSlice{src_buffer, instr->operand(0)->shape()},
      /*destination_buffer=*/ShapedSlice{dst_buffer, instr->shape()},
      /*mem_size=*/src_buffer.size());
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

Future<ThunkSequence> ThunkEmitter::EmitWhile(const HloInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto config,
                   instr->backend_config<xla::WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count = std::nullopt;
  if (config.has_known_trip_count()) {
    trip_count = config.known_trip_count().n();
  }

  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Buffer slice holding while loop predicate.
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice pred,
                   GetAllocationSlice(condition->root_instruction(), {}));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  return std::move(tsl::JoinFutures(EmitHloComputation(condition),
                                    EmitHloComputation(body)))
      .Map([info = std::move(info), pred = pred, trip_count = trip_count](
               std::tuple<ThunkSequence, ThunkSequence> tuple) mutable {
        auto [cond_thunks, body_thunks] = std::move(tuple);
        return ThunkSequence::Of<WhileThunk>(
            std::move(info), std::move(pred), std::move(cond_thunks),
            std::move(body_thunks), trip_count);
      });
}

Future<ThunkSequence> ThunkEmitter::EmitCall(const HloInstruction* instr) {
  DCHECK_EQ(instr->opcode(), HloOpcode::kCall);
  DCHECK_EQ(instr->called_computations().size(), 1);
  const HloComputation* computation = instr->called_computations().front();
  return EmitHloComputation(computation);
}

Future<ThunkSequence> ThunkEmitter::EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  ABSL_ASSIGN_OR_RETURN(KernelDefinition<LlvmKernelSource> kernel_def,
                   EmitRngGetAndUpdateStateLLVMIR(instr, ir_emitter_context_,
                                                  kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ABSL_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](auto thunk) { return ThunkSequence::Of(std::move(thunk)); });
}

Future<ThunkSequence> ThunkEmitter::EmitSort(const HloSortInstruction* sort) {
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
    ABSL_ASSIGN_OR_RETURN(destination_buffer, GetAllocationSlice(sort, shape_index));
    ABSL_ASSIGN_OR_RETURN(source_address, GetAllocationSlice(sort->operand(i), {}));

    if (destination_buffer != source_address) {
      // TODO(b/26783907): Figure out why we never seem to share
      // buffers for key/value sort.
      VLOG(2) << op_name << " requires initial D2D copy for operand " << i;
      thunks.Emplace<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              sort, ir_emitter_context_->GetNextThunkId()),
          /*source_buffer=*/
          ShapedSlice{source_address, sort->operand(i)->shape()},
          /*destination_buffer=*/
          ShapedSlice{destination_buffer, sort->operand(i)->shape()},
          ShapeUtil::ByteSizeOf(sort->operand(i)->shape()));
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
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                   GetAllocationSlice(instr, {}));
  return ThunkSequence::Of<ThunkType>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      result_slice);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRngSeed(
    const HloInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                   GetAllocationSlice(instr, {}));
  return ThunkSequence::Of<RngSeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      result_slice);
}

Future<ThunkSequence> ThunkEmitter::EmitCollective(
    const HloInstruction* collective) {
  switch (collective->opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart: {
      auto* all_reduce = Cast<HloAllReduceInstruction>(collective);
      return EmitCollective<AllReduceThunk, HloAllReduceInstruction>(
          Thunk::kAllReduce, all_reduce, all_reduce->use_global_device_ids());
    }

    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart: {
      auto* all_gather = Cast<HloAllGatherInstruction>(collective);
      return EmitCollective<AllGatherThunk, HloAllGatherInstruction>(
          Thunk::kAllGather, all_gather, all_gather->use_global_device_ids());
    }

    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
      return EmitCollective<CollectivePermuteThunk,
                            HloCollectivePermuteInstruction>(
          Thunk::kCollectivePermute,
          Cast<HloCollectivePermuteInstruction>(collective), std::nullopt);

    case HloOpcode::kReduceScatter: {
      auto* reduce_scatter = Cast<HloReduceScatterInstruction>(collective);
      return EmitCollective<ReduceScatterThunk, HloReduceScatterInstruction>(
          Thunk::kReduceScatter, reduce_scatter,
          reduce_scatter->use_global_device_ids());
    }

    case HloOpcode::kAllToAll:
      return EmitCollective<AllToAllThunk, HloAllToAllInstruction>(
          Thunk::kAllToAll, Cast<HloAllToAllInstruction>(collective),
          std::nullopt);

    case HloOpcode::kRaggedAllToAll:
      return EmitCollective<RaggedAllToAllThunk, HloRaggedAllToAllInstruction>(
          Thunk::kRaggedAllToAll,
          Cast<HloRaggedAllToAllInstruction>(collective), std::nullopt);

    case HloOpcode::kCollectiveBroadcast:
      return EmitCollective<CollectiveBroadcastThunk,
                            HloCollectiveBroadcastInstruction>(
          Thunk::kCollectiveBroadcast,
          Cast<HloCollectiveBroadcastInstruction>(collective), std::nullopt);

    default:
      return Internal("Unsupported collective instruction: %s",
                      collective->ToString());
  }
}

Future<ThunkSequence> ThunkEmitter::EmitCollectiveGroup(
    const HloInstruction* instr) {
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return EmitHloComputation(instr->async_wrapped_computation())
      .Map([info = std::move(info)](ThunkSequence thunks) mutable {
        return ThunkSequence::Of<CollectiveGroupThunk>(
            std::move(info), Thunk::Kind::kGroup, std::move(thunks));
      });
}

template <typename CollectiveThunkType, typename HloInstType>
Future<ThunkSequence> ThunkEmitter::EmitCollective(
    Thunk::Kind kind, const HloInstType* inst,
    std::optional<bool> use_global_device_ids) {
  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  int64_t replica_count = hlo_config.replica_count();
  int64_t partition_count = hlo_config.num_partitions();
  int64_t operand_count = inst->operand_count();
  VLOG(2) << CollectiveThunkType::GetHloOpName()
          << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << operand_count;

  // A collective-broadcast may select its root rank at runtime, in which case
  // the last operand is a root-rank vector rather than data to broadcast.
  const bool has_dynamic_root = [](const HloInstType* inst) {
    if constexpr (std::is_same_v<HloInstType,
                                 HloCollectiveBroadcastInstruction>) {
      return inst->has_dynamic_root();
    }
    return false;
  }(inst);

  // CollectivePermuteThunk has its own degeneracy predicate and a different
  // constructor that requires replica/partition counts and permute options.
  constexpr bool is_collective_permute =
      std::is_same_v<CollectiveThunkType, CollectivePermuteThunk>;

  // Stash relevant information in CollectiveThunk::Buffer even if
  // we may not generate a CollectiveThunk.
  ABSL_ASSIGN_OR_RETURN(
      std::vector<CollectiveThunk::Buffer> buffers,
      GetCollectiveBuffers(ir_emitter_context_->buffer_assignment(), inst, kind,
                           has_dynamic_root));

  // A collective permute with no source-target pairs receives no data on any
  // participant, which the collective runtimes implement by zeroing the
  // output (see RunCollectivePermute). Emit the memzero directly and skip the
  // collective thunk; besides avoiding a pointless communicator acquisition,
  // this keeps such programs (e.g. the gradient of a single-device ppermute)
  // working on builds without collectives support.
  if constexpr (is_collective_permute) {
    if (inst->source_target_pairs().empty()) {
      ThunkSequence thunks;
      for (int64_t i = 0; i < buffers.size(); ++i) {
        thunks.Emplace<MemzeroThunk>(
            Thunk::ThunkInfo::WithProfileAnnotation(
                inst, ir_emitter_context_->GetNextThunkId()),
            ShapedSlice{buffers[i].destination_buffer.slice,
                        inst->operand(i)->shape()});
      }
      return thunks;
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
  bool is_degenerate = false;
  if (kind != Thunk::Kind::kRaggedAllToAll) {
    if constexpr (is_collective_permute) {
      is_degenerate = CollectivePermuteThunk::IsDegenerate(inst, replica_count,
                                                           partition_count);
    } else {
      is_degenerate = GetCollectiveConfig(inst, use_global_device_ids)
                          .IsDegenerate(replica_count, partition_count);
    }
  }

  if (is_degenerate) {
    return EmitDegeneratedCollective(buffers, inst);
  }

  if constexpr (!is_collective_permute) {
    ABSL_RETURN_IF_ERROR(CollectiveThunkType::CheckImplementable(inst, replica_count,
                                                            partition_count));
  }

  auto info = Thunk::ThunkInfo::WithProfileAnnotation(
      inst, ir_emitter_context_->GetNextThunkId());
  Future<ThunkSequence> thunks;
  // For AllGather the strategy is now determined by the annotation written
  // by CollectiveKernelStrategyAnnotator.
  // `use_triton` was already set above by reading the backend_config
  // annotation.
  if constexpr (is_collective_permute) {
    thunks = ThunkSequence::Of<CollectivePermuteThunk>(
        info, inst, replica_count, partition_count, std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_collective_permute_mode(),
        ir_emitter_context_->debug_options()
            .xla_gpu_collective_permute_connected_components());
  } else if constexpr (std::is_same_v<CollectiveThunkType,
                                      CollectiveBroadcastThunk>) {
    // CollectiveBroadcastThunk needs the dynamic-root flag so it can treat
    // the trailing root-rank buffer specially at run time.
    thunks = ThunkSequence::Of<CollectiveThunkType>(
        info, inst, /*buffers=*/std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p(),
        has_dynamic_root);
  } else if constexpr (std::is_constructible_v<
                           CollectiveThunkType, Thunk::ThunkInfo,
                           decltype(inst),
                           std::vector<CollectiveThunk::Buffer>>) {
    thunks = ThunkSequence::Of<CollectiveThunkType>(
        info, inst, /*buffers=*/std::move(buffers));
  } else {
    thunks = ThunkSequence::Of<CollectiveThunkType>(
        info, inst, /*buffers=*/std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p());
  }
  return thunks;
}

template <typename HloInstType>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitDegeneratedCollective(
    std::vector<CollectiveThunk::Buffer>& buffers, const HloInstType* inst) {
  // Degenerate collectives are simply identity function. Buffer
  // assignment expects a copy, so that's what we do.
  ThunkSequence thunks;
  for (int64_t i = 0; i < buffers.size(); i++) {
    const Shape shape = inst->operand(i)->shape();
    thunks.Emplace<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            inst, ir_emitter_context_->GetNextThunkId()),
        ShapedSlice{buffers[i].source_buffer.slice, shape},
        ShapedSlice{buffers[i].destination_buffer.slice, shape},
        ShapeUtil::ByteSizeOf(shape));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitInfeed(
    const HloInfeedInstruction* instr) {
  // Infeed instruction returns a tuple containing the result data
  // and a token. We only need the result data to construct the
  // infeed thunk.
  std::vector<ShapedSlice> shaped_slices;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple() || subshape.IsToken()) return absl::OkStatus();
        if (subshape.IsArray()) {
          ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                           GetAllocationSlice(instr, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        instr->ToString(), index.ToString());
      }));

  return ThunkSequence::Of<InfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(shaped_slices));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitOutfeed(
    const HloOutfeedInstruction* instr) {
  // HLO outfeed instruction has 2 operands, the source and a token,
  // and a single token output.
  const HloInstruction* source = instr->operand(0);
  std::vector<ShapedSlice> shaped_slices;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      source->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple()) return absl::OkStatus();
        if (subshape.IsArray()) {
          ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                           GetAllocationSlice(source, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        source->ToString(), index.ToString());
      }));

  return ThunkSequence::Of<OutfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(shaped_slices));
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

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyStart(
    const HloCopyStartInstruction* copy_start_instr) {
  // copy-start has a tuple shape: {host, device, context},
  // or {device, host, context}.
  // Only the destination shape is needed to get the output buffer.
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                   GetAllocationSlice(copy_start_instr,
                                      /*index=*/{0}));

  const HloInstruction* src = copy_start_instr->operand(0);
  const Shape& input_shape = src->shape();
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                   GetAllocationSlice(src, {}));
  const Shape& shape = copy_start_instr->shape();
  CHECK(shape.IsTuple());
  auto host_memory_space =
      static_cast<int>(stream_executor::MemorySpace::kHost);
  ABSL_ASSIGN_OR_RETURN(bool is_dst_host_memory,
                   ShapeHasHostMemorySpace(shape, 0, host_memory_space));
  ABSL_ASSIGN_OR_RETURN(bool is_src_host_memory,
                   ShapeHasHostMemorySpace(shape, 1, host_memory_space));
  if (is_dst_host_memory && is_src_host_memory) {
    return absl::InternalError(
        absl::StrFormat("Copy-start %s has host memory space S(%d) on both "
                        "source and destination, which is unsupported",
                        copy_start_instr->ToString(),
                        static_cast<int>(stream_executor::MemorySpace::kHost)));
  }

  // Create the copy thunk with ThunkInfo derived from copy-start.
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      copy_start_instr, ir_emitter_context_->GetNextThunkId());

  std::unique_ptr<Thunk> copy_thunk;
  if (!is_dst_host_memory && !is_src_host_memory) {
    // D2D async copy: both source and destination reside in device memory.
    // The thunk is a raw memcpy, so source and destination layouts must
    // match; layout-changing copies must not reach this path.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        shape.tuple_shapes(0), input_shape, Layout::Equal().MinorToMajorOnly()))
        << "Copy-start " << copy_start_instr->ToString()
        << " has mismatched source/destination layouts";
    copy_thunk = std::make_unique<DeviceToDeviceCopyThunk>(
        info,
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape));
  } else if (is_dst_host_memory) {
    copy_thunk = std::make_unique<DeviceToHostCopyThunk>(
        info,
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape));
  } else {
    copy_thunk = std::make_unique<HostToDeviceCopyThunk>(
        info,
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
    return ThunkSequence::Of(std::move(copy_thunk));
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

  return ThunkSequence::Of(std::move(start_thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyDone(
    const HloInstruction* instr) {
  const HloInstruction* copy_start_instr = instr->operand(0);
  CHECK(copy_start_instr->opcode() == HloOpcode::kCopyStart);

  // If the copy-start was asynchronous, emit an AsyncDoneThunk.
  auto it = hlo_async_executions_.find(copy_start_instr);
  if (it != hlo_async_executions_.end()) {
    return ThunkSequence::Of<AsyncDoneThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        it->second);
  }

  // Synchronous copy: copy-done is a no-op.
  return ThunkSequence::Empty();
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSend(
    const HloSendInstruction* instr) {
  TF_RET_CHECK(!instr->is_host_transfer());

  const HloInstruction* src = instr->operand(0);
  ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(src, {}));

  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  const int64_t replica_count = hlo_config.replica_count();
  const int64_t partition_count = hlo_config.num_partitions();
  const int64_t memory_space =
      instr->shape().IsTuple()
          ? instr->shape().tuple_shapes(0).layout().memory_space()
          : instr->shape().layout().memory_space();

  const CollectiveThunk::Buffer buffer = {
      /*element_count=*/ShapeUtil::ElementsIn(src->shape()),
      /*source_buffer=*/slice,
      /*destination_buffer=*/slice,
      /*source_memory_space=*/memory_space,
      /*destination_memory_space=*/memory_space};
  return ThunkSequence::Of<SendThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr, replica_count, partition_count, buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRecv(
    const HloRecvInstruction* instr) {
  TF_RET_CHECK(!instr->is_host_transfer());
  TF_RET_CHECK(instr->shape().IsTuple());

  ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(instr, {0}));

  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  const int64_t replica_count = hlo_config.replica_count();
  const int64_t partition_count = hlo_config.num_partitions();
  const int64_t memory_space =
      instr->shape().tuple_shapes(0).layout().memory_space();

  const CollectiveThunk::Buffer buffer = {
      /*element_count=*/ShapeUtil::ElementsIn(instr->shape().tuple_shapes(0)),
      /*source_buffer=*/slice,
      /*destination_buffer=*/slice,
      /*source_memory_space=*/memory_space,
      /*destination_memory_space=*/memory_space};
  return ThunkSequence::Of<RecvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr, replica_count, partition_count, buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSendDone(
    const HloSendDoneInstruction* instr) {
  TF_RET_CHECK(!instr->is_host_transfer());
  return EmitAsyncDone(instr, FindCanonicalSendRecvStartOp(instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRecvDone(
    const HloRecvDoneInstruction* instr) {
  TF_RET_CHECK(!instr->is_host_transfer());
  return EmitAsyncDone(instr, FindCanonicalSendRecvStartOp(instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostSend(
    const HloSendInstruction* instr) {
  TF_RET_CHECK(instr->is_host_transfer());

  const HloInstruction* src = instr->operand(0);
  ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(src, {}));

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send instruction");
  }

  return ThunkSequence::Of<HostSendThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      src->shape(), slice.slice, *instr->channel_id(), send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostRecv(
    const HloRecvInstruction* instr) {
  TF_RET_CHECK(instr->is_host_transfer());
  TF_RET_CHECK(instr->shape().IsTuple());

  ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(instr, {0}));

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv instruction");
  }

  return ThunkSequence::Of<HostRecvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr->shape().tuple_shapes()[0], slice.slice, *instr->channel_id(),
      send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostSendDone(
    const HloInstruction* done, const HloSendRecvInstruction* host_transfer) {
  TF_RET_CHECK(host_transfer->is_host_transfer());
  if (!host_transfer->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send done instruction");
  }

  return ThunkSequence::Of<HostSendDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          done, ir_emitter_context_->GetNextThunkId()),
      *host_transfer->channel_id(), send_recv_events_,
      DeviceConstraint(host_transfer));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostRecvDone(
    const HloInstruction* done, const HloSendRecvInstruction* host_transfer) {
  TF_RET_CHECK(host_transfer->is_host_transfer());
  if (!host_transfer->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv done instruction");
  }

  return ThunkSequence::Of<HostRecvDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          done, ir_emitter_context_->GetNextThunkId()),
      *host_transfer->channel_id(), send_recv_events_,
      DeviceConstraint(host_transfer));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostExecuteStart(
    const HloInstruction* async_start,
    const HloCustomCallInstruction* host_execute) {
  TF_RET_CHECK(IsHostExecuteCustomCall(*host_execute));

  std::unique_ptr<HloModule> hlo_module =
      ExtractComputationIntoNewModule(*host_execute->called_computation());

  // All offloaded computations are marked as host computations from the
  // perspective of the GPU backend. Since these will execute on the main
  // thread from the CPU backend perspective, mark them as such.
  for (auto* computation : hlo_module->computations()) {
    computation->SetExecutionThread(HloInstruction::kMainExecutionThread);
  }

  absl::InlinedVector<ShapedSlice, 4> operand_slices;
  for (HloInstruction* operand : host_execute->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      ABSL_ASSIGN_OR_RETURN(auto slice,
                       ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                           operand, indexed.index));
      operand_slices.push_back({slice, indexed.shape});
    }
  }

  absl::InlinedVector<ShapedSlice, 4> result_slices;
  for (auto& indexed : ShapeUtil::GetLeafShapes(host_execute->shape())) {
    ABSL_ASSIGN_OR_RETURN(auto slice,
                     ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                         host_execute, indexed.index));
    result_slices.push_back({slice, indexed.shape});
  }

  HostOffloadingExecutableProto host_offloading_executable_proto;
  *host_offloading_executable_proto.mutable_hlo_module() =
      hlo_module->ToProto();
  host_offloading_executable_proto.set_executable_type(
      HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT);

  ABSL_ASSIGN_OR_RETURN(auto thunk,
                   HostExecuteStartThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           async_start, ir_emitter_context_->GetNextThunkId()),
                       std::move(host_offloading_executable_proto),
                       std::move(operand_slices), std::move(result_slices)));

  auto [it, inserted] = GetInstructionToHostExecuteAsyncEvents().emplace(
      host_execute, thunk->async_events());
  if (!inserted) {
    return Internal(
        "Async events already exist for host offloading custom call %s.",
        host_execute->ToString());
  }
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostExecuteDone(
    const HloInstruction* async_done,
    const HloCustomCallInstruction* host_execute) {
  TF_RET_CHECK(IsHostExecuteCustomCall(*host_execute));

  auto it = GetInstructionToHostExecuteAsyncEvents().find(host_execute);
  TF_RET_CHECK(it != GetInstructionToHostExecuteAsyncEvents().end())
      << "could not find async events for host execute operation";

  absl::InlinedVector<ShapedSlice, 4> result_slices;
  for (auto& indexed : ShapeUtil::GetLeafShapes(host_execute->shape())) {
    ABSL_ASSIGN_OR_RETURN(auto slice,
                          ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                              host_execute, indexed.index));
    result_slices.push_back({slice, indexed.shape});
  }

  return ThunkSequence::Of<HostExecuteDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          async_done, ir_emitter_context_->GetNextThunkId()),
      std::move(result_slices), it->second);
}

Future<ThunkSequence> ThunkEmitter::EmitAsyncStart(
    const HloInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<AsyncExecution> execution,
                   RegisterAsyncExecution(instr));

  Future<ThunkSequence> nested =
      HasCollectivesGroupAttribute(instr)
          ? EmitCollectiveGroup(instr)
          : EmitHloComputation(instr->async_wrapped_computation());

  return std::move(nested).Map([this, instr, execution = std::move(execution)](
                                   ThunkSequence thunks) mutable {
    return EmitAsyncStart(std::move(execution), instr, std::move(thunks));
  });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncStart(
    std::shared_ptr<AsyncExecution> execution,
    const HloInstruction* async_start, ThunkSequence thunks) {
  const ExecutionStreamAssignment& streams =
      ir_emitter_context_->execution_stream_assignment();
  ABSL_ASSIGN_OR_RETURN(ExecutionStreamId stream_id,
                   streams.GetExecutionStreamId(async_start));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      async_start, execution->start_thunk_id());
  return ThunkSequence::Of<AsyncStartThunk>(
      std::move(info), stream_id, std::move(thunks), std::move(execution));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncSendRecvStart(
    const HloSendRecvInstruction* async_start, ThunkSequence thunks) {
  // Device send/recv outside an async computation can pipeline multiple start
  // thunks through one AsyncExecution. They use a canonical owner so every
  // pipelined start shares the same execution.
  const HloInstruction* owner = FindCanonicalSendRecvStartOp(async_start);

  const ExecutionStreamAssignment& streams =
      ir_emitter_context_->execution_stream_assignment();
  ABSL_ASSIGN_OR_RETURN(ExecutionStreamId stream_id,
                   streams.GetExecutionStreamId(owner));

  if (auto it = hlo_async_executions_.find(owner);
      it != hlo_async_executions_.end()) {
    Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
        async_start, ir_emitter_context_->GetNextThunkId());
    return ThunkSequence::Of<AsyncStartThunk>(std::move(info), stream_id,
                                              std::move(thunks), it->second);
  }

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      async_start, ir_emitter_context_->GetNextThunkId());
  auto [it, inserted] = hlo_async_executions_.emplace(
      owner, std::make_shared<AsyncExecution>(info));
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    owner->ToString());
  }

  return ThunkSequence::Of<AsyncStartThunk>(std::move(info), stream_id,
                                            std::move(thunks), it->second);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncDone(
    const HloInstruction* done, const HloInstruction* start) {
  auto it = hlo_async_executions_.find(start);
  TF_RET_CHECK(it != hlo_async_executions_.end())
      << "could not find async execution for start operation";
  return ThunkSequence::Of<AsyncDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          done, ir_emitter_context_->GetNextThunkId()),
      it->second);
}

Future<ThunkSequence> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* hlo) {
  switch (hlo->opcode()) {
    // Legacy non-async-wrapped collective-start operations.
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCollectivePermuteStart:
      return DispatchLegacyCollectiveStart(hlo);

    // Legacy non-async-wrapped collective-done operations.
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectivePermuteDone:
      return DispatchAsyncDone(hlo);

    // Synchronous collective operations (async execution if needed added by
    // wrapping into generic async-start/async-done).
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kReduceScatter:
      return EmitCollective(hlo);

    // Generic async start/done wrapping asynchronous computation (operation).
    case HloOpcode::kAsyncStart:
      return DispatchAsyncStart(hlo);
    case HloOpcode::kAsyncDone:
      return DispatchAsyncDone(hlo);

    // Send/recv and their done operations dispatch first by
    // `is_host_transfer()`. Device transfer start emission then depends on
    // whether it is inside an async computation.
    case HloOpcode::kSend:
      return DispatchSend(Cast<HloSendInstruction>(hlo));
    case HloOpcode::kSendDone:
      return DispatchSendDone(Cast<HloSendDoneInstruction>(hlo));
    case HloOpcode::kRecv:
      return DispatchRecv(Cast<HloRecvInstruction>(hlo));
    case HloOpcode::kRecvDone:
      return DispatchRecvDone(Cast<HloRecvDoneInstruction>(hlo));

    case HloOpcode::kCall:
      return EmitCall(hlo);
    case HloOpcode::kConditional:
      return EmitConditional(hlo);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(hlo));
    case HloOpcode::kCustomCall:
      return DispatchCustomCall(hlo);
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
      return EmitFft(Cast<HloFftInstruction>(hlo));

    case HloOpcode::kReplicaId:
      return EmitReplicaOrPartitionId<ReplicaIdThunk>(hlo);
    case HloOpcode::kRngGetAndUpdateState:
      return EmitRngGetAndUpdateState(
          Cast<HloRngGetAndUpdateStateInstruction>(hlo));

    case HloOpcode::kSort:
      return EmitSort(Cast<HloSortInstruction>(hlo));
    case HloOpcode::kWhile:
      return EmitWhile(hlo);
    case HloOpcode::kCopyStart:
      return EmitCopyStart(Cast<HloCopyStartInstruction>(hlo));
    case HloOpcode::kCopyDone:
      return EmitCopyDone(hlo);

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
      return ThunkSequence::Empty();
    default:
      return Internal("Unsupported instruction opcode: %s",
                      HloOpcodeString(hlo->opcode()));
  }
  return Internal("Unhandled HLO instruction");
}

Future<ThunkSequence> ThunkEmitter::EmitHloEntryComputation(
    const HloModule* module) {
  return EmitHloComputation(module->entry_computation());
}

Future<ThunkSequence> ThunkEmitter::EmitHloComputation(
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
  std::vector<Future<ThunkSequence>> futures(instructions.size());
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
