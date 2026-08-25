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

#include "xla/backends/gpu/autotuner/triton/aiter_unified_attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/autotuner/triton/embed_aiter_unified_attention.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/util/file_toc.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kTritonCallTarget = "__gpu$xla.gpu.triton";
constexpr absl::string_view kKernelName =
    "aiter_triton_paged_attention_producer";

absl::string_view GetTemplate() {
  const FileToc* toc =
      aiter_unified_attention::embed_aiter_unified_attention_create();
  for (size_t i = 0;
       i < aiter_unified_attention::embed_aiter_unified_attention_size();
       ++i) {
    if (absl::string_view(toc[i].name) ==
        "aiter_unified_attention_3d.ttir.tpl") {
      return absl::string_view(toc[i].data, toc[i].size);
    }
  }
  return {};
}

std::string MlirFloatLiteral(double value) {
  std::string result = absl::StrFormat("%.17g", value);
  if (!absl::StrContains(result, ".") &&
      !absl::StrContainsIgnoreCase(result, "e")) {
    absl::StrAppend(&result, ".0");
  }
  return result;
}

absl::StatusOr<HloInstruction*> OuterOperand(
    HloInstruction& fusion, const HloInstruction* inner_parameter) {
  if (inner_parameter == nullptr ||
      inner_parameter->opcode() != HloOpcode::kParameter) {
    return absl::InvalidArgumentError(
        "AITER Triton paged attention requires fusion parameter operands.");
  }
  const int64_t number = inner_parameter->parameter_number();
  if (number < 0 || number >= fusion.operand_count()) {
    return absl::InvalidArgumentError(
        "AITER Triton paged-attention parameter is outside the fusion ABI.");
  }
  return fusion.mutable_operand(number);
}

std::string BuildBackendConfig(absl::string_view ttir,
                               const BlockLevelFusionConfig& config,
                               int64_t sequences, int64_t kv_heads,
                               int64_t num_segments) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  std::vector<mlir::NamedAttribute> attributes = {
      builder.getNamedAttr("name", builder.getStringAttr(kKernelName)),
      builder.getNamedAttr("ir", builder.getStringAttr(ttir)),
      builder.getNamedAttr(
          "num_stages",
          builder.getI32IntegerAttr(static_cast<int32_t>(config.num_stages()))),
      builder.getNamedAttr(
          "num_warps",
          builder.getI32IntegerAttr(static_cast<int32_t>(config.num_warps()))),
      builder.getNamedAttr("grid_x", builder.getI32IntegerAttr(sequences)),
      builder.getNamedAttr("grid_y", builder.getI32IntegerAttr(kv_heads)),
      builder.getNamedAttr("grid_z",
                           builder.getI32IntegerAttr(num_segments)),
      builder.getNamedAttr("debug", builder.getBoolAttr(false)),
      builder.getNamedAttr("is_tma_allowed", builder.getBoolAttr(false)),
      builder.getNamedAttr("global_scratch_memory_size",
                           builder.getI32IntegerAttr(0)),
      builder.getNamedAttr(
          "waves_per_eu",
          builder.getI32IntegerAttr(config.waves_per_eu())),
  };
  mlir::DictionaryAttr dictionary =
      mlir::DictionaryAttr::get(&context, llvm::ArrayRef(attributes));
  std::string result;
  llvm::raw_string_ostream(result) << dictionary;
  return result;
}

}  // namespace

absl::StatusOr<flydsl::FlyPagedAttentionSegmentedProducerDescriptor>
GetAiterTritonPagedAttentionDescriptor(const HloInstruction& instruction) {
  if (instruction.opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError(
        "AITER Triton paged attention requires a producer fusion.");
  }
  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(&instruction);
  auto descriptor = flydsl::GetFlyPagedAttentionSegmentedProducerDescriptor(
      *fusion->fused_expression_root());
  if (!descriptor.has_value()) {
    return absl::InvalidArgumentError(
        "Fusion is not a supported segmented paged-attention producer.");
  }
  // This TTIR is AITER's MI300 all-decode specialization: D128, TILE_SIZE=16,
  // BLOCK_M=16. Page-size variants need separately generated source TTIR.
  if (descriptor->attention.page_size != 16) {
    return absl::InvalidArgumentError(
        "The embedded AITER Triton producer currently requires page size 16.");
  }
  return *descriptor;
}

absl::StatusOr<std::string> BuildAiterTritonPagedAttentionTtir(
    const flydsl::FlyPagedAttentionSegmentedProducerDescriptor& descriptor) {
  const auto& attention = descriptor.attention;
  if (attention.page_size != 16 || attention.head_dimension != 128 ||
      (attention.element_type != BF16 && attention.element_type != F16) ||
      descriptor.num_segments <= 1 || descriptor.num_segments > 256 ||
      attention.gqa_group <= 0 || attention.gqa_group > 16) {
    return absl::InvalidArgumentError(
        "Unsupported shape for embedded AITER unified-attention TTIR.");
  }
  absl::string_view source = GetTemplate();
  if (source.empty()) {
    return absl::InternalError("Embedded AITER TTIR template is missing.");
  }

  constexpr int64_t kTileSize = 16;
  const int64_t segment_denominator = descriptor.num_segments * kTileSize;
  const int64_t state_sequence_stride =
      attention.query_heads * descriptor.num_segments;
  const int64_t output_head_stride =
      descriptor.num_segments * attention.head_dimension;
  const int64_t output_sequence_stride =
      attention.query_heads * output_head_stride;
  const int64_t cache_stride_1 =
      attention.kv_heads * attention.head_dimension;
  const int64_t cache_stride_0 = attention.page_size * cache_stride_1;
  const int64_t query_span_minus_one = 15 / attention.gqa_group;

  std::string result = absl::StrReplaceAll(
      source,
      {{"__ELEMENT_TYPE__",
        attention.element_type == BF16 ? "bf16" : "f16"},
       {"__SCALE__", MlirFloatLiteral(attention.scale)},
       {"__BLOCK_TABLE_STRIDE__",
        absl::StrCat(attention.pages_per_sequence)},
       {"__QUERY_STRIDE_0__",
        absl::StrCat(attention.query_heads * attention.head_dimension)},
       {"__CACHE_STRIDE_0__", absl::StrCat(cache_stride_0)},
       {"__CACHE_STRIDE_1__", absl::StrCat(cache_stride_1)},
       {"__SEGMENT_DEN_MINUS_ONE__",
        absl::StrCat(segment_denominator - 1)},
       {"__SEGMENT_DEN__", absl::StrCat(segment_denominator)},
       {"__NUM_SEGMENTS__", absl::StrCat(descriptor.num_segments)},
       {"__STATE_SEQUENCE_STRIDE__",
        absl::StrCat(state_sequence_stride)},
       {"__OUTPUT_HEAD_STRIDE__", absl::StrCat(output_head_stride)},
       {"__OUTPUT_SEQUENCE_STRIDE__",
        absl::StrCat(output_sequence_stride)},
       {"__PAGE_SIZE__", absl::StrCat(attention.page_size)},
       {"__QUERY_HEADS__", absl::StrCat(attention.query_heads)},
       {"__GQA_GROUP__", absl::StrCat(attention.gqa_group)},
       {"__QUERY_SPAN_MINUS_ONE__",
        absl::StrCat(query_span_minus_one)}});
  if (absl::StrContains(result, "__")) {
    return absl::InternalError(
        "Unresolved placeholder in embedded AITER unified-attention TTIR.");
  }
  return result;
}

absl::Status ApplyAiterTritonPagedAttentionConfig(
    HloInstruction& instruction, const BlockLevelFusionConfig& config) {
  TF_ASSIGN_OR_RETURN(auto descriptor,
                      GetAiterTritonPagedAttentionDescriptor(instruction));
  if (config.num_warps() != 2 || config.num_stages() < 1 ||
      config.num_stages() > 3 || config.waves_per_eu() < 0) {
    return absl::InvalidArgumentError(
        "Invalid launch config for AITER Triton paged attention.");
  }
  TF_ASSIGN_OR_RETURN(std::string ttir,
                      BuildAiterTritonPagedAttentionTtir(descriptor));
  TF_ASSIGN_OR_RETURN(HloInstruction * query,
                      OuterOperand(instruction, descriptor.attention.query));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * key_cache,
      OuterOperand(instruction, descriptor.attention.key_cache));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * value_cache,
      OuterOperand(instruction, descriptor.attention.value_cache));
  TF_ASSIGN_OR_RETURN(HloInstruction * used_k,
                      OuterOperand(instruction, descriptor.attention.used_k));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * block_table,
      OuterOperand(instruction, descriptor.attention.block_table));
  std::vector<HloInstruction*> operands = {query, key_cache, value_cache,
                                           used_k, block_table};
  std::string backend_config = BuildBackendConfig(
      ttir, config, descriptor.attention.sequences,
      descriptor.attention.kv_heads, descriptor.num_segments);
  std::unique_ptr<HloInstruction> custom_call =
      HloInstruction::CreateCustomCall(instruction.shape(), operands,
                                       kTritonCallTarget, backend_config);
  instruction.SetupDerivedInstruction(custom_call.get());
  return instruction.parent()->ReplaceWithNewInstruction(
      &instruction, std::move(custom_call));
}

}  // namespace xla::gpu
