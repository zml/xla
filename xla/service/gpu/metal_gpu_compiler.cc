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

#include "xla/service/gpu/metal_gpu_compiler.h"

#include <cmath>
#include <complex>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/metal_msl_emitter.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

constexpr char kMetalTargetTriple[] = "air64-apple-macosx";
constexpr char kMetalDataLayout[] = "e-p:64:64-i64:64-n32:64-S128";
constexpr double kPi = 3.141592653589793238462643383279502884;

PrimitiveType ComplexTypeForFft(const HloFftInstruction& fft) {
  PrimitiveType input_type = fft.operand(0)->shape().element_type();
  PrimitiveType output_type = fft.shape().element_type();
  return input_type == C128 || output_type == C128 ? C128 : C64;
}

int64_t Product(const std::vector<int64_t>& values) {
  int64_t product = 1;
  for (int64_t value : values) product *= value;
  return product;
}

std::vector<int64_t> TrailingDimensions(const Shape& shape, int64_t count) {
  std::vector<int64_t> dimensions;
  dimensions.reserve(count);
  int64_t rank = shape.dimensions().size();
  for (int64_t i = rank - count; i < rank; ++i) {
    dimensions.push_back(shape.dimensions(i));
  }
  return dimensions;
}

std::vector<int64_t> BatchDimensions(const Shape& shape, int64_t fft_rank) {
  std::vector<int64_t> dimensions;
  int64_t rank = shape.dimensions().size();
  dimensions.reserve(rank - fft_rank);
  for (int64_t i = 0; i < rank - fft_rank; ++i) {
    dimensions.push_back(shape.dimensions(i));
  }
  return dimensions;
}

std::vector<int64_t> WithFlatFftDimension(std::vector<int64_t> dimensions,
                                          int64_t flat_fft_size) {
  dimensions.push_back(flat_fft_size);
  return dimensions;
}

std::vector<int64_t> UnflattenIndex(int64_t index,
                                    const std::vector<int64_t>& dimensions) {
  std::vector<int64_t> result(dimensions.size());
  for (int64_t i = dimensions.size() - 1; i >= 0; --i) {
    result[i] = index % dimensions[i];
    index /= dimensions[i];
  }
  return result;
}

Literal CreateDftKernel(PrimitiveType type,
                        const std::vector<int64_t>& fft_lengths,
                        const std::vector<int64_t>& input_dims,
                        const std::vector<int64_t>& output_dims,
                        FftType fft_type) {
  int64_t rows = Product(input_dims);
  int64_t cols = Product(output_dims);
  int64_t element_count = Product(fft_lengths);
  Literal literal(ShapeUtil::MakeShape(type, {rows, cols}));
  for (int64_t row = 0; row < rows; ++row) {
    std::vector<int64_t> row_coords = UnflattenIndex(row, input_dims);
    for (int64_t col = 0; col < cols; ++col) {
      std::vector<int64_t> col_coords = UnflattenIndex(col, output_dims);
      double scale = 1.0;
      double sign = -1.0;

      if (fft_type == FftType::IFFT) {
        sign = 1.0;
        scale = 1.0 / static_cast<double>(element_count);
      } else if (fft_type == FftType::IRFFT) {
        sign = 1.0;
        scale = 1.0 / static_cast<double>(element_count);
        int64_t last_frequency = row_coords.back();
        int64_t last_length = fft_lengths.back();
        if (last_frequency != 0 &&
            !(last_length % 2 == 0 && last_frequency == last_length / 2)) {
          scale *= 2.0;
        }
      }

      double phase = 0.0;
      for (int64_t i = 0; i < fft_lengths.size(); ++i) {
        phase += static_cast<double>(row_coords[i] * col_coords[i]) /
                 static_cast<double>(fft_lengths[i]);
      }
      double angle = sign * 2.0 * kPi * phase;
      if (type == C128) {
        literal.Set<complex128>({row, col},
                                scale * complex128(std::cos(angle),
                                                    std::sin(angle)));
      } else {
        literal.Set<complex64>({row, col},
                               complex64(scale * std::cos(angle),
                                         scale * std::sin(angle)));
      }
    }
  }
  return literal;
}

class MetalFftExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "metal-fft-expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override {
    if (instruction->opcode() != HloOpcode::kFft) return false;
    const HloFftInstruction* fft = Cast<HloFftInstruction>(instruction);
    if (fft->fft_length().empty()) return false;

    PrimitiveType input_type = fft->operand(0)->shape().element_type();
    PrimitiveType output_type = fft->shape().element_type();
    switch (fft->fft_type()) {
      case FftType::FFT:
      case FftType::IFFT:
        return (input_type == C64 || input_type == C128) &&
               input_type == output_type;
      case FftType::RFFT:
        return (input_type == F32 && output_type == C64) ||
               (input_type == F64 && output_type == C128);
      case FftType::IRFFT:
        return (input_type == C64 && output_type == F32) ||
               (input_type == C128 && output_type == F64);
    }
  }

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override {
    HloComputation* computation = instruction->parent();
    const HloFftInstruction* fft = Cast<HloFftInstruction>(instruction);
    HloInstruction* operand = instruction->mutable_operand(0);

    std::vector<int64_t> fft_lengths(fft->fft_length().begin(),
                                     fft->fft_length().end());
    const int64_t fft_rank = fft_lengths.size();
    std::vector<int64_t> input_dims =
        TrailingDimensions(operand->shape(), fft_rank);
    std::vector<int64_t> output_dims =
        TrailingDimensions(instruction->shape(), fft_rank);
    const int64_t input_size = Product(input_dims);
    const int64_t output_size = Product(output_dims);
    std::vector<int64_t> batch_dims = BatchDimensions(operand->shape(), fft_rank);
    PrimitiveType complex_type = ComplexTypeForFft(*fft);

    if (fft->fft_type() == FftType::RFFT) {
      Shape complex_input_shape =
          ShapeUtil::ChangeElementType(operand->shape(), complex_type);
      operand = computation->AddInstruction(
          HloInstruction::CreateConvert(complex_input_shape, operand));
    }

    Shape flat_input_shape = ShapeUtil::MakeShape(
        operand->shape().element_type(),
        WithFlatFftDimension(batch_dims, input_size));
    operand = computation->AddInstruction(
        HloInstruction::CreateReshape(flat_input_shape, operand));

    HloInstruction* kernel =
        computation->AddInstruction(HloInstruction::CreateConstant(
            CreateDftKernel(complex_type, fft_lengths, input_dims, output_dims,
                            fft->fft_type())));

    DotDimensionNumbers dnums;
    dnums.add_lhs_contracting_dimensions(operand->shape().dimensions().size() -
                                         1);
    dnums.add_rhs_contracting_dimensions(0);
    PrecisionConfig precision_config;

    Shape flat_output_shape = ShapeUtil::MakeShape(
        instruction->shape().element_type(),
        WithFlatFftDimension(batch_dims, output_size));
    Shape dot_shape = flat_output_shape;
    if (fft->fft_type() == FftType::IRFFT) {
      dot_shape = ShapeUtil::ChangeElementType(dot_shape, complex_type);
    }
    HloInstruction* dot = computation->AddInstruction(HloInstruction::CreateDot(
        dot_shape, operand, kernel, dnums, precision_config));

    if (fft->fft_type() == FftType::IRFFT) {
      HloInstruction* real = computation->AddInstruction(
          HloInstruction::CreateUnary(flat_output_shape, HloOpcode::kReal, dot));
      return computation->AddInstruction(
          HloInstruction::CreateReshape(instruction->shape(), real));
    }
    return computation->AddInstruction(
        HloInstruction::CreateReshape(instruction->shape(), dot));
  }
};

}  // namespace

MetalGpuCompiler::MetalGpuCompiler()
    : GpuCompiler(stream_executor::metal::kMetalPlatformId, kMetalTargetTriple,
                  kMetalDataLayout) {}

absl::Status MetalGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  HloPassPipeline pre_pipeline("metal triangular solve expansion",
                               compilation_stats);
  pre_pipeline.AddPass<TriangularSolveExpander>();
  pre_pipeline.AddPass<MetalFftExpander>();
  return pre_pipeline.Run(hlo_module, {HloInstruction::kMainExecutionThread})
      .status();
}

void MetalGpuCompiler::AddPaddingForGpublasGemms(
    HloPassPipeline& pipeline, const DebugOptions& debug_options,
    const se::GpuComputeCapability& gpu_version) {}

absl::Status MetalGpuCompiler::AddConvAndGemmAutotuningPass(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    const se::GpuComputeCapability& gpu_version, const CompileOptions& options,
    tsl::thread::ThreadPool* thread_pool, se::StreamExecutor* stream_exec,
    const Compiler::GpuTargetConfig* target_config,
    const MultiProcessKeyValueStore& key_value_store,
    const se::SemanticVersion& toolkit_version, const AliasInfo* alias_info,
    const DebugOptions& debug_options, mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  return absl::OkStatus();
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
MetalGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    std::optional<int> shard_number) {
  TF_ASSIGN_OR_RETURN(std::string msl, EmitMslFromLlvmModule(*llvm_module));
  std::vector<uint8_t> binary(msl.begin(), msl.end());
  return BackendCompileResult{/*asm_text=*/std::move(msl),
                              /*binary=*/std::move(binary)};
}

std::vector<std::string> MetalGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return {};
}

}  // namespace gpu
}  // namespace xla
