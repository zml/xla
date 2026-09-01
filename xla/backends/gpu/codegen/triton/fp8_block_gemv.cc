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

#include "xla/backends/gpu/codegen/triton/fp8_block_gemv.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::mlir::Value;
using ::xla::xtile::CreateConst;

constexpr int64_t kScaleBlock = 128;

constexpr int64_t kMaxBlockK = 512;

const HloInstruction* SkipBitcasts(const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape) {
    instr = instr->operand(0);
  }
  return instr;
}

bool IsRowMajor(const Shape& shape) {
  return LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

// A block scale is either a bf16 value or a UE8M0 exponent.
bool IsScaleType(PrimitiveType type) {
  // F32 is the form the CUTLASS rung requires (its collective ties the scale
  // element to the f32 accumulator); the Triton emitter widens any of these to
  // f32 before folding, so accepting it costs nothing there.
  return type == BF16 || type == F8E8M0FNU || type == F32;
}

bool IsIdentityScale(const HloInstruction* scale) {
  scale = SkipBitcasts(scale);
  if (scale->shape().element_type() != BF16) return false;
  for (int64_t dim : scale->shape().dimensions()) {
    if (dim != 1) return false;
  }
  return scale->opcode() == HloOpcode::kConstant &&
         scale->literal().IsAllFloat(1.0f);
}

std::optional<Fp8BlockGemvSpec> MatchScaledDotStructure(
    const HloScaledDotInstruction& dot,
    absl::FunctionRef<const HloInstruction*(const HloInstruction*)> resolve,
    const char** reason = nullptr, bool scale_proven_identity = false) {
  auto no = [&](const char* why) -> std::optional<Fp8BlockGemvSpec> {
    if (reason != nullptr) *reason = why;
    return std::nullopt;
  };
  const DotDimensionNumbers& dims = dot.dot_dimension_numbers();
  if (!dims.lhs_batch_dimensions().empty() ||
      !dims.rhs_batch_dimensions().empty() ||
      dims.lhs_contracting_dimensions_size() != 1 ||
      dims.rhs_contracting_dimensions_size() != 1) {
    return no("dims");
  }
  if (dot.shape().element_type() != BF16) return no("out-not-bf16");

  const HloInstruction* activation = dot.operand(0);
  const HloInstruction* weight = dot.operand(1);
  const HloInstruction* act_scale = dot.operand(2);
  const HloInstruction* scale = dot.operand(3);

  const bool w8a8 = activation->shape().element_type() == F8E4M3FN;
  if (!w8a8) {
    if (!scale_proven_identity && !IsIdentityScale(resolve(act_scale))) {
      return no("act-scale-not-identity");
    }
    if (activation->shape().element_type() != BF16) return no("act-type");
  }
  if (activation->shape().dimensions().size() != 2 ||
      !IsRowMajor(activation->shape())) {
    return no("act-shape");
  }
  if (dims.rhs_contracting_dimensions(0) != 1) return no("weight-contracting");
  const int64_t act_contracting = dims.lhs_contracting_dimensions(0);
  const bool batch_major = act_contracting == 1;
  const int64_t batch = activation->shape().dimensions(batch_major ? 0 : 1);
  if (batch < 1) return no("batch-range");
  // One row takes the reduction path; the mma path tiles rows by 16. A batch
  // neither of those covers is still claimable, but only where the CUTLASS
  // rung is -- see Fp8BlockGemvBatchNeedsCutlass, which the arm applies with a
  // compute capability in hand.
  if (!w8a8 && batch > 1 && batch % 16 != 0) return no("batch-range");
  // A bf16 activation has to be converted before the mma, and above 16 rows
  // that costs 3.5-4.2x against the generic route -- measured. Only the W8A8
  // form, which feeds the tensor cores directly, claims wider than a decode.
  if (!w8a8 && batch > 16) return no("w8a16-batch-cap");
  // The per-row activation scale is spelled [batch, k/128] only.
  if (w8a8 && !batch_major) return no("w8a8-act-layout");

  if (weight->shape().element_type() != F8E4M3FN ||
      weight->shape().dimensions().size() != 2 ||
      !IsRowMajor(weight->shape())) {
    return no("weight-type");
  }
  if (!IsScaleType(scale->shape().element_type()) ||
      scale->shape().dimensions().size() != 2 ||
      !IsRowMajor(scale->shape())) {
    return no("scale-type");
  }

  const int64_t n = weight->shape().dimensions(0);
  const int64_t k = weight->shape().dimensions(1);
  if (n % kScaleBlock != 0 || k % kScaleBlock != 0) {
    return no("scale-block-align");
  }
  if (scale->shape().dimensions(0) != n / kScaleBlock ||
      scale->shape().dimensions(1) != k / kScaleBlock) {
    return no("scale-grid");
  }
  if (activation->shape().dimensions(act_contracting) != k) {
    return no("k-mismatch");
  }
  if (w8a8) {
    if (!IsScaleType(act_scale->shape().element_type()) ||
        act_scale->shape().dimensions().size() != 2 ||
        !IsRowMajor(act_scale->shape())) {
      return no("act-scale-type");
    }
    if (act_scale->shape().dimensions(0) != batch ||
        act_scale->shape().dimensions(1) != k / kScaleBlock) {
      return no("act-scale-grid");
    }
  }

  return Fp8BlockGemvSpec{
      /*activation_index=*/0,
      /*weight_index=*/0,
      /*scale_index=*/0,
      /*act_scale_index=*/0,
      /*n=*/n,
      /*k=*/k,
      /*batch=*/batch,
      /*activation_batch_major=*/batch_major,
      /*w8a8=*/w8a8,
      /*weight_scale_type=*/scale->shape().element_type(),
  };
}

}  // namespace

namespace {
int64_t ChooseBlockK(int64_t block_n, int64_t k, int64_t max_block_k) {
  const int64_t budget = 32 * 1024;
  int64_t block_k = budget / (block_n * static_cast<int64_t>(sizeof(float)));
  block_k = std::min<int64_t>(block_k, max_block_k);
  block_k = std::max<int64_t>(block_k, kScaleBlock);
  block_k = (block_k / kScaleBlock) * kScaleBlock;
  while (block_k > kScaleBlock && k % block_k != 0) {
    block_k -= kScaleBlock;
  }
  return block_k;
}
}  // namespace

bool Fp8BlockGemvBatchNeedsCutlass(int64_t batch) {
  return batch != 1 && batch % 16 != 0;
}

bool HasCutlassBlockGemm(const se::GpuComputeCapability& gpu_version) {
  const se::CudaComputeCapability* cc = gpu_version.cuda_compute_capability();
  if (cc == nullptr) return false;
  // Both Blackwell families, which take different collectives: the datacenter
  // parts get tcgen05, the consumer one warp-level mma. The table carries an
  // instantiation for each, and CanRun applies this same test per config.
  //
  // Consumer Blackwell is 12.0 only. The build targets 12.0a, and a plain
  // sm_121 cubin has TMA compiled out of the mainloop, so an sm_121 part must
  // not be told the collective is there -- otherwise the arm claims the dot
  // and every config then declines at emit time.
  return cc->major == se::CudaComputeCapability::kBlackwell ||
         (cc->major == se::CudaComputeCapability::kBlackwell_12 &&
          cc->minor == 0);
}

std::optional<Fp8BlockGemvConfig> Fp8BlockGemvConfigFor(
    const HloScaledDotInstruction& dot,
    const se::GpuComputeCapability& gpu_version) {
  std::optional<Fp8BlockGemvSpec> spec =
      MatchScaledDotStructure(dot, [](const HloInstruction* v) { return v; });
  if (!spec.has_value()) return std::nullopt;
  if (Fp8BlockGemvBatchNeedsCutlass(spec->batch) &&
      !HasCutlassBlockGemm(gpu_version)) {
    return std::nullopt;
  }

  const bool single_row = spec->batch == 1;
  if (spec->batch > 16) {
    int64_t block_m = 128;
    while (block_m > 16 && spec->batch % block_m != 0) block_m /= 2;
    return Fp8BlockGemvConfig{block_m, /*block_n=*/128,
                              /*block_k=*/kScaleBlock, /*num_warps=*/8,
                              /*num_stages=*/3};
  }
  const int64_t block_n = single_row ? 16 : 32;
  int64_t block_m = std::min<int64_t>(spec->batch, 64);
  while (block_m > 1 && spec->batch % block_m != 0) block_m /= 2;
  return Fp8BlockGemvConfig{
      block_m, block_n, ChooseBlockK(block_n, spec->k, kMaxBlockK),
      single_row ? 8 : 4, single_row ? 3 : 4};
}

bool Fp8BlockGemvSupportsScaledDot(
    const HloScaledDotInstruction& dot,
    const se::GpuComputeCapability& gpu_version) {
  // Deliberately the same predicate the arm applies: this answer keeps the op
  // alive past the pre-layout ScaledDotRewriter, and an op kept alive that the
  // arm then declines reaches a floor that fails the compile.
  const bool supported = Fp8BlockGemvConfigFor(dot, gpu_version).has_value();
  VLOG(1) << "fp8 block gemv scaled-dot capability: supported=" << supported
          << " " << dot.ToString();
  return supported;
}

std::optional<Fp8BlockGemvSpec> MatchFp8BlockGemv(
    const HloFusionInstruction& fusion) {
  const HloComputation* computation = fusion.fused_instructions_computation();
  const HloInstruction* root = SkipBitcasts(computation->root_instruction());

  if (root->opcode() == HloOpcode::kScaledDot) {
    const auto* scaled = Cast<HloScaledDotInstruction>(root);
    const char* reason = "?";
    const bool scale_proven_identity = absl::StartsWith(
        computation->name(), kFp8BlockGemvComputationPrefix);
    std::optional<Fp8BlockGemvSpec> spec = MatchScaledDotStructure(
        *scaled,
        [&](const HloInstruction* value) -> const HloInstruction* {
          if (value->opcode() != HloOpcode::kParameter) return value;
          return fusion.operand(value->parameter_number());
        },
        &reason, scale_proven_identity);
    if (!spec.has_value()) {
      VLOG(1) << "fp8 block gemv scaled-dot no match " << fusion.name() << ": "
              << reason;
      return std::nullopt;
    }
    const HloInstruction* activation = SkipBitcasts(scaled->operand(0));
    const HloInstruction* weight = SkipBitcasts(scaled->operand(1));
    const HloInstruction* act_scale = SkipBitcasts(scaled->operand(2));
    const HloInstruction* scale = SkipBitcasts(scaled->operand(3));
    if (activation->opcode() != HloOpcode::kParameter ||
        weight->opcode() != HloOpcode::kParameter ||
        scale->opcode() != HloOpcode::kParameter ||
        (spec->w8a8 && act_scale->opcode() != HloOpcode::kParameter)) {
      VLOG(1) << "fp8 block gemv scaled-dot no match " << fusion.name()
              << ": operands not parameters (" << activation->opcode() << ", "
              << weight->opcode() << ", " << scale->opcode() << ")";
      return std::nullopt;
    }
    spec->activation_index = activation->parameter_number();
    spec->weight_index = weight->parameter_number();
    spec->scale_index = scale->parameter_number();
    spec->act_scale_index =
        spec->w8a8 ? act_scale->parameter_number() : spec->scale_index;
    return spec;
  }

  return std::nullopt;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitFp8BlockGemvXTileModule(
    absl::string_view fn_name, const HloFusionInstruction& fusion,
    const Fp8BlockGemvSpec& spec,
    const xla::xtile::BlockLevelParameters& block_level_parameters,
    mlir::MLIRContext& mlir_context) {
  if (block_level_parameters.output_tile_sizes.size() != 1 ||
      block_level_parameters.output_tile_sizes.front().size() != 2) {
    return absl::InvalidArgumentError(
        "fp8 block gemv needs a [block_m, block_n] output tile");
  }
  const int64_t block_m = block_level_parameters.output_tile_sizes.front()[0];
  const int64_t block_n = block_level_parameters.output_tile_sizes.front()[1];
  if (block_m <= 0 || spec.batch % block_m != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("fp8 block gemv needs block_m dividing m, got ", block_m,
                     " for m ", spec.batch));
  }
  if (spec.batch > 1 && block_m == 1) {
    return absl::InvalidArgumentError(
        "fp8 block gemv reduces a single row and mma's the rest; a one-row "
        "tile of a taller output would take the reduction path per row");
  }
  if (spec.batch > 1 && block_m < 16) {
    return absl::InvalidArgumentError(absl::StrCat(
        "fp8 block gemv mma tiles rows by at least 16, got ", block_m));
  }

  const HloComputation* computation = fusion.fused_instructions_computation();

  int64_t block_k = 0;
  if (absl::StatusOr<xla::xtile::Tile> tile =
          SkipBitcasts(computation->root_instruction())->backend_config<xla::xtile::Tile>();
      tile.ok() && tile->sizes_size() > 0) {
    block_k = tile->sizes(tile->sizes_size() - 1);
  }
  if (block_k <= 0 || block_k % kScaleBlock != 0 || spec.k % block_k != 0) {
    block_k = ChooseBlockK(block_n, spec.k, kMaxBlockK);
  }
  if (spec.n % block_n != 0 || spec.k % block_k != 0 ||
      block_k % kScaleBlock != 0) {
    return absl::InvalidArgumentError("fp8 block gemv needs whole tiles");
  }
  // block_n must divide the scale block: a program reads one scale row, and a
  // wider tile would dequantize its upper columns with the wrong scale.
  if (kScaleBlock % block_n != 0) {
    return absl::InvalidArgumentError(
        "fp8 block gemv needs block_n aligned to the scale grid");
  }

  VLOG(1) << "fp8 block gemv " << computation->name() << ": "
          << (spec.w8a8 ? "w8a8" : "w8a16") << " batch " << spec.batch
          << ", n " << spec.n << ", k " << spec.k << ", tile " << block_m << "x"
          << block_n << "x" << block_k << ", "
          << block_level_parameters.num_warps << " warps, "
          << block_level_parameters.num_stages << " stages";

  auto loc = mlir::NameLoc::get(
      mlir::StringAttr::get(&mlir_context, computation->name()));
  mlir::ImplicitLocOpBuilder b(loc, &mlir_context);

  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);
  b.setInsertionPointToEnd(module->getBody());

  ABSL_ASSIGN_OR_RETURN(auto fn_arg_types,
                   xtile::GetFnArgTypes(b, fusion, /*opaque_args_types=*/{},
                                        /*gpu_cc=*/std::nullopt));
  llvm::SmallVector<mlir::NamedAttribute> named_attributes{
      b.getNamedAttr("num_opaque_args", b.getI32IntegerAttr(0))};
  auto fn = xtile::EntryFuncOp::create(b, fn_name, fn_arg_types,
                                       named_attributes, {});
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  mlir::ValueRange buffers = fn.getBufferArgs();
  Value activation = buffers[spec.activation_index];
  Value weight = buffers[spec.weight_index];
  Value scale = buffers[spec.scale_index];
  Value act_scale = spec.w8a8 ? buffers[spec.act_scale_index] : Value();
  Value output = buffers[computation->num_parameters()];

  mlir::Type f32 = b.getF32Type();
  auto index = [&](int64_t v) -> Value {
    return mlir::arith::ConstantIndexOp::create(b, v);
  };
  auto mul_index = [&](Value v, int64_t c) -> Value {
    return mlir::arith::MulIOp::create(b, v, index(c));
  };
  auto add_index = [&](Value v, int64_t c) -> Value {
    return mlir::arith::AddIOp::create(b, v, index(c));
  };
  auto tensor_of = [&](Value buffer, llvm::ArrayRef<int64_t> sizes) {
    return mlir::RankedTensorType::get(
        sizes,
        mlir::cast<mlir::MemRefType>(buffer.getType()).getElementType());
  };
  auto extract = [&](Value buffer, llvm::ArrayRef<Value> offsets,
                     llvm::ArrayRef<int64_t> sizes) -> Value {
    return xtile::ExtractTileOp::create(
        b, tensor_of(buffer, sizes), buffer, mlir::ValueRange(offsets),
        llvm::SmallVector<int64_t>(sizes.begin(), sizes.end()),
        llvm::SmallVector<int64_t>(sizes.size(), 1));
  };
  auto reshape = [&](Value v, llvm::ArrayRef<int64_t> sizes) -> Value {
    return mlir::stablehlo::ReshapeOp::create(
        b,
        mlir::RankedTensorType::get(
            sizes, mlir::cast<mlir::RankedTensorType>(v.getType())
                       .getElementType()),
        v);
  };
  auto broadcast = [&](Value v, llvm::ArrayRef<int64_t> sizes,
                       llvm::ArrayRef<int64_t> dims) -> Value {
    return mlir::stablehlo::BroadcastInDimOp::create(
        b,
        mlir::RankedTensorType::get(
            sizes, mlir::cast<mlir::RankedTensorType>(v.getType())
                       .getElementType()),
        v, llvm::SmallVector<int64_t>(dims.begin(), dims.end()));
  };
  // A UE8M0 scale is 2^(e-127): its f32 bit pattern is the exponent byte
  // shifted into the exponent field. Triton has no e8m0 type, so the byte is
  // reinterpreted as an integer first.
  auto scale_to_f32 = [&](Value tile) -> Value {
    auto type = mlir::cast<mlir::RankedTensorType>(tile.getType());
    if (!mlir::isa<mlir::Float8E8M0FNUType>(type.getElementType())) {
      return xtile::Cast(b, tile, f32);
    }
    Value bits = mlir::arith::BitcastOp::create(
        b, type.clone(b.getI8Type()), tile);
    Value wide =
        mlir::arith::ExtUIOp::create(b, type.clone(b.getI32Type()), bits);
    Value shifted = mlir::arith::ShLIOp::create(
        b, wide, CreateConst(b, b.getI32Type(), 23, type.getShape()));
    // tensor.bitcast is the spelling that reaches tt.bitcast; arith.bitcast
    // only survives where the type conversion pass folds it away.
    return mlir::tensor::BitcastOp::create(b, type.clone(f32), shifted);
  };
  // e4m3 to f16 is exact and converts two elements per instruction.
  auto fp8_to_f32 = [&](Value tile) -> Value {
    return xtile::Cast(b, xtile::Cast(b, tile, b.getF16Type()), f32);
  };

  const int64_t n_tiles = spec.n / block_n;
  Value pid = fn.getProgramId();
  Value m0 = mul_index(mlir::arith::DivUIOp::create(b, pid, index(n_tiles)),
                       block_m);
  Value n0 = mul_index(mlir::arith::RemUIOp::create(b, pid, index(n_tiles)),
                       block_n);
  Value scale_row = mlir::arith::DivUIOp::create(b, n0, index(kScaleBlock));

  const bool is_single_row = spec.batch == 1;
  Value acc_init =
      is_single_row ? CreateConst(b, f32, 0.0f, {block_n, 1})
                    : CreateConst(b, f32, 0.0f, {block_m, block_n});
  auto loop = mlir::scf::ForOp::create(b, index(0), index(spec.k / block_k),
                                       index(1), mlir::ValueRange{acc_init});
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value acc = loop.getRegionIterArg(0);
    Value k0 = mul_index(loop.getInductionVar(), block_k);
    const int64_t scale_cols = block_k / kScaleBlock;
    Value scale_col = mlir::arith::DivUIOp::create(b, k0, index(kScaleBlock));

    // One scale row [1, scale_cols], expanded to one value per column of the
    // contracting tile.
    auto expand_scale_row = [&](Value buffer, Value row) -> Value {
      Value tile = extract(buffer, {row, scale_col}, {1, scale_cols});
      Value vec = scale_to_f32(reshape(tile, {scale_cols}));
      Value wide = broadcast(vec, {scale_cols, kScaleBlock}, {0});
      return reshape(wide, {block_k});
    };

    const llvm::SmallVector<int64_t, 2> act_shape =
        spec.activation_batch_major
            ? llvm::SmallVector<int64_t, 2>{block_m, block_k}
            : llvm::SmallVector<int64_t, 2>{block_k, block_m};
    auto load_act_tile = [&]() -> Value {
      if (spec.activation_batch_major) {
        return extract(activation, {m0, k0}, act_shape);
      }
      return extract(activation, {k0, m0}, act_shape);
    };

    Value next;
    if (is_single_row) {
      Value scale_wide = expand_scale_row(scale, scale_row);
      Value act_row = reshape(load_act_tile(), {block_k});
      Value act_f32 =
          spec.w8a8 ? fp8_to_f32(act_row) : xtile::Cast(b, act_row, f32);
      Value scaled_act = mlir::stablehlo::MulOp::create(b, act_f32, scale_wide);
      if (spec.w8a8) {
        scaled_act = mlir::stablehlo::MulOp::create(
            b, scaled_act, expand_scale_row(act_scale, m0));
      }
      Value act_2d = broadcast(scaled_act, {block_n, block_k}, {1});
      Value weight_f32 =
          fp8_to_f32(extract(weight, {n0, k0}, {block_n, block_k}));
      Value product = mlir::stablehlo::MulOp::create(b, weight_f32, act_2d);

      Value zero = CreateConst(b, f32, 0.0f, {});
      auto reduction = mlir::stablehlo::ReduceOp::create(
          b, mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(product),
          zero, llvm::SmallVector<int64_t>{1});
      {
        mlir::OpBuilder::InsertionGuard reducer_guard(b);
        auto scalar = mlir::RankedTensorType::get({}, f32);
        mlir::Block* reducer = b.createBlock(&reduction->getRegion(0), {},
                                             {scalar, scalar}, {loc, loc});
        b.setInsertionPointToStart(reducer);
        mlir::stablehlo::ReturnOp::create(
            b, mlir::SmallVector<Value>{mlir::stablehlo::AddOp::create(
                   b, reducer->getArgument(0), reducer->getArgument(1))});
      }
      Value reduced = reshape(reduction.getResult(0), {block_n, 1});
      next = mlir::stablehlo::AddOp::create(b, acc, reduced);
    } else {
      auto dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
          b.getContext(), /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{1},
          /*rhsContractingDimensions=*/{1});
      // As the generic dot emitter spells it: a null precision config does not
      // lower, and the accumulate must be an arith add to reach Triton.
      auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
          b.getContext(), {mlir::stablehlo::Precision::DEFAULT,
                           mlir::stablehlo::Precision::DEFAULT});
      auto dot = [&](Value lhs, Value rhs) -> Value {
        return mlir::stablehlo::DotGeneralOp::create(
            b, mlir::RankedTensorType::get({block_m, block_n}, f32), lhs, rhs,
            dims, precision_config, /*algorithm=*/nullptr);
      };
      if (!spec.w8a8) {
        Value scale_wide = expand_scale_row(scale, scale_row);
        Value weight_bf16 = xtile::Cast(
            b, extract(weight, {n0, k0}, {block_n, block_k}), b.getBF16Type());
        Value act_tile = load_act_tile();
        Value act_t =
            spec.activation_batch_major
                ? act_tile
                : mlir::stablehlo::TransposeOp::create(
                      b,
                      mlir::RankedTensorType::get({block_m, block_k},
                                                  b.getBF16Type()),
                      act_tile, llvm::SmallVector<int64_t>{1, 0})
                      .getResult();
        Value scale_2d = broadcast(scale_wide, {block_m, block_k}, {1});
        Value act_bf16 = xtile::Cast(
            b,
            mlir::stablehlo::MulOp::create(b, xtile::Cast(b, act_t, f32),
                                           scale_2d),
            b.getBF16Type());
        next = mlir::arith::AddFOp::create(b, acc, dot(act_bf16, weight_bf16));
      } else {
        // One fp8 x fp8 dot per 128-deep scale block, its product scaled on
        // the accumulator by the row scale times the tile's weight scale.
        next = acc;
        for (int64_t j = 0; j < scale_cols; ++j) {
          Value kj = add_index(k0, j * kScaleBlock);
          Value kbj = add_index(scale_col, j);
          Value act_tile = extract(activation, {m0, kj}, {block_m, kScaleBlock});
          Value weight_tile =
              extract(weight, {n0, kj}, {block_n, kScaleBlock});
          Value zero = CreateConst(b, f32, 0.0f, {block_m, block_n});
          Value partial =
              mlir::arith::AddFOp::create(b, zero, dot(act_tile, weight_tile));
          Value row_scale =
              scale_to_f32(extract(act_scale, {m0, kbj}, {block_m, 1}));
          Value tile_scale = scale_to_f32(
              reshape(extract(scale, {scale_row, kbj}, {1, 1}), {1}));
          Value s = mlir::stablehlo::MulOp::create(
              b, row_scale, broadcast(tile_scale, {block_m, 1}, {1}));
          Value s_2d = broadcast(reshape(s, {block_m}), {block_m, block_n}, {0});
          next = mlir::arith::AddFOp::create(
              b, next, mlir::stablehlo::MulOp::create(b, partial, s_2d));
        }
      }
    }
    mlir::scf::YieldOp::create(b, mlir::ValueRange{next});
  }

  auto out_element = mlir::cast<mlir::MemRefType>(output.getType())
                         .getElementType();
  Value result = xtile::Cast(b, loop.getResult(0), out_element);
  if (is_single_row) {
    result = mlir::stablehlo::TransposeOp::create(
        b, mlir::RankedTensorType::get({block_m, block_n}, out_element),
        result, llvm::SmallVector<int64_t>{1, 0});
  }
  xtile::InsertTileOp::create(b, result, output,
                              mlir::ValueRange{m0, n0},
                              llvm::SmallVector<int64_t>{block_m, block_n},
                              llvm::SmallVector<int64_t>{1, 1});

  b.create<xtile::EntryFuncReturnOp>();

  {
    mlir::PassManager pm(&mlir_context);
    pm.addPass(xtile::createVerifyLegalXTileOpsPass());
    tsl::StatusScopedDiagnosticHandler diagnostic_handler(&mlir_context);
    ABSL_RETURN_IF_ERROR(diagnostic_handler.consumeStatus(pm.run(*module)));
  }
  return module;
}

}  // namespace xla::gpu
