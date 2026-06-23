/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h"

#include <cstdint>
#include <memory>
#include <string>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "google/protobuf/text_format.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/transforms/lower_to_llvm_common.h"
#include "xla/codegen/emitters/transforms/lowering_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace emitters {

// Apple Metal/AIR: thread positions are kernel ARGUMENTS tagged in the
// !air.kernel metadata (air.thread_position_*), NOT readable special-register
// intrinsics like NVVM/AMDGPU. A dialect-conversion pattern cannot append a
// function argument (FuncToLLVM seeds its signature conversion with the existing
// arg count), so — before the conversion runs — we imperatively (1) append two
// i32 params [tid, tgid] to each `xla.entry` kernel and (2) rewrite
// gpu.thread_id -> index_cast(tid) and gpu.block_id -> index_cast(tgid) for the
// x dimension (y/z collapse to 0: the loop launch is 1-D). XLA then reconstructs
// the global linear index as `tid + tgid*num_work_items`, which equals Metal's
// thread_position_in_grid as long as the dispatch threadgroup size ==
// num_work_items (both derive from the fusion's LaunchDimensions). The matching
// !air.kernel metadata for the two new args is attached after translation, in
// MetalMlirKernelFusion::CreateLLVMModule (via AttachAirKernelMetadata).
//
// This mirrors the Intel arm's in-lambda module mutation; after it runs no gpu
// dialect ops remain, so the generic arith/func/math lowering finishes the
// module without any gpu->target conversion.
void InjectAirThreadPositions(mlir::ModuleOp module_op) {
  mlir::OpBuilder builder(module_op.getContext());
  mlir::Type i32 = builder.getI32Type();
  module_op.walk([&](mlir::func::FuncOp fn) {
    if (!fn->hasAttr("xla.entry") || fn.getBody().empty()) {
      return;
    }
    unsigned tid_index = fn.getNumArguments();
    auto role_attrs = [&](const char* role) {
      return builder.getDictionaryAttr({builder.getNamedAttr(
          "xla.air_role", builder.getStringAttr(role))});
    };
    mlir::Location loc = fn.getLoc();
    fn.insertArgument(tid_index, i32,
                      role_attrs("thread_position_in_threadgroup"), loc);
    fn.insertArgument(tid_index + 1, i32,
                      role_attrs("threadgroup_position_in_grid"), loc);
    mlir::Value tid = fn.getArgument(tid_index);
    mlir::Value tgid = fn.getArgument(tid_index + 1);

    llvm::SmallVector<mlir::Operation*> to_erase;
    fn.walk([&](mlir::Operation* op) {
      mlir::Value src;
      bool is_x = false;
      if (auto t = mlir::dyn_cast<mlir::gpu::ThreadIdOp>(op)) {
        is_x = t.getDimension() == mlir::gpu::Dimension::x;
        src = tid;
      } else if (auto b = mlir::dyn_cast<mlir::gpu::BlockIdOp>(op)) {
        is_x = b.getDimension() == mlir::gpu::Dimension::x;
        src = tgid;
      } else {
        return;
      }
      mlir::OpBuilder b(op);
      mlir::Value idx =
          is_x ? b.create<mlir::arith::IndexCastOp>(op->getLoc(),
                                                    b.getIndexType(), src)
                     .getResult()
               : b.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 0)
                     .getResult();
      op->getResult(0).replaceAllUsesWith(idx);
      to_erase.push_back(op);
    });
    for (mlir::Operation* op : to_erase) {
      op->erase();
    }
  });
}

// Metal/AIR: kernel buffer arguments must be `ptr addrspace(1)` (device memory)
// and accessed as addrspace(1) throughout, but LowerTensors creates kernel
// pointer args + their GEPs in addrspace(0) (lower_tensors.cc — NVVM relies on
// its kernel calling convention to treat them as global; AIR has no such
// convention and a device buffer accessed via addrspace(0) reads thread/private
// memory -> all-zero output). Promote each xla.entry pointer arg and every
// pointer SSA value transitively derived from it (GEPs) to addrspace(1), in
// place (MLIR Value::setType). Done before the func/arith->LLVM conversion so
// FuncToLLVM carries the addrspace through. (Opaque pointers: GEP result
// addrspace = base operand addrspace; load/store inherit the operand's space, so
// only args + GEP results need retyping.)
//
// This is INTERPROCEDURAL: a fusion's entry passes its device buffers into
// subroutine/reducer clones (func.call, after convert-pure-call-ops), e.g.
// softmax's fused_divide_exponential clone takes the input buffer as a `ptr`
// parameter. Promoting only the entry leaves that callee's parameter
// addrspace(0), so the entry passes an addrspace(1) pointer to an addrspace(0)
// parameter and the LLVM verifier rejects the module ("Call parameter type does
// not match function signature" -> "Failed to translate module to LLVM IR").
// So we run a module-wide fixpoint that also pushes addrspace(1) across call
// edges into callee parameters (and then through their GEPs / further calls).
void PromoteEntryBuffersToDevice(mlir::ModuleOp module_op) {
  mlir::MLIRContext* ctx = module_op.getContext();
  auto ptr1 = mlir::LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
  auto addr_space_of = [](mlir::Type t) -> int {
    auto p = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(t);
    return p ? static_cast<int>(p.getAddressSpace()) : -1;
  };
  mlir::SymbolTable symtab(module_op);

  // Set parameter `i` of `fn` (its function type + entry-block argument) to
  // addrspace(1). Idempotent: a param already addrspace(1) is left alone.
  auto promote_param = [&](mlir::func::FuncOp fn, unsigned i) {
    llvm::SmallVector<mlir::Type> inputs(fn.getFunctionType().getInputs());
    inputs[i] = ptr1;
    fn.setType(mlir::FunctionType::get(ctx, inputs,
                                       fn.getFunctionType().getResults()));
    if (!fn.getBody().empty()) {
      fn.getBody().front().getArgument(i).setType(ptr1);
    }
  };

  // Seed: every xla.entry pointer arg (the device buffers) -> addrspace(1).
  module_op.walk([&](mlir::func::FuncOp fn) {
    if (!fn->hasAttr("xla.entry") || fn.getBody().empty()) return;
    auto inputs = fn.getFunctionType().getInputs();
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (addr_space_of(inputs[i]) == 0) promote_param(fn, i);
    }
  });

  // Module-wide fixpoint: (1) push addrspace(1) through GEP chains within each
  // function; (2) propagate it across call edges into callee parameters. A
  // newly-promoted callee parameter feeds (1) on the next round (its GEPs), and
  // if that callee calls a deeper clone, (2) propagates further — so the loop
  // converges over an arbitrary call graph.
  bool changed = true;
  while (changed) {
    changed = false;
    module_op.walk([&](mlir::LLVM::GEPOp gep) {
      if (addr_space_of(gep.getBase().getType()) == 1 &&
          addr_space_of(gep.getResult().getType()) == 0) {
        gep.getResult().setType(ptr1);
        changed = true;
      }
    });
    module_op.walk([&](mlir::func::CallOp call) {
      auto callee = symtab.lookup<mlir::func::FuncOp>(call.getCallee());
      if (!callee || callee.getBody().empty()) return;
      auto params = callee.getFunctionType().getInputs();
      auto args = call.getArgOperands();
      for (unsigned i = 0; i < args.size() && i < params.size(); ++i) {
        if (addr_space_of(args[i].getType()) == 1 &&
            addr_space_of(params[i]) == 0) {
          promote_param(callee, i);
          changed = true;
        }
      }
    });
  }
}

// Metal/AIR: a REAL warp reduction (MlirReductionFusion) lowers (via
// lower_xla_to_scf) to `gpu.shuffle{down}` + `gpu.barrier` ops, which the
// GpuToAIR arm must legalize. There is no MLIR gpu->AIR conversion, so rewrite
// them imperatively to bare external AIR calls (probe-verified spellings):
//   gpu.barrier            -> air.wg.barrier(i32 2 [mem_threadgroup], i32 1)
//   gpu.shuffle down v,d,w -> air.simd_shuffle_down.f32(f32 v, i16 trunc(d))
// The Apple SIMD width is hardwired to 32 so the `width` operand is dropped; the
// `valid` result is unused by the reduction so it is replaced with `true`. The
// caller's lane is implicit in the SIMD hardware, and MlirReductionFusion derives
// lane/warp identity arithmetically from thread_id (no gpu.lane_id/subgroup_id),
// so NO simdgroup position kernel args are needed. The air.* decls + the kernel
// are marked Convergent in the Metal CreateLLVMModule tail so air-opt --O3 won't
// reorder across the barrier/shuffle. f32 accumulators shuffle directly; i32
// (argmax index) bitcasts through the f32 builtin (data movement, type-agnostic);
// other element types/modes are left to fail loud.
void RewriteGpuWarpOpsToAir(mlir::ModuleOp module_op) {
  mlir::MLIRContext* ctx = module_op.getContext();
  auto i16 = mlir::IntegerType::get(ctx, 16);
  auto i32 = mlir::IntegerType::get(ctx, 32);
  auto f32 = mlir::Float32Type::get(ctx);
  auto void_ty = mlir::LLVM::LLVMVoidType::get(ctx);

  auto get_or_insert = [&](llvm::StringRef name, mlir::Type result,
                           llvm::ArrayRef<mlir::Type> args)
      -> mlir::LLVM::LLVMFuncOp {
    if (auto f = module_op.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
      return f;
    }
    mlir::OpBuilder b = mlir::OpBuilder::atBlockBegin(module_op.getBody());
    auto fty = mlir::LLVM::LLVMFunctionType::get(result, args,
                                                 /*isVarArg=*/false);
    return b.create<mlir::LLVM::LLVMFuncOp>(module_op.getLoc(), name, fty);
  };

  module_op.walk([&](mlir::func::FuncOp fn) {
    if (!fn->hasAttr("xla.entry") || fn.getBody().empty()) {
      return;
    }
    llvm::SmallVector<mlir::Operation*> to_erase;
    fn.walk([&](mlir::Operation* op) {
      if (auto bar = mlir::dyn_cast<mlir::gpu::BarrierOp>(op)) {
        mlir::OpBuilder b(op);
        auto fn_barrier = get_or_insert("air.wg.barrier", void_ty, {i32, i32});
        mlir::Value c2 = b.create<mlir::LLVM::ConstantOp>(
            op->getLoc(), i32, b.getI32IntegerAttr(2));
        mlir::Value c1 = b.create<mlir::LLVM::ConstantOp>(
            op->getLoc(), i32, b.getI32IntegerAttr(1));
        b.create<mlir::LLVM::CallOp>(op->getLoc(), fn_barrier,
                                     mlir::ValueRange{c2, c1});
        to_erase.push_back(op);
      } else if (auto shf = mlir::dyn_cast<mlir::gpu::ShuffleOp>(op)) {
        mlir::Type vty = shf.getValue().getType();
        // Sum/max reductions shuffle f32 accumulators; an ARGMAX (e.g. the
        // lm_head greedy/topk-1 vocab reduction) also shuffles the i32 index
        // lane-to-lane. simd_shuffle is pure 32-bit data movement, so route i32
        // through the (numerically verified) f32 builtin by bitcasting both ways
        // — no separate i32 builtin to trust (all air.simd_shuffle_down.*i32
        // spellings pipeline-create regardless of correctness; probe can't tell
        // them apart). Other modes/widths fail loud.
        if (shf.getMode() != mlir::gpu::ShuffleMode::DOWN ||
            (vty != f32 && vty != i32)) {
          return;
        }
        mlir::OpBuilder b(op);
        mlir::Value in = shf.getValue();
        if (vty == i32) {
          in = b.create<mlir::arith::BitcastOp>(op->getLoc(), f32, in);
        }
        mlir::Value delta =
            b.create<mlir::arith::TruncIOp>(op->getLoc(), i16, shf.getOffset());
        auto fn_shuffle =
            get_or_insert("air.simd_shuffle_down.f32", f32, {f32, i16});
        mlir::Value res =
            b.create<mlir::LLVM::CallOp>(op->getLoc(), fn_shuffle,
                                         mlir::ValueRange{in, delta})
                .getResult();
        if (vty == i32) {
          res = b.create<mlir::arith::BitcastOp>(op->getLoc(), i32, res);
        }
        shf.getShuffleResult().replaceAllUsesWith(res);
        mlir::Value vtrue =
            b.create<mlir::arith::ConstantOp>(op->getLoc(), b.getBoolAttr(true));
        shf.getValid().replaceAllUsesWith(vtrue);
        to_erase.push_back(op);
      }
    });
    for (mlir::Operation* op : to_erase) {
      op->erase();
    }
  });
}

// Metal/AIR: air-as assembles AND metallib emits the standard llvm.* math
// intrinsics, but Apple's GPU driver then CRASHES compiling some of them to GPU
// ISA at pipeline-state creation (newComputePipelineState fails "Metal failed:
// Internal" / XPC_ERROR_CONNECTION_INTERRUPTED), unlike exp/log/sqrt which lower
// fine. So rewrite those specific math ops to the bare external air.fast_*.f32
// builtins (probe-verified) before the conversion turns them into the
// unsupported intrinsics.
//
// Probe (metal-xla-docs/scratch/air-ref/probe_f16_intrinsics.sh) on this
// toolchain: llvm.{maximum,minimum,sin,cos}.{f32,f16} all crash the driver;
// llvm.{maxnum,minnum,exp,log,sqrt,fabs}.{f32,f16} are fine. So we rewrite
// maximumf/minimumf/sin/cos for f32 AND for f16/bf16 (the crash is dtype-wide).
// arith.maximumf/minimumf lower to llvm.maximum/minimum (NaN-propagating,
// signed-zero-aware), the crashing variant — not the safe llvm.maxnum/minnum.
// f16/bf16 are handled by widening operands to f32, calling the f32 builtin, and
// fptrunc'ing back: the air.fast_*.f32 result is rounded to the narrow dtype
// (≥ as accurate as a native f16 sin, and exact for max/min). air.fast_*.f16
// builtins do exist, but the widen path is one uniform rule for both narrow
// dtypes (no bf16 builtin needed) and matches the f32 compute the emitter
// already uses elsewhere.
//
// This MUST visit EVERY function with a body, not just xla.entry: a reduction's
// reducer is emitted as a separate clone function (region_*_clone_maximum_*)
// that the entry calls, and the max/min lives THERE. Restricting to xla.entry
// left reducer-clone maximumf/minimumf as llvm.maximum -> every max/min
// REDUCTION (e.g. softmax's max, max-reductions) crashed the driver at
// pipeline-state creation, while bare elementwise max (op in the entry) worked.
void RewriteMathToAir(mlir::ModuleOp module_op) {
  mlir::MLIRContext* ctx = module_op.getContext();
  auto f32 = mlir::Float32Type::get(ctx);
  auto decl = [&](llvm::StringRef name, unsigned nargs) -> mlir::LLVM::LLVMFuncOp {
    if (auto f = module_op.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) return f;
    mlir::OpBuilder b = mlir::OpBuilder::atBlockBegin(module_op.getBody());
    llvm::SmallVector<mlir::Type> args(nargs, f32);
    return b.create<mlir::LLVM::LLVMFuncOp>(
        module_op.getLoc(), name,
        mlir::LLVM::LLVMFunctionType::get(f32, args, /*isVarArg=*/false));
  };
  module_op.walk([&](mlir::func::FuncOp fn) {
    if (fn.getBody().empty()) return;  // entry AND reducer clones (see above).
    llvm::SmallVector<mlir::Operation*> to_erase;
    fn.walk([&](mlir::Operation* op) {
      mlir::OpBuilder b(op);
      auto emit = [&](llvm::StringRef name, mlir::ValueRange args,
                      mlir::Value result) {
        mlir::Type rt = result.getType();
        // Scalar f32 directly; scalar f16/bf16 by widening to f32 and narrowing
        // the result back. Vectors / other types: leave as-is (none reach here).
        const bool narrow = rt.isF16() || rt.isBF16();
        if (!rt.isF32() && !narrow) return false;
        llvm::SmallVector<mlir::Value> f32args;
        f32args.reserve(args.size());
        for (mlir::Value a : args) {
          f32args.push_back(a.getType() == f32
                                ? a
                                : b.create<mlir::arith::ExtFOp>(op->getLoc(),
                                                                f32, a)
                                      .getResult());
        }
        mlir::Value r =
            b.create<mlir::LLVM::CallOp>(op->getLoc(),
                                         decl(name, f32args.size()), f32args)
                .getResult();
        if (narrow) {
          r = b.create<mlir::arith::TruncFOp>(op->getLoc(), rt, r).getResult();
        }
        result.replaceAllUsesWith(r);
        to_erase.push_back(op);
        return true;
      };
      if (auto m = mlir::dyn_cast<mlir::arith::MaximumFOp>(op)) {
        emit("air.fast_fmax.f32", {m.getLhs(), m.getRhs()}, m.getResult());
      } else if (auto m = mlir::dyn_cast<mlir::arith::MinimumFOp>(op)) {
        emit("air.fast_fmin.f32", {m.getLhs(), m.getRhs()}, m.getResult());
      } else if (auto s = mlir::dyn_cast<mlir::math::SinOp>(op)) {
        emit("air.fast_sin.f32", {s.getOperand()}, s.getResult());
      } else if (auto c = mlir::dyn_cast<mlir::math::CosOp>(op)) {
        emit("air.fast_cos.f32", {c.getOperand()}, c.getResult());
      }
    });
    for (mlir::Operation* op : to_erase) op->erase();
  });
}

// Metal/AIR: lower_tensors lowers threadgroup shared memory (AllocateSharedOp) as
// an addrspace(3) global addrspacecast'd to the generic addrspace(0) and accessed
// through that. But AIR addrspace(0) is thread/private, not a generic space that
// reaches threadgroup memory, so multi-warp reductions read zero. Eliminate every
// addrspacecast(N->0) for N in {1 device, 3 threadgroup} by replacing the cast
// result with its real-addrspace source, then fix every GEP so its result
// addrspace matches its base (so loads/stores happen in the real addrspace).
void FixupAirPointerAddrSpaces(mlir::ModuleOp module_op) {
  mlir::MLIRContext* ctx = module_op.getContext();
  auto as_ptr = [](mlir::Type t) {
    return mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(t);
  };
  llvm::SmallVector<mlir::LLVM::AddrSpaceCastOp> casts;
  module_op.walk([&](mlir::LLVM::AddrSpaceCastOp c) {
    auto src = as_ptr(c.getArg().getType());
    auto res = as_ptr(c.getResult().getType());
    if (src && res && res.getAddressSpace() == 0 &&
        (src.getAddressSpace() == 1 || src.getAddressSpace() == 3)) {
      casts.push_back(c);
    }
  });
  for (auto c : casts) {
    c.getResult().replaceAllUsesWith(c.getArg());
    c.erase();
  }
  bool again = true;
  while (again) {
    again = false;
    module_op.walk([&](mlir::LLVM::GEPOp gep) {
      auto base = as_ptr(gep.getBase().getType());
      auto res = as_ptr(gep.getResult().getType());
      if (base && res && base.getAddressSpace() != res.getAddressSpace()) {
        gep.getResult().setType(mlir::LLVM::LLVMPointerType::get(
            ctx, base.getAddressSpace()));
        again = true;
      }
    });
  }
}

#define GEN_PASS_DEF_LOWERTOLLVMGPUPASS
#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h.inc"

namespace {

namespace se = ::stream_executor;

// log2(e), used to express exp(x) = exp2(x * log2(e)).
constexpr double kLog2E = 1.4426950408889634;

// Lowers a scalar bf16 `math.exp2` to the native gfx1250 `v_exp_bf16`
// instruction via the `llvm.amdgcn.exp2` intrinsic. Without this, the default
// MathToROCDL lowering upcasts bf16 to f32 and calls `__ocml_exp2_f32`, never
// using the hardware bf16 transcendental unit. Vector ops are scalarized first
// by MathToROCDL's ScalarizeVectorOpLowering (lower benefit), so this pattern
// only needs to handle the scalar case.
struct Exp2BF16ToAMDGPU
    : public mlir::ConvertOpToLLVMPattern<mlir::math::Exp2Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::Exp2Op op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 exp2");
    }
    mlir::Value operand = adaptor.getOperands().front();
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, /*resultType=*/operand.getType(),
        rewriter.getStringAttr("llvm.amdgcn.exp2"), mlir::ValueRange{operand});
    return mlir::success();
  }
};

// Lowers a scalar bf16 `math.exp` to the native gfx1250 `v_exp_bf16`
// instruction by rewriting exp(x) = exp2(x * log2(e)) and emitting the
// `llvm.amdgcn.exp2` intrinsic. See Exp2BF16ToAMDGPU for the rationale.
struct ExpBF16ToAMDGPU
    : public mlir::ConvertOpToLLVMPattern<mlir::math::ExpOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 exp");
    }
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperands().front();
    mlir::Type bf16 = operand.getType();
    mlir::Value log2e = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, bf16, rewriter.getFloatAttr(bf16, kLog2E));
    mlir::Value scaled =
        rewriter.create<mlir::LLVM::FMulOp>(loc, operand, log2e);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, /*resultType=*/bf16, rewriter.getStringAttr("llvm.amdgcn.exp2"),
        mlir::ValueRange{scaled});
    return mlir::success();
  }
};

class LowerToLLVMGPUPass
    : public impl::LowerToLLVMGPUPassBase<LowerToLLVMGPUPass> {
 public:
  LowerToLLVMGPUPass() = default;

  explicit LowerToLLVMGPUPass(const LowerToLLVMGPUPassOptions& options)
      : LowerToLLVMGPUPassBase(options) {}

  explicit LowerToLLVMGPUPass(const se::DeviceDescription& device_description)
      : device_spec_(device_description) {}

  void runOnOperation() override {
    if (!gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      absl::StatusOr<se::DeviceDescription> device_description =
          se::DeviceDescription::FromProto(device_info);
      CHECK_OK(device_description.status());
      *device_spec_.mutable_type() = *device_description;
    }

    auto populate_patterns =
        [&](mlir::LLVMTypeConverter& converter,
            mlir::RewritePatternSet& patterns,
            mlir::ConversionTarget& target) -> mlir::LogicalResult {
      if (device_spec_.IsAmdGpu()) {
        std::string chipset =
            device_spec_.gpu().rocm_compute_capability().gfx_version();
        llvm::FailureOr<mlir::amdgpu::Chipset> maybeChipset =
            mlir::amdgpu::Chipset::parse(chipset);
        if (mlir::failed(maybeChipset)) {
          mlir::emitError(mlir::UnknownLoc::get(&getContext()),
                          "Invalid chipset name: " + chipset);
          return mlir::failure();
        }
        mlir::populateGpuToROCDLConversionPatterns(
            converter, patterns, mlir::gpu::amd::Runtime::Unknown,
            *maybeChipset);
        mlir::configureGpuToROCDLConversionLegality(target);
        mlir::populateAMDGPUToROCDLConversionPatterns(converter, patterns,
                                                      *maybeChipset);
        // On gfx1250 emit native bf16 exp via v_exp_bf16 instead of upcasting
        // to f32 and calling __ocml_exp(2)_f32. Higher benefit than the default
        // MathToROCDL patterns so it wins for scalar bf16 ops.
        if (device_spec_.gpu()
                .rocm_compute_capability()
                .has_bf16_transcendental_support()) {
          patterns.add<ExpBF16ToAMDGPU, Exp2BF16ToAMDGPU>(
              converter, /*benefit=*/mlir::PatternBenefit(2));
        }
        target.addIllegalDialect<mlir::amdgpu::AMDGPUDialect>();
      } else if (device_spec_.IsIntelGpu()) {
        // Add sub-group-size attribute to functions.
        int32_t sub_group_size = device_spec_.gpu().threads_per_warp();
        if (auto module_op = mlir::dyn_cast<mlir::ModuleOp>(getOperation())) {
          module_op.walk([sub_group_size](mlir::func::FuncOp func) {
            if (!func.getBody().empty()) {
              mlir::OpBuilder b(func.getContext());
              auto sub_group_attr = b.getI32IntegerAttr(sub_group_size);
              func->setAttr("intel_reqd_sub_group_size", sub_group_attr);
            }
          });
        }
        populateGpuToLLVMSPVConversionPatterns(converter, patterns);
        spirv::populateMathToLLVMSPVConversionPatterns(spirv::getSPIRVMathOps(),
                                                       converter, patterns);
        populateGpuMemorySpaceAttributeConversions(converter);
      } else if (device_spec_.IsMetal()) {
        // Metal/AIR: no MLIR gpu->target conversion exists. Inject the AIR
        // thread-position kernel args + rewrite gpu.thread_id/gpu.block_id reads
        // imperatively here (a conversion pattern cannot append a func arg).
        // After this no gpu ops remain, so the generic arith/func/math lowering
        // completes the module; the !air.kernel metadata for the new args is
        // attached post-translation in MetalMlirKernelFusion::CreateLLVMModule.
        // NOTE: must precede the NVVM `else` — Metal masquerades as CUDA {7,5}
        // so IsNvidiaGpu() is also true.
        if (auto module_op = mlir::dyn_cast<mlir::ModuleOp>(getOperation())) {
          InjectAirThreadPositions(module_op);
          PromoteEntryBuffersToDevice(module_op);
          RewriteGpuWarpOpsToAir(module_op);
          RewriteMathToAir(module_op);
          FixupAirPointerAddrSpaces(module_op);
        }
      } else {
        mlir::populateGpuToNVVMConversionPatterns(converter, patterns);
        mlir::configureGpuToNVVMConversionLegality(target);
      }
      return mlir::success();
    };

    if (mlir::failed(LowerToLLVM(getOperation(), populate_patterns))) {
      signalPassFailure();
      return;
    }

    if (device_spec_.IsAmdGpu()) {
      EnsureAMDGPUAllocasUseAS5(getOperation());
    }
  }

 private:
  DeviceSpec device_spec_;
};

}  // namespace

std::unique_ptr<::mlir::Pass> createLowerToLLVMGPUPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<LowerToLLVMGPUPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
