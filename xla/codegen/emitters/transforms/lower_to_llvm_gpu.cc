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

void PromoteEntryBuffersToDevice(mlir::ModuleOp module_op) {
  mlir::MLIRContext* ctx = module_op.getContext();
  auto ptr1 = mlir::LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
  auto addr_space_of = [](mlir::Type t) -> int {
    auto p = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(t);
    return p ? static_cast<int>(p.getAddressSpace()) : -1;
  };
  mlir::SymbolTable symtab(module_op);

  auto promote_param = [&](mlir::func::FuncOp fn, unsigned i) {
    llvm::SmallVector<mlir::Type> inputs(fn.getFunctionType().getInputs());
    inputs[i] = ptr1;
    fn.setType(mlir::FunctionType::get(ctx, inputs,
                                       fn.getFunctionType().getResults()));
    if (!fn.getBody().empty()) {
      fn.getBody().front().getArgument(i).setType(ptr1);
    }
  };

  module_op.walk([&](mlir::func::FuncOp fn) {
    if (!fn->hasAttr("xla.entry") || fn.getBody().empty()) return;
    auto inputs = fn.getFunctionType().getInputs();
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (addr_space_of(inputs[i]) == 0) promote_param(fn, i);
    }
  });

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
    module_op.walk([&](mlir::UnrealizedConversionCastOp ucc) {
      if (ucc.getNumResults() != 1 || ucc.getInputs().size() != 1) return;
      if (addr_space_of(ucc.getResult(0).getType()) != 0) return;
      mlir::Value v = ucc.getInputs()[0];
      while (auto def = v.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (def.getInputs().size() != 1) return;
        v = def.getInputs()[0];
      }
      if (addr_space_of(v.getType()) == 1) {
        ucc.getResult(0).setType(ptr1);
        changed = true;
      }
    });
  }
}

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
        // i32 goes through the f32 shuffle builtin by bitcast: an i32 spelling of the
        // builtin creates a pipeline whether or not it is correct.
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

void RewriteAtomicsToAir(mlir::ModuleOp module_op) {
  mlir::MLIRContext* ctx = module_op.getContext();
  auto i32 = mlir::IntegerType::get(ctx, 32);
  auto i1 = mlir::IntegerType::get(ctx, 1);
  auto ptr0 = mlir::LLVM::LLVMPointerType::get(ctx, 0);
  auto ptr1 = mlir::LLVM::LLVMPointerType::get(ctx, 1);
  auto get_decl = [&]() -> mlir::LLVM::LLVMFuncOp {
    llvm::StringRef name = "air.atomic.global.cmpxchg.weak.i32";
    if (auto f = module_op.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) return f;
    mlir::OpBuilder b = mlir::OpBuilder::atBlockBegin(module_op.getBody());
    auto fty = mlir::LLVM::LLVMFunctionType::get(
        i32, {ptr1, ptr0, i32, i32, i32, i32, i1}, /*isVarArg=*/false);
    return b.create<mlir::LLVM::LLVMFuncOp>(module_op.getLoc(), name, fty);
  };
  module_op.walk([&](mlir::func::FuncOp fn) {
    if (fn.getBody().empty()) return;
    llvm::SmallVector<mlir::LLVM::AtomicCmpXchgOp> ops;
    fn.walk([&](mlir::LLVM::AtomicCmpXchgOp op) { ops.push_back(op); });
    if (ops.empty()) return;
    mlir::OpBuilder eb = mlir::OpBuilder::atBlockBegin(&fn.getBody().front());
    mlir::Value one = eb.create<mlir::LLVM::ConstantOp>(fn.getLoc(), i32,
                                                        eb.getI32IntegerAttr(1));
    mlir::Value expected_slot = eb.create<mlir::LLVM::AllocaOp>(
        fn.getLoc(), ptr0, i32, one, /*alignment=*/4);
    mlir::LLVM::LLVMFuncOp decl = get_decl();
    for (auto op : ops) {
      mlir::OpBuilder b(op);
      mlir::Location loc = op.getLoc();
      mlir::Value cmp = op.getCmp();
      auto c0 = b.create<mlir::LLVM::ConstantOp>(loc, i32, b.getI32IntegerAttr(0));
      auto c2 = b.create<mlir::LLVM::ConstantOp>(loc, i32, b.getI32IntegerAttr(2));
      auto weak =
          b.create<mlir::LLVM::ConstantOp>(loc, i1, b.getIntegerAttr(i1, 1));
      b.create<mlir::LLVM::StoreOp>(loc, cmp, expected_slot);
      auto call = b.create<mlir::LLVM::CallOp>(
          loc, decl,
          mlir::ValueRange{op.getPtr(), expected_slot, op.getVal(), c0, c0, c2,
                           weak});
      mlir::Value old = call.getResult();
      mlir::Value success = b.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, old, cmp);
      mlir::Type struct_ty = op.getResult().getType();
      mlir::Value s = b.create<mlir::LLVM::UndefOp>(loc, struct_ty);
      s = b.create<mlir::LLVM::InsertValueOp>(loc, s, old,
                                              llvm::ArrayRef<int64_t>{0});
      s = b.create<mlir::LLVM::InsertValueOp>(loc, s, success,
                                              llvm::ArrayRef<int64_t>{1});
      op.getResult().replaceAllUsesWith(s);
      op.erase();
    }
  });
}

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

// ln(2), used to express log(x) = log2(x) * ln(2).
constexpr double kLn2 = 0.6931471805599453;

// log2(e), used to express exp(x) = exp2(x * log2(e)).
constexpr double kLog2e = 1.4426950408889634;

// Lowers a scalar bf16 unary `math` op to the matching native gfx1250 bf16
// transcendental instruction (v_exp_bf16, v_sqrt_bf16, v_rsq_bf16, v_tanh_bf16,
// v_log_bf16, ...) via its `llvm.amdgcn.*` intrinsic, when the op maps 1:1 to
// the instruction. Without this, the default MathToROCDL lowering upcasts bf16
// to f32 and calls an `__ocml_*_f32` library function, never using the hardware
// bf16 transcendental unit. Vector ops are scalarized first by MathToROCDL's
// ScalarizeVectorOpLowering (lower benefit), so this pattern only needs to
// handle the scalar case.
template <typename OpTy>
struct TranscendentalBF16ToAMDGPU : public mlir::ConvertOpToLLVMPattern<OpTy> {
  TranscendentalBF16ToAMDGPU(const mlir::LLVMTypeConverter& converter,
                             llvm::StringRef intrinsic,
                             mlir::PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<OpTy>(converter, benefit),
        intrinsic(intrinsic) {}

  mlir::LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 op");
    }
    mlir::Value operand = adaptor.getOperands().front();
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, /*resultType=*/operand.getType(), rewriter.getStringAttr(intrinsic),
        mlir::ValueRange{operand});
    return mlir::success();
  }

  llvm::StringRef intrinsic;
};

// Lowers a scalar bf16 `math.log` on gfx1250 by rewriting
// log(x) = log2(x) * ln(2) and computing log2 with the native `v_log_f32`
// transcendental (the `llvm.amdgcn.log` intrinsic) in f32.
//
// Everything is computed in f32 and rounded to bf16 only once, at the end.
// Using f32 log2 rather than the native bf16 `v_log_bf16` (which only has
// bf16 mantissa precision) makes the result correctly-rounded, for one extra
// input widening. The input `fpext` is exact (a bit
// shift) and the trailing `fmul` + `fptrunc` fuse into one `v_fma_mixlo_bf16`,
// so this lowers to `v_lshlrev_b32` + `v_log_f32` + `v_fma_mixlo_bf16`.
struct LogBF16ToAMDGPU
    : public mlir::ConvertOpToLLVMPattern<mlir::math::LogOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::LogOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 log");
    }
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperands().front();
    mlir::Type bf16 = operand.getType();
    mlir::Type f32 = rewriter.getF32Type();
    mlir::Value x_f32 =
        mlir::LLVM::FPExtOp::create(rewriter, loc, f32, operand);
    mlir::Value log2x =
        mlir::LLVM::CallIntrinsicOp::create(
            rewriter, loc, /*resultType=*/f32,
            rewriter.getStringAttr("llvm.amdgcn.log"), mlir::ValueRange{x_f32})
            .getResults();
    mlir::Value ln2_f32 = mlir::LLVM::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, kLn2));
    mlir::Value logx_f32 =
        mlir::LLVM::FMulOp::create(rewriter, loc, log2x, ln2_f32);
    rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(op, bf16, logx_f32);
    return mlir::success();
  }
};

// Lowers a scalar bf16 `math.exp` on gfx1250 by rewriting
// exp(x) = 2^(x * log2(e)) and computing exp2 with the native `v_exp_f32`
// transcendental (the `llvm.amdgcn.exp2` intrinsic) in f32.
//
// Everything is computed in f32: the bf16 input is widened to f32, scaled by
// an f32 log2(e), exponentiated in f32, and the result rounded once to bf16.
// Scaling and exponentiating in f32 rather than using the native bf16
// `v_exp_bf16` (which would round x * log2(e) to bf16 before exponentiating)
// keeps the result accurate: the bf16-rounded-exponent error otherwise grows
// with |x| and can overflow to inf, which is why the native bf16 exp path was
// removed. Lowers to
// `v_lshlrev_b32` + `v_mul_f32` + `v_exp_f32` + `v_cvt_pk_bf16_f32`.
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
    mlir::Type f32 = rewriter.getF32Type();
    mlir::Value x_f32 =
        mlir::LLVM::FPExtOp::create(rewriter, loc, f32, operand);
    mlir::Value log2e = mlir::LLVM::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, kLog2e));
    mlir::Value scaled =
        mlir::LLVM::FMulOp::create(rewriter, loc, x_f32, log2e);
    mlir::Value exp2x = mlir::LLVM::CallIntrinsicOp::create(
                            rewriter, loc, /*resultType=*/f32,
                            rewriter.getStringAttr("llvm.amdgcn.exp2"),
                            mlir::ValueRange{scaled})
                            .getResults();
    rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(op, bf16, exp2x);
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
        // On gfx1250 emit native bf16 transcendentals (v_exp_bf16, v_sqrt_bf16,
        // v_rsq_bf16, v_tanh_bf16, v_log_bf16, ...) instead of upcasting to f32
        // and calling __ocml_*_f32. Higher benefit than the default MathToROCDL
        // patterns so it wins for scalar bf16 ops.
        if (device_spec_.gpu()
                .rocm_compute_capability()
                .has_bf16_transcendental_support()) {
          mlir::PatternBenefit benefit(2);
          patterns.add<LogBF16ToAMDGPU>(converter, benefit);
          patterns.add<ExpBF16ToAMDGPU>(converter, benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::Exp2Op>>(
              converter, "llvm.amdgcn.exp2", benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::SqrtOp>>(
              converter, "llvm.amdgcn.sqrt", benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::RsqrtOp>>(
              converter, "llvm.amdgcn.rsq", benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::TanhOp>>(
              converter, "llvm.amdgcn.tanh", benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::Log2Op>>(
              converter, "llvm.amdgcn.log", benefit);
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
        if (auto module_op = mlir::dyn_cast<mlir::ModuleOp>(getOperation())) {
          InjectAirThreadPositions(module_op);
          PromoteEntryBuffersToDevice(module_op);
          RewriteGpuWarpOpsToAir(module_op);
          RewriteMathToAir(module_op);
          FixupAirPointerAddrSpaces(module_op);
          RewriteAtomicsToAir(module_op);
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
