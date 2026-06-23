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

#include "xla/backends/gpu/codegen/emitters/metal_mlir_kernel_fusion.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/metal_air_metadata.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// The AIR `arg_type_name` / `arg_type_size` / `arg_type_align_size` reflection
// metadata for a device buffer of the given element type. Spellings/sizes are
// the ground truth `xcrun metal -emit-llvm` emits for `device <T>*` args. An
// unmapped type keeps a 4-byte "float" descriptor (the historical default);
// this is reflection metadata (loads/stores carry their own IR types), so a
// wrong descriptor does not miscompile — but matching the real element type is
// the honest thing and keeps the metadata usable.
metal::AirArg AirBufferArg(PrimitiveType elem, std::string name,
                           bool read_only) {
  metal::AirArg arg;
  arg.kind = metal::AirArg::Kind::kDeviceBuffer;
  arg.name = std::move(name);
  arg.read_only = read_only;
  const char* type_name;
  switch (elem) {
    case F32: type_name = "float"; break;
    case F16: type_name = "half"; break;
    case BF16: type_name = "bfloat"; break;
    case S8: type_name = "char"; break;
    case U8: type_name = "uchar"; break;
    case S16: type_name = "short"; break;
    case U16: type_name = "ushort"; break;
    case S32: type_name = "int"; break;
    case U32: type_name = "uint"; break;
    case S64: type_name = "long"; break;
    case U64: type_name = "ulong"; break;
    case PRED: type_name = "bool"; break;
    default: type_name = nullptr; break;
  }
  if (type_name == nullptr) {
    arg.type_name = "float";
    arg.type_size = 4;
    arg.type_align = 4;
  } else {
    arg.type_name = type_name;
    const int bytes = primitive_util::ByteWidth(elem);
    arg.type_size = bytes;
    arg.type_align = bytes;
  }
  return arg;
}

// Apple's air-as is an ~LLVM-15-era parser, but LLVM 23's translateModuleToLLVMIR
// emits post-15 function attributes (memory(...), nocreateundeforpoison, ...) on
// the kernel and on intrinsic declarations (e.g. @llvm.maximum.f32), which air-as
// cannot lex ("unterminated attribute group"). Strip every FUNCTION-level
// attribute down to the air-as-safe set the hand emitter uses (NoUnwind, plus
// Convergent if present for future threadgroup kernels); parameter/return attrs
// (noalias/align/dereferenceable) are pre-15 and kept. Also strip call-site
// function attributes for the same reason.
void ScrubAirIncompatibleAttrs(llvm::Module& module) {
  llvm::LLVMContext& ctx = module.getContext();
  for (llvm::Function& fn : module.functions()) {
    const bool nounwind = fn.hasFnAttribute(llvm::Attribute::NoUnwind);
    bool convergent = fn.hasFnAttribute(llvm::Attribute::Convergent);
    // The AIR warp/threadgroup builtins must stay Convergent so air-opt --O3
    // won't reorder code across them (else silently-wrong reductions).
    const llvm::StringRef name = fn.getName();
    if (name == "air.wg.barrier" || name.starts_with("air.simd_")) {
      convergent = true;
    }
    fn.setAttributes(fn.getAttributes().removeFnAttributes(ctx));
    if (nounwind) fn.addFnAttr(llvm::Attribute::NoUnwind);
    if (convergent) fn.addFnAttr(llvm::Attribute::Convergent);
    for (llvm::BasicBlock& bb : fn) {
      for (llvm::Instruction& inst : bb) {
        if (auto* call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
          call->setAttributes(call->getAttributes().removeFnAttributes(ctx));
        }
      }
    }
  }
}

}  // namespace

xla::Future<LlvmKernelSource> MetalMlirKernelFusion::CreateLLVMModule(
    const se::DeviceDescription& device, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment, KernelCompiler* kernel_compiler,
    BorrowedMlirContext borrowed_context) const {
  // Recover each buffer's element type from the fusion (the kernel ABI is
  // [operands..., outputs..., tid, tgid]) while we still have the fusion
  // context, so the AIR arg descriptors match. Flatten a tuple root to its leaf
  // shapes (multi-output fusions bind one buffer per leaf).
  std::vector<PrimitiveType> buffer_elems;
  buffer_elems.reserve(fusion.operand_count() + 1);
  for (const HloInstruction* operand : fusion.operands()) {
    buffer_elems.push_back(operand->shape().element_type());
  }
  const Shape& root_shape = fusion.shape();
  if (root_shape.IsTuple()) {
    for (const Shape& leaf : root_shape.tuple_shapes()) {
      buffer_elems.push_back(leaf.element_type());
    }
  } else {
    buffer_elems.push_back(root_shape.element_type());
  }
  const size_t num_outputs =
      root_shape.IsTuple() ? root_shape.tuple_shapes().size() : 1;
  std::string fusion_name(fusion.name());

  // Native GpuToAIR path: ride XLA's real MLIR->LLVM lowering (the base), whose
  // Metal arm (lower_to_llvm_gpu.cc) injects the AIR thread-position kernel args
  // and rewrites the gpu.* index reads to read them. There is intentionally NO
  // hand-emitter fallback (load-bearing rule: the native path is the SOLE
  // compile path; a silent fallback once masked the emitter not running). An
  // unsupported fusion surfaces as a loud error from the base lowering. Then
  // post-process the lowered module into AIR (the !air.kernel metadata + the
  // attr scrub); StampAirModuleEnvelope runs later on the linked whole module
  // in MetalGpuCompiler::CompileTargetBinary.
  return MlirKernelFusion::CreateLLVMModule(
             device, fusion, entry_function_name, buffer_assignment,
             kernel_compiler, std::move(borrowed_context))
      .Map([kernel_name = std::string(entry_function_name),
            buffer_elems = std::move(buffer_elems), num_outputs,
            fusion_name = std::move(fusion_name)](
               LlvmKernelSource source) -> absl::StatusOr<LlvmKernelSource> {
        llvm::orc::ThreadSafeModule tsm = std::move(source).thread_safe_module();
        llvm::Module* module = tsm.getModuleUnlocked();

        // Stamp the !air.kernel metadata air-as requires. The lowered kernel's
        // LLVM args are [buffers..., tid, tgid]: pointer args are device buffers
        // (last num_outputs = the read_write outputs), and the two trailing i32s
        // are the thread-position args the Metal lowering arm injected.
        llvm::Function* f = module->getFunction(kernel_name);
        if (f == nullptr) {
          return absl::InternalError(
              absl::StrCat("Metal AIR: lowered module has no kernel function '",
                           kernel_name, "'"));
        }

        std::vector<llvm::Argument*> ptr_args;
        std::vector<llvm::Argument*> scalar_args;
        for (llvm::Argument& a : f->args()) {
          if (a.getType()->isPointerTy()) {
            ptr_args.push_back(&a);
          } else {
            scalar_args.push_back(&a);
          }
        }
        if (scalar_args.size() != 2 || ptr_args.empty()) {
          return absl::UnimplementedError(absl::StrCat(
              "Metal AIR backend cannot lower fusion '", fusion_name,
              "': expected the [buffers..., tid, tgid] kernel ABI but got ",
              ptr_args.size(), " buffer arg(s) and ", scalar_args.size(),
              " scalar arg(s)."));
        }

        std::vector<metal::AirArg> air_args;
        air_args.reserve(ptr_args.size() + 2);
        for (size_t i = 0; i < ptr_args.size(); ++i) {
          const bool is_output = (i + num_outputs >= ptr_args.size());
          // Fall back to f32 if the ABI arg count doesn't line up with operands
          // + outputs (defensive — preserves the historical descriptor).
          const PrimitiveType elem =
              i < buffer_elems.size() ? buffer_elems[i] : F32;
          std::string name;
          if (!is_output) {
            name = absl::StrCat("in", i);
          } else if (num_outputs == 1) {
            name = "out";
          } else {
            name = absl::StrCat("out", i + num_outputs - ptr_args.size());
          }
          air_args.push_back(AirBufferArg(elem, std::move(name),
                                          /*read_only=*/!is_output));
        }
        air_args.push_back({metal::AirArg::Kind::kThreadPositionInThreadgroup,
                            "tid", "uint", 4, 4, true});
        air_args.push_back({metal::AirArg::Kind::kThreadgroupPositionInGrid,
                            "tgid", "uint", 4, 4, true});

        ScrubAirIncompatibleAttrs(*module);
        // A reduction kernel that calls the AIR warp/barrier builtins must itself
        // be Convergent (so air-opt --O3 won't sink/hoist code across them).
        if (module->getFunction("air.wg.barrier") != nullptr ||
            module->getFunction("air.simd_shuffle_down.f32") != nullptr) {
          f->addFnAttr(llvm::Attribute::Convergent);
        }
        metal::AttachAirKernelMetadata(f, air_args);
        return LlvmKernelSource{std::move(tsm)};
      });
}

}  // namespace xla::gpu
