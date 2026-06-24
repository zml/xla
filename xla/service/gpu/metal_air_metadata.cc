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

#include "xla/service/gpu/metal_air_metadata.h"

#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/TargetParser/Triple.h"

namespace xla {
namespace gpu {
namespace metal {

namespace {

// This machine's Metal toolchain emits this triple/datalayout (captured using
// the Metal compiler with `-S -emit-llvm`). TODO: detect the AIR version from
// the toolchain instead of hardcoding v28/macosx26 (other toolchains emit
// air64_v27/macosx15) — tracked in LEARNINGS.md.
constexpr char kAirTriple[] = "air64_v28-apple-macosx26.0.0";
constexpr char kAirDataLayout[] =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
    "f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-"
    "v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-"
    "n8:16:32";

}  // namespace

void StampAirModuleEnvelope(llvm::Module& module) {
  llvm::LLVMContext& ctx = module.getContext();
  auto I32 = [&ctx](int v) -> llvm::Metadata* {
    return llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), v));
  };
  auto Str = [&ctx](llvm::StringRef s) -> llvm::Metadata* {
    return llvm::MDString::get(ctx, s);
  };

  module.setTargetTriple(llvm::Triple(kAirTriple));
  module.setDataLayout(kAirDataLayout);

  // Idempotent: this may be called twice on the GpuCompiler-subclass path — once
  // per fusion kernel and again on the linked whole module in
  // CompileTargetBinary. addModuleFlag/addOperand APPEND, so a second unguarded
  // stamp produces duplicate module flags that air-as rejects ("module flag
  // identifiers must be unique"). Add each flag/metadata only if absent. (For a
  // fresh module this adds them all.)
  using MF = llvm::Module::ModFlagBehavior;
  auto add_flag_once = [&](MF behavior, llvm::StringRef key, unsigned val) {
    if (module.getModuleFlag(key) == nullptr) {
      module.addModuleFlag(behavior, key, val);
    }
  };
  add_flag_once(MF::Error, "wchar_size", 4u);
  add_flag_once(MF::Max, "frame-pointer", 2u);
  add_flag_once(MF::Max, "air.max_device_buffers", 31u);
  add_flag_once(MF::Max, "air.max_constant_buffers", 31u);
  add_flag_once(MF::Max, "air.max_threadgroup_buffers", 31u);
  add_flag_once(MF::Max, "air.max_textures", 128u);
  add_flag_once(MF::Max, "air.max_read_write_textures", 8u);
  add_flag_once(MF::Max, "air.max_samplers", 16u);

  // !air.version = !{i32 2, i32 8, i32 0}
  if (module.getNamedMetadata("air.version") == nullptr) {
    module.getOrInsertNamedMetadata("air.version")
        ->addOperand(llvm::MDNode::get(ctx, {I32(2), I32(8), I32(0)}));
  }
  // !air.language_version = !{!"Metal", i32 4, i32 0, i32 0}
  if (module.getNamedMetadata("air.language_version") == nullptr) {
    module.getOrInsertNamedMetadata("air.language_version")
        ->addOperand(
            llvm::MDNode::get(ctx, {Str("Metal"), I32(4), I32(0), I32(0)}));
  }
  // !air.compile_options = !{ denorms_disable, fast_math_enable, fb_fetch }
  if (module.getNamedMetadata("air.compile_options") == nullptr) {
    llvm::NamedMDNode* opts =
        module.getOrInsertNamedMetadata("air.compile_options");
    opts->addOperand(
        llvm::MDNode::get(ctx, {Str("air.compile.denorms_disable")}));
    opts->addOperand(
        llvm::MDNode::get(ctx, {Str("air.compile.fast_math_enable")}));
    opts->addOperand(
        llvm::MDNode::get(ctx, {Str("air.compile.framebuffer_fetch_enable")}));
  }
}

void AttachAirKernelMetadata(llvm::Function* f, absl::Span<const AirArg> args) {
  llvm::LLVMContext& ctx = f->getContext();
  auto I32 = [&ctx](int v) -> llvm::Metadata* {
    return llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), v));
  };
  auto Str = [&ctx](llvm::StringRef s) -> llvm::Metadata* {
    return llvm::MDString::get(ctx, s);
  };
  auto descriptor = [&](unsigned index, const AirArg& arg) -> llvm::MDNode* {
    std::vector<llvm::Metadata*> e;
    e.push_back(I32(index));
    switch (arg.kind) {
      case AirArg::Kind::kDeviceBuffer:
        e.push_back(Str("air.buffer"));
        e.push_back(Str("air.location_index"));
        e.push_back(I32(index));
        e.push_back(I32(1));
        e.push_back(Str(arg.read_only ? "air.read" : "air.read_write"));
        e.push_back(Str("air.address_space"));
        e.push_back(I32(1));
        e.push_back(Str("air.arg_type_size"));
        e.push_back(I32(arg.type_size));
        e.push_back(Str("air.arg_type_align_size"));
        e.push_back(I32(arg.type_align));
        e.push_back(Str("air.arg_type_name"));
        e.push_back(Str(arg.type_name));
        e.push_back(Str("air.arg_name"));
        e.push_back(Str(arg.name));
        break;
      case AirArg::Kind::kConstantBuffer:
        e.push_back(Str("air.buffer"));
        e.push_back(Str("air.buffer_size"));
        e.push_back(I32(arg.type_size));
        e.push_back(Str("air.location_index"));
        e.push_back(I32(index));
        e.push_back(I32(1));
        e.push_back(Str("air.read"));
        e.push_back(Str("air.address_space"));
        e.push_back(I32(2));
        e.push_back(Str("air.arg_type_size"));
        e.push_back(I32(arg.type_size));
        e.push_back(Str("air.arg_type_align_size"));
        e.push_back(I32(arg.type_align));
        e.push_back(Str("air.arg_type_name"));
        e.push_back(Str(arg.type_name));
        e.push_back(Str("air.arg_name"));
        e.push_back(Str(arg.name));
        break;
      case AirArg::Kind::kThreadPositionInGrid:
        e.push_back(Str("air.thread_position_in_grid"));
        e.push_back(Str("air.arg_type_name"));
        e.push_back(Str(arg.type_name));
        e.push_back(Str("air.arg_name"));
        e.push_back(Str(arg.name));
        break;
      case AirArg::Kind::kThreadPositionInThreadgroup:
        e.push_back(Str("air.thread_position_in_threadgroup"));
        e.push_back(Str("air.arg_type_name"));
        e.push_back(Str(arg.type_name));
        e.push_back(Str("air.arg_name"));
        e.push_back(Str(arg.name));
        break;
      case AirArg::Kind::kThreadgroupPositionInGrid:
        e.push_back(Str("air.threadgroup_position_in_grid"));
        e.push_back(Str("air.arg_type_name"));
        e.push_back(Str(arg.type_name));
        e.push_back(Str("air.arg_name"));
        e.push_back(Str(arg.name));
        break;
    }
    return llvm::MDNode::get(ctx, e);
  };

  std::vector<llvm::Metadata*> descriptors;
  descriptors.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    descriptors.push_back(descriptor(static_cast<unsigned>(i), args[i]));
  }
  llvm::Metadata* kernel_entry[] = {
      llvm::ConstantAsMetadata::get(f),
      llvm::MDNode::get(ctx, llvm::ArrayRef<llvm::Metadata*>()),  // outputs
      llvm::MDNode::get(ctx, descriptors),                        // args
  };
  f->getParent()
      ->getOrInsertNamedMetadata("air.kernel")
      ->addOperand(llvm::MDNode::get(ctx, kernel_entry));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xla
