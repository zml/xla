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

#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/lib/Target/SPIRV/MCTargetDesc/SPIRVBaseInfo.h"
#include "llvm/lib/Target/SPIRV/SPIRVAPI.h"
#include "llvm/lib/Target/SPIRV/SPIRVSubtarget.h"
#include "llvm/lib/Target/SPIRV/SPIRVTargetMachine.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "tsl/platform/errors.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace xla::gpu::spirv {

namespace {

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

void SPIRVBackendInit() {
  LLVMInitializeSPIRVTargetInfo();
  LLVMInitializeSPIRVTarget();
  LLVMInitializeSPIRVTargetMC();
  LLVMInitializeSPIRVAsmPrinter();

  // Initialize the LLVM optimization passes.
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

absl::Status SPIRVTargetModuleLinker(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, const std::string& device_bitcode_path) {
  return absl::OkStatus();
}

std::string EmitModuleToSPIRV(llvm::Module* module,
                              llvm::TargetMachine* target_machine) {
  std::string spirv_binary;
  llvm::raw_string_ostream string_stream(spirv_binary);
  llvm::buffer_ostream buffered_stream(string_stream);
  llvm::legacy::PassManager pm;
  // Unlike other GPU backends like NVTPTX and AMDGPU, SPIRV does not have
  // address inference pass in the TargetPassConfig. So we do it here
  // explicitly.
  pm.add(llvm::createInferAddressSpacesPass(0));
  llvm::ScalarizerPassOptions scalarizer_options;
  scalarizer_options.ScalarizeLoadStore = true;
  pm.add(llvm::createScalarizerPass(scalarizer_options));

  int scalarized_dump_fd = -1;
  llvm::SmallString<128> scalarized_dump_path;
  std::unique_ptr<llvm::raw_fd_ostream> scalarized_dump_stream;
  std::error_code scalarized_dump_error = llvm::sys::fs::createUniqueFile(
      "/tmp/xla-vulkan-after-scalarizer-%%%%%%.ll", scalarized_dump_fd,
      scalarized_dump_path);
  if (!scalarized_dump_error) {
    scalarized_dump_stream = std::make_unique<llvm::raw_fd_ostream>(
        scalarized_dump_fd, /*shouldClose=*/true);
    scalarized_dump_stream->SetUnbuffered();
    pm.add(llvm::createPrintModulePass(*scalarized_dump_stream));
    llvm::errs() << "Scalarized Vulkan LLVM module: " << scalarized_dump_path
                 << '\n';
  } else {
    llvm::errs() << "Could not create scalarized Vulkan LLVM dump: "
                 << scalarized_dump_error.message() << '\n';
  }

  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  std::unique_ptr<llvm::MachineModuleInfoWrapperPass> mmiwp(
      new llvm::MachineModuleInfoWrapperPass(target_machine));
  target_machine->getObjFileLowering()->Initialize(mmiwp->getMMI().getContext(),
                                                   *target_machine);
  
  llvm::errs() << "[Vulkan] before addPassesToEmitFile\n";

  bool failed = target_machine->addPassesToEmitFile(
      pm, buffered_stream, nullptr, llvm::CodeGenFileType::ObjectFile);

  llvm::errs() << "[Vulkan] after addPassesToEmitFile, failed=" << failed << '\n';
  llvm::errs() << "[Vulkan] before pm.run\n";

  pm.run(*module);

  llvm::errs() << "[Vulkan] after pm.run\n";
  if (scalarized_dump_stream) scalarized_dump_stream->flush();

  return spirv_binary;
}

llvm::IntegerType* GetSubByteElementType(llvm::Type* call_type) {
  llvm::Type* scalar = call_type->getScalarType();
  if (auto* type = llvm::dyn_cast<llvm::IntegerType>(scalar)) {
    return type->getBitWidth() < 8 ? type : nullptr;
  }
  return nullptr;
}

// Expand sub-byte integer llvm bitreverse intrinsic calls because SPIR-V has no
// native support for such integer types.
void ExpandSubByteBitReverse(llvm::Module* module) {
  llvm::SmallVector<llvm::CallInst*> calls;
  for (auto& func : *module) {
    for (auto& bb : func) {
      for (auto& inst : bb) {
        llvm::IntrinsicInst* call = llvm::dyn_cast<llvm::IntrinsicInst>(&inst);
        if (call && call->getIntrinsicID() == llvm::Intrinsic::bitreverse &&
            GetSubByteElementType(call->getType())) {
          calls.push_back(call);
        }
      }
    }
  }
  for (llvm::CallInst* call : calls) {
    llvm::IRBuilder<> builder(call);
    llvm::Type* type = call->getType();
    llvm::Type* wide_type = llvm::Type::getInt32Ty(module->getContext());
    if (auto* vec_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
      wide_type = llvm::FixedVectorType::get(builder.getInt32Ty(),
                                             vec_type->getNumElements());
    }
    llvm::Value* zext = builder.CreateZExt(call->getArgOperand(0), wide_type);
    llvm::Function* bitrev = llvm::Intrinsic::getOrInsertDeclaration(
        module, llvm::Intrinsic::bitreverse, {wide_type});
    llvm::Value* rev = builder.CreateCall(bitrev, {zext});
    llvm::Value* shr = builder.CreateLShr(
        rev, 32 - GetSubByteElementType(type)->getBitWidth());
    llvm::Value* result = builder.CreateTrunc(shr, type);
    call->replaceAllUsesWith(result);
    call->eraseFromParent();
  }
}

absl::Status ValidateVulkanModule(const llvm::Module& module) {
  for (const llvm::Function& function : module) {
    if (function.isDeclaration() &&
        (function.getName().contains("__spirv_ocl_") ||
         function.getName() == "_Z7barrierj")) {
      return absl::UnimplementedError(absl::StrCat(
          "OpenCL device function is not supported by the Vulkan backend: ",
          function.getName().str()));
    }
  }
  return absl::OkStatus();
}

absl::Status WrapVulkanEntryPoints(llvm::Module* module) {
  llvm::LLVMContext& context = module->getContext();
  llvm::SmallVector<llvm::Function*> entries;
  for (llvm::Function& function : *module) {
    if (!function.isDeclaration() && function.hasFnAttribute("hlsl.shader")) {
      entries.push_back(&function);
    }
  }
  if (entries.empty()) {
    return absl::OkStatus();
  }

  for (llvm::Function* implementation : entries) {
    const std::string entry_name = implementation->getName().str();
    const std::string numthreads =
        implementation->hasFnAttribute("hlsl.numthreads")
            ? implementation->getFnAttribute("hlsl.numthreads")
                  .getValueAsString()
                  .str()
            : "1,1,1";
    for (llvm::Argument& argument : implementation->args()) {
      if (!llvm::isa<llvm::PointerType>(argument.getType())) {
        return absl::UnimplementedError(absl::StrCat(
            "Vulkan entry point ", entry_name,
            " has a non-pointer argument; only managed buffer arguments are "
            "supported"));
      }
    }

    implementation->setName(absl::StrCat(entry_name, ".impl"));
    implementation->setLinkage(llvm::GlobalValue::InternalLinkage);
    implementation->addFnAttr(llvm::Attribute::AlwaysInline);
    implementation->removeFnAttr("hlsl.shader");
    implementation->removeFnAttr("hlsl.numthreads");

    llvm::Function* entry = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
        llvm::GlobalValue::ExternalLinkage, entry_name, module);
    entry->setCallingConv(llvm::CallingConv::C);
    entry->addFnAttr("hlsl.shader", "compute");
    entry->addFnAttr("hlsl.numthreads", numthreads);

    llvm::BasicBlock* block = llvm::BasicBlock::Create(context, "entry", entry);
    llvm::IRBuilder<> builder(block);
    llvm::SmallVector<llvm::Value*> arguments;
    arguments.reserve(implementation->arg_size());
    unsigned binding = 0;
    for (llvm::Argument& argument : implementation->args()) {
      const bool writable = !argument.hasAttribute(llvm::Attribute::ReadOnly);
      llvm::Type* element_type = nullptr;
      for (llvm::User* user : argument.users()) {
        llvm::Type* accessed_type = nullptr;
        if (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {
          accessed_type = gep->getSourceElementType();
        } else if (auto* load = llvm::dyn_cast<llvm::LoadInst>(user)) {
          accessed_type = load->getType();
        } else if (auto* store = llvm::dyn_cast<llvm::StoreInst>(user)) {
          accessed_type = store->getValueOperand()->getType();
        }
        while (auto* array = llvm::dyn_cast_or_null<llvm::ArrayType>(
                   accessed_type)) {
          accessed_type = array->getElementType();
        }
        if (auto* vector =
                llvm::dyn_cast_or_null<llvm::VectorType>(accessed_type)) {
          accessed_type = vector->getElementType();
        }
        if (accessed_type == nullptr) {
          continue;
        }
        if (element_type != nullptr && element_type != accessed_type) {
          return absl::UnimplementedError(absl::StrCat(
              "Vulkan entry point ", entry_name, " buffer argument ", binding,
              " is accessed with incompatible element types"));
        }
        element_type = accessed_type;
      }
      if (element_type == nullptr) {
        element_type = builder.getInt32Ty();
      }
      llvm::Type* runtime_array = llvm::ArrayType::get(element_type, 0);
      llvm::Type* resource_type = llvm::TargetExtType::get(
          context, "spirv.VulkanBuffer", {runtime_array},
          {12, writable ? 1U : 0U});
      llvm::Function* get_handle = llvm::Intrinsic::getOrInsertDeclaration(
          module, llvm::Intrinsic::spv_resource_handlefrombinding,
          {resource_type});
      llvm::Value* handle = builder.CreateCall(
          get_handle,
          {builder.getInt32(0), builder.getInt32(binding), builder.getInt32(1),
           builder.getInt32(0),
           llvm::ConstantPointerNull::get(builder.getPtrTy())});
      llvm::PointerType* storage_buffer_pointer =
          llvm::PointerType::get(context, 11);
      llvm::Function* get_pointer = llvm::Intrinsic::getOrInsertDeclaration(
          module, llvm::Intrinsic::spv_resource_getbasepointer,
          {storage_buffer_pointer, resource_type});
      llvm::Value* pointer = builder.CreateCall(get_pointer, {handle});
      if (pointer->getType() != argument.getType()) {
        pointer = builder.CreateAddrSpaceCast(pointer, argument.getType());
      }
      arguments.push_back(pointer);
      ++binding;
    }
    builder.CreateCall(implementation, arguments);
    builder.CreateRetVoid();
  }

  int dump_fd = -1;
  llvm::SmallString<128> dump_path;
  std::error_code error = llvm::sys::fs::createUniqueFile(
      "/tmp/xla-vulkan-wrapped-%%%%%%.ll", dump_fd, dump_path);
  if (error) {
    return absl::InternalError(absl::StrCat(
        "Failed to create wrapped Vulkan LLVM dump: ", error.message()));
  }

  llvm::raw_fd_ostream dump_stream(dump_fd, /*shouldClose=*/true);
  module->print(dump_stream, nullptr);
  dump_stream.flush();

  llvm::errs() << "Wrapped Vulkan LLVM module: " << dump_path << '\n';

  return absl::OkStatus();
}

}  // namespace

std::vector<std::string> SPIRVExtensionsEnumToString(
    const llvm::ExtensionSet& enum_extensions) {
  std::vector<std::string> str_extensions;
  for (auto& ext : enum_extensions) {
    str_extensions.push_back(llvm::getSymbolicOperandMnemonic(
        llvm::SPIRV::OperandCategory::ExtensionOperand, ext));
  }
  return str_extensions;
}

std::vector<std::string> GetSPIRVBackendOptions(
    const DebugOptions& debug_options) {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.
  std::vector<std::string> backend_llvm_opts;

  auto backend_extra_llvm_opts = llvm_ir::ExtractXlaBackendExtraOptions(
      debug_options.xla_backend_extra_options());
  backend_llvm_opts.insert(backend_llvm_opts.end(),
                           backend_extra_llvm_opts.cbegin(),
                           backend_extra_llvm_opts.cend());

  return backend_llvm_opts;
}

absl::StatusOr<std::string> CompileToSPIRV(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  static absl::once_flag backend_init_flag;
  static absl::NoDestructor<std::vector<std::string>> spirv_extensions;
  absl::call_once(backend_init_flag, SPIRVBackendInit);
  auto llvm_opts = GetSPIRVBackendOptions(debug_options);
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_opts);

  // SPIRV Kernel functions expect their arguments' address spaces to be global,
  // i.e., addrspace(1). Here we only change kernel argument's address space if
  // it is addrspace(0). Then an address space cast to original is applied so
  // that users still have old address space.
  llvm::LLVMContext& context = module->getContext();
  llvm::SmallVector<llvm::Function*> kernel_funcs;
  for (auto& func : *module) {
    if (func.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      kernel_funcs.push_back(&func);
    }
  }
  for (auto old_func : kernel_funcs) {
    if (old_func->getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      std::vector<llvm::Type*> new_arg_types;
      for (auto& old_arg : old_func->args()) {
        llvm::Type* old_arg_type = old_arg.getType();
        auto ptr_type = llvm::dyn_cast<llvm::PointerType>(old_arg_type);
        if (ptr_type->getAddressSpace() == 0) {
          auto new_arg_type = llvm::PointerType::get(context, 1);
          new_arg_types.push_back(new_arg_type);
        } else {
          new_arg_types.push_back(old_arg_type);
        }
      }
      auto new_func_type = llvm::FunctionType::get(
          old_func->getReturnType(), new_arg_types, old_func->isVarArg());

      auto new_func = llvm::Function::Create(
          new_func_type, old_func->getLinkage(), old_func->getName(), module);

      // We do not want to modify type of arguments of old func at the uses,
      // hence using identity map.
      llvm::ValueToValueMapTy identity_map;
      for (auto& old_arg : old_func->args()) {
        identity_map[&old_arg] = &old_arg;
      }
      llvm::SmallVector<llvm::ReturnInst*> returns;
      llvm::CloneFunctionInto(new_func, old_func, identity_map,
                              llvm::CloneFunctionChangeType::LocalChangesOnly,
                              returns);

      llvm::IRBuilder<> builder(&(*new_func->begin()->begin()));
      auto new_arg_it = new_func->arg_begin();
      auto old_arg_it = old_func->arg_begin();
      for (; old_arg_it != old_func->arg_end(); ++old_arg_it, ++new_arg_it) {
        if (auto old_ptr_type =
                llvm::dyn_cast<llvm::PointerType>(old_arg_it->getType())) {
          auto cast = builder.CreateAddrSpaceCast(new_arg_it, old_ptr_type);
          old_arg_it->replaceAllUsesWith(cast);
        }
      }
      // TODO: Update kernel function's uses. Currently, we are assuming that
      // kernel function is not called by any other functions in the current
      // LLVM module.
      new_func->takeName(old_func);
      old_func->eraseFromParent();
    }
  }

  llvm::Triple default_target_triple("spirv64-unknown-unknown");
  std::unique_ptr<llvm::TargetMachine> target_machine =
      GetTargetMachine(default_target_triple, "", debug_options, "");
  // Set datalayout and spirv extenstions.
  module->setDataLayout(target_machine->createDataLayout());
  llvm::SPIRVTargetMachine* sub_target =
      static_cast<llvm::SPIRVTargetMachine*>(target_machine.get());
  const_cast<llvm::SPIRVSubtarget*>(sub_target->getSubtargetImpl())
      ->initAvailableExtensions(common_spirv_extensions);

  RETURN_IF_ERROR(LinkAndOptimizeModule(
      module, gpu_version, debug_options, "", SPIRVTargetModuleLinker,
      default_target_triple, target_machine.get(), kDefaultInlineThreshold));

  ExpandSubByteBitReverse(module);

  // The LLVM SPIR-V backend removes unused globals during its passes for
  // translation to SPIR-V. To prevent this, we create a fake use of those
  // globals with a minimal fake use function. We first create a global pointer
  // list with appending linkage containing pointers to all globals in the
  // module. And then the fake use function uses getelementptr and load
  // instruction to load the first element of the global pointer list.
  llvm::Type* ptr_type = llvm::PointerType::getUnqual(context);
  llvm::SmallVector<llvm::Constant*> global_ptrs;
  for (llvm::GlobalVariable& global_var : module->globals()) {
    global_ptrs.push_back(
        llvm::ConstantExpr::getPointerCast(&global_var, ptr_type));
  }
  if (!global_ptrs.empty()) {
    auto* arr_type = llvm::ArrayType::get(ptr_type, global_ptrs.size());
    auto* arr_init = llvm::ConstantArray::get(arr_type, global_ptrs);
    auto* global_ptr_arr = new llvm::GlobalVariable(
        arr_type, /*isConstant=*/false, llvm::GlobalValue::AppendingLinkage,
        /*Initializer=*/arr_init, "global_ptr_list",
        /*ThreadLocalMode=*/llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/1, /*isExternallyInitialized=*/false);
    global_ptr_arr->setAlignment(llvm::Align(kConstantBufferAlignBytes));
    module->insertGlobalVariable(global_ptr_arr);

    // Create a fake use function that loads the first element of
    // global_ptr_list to prevent it from being optimized away.
    auto* fake_use_func_type =
        llvm::FunctionType::get(ptr_type, /*isVarArg=*/false);
    auto* fake_use_func = llvm::Function::Create(
        fake_use_func_type, llvm::GlobalValue::ExternalLinkage,
        "fake_use_globals", module);
    fake_use_func->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    auto* bb = llvm::BasicBlock::Create(context, "entry", fake_use_func);
    llvm::IRBuilder<> ir_builder(bb);
    auto* gep = ir_builder.CreateConstGEP2_64(arr_type, global_ptr_arr,
                                              /*Idx0=*/0, /*Idx1=*/0);
    auto* load = ir_builder.CreateLoad(ptr_type, gep);
    ir_builder.CreateRet(load);
  }

  return EmitModuleToSPIRV(module, target_machine.get());
}

absl::StatusOr<std::string> CompileToVulkanSPIRV(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, SPIRVBackendInit);
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(
      GetSPIRVBackendOptions(debug_options));

  RETURN_IF_ERROR(ValidateVulkanModule(*module));
  RETURN_IF_ERROR(WrapVulkanEntryPoints(module));

  llvm::errs() << "[Vulkan] before target machine\n";

  llvm::Triple target_triple("spirv1.5-unknown-vulkan1.2-compute");
  std::unique_ptr<llvm::TargetMachine> target_machine =
      GetTargetMachine(target_triple, "", debug_options, "");

  llvm::errs() << "[Vulkan] after target machine\n";

  module->setTargetTriple(target_triple);
  module->setDataLayout(target_machine->createDataLayout());

  llvm::SPIRVTargetMachine* spirv_target =
      static_cast<llvm::SPIRVTargetMachine*>(target_machine.get());
  const_cast<llvm::SPIRVSubtarget*>(spirv_target->getSubtargetImpl())
      ->initAvailableExtensions({});

  llvm::errs() << "[Vulkan] before LinkAndOptimizeModule\n";

  RETURN_IF_ERROR(LinkAndOptimizeModule(
      module, gpu_version, debug_options, "", SPIRVTargetModuleLinker,
      target_triple, target_machine.get(), kDefaultInlineThreshold));

  llvm::errs() << "[Vulkan] after LinkAndOptimizeModule\n";

  ExpandSubByteBitReverse(module);

  llvm::errs() << "[Vulkan] before EmitModuleToSPIRV\n";

  std::string spirv = EmitModuleToSPIRV(module, target_machine.get());

  llvm::errs() << "[Vulkan] after EmitModuleToSPIRV\n";
  return spirv;
}

}  // namespace xla::gpu::spirv
