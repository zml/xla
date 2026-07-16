// Copyright 2026 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include "llvm/ADT/Triple.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

int main() {
  llvm::LLVMContext context;
  context.enableOpaquePointers();
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
      "target triple = \"mtgpu-mt-musa\"\n"
      "define void @kernel(ptr addrspace(1) %out) { ret void }\n",
      diagnostic, context);
  if (module == nullptr) return 1;

  llvm::Function* kernel = module->getFunction("kernel");
  if (kernel == nullptr) return 2;
  kernel->setCallingConv(llvm::CallingConv::MTGPU_KERNEL);

  llvm::Triple triple(module->getTargetTriple());
  if (!triple.isMTGPU() || triple.getOS() != llvm::Triple::MUSA) return 3;
  if (kernel->getCallingConv() != llvm::CallingConv::MTGPU_KERNEL) return 4;
  return 0;
}
