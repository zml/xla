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

#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"

// Loaded only by the pinned SDK compiler, never by the PJRT process. Keep the
// remainder of O2 intact: SDK 5.1 loop-load elimination crashes while forming
// runtime pointer checks for qualified S4000 shared-memory kernels.
extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "musa-sdk51-loop-load-guard", "1",
          [](llvm::PassBuilder& builder) {
            auto* callbacks = builder.getPassInstrumentationCallbacks();
            if (callbacks == nullptr) {
              llvm::report_fatal_error("MUSA pass guard requires instrumentation");
            }
            callbacks->registerShouldRunOptionalPassCallback(
                [](llvm::StringRef name, llvm::Any) {
                  return name != "LoopLoadEliminationPass";
                });
          }};
}
