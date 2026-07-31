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

#include "xla/service/gpu/musa/musa_custom_kernel_source.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::string Module(absl::string_view entries,
                   absl::string_view globals = "") {
  return absl::StrCat(
      "target datalayout = \"", kMusaDataLayout, "\"\n",
      "target triple = \"", kMusaTargetTriple, "\"\n\n", globals,
      entries);
}

std::string Kernel(absl::string_view name, bool marked = true) {
  return absl::StrCat(
      "define void @", name, "(ptr addrspace(1) %out)",
      marked ? " #0" : "", " {\nentry:\n  store i32 7, ptr addrspace(1) "
      "%out, align 4\n  ret void\n}\n",
      marked ? "attributes #0 = { \"xla.musa.kernel.v1\" }\n" : "");
}

TEST(MusaCustomKernelSourceTest, AcceptsOneMatchingVersionedKernel) {
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource source,
      ParseMusaCustomKernelSource("store_seven", Module(Kernel("store_seven"))));
  ASSERT_NE(source.module(), nullptr);
  EXPECT_NE(source.module()->getFunction("store_seven"), nullptr);
}

TEST(MusaCustomKernelSourceTest, RejectsMalformedCurrentLlvm) {
  EXPECT_THAT(ParseMusaCustomKernelSource("kernel", "not LLVM"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("parser rejected")));
}

TEST(MusaCustomKernelSourceTest, RejectsEntryNameMismatch) {
  EXPECT_THAT(ParseMusaCustomKernelSource("requested", Module(Kernel("actual"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match requested name")));
}

TEST(MusaCustomKernelSourceTest, RejectsUnmarkedExternalDefinition) {
  EXPECT_THAT(ParseMusaCustomKernelSource("kernel",
                                          Module(Kernel("kernel", false))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("kernel-marker")));
}

TEST(MusaCustomKernelSourceTest, RejectsMultipleKernelEntries) {
  std::string entries =
      absl::StrCat(Kernel("first"), "\n", Kernel("second"));
  EXPECT_THAT(ParseMusaCustomKernelSource("first", Module(entries)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exactly one marked kernel")));
}

TEST(MusaCustomKernelSourceTest, RejectsExportedGlobals) {
  EXPECT_THAT(
      ParseMusaCustomKernelSource(
          "kernel",
          Module(Kernel("kernel"), "@value = addrspace(1) global i32 0\n")),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("does not yet support exported globals")));
}

TEST(MusaCustomKernelSourceTest, RejectsWrongTargetBeforeBridge) {
  std::string source = Module(Kernel("kernel"));
  const std::string triple =
      absl::StrCat("target triple = \"", kMusaTargetTriple, "\"");
  source.replace(source.find(triple), triple.size(),
                 "target triple = \"nvptx64-nvidia-cuda\"");
  EXPECT_THAT(ParseMusaCustomKernelSource("kernel", source),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("target-triple")));
}

}  // namespace
}  // namespace xla::gpu::musa
