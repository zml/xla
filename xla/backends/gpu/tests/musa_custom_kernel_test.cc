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

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla::gpu {
namespace {

class MusaCustomKernelE2eTest : public HloPjRtTestBase {};

TEST_F(MusaCustomKernelE2eTest, ScalarAddUsesOwnedMubinKernel) {
  const std::string llvm_ir = absl::StrCat(
      "target datalayout = \"", musa::kMusaDataLayout, "\"\n",
      "target triple = \"", musa::kMusaTargetTriple, "\"\n\n",
      "define void @musa_scalar_add(ptr addrspace(1) %a, "
      "ptr addrspace(1) %b, ptr addrspace(1) %out) #0 {\n",
      "entry:\n",
      "  %av = load float, ptr addrspace(1) %a, align 4\n",
      "  %bv = load float, ptr addrspace(1) %b, align 4\n",
      "  %sum = fadd float %av, %bv\n",
      "  store float %sum, ptr addrspace(1) %out, align 4\n",
      "  ret void\n",
      "}\n\n",
      "attributes #0 = { \"", musa::kMusaLlvmKernelMarker, "\" }\n");
  const std::string backend_config = absl::StrCat(
      "{ name = \"musa_scalar_add\", kernel_type = \"musa_llvm\", "
      "kernel_data = \"", absl::CEscape(llvm_ir),
      "\", grid_x = 1, grid_y = 1, grid_z = 1, "
      "block_x = 1, block_y = 1, block_z = 1, "
      "shared_mem_bytes = 0, output_indices = [2] }");
  const std::string hlo = absl::StrCat(
      "HloModule musa_custom_kernel_test\n\n",
      "ENTRY main {\n",
      "  a = f32[] constant(3.0)\n",
      "  b = f32[] constant(4.0)\n",
      "  ROOT out = f32[] custom-call(a, b),\n",
      "    custom_call_target=\"__gpu$xla.gpu.musa_llvm\",\n",
      "    backend_config=\"", absl::CEscape(backend_config), "\"\n",
      "}\n");

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  EXPECT_EQ(result.Get<float>({}), 7.0f);
}

}  // namespace
}  // namespace xla::gpu
