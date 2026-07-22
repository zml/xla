/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/target_util.h"

#include <gtest/gtest.h>
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class TargetUtilTest : public testing::Test {
 public:
  TargetUtilTest() : module_("test", ctx_), builder_(ctx_) {}

 protected:
  void SetUp() override {
    auto fn = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {}),
        llvm::Function::LinkageTypes::ExternalLinkage, "fn", module_);
    auto block = llvm::BasicBlock::Create(ctx_, "blk", fn);
    builder_.SetInsertPoint(block);
  }

  llvm::LLVMContext ctx_;
  llvm::Module module_;
  llvm::IRBuilder<> builder_;
};

TEST_F(TargetUtilTest, NVPTXGroupBarrier) {
  module_.setTargetTriple(llvm::Triple("nvptx"));
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId,
                            {/*membermask=*/builder_.getInt32(-1)}, {},
                            &builder_);
  builder_.CreateRetVoid();
  EXPECT_FALSE(llvm::verifyModule(module_, &llvm::errs()));
}

TEST_F(TargetUtilTest, AMDGCNGroupBarrier) {
  module_.setTargetTriple(llvm::Triple("amdgcn"));
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId, {}, {},
                            &builder_);
  builder_.CreateRetVoid();
  EXPECT_FALSE(llvm::verifyModule(module_, &llvm::errs()));
}

TEST_F(TargetUtilTest, MusaCoordinatesAndBarrierUseVersionedShimAbi) {
  module_.setTargetTriple(llvm::Triple(musa::kMusaTargetTriple));
  struct CoordinateShim {
    TargetIntrinsicID intrinsic;
    const char* symbol;
    bool convergent;
  };
  constexpr CoordinateShim kCoordinateShims[] = {
      {TargetIntrinsicID::kBlockIdx, "__xla_musa_v1_read_ctaid_x", true},
      {TargetIntrinsicID::kBlockIdy, "__xla_musa_v1_read_ctaid_y", true},
      {TargetIntrinsicID::kBlockIdz, "__xla_musa_v1_read_ctaid_z", true},
      {TargetIntrinsicID::kThreadIdx, "__xla_musa_v1_read_tid_x", false},
      {TargetIntrinsicID::kThreadIdy, "__xla_musa_v1_read_tid_y", false},
      {TargetIntrinsicID::kThreadIdz, "__xla_musa_v1_read_tid_z", false},
      {TargetIntrinsicID::kBlockDimx, "__xla_musa_v1_read_ntid_x", false},
      {TargetIntrinsicID::kBlockDimy, "__xla_musa_v1_read_ntid_y", false},
      {TargetIntrinsicID::kBlockDimz, "__xla_musa_v1_read_ntid_z", false},
  };
  for (const CoordinateShim& shim : kCoordinateShims) {
    EmitCallToTargetIntrinsic(shim.intrinsic, {}, {}, &builder_);
  }
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kBarrierId, {}, {}, &builder_);
  builder_.CreateRetVoid();
  ASSERT_FALSE(llvm::verifyModule(module_, &llvm::errs()));

  llvm::Function* barrier =
      module_.getFunction("__xla_musa_v1_workgroup_barrier");
  ASSERT_NE(barrier, nullptr);
  for (const CoordinateShim& shim : kCoordinateShims) {
    llvm::Function* function = module_.getFunction(shim.symbol);
    ASSERT_NE(function, nullptr) << shim.symbol;
    EXPECT_EQ(function->getCallingConv(), llvm::CallingConv::C);
    EXPECT_TRUE(function->hasFnAttribute(llvm::Attribute::NoUnwind));
    EXPECT_EQ(function->hasFnAttribute(llvm::Attribute::Convergent),
              shim.convergent)
        << shim.symbol;
    EXPECT_EQ(function->getMemoryEffects(), llvm::MemoryEffects::none())
        << shim.symbol;
  }
  EXPECT_EQ(barrier->getCallingConv(), llvm::CallingConv::C);
  EXPECT_TRUE(barrier->hasFnAttribute(llvm::Attribute::NoUnwind));
  EXPECT_TRUE(barrier->hasFnAttribute(llvm::Attribute::Convergent));
  EXPECT_FALSE(barrier->hasFnAttribute(llvm::Attribute::Memory));

  for (const llvm::Function& function : module_.functions()) {
    EXPECT_FALSE(function.getName().starts_with("llvm.nvvm."));
    EXPECT_FALSE(function.getName().starts_with("llvm.amdgcn."));
  }
}

TEST_F(TargetUtilTest, MusaKernelUsesBridgeMarker) {
  module_.setTargetTriple(llvm::Triple(musa::kMusaTargetTriple));
  llvm::Function* function = module_.getFunction("fn");
  ASSERT_NE(function, nullptr);
  AnnotateFunctionAsGpuKernel(&module_, function, &builder_);
  EXPECT_EQ(function->getCallingConv(), llvm::CallingConv::C);
  llvm::Attribute marker =
      function->getFnAttribute(musa::kMusaLlvmKernelMarker);
  ASSERT_TRUE(marker.isStringAttribute());
  EXPECT_TRUE(marker.getValueAsString().empty());
}

TEST_F(TargetUtilTest, MusaGroupBarrierFailsClosed) {
  module_.setTargetTriple(llvm::Triple(musa::kMusaTargetTriple));
  EXPECT_DEATH(EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId, {},
                                         {}, &builder_),
               "outside the versioned shim ABI");
}

TEST(TargetUtil, ObtainDeviceFunctionNameExp) {
  llvm::Triple triple("nvptx64-unknown-unknown");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/F32, triple),
            "__nv_expf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/BF16, triple),
            "__nv_fast_expf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/F16, triple),
            "__nv_fast_expf");
}

TEST(TargetUtil, ObtainDeviceFunctionNameLog) {
  llvm::Triple triple("nvptx64-unknown-unknown");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/F32, triple),
            "__nv_logf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/BF16, triple),
            "__nv_fast_logf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/F16, triple),
            "__nv_fast_logf");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
