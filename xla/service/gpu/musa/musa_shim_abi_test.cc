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

#include "xla/service/gpu/musa/musa_shim_abi.h"

#include <array>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::IsOk;

std::string Sha256Hex(absl::string_view value) {
  llvm::SHA256 sha256;
  sha256.update(llvm::StringRef(value.data(), value.size()));
  std::array<uint8_t, 32> digest = sha256.final();
  std::string result;
  result.reserve(64);
  for (uint8_t byte : digest) {
    absl::StrAppend(&result, absl::StrFormat("%02x", byte));
  }
  return result;
}

TEST(MusaShimAbiTest, MappingIsFiniteVersionedAndSelfConsistent) {
  EXPECT_THAT(ValidateMusaShimTable(), IsOk());
  EXPECT_EQ(MusaShimSpecs().size(), 15);
  for (const MusaShimSpec& spec : MusaShimSpecs()) {
    EXPECT_TRUE(absl::StartsWith(spec.xla_symbol, "__xla_musa_v1_"));
    EXPECT_TRUE(absl::StartsWith(spec.vendor_intrinsic, "llvm.musa."));
    EXPECT_EQ(spec.minimum_mapping_version, 1);
    EXPECT_FALSE(MusaShimSignatureText(spec.signature).empty());
    EXPECT_FALSE(MusaMemoryEffectsText(spec.memory_effects).empty());
    EXPECT_EQ(FindMusaShim(spec.xla_symbol), &spec);
  }
  EXPECT_EQ(FindMusaShim("__xla_musa_v1_unknown"), nullptr);
  EXPECT_EQ(FindMusaShim("__xla_musa_v2_read_tid_x"), nullptr);
}

TEST(MusaShimAbiTest, AddressSpacesMatchPinnedSdk) {
  ASSERT_EQ(MusaAddressSpaceSpecs().size(), 6);
  for (uint32_t address_space : {0, 1, 2, 3, 5}) {
    const MusaAddressSpaceSpec* spec = FindMusaAddressSpace(address_space);
    ASSERT_NE(spec, nullptr);
    EXPECT_TRUE(spec->allowed_in_interchange);
  }
  const MusaAddressSpaceSpec* reserved = FindMusaAddressSpace(4);
  ASSERT_NE(reserved, nullptr);
  EXPECT_FALSE(reserved->allowed_in_interchange);
  EXPECT_EQ(reserved->name, "reserved");
  EXPECT_EQ(reserved->pointer_width, 32);
  const MusaAddressSpaceSpec* private_scratch = FindMusaAddressSpace(5);
  ASSERT_NE(private_scratch, nullptr);
  EXPECT_EQ(private_scratch->name, "private_scratch");
  EXPECT_EQ(private_scratch->pointer_width, 64);
  EXPECT_EQ(FindMusaAddressSpace(6), nullptr);
}

TEST(MusaShimAbiTest, RiskyCapabilitiesAreExplicitlyRejected) {
  constexpr absl::string_view kRequired[] = {
      "atomics", "non_generic_math", "subgroup_barrier", "subgroup_shuffle",
      "subgroup_vote"};
  ASSERT_EQ(MusaUnsupportedCapabilities().size(), std::size(kRequired));
  for (int i = 0; i < std::size(kRequired); ++i) {
    EXPECT_EQ(MusaUnsupportedCapabilities()[i].name, kRequired[i]);
    EXPECT_FALSE(MusaUnsupportedCapabilities()[i].reason.empty());
  }
}

TEST(MusaShimAbiTest, CanonicalMappingHasReviewedSha256) {
  const std::string canonical = MusaShimCanonicalText();
  EXPECT_TRUE(absl::StrContains(canonical, "triple=mtgpu-mt-musa"));
  EXPECT_TRUE(absl::StrContains(canonical, "arch=mp_21"));
  EXPECT_TRUE(
      absl::StrContains(canonical, "as\t1\tglobal\tglobal\t64\tinterchange"));
  EXPECT_TRUE(absl::StrContains(
      canonical, "as\t4\tvendor-internal\treserved\t32\tvendor"));
  EXPECT_FALSE(absl::StrContains(canonical, "llvm.nvvm"));
  EXPECT_FALSE(absl::StrContains(canonical, "llvm.amdgcn"));
  EXPECT_EQ(Sha256Hex(canonical), kMusaShimMappingSha256);
}

TEST(MusaShimAbiTest, ShimSemanticsMatchPinnedCompilerDeclarations) {
  const MusaShimSpec* tid = FindMusaShim("__xla_musa_v1_read_tid_x");
  ASSERT_NE(tid, nullptr);
  EXPECT_EQ(tid->signature, MusaShimSignature::kI32Void);
  EXPECT_EQ(tid->memory_effects, MusaMemoryEffects::kNone);
  EXPECT_FALSE(tid->convergent);

  const MusaShimSpec* ctaid = FindMusaShim("__xla_musa_v1_read_ctaid_x");
  ASSERT_NE(ctaid, nullptr);
  EXPECT_EQ(ctaid->memory_effects, MusaMemoryEffects::kNone);
  EXPECT_TRUE(ctaid->convergent);

  const MusaShimSpec* clock = FindMusaShim("__xla_musa_v1_clock64");
  ASSERT_NE(clock, nullptr);
  EXPECT_EQ(clock->signature, MusaShimSignature::kI64Void);
  EXPECT_EQ(clock->memory_effects, MusaMemoryEffects::kInaccessibleReadWrite);

  const MusaShimSpec* barrier = FindMusaShim("__xla_musa_v1_workgroup_barrier");
  ASSERT_NE(barrier, nullptr);
  EXPECT_EQ(barrier->signature, MusaShimSignature::kVoidVoid);
  EXPECT_EQ(barrier->vendor_intrinsic, "llvm.musa.barrier0");
  EXPECT_EQ(barrier->memory_effects, MusaMemoryEffects::kReadWrite);
  EXPECT_TRUE(barrier->convergent);
}

}  // namespace
}  // namespace xla::gpu::musa
