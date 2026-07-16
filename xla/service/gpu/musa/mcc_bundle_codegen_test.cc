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

#include "xla/service/gpu/musa/mcc_bundle_codegen.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include <unistd.h>
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/resource_loader.h"
#include "xla/tsl/testing/temporary_directory.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

std::string HelperPath() {
  return tsl::GetDataDependencyFilepath(
      "xla/service/gpu/musa/mcc_bundle_codegen_test_helper");
}

std::string VendorLlvm(std::string mode = "success") {
  return absl::StrCat("; FAKE_MODE=", mode, "\n",
                      "target triple = \"mtgpu-mt-musa\"\n",
                      "define void @kernel(ptr %out) { ret void }\n");
}

class MccBundleCodegenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<tsl::testing::TemporaryDirectory> temp =
        tsl::testing::TemporaryDirectory::CreateForCurrentTestcase();
    ASSERT_TRUE(temp.ok()) << temp.status();
    temp_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(*std::move(temp));
  }

  MccBundleCodegenOptions Options() const {
    MccBundleCodegenOptions options;
    options.mcc_path = HelperPath();
    options.clang_offload_bundler_path = HelperPath();
    options.temporary_directory_root = temp_->path();
    return options;
  }

  void ExpectProviderDirectoryWasRemoved() const {
    std::vector<std::string> children;
    ASSERT_TRUE(
        tsl::Env::Default()->GetChildren(temp_->path(), &children).ok());
    EXPECT_TRUE(children.empty());
  }

 private:
  std::unique_ptr<tsl::testing::TemporaryDirectory> temp_;
};

TEST_F(MccBundleCodegenTest, CanonicalProviderContractFreezesBehavior) {
  EXPECT_EQ(MccBundleProviderName(), "mcc-bundle-v1");
  const std::string contract(MccBundleProviderCanonicalText());
  EXPECT_THAT(contract, HasSubstr("schema=xla-musa-mcc-bundle-v1\n"));
  EXPECT_THAT(contract, HasSubstr("mcc_argv=-mtgpu|--musa-device-only|"));
  EXPECT_THAT(contract, HasSubstr("cardinality:exactly-one\n"));
  EXPECT_THAT(contract, HasSubstr("--targets={reported_bundle_id}"));
  EXPECT_EQ(contract.find("/tmp"), std::string::npos);
  EXPECT_EQ(contract.back(), '\n');

  MccBundleCodegenOptions options;
  EXPECT_EQ(options.max_llvm_bytes, kMusaBridgeMaxLlvmBytes);
  EXPECT_EQ(options.max_mubin_bytes, kMusaBridgeMaxMubinBytes);
  EXPECT_EQ(options.max_diagnostic_bytes, kMusaBridgeMaxDiagnosticMessageBytes);
}

TEST_F(MccBundleCodegenTest, UsesExactArgvEnvironmentAndReportedBundleId) {
  absl::StatusOr<MccBundleCodegenResult> result =
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm(), Options());
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->selected_bundle_id, "musa-mtgpu-mt-musa---mp_21");
  const std::string mubin(result->mubin.begin(), result->mubin.end());
  EXPECT_THAT(mubin, HasSubstr("MUBIN:"));
  const std::string private_directory = mubin.substr(strlen("MUBIN:"));
  EXPECT_NE(access(private_directory.c_str(), F_OK), 0);
  ExpectProviderDirectoryWasRemoved();
}

TEST_F(MccBundleCodegenTest, RejectsAnythingOutsideFrozenContract) {
  MccBundleCodegenOptions options = Options();
  options.architecture = "mp_31";
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm(), options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("frozen")));

  options = Options();
  options.mcc_path = "relative/mcc";
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm(), options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("absolute")));

  options = Options();
  options.max_llvm_bytes = 8;
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm(), options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("too large")));
  ExpectProviderDirectoryWasRemoved();
}

TEST_F(MccBundleCodegenTest, ReportsEveryToolStageFailureAndCleansUp) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"bundle_fail", "LLVM bundling"},
      {"mcc_fail", "MCC codegen"},
      {"list_fail", "bundle target listing"},
      {"unbundle_fail", "MUBIN extraction"},
  };
  for (const auto& [mode, expected_stage] : cases) {
    EXPECT_THAT(
        CompileVerifiedMusaLlvmWithMcc(VendorLlvm(mode), Options()),
        StatusIs(absl::StatusCode::kInternal, HasSubstr(expected_stage)));
    ExpectProviderDirectoryWasRemoved();
  }
}

TEST_F(MccBundleCodegenTest, BoundsTimeoutAndToolOutputAndCleansUp) {
  MccBundleCodegenOptions options = Options();
  options.subprocess_limits.timeout = std::chrono::milliseconds(40);
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm("timeout"), options),
      StatusIs(absl::StatusCode::kDeadlineExceeded, HasSubstr("timed out")));
  ExpectProviderDirectoryWasRemoved();

  options = Options();
  options.subprocess_limits.max_stdout_bytes = 1024;
  EXPECT_THAT(CompileVerifiedMusaLlvmWithMcc(VendorLlvm("output"), options),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("output limit")));
  ExpectProviderDirectoryWasRemoved();
}

TEST_F(MccBundleCodegenTest, RejectsWrongDuplicateAndMalformedBundleIds) {
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm("wrong_bundle"), Options()),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("exactly one")));
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm("duplicate_bundle"), Options()),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("exactly one")));
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm("malformed_bundle"), Options()),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("malformed")));
  ExpectProviderDirectoryWasRemoved();
}

TEST_F(MccBundleCodegenTest, BoundsExtractedMubinAndCleansUp) {
  MccBundleCodegenOptions options = Options();
  options.max_mubin_bytes = 1024;
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm("large_mubin"), options),
      StatusIs(absl::StatusCode::kResourceExhausted,
               HasSubstr("MUBIN exceeds")));
  ExpectProviderDirectoryWasRemoved();
}

TEST_F(MccBundleCodegenTest, RejectsFifoMubinWithoutBlocking) {
  EXPECT_THAT(
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm("fifo_mubin"), Options()),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("regular file")));
  ExpectProviderDirectoryWasRemoved();
}

TEST_F(MccBundleCodegenTest, SanitizesRedactsAndBoundsDiagnostics) {
  MccBundleCodegenOptions options = Options();
  options.max_diagnostic_bytes = 96;
  absl::StatusOr<MccBundleCodegenResult> result =
      CompileVerifiedMusaLlvmWithMcc(VendorLlvm("diagnostic"), options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_LE(result->diagnostics.size(), options.max_diagnostic_bytes);
  EXPECT_THAT(result->diagnostics, HasSubstr("warning: ?path=<temp>"));
  EXPECT_THAT(result->diagnostics,
              Not(HasSubstr(options.temporary_directory_root)));
  ExpectProviderDirectoryWasRemoved();
}

}  // namespace
}  // namespace xla::gpu::musa
