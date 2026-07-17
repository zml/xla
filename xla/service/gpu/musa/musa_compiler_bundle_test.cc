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

#include "xla/service/gpu/musa/musa_compiler_bundle.h"

#include <array>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include <sys/stat.h>
#include "xla/tsl/platform/env.h"
#include "xla/tsl/testing/temporary_directory.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr std::array<const char*, 10> kComponentNames = {
    "bridge",   "identity",  "libclang.so", "mcc",      "bundler",
    "ld.lld",   "readobj",   "libdevice.bc", "intrinsics.td",
    "builtins.def",
};

std::string Manifest() {
  const std::string sha(64, '0');
  return absl::StrCat(
      "schema=", kMusaCompilerBundleSchema, "\n",
      "xla_revision=c10-test\n",
      "current_llvm_revision=llvm-test\n",
      "provider_name=mcc-bundle-v1\n",
      "provider_fingerprint=", sha, "\n",
      "bridge_fingerprint=", sha, "\n",
      "toolchain_fingerprint=", sha, "\n",
      "libdevice_fingerprint=", sha, "\n",
      "driver_compatibility=musa-driver-3.0-compatible\n",
      "runtime_compatibility=musa-runtime-4.0.1-compatible\n",
      "bridge_executable=components/bridge\n",
      "toolchain_identity=components/identity\n",
      "libclang_cpp=components/libclang.so\n",
      "mcc=components/mcc\n",
      "clang_offload_bundler=components/bundler\n",
      "lld=components/ld.lld\n",
      "llvm_readobj=components/readobj\n",
      "libdevice=components/libdevice.bc\n",
      "intrinsics_musa_td=components/intrinsics.td\n",
      "builtins_mtgpu_def=components/builtins.def\n");
}

class MusaCompilerBundleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<tsl::testing::TemporaryDirectory> root =
        tsl::testing::TemporaryDirectory::CreateForCurrentTestcase();
    ASSERT_THAT(root, IsOk());
    root_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(*std::move(root));
    components_ = absl::StrCat(root_->path(), "/components");
    ASSERT_THAT(tsl::Env::Default()->CreateDir(components_), IsOk());
    for (const char* name : kComponentNames) {
      const std::string path = absl::StrCat(components_, "/", name);
      ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), path, "component"),
                  IsOk());
    }
    ASSERT_EQ(chmod(absl::StrCat(components_, "/bridge").c_str(), 0700), 0);
    manifest_path_ = absl::StrCat(root_->path(), "/bundle.conf");
  }

  void WriteManifest(const std::string& manifest) {
    ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), manifest_path_,
                                       manifest),
                IsOk());
  }

  std::unique_ptr<tsl::testing::TemporaryDirectory> root_;
  std::string components_;
  std::string manifest_path_;
};

TEST_F(MusaCompilerBundleTest, LoadsRelocatableStrictManifest) {
  WriteManifest(Manifest());
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> provider =
      LoadMusaCompilationProviderFromBundle(manifest_path_);
  ASSERT_THAT(provider, IsOk());
  EXPECT_EQ((*provider)->identity().xla_revision, "c10-test");
  EXPECT_EQ((*provider)->identity().provider_name, "mcc-bundle-v1");
  EXPECT_TRUE((*provider)->capabilities().vendor_llvm_isolated);
  EXPECT_EQ((*provider)->capabilities().binary_kind, "mubin");
}

TEST_F(MusaCompilerBundleTest, RejectsNonCanonicalOrEscapingManifest) {
  WriteManifest(absl::StrCat(Manifest(), "unknown=value\n"));
  EXPECT_THAT(LoadMusaCompilationProviderFromBundle(manifest_path_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unknown or duplicate")));

  WriteManifest(absl::StrReplaceAll(
      Manifest(), {{"bridge_executable=components/bridge",
                    "bridge_executable=/tmp/bridge"}}));
  EXPECT_THAT(LoadMusaCompilationProviderFromBundle(manifest_path_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("safe and relative")));
}

}  // namespace
}  // namespace xla::gpu::musa
