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

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/musa/mcc_bundle_codegen.h"
#include "xla/service/gpu/musa/musa_compilation_provider.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "xla/tools/musa_llvm_bridge/toolchain_fingerprint.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/resource_loader.h"
#include "xla/tsl/testing/temporary_directory.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::IsOk;
using bridge::MusaBridgeFingerprints;
using bridge::MusaBridgeToolchainPaths;

std::string CanonicalPath(const std::string& path) {
  char* resolved = realpath(path.c_str(), nullptr);
  EXPECT_NE(resolved, nullptr) << path;
  if (resolved == nullptr) return {};
  std::string result(resolved);
  std::free(resolved);
  return result;
}

std::string BridgePath() {
  return CanonicalPath(tsl::GetDataDependencyFilepath(
      "xla/tools/musa_llvm_bridge/musa-llvm-bridge"));
}

std::string RunfileFromEnvironment(absl::string_view name) {
  const char* logical = std::getenv(std::string(name).c_str());
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  EXPECT_NE(logical, nullptr) << name;
  EXPECT_NE(test_srcdir, nullptr);
  return CanonicalPath(absl::StrCat(test_srcdir == nullptr ? "" : test_srcdir,
                                    "/", logical == nullptr ? "" : logical));
}

std::string Dirname(absl::string_view path) {
  const size_t slash = path.rfind('/');
  if (slash == 0) return "/";
  return std::string(path.substr(0, slash));
}

MusaBridgeToolchainPaths ToolchainPaths() {
  MusaBridgeToolchainPaths paths;
  paths.bridge_executable = BridgePath();
  paths.toolchain_identity = RunfileFromEnvironment("MUSA_TEST_IDENTITY");
  paths.libclang_cpp = RunfileFromEnvironment("MUSA_TEST_LIBCLANG_CPP");
  paths.mcc = RunfileFromEnvironment("MUSA_TEST_MCC");
  paths.clang_offload_bundler = RunfileFromEnvironment("MUSA_TEST_BUNDLER");
  paths.lld = RunfileFromEnvironment("MUSA_TEST_LLD");
  paths.llvm_readobj = RunfileFromEnvironment("MUSA_TEST_READOBJ");
  paths.libdevice = RunfileFromEnvironment("MUSA_TEST_LIBDEVICE");
  const std::string sdk_root = Dirname(Dirname(paths.mcc));
  paths.intrinsics_musa_td =
      absl::StrCat(sdk_root, "/include/llvm/IR/IntrinsicsMUSA.td");
  paths.builtins_mtgpu_def =
      absl::StrCat(sdk_root, "/include/clang/Basic/BuiltinsMTGPU.def");
  return paths;
}

std::string ReadGolden() {
  std::ifstream input(tsl::GetDataDependencyFilepath(
                          "xla/service/gpu/musa/testdata/"
                          "llvm14_compatibility/elemental.llvm14.ll"),
                      std::ios::binary);
  EXPECT_TRUE(input.is_open());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::string ComponentSha(const MusaBridgeFingerprints& fingerprints,
                         absl::string_view name) {
  const auto found = std::find_if(
      fingerprints.components.begin(), fingerprints.components.end(),
      [&](const bridge::MusaBridgeComponentFingerprint& component) {
        return component.name == name;
      });
  EXPECT_NE(found, fingerprints.components.end()) << name;
  return found == fingerprints.components.end() ? "" : found->sha256;
}

TEST(MusaCompilationProviderProcessTest,
     CompilesCheckedMubinThroughPinnedIsolatedBridgeAndCachesIt) {
  const MusaBridgeToolchainPaths bridge_paths = ToolchainPaths();
  absl::StatusOr<MusaBridgeFingerprints> fingerprints =
      bridge::FingerprintMusaBridgeToolchain(
          bridge_paths, std::string(MccBundleProviderName()),
          std::string(MccBundleProviderCanonicalText()));
  ASSERT_THAT(fingerprints, IsOk());

  absl::StatusOr<tsl::testing::TemporaryDirectory> root =
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase();
  ASSERT_THAT(root, IsOk());
  const std::string temporary = absl::StrCat(root->path(), "/temporary");
  const std::string cache = absl::StrCat(root->path(), "/cache");
  ASSERT_THAT(tsl::Env::Default()->CreateDir(temporary), IsOk());
  ASSERT_THAT(tsl::Env::Default()->CreateDir(cache), IsOk());

  MusaCompilationProviderSelection selection;
  selection.subprocess.identity = {
      .xla_revision = "ed8f8caf84",
      .current_llvm_revision = "llvm-7dd886751573",
      .provider_name = fingerprints->provider_name,
      .provider_fingerprint = fingerprints->provider_fingerprint,
      .bridge_fingerprint = fingerprints->bridge_fingerprint,
      .toolchain_fingerprint = fingerprints->toolchain_fingerprint,
      .libdevice_fingerprint = ComponentSha(*fingerprints, "libdevice"),
      .driver_compatibility = "musa-driver-3.0-compatible",
      .runtime_compatibility = "musa-runtime-4.0.1-compatible",
  };
  MusaSubprocessBridgePaths& paths = selection.subprocess.paths;
  paths.bridge_executable = bridge_paths.bridge_executable;
  paths.toolchain_identity = bridge_paths.toolchain_identity;
  paths.libclang_cpp = bridge_paths.libclang_cpp;
  paths.mcc = bridge_paths.mcc;
  paths.clang_offload_bundler = bridge_paths.clang_offload_bundler;
  paths.lld = bridge_paths.lld;
  paths.llvm_readobj = bridge_paths.llvm_readobj;
  paths.libdevice = bridge_paths.libdevice;
  paths.intrinsics_musa_td = bridge_paths.intrinsics_musa_td;
  paths.builtins_mtgpu_def = bridge_paths.builtins_mtgpu_def;
  selection.subprocess.temporary_directory_root = temporary;
  selection.subprocess.cache_directory = cache;

  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> provider =
      AssembleMusaCompilationProvider(selection);
  ASSERT_THAT(provider, IsOk());
  MusaLlvm14CompatibilityResult module;
  module.normalized_llvm = ReadGolden();
  module.normalized_llvm_sha256 = MusaBridgeSha256Hex(module.normalized_llvm);
  module.metadata.module_name = "provider_process_test";
  module.metadata.kernel_entry_names = {"kernel"};

  absl::StatusOr<MusaCompilationArtifact> compiled =
      (*provider)->Compile(module, {});
  ASSERT_THAT(compiled, IsOk());
  EXPECT_FALSE(compiled->cache_hit);
  EXPECT_THAT(stream_executor::musa::ValidateMusaMubin(compiled->mubin),
              IsOk());
  EXPECT_EQ(compiled->mubin_sha256,
            MusaBridgeSha256Hex(absl::string_view(
                reinterpret_cast<const char*>(compiled->mubin.data()),
                compiled->mubin.size())));

  absl::StatusOr<MusaCompilationArtifact> cached =
      (*provider)->Compile(module, {});
  ASSERT_THAT(cached, IsOk());
  EXPECT_TRUE(cached->cache_hit);
  EXPECT_EQ(cached->mubin, compiled->mubin);
  EXPECT_EQ(cached->cache_key, compiled->cache_key);
}

}  // namespace
}  // namespace xla::gpu::musa
