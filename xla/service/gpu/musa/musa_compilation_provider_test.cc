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

#include "xla/service/gpu/musa/musa_compilation_provider.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include <sys/stat.h>
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/resource_loader.h"
#include "xla/tsl/testing/temporary_directory.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::string HelperPath() {
  return tsl::GetDataDependencyFilepath(
      "xla/service/gpu/musa/musa_compilation_provider_test_helper");
}

MusaCompilationIdentity Identity() {
  return {
      .xla_revision = "ed8f8caf84",
      .current_llvm_revision = "llvm-head",
      .provider_name = "subprocess-test-v1",
      .provider_fingerprint = std::string(64, 'a'),
      .bridge_fingerprint = std::string(64, 'b'),
      .toolchain_fingerprint = std::string(64, 'c'),
      .libdevice_fingerprint = std::string(64, 'd'),
      .driver_compatibility = "musa-driver-3.0-compatible",
      .runtime_compatibility = "musa-runtime-4.0.1-compatible",
  };
}

MusaLlvm14CompatibilityResult Module(std::string name) {
  MusaLlvm14CompatibilityResult result;
  result.normalized_llvm = absl::StrCat(
      "; ModuleID = '", name, "'\n", "source_filename = \"", name, "\"\n",
      "target datalayout = \"", kMusaDataLayout, "\"\n", "target triple = \"",
      kMusaTargetTriple, "\"\n\n",
      "define void @kernel(ptr addrspace(1) %out) {\n", "entry:\n",
      "  %tid = call i32 @__xla_musa_v1_read_tid_x()\n",
      "  store i32 %tid, ptr addrspace(1) %out, align 4\n", "  ret void\n",
      "}\n\n", "declare i32 @__xla_musa_v1_read_tid_x() #0\n\n",
      "attributes #0 = { nounwind readnone }\n");
  result.normalized_llvm_sha256 = MusaBridgeSha256Hex(result.normalized_llvm);
  result.metadata.module_name = std::move(name);
  result.metadata.kernel_entry_names = {"kernel"};
  return result;
}

struct State {
  int active = 0;
  int maximum = 0;
  int calls = 0;
};

State ReadState(const std::string& path) {
  std::ifstream input(path);
  State state;
  input >> state.active >> state.maximum >> state.calls;
  return state;
}

class MusaCompilationProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<tsl::testing::TemporaryDirectory> temp =
        tsl::testing::TemporaryDirectory::CreateForCurrentTestcase();
    ASSERT_THAT(temp, IsOk());
    temp_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(*std::move(temp));
    temporary_root_ = absl::StrCat(temp_->path(), "/temporary");
    cache_root_ = absl::StrCat(temp_->path(), "/cache");
    state_path_ = absl::StrCat(temp_->path(), "/state");
    ASSERT_THAT(tsl::Env::Default()->CreateDir(temporary_root_), IsOk());
    ASSERT_THAT(tsl::Env::Default()->CreateDir(cache_root_), IsOk());
  }

  MusaCompilationProviderSelection Selection(bool cache = false) const {
    MusaCompilationProviderSelection selection;
    selection.subprocess.identity = Identity();
    MusaSubprocessBridgePaths& paths = selection.subprocess.paths;
    paths.bridge_executable = HelperPath();
    paths.toolchain_identity = state_path_;
    paths.libclang_cpp = HelperPath();
    paths.mcc = HelperPath();
    paths.clang_offload_bundler = HelperPath();
    paths.lld = HelperPath();
    paths.llvm_readobj = HelperPath();
    paths.libdevice = HelperPath();
    paths.intrinsics_musa_td = HelperPath();
    paths.builtins_mtgpu_def = HelperPath();
    selection.subprocess.temporary_directory_root = temporary_root_;
    selection.subprocess.cache_directory = cache ? cache_root_ : "";
    selection.subprocess.subprocess_limits.timeout = std::chrono::seconds(10);
    selection.subprocess.subprocess_limits.max_stdout_bytes = 1 << 20;
    selection.subprocess.subprocess_limits.max_stderr_bytes = 1 << 20;
    return selection;
  }

  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> Provider(
      bool cache = false) const {
    return AssembleMusaCompilationProvider(Selection(cache));
  }

  void ExpectTemporaryRootEmpty() const {
    std::vector<std::string> children;
    ASSERT_THAT(tsl::Env::Default()->GetChildren(temporary_root_, &children),
                IsOk());
    EXPECT_TRUE(children.empty());
  }

  std::unique_ptr<tsl::testing::TemporaryDirectory> temp_;
  std::string temporary_root_;
  std::string cache_root_;
  std::string state_path_;
};

TEST_F(MusaCompilationProviderTest, FreezesProviderCapabilitiesAndRequest) {
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> provider =
      Provider();
  ASSERT_THAT(provider, IsOk());
  EXPECT_EQ((*provider)->name(), "subprocess-test-v1");
  EXPECT_EQ((*provider)->identity().toolchain_fingerprint,
            std::string(64, 'c'));
  const MusaCompilationCapabilities capabilities = (*provider)->capabilities();
  EXPECT_TRUE(capabilities.supports_compile);
  EXPECT_FALSE(capabilities.supports_relocatable);
  EXPECT_FALSE(capabilities.supports_compile_and_link);
  EXPECT_TRUE(capabilities.vendor_llvm_isolated);
  EXPECT_EQ(capabilities.binary_kind, "mubin");
  EXPECT_EQ(capabilities.target, "mtgpu-mt-musa/mp_21");

  MusaCompilationOptions options;
  options.optimization_level = 3;
  options.fast_math = true;
  absl::StatusOr<MusaBridgeCompileRequest> request =
      BuildMusaBridgeCompileRequest(Module("request_contract"), Identity(),
                                    options);
  ASSERT_THAT(request, IsOk());
  EXPECT_EQ(request->optimization_level(), 3);
  EXPECT_TRUE(request->numerical_flags().fast_math());
  EXPECT_EQ(request->provider_fingerprint(), Identity().provider_fingerprint);
  EXPECT_EQ(request->kernel_entry_names_size(), 1);
  EXPECT_EQ(request->exported_symbol_names_size(), 1);
  EXPECT_THAT(MusaCompilationCacheKey(*request, Identity()), IsOk());
}

TEST_F(MusaCompilationProviderTest, CacheIdentityObservesEveryExternalClosure) {
  const MusaLlvm14CompatibilityResult module = Module("cache_identity");
  const MusaCompilationOptions options;
  const MusaCompilationIdentity identity = Identity();
  absl::StatusOr<MusaBridgeCompileRequest> request =
      BuildMusaBridgeCompileRequest(module, identity, options);
  ASSERT_THAT(request, IsOk());
  absl::StatusOr<std::string> baseline =
      MusaCompilationCacheKey(*request, identity);
  ASSERT_THAT(baseline, IsOk());

  for (int mutation = 0; mutation < 3; ++mutation) {
    MusaCompilationIdentity changed = identity;
    if (mutation == 0) changed.libdevice_fingerprint = std::string(64, 'e');
    if (mutation == 1) changed.driver_compatibility += ".new";
    if (mutation == 2) changed.runtime_compatibility += ".new";
    absl::StatusOr<std::string> key =
        MusaCompilationCacheKey(*request, changed);
    ASSERT_THAT(key, IsOk());
    EXPECT_NE(*key, *baseline);
  }

  MusaCompilationOptions changed_options = options;
  changed_options.deterministic = false;
  absl::StatusOr<MusaBridgeCompileRequest> changed_request =
      BuildMusaBridgeCompileRequest(module, identity, changed_options);
  ASSERT_THAT(changed_request, IsOk());
  absl::StatusOr<std::string> changed_key =
      MusaCompilationCacheKey(*changed_request, identity);
  ASSERT_THAT(changed_key, IsOk());
  EXPECT_NE(*changed_key, *baseline);

  MusaLlvm14CompatibilityResult changed_module = Module("cache_identity_2");
  changed_request =
      BuildMusaBridgeCompileRequest(changed_module, identity, options);
  ASSERT_THAT(changed_request, IsOk());
  changed_key = MusaCompilationCacheKey(*changed_request, identity);
  ASSERT_THAT(changed_key, IsOk());
  EXPECT_NE(*changed_key, *baseline);
}

TEST_F(MusaCompilationProviderTest, CompilesCachesAndRecoversCorruption) {
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> provider =
      Provider(/*cache=*/true);
  ASSERT_THAT(provider, IsOk());
  const MusaLlvm14CompatibilityResult module = Module("success");

  absl::StatusOr<MusaCompilationArtifact> first =
      (*provider)->Compile(module, {});
  ASSERT_THAT(first, IsOk());
  EXPECT_FALSE(first->cache_hit);
  EXPECT_FALSE(first->recovered_invalid_cache_entry);
  EXPECT_EQ(first->mubin_sha256,
            MusaBridgeSha256Hex(std::string_view(
                reinterpret_cast<const char*>(first->mubin.data()),
                first->mubin.size())));
  EXPECT_THAT(stream_executor::musa::ValidateMusaMubin(first->mubin), IsOk());
  ExpectTemporaryRootEmpty();

  absl::StatusOr<MusaCompilationArtifact> second =
      (*provider)->Compile(module, {});
  ASSERT_THAT(second, IsOk());
  EXPECT_TRUE(second->cache_hit);
  EXPECT_EQ(second->mubin_sha256, first->mubin_sha256);
  EXPECT_EQ(second->mubin, first->mubin);
  EXPECT_EQ(ReadState(state_path_).calls, 1);

  const std::string cache_path =
      absl::StrCat(cache_root_, "/", first->cache_key, ".musa-cache-v1");
  std::ofstream corrupt(cache_path, std::ios::binary | std::ios::trunc);
  corrupt << "corrupt";
  corrupt.close();
  absl::StatusOr<MusaCompilationArtifact> recovered =
      (*provider)->Compile(module, {});
  ASSERT_THAT(recovered, IsOk());
  EXPECT_FALSE(recovered->cache_hit);
  EXPECT_TRUE(recovered->recovered_invalid_cache_entry);
  EXPECT_EQ(ReadState(state_path_).calls, 2);
  struct stat metadata;
  ASSERT_EQ(stat(cache_path.c_str(), &metadata), 0);
  EXPECT_EQ(metadata.st_mode & 0777, 0600);
}

TEST_F(MusaCompilationProviderTest, MapsProcessAndProtocolFailures) {
  struct Case {
    const char* mode;
    absl::StatusCode code;
  };
  for (const Case& test : {
           Case{"crash", absl::StatusCode::kInternal},
           Case{"malformed", absl::StatusCode::kDataLoss},
           Case{"oversized", absl::StatusCode::kResourceExhausted},
           Case{"rejected", absl::StatusCode::kInvalidArgument},
           Case{"compile_error", absl::StatusCode::kInternal},
           Case{"internal_error", absl::StatusCode::kInternal},
           Case{"identity_mismatch", absl::StatusCode::kFailedPrecondition},
           Case{"invalid_mubin", absl::StatusCode::kDataLoss},
           Case{"stderr", absl::StatusCode::kDataLoss},
       }) {
    MusaCompilationProviderSelection selection = Selection();
    if (std::string(test.mode) == "oversized") {
      selection.subprocess.subprocess_limits.max_stdout_bytes = 1024;
    }
    absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> provider =
        AssembleMusaCompilationProvider(selection);
    ASSERT_THAT(provider, IsOk()) << test.mode;
    EXPECT_THAT((*provider)->Compile(Module(test.mode), {}),
                StatusIs(test.code))
        << test.mode;
    ExpectTemporaryRootEmpty();
  }
}

TEST_F(MusaCompilationProviderTest, EnforcesTimeoutAndCancellation) {
  MusaCompilationProviderSelection selection = Selection();
  selection.subprocess.subprocess_limits.timeout =
      std::chrono::milliseconds(40);
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> timeout_provider =
      AssembleMusaCompilationProvider(selection);
  ASSERT_THAT(timeout_provider, IsOk());
  EXPECT_THAT((*timeout_provider)->Compile(Module("timeout"), {}),
              StatusIs(absl::StatusCode::kDeadlineExceeded));

  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> cancel_provider =
      Provider();
  ASSERT_THAT(cancel_provider, IsOk());
  std::atomic<bool> cancel = false;
  MusaCompilationOptions options;
  options.cancellation_requested = [&] { return cancel.load(); };
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cancel.store(true);
  });
  EXPECT_THAT((*cancel_provider)->Compile(Module("cancel"), options),
              StatusIs(absl::StatusCode::kCancelled));
  canceller.join();
  ExpectTemporaryRootEmpty();
}

TEST_F(MusaCompilationProviderTest, CancelsWhileWaitingForProviderSlot) {
  MusaCompilationProviderSelection selection = Selection();
  selection.subprocess.max_concurrent_compilations = 1;
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> provider =
      AssembleMusaCompilationProvider(selection);
  ASSERT_THAT(provider, IsOk());

  absl::Status first_status;
  std::thread first([&] {
    first_status =
        (*provider)->Compile(Module("concurrent_holder"), {}).status();
  });
  for (int attempt = 0; attempt < 100 && ReadState(state_path_).active != 1;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (ReadState(state_path_).active != 1) {
    first.join();
    FAIL() << "first compilation did not acquire the provider slot";
  }

  std::atomic<bool> cancel = false;
  MusaCompilationOptions options;
  options.cancellation_requested = [&] { return cancel.load(); };
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    cancel.store(true);
  });
  EXPECT_THAT((*provider)->Compile(Module("concurrent_waiter"), options),
              StatusIs(absl::StatusCode::kCancelled,
                       HasSubstr("waiting for a provider slot")));
  canceller.join();
  first.join();
  EXPECT_THAT(first_status, IsOk());
  EXPECT_EQ(ReadState(state_path_).calls, 1);
  ExpectTemporaryRootEmpty();
}

TEST_F(MusaCompilationProviderTest, BoundsConcurrentBridgeProcesses) {
  MusaCompilationProviderSelection selection = Selection();
  selection.subprocess.max_concurrent_compilations = 2;
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> provider =
      AssembleMusaCompilationProvider(selection);
  ASSERT_THAT(provider, IsOk());

  std::atomic<int> failures = 0;
  std::vector<std::thread> workers;
  for (int i = 0; i < 6; ++i) {
    workers.emplace_back([&, i] {
      if (!(*provider)
               ->Compile(Module(absl::StrCat("concurrent_", i)), {})
               .ok()) {
        ++failures;
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  EXPECT_EQ(failures.load(), 0);
  const State state = ReadState(state_path_);
  EXPECT_EQ(state.active, 0);
  EXPECT_EQ(state.calls, 6);
  EXPECT_LE(state.maximum, 2);
  EXPECT_GE(state.maximum, 2);
  ExpectTemporaryRootEmpty();
}

TEST_F(MusaCompilationProviderTest, RejectsUnsafeOrUnqualifiedSelections) {
  MusaCompilationProviderSelection selection = Selection();
  selection.kind = MusaCompilationProviderKind::kMccBundleInProcess;
  EXPECT_THAT(AssembleMusaCompilationProvider(selection),
              StatusIs(absl::StatusCode::kUnavailable, HasSubstr("isolated")));
  selection.kind = MusaCompilationProviderKind::kDirectInternalTools;
  EXPECT_THAT(
      AssembleMusaCompilationProvider(selection),
      StatusIs(absl::StatusCode::kUnavailable, HasSubstr("diagnostic")));

  selection = Selection();
  selection.subprocess.max_concurrent_compilations = 0;
  EXPECT_THAT(AssembleMusaCompilationProvider(selection),
              StatusIs(absl::StatusCode::kInvalidArgument));
  selection = Selection();
  selection.subprocess.paths.bridge_executable = "relative/bridge";
  EXPECT_THAT(AssembleMusaCompilationProvider(selection),
              StatusIs(absl::StatusCode::kInvalidArgument));
  selection = Selection();
  selection.subprocess.identity.provider_name = "provider name with spaces";
  EXPECT_THAT(AssembleMusaCompilationProvider(selection),
              StatusIs(absl::StatusCode::kInvalidArgument));
  selection = Selection();
  selection.subprocess.identity.libdevice_fingerprint = "not-a-digest";
  EXPECT_THAT(AssembleMusaCompilationProvider(selection),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu::musa
