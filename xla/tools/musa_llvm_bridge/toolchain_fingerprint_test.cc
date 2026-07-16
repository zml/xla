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

#include "xla/tools/musa_llvm_bridge/toolchain_fingerprint.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "xla/service/gpu/musa/protocol.h"

namespace xla::gpu::musa::bridge {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

void Write(const std::string& path, const std::string& contents) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(stream.good());
  stream << contents;
  ASSERT_TRUE(stream.good());
}

struct FixturePaths {
  std::string root;
  MusaBridgeToolchainPaths paths;
};

FixturePaths MakeFixture() {
  FixturePaths fixture;
  fixture.root = absl::StrCat(::testing::TempDir(), "/musa-fingerprint");
  if (mkdir(fixture.root.c_str(), 0700) != 0 && errno != EEXIST) {
    std::abort();
  }
  const auto path = [&](const char* name) {
    return absl::StrCat(fixture.root, "/", name);
  };
  fixture.paths.bridge_executable = path("bridge");
  fixture.paths.toolchain_identity = path("identity");
  fixture.paths.libclang_cpp = path("libclang-cpp.so.14");
  fixture.paths.mcc = path("mcc");
  fixture.paths.clang_offload_bundler = path("bundler");
  fixture.paths.lld = path("lld");
  fixture.paths.llvm_readobj = path("readobj");
  fixture.paths.libdevice = path("libdevice.bc");
  fixture.paths.intrinsics_musa_td = path("IntrinsicsMUSA.td");
  fixture.paths.builtins_mtgpu_def = path("BuiltinsMTGPU.def");
  Write(fixture.paths.bridge_executable, "bridge");
  Write(fixture.paths.toolchain_identity,
        "schema=xla-musa-toolchain-v1\n"
        "enabled=1\n"
        "distro_sha256=abcd\n");
  Write(fixture.paths.libclang_cpp, "vendor-llvm");
  Write(fixture.paths.mcc, "mcc");
  Write(fixture.paths.clang_offload_bundler, "bundler");
  Write(fixture.paths.lld, "lld");
  Write(fixture.paths.llvm_readobj, "readobj");
  Write(fixture.paths.libdevice, "libdevice");
  Write(fixture.paths.intrinsics_musa_td, "intrinsics");
  Write(fixture.paths.builtins_mtgpu_def, "builtins");
  return fixture;
}

constexpr char kProviderContract[] =
    "schema=xla-musa-mcc-bundle-v1\n"
    "target=mtgpu-mt-musa/mp_21\n"
    "optimization=O2\n";

TEST(MusaBridgeToolchainFingerprintTest, IsPathIndependentAndComplete) {
  FixturePaths first = MakeFixture();
  absl::StatusOr<MusaBridgeFingerprints> fingerprint =
      FingerprintMusaBridgeToolchain(first.paths, "mcc-bundle-v1",
                                     kProviderContract);
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  EXPECT_EQ(fingerprint->components.size(), 10);
  EXPECT_EQ(fingerprint->bridge_fingerprint, MusaBridgeSha256Hex("bridge"));
  EXPECT_EQ(fingerprint->provider_fingerprint.size(), 64);
  EXPECT_EQ(fingerprint->toolchain_fingerprint.size(), 64);
  EXPECT_THAT(fingerprint->canonical_toolchain_manifest,
              HasSubstr("component\tlibdevice\t9\t"));
  EXPECT_EQ(fingerprint->canonical_toolchain_manifest.find(first.root),
            std::string::npos);
}

TEST(MusaBridgeToolchainFingerprintTest, EveryRelevantChangeInvalidates) {
  FixturePaths fixture = MakeFixture();
  absl::StatusOr<MusaBridgeFingerprints> before =
      FingerprintMusaBridgeToolchain(fixture.paths, "mcc-bundle-v1",
                                     kProviderContract);
  ASSERT_TRUE(before.ok()) << before.status();

  Write(fixture.paths.libdevice, "different-libdevice");
  absl::StatusOr<MusaBridgeFingerprints> after = FingerprintMusaBridgeToolchain(
      fixture.paths, "mcc-bundle-v1", kProviderContract);
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_NE(after->toolchain_fingerprint, before->toolchain_fingerprint);
  EXPECT_EQ(after->provider_fingerprint, before->provider_fingerprint);

  Write(fixture.paths.mcc, "different-mcc");
  after = FingerprintMusaBridgeToolchain(fixture.paths, "mcc-bundle-v1",
                                         kProviderContract);
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_NE(after->provider_fingerprint, before->provider_fingerprint);

  after = FingerprintMusaBridgeToolchain(
      fixture.paths, "mcc-bundle-v1",
      absl::StrCat(kProviderContract, "revision=2\n"));
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_NE(after->provider_fingerprint, before->provider_fingerprint);
}

TEST(MusaBridgeToolchainFingerprintTest, RejectsUnsafeIdentityAndInputs) {
  FixturePaths fixture = MakeFixture();
  Write(fixture.paths.toolchain_identity,
        "schema=xla-musa-toolchain-v1\nroot=/absolute/path\n");
  EXPECT_THAT(
      FingerprintMusaBridgeToolchain(fixture.paths, "mcc-bundle-v1",
                                     kProviderContract),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unsafe value")));

  fixture = MakeFixture();
  Write(fixture.paths.libdevice, "");
  EXPECT_THAT(FingerprintMusaBridgeToolchain(fixture.paths, "mcc-bundle-v1",
                                             kProviderContract),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("nonempty regular file")));

  EXPECT_THAT(
      FingerprintMusaBridgeToolchain(fixture.paths, "provider with spaces",
                                     kProviderContract),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not canonical")));
}

TEST(MusaBridgeToolchainFingerprintTest, RejectsSameSizeMutationWhileHashing) {
  FixturePaths fixture = MakeFixture();
  Write(fixture.paths.libclang_cpp, std::string(32 << 20, 'a'));

  std::atomic<bool> started = false;
  std::atomic<bool> stop = false;
  std::atomic<bool> mutation_failed = false;
  std::thread mutator([&] {
    const int fd = open(fixture.paths.libclang_cpp.c_str(), O_WRONLY);
    if (fd < 0) {
      mutation_failed.store(true, std::memory_order_relaxed);
      started.store(true, std::memory_order_release);
      return;
    }
    const std::string blocks[] = {std::string(4096, 'b'),
                                  std::string(4096, 'c')};
    size_t iteration = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      if (pwrite(fd, blocks[iteration++ % 2].data(), blocks[0].size(), 0) !=
          static_cast<ssize_t>(blocks[0].size())) {
        mutation_failed.store(true, std::memory_order_relaxed);
        break;
      }
      started.store(true, std::memory_order_release);
    }
    close(fd);
  });
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
  absl::StatusOr<MusaBridgeFingerprints> result =
      FingerprintMusaBridgeToolchain(fixture.paths, "mcc-bundle-v1",
                                     kProviderContract);
  stop.store(true, std::memory_order_relaxed);
  mutator.join();
  ASSERT_FALSE(mutation_failed.load(std::memory_order_relaxed));
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kAborted, HasSubstr("changed")));
}

}  // namespace
}  // namespace xla::gpu::musa::bridge
