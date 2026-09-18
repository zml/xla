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
#include <cstdlib>
#include <optional>
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
#include <unistd.h>
#include "xla/service/gpu/musa/protocol.h"
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

constexpr std::array<const char*, 11> kComponents = {
    "libpjrt_musa.so",
    "musa-compiler/musa-llvm-bridge",
    "musa-sdk/toolchain-identity.txt",
    "musa-sdk/lib/libclang-cpp.so.14",
    "musa-sdk/bin/clang-14",
    "musa-sdk/bin/clang-offload-bundler",
    "musa-sdk/bin/lld",
    "musa-sdk/bin/llvm-readobj",
    "musa-sdk/mtgpu/bitcode/libdevice.bc",
    "musa-sdk/include/llvm/IR/IntrinsicsMUSA.td",
    "musa-sdk/include/clang/Basic/BuiltinsMTGPU.def",
};
constexpr std::array<const char*, 3> kEnvironment = {
    "MUSA_PATH", "XLA_MUSA_COMPILATION_TEMP_ROOT",
    "XLA_MUSA_COMPILATION_CACHE_DIR"};

std::string QualifiedIdentity() {
  return "schema=xla-musa-toolchain-v1\n"
         "musa_version=5.1.0\n"
         "musa_version_number=50100\n"
         "musa_device=S4000\n"
         "musa_gpu_architectures=mp_22\n"
         "distro_sha256="
         "5407266eab8fe42caee83f6a7a979edeaa9ea6e542e197bf65fb5f394f1980b2\n";
}

std::string IdentityText(const MusaCompilationIdentity& identity) {
  return absl::StrCat(
      identity.xla_revision, "\n", identity.current_llvm_revision, "\n",
      identity.provider_name, "\n", identity.provider_fingerprint, "\n",
      identity.bridge_fingerprint, "\n", identity.toolchain_fingerprint, "\n",
      identity.libdevice_fingerprint, "\n", identity.driver_compatibility, "\n",
      identity.runtime_compatibility);
}

class MusaCompilerLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (size_t i = 0; i < kEnvironment.size(); ++i) {
      if (const char* value = std::getenv(kEnvironment[i]); value != nullptr) {
        saved_environment_[i] = value;
      }
      ASSERT_EQ(unsetenv(kEnvironment[i]), 0);
    }
    absl::StatusOr<tsl::testing::TemporaryDirectory> root =
        tsl::testing::TemporaryDirectory::CreateForCurrentTestcase();
    ASSERT_THAT(root, IsOk());
    root_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(*std::move(root));
    directory_ = absl::StrCat(root_->path(), "/bundle");
    Populate(directory_);
  }

  void TearDown() override {
    for (size_t i = 0; i < kEnvironment.size(); ++i) {
      if (saved_environment_[i].has_value()) {
        EXPECT_EQ(setenv(kEnvironment[i], saved_environment_[i]->c_str(), 1),
                  0);
      } else {
        EXPECT_EQ(unsetenv(kEnvironment[i]), 0);
      }
    }
  }

  void Write(const std::string& path, const std::string& contents) {
    ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(
                    path.substr(0, path.rfind('/'))),
                IsOk());
    ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), path, contents),
                IsOk());
  }

  void Populate(const std::string& directory) {
    for (size_t i = 0; i < kComponents.size(); ++i) {
      const std::string path = absl::StrCat(directory, "/", kComponents[i]);
      Write(path, i == 2 ? QualifiedIdentity() : kComponents[i]);
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
  }

  std::string Path(size_t component) const {
    return absl::StrCat(directory_, "/", kComponents[component]);
  }

  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> Load() {
    return LoadMusaCompilationProviderFromLayout(Path(0));
  }

  std::array<std::optional<std::string>, kEnvironment.size()>
      saved_environment_;
  std::unique_ptr<tsl::testing::TemporaryDirectory> root_;
  std::string directory_;
};

TEST_F(MusaCompilerLayoutTest, LoadsFixedLayoutWithoutManifest) {
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> provider = Load();
  ASSERT_THAT(provider, IsOk());
  const MusaCompilationIdentity& identity = (*provider)->identity();
  EXPECT_EQ(identity.xla_revision, MusaBridgeSha256Hex(kComponents[0]));
  EXPECT_EQ(identity.current_llvm_revision,
            absl::StrCat("linked-", identity.xla_revision));
  EXPECT_EQ(identity.bridge_fingerprint, MusaBridgeSha256Hex(kComponents[1]));
  EXPECT_EQ(identity.libdevice_fingerprint,
            MusaBridgeSha256Hex(kComponents[8]));
  EXPECT_EQ(identity.provider_name, "mcc-bundle-v1");
  EXPECT_EQ(identity.runtime_compatibility, "musa-runtime-5.1.0-compatible");
  EXPECT_EQ(identity.driver_compatibility, "musa-driver-5.1-compatible");
  EXPECT_TRUE((*provider)->capabilities().vendor_llvm_isolated);
  EXPECT_EQ((*provider)->capabilities().binary_kind, "mubin");
}

TEST_F(MusaCompilerLayoutTest, RelocationIncludingSpacesPreservesIdentity) {
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> first = Load();
  ASSERT_THAT(first, IsOk());
  const std::string relocated =
      absl::StrCat(root_->path(), "/relocated bundle");
  Populate(relocated);
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> second =
      LoadMusaCompilationProviderFromLayout(
          absl::StrCat(relocated, "/libpjrt_musa.so"));
  ASSERT_THAT(second, IsOk());
  EXPECT_EQ(IdentityText((*first)->identity()),
            IdentityText((*second)->identity()));
}

TEST_F(MusaCompilerLayoutTest, EveryPackagedComponentParticipatesInIdentity) {
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> before = Load();
  ASSERT_THAT(before, IsOk());
  const MusaCompilationIdentity original = (*before)->identity();
  for (size_t i = 0; i < kComponents.size(); ++i) {
    SCOPED_TRACE(kComponents[i]);
    const std::string contents = i == 2 ? QualifiedIdentity() : kComponents[i];
    Write(Path(i),
          absl::StrCat(contents, i == 2 ? "revision=changed\n" : "-changed"));
    absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> after = Load();
    ASSERT_THAT(after, IsOk());
    const MusaCompilationIdentity& changed = (*after)->identity();
    EXPECT_NE(IdentityText(original), IdentityText(changed));
    if (i == 0) {
      EXPECT_NE(original.xla_revision, changed.xla_revision);
      EXPECT_NE(original.current_llvm_revision, changed.current_llvm_revision);
    } else if (i == 1) {
      EXPECT_NE(original.bridge_fingerprint, changed.bridge_fingerprint);
    } else {
      EXPECT_NE(original.toolchain_fingerprint, changed.toolchain_fingerprint);
    }
    if (i == 4 || i == 5) {
      EXPECT_NE(original.provider_fingerprint, changed.provider_fingerprint);
    }
    if (i == 8) {
      EXPECT_NE(original.libdevice_fingerprint, changed.libdevice_fingerprint);
    }
    Write(Path(i), contents);
  }
}

TEST_F(MusaCompilerLayoutTest, RejectsUnqualifiedSdkIdentity) {
  const std::pair<const char*, const char*> replacements[] = {
      {"5.1.0", "5.2.0"},
      {"50100", "50200"},
      {"S4000", "S80"},
      {"mp_22", "mp_21"},
      {"5407266e", "00000000"}};
  for (const auto& [from, to] : replacements) {
    SCOPED_TRACE(from);
    Write(Path(2), absl::StrReplaceAll(QualifiedIdentity(), {{from, to}}));
    EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                 HasSubstr("qualified SDK 5.1.0 S4000/mp_22")));
  }
}

TEST_F(MusaCompilerLayoutTest, AcceptsPublishedSdkIdentity) {
  const std::string published = absl::StrReplaceAll(
      QualifiedIdentity(),
      {{"5407266eab8fe42caee83f6a7a979edeaa9ea6e542e197bf65fb5f394f1980b2",
        "afa05b1e73c4816e063fb695c889e37877599aa021a4ef8dba08998c1f3b1f9f"}});
  Write(Path(2), published);
  EXPECT_THAT(Load(), IsOk());
}

TEST_F(MusaCompilerLayoutTest, RejectsMalformedOrDuplicateSdkIdentity) {
  Write(Path(2), absl::StrCat(QualifiedIdentity(), "musa_device=S80\n"));
  EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("duplicate")));
  Write(Path(2), absl::StrReplaceAll(QualifiedIdentity(),
                                     {{"xla-musa-toolchain-v1", "unknown"}}));
  EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("schema")));
}

TEST_F(MusaCompilerLayoutTest, RejectsMissingOrEmptyComponents) {
  for (size_t i = 0; i < kComponents.size(); ++i) {
    SCOPED_TRACE(kComponents[i]);
    ASSERT_EQ(unlink(Path(i).c_str()), 0);
    EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kNotFound));
    Write(Path(i), "");
    ASSERT_EQ(chmod(Path(i).c_str(), 0700), 0);
    EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kInvalidArgument,
                                 HasSubstr("nonempty regular file")));
    Write(Path(i), i == 2 ? QualifiedIdentity() : kComponents[i]);
  }
}

TEST_F(MusaCompilerLayoutTest, RejectsNonExecutableTools) {
  for (size_t i : {1, 4, 5, 6, 7}) {
    SCOPED_TRACE(kComponents[i]);
    ASSERT_EQ(chmod(Path(i).c_str(), 0600), 0);
    EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kPermissionDenied,
                                 HasSubstr("executable")));
    ASSERT_EQ(chmod(Path(i).c_str(), 0700), 0);
  }
}

TEST_F(MusaCompilerLayoutTest, RejectsComponentSymlinkOutsideSelectedRoot) {
  const std::string outside = absl::StrCat(root_->path(), "/outside-tool");
  Write(outside, "foreign");
  ASSERT_EQ(chmod(outside.c_str(), 0700), 0);
  ASSERT_EQ(unlink(Path(4).c_str()), 0);
  ASSERT_EQ(symlink(outside.c_str(), Path(4).c_str()), 0);
  EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("escapes its bundle")));
}

TEST_F(MusaCompilerLayoutTest, MusaPathSelectsToolkitWithoutManifest) {
  const std::string other = absl::StrCat(root_->path(), "/other");
  Populate(other);
  Write(absl::StrCat(other, "/", kComponents[4]), "other compiler");
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> first = Load();
  ASSERT_THAT(first, IsOk());
  ASSERT_EQ(setenv("MUSA_PATH", absl::StrCat(other, "/musa-sdk").c_str(), 1),
            0);
  absl::StatusOr<std::unique_ptr<MusaCompilationProvider>> second = Load();
  ASSERT_THAT(second, IsOk());
  EXPECT_NE((*first)->identity().provider_fingerprint,
            (*second)->identity().provider_fingerprint);
}

TEST_F(MusaCompilerLayoutTest, InvalidMusaPathDoesNotFallBack) {
  ASSERT_EQ(setenv("MUSA_PATH", "/missing-musa-sdk", 1), 0);
  EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kNotFound));
  ASSERT_EQ(setenv("MUSA_PATH", "relative-sdk", 1), 0);
  EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("absolute safe path")));
  ASSERT_EQ(setenv("MUSA_PATH", Path(0).c_str(), 1), 0);
  EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("filesystem type")));
}

TEST_F(MusaCompilerLayoutTest, ExplicitToolkitTakesPrecedenceOverEnvironment) {
  ASSERT_EQ(setenv("MUSA_PATH", "/missing-musa-sdk", 1), 0);
  EXPECT_THAT(LoadMusaCompilationProviderFromLayout(
                  Path(0), absl::StrCat(directory_, "/musa-sdk")),
              IsOk());
}

TEST_F(MusaCompilerLayoutTest, RejectsUnsafeExplicitPaths) {
  EXPECT_THAT(LoadMusaCompilationProviderFromLayout("relative.so"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      LoadMusaCompilationProviderFromLayout(
          std::string(Path(0)).append("\0ignored", 8)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("control byte")));
}

TEST_F(MusaCompilerLayoutTest, ValidatesTemporaryAndCacheDirectories) {
  const std::string cache = absl::StrCat(root_->path(), "/cache");
  ASSERT_THAT(tsl::Env::Default()->CreateDir(cache), IsOk());
  ASSERT_EQ(setenv("XLA_MUSA_COMPILATION_CACHE_DIR", cache.c_str(), 1), 0);
  EXPECT_THAT(Load(), IsOk());
  ASSERT_EQ(setenv("XLA_MUSA_COMPILATION_CACHE_DIR", Path(0).c_str(), 1), 0);
  EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_EQ(unsetenv("XLA_MUSA_COMPILATION_CACHE_DIR"), 0);
  ASSERT_EQ(setenv("XLA_MUSA_COMPILATION_TEMP_ROOT", "/missing-musa-temp", 1),
            0);
  EXPECT_THAT(Load(), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xla::gpu::musa
