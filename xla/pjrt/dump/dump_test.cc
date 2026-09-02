/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/dump/dump.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace {

class TestTopology : public xla::PjRtTopologyDescription {
 public:
  absl::StatusOr<xla::PjRtTopologyDescriptionProto> ToProto() const override {
    xla::PjRtTopologyDescriptionProto proto;
    proto.set_platform_id(123);
    proto.set_platform_name("test_topology");
    proto.set_platform_version("test_topology_version");
    proto.set_is_subslice_topology(false);
    return proto;
  }

  absl::string_view platform_name() const override { return "test_topology"; }

  absl::string_view platform_version() const override {
    return "test_topology_version";
  }

  xla::PjRtPlatformId platform_id() const override { return 123; }

  bool is_subslice_topology() const override { return false; }

  std::vector<std::unique_ptr<const xla::PjRtDeviceDescription>>
  DeviceDescriptions() const override {
    return {};
  }

  absl::StatusOr<uint64_t> Fingerprint() const override { return 123; }

  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::StatusOr<xla::Layout> GetDefaultLayout(
      xla::PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return xla::Layout();
  }

 private:
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;

TEST(DumpTest, ResolveSpongeDumpPath) {
  EXPECT_THAT(pjrt::ResolveTestingDumpPath("sponge"),
              IsOkAndHolds(testing::Not(testing::IsEmpty())));
  EXPECT_THAT(pjrt::ResolveTestingDumpPath("TEST_UNDECLARED_OUTPUTS_DIR"),
              IsOkAndHolds(testing::Not(testing::IsEmpty())));
  EXPECT_THAT(pjrt::ResolveTestingDumpPath("/tmp/foo"),
              IsOkAndHolds("/tmp/foo"));
  EXPECT_THAT(pjrt::ResolveTestingDumpPath(""), IsOkAndHolds(""));
}

TEST(DumpTest, GetDumpSubdirPath) {
  const std::string temp_dir = tsl::testing::TmpDir();
  TF_ASSERT_OK_AND_ASSIGN(std::string dump_subdir,
                          pjrt::GetDumpSubdirPath(temp_dir, "my_module", 0));
  EXPECT_THAT(dump_subdir, HasSubstr(temp_dir));
  EXPECT_THAT(dump_subdir, HasSubstr("my_module"));
  EXPECT_THAT(tsl::Env::Default()->IsDirectory(dump_subdir), IsOk());
}

TEST(DumpTest, GetDumpSubdirPathEmptyPath) {
  TF_ASSERT_OK_AND_ASSIGN(std::string dump_subdir,
                          pjrt::GetDumpSubdirPath("", "my_module", 0));
  EXPECT_EQ(dump_subdir, "");
}

TEST(DumpTest, DumpCompileInputs) {
  const std::string temp_test_dir = tsl::testing::TmpDir();
  const std::string temp_test_subdir =
      tsl::io::JoinPath(temp_test_dir, "compile_dump_test",
                        absl::StrCat(absl::ToUnixMillis(absl::Now())));
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(temp_test_subdir));
  xla::CompileOptions compile_options;
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      xla::llvm_ir::CreateMlirModuleOp(builder.getUnknownLoc());
  auto topology = std::make_unique<TestTopology>();

  // Dump compile inputs.
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_dump_to(temp_test_subdir);

  TF_ASSERT_OK(pjrt::DumpCompileInputs(temp_test_subdir, compile_options,
                                       *module, *topology.get(), 0));
  std::vector<std::string> files;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(temp_test_subdir, "*"), &files));

  ASSERT_EQ(files.size(), 1);
  std::string dump_subdir = files[0];
  EXPECT_THAT(tsl::Env::Default()->IsDirectory(dump_subdir), IsOk());

  std::vector<std::string> dump_files;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_subdir, "*"), &dump_files));
  EXPECT_EQ(dump_files.size(), 3);
  EXPECT_THAT(dump_files, testing::UnorderedElementsAre(
                              HasSubstr("module.mlir"),
                              HasSubstr("compile_options.textproto"),
                              HasSubstr("topology.textproto")));
}

TEST(DumpTest, DumpCompileInputsAsProto) {
  const std::string temp_test_dir = tsl::testing::TmpDir();
  const std::string temp_test_subdir =
      tsl::io::JoinPath(temp_test_dir, "compile_dump_as_proto_test",
                        absl::StrCat(absl::ToUnixMillis(absl::Now())));
  TF_ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(temp_test_subdir));
  xla::CompileOptions compile_options;
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      xla::llvm_ir::CreateMlirModuleOp(builder.getUnknownLoc());
  auto topology = std::make_unique<TestTopology>();

  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_dump_to(temp_test_subdir);
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_dump_hlo_as_proto(true);

  TF_ASSERT_OK(pjrt::DumpCompileInputs(temp_test_subdir, compile_options,
                                       *module, *topology.get(), 0));
  std::vector<std::string> files;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(temp_test_subdir, "*"), &files));

  ASSERT_EQ(files.size(), 1);
  std::string dump_subdir = files[0];
  EXPECT_THAT(tsl::Env::Default()->IsDirectory(dump_subdir), IsOk());

  std::vector<std::string> dump_files;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_subdir, "*"), &dump_files));
  EXPECT_EQ(dump_files.size(), 3);
  EXPECT_THAT(dump_files,
              testing::UnorderedElementsAre(HasSubstr("module.mlir"),
                                            HasSubstr("compile_options.pb"),
                                            HasSubstr("topology.pb")));

  xla::CompileOptionsProto dumped_options;
  ASSERT_OK(tsl::ReadBinaryProto(
      tsl::Env::Default(), tsl::io::JoinPath(dump_subdir, "compile_options.pb"),
      &dumped_options));
  const xla::DebugOptions& dumped_debug_options =
      dumped_options.executable_build_options().debug_options();
  EXPECT_TRUE(dumped_debug_options.xla_dump_to().empty());
  EXPECT_FALSE(dumped_debug_options.xla_enable_dumping());
  EXPECT_FALSE(dumped_debug_options.xla_dump_hlo_as_proto());
}

TEST(MaybeDumpCompileInputsTest, XlaDumpToNotSet) {
  const std::string temp_test_dir = tsl::testing::TmpDir();
  const std::string temp_test_subdir =
      tsl::io::JoinPath(temp_test_dir, "compile_maybe_dump_test_not_set",
                        absl::StrCat(absl::ToUnixMillis(absl::Now())));
  TF_ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(temp_test_subdir));
  xla::CompileOptions compile_options;
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      xla::llvm_ir::CreateMlirModuleOp(builder.getUnknownLoc());
  auto topology = std::make_unique<TestTopology>();

  // xla_dump_to not set
  TF_ASSERT_OK(pjrt::MaybeDumpCompileInputs(compile_options, *module,
                                            *topology.get(), 0));

  std::vector<std::string> no_files;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(temp_test_subdir, "*"), &no_files));

  ASSERT_EQ(no_files.size(), 0);
}

TEST(MaybeDumpCompileInputsTest, XlaDumpToSet) {
  const std::string temp_test_dir = tsl::testing::TmpDir();
  const std::string temp_test_subdir =
      tsl::io::JoinPath(temp_test_dir, "compile_maybe_dump_test_set",
                        absl::StrCat(absl::ToUnixMillis(absl::Now())));
  TF_ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(temp_test_subdir));
  xla::CompileOptions compile_options;
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      xla::llvm_ir::CreateMlirModuleOp(builder.getUnknownLoc());
  auto topology = std::make_unique<TestTopology>();

  // Set xla_dump_to and dump compile inputs.
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_dump_to(temp_test_subdir);

  TF_ASSERT_OK(pjrt::MaybeDumpCompileInputs(compile_options, *module,
                                            *topology.get(), 0));
  std::vector<std::string> files;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(temp_test_subdir, "*"), &files));

  ASSERT_EQ(files.size(), 1);
  std::string dump_subdir = files[0];
  EXPECT_THAT(tsl::Env::Default()->IsDirectory(dump_subdir), IsOk());

  std::vector<std::string> dump_files;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_subdir, "*"), &dump_files));
  EXPECT_EQ(dump_files.size(), 3);
  EXPECT_THAT(dump_files, testing::UnorderedElementsAre(
                              HasSubstr("module.mlir"),
                              HasSubstr("compile_options.textproto"),
                              HasSubstr("topology.textproto")));
}

TEST(MaybeDumpCompileInputsTest, XlaDumpToOptionOverrideSet) {
  const std::string temp_test_dir = tsl::testing::TmpDir();
  const std::string temp_test_subdir =
      tsl::io::JoinPath(temp_test_dir, "compile_maybe_dump_test_override_set",
                        absl::StrCat(absl::ToUnixMillis(absl::Now())));
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(temp_test_subdir));
  xla::CompileOptions compile_options;
  compile_options.env_option_overrides.emplace_back("xla_dump_to",
                                                    temp_test_subdir);
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      xla::llvm_ir::CreateMlirModuleOp(builder.getUnknownLoc());
  auto topology = std::make_unique<TestTopology>();

  ASSERT_OK(
      pjrt::MaybeDumpCompileInputs(compile_options, *module, *topology, 0));
  std::vector<std::string> files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(temp_test_subdir, "*"), &files));

  ASSERT_EQ(files.size(), 1);
  std::vector<std::string> dump_files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(files[0], "*"), &dump_files));
  EXPECT_THAT(dump_files, testing::UnorderedElementsAre(
                              HasSubstr("module.mlir"),
                              HasSubstr("compile_options.textproto"),
                              HasSubstr("topology.textproto")));

  xla::CompileOptionsProto dumped_options;
  ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(),
      tsl::io::JoinPath(files[0], "compile_options.textproto"),
      &dumped_options));
  EXPECT_EQ(dumped_options.env_option_overrides().find("xla_dump_to"),
            dumped_options.env_option_overrides().end());
  EXPECT_TRUE(dumped_options.executable_build_options()
                  .debug_options()
                  .xla_dump_to()
                  .empty());
}

class MaybeDumpCompileInputsModuleFilterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string temp_test_dir = tsl::testing::TmpDir();
    dump_subdir_ = tsl::io::JoinPath(
        temp_test_dir, absl::StrCat("compile_maybe_dump_filter_",
                                    absl::ToUnixMillis(absl::Now())));
    TF_ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(dump_subdir_));
    module_ = xla::llvm_ir::CreateMlirModuleOp(builder_.getUnknownLoc());
    module_->setName("foo");
    topology_ = std::make_unique<TestTopology>();
    debug_options_ =
        compile_options_.executable_build_options.mutable_debug_options();
    debug_options_->set_xla_dump_to(dump_subdir_);
  }

  bool Run(absl::string_view module_re) {
    debug_options_->set_xla_dump_hlo_module_re(module_re);
    EXPECT_OK(pjrt::MaybeDumpCompileInputs(compile_options_, *module_,
                                           *topology_, 0));
    std::vector<std::string> files;
    EXPECT_OK(tsl::Env::Default()->GetMatchingPaths(
        tsl::io::JoinPath(dump_subdir_, "*"), &files));
    return !files.empty();
  }

  std::string dump_subdir_;
  xla::CompileOptions compile_options_;
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_{&context_};
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<TestTopology> topology_;
  xla::DebugOptions* debug_options_;
};

TEST_F(MaybeDumpCompileInputsModuleFilterTest,
       DisablesDumpWhenFilterDoesNotMatch) {
  EXPECT_FALSE(Run("bar"));
}

TEST_F(MaybeDumpCompileInputsModuleFilterTest, EnablesDumpWhenFilterMatches) {
  EXPECT_TRUE(Run("foo"));
}

}  // namespace
