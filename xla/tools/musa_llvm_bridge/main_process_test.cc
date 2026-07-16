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

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "xla/service/gpu/musa/mcc_bundle_codegen.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"
#include "xla/tools/musa_llvm_bridge/toolchain_fingerprint.h"
#include "xla/tsl/platform/resource_loader.h"

namespace xla::gpu::musa::bridge {
namespace {

using ::absl_testing::IsOk;
using ::testing::HasSubstr;

struct ProcessResult {
  int exit_code;
  std::string stdout_text;
  std::string stderr_text;
};

bool WriteAll(int fd, absl::string_view value) {
  while (!value.empty()) {
    const ssize_t count = write(fd, value.data(), value.size());
    if (count > 0) {
      value.remove_prefix(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

std::string ReadAll(int fd) {
  std::string output;
  char buffer[4096];
  while (true) {
    const ssize_t count = read(fd, buffer, sizeof(buffer));
    if (count > 0) {
      output.append(buffer, count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    break;
  }
  close(fd);
  return output;
}

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

std::string ReadTestdata(const std::string& name) {
  std::ifstream input(tsl::GetDataDependencyFilepath(absl::StrCat(
                          "xla/tools/musa_llvm_bridge/testdata/", name)),
                      std::ios::binary);
  EXPECT_TRUE(input.is_open()) << name;
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
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

std::vector<std::string> Arguments(const MusaBridgeToolchainPaths& paths) {
  return {
      paths.bridge_executable,
      absl::StrCat("--bridge-executable=", paths.bridge_executable),
      absl::StrCat("--toolchain-identity=", paths.toolchain_identity),
      absl::StrCat("--libclang-cpp=", paths.libclang_cpp),
      absl::StrCat("--mcc=", paths.mcc),
      absl::StrCat("--clang-offload-bundler=", paths.clang_offload_bundler),
      absl::StrCat("--lld=", paths.lld),
      absl::StrCat("--llvm-readobj=", paths.llvm_readobj),
      absl::StrCat("--libdevice=", paths.libdevice),
      absl::StrCat("--intrinsics-musa-td=", paths.intrinsics_musa_td),
      absl::StrCat("--builtins-mtgpu-def=", paths.builtins_mtgpu_def),
      absl::StrCat("--temp-root=", CanonicalPath(::testing::TempDir())),
  };
}

MusaBridgeCompileRequest Request(const MusaBridgeToolchainPaths& paths,
                                 const std::string& llvm_ir) {
  absl::StatusOr<MusaBridgeFingerprints> fingerprints =
      FingerprintMusaBridgeToolchain(
          paths, std::string(MccBundleProviderName()),
          std::string(MccBundleProviderCanonicalText()));
  EXPECT_THAT(fingerprints, IsOk());
  MusaBridgeCompileRequest request;
  if (!fingerprints.ok()) return request;
  request.set_protocol_version(kMusaBridgeProtocolVersion);
  request.set_shim_abi_version(kMusaShimAbiVersion);
  request.set_mapping_version(kMusaShimMappingVersion);
  request.set_mapping_fingerprint(kMusaShimMappingSha256);
  request.set_module_name("main_process_test");
  request.set_normalized_llvm(llvm_ir);
  request.set_normalized_llvm_bytes(llvm_ir.size());
  request.set_normalized_llvm_sha256(MusaBridgeSha256Hex(llvm_ir));
  request.add_kernel_entry_names("kernel");
  request.add_exported_symbol_names("kernel");
  request.set_target_triple(kMusaTargetTriple);
  request.set_architecture(kMusaTargetArchitecture);
  request.set_data_layout(kMusaDataLayout);
  request.set_pointer_model(MUSA_BRIDGE_POINTER_MODEL_OPAQUE);
  request.set_pointer_width_bits(kMusaInterchangePointerWidth);
  request.set_byte_order(MUSA_BRIDGE_BYTE_ORDER_LITTLE_ENDIAN);
  request.mutable_numerical_flags();
  request.set_optimization_level(2);
  request.set_deterministic(true);
  request.set_xla_revision("main-process-test");
  request.set_current_llvm_revision("current-llvm-test");
  request.set_provider_name(fingerprints->provider_name);
  request.set_provider_fingerprint(fingerprints->provider_fingerprint);
  request.set_bridge_fingerprint(fingerprints->bridge_fingerprint);
  request.set_toolchain_fingerprint(fingerprints->toolchain_fingerprint);
  return request;
}

ProcessResult RunProcess(const std::vector<std::string>& arguments,
                         absl::string_view input, int input_file = -1) {
  int stdin_pipe[2] = {-1, -1};
  int stdout_pipe[2];
  int stderr_pipe[2];
  if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0 ||
      (input_file < 0 && pipe(stdin_pipe) != 0)) {
    ADD_FAILURE() << "failed to create process-test pipes";
    return {-1, {}, {}};
  }

  const pid_t child = fork();
  if (child < 0) {
    ADD_FAILURE() << "failed to fork bridge process";
    return {-1, {}, {}};
  }
  if (child == 0) {
    const int child_input = input_file >= 0 ? input_file : stdin_pipe[0];
    if (dup2(child_input, STDIN_FILENO) < 0 ||
        dup2(stdout_pipe[1], STDOUT_FILENO) < 0 ||
        dup2(stderr_pipe[1], STDERR_FILENO) < 0) {
      _exit(125);
    }
    if (stdin_pipe[0] >= 0) close(stdin_pipe[0]);
    if (stdin_pipe[1] >= 0) close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    if (input_file >= 0) close(input_file);
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) {
      argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    execv(argv[0], argv.data());
    _exit(126);
  }

  close(stdout_pipe[1]);
  close(stderr_pipe[1]);
  if (input_file < 0) {
    close(stdin_pipe[0]);
    EXPECT_TRUE(WriteAll(stdin_pipe[1], input));
    close(stdin_pipe[1]);
  } else {
    close(input_file);
  }
  ProcessResult result;
  result.stdout_text = ReadAll(stdout_pipe[0]);
  result.stderr_text = ReadAll(stderr_pipe[0]);
  int status = 0;
  EXPECT_EQ(waitpid(child, &status, 0), child);
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return result;
}

std::string Wire(const MusaBridgeCompileRequest& request) {
  absl::StatusOr<std::string> wire = EncodeMusaBridgeCompileRequest(request);
  EXPECT_THAT(wire, IsOk());
  return wire.ok() ? *wire : std::string();
}

MusaBridgeCompileResponse Response(const ProcessResult& process) {
  EXPECT_EQ(process.exit_code, 0);
  EXPECT_TRUE(process.stderr_text.empty()) << process.stderr_text;
  absl::StatusOr<MusaBridgeCompileResponse> response =
      DecodeMusaBridgeCompileResponse(process.stdout_text);
  EXPECT_THAT(response, IsOk());
  return response.ok() ? *response : MusaBridgeCompileResponse();
}

TEST(MusaLlvmBridgeMainProcessTest,
     CanonicalRequestCompilesWithPinnedSdkOnStdoutOnly) {
  const MusaBridgeToolchainPaths paths = ToolchainPaths();
  MusaBridgeCompileRequest request = Request(paths, ReadTestdata("minimal.ll"));
  ProcessResult process = RunProcess(Arguments(paths), Wire(request));
  MusaBridgeCompileResponse response = Response(process);
  EXPECT_EQ(response.status(), MUSA_BRIDGE_STATUS_OK);
  EXPECT_FALSE(response.mubin().empty());
  EXPECT_EQ(response.mubin_sha256(), MusaBridgeSha256Hex(response.mubin()));
  EXPECT_THAT(ValidateMusaBridgeExchange(request, response), IsOk());
}

TEST(MusaLlvmBridgeMainProcessTest,
     UnsupportedSemanticOptionsAreRejectedBeforeProvider) {
  const MusaBridgeToolchainPaths paths = ToolchainPaths();
  const MusaBridgeCompileRequest base =
      Request(paths, ReadTestdata("minimal.ll"));
  std::vector<MusaBridgeCompileRequest> cases;
  cases.push_back(base);
  cases.back().set_optimization_level(1);
  cases.push_back(base);
  cases.back().set_emit_debug_information(true);
  cases.push_back(base);
  cases.back().set_deterministic(false);
  cases.push_back(base);
  cases.back().mutable_numerical_flags()->set_fast_math(true);
  cases.push_back(base);
  cases.back().mutable_numerical_flags()->set_flush_denormals_to_zero(true);
  cases.push_back(base);
  cases.back().mutable_numerical_flags()->set_finite_math_only(true);
  cases.push_back(base);
  cases.back().mutable_numerical_flags()->set_unsafe_math_optimizations(true);
  cases.push_back(base);
  cases.back().mutable_numerical_flags()->set_no_signed_zeros(true);
  cases.push_back(base);
  cases.back().mutable_numerical_flags()->set_allow_fp_contract(true);

  for (const MusaBridgeCompileRequest& request : cases) {
    MusaBridgeCompileResponse response =
        Response(RunProcess(Arguments(paths), Wire(request)));
    EXPECT_EQ(response.status(), MUSA_BRIDGE_STATUS_REJECTED);
    ASSERT_EQ(response.diagnostics_size(), 1);
    EXPECT_EQ(response.diagnostics(0).code(), "unsupported-options");
    EXPECT_THAT(ValidateMusaBridgeExchange(request, response), IsOk());
  }
}

TEST(MusaLlvmBridgeMainProcessTest, ActualFingerprintMismatchIsRejected) {
  const MusaBridgeToolchainPaths paths = ToolchainPaths();
  MusaBridgeCompileRequest request = Request(paths, ReadTestdata("minimal.ll"));
  request.set_toolchain_fingerprint(std::string(64, '0'));
  MusaBridgeCompileResponse response =
      Response(RunProcess(Arguments(paths), Wire(request)));
  EXPECT_EQ(response.status(), MUSA_BRIDGE_STATUS_REJECTED);
  ASSERT_EQ(response.diagnostics_size(), 1);
  EXPECT_EQ(response.diagnostics(0).code(), "identity-mismatch");
  EXPECT_NE(response.toolchain_fingerprint(), request.toolchain_fingerprint());
}

TEST(MusaLlvmBridgeMainProcessTest,
     DeclaredVendorLlvmMustMatchTheLoadedSharedObject) {
  const MusaBridgeToolchainPaths paths = ToolchainPaths();
  MusaBridgeCompileRequest request = Request(paths, ReadTestdata("minimal.ll"));
  std::vector<std::string> arguments = Arguments(paths);
  for (std::string& argument : arguments) {
    if (absl::StartsWith(argument, "--libclang-cpp=")) {
      argument = absl::StrCat("--libclang-cpp=", CanonicalPath("/bin/false"));
    }
  }
  MusaBridgeCompileResponse response =
      Response(RunProcess(arguments, Wire(request)));
  EXPECT_EQ(response.status(), MUSA_BRIDGE_STATUS_INTERNAL_ERROR);
  ASSERT_EQ(response.diagnostics_size(), 1);
  EXPECT_EQ(response.diagnostics(0).code(), "startup-config");
  EXPECT_THAT(ValidateMusaBridgeExchange(request, response), IsOk());
}

TEST(MusaLlvmBridgeMainProcessTest,
     MixedSdkComponentsAreRejectedBeforeFingerprinting) {
  const MusaBridgeToolchainPaths paths = ToolchainPaths();
  MusaBridgeCompileRequest request = Request(paths, ReadTestdata("minimal.ll"));
  std::vector<std::string> arguments = Arguments(paths);
  for (std::string& argument : arguments) {
    if (absl::StartsWith(argument, "--lld=")) {
      argument = absl::StrCat("--lld=", CanonicalPath("/bin/false"));
    }
  }
  MusaBridgeCompileResponse response =
      Response(RunProcess(arguments, Wire(request)));
  EXPECT_EQ(response.status(), MUSA_BRIDGE_STATUS_INTERNAL_ERROR);
  ASSERT_EQ(response.diagnostics_size(), 1);
  EXPECT_EQ(response.diagnostics(0).code(), "startup-config");
  EXPECT_THAT(response.diagnostics(0).message(), HasSubstr("pinned MCC SDK"));
  EXPECT_THAT(ValidateMusaBridgeExchange(request, response), IsOk());
}

TEST(MusaLlvmBridgeMainProcessTest,
     MissingDuplicateAndUnknownFlagsBecomeCanonicalResponses) {
  const MusaBridgeToolchainPaths paths = ToolchainPaths();
  MusaBridgeCompileRequest request = Request(paths, ReadTestdata("minimal.ll"));
  const std::string wire = Wire(request);
  std::vector<std::vector<std::string>> cases;
  std::vector<std::string> missing = Arguments(paths);
  missing.pop_back();
  cases.push_back(std::move(missing));
  std::vector<std::string> duplicate = Arguments(paths);
  duplicate.push_back(duplicate.back());
  cases.push_back(std::move(duplicate));
  std::vector<std::string> unknown = Arguments(paths);
  unknown.push_back("--unknown=/tmp");
  cases.push_back(std::move(unknown));
  for (const std::vector<std::string>& arguments : cases) {
    MusaBridgeCompileResponse response = Response(RunProcess(arguments, wire));
    EXPECT_EQ(response.status(), MUSA_BRIDGE_STATUS_INTERNAL_ERROR);
    ASSERT_EQ(response.diagnostics_size(), 1);
    EXPECT_EQ(response.diagnostics(0).code(), "startup-config");
    EXPECT_THAT(ValidateMusaBridgeExchange(request, response), IsOk());
  }
}

TEST(MusaLlvmBridgeMainProcessTest,
     MalformedAndNoncanonicalWireAreTransportFailuresWithoutStdout) {
  const MusaBridgeToolchainPaths paths = ToolchainPaths();
  MusaBridgeCompileRequest request = Request(paths, ReadTestdata("minimal.ll"));
  for (const std::string& wire : {std::string("sensitive malformed wire\n"),
                                  absl::StrCat(Wire(request), "\n")}) {
    ProcessResult process = RunProcess(Arguments(paths), wire);
    EXPECT_EQ(process.exit_code, 2);
    EXPECT_TRUE(process.stdout_text.empty());
    EXPECT_THAT(process.stderr_text, HasSubstr("musa-llvm-bridge:"));
    EXPECT_FALSE(absl::StrContains(process.stderr_text, "sensitive"));
  }
}

TEST(MusaLlvmBridgeMainProcessTest,
     OversizedRegularStdinIsRejectedWithoutReadingSparsePayload) {
  const MusaBridgeToolchainPaths paths = ToolchainPaths();
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/oversized-bridge-request");
  const int fd =
      open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, kMusaBridgeMaxRequestWireBytes + 1), 0);
  ASSERT_EQ(lseek(fd, 0, SEEK_SET), 0);
  ProcessResult process = RunProcess(Arguments(paths), /*input=*/{}, fd);
  EXPECT_EQ(process.exit_code, 2);
  EXPECT_TRUE(process.stdout_text.empty());
  EXPECT_THAT(process.stderr_text, HasSubstr("wire limit"));
}

}  // namespace
}  // namespace xla::gpu::musa::bridge
