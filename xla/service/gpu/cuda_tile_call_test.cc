#include "xla/service/gpu/cuda_tile_call.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

// A complete config, optionally with one attribute dropped and extra ones
// appended.
std::string Config(absl::string_view without = "",
                   absl::string_view extra = "") {
  std::vector<std::string> fields;
  auto add = [&](absl::string_view key, absl::string_view value) {
    if (key == without) return;
    fields.push_back(absl::StrCat(key, " = ", value));
  };
  add("name", "\"k\"");
  add("kernel_type", "\"cuda_tile\"");
  add("ir", "\"cuda_tile.module @m {}\"");
  add("grid_x", "1");
  add("grid_y", "2 : i32");
  add("grid_z", "3 : i64");
  if (!extra.empty()) fields.push_back(std::string(extra));
  return absl::StrCat("{", absl::StrJoin(fields, ", "), "}");
}

class CudaTileCallTest : public ::testing::Test {
 protected:
  absl::StatusOr<CudaTileCall> Parse(absl::string_view config) {
    return CudaTileCall::Parse(config, &ctx_);
  }
  absl::Status ParseStatus(absl::string_view config) {
    return Parse(config).status();
  }

  mlir::MLIRContext ctx_;
};

TEST_F(CudaTileCallTest, ParsesTheMinimalConfig) {
  TF_ASSERT_OK_AND_ASSIGN(CudaTileCall call, Parse(Config()));
  EXPECT_EQ(call.name, "k");
  EXPECT_EQ(call.ir, "cuda_tile.module @m {}");
  EXPECT_EQ(call.grid_x, 1);
  EXPECT_EQ(call.grid_y, 2);
  EXPECT_EQ(call.grid_z, 3);
  EXPECT_EQ(call.bytecode_major, 13);
  EXPECT_EQ(call.bytecode_minor, 3);
  EXPECT_TRUE(call.output_indices.empty());
}

TEST_F(CudaTileCallTest, EveryRequiredKeyIsNamedWhenMissing) {
  for (absl::string_view key :
       {"name", "kernel_type", "ir", "grid_x", "grid_y", "grid_z"}) {
    absl::Status status = ParseStatus(Config(key));
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << key;
    EXPECT_THAT(status.message(), HasSubstr(absl::StrCat("'", key, "'")));
  }
}

TEST_F(CudaTileCallTest, NonDictionaryConfigIsAnErrorNotACrash) {
  for (absl::string_view config : {"42 : i32", "{", "", "\"str\""}) {
    EXPECT_EQ(ParseStatus(config).code(), absl::StatusCode::kInvalidArgument)
        << config;
  }
}

TEST_F(CudaTileCallTest, RejectsLaunchShapeKeys) {
  for (absl::string_view key :
       {"block_x", "block_y", "block_z", "shared_mem_bytes", "num_warps",
        "num_stages", "num_ctas", "kernel_data", "zeroed_args"}) {
    absl::Status status =
        ParseStatus(Config("", absl::StrCat(key, " = 1")));
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << key;
    EXPECT_THAT(status.message(), HasSubstr(absl::StrCat("'", key, "'")));
  }
}

TEST_F(CudaTileCallTest, RejectsOtherKernelTypes) {
  for (absl::string_view type : {"ptx", "cubin", "tilebc", "xtile"}) {
    absl::Status status = ParseStatus(
        Config("kernel_type", absl::StrCat("kernel_type = \"", type, "\"")));
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << type;
    EXPECT_THAT(status.message(), HasSubstr(type));
  }
}

TEST_F(CudaTileCallTest, GridAxesAreRangeChecked) {
  EXPECT_THAT(ParseStatus(Config("grid_x", "grid_x = 0")).message(),
              HasSubstr("'grid_x' is 0"));
  EXPECT_THAT(ParseStatus(Config("grid_z", "grid_z = 16777216")).message(),
              HasSubstr("'grid_z' is 16777216"));
  TF_ASSERT_OK_AND_ASSIGN(CudaTileCall call,
                          Parse(Config("grid_y", "grid_y = 16777215")));
  EXPECT_EQ(call.grid_y, 16777215);
}

TEST_F(CudaTileCallTest, ParsesIrVersion) {
  TF_ASSERT_OK_AND_ASSIGN(CudaTileCall call,
                          Parse(Config("", "ir_version = \"13.1\"")));
  EXPECT_EQ(call.bytecode_major, 13);
  EXPECT_EQ(call.bytecode_minor, 1);
  for (absl::string_view bad : {"\"13\"", "\"a.b\"", "\"13.1.0\"", "13"}) {
    EXPECT_EQ(
        ParseStatus(Config("", absl::StrCat("ir_version = ", bad))).code(),
        absl::StatusCode::kInvalidArgument)
        << bad;
  }
}

TEST_F(CudaTileCallTest, ParsesOutputIndices) {
  TF_ASSERT_OK_AND_ASSIGN(CudaTileCall call,
                          Parse(Config("", "output_indices = [1, 3]")));
  EXPECT_THAT(call.output_indices, ElementsAre(1, 3));
  for (absl::string_view bad :
       {"output_indices = [\"a\"]", "output_indices = 0",
        "output_indices = dense<[0]> : tensor<1xi32>",
        "output_indices = [4294967296]", "output_indices = [-1]"}) {
    EXPECT_EQ(ParseStatus(Config("", bad)).code(),
              absl::StatusCode::kInvalidArgument)
        << bad;
  }
}

TEST_F(CudaTileCallTest, WideIntegerAttributeIsAnError) {
  absl::Status status = ParseStatus(
      Config("grid_x", "grid_x = 18446744073709551617 : i128"));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("'grid_x'"));
}

TEST_F(CudaTileCallTest, ParseErrorCarriesTheDiagnostic) {
  absl::Status status = ParseStatus("{name = ");
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("expected"));
}

TEST_F(CudaTileCallTest, FingerprintCarriesTheArgumentCount) {
  EXPECT_NE(CudaTileKernelFingerprint(Config(), 2),
            CudaTileKernelFingerprint(Config(), 3));
  EXPECT_EQ(CudaTileKernelFingerprint(Config(), 2),
            CudaTileKernelFingerprint(Config(), 2));
}

TEST_F(CudaTileCallTest, IrEscapesRoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(
      CudaTileCall call,
      Parse(Config("ir", R"(ir = "a\"b\\c\nd")")));
  EXPECT_EQ(call.ir, "a\"b\\c\nd");
}

TEST_F(CudaTileCallTest, ParsesZeroedOutputs) {
  TF_ASSERT_OK_AND_ASSIGN(CudaTileCall call,
                          Parse(Config("", "zeroed_outputs = [0, 2]")));
  EXPECT_THAT(call.zeroed_outputs, ElementsAre(0, 2));
  EXPECT_TRUE(Parse(Config()).value().zeroed_outputs.empty());
  for (absl::string_view bad :
       {"zeroed_outputs = [\"a\"]", "zeroed_outputs = 0",
        "zeroed_outputs = [-1]", "zeroed_outputs = [4294967296]"}) {
    EXPECT_EQ(ParseStatus(Config("", bad)).code(),
              absl::StatusCode::kInvalidArgument)
        << bad;
  }
}

// Ascending lets the emitter skip sorting and de-duplicating.
TEST_F(CudaTileCallTest, ZeroedOutputsMustBeStrictlyAscending) {
  for (absl::string_view bad :
       {"zeroed_outputs = [1, 0]", "zeroed_outputs = [1, 1]"}) {
    absl::Status status = ParseStatus(Config("", bad));
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << bad;
    EXPECT_THAT(status.message(), HasSubstr("strictly ascending"));
  }
}

}  // namespace
}  // namespace xla::gpu
