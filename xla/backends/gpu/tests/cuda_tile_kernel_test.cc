#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/tile_ir/tileiras_compiler.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

constexpr int kN = 128;

// Kernels over 128 f32 elements, pointer arithmetic only, so a failure can
// only be in XLA's plumbing.

// out[i] = in[i] + 1
constexpr absl::string_view kAddOne = R"(
cuda_tile.module @m {
  entry @add_one(%in : tile<ptr<f32>>, %out : tile<ptr<f32>>) {
    %offsets = iota : tile<128xi32>
    %in_r = reshape %in : tile<ptr<f32>> -> tile<1xptr<f32>>
    %in_b = broadcast %in_r : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %in_p = offset %in_b, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
    %v, %t0 = load_ptr_tko weak %in_p : tile<128xptr<f32>> -> tile<128xf32>, token
    %one = constant <f32: 1.000000e+00> : tile<f32>
    %one_r = reshape %one : tile<f32> -> tile<1xf32>
    %one_b = broadcast %one_r : tile<1xf32> -> tile<128xf32>
    %sum = addf %v, %one_b rounding<nearest_even> : tile<128xf32>
    %out_r = reshape %out : tile<ptr<f32>> -> tile<1xptr<f32>>
    %out_b = broadcast %out_r : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %out_p = offset %out_b, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
    %t1 = store_ptr_tko weak %out_p, %sum : tile<128xptr<f32>>, tile<128xf32> -> token
    return
  }
}
)";

// out[i] = in[i] + 1, with a third parameter the kernel never touches.
constexpr absl::string_view kAddOneWithScratch = R"(
cuda_tile.module @m {
  entry @add_one(%in : tile<ptr<f32>>, %out : tile<ptr<f32>>, %scratch : tile<ptr<i8>>) {
    %offsets = iota : tile<128xi32>
    %in_r = reshape %in : tile<ptr<f32>> -> tile<1xptr<f32>>
    %in_b = broadcast %in_r : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %in_p = offset %in_b, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
    %v, %t0 = load_ptr_tko weak %in_p : tile<128xptr<f32>> -> tile<128xf32>, token
    %one = constant <f32: 1.000000e+00> : tile<f32>
    %one_r = reshape %one : tile<f32> -> tile<1xf32>
    %one_b = broadcast %one_r : tile<1xf32> -> tile<128xf32>
    %sum = addf %v, %one_b rounding<nearest_even> : tile<128xf32>
    %out_r = reshape %out : tile<ptr<f32>> -> tile<1xptr<f32>>
    %out_b = broadcast %out_r : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %out_p = offset %out_b, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
    %t1 = store_ptr_tko weak %out_p, %sum : tile<128xptr<f32>>, tile<128xf32> -> token
    return
  }
}
)";

// out[i] = a[i] + b[i]; `signature` orders the three parameters.
std::string AddTwo(absl::string_view signature) {
  return absl::StrCat(R"(
cuda_tile.module @m {
  entry @add_two()", signature, R"() {
    %offsets = iota : tile<128xi32>
    %a_r = reshape %a : tile<ptr<f32>> -> tile<1xptr<f32>>
    %a_b = broadcast %a_r : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %a_p = offset %a_b, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
    %av, %t0 = load_ptr_tko weak %a_p : tile<128xptr<f32>> -> tile<128xf32>, token
    %b_r = reshape %b : tile<ptr<f32>> -> tile<1xptr<f32>>
    %b_b = broadcast %b_r : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %b_p = offset %b_b, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
    %bv, %t1 = load_ptr_tko weak %b_p : tile<128xptr<f32>> -> tile<128xf32>, token
    %sum = addf %av, %bv rounding<nearest_even> : tile<128xf32>
    %out_r = reshape %out : tile<ptr<f32>> -> tile<1xptr<f32>>
    %out_b = broadcast %out_r : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %out_p = offset %out_b, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
    %t2 = store_ptr_tko weak %out_p, %sum : tile<128xptr<f32>>, tile<128xf32> -> token
    return
  }
}
)");
}

// Each of the tile blocks writes its linear id t = x + 2y + 6z (a (2,3,4)
// grid) over out[128*t, 128*t + 128); the input is unused.
constexpr absl::string_view kGrid = R"(
cuda_tile.module @m {
  entry @grid(%in : tile<ptr<f32>>, %out : tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c2 = constant <i32: 2> : tile<i32>
    %c6 = constant <i32: 6> : tile<i32>
    %c128 = constant <i32: 128> : tile<i32>
    %y2 = muli %by, %c2 : tile<i32>
    %z6 = muli %bz, %c6 : tile<i32>
    %xy = addi %bx, %y2 : tile<i32>
    %t = addi %xy, %z6 : tile<i32>
    %base = muli %t, %c128 : tile<i32>
    %offsets = iota : tile<128xi32>
    %base_r = reshape %base : tile<i32> -> tile<1xi32>
    %base_b = broadcast %base_r : tile<1xi32> -> tile<128xi32>
    %idx = addi %offsets, %base_b : tile<128xi32>
    %out_r = reshape %out : tile<ptr<f32>> -> tile<1xptr<f32>>
    %out_b = broadcast %out_r : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %out_p = offset %out_b, %idx : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
    %tf = itof %t signed rounding<nearest_even> : tile<i32> -> tile<f32>
    %tf_r = reshape %tf : tile<f32> -> tile<1xf32>
    %tf_b = broadcast %tf_r : tile<1xf32> -> tile<128xf32>
    %tok = store_ptr_tko weak %out_p, %tf_b : tile<128xptr<f32>>, tile<128xf32> -> token
    return
  }
}
)";

// MLIR text inside a printed dictionary inside HLO text: every level escapes
// backslash and quote.
std::string EscapeIr(absl::string_view ir) {
  return absl::StrReplaceAll(
      ir, {{"\\", "\\\\\\\\"}, {"\"", "\\\\\\\""}, {"\n", "\\\\n"}});
}

struct Grid {
  int x = 1;
  int y = 1;
  int z = 1;
};

std::string Config(absl::string_view name, absl::string_view ir,
                   absl::string_view extra = "", Grid grid = {}) {
  return absl::StrCat("{ name = \\\"", name,
                      "\\\", kernel_type = \\\"cuda_tile\\\", ir = \\\"",
                      EscapeIr(ir), "\\\", grid_x = ", grid.x,
                      ", grid_y = ", grid.y, ", grid_z = ", grid.z, extra,
                      " }");
}

std::string UnaryHlo(absl::string_view config, absl::string_view attrs = "") {
  return absl::StrCat(R"(
    HloModule cuda_tile_test

    ENTRY main {
      a = f32[128] parameter(0)
      ROOT out = f32[128] custom-call(a),
        custom_call_target="__gpu$xla.gpu.cuda_tile", )",
                      attrs, R"(
        backend_config=")",
                      config, R"("
    })");
}

std::string BinaryHlo(absl::string_view config) {
  return absl::StrCat(R"(
    HloModule cuda_tile_test

    ENTRY main {
      a = f32[128] parameter(0)
      b = f32[128] parameter(1)
      ROOT out = f32[128] custom-call(a, b),
        custom_call_target="__gpu$xla.gpu.cuda_tile",
        backend_config=")",
                      config, R"("
    })");
}

class CudaTileKernelE2ETest : public HloTestBase {
 protected:
  void SetUp() override {
    absl::StatusOr<std::string> tileiras = tile_ir::FindTileIrAssembler(
        GetDebugOptionsFromFlags().xla_gpu_cuda_data_dir());
    if (!tileiras.ok()) {
      GTEST_SKIP() << "no tileiras: " << tileiras.status().message();
    }
  }

  static Literal Ramp(float offset) {
    std::vector<float> values(kN);
    for (int i = 0; i < kN; ++i) values[i] = offset + i;
    return LiteralUtil::CreateR1<float>(values);
  }

  static void ExpectRamp(const Literal& result, float offset) {
    for (int i = 0; i < kN; ++i) {
      EXPECT_EQ(result.Get<float>({i}), offset + i) << "at " << i;
    }
  }
};

TEST_F(CudaTileKernelE2ETest, AddOne) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(UnaryHlo(Config("add_one", kAddOne))));
  Literal a = Ramp(10);
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&a}));
  ExpectRamp(result, 11);
}

TEST_F(CudaTileKernelE2ETest, TwoOperands) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(BinaryHlo(Config(
          "add_two",
          AddTwo("%a : tile<ptr<f32>>, %b : tile<ptr<f32>>, "
                 "%out : tile<ptr<f32>>")))));
  Literal a = Ramp(10);
  Literal b = Ramp(1000);
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Execute(std::move(module), {&a, &b}));
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(result.Get<float>({i}), 1010 + 2 * i) << "at " << i;
  }
}

TEST_F(CudaTileKernelE2ETest, NonTrivialOutputIndices) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(BinaryHlo(Config(
          "add_two",
          AddTwo("%a : tile<ptr<f32>>, %out : tile<ptr<f32>>, "
                 "%b : tile<ptr<f32>>"),
          ", output_indices = [1]"))));
  Literal a = Ramp(10);
  Literal b = Ramp(1000);
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Execute(std::move(module), {&a, &b}));
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(result.Get<float>({i}), 1010 + 2 * i) << "at " << i;
  }
}

// An aliased operand appears twice in the kernel arguments with one device
// pointer; the entry declares both.
TEST_F(CudaTileKernelE2ETest, AliasedInPlace) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(UnaryHlo(
          Config("add_one", kAddOne), "output_to_operand_aliasing={{}: (0, {})},")));
  Literal a = Ramp(10);
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&a}));
  ExpectRamp(result, 11);
}

// One kernel argument per result leaf; a u8 leaf is how a kernel asks for
// scratch.
TEST_F(CudaTileKernelE2ETest, TupleOutputWithScratch) {
  std::string hlo = absl::StrCat(R"(
    HloModule cuda_tile_test

    ENTRY main {
      a = f32[128] parameter(0)
      cc = (f32[128], u8[256]) custom-call(a),
        custom_call_target="__gpu$xla.gpu.cuda_tile",
        backend_config=")",
                                 Config("add_one", kAddOneWithScratch), R"("
      ROOT out = f32[128] get-tuple-element(cc), index=0
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  Literal a = Ramp(10);
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&a}));
  ExpectRamp(result, 11);
}

TEST_F(CudaTileKernelE2ETest, ArityMismatchIsACompileError) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(
                       UnaryHlo(Config("add_one", kAddOneWithScratch))));
  Literal a = Ramp(10);
  absl::Status status = Execute(std::move(module), {&a}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("takes 3 arguments but the custom call supplies 2"));
}

// Proves the dispatch arm: a payload error comes from the parser, not from
// the FFI registry the call would fall into otherwise.
TEST_F(CudaTileKernelE2ETest, MalformedPayloadIsAnError) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(UnaryHlo(
                       Config("add_one", "cuda_tile.module @m {"))));
  Literal a = Ramp(10);
  absl::Status status = Execute(std::move(module), {&a}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("does not parse"));
  EXPECT_THAT(status.message(),
              Not(HasSubstr("No registered implementation")));
}

// Pins the (x, y, z) -> BlockDim mapping: every block writes its own tile
// with its linear id, so a transposed or flattened grid changes the output.
TEST_F(CudaTileKernelE2ETest, ThreeDGrid) {
  constexpr int kTiles = 2 * 3 * 4;
  std::string hlo = absl::StrCat(R"(
    HloModule cuda_tile_test

    ENTRY main {
      a = f32[3072] parameter(0)
      ROOT out = f32[3072] custom-call(a),
        custom_call_target="__gpu$xla.gpu.cuda_tile",
        backend_config=")",
                                 Config("grid", kGrid, "", {2, 3, 4}), R"("
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  Literal a = LiteralUtil::CreateR1<float>(std::vector<float>(kTiles * kN, 0));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&a}));
  for (int t = 0; t < kTiles; ++t) {
    for (int i = 0; i < kN; ++i) {
      EXPECT_EQ(result.Get<float>({t * kN + i}), t) << "tile " << t << " at "
                                                    << i;
    }
  }
}

// The cache is keyed by config and argument count: one more operand must not
// reuse the first kernel.
TEST_F(CudaTileKernelE2ETest, CachedKernelStillChecksArity) {
  std::string config = Config("add_one", kAddOne);
  std::string hlo = absl::StrCat(R"(
    HloModule cuda_tile_test

    ENTRY main {
      a = f32[128] parameter(0)
      c0 = f32[128] custom-call(a),
        custom_call_target="__gpu$xla.gpu.cuda_tile",
        backend_config=")",
                                 config, R"("
      ROOT c1 = f32[128] custom-call(c0, a),
        custom_call_target="__gpu$xla.gpu.cuda_tile",
        backend_config=")",
                                 config, R"("
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  Literal a = Ramp(10);
  absl::Status status = Execute(std::move(module), {&a}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("takes 2 arguments but the custom call supplies 3"));
}

TEST_F(CudaTileKernelE2ETest, CommandBufferCapture) {
  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(
                       UnaryHlo(Config("add_one", kAddOne)), config));
  Literal a = Ramp(10);
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&a}));
  ExpectRamp(result, 11);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
