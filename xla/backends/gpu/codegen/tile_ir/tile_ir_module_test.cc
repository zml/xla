#include "xla/backends/gpu/codegen/tile_ir/tile_ir_module.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu::tile_ir {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;

// The milestone kernel: out[i] = in[i] + 1 over 128 f32, pointer arithmetic
// only, in the shape of cuda-tile's README example plus a store.
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

class TileIrModuleTest : public ::testing::Test {
 protected:
  TileIrModuleTest() { LoadMlirDialectsForCudaTile(ctx_); }

  std::vector<std::string> EntryNames(absl::string_view bytecode) {
    mlir::OwningOpRef<mlir::cuda_tile::ModuleOp> module =
        mlir::cuda_tile::readBytecode(
            llvm::MemoryBufferRef(llvm::StringRef(bytecode.data(),
                                                  bytecode.size()),
                                  "bytecode"),
            ctx_);
    std::vector<std::string> names;
    if (!module) return names;
    module->walk([&](mlir::cuda_tile::EntryOp e) {
      names.push_back(e.getSymName().str());
    });
    return names;
  }

  mlir::MLIRContext ctx_;
};

TEST_F(TileIrModuleTest, MilestoneKernelRoundTrips) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::string bytecode,
      CudaTileTextToBytecode({kAddOne, "add_one", "", 2}, ctx_, nullptr));
  EXPECT_TRUE(mlir::cuda_tile::isTileIRBytecode(llvm::MemoryBufferRef(
      llvm::StringRef(bytecode.data(), bytecode.size()), "bytecode")));
  EXPECT_THAT(EntryNames(bytecode), ElementsAre("add_one"));
}

TEST_F(TileIrModuleTest, RenamesTheEntry) {
  TileIrModuleText dump;
  TF_ASSERT_OK_AND_ASSIGN(
      std::string bytecode,
      CudaTileTextToBytecode({kAddOne, "add_one", "renamed", 2}, ctx_,
                             &dump));
  EXPECT_THAT(EntryNames(bytecode), ElementsAre("renamed"));
  EXPECT_THAT(dump.cuda_tile_text, HasSubstr("@renamed"));
  EXPECT_THAT(dump.cuda_tile_text, Not(HasSubstr("@add_one")));
}

TEST_F(TileIrModuleTest, AcceptsABuiltinModuleWrapper) {
  std::string wrapped = absl::StrCat("module {", kAddOne, "}");
  TF_ASSERT_OK_AND_ASSIGN(
      std::string bytecode,
      CudaTileTextToBytecode({wrapped, "add_one", "", 2}, ctx_, nullptr));
  EXPECT_THAT(EntryNames(bytecode), ElementsAre("add_one"));
}

TEST_F(TileIrModuleTest, ScrubsNamedLocations) {
  constexpr absl::string_view kWithLoc = R"(
cuda_tile.module @m {
  entry @k(%in : tile<ptr<f32>>) {
    %offsets = iota : tile<128xi32> loc("iota-loc")
    return loc("ret-loc")
  } loc("entry-loc")
} loc("module-loc")
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::string bytecode,
      CudaTileTextToBytecode({kWithLoc, "k", "", 1}, ctx_, nullptr));
  EXPECT_THAT(EntryNames(bytecode), ElementsAre("k"));
}

TEST_F(TileIrModuleTest, ScalarParameterIsInvalidArgument) {
  // Arity 2 passes; the type check must catch the scalar, which would
  // otherwise receive the low bits of a device address.
  constexpr absl::string_view kScalar = R"(
cuda_tile.module @m {
  entry @k(%n : tile<i32>, %out : tile<ptr<f32>>) {
    return
  }
}
)";
  absl::Status status =
      CudaTileTextToBytecode({kScalar, "k", "", 2}, ctx_, nullptr).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("parameter 0"));
  EXPECT_THAT(status.message(), HasSubstr("tile<ptr<T>>"));
}

TEST_F(TileIrModuleTest, TokenParameterIsInvalidArgument) {
  constexpr absl::string_view kToken = R"(
cuda_tile.module @m {
  entry @k(%p : tile<ptr<f32>>, %t : !cuda_tile.token) {
    return
  }
}
)";
  absl::Status status =
      CudaTileTextToBytecode({kToken, "k", "", -1}, ctx_, nullptr).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("parameter 1"));
}

// The kernel the 3-D grid e2e test launches; parsed here first so a syntax
// slip is caught without a GPU.
TEST_F(TileIrModuleTest, ThreeDGridKernelRoundTrips) {
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
  TF_ASSERT_OK_AND_ASSIGN(
      std::string bytecode,
      CudaTileTextToBytecode({kGrid, "grid", "", 2}, ctx_, nullptr));
  EXPECT_THAT(EntryNames(bytecode), ElementsAre("grid"));
}

TEST_F(TileIrModuleTest, ArityMismatchIsInvalidArgument) {
  absl::Status status =
      CudaTileTextToBytecode({kAddOne, "add_one", "", 3}, ctx_, nullptr)
          .status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("takes 2 arguments but the custom call supplies 3"));
}

TEST_F(TileIrModuleTest, UnknownEntryListsTheSymbols) {
  absl::Status status =
      CudaTileTextToBytecode({kAddOne, "nope", "", -1}, ctx_, nullptr)
          .status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("'nope'"));
  EXPECT_THAT(status.message(), HasSubstr("[add_one]"));
}

TEST_F(TileIrModuleTest, UnparseableTextIsInvalidArgument) {
  absl::Status status =
      CudaTileTextToBytecode({"cuda_tile.module @m {", "k", "", -1}, ctx_,
                             nullptr)
          .status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("does not parse"));
}

TEST_F(TileIrModuleTest, UnsupportedBytecodeVersionIsInvalidArgument) {
  absl::Status status =
      CudaTileTextToBytecode({kAddOne, "add_one", "", 2, 99, 9}, ctx_, nullptr)
          .status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("99.9"));
}

}  // namespace
}  // namespace xla::gpu::tile_ir
