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

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

class MusaScatterTest : public HloPjRtTestBase {};

TEST_F(MusaScatterTest, PackedBf16ReplacementPreservesBitsUnderContention) {
  // Distinct indices that share a word must not overwrite each other. Repeated
  // indices deliberately use identical values, making the expected result
  // independent of the unspecified order of conflicting scatter updates.
  constexpr char kHlo[] = R"(
HloModule packed_replacement
replace {
  old = bf16[] parameter(0)
  ROOT value = bf16[] parameter(1)
}

ENTRY main {
  input_bits = u16[8194] parameter(0)
  indices = s32[8192,1] parameter(1)
  update_bits = u16[8192] parameter(2)
  input = bf16[8194] bitcast-convert(input_bits)
  updates = bf16[8192] bitcast-convert(update_bits)
  result = bf16[8194] scatter(input, indices, updates),
    update_window_dims={}, inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1,
    to_apply=replace
  ROOT bits = u16[8194] bitcast-convert(result)
}
)";
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<uint16_t> initial(8194, 0x7fa5);
    std::vector<uint16_t> expected = initial;
    std::vector<uint16_t> updates(8192);
    Literal indices = LiteralUtil::CreateFull<int32_t>({8192, 1}, 0);
    for (int i = 0; i < 8192; ++i) {
      int index;
      switch (mode) {
        case 0:
          index = 2 * (i % 4096) + 1;
          break;
        case 1:
          index = i + 1;
          break;
        case 2:
          index = 4096 + i % 2;
          break;
        default:
          index = i % 2 == 0 ? 0 : 8193;
          break;
      }
      indices.Set<int32_t>({i, 0}, index);
      updates[i] = static_cast<uint16_t>(index * 40503 + 17);
      expected[index] = updates[i];
    }
    Literal input = LiteralUtil::CreateR1<uint16_t>(initial);
    Literal update = LiteralUtil::CreateR1<uint16_t>(updates);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(kHlo));
    ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module),
                                                 {&input, &indices, &update}));
    for (int i = 0; i < 8194; ++i) {
      ASSERT_EQ(result.Get<uint16_t>({i}), expected[i]) << "element " << i;
    }
  }
}

TEST_F(MusaScatterTest, BfloatTransposePreservesBitsAcrossPartialTiles) {
  constexpr char kHlo[] = R"(
HloModule shared_transpose
ENTRY main {
  input = bf16[65,67]{1,0} parameter(0)
  ROOT output = bf16[67,65]{1,0} transpose(input), dimensions={1,0}
}
)";
  std::vector<uint16_t> bits(65 * 67);
  for (int i = 0; i < 65 * 67; ++i) bits[i] = i * 40503 + 17;
  Literal input =
      Literal::CreateFromShape(ShapeUtil::MakeShape(BF16, {65, 67}));
  std::memcpy(input.untyped_data(), bits.data(),
              bits.size() * sizeof(uint16_t));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input}));
  std::vector<uint16_t> actual(bits.size());
  std::memcpy(actual.data(), result.untyped_data(),
              actual.size() * sizeof(uint16_t));
  for (int row = 0; row < 65; ++row) {
    for (int col = 0; col < 67; ++col) {
      ASSERT_EQ(actual[col * 65 + row], bits[row * 67 + col])
          << "row " << row << " col " << col;
    }
  }
}

TEST_F(MusaScatterTest, SelectedBfloatConcatTransposePreservesEveryBranch) {
  constexpr char kHlo[] = R"(
HloModule selected_concat_transpose
ENTRY main {
  a = bf16[65,67]{1,0} parameter(0)
  b = bf16[65,67]{1,0} parameter(1)
  mask = pred[65,67]{1,0} parameter(2)
  selected = bf16[65,67]{1,0} select(mask, a, b)
  joined = bf16[130,67]{1,0} concatenate(selected, a), dimensions={0}
  ROOT output = bf16[67,130]{1,0} transpose(joined), dimensions={1,0}
}
)";
  std::vector<uint16_t> a_bits(65 * 67), b_bits(65 * 67);
  Literal a = Literal::CreateFromShape(ShapeUtil::MakeShape(BF16, {65, 67}));
  Literal b = Literal::CreateFromShape(ShapeUtil::MakeShape(BF16, {65, 67}));
  Literal mask = LiteralUtil::CreateFull<bool>({65, 67}, false);
  for (int row = 0; row < 65; ++row) {
    for (int col = 0; col < 67; ++col) {
      const int index = row * 67 + col;
      a_bits[index] = index * 40503 + 17;
      b_bits[index] = index * 23003 + 31;
      mask.Set<bool>({row, col}, (row + col) % 2 == 0);
    }
  }
  std::memcpy(a.untyped_data(), a_bits.data(),
              a_bits.size() * sizeof(uint16_t));
  std::memcpy(b.untyped_data(), b_bits.data(),
              b_bits.size() * sizeof(uint16_t));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&a, &b, &mask}));
  std::vector<uint16_t> actual(130 * 67);
  std::memcpy(actual.data(), result.untyped_data(),
              actual.size() * sizeof(uint16_t));
  for (int row = 0; row < 130; ++row) {
    for (int col = 0; col < 67; ++col) {
      const int source_row = row % 65;
      const int index = source_row * 67 + col;
      const bool choose_a = row >= 65 || (source_row + col) % 2 == 0;
      ASSERT_EQ(actual[col * 130 + row],
                choose_a ? a_bits[index] : b_bits[index])
          << "row " << row << " col " << col;
    }
  }
}

}  // namespace
}  // namespace xla::gpu
