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

#include "xla/backends/gpu/transforms/nvfp4_scale_swizzle.h"

#include <cstdint>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

// The involution, as dimension numbers on the 5-D view.
constexpr int64_t kSfPerm[] = {0, 3, 2, 1, 4};

bool DimsAre(const Shape& s, absl::Span<const int64_t> dims) {
  if (s.dimensions().size() != dims.size()) return false;
  for (int64_t i = 0; i < dims.size(); ++i) {
    if (s.dimensions(i) != dims[i]) return false;
  }
  return true;
}

}  // namespace

HloInstruction* EmitNvfp4ScaleUnswizzle(HloInstruction* sf, int64_t n,
                                        int64_t kg) {
  const Shape& s = sf->shape();
  if (s.dimensions().size() != 3) return nullptr;
  if (n % kSfRowsPerBlock != 0 || kg % kSfGroupsPerBlock != 0) return nullptr;
  const int64_t nb = n / kSfRowsPerBlock;
  const int64_t kb = kg / kSfGroupsPerBlock;
  if (!DimsAre(s, {nb, kb, kSfTileElems})) return nullptr;

  HloComputation* comp = sf->parent();
  const PrimitiveType t = s.element_type();

  // [nb, kb, 512] -> [nb, kb, 32, 4, 4]
  HloInstruction* split =
      comp->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(t, {nb, kb, 32, kSfGroupsPerBlock,
                                   kSfGroupsPerBlock}),
          sf));
  // -> [nb, 4, 32, kb, 4]
  HloInstruction* permuted =
      comp->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(t, {nb, kSfGroupsPerBlock, 32, kb,
                                   kSfGroupsPerBlock}),
          split, kSfPerm));
  // -> [N, kg]; the leading three axes flatten to n_blk*128 + m1*32 + m0.
  return comp->AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(t, {n, kg}), permuted));
}

HloInstruction* MatchNvfp4ScaleUnswizzle(HloInstruction* natural) {
  if (natural->opcode() != HloOpcode::kReshape) return nullptr;
  HloInstruction* permuted = natural->mutable_operand(0);
  if (permuted->opcode() != HloOpcode::kTranspose) return nullptr;
  if (!absl::c_equal(permuted->dimensions(), absl::MakeSpan(kSfPerm))) {
    return nullptr;
  }
  HloInstruction* split = permuted->mutable_operand(0);
  if (split->opcode() != HloOpcode::kReshape) return nullptr;
  HloInstruction* sf = split->mutable_operand(0);

  // Re-derive rather than trust: the shapes must be exactly what
  // EmitNvfp4ScaleUnswizzle would have produced for this [N, kg].
  const Shape& out = natural->shape();
  if (out.dimensions().size() != 2) return nullptr;
  const int64_t n = out.dimensions(0);
  const int64_t kg = out.dimensions(1);
  if (n % kSfRowsPerBlock != 0 || kg % kSfGroupsPerBlock != 0) return nullptr;
  const int64_t nb = n / kSfRowsPerBlock;
  const int64_t kb = kg / kSfGroupsPerBlock;
  if (!DimsAre(sf->shape(), {nb, kb, kSfTileElems})) return nullptr;
  if (!DimsAre(split->shape(),
               {nb, kb, 32, kSfGroupsPerBlock, kSfGroupsPerBlock})) {
    return nullptr;
  }
  if (!DimsAre(permuted->shape(),
               {nb, kSfGroupsPerBlock, 32, kb, kSfGroupsPerBlock})) {
    return nullptr;
  }
  if (sf->shape().element_type() != out.element_type()) return nullptr;
  return sf;
}

}  // namespace gpu
}  // namespace xla
