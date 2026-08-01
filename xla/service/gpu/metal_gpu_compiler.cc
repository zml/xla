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

#include "xla/service/gpu/metal_gpu_compiler.h"

#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/metal_air_metadata.h"
#include "xla/backends/gpu/runtime/metal_flash_attn_thunk.h"
#include "xla/backends/gpu/runtime/metal_kv_write_thunk.h"
#include "xla/backends/gpu/runtime/metal_paged_attn_thunk.h"
#include "xla/backends/gpu/runtime/metal_sort_thunk.h"
#include "xla/backends/gpu/runtime/metal_topk_thunk.h"
#include "xla/comparison_util.h"
#include "xla/service/gpu/metal_custom_calls.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

absl::Status StripTransparentResultWrappers(HloModule* module) {
  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();
  HloInstruction* unwrapped = root;
  while (true) {
    if (unwrapped->opcode() == HloOpcode::kGetTupleElement &&
        unwrapped->operand(0)->opcode() == HloOpcode::kTuple) {
      unwrapped = unwrapped->mutable_operand(0)->mutable_operand(
          unwrapped->tuple_index());
    } else if (unwrapped->opcode() == HloOpcode::kCustomCall &&
               unwrapped->operand_count() == 1 &&
               ShapeUtil::Compatible(unwrapped->shape(),
                                     unwrapped->operand(0)->shape()) &&
               unwrapped->IsCustomCall({"Sharding", "LayoutConstraint",
                                        "xla.sdy.FuncResultSharding"})) {
      unwrapped = unwrapped->mutable_operand(0);
    } else {
      break;
    }
  }
  if (unwrapped != root) {
    entry->set_root_instruction(unwrapped, /*accept_different_shape=*/true);
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }
  return absl::OkStatus();
}

bool TraceKVCacheSlice(HloInstruction* op, HloInstruction** cache,
                       HloInstruction** layer) {
  if (op->opcode() != HloOpcode::kTranspose) return false;
  HloInstruction* r = op->mutable_operand(0);
  HloInstruction* ds = r->opcode() == HloOpcode::kReshape ? r->mutable_operand(0) : r;
  if (ds->opcode() != HloOpcode::kDynamicSlice) return false;
  if (ds->operand(0)->shape().dimensions().size() != 4) return false;
  auto* dsi = DynCast<HloDynamicSliceInstruction>(ds);
  if (dsi == nullptr || dsi->dynamic_slice_sizes().empty() ||
      dsi->dynamic_slice_sizes()[0] != 1) {
    return false;  // must extract exactly one layer (dim 0)
  }
  *cache = ds->mutable_operand(0);
  *layer = ds->mutable_operand(1);
  return true;
}

HloInstruction* TraceRawTokenParam(HloInstruction* tok) {
  HloInstruction* t = tok;
  while (t->opcode() == HloOpcode::kConvert ||
         t->opcode() == HloOpcode::kBitcast ||
         t->opcode() == HloOpcode::kReshape) {
    t = t->mutable_operand(0);
  }
  if (t->opcode() != HloOpcode::kParameter) return nullptr;
  const Shape& s = t->shape();
  if (!s.IsArray() || ShapeUtil::ElementsIn(s) != 1) return nullptr;
  if (s.element_type() != S32 && s.element_type() != U32) return nullptr;
  return t;
}

absl::Status RelaxFlashAttnKVLayout(HloModule* module) {
  for (HloComputation* comp : module->computations()) {
    std::vector<HloInstruction*> ccs;
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kCustomCall &&
          Cast<HloCustomCallInstruction>(instr)->custom_call_target() ==
              "zml$flash_attn") {
        ccs.push_back(instr);
      }
    }
    for (HloInstruction* instr : ccs) {
      auto* cc = Cast<HloCustomCallInstruction>(instr);
      const int oc = cc->operand_count();
      if ((oc != 4 && oc != 5) || !cc->layout_constrained()) continue;
      HloInstruction* num_tokens = (oc == 5) ? cc->mutable_operand(4) : nullptr;

      HloInstruction *cache_k = nullptr, *layer_k = nullptr;
      HloInstruction *cache_v = nullptr, *layer_v = nullptr;
      if (TraceKVCacheSlice(cc->mutable_operand(1), &cache_k, &layer_k) &&
          TraceKVCacheSlice(cc->mutable_operand(2), &cache_v, &layer_v)) {
        const std::vector<Shape>& old = cc->operand_shapes_with_layout();
        HloInstruction* raw_tok = TraceRawTokenParam(cc->mutable_operand(3));
        HloInstruction* tok = raw_tok ? raw_tok : cc->mutable_operand(3);
        const Shape& tok_layout = raw_tok ? raw_tok->shape() : old[3];
        std::vector<HloInstruction*> ops = {cc->mutable_operand(0), cache_k,
                                            cache_v, tok, layer_k};
        Shape q_lay = old[0];
        if (q_lay.dimensions().size() == 3 && q_lay.dimensions(1) > 1) {
          *q_lay.mutable_layout() = LayoutUtil::MakeLayout({2, 0, 1});
        }
        std::vector<Shape> layouts = {q_lay, cache_k->shape(), cache_v->shape(),
                                      tok_layout, layer_k->shape()};
        if (num_tokens != nullptr) {
          ops.push_back(num_tokens);
          layouts.push_back(num_tokens->shape());
        }
        HloInstruction* new_cc = comp->AddInstruction(
            HloInstruction::CreateCustomCall(
                cc->shape(), ops, cc->custom_call_target(), layouts,
                cc->raw_backend_config_string(), cc->api_version()));
        TF_RETURN_IF_ERROR(cc->ReplaceAllUsesWith(new_cc));
        TF_RETURN_IF_ERROR(comp->RemoveInstruction(cc));
        continue;
      }

      std::vector<Shape> shapes = cc->operand_shapes_with_layout();
      bool changed = false;
      for (int i : {0, 1, 2}) {
        if (i >= static_cast<int>(shapes.size())) continue;
        if (shapes[i].dimensions().size() != 3) continue;
        if (i == 0 && shapes[0].dimensions(1) <= 1) continue;
        *shapes[i].mutable_layout() = LayoutUtil::MakeLayout({2, 0, 1});
        changed = true;
      }
      if (changed) cc->set_operand_shapes_with_layout(std::move(shapes));
    }
  }
  return absl::OkStatus();
}

void PrewarmMetalPipelines(HloModule* module, se::StreamExecutor* stream_exec) {
  absl::flat_hash_set<std::pair<bool, int64_t>> fa_seen;
  absl::flat_hash_set<std::tuple<int, int64_t, int64_t, int64_t>> paged_seen;
  absl::flat_hash_set<std::tuple<int64_t, int64_t, int64_t>> kvw_seen;
  absl::flat_hash_set<std::pair<int, int64_t>> topk_seen;
  absl::flat_hash_set<std::pair<int, bool>> sort_seen;
  for (const HloComputation* comp : module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != HloOpcode::kCustomCall) continue;
      const absl::string_view target =
          Cast<HloCustomCallInstruction>(instr)->custom_call_target();

      if (target == "zml$flash_attn") {
        const int oc = instr->operand_count();
        if (oc < 4 || oc > 6) continue;  // q,k,v,tok[,layer][,num_tokens]
        const Shape& q = instr->operand(0)->shape();
        const Shape& k = instr->operand(1)->shape();
        if (q.dimensions().size() != 3) continue;
        const bool is_prefill = q.dimensions(1) > 1;
        const int64_t hd = q.dimensions(2);
        const bool full_cache = (k.dimensions().size() == 4);
        const int64_t n_kv = full_cache ? k.dimensions(2) : k.dimensions(0);
        const int64_t seqlen = k.dimensions(1);
        bool pos_major = full_cache;
        if (!full_cache && k.layout().minor_to_major().size() == 3) {
          pos_major = k.layout().minor_to_major(0) == 2 &&
                      k.layout().minor_to_major(1) == 0 &&
                      k.layout().minor_to_major(2) == 1;
        }
        const int64_t kv_pos_stride = pos_major ? n_kv * hd : hd;
        if (fa_seen.insert({is_prefill, kv_pos_stride}).second) {
          MetalFlashAttnThunk::PrewarmPipeline(stream_exec, is_prefill,
                                               kv_pos_stride, seqlen, hd);
        }
      } else if (target == "zml$paged_attn") {
        if (instr->operand_count() != 6) continue;
        const Shape& q = instr->operand(0)->shape();       // [total_q,heads,hd]
        const Shape& kc = instr->operand(1)->shape();      // [P,B,n_kv,hd]
        if (q.dimensions().size() != 3 || kc.dimensions().size() != 4) continue;
        const int64_t head_dim = q.dimensions(2);
        const int64_t block_size = kc.dimensions(1);
        const int64_t num_kv_heads = kc.dimensions(2);
        const int dt = static_cast<int>(q.element_type());
        if (paged_seen.insert({dt, head_dim, block_size, num_kv_heads}).second) {
          MetalPagedAttnThunk::Prewarm(stream_exec, q.element_type(), head_dim,
                                       block_size, num_kv_heads);
        }
      } else if (target == "metal$kv_write") {
        if (instr->operand_count() != 7) continue;
        const Shape& kc = instr->operand(0)->shape();      // [P,B,H,D]
        if (kc.dimensions().size() != 4) continue;
        const int64_t num_slots = kc.dimensions(0) * kc.dimensions(1);
        const int64_t kv_heads = kc.dimensions(2);
        const int64_t head_dim = kc.dimensions(3);
        if (kvw_seen.insert({num_slots, kv_heads, head_dim}).second) {
          MetalKvWriteThunk::Prewarm(stream_exec, num_slots, kv_heads, head_dim);
        }
      } else if (target == "__gpu$TopK") {
        if (instr->operand_count() != 1 || !instr->shape().IsTuple() ||
            instr->shape().tuple_shapes().size() != 2) {
          continue;
        }
        const Shape& data = instr->operand(0)->shape();
        const Shape& vals = instr->shape().tuple_shapes()[0];
        if (data.dimensions().empty() || vals.dimensions().empty()) continue;
        const int dt = static_cast<int>(data.element_type());
        const int64_t k = vals.dimensions(vals.dimensions().size() - 1);
        if (topk_seen.insert({dt, k}).second) {
          MetalTopKThunk::Prewarm(stream_exec, data.element_type(), k);
        }
      } else if (target == "metal$sort") {
        if (instr->operand_count() != 1) continue;
        const Shape& v = instr->operand(0)->shape();
        const int dt = static_cast<int>(v.element_type());
        const bool desc =
            Cast<HloCustomCallInstruction>(instr)->opaque() == "desc";
        if (sort_seen.insert({dt, desc}).second) {
          MetalSortThunk::Prewarm(stream_exec, v.element_type(), desc);
        }
      }
    }
  }
}

const HloInstruction* StripElementwisePlumbing(const HloInstruction* i) {
  while (i->opcode() == HloOpcode::kConvert ||
         i->opcode() == HloOpcode::kBroadcast ||
         i->opcode() == HloOpcode::kBitcast) {
    i = i->operand(0);
  }
  return i;
}

const HloInstruction* SkipBitcastsOnly(const HloInstruction* i) {
  while (i->opcode() == HloOpcode::kBitcast) i = i->operand(0);
  return i;
}

void CollectCone(const HloInstruction* i,
                 absl::flat_hash_set<const HloInstruction*>* cone) {
  if (!cone->insert(i).second) return;
  for (const HloInstruction* op : i->operands()) CollectCone(op, cone);
}

struct KvWriteArm {
  const HloInstruction* target_param = nullptr;  // cache parameter
  const HloInstruction* pred_param = nullptr;    // pred[1,1,H,D] parameter
  const HloInstruction* old_param = nullptr;     // old-slice parameter
  const HloInstruction* new_param = nullptr;     // k_new / v_new parameter
  const HloInstruction* pos_param = nullptr;     // s32[1] (rope arm only)
  const HloInstruction* freq_param = nullptr;    // f32[1,D/2] (rope arm only)
  const HloInstruction* slot_param = nullptr;    // s32[1] from the DUS indices
  bool has_rope = false;
};

bool MatchSlotIndexChain(const HloInstruction* idx0, int64_t block,
                         const HloInstruction** slot_param) {
  absl::flat_hash_set<const HloInstruction*> cone;
  CollectCone(idx0, &cone);
  const HloInstruction* param = nullptr;
  bool saw_shift = false, saw_mask = false;
  for (const HloInstruction* i : cone) {
    if (i->opcode() == HloOpcode::kParameter) {
      if (param != nullptr && param != i) return false;
      param = i;
    } else if (i->opcode() == HloOpcode::kShiftRightLogical &&
               i->operand(1)->IsConstant()) {
      int32_t log2b = 0;
      while ((int64_t{1} << log2b) < block) ++log2b;
      const auto& lit = i->operand(1)->literal();
      if (lit.element_count() >= 1 &&
          lit.GetFirstElement<int32_t>() == log2b) {
        saw_shift = true;
      }
    } else if (i->opcode() == HloOpcode::kAnd && i->operand(1)->IsConstant()) {
      const auto& lit = i->operand(1)->literal();
      if (lit.element_count() >= 1 &&
          lit.GetFirstElement<int32_t>() == static_cast<int32_t>(block - 1)) {
        saw_mask = true;
      }
    }
  }
  if (param == nullptr || !saw_shift || !saw_mask) return false;
  if (param->shape().dimensions().size() != 1 ||
      param->shape().dimensions(0) != 1 ||
      param->shape().element_type() != S32) {
    return false;
  }
  *slot_param = param;
  return true;
}

bool MatchKvWriteArm(const HloInstruction* dus, int64_t block, KvWriteArm* arm) {
  if (dus->opcode() != HloOpcode::kDynamicUpdateSlice) return false;
  const HloInstruction* target = SkipBitcastsOnly(dus->operand(0));
  if (target->opcode() != HloOpcode::kParameter) return false;
  const Shape& cache = target->shape();
  if (cache.dimensions().size() != 4 || cache.element_type() != BF16) {
    return false;
  }
  const HloInstruction* sel = dus->operand(1);
  if (sel->opcode() != HloOpcode::kSelect) return false;
  if (sel->operand(0)->opcode() != HloOpcode::kParameter) return false;
  const HloInstruction* old_p = SkipBitcastsOnly(sel->operand(2));
  if (old_p->opcode() != HloOpcode::kParameter) return false;
  if (!MatchSlotIndexChain(dus->operand(2), block, &arm->slot_param) &&
      !MatchSlotIndexChain(dus->operand(3), block, &arm->slot_param)) {
    return false;
  }
  arm->target_param = target;
  arm->pred_param = sel->operand(0);
  arm->old_param = old_p;

  const int64_t heads = cache.dimensions(2), hd = cache.dimensions(3);
  const HloInstruction* on_true = SkipBitcastsOnly(sel->operand(1));
  if (on_true->opcode() == HloOpcode::kParameter) {
    if (ShapeUtil::ElementsIn(on_true->shape()) != heads * hd) return false;
    arm->new_param = on_true;
    arm->has_rope = false;
    return true;
  }
  if (on_true->opcode() != HloOpcode::kConcatenate ||
      on_true->operand_count() != 2) {
    return false;
  }
  const HloInstruction* sub = on_true->operand(0);
  if (sub->opcode() == HloOpcode::kConvert) sub = sub->operand(0);
  const HloInstruction* add = on_true->operand(1);
  if (add->opcode() == HloOpcode::kConvert) add = add->operand(0);
  if (sub->opcode() != HloOpcode::kSubtract || add->opcode() != HloOpcode::kAdd)
    return false;
  const HloInstruction* knew = nullptr;
  const HloInstruction* angle = nullptr;  // the shared multiply under cos/sin
  auto classify = [&](const HloInstruction* mul, bool* lo_half,
                      bool* is_cos) -> bool {
    if (mul->opcode() != HloOpcode::kMultiply) return false;
    const HloInstruction* slice = nullptr;
    const HloInstruction* trig = nullptr;
    for (int k = 0; k < 2; ++k) {
      const HloInstruction* s = StripElementwisePlumbing(mul->operand(k));
      if (s->opcode() == HloOpcode::kSlice) {
        slice = s;
      } else if (s->opcode() == HloOpcode::kCos ||
                 s->opcode() == HloOpcode::kSin) {
        trig = s;
      }
    }
    if (slice == nullptr || trig == nullptr) return false;
    const HloInstruction* src = SkipBitcastsOnly(slice->operand(0));
    if (src->opcode() != HloOpcode::kParameter) return false;
    if (ShapeUtil::ElementsIn(src->shape()) != heads * hd) return false;
    if (knew != nullptr && knew != src) return false;
    knew = src;
    const int last = slice->shape().dimensions().size() - 1;
    const int64_t lo = slice->slice_starts(last), hi = slice->slice_limits(last);
    if (lo == 0 && hi == hd / 2) {
      *lo_half = true;
    } else if (lo == hd / 2 && hi == hd) {
      *lo_half = false;
    } else {
      return false;
    }
    *is_cos = trig->opcode() == HloOpcode::kCos;
    if (angle != nullptr && angle != trig->operand(0)) return false;
    angle = trig->operand(0);
    return true;
  };
  bool lo0, cos0, lo1, cos1, lo2, cos2, lo3, cos3;
  if (!classify(sub->operand(0), &lo0, &cos0) ||
      !classify(sub->operand(1), &lo1, &cos1) ||
      !classify(add->operand(0), &lo2, &cos2) ||
      !classify(add->operand(1), &lo3, &cos3)) {
    return false;
  }
  if (!(lo0 && cos0 && !lo1 && !cos1 && lo2 && !cos2 && !lo3 && cos3)) {
    return false;
  }
  if (angle->opcode() != HloOpcode::kMultiply) return false;
  const HloInstruction* pos = nullptr;
  const HloInstruction* freq = nullptr;
  for (int k = 0; k < 2; ++k) {
    const HloInstruction* s = StripElementwisePlumbing(angle->operand(k));
    if (s->opcode() != HloOpcode::kParameter) continue;
    if (s->shape().element_type() == S32) {
      pos = s;
    } else if (s->shape().element_type() == F32 &&
               ShapeUtil::ElementsIn(s->shape()) == hd / 2) {
      freq = s;
    }
  }
  if (pos == nullptr || freq == nullptr) return false;
  arm->new_param = knew;
  arm->pos_param = pos;
  arm->freq_param = freq;
  arm->has_rope = true;
  return true;
}

bool MatchPredChain(const HloInstruction* pred_operand,
                    const HloInstruction* slot_instr, int64_t pages,
                    int64_t block) {
  if ((pred_operand->opcode() != HloOpcode::kFusion &&
       pred_operand->opcode() != HloOpcode::kBroadcast) ||
      pred_operand->user_count() != 1 || pred_operand->operand_count() != 1) {
    return false;
  }
  const HloInstruction* reduce_f = pred_operand->operand(0);
  if (reduce_f->opcode() != HloOpcode::kFusion ||
      reduce_f->user_count() != 1 ||
      reduce_f->shape().element_type() != PRED ||
      !reduce_f->shape().dimensions().empty()) {
    return false;
  }
  bool saw_bounds = false, saw_slot = false;
  for (const HloInstruction* op : reduce_f->operands()) {
    if (op == slot_instr) saw_slot = true;
    if (op->opcode() == HloOpcode::kConstant &&
        op->shape().element_type() == S32 &&
        ShapeUtil::ElementsIn(op->shape()) == 4) {
      const auto& lit = op->literal();
      saw_bounds = lit.Get<int32_t>({0}) == static_cast<int32_t>(pages - 1) &&
                   lit.Get<int32_t>({1}) == static_cast<int32_t>(block - 1) &&
                   lit.Get<int32_t>({2}) == 0 && lit.Get<int32_t>({3}) == 0;
    }
  }
  const HloInstruction* root = reduce_f->fused_expression_root();
  if (root->opcode() != HloOpcode::kReduce ||
      root->to_apply()->root_instruction()->opcode() != HloOpcode::kAnd) {
    return false;
  }
  return saw_bounds && saw_slot;
}

bool MatchOldSliceFusion(const HloInstruction* f,
                         const HloInstruction* cache_instr,
                         const HloInstruction* slot_instr, int64_t block) {
  if (f->opcode() != HloOpcode::kFusion || f->user_count() != 1 ||
      f->operand_count() != 2) {
    return false;
  }
  if (!((f->operand(0) == cache_instr && f->operand(1) == slot_instr) ||
        (f->operand(1) == cache_instr && f->operand(0) == slot_instr))) {
    return false;
  }
  const HloInstruction* root = f->fused_expression_root();
  if (root->opcode() != HloOpcode::kDynamicSlice) return false;
  const HloInstruction* slot_param = nullptr;
  return MatchSlotIndexChain(root->operand(1), block, &slot_param) ||
         MatchSlotIndexChain(root->operand(2), block, &slot_param);
}

absl::StatusOr<bool> RewriteKvCacheWrites(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    std::vector<HloInstruction*> fusions;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion &&
          instr->fused_expression_root()->opcode() == HloOpcode::kTuple &&
          instr->fused_expression_root()->operand_count() == 2) {
        fusions.push_back(instr);
      }
    }
    for (HloInstruction* f : fusions) {
      const HloInstruction* root = f->fused_expression_root();
      if (root->operand(0)->opcode() != HloOpcode::kDynamicUpdateSlice) {
        continue;
      }
      const HloInstruction* t0 = SkipBitcastsOnly(root->operand(0)->operand(0));
      if (t0->opcode() != HloOpcode::kParameter ||
          t0->shape().dimensions().size() != 4) {
        continue;
      }
      const int64_t pages = t0->shape().dimensions(0);
      const int64_t block = t0->shape().dimensions(1);
      const int64_t heads = t0->shape().dimensions(2);
      const int64_t hd = t0->shape().dimensions(3);
      if (block < 2 || (block & (block - 1)) != 0 || hd % 2 != 0) continue;

      KvWriteArm a0, a1;
      if (!MatchKvWriteArm(root->operand(0), block, &a0)) {
        continue;
      }
      if (!MatchKvWriteArm(root->operand(1), block, &a1)) {
        continue;
      }
      if (a0.has_rope == a1.has_rope) {
        continue;
      }
      const KvWriteArm& rope = a0.has_rope ? a0 : a1;
      const KvWriteArm& raw = a0.has_rope ? a1 : a0;
      const int64_t rope_idx = a0.has_rope ? 0 : 1;
      if (rope.pred_param != raw.pred_param ||
          rope.slot_param != raw.slot_param) {
        continue;
      }
      auto op_of = [&](const HloInstruction* param) -> HloInstruction* {
        return f->mutable_operand(param->parameter_number());
      };
      HloInstruction* k_cache = op_of(rope.target_param);
      HloInstruction* v_cache = op_of(raw.target_param);
      HloInstruction* k_new = op_of(rope.new_param);
      HloInstruction* v_new = op_of(raw.new_param);
      HloInstruction* slot = op_of(rope.slot_param);
      HloInstruction* pos = op_of(rope.pos_param);
      HloInstruction* freq = op_of(rope.freq_param);
      HloInstruction* pred_operand = op_of(rope.pred_param);
      HloInstruction* old_k = op_of(rope.old_param);
      HloInstruction* old_v = op_of(raw.old_param);
      if (k_cache == v_cache) {
        continue;
      }
      if (!MatchPredChain(pred_operand, slot, pages, block)) {
        continue;
      }
      if (!MatchOldSliceFusion(old_k, k_cache, slot, block) ||
          !MatchOldSliceFusion(old_v, v_cache, slot, block)) {
        continue;
      }

      HloInstruction* cc =
          computation->AddInstruction(HloInstruction::CreateCustomCall(
              f->shape(), {k_cache, k_new, v_cache, v_new, slot, pos, freq},
              "metal$kv_write"));
      Cast<HloCustomCallInstruction>(cc)->set_output_to_operand_aliasing(
          {{ShapeIndex{static_cast<int64_t>(rope_idx)}, {0, {}}},
           {ShapeIndex{static_cast<int64_t>(1 - rope_idx)}, {2, {}}}});
      VLOG(1) << "Metal: fusing decode KV-cache write cluster " << f->name()
              << " -> metal$kv_write (P=" << pages << " B=" << block
              << " H=" << heads << " D=" << hd << ")";
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(f, cc));
      changed = true;
    }
  }
  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }
  return changed;
}

bool IsMetalSortKeyType(PrimitiveType t) {
  return t == BF16 || t == F16 || t == F32 || t == S32 || t == S16 || t == S8 ||
         t == U32 || t == U16 || t == U8;
}

struct SortMatch {
  HloInstruction* value_input;  // external key tensor
  int64_t kidx;                 // which sort operand is the key
  int64_t sort_dim;             // logical axis being sorted
  bool descending;
};

std::optional<SortMatch> MatchMetalSort(HloInstruction* container) {
  const HloSortInstruction* sort = nullptr;
  bool fused = false;
  if (container->opcode() == HloOpcode::kSort) {
    sort = Cast<HloSortInstruction>(container);
  } else if (container->opcode() == HloOpcode::kFusion &&
             container->fused_expression_root()->opcode() == HloOpcode::kSort) {
    sort = Cast<HloSortInstruction>(container->fused_expression_root());
    fused = true;
  } else {
    return std::nullopt;
  }
  const HloComputation* comp = sort->called_computations().front();
  int64_t kidx = -1;
  bool descending = false;
  for (const HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() != HloOpcode::kCompare) continue;
    const HloInstruction* l = instr->operand(0);
    const HloInstruction* r = instr->operand(1);
    if (l->opcode() != HloOpcode::kParameter ||
        r->opcode() != HloOpcode::kParameter) {
      continue;
    }
    const int64_t pl = l->parameter_number(), pr = r->parameter_number();
    if (pl % 2 != 0 || pr != pl + 1) continue;  // canonical (non-reversed) pair
    const PrimitiveType cet = sort->operand(pl / 2)->shape().element_type();
    if (!IsMetalSortKeyType(cet)) continue;  // not the (float/int) key compare
    kidx = pl / 2;
    descending = instr->comparison_direction() == ComparisonDirection::kGt ||
                 instr->comparison_direction() == ComparisonDirection::kGe;
    break;
  }
  if (kidx < 0) return std::nullopt;

  const Shape& kshape = sort->operand(kidx)->shape();
  const int64_t rank = kshape.dimensions().size();
  if (rank < 1) return std::nullopt;
  if (kshape.has_layout() &&
      !LayoutUtil::IsMonotonicWithDim0Major(kshape.layout())) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    if (i == kidx) continue;
    const HloInstruction* op = sort->operand(i);
    if (op->opcode() != HloOpcode::kIota ||
        Cast<HloIotaInstruction>(op)->iota_dimension() != sort->sort_dimension()) {
      return std::nullopt;
    }
  }
  HloInstruction* value_input = nullptr;
  if (fused) {
    const HloInstruction* kp = sort->operand(kidx);
    if (kp->opcode() != HloOpcode::kParameter) return std::nullopt;
    value_input = container->mutable_operand(kp->parameter_number());
  } else {
    value_input = container->mutable_operand(kidx);
  }
  return SortMatch{value_input, kidx, sort->sort_dimension(), descending};
}

absl::StatusOr<bool> RewriteSortToMetalThunk(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    std::vector<HloInstruction*> containers;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kSort ||
          (instr->opcode() == HloOpcode::kFusion &&
           instr->fused_expression_root()->opcode() == HloOpcode::kSort)) {
        containers.push_back(instr);
      }
    }
    for (HloInstruction* container : containers) {
      std::optional<SortMatch> m = MatchMetalSort(container);
      if (!m.has_value()) continue;
      const Shape kshape = m->value_input->shape();
      const Shape idx_shape = ShapeUtil::ChangeElementType(kshape, S32);
      const int64_t rank = kshape.dimensions().size();
      const char* opaque = m->descending ? "desc" : "asc";
      VLOG(1) << "Metal: rewriting Sort " << container->name()
              << " -> metal$sort (" << opaque << ", dim " << m->sort_dim << ")";
      HloInstruction* svals = nullptr;
      HloInstruction* sidxs = nullptr;
      if (m->sort_dim == rank - 1) {
        HloInstruction* cc =
            computation->AddInstruction(HloInstruction::CreateCustomCall(
                ShapeUtil::MakeTupleShape({kshape, idx_shape}), {m->value_input},
                kMetalSortCallTarget, opaque));
        svals = computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(kshape, cc, 0));
        sidxs = computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(idx_shape, cc, 1));
      } else {
        std::vector<int64_t> perm;
        std::vector<int64_t> inv_perm(rank);
        for (int64_t i = 0; i < rank; ++i) {
          if (i != m->sort_dim) perm.push_back(i);
        }
        perm.push_back(m->sort_dim);
        std::vector<int64_t> tdims;
        for (int64_t p : perm) tdims.push_back(kshape.dimensions(p));
        for (int64_t i = 0; i < rank; ++i) inv_perm[perm[i]] = i;
        const Shape vt_shape = ShapeUtil::MakeShape(kshape.element_type(), tdims);
        const Shape vt_idx = ShapeUtil::ChangeElementType(vt_shape, S32);
        HloInstruction* vt = computation->AddInstruction(
            HloInstruction::CreateTranspose(vt_shape, m->value_input, perm));
        HloInstruction* cc =
            computation->AddInstruction(HloInstruction::CreateCustomCall(
                ShapeUtil::MakeTupleShape({vt_shape, vt_idx}), {vt},
                kMetalSortCallTarget, opaque));
        HloInstruction* svt = computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(vt_shape, cc, 0));
        HloInstruction* sit = computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(vt_idx, cc, 1));
        svals = computation->AddInstruction(
            HloInstruction::CreateTranspose(kshape, svt, inv_perm));
        sidxs = computation->AddInstruction(
            HloInstruction::CreateTranspose(idx_shape, sit, inv_perm));
      }
      if (!container->shape().IsTuple()) {
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(container, svals));
      } else {
        std::vector<HloInstruction*> elems(
            container->shape().tuple_shapes().size());
        for (int64_t i = 0; i < static_cast<int64_t>(elems.size()); ++i) {
          if (i == m->kidx) {
            elems[i] = svals;
          } else if (container->shape().tuple_shapes(i).element_type() == S32) {
            elems[i] = sidxs;
          } else {
            elems[i] = computation->AddInstruction(HloInstruction::CreateConvert(
                container->shape().tuple_shapes(i), sidxs));
          }
        }
        HloInstruction* new_tuple =
            computation->AddInstruction(HloInstruction::CreateTuple(elems));
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(container, new_tuple));
      }
      changed = true;
    }
  }
  if (changed) TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  return changed;
}

}  // namespace

MetalGpuCompiler::MetalGpuCompiler()
    : GpuCompiler(stream_executor::metal::kMetalPlatformId,
                  metal::TargetTriple(), metal::DataLayout()) {}

absl::StatusOr<std::unique_ptr<HloModule>> MetalGpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  module->mutable_config().set_num_partitions(1);
  TF_RETURN_IF_ERROR(StripTransparentResultWrappers(module.get()));
  TF_RETURN_IF_ERROR(RelaxFlashAttnKVLayout(module.get()));
  // A fresh config through set_config, not mutable_config(): a later sharding
  // pass re-shares the config and would drop the change.
  {
    HloModuleConfig cfg = module->config();
    cfg.mutable_debug_options().set_xla_gpu_dot_merger_threshold_mb(0);
    cfg.mutable_debug_options().set_xla_gpu_enable_cub_radix_sort(false);
    module->set_config(std::move(cfg));
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> optimized,
      GpuCompiler::RunHloPasses(std::move(module), stream_exec, options));
  TF_RETURN_IF_ERROR(RewriteKvCacheWrites(optimized.get()).status());
  TF_RETURN_IF_ERROR(RewriteSortToMetalThunk(optimized.get()).status());
  PrewarmMetalPipelines(optimized.get(), stream_exec);
  return optimized;
}

absl::Status MetalGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version, const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  return absl::OkStatus();
}

void MetalGpuCompiler::AddPaddingForGpublasGemms(
    HloPassPipeline& pipeline, const DebugOptions& debug_options,
    const se::GpuComputeCapability& gpu_version) {
}

absl::Status MetalGpuCompiler::AddAutotunerPass(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    const se::GpuComputeCapability& gpu_version, const CompileOptions& options,
    tsl::thread::ThreadPool* thread_pool,
    stream_executor::StreamExecutor* stream_executor,
    const GpuTargetConfig* target_config, const AliasInfo* alias_info,
    mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const MultiProcessKeyValueStore& key_value_store) {
  return absl::OkStatus();
}

std::vector<std::string> MetalGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return {};
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
MetalGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const se::DeviceDescription& device_description, bool relocatable,
    const HloModule* debug_module, std::optional<int> shard_number) {
  if (relocatable) {
    return absl::UnimplementedError(
        "Metal AIR backend does not support relocatable compilation.");
  }

  metal::StampAirModuleEnvelope(*llvm_module);

  std::string air_ll;
  llvm::raw_string_ostream os(air_ll);
  llvm_module->print(os, /*AAW=*/nullptr);
  os.flush();

  const std::string name =
      debug_module != nullptr ? debug_module->name() : "metal_air_module";
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> metallib,
                      CompileMetalAirToMetallib(air_ll, name));

  return BackendCompileResult{/*asm_text=*/"", /*binary=*/std::move(metallib)};
}

}  // namespace gpu
}  // namespace xla
