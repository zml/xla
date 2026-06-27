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
#include "xla/backends/gpu/runtime/metal_topk_thunk.h"
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

// Peels the value-preserving result wrappers ZML/Shardy add — a
// GetTupleElement-of-Tuple, and single-operand shape-preserving Sharding /
// LayoutConstraint / xla.sdy.FuncResultSharding custom-calls — off the entry
// root and re-roots to the real computation. GpuCompiler's sharding-removal
// pass strips the sharding custom-calls but does NOT re-root the gte-of-tuple
// result wrapper; without re-rooting, the real output stays an interior value
// that BufferAssignment places in a temp, so the AIR kernel writes a buffer
// that isn't the returned result.
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

// Relax the zml$flash_attn K/V operand layout from row-major {2,1,0} to {2,0,1}.
// The decode KV cache is stored position-major ([..,seqlen,n_kv,hd]); ZML's
// `k.transpose(.h,.k,.hd)` reorders it to head-major [n_kv,seqlen,hd] for the
// custom call. That transpose is a pure LAYOUT change ({2,0,1} on [n_kv,seqlen,hd]
// is byte-identical to the slice) — but the custom call's row-major {2,1,0}
// constraint forces layout assignment to MATERIALIZE it as an O(seqlen) copy per
// layer per token (~460MB/token at seqlen 2048, the dominant decode-collapse
// cost — see project_metal_fa_vec_port). Relaxing the constraint to {2,0,1} lets
// the transpose fold to a free bitcast; MetalFlashAttnThunk then reads K/V via
// position-major strides (it derives them from the operand layout). Backend-only,
// ZML untouched, free on CUDA/AMD (cuda_fa reads the cache natively).
// Trace a flash_attn K/V operand `transpose(reshape?(dynamic-slice(cache, L)))`
// back to the whole [n_layer, seqlen, n_kv, hd] cache + the layer index L.
// Returns false (no rewrite) if the operand isn't this exact per-token
// layer-extraction shape.
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

// Trace the flash_attn `tok` (token-position) operand back to its raw host-set
// entry parameter. ZML feeds the decode position as a u32[] entry parameter
// wrapped in a value-preserving integer convert (u32->s32); the convert is a
// DEVICE op whose output buffer is NOT host-valid when MetalFlashAttnThunk
// encodes (the convert thunk hasn't run yet), so the thunk reads a stale/garbage
// position if it dereferences that operand host-side. The RAW parameter buffer
// IS host-valid at encode time (host-set per layer-exe invocation, exactly like
// the layer index Arg_13, which already reads correctly). Substituting the raw
// parameter for the convert lets the thunk read the live position host-side to
// pick a per-execute kernel variant (nsg). The kernel's on-device tok[0] read is
// unaffected: u32 and s32 are bit-identical for positions < 2^31 (always true
// here, ctx <= 8192), and it reads the operand bits as `int` either way.
// Returns the raw parameter iff `tok` is convert/bitcast/reshape* over a 4-byte
// integer scalar entry parameter; nullptr otherwise (leave the operand as-is).
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

// Eliminate the per-token KV-cache transpose AND layer dynamic-slice that feed
// the zml$flash_attn custom call (the dominant decode-with-context cost — see
// project_metal_fa_vec_port). The cache is stored [n_layer, seqlen, n_kv, hd];
// ZML slices the current layer + transposes to head-major [n_kv, seqlen, hd].
// Both are O(seqlen) copies per layer per token. Instead, rewrite the custom
// call to read the WHOLE cache + a layer-index operand (MetalFlashAttnThunk
// offsets to the layer and reads position-major in place). If the K/V operands
// aren't this exact slice+transpose shape, fall back to just relaxing their
// layout to {2,0,1} (folds the transpose alone). Backend-only, ZML untouched.
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
      // PREFILL carries num_tokens (the real prompt length) as a trailing 5th
      // operand; preserve it through the whole-cache rewrite (appended after the
      // layer index, so the existing operand indices are unchanged). The fallback
      // path keeps the whole operand list intact.
      HloInstruction* num_tokens = (oc == 5) ? cc->mutable_operand(4) : nullptr;

      // Preferred: whole-cache feed (drop slice + transpose).
      HloInstruction *cache_k = nullptr, *layer_k = nullptr;
      HloInstruction *cache_v = nullptr, *layer_v = nullptr;
      if (TraceKVCacheSlice(cc->mutable_operand(1), &cache_k, &layer_k) &&
          TraceKVCacheSlice(cc->mutable_operand(2), &cache_v, &layer_v)) {
        const std::vector<Shape>& old = cc->operand_shapes_with_layout();
        // Feed the RAW host-set token-position parameter (host-valid at encode
        // time) instead of the device convert, so the thunk can read the live
        // position host-side for per-execute nsg selection. Falls back to the
        // convert operand if the expected convert(parameter) shape isn't found.
        HloInstruction* raw_tok = TraceRawTokenParam(cc->mutable_operand(3));
        HloInstruction* tok = raw_tok ? raw_tok : cc->mutable_operand(3);
        const Shape& tok_layout = raw_tok ? raw_tok->shape() : old[3];
        std::vector<HloInstruction*> ops = {cc->mutable_operand(0), cache_k,
                                            cache_v, tok, layer_k};
        // Relax the PREFILL Q operand (q_len = dim1 > 1) layout to {2,0,1} too.
        // ZML's RoPE emits Q as transpose({2,0,1})(token-major rotate-half concat),
        // byte-identical to {2,0,1}; the {2,1,0} constraint forces a materializing
        // head-major copy (the wrapped_transpose seqlen-scaler). Folding it to a
        // bitcast keeps the RoPE slice/concat fusions logical [q_len,n_head,hd]
        // (dim0 = token) → num_tokens-clampable by the existing token-row clamp.
        // The thunk derives Q strides from this layout. Decode (q_len==1) is left
        // head-major.
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

      // Fallback: relax k/v layout to {2,0,1} (fold the transpose only).
      std::vector<Shape> shapes = cc->operand_shapes_with_layout();
      bool changed = false;
      for (int i : {0, 1, 2}) {
        if (i >= static_cast<int>(shapes.size())) continue;
        if (shapes[i].dimensions().size() != 3) continue;
        // Q (operand 0): relax only for prefill (q_len = dim1 > 1) — folds the RoPE
        // head-major transpose (see the whole-cache-feed path). K/V (1,2): always.
        if (i == 0 && shapes[0].dimensions(1) <= 1) continue;
        *shapes[i].mutable_layout() = LayoutUtil::MakeLayout({2, 0, 1});
        changed = true;
      }
      if (changed) cc->set_operand_shapes_with_layout(std::move(shapes));
    }
  }
  return absl::OkStatus();
}

// Walk the optimized module and precompile EVERY lazily-JIT'd Metal kernel's
// pipeline (metallib compile + PSO) now, while we still have a live executor, so
// no first-execute path pays the one-time compile (~compile + ~70ms PSO). Each
// thunk's Prewarm warms the metallib cache and Apple's driver pipeline cache for
// the exact (config, function-constant) variants its Ensure* will request, so
// the first execute is a pure cache hit. Best-effort per op; dedups configs.
// (GEMM kernels are already compiled at thunk-emit time, inside "Compiled all
// models"; only their PSO load is lazy — a smaller residual left for later.)
void PrewarmMetalPipelines(HloModule* module, se::StreamExecutor* stream_exec) {
  absl::flat_hash_set<std::pair<bool, int64_t>> fa_seen;
  absl::flat_hash_set<std::tuple<int, int64_t, int64_t, int64_t>> paged_seen;
  absl::flat_hash_set<std::tuple<int64_t, int64_t, int64_t>> kvw_seen;
  absl::flat_hash_set<std::pair<int, int64_t>> topk_seen;
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
      }
    }
  }
}

// ── Decode KV-cache write fusion ────────────────────────────────────────────
//
// Each decode layer's conditional KV-cache update costs SIX kernels: pure HLO
// cannot express a predicated store, so zml's "write k/v at slot unless the
// slot is padding (-1)" compiles to
//
//   Z = s32[4] zeros                                   (1 kLoop fusion)
//   valid = and-reduce(Z <= [page,off,0,0] <= bounds)  (1 pred[] reduce fusion)
//   pred = broadcast(valid) [1,1,H,D]                  (1 kLoop fusion)
//   old_k = dynamic_slice(k_cache, page, off)          (1 kLoop fusion)
//   old_v = dynamic_slice(v_cache, page, off)          (1 kLoop fusion)
//   (k_cache, v_cache) = in-place DUS fusion:          (1 kLoop fusion)
//       k_cache[page,off] = select(pred, rope(k_new), old_k)
//       v_cache[page,off] = select(pred, v_new,       old_v)
//
// — five launch-floor dispatches (~12us/layer/token) of index plumbing around
// one real write. The slices/select can't merge into the DUS fusion: an extra
// read of the DUS target breaks the in-place property and would materialize
// the whole cache. RewriteKvCacheWrites pattern-matches the cluster and
// replaces it with ONE "metal$kv_write" custom call (MetalKvWriteThunk: a
// predicated store, `if (slot<0 || slot>=P*B) return`), aliasing both tuple
// outputs to the cache operands like the DUS did. The dead producers DCE away.
//
// Soundness of dropping pred/slices: pred ⟺ slot ∈ [0, P*B) (verified via the
// bounds constant {P-1,B-1,0,0}, the shift==log2(B) / mask==B-1 chain, and the
// and-reduce); when pred is false the DUS wrote old values back to the same
// clamped location — a no-op the kernel's early return replicates.

// Strips converts/broadcasts/bitcasts (the rope chain's plumbing).
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

// Collects the transitive operand cone of `i` (within one fused computation).
void CollectCone(const HloInstruction* i,
                 absl::flat_hash_set<const HloInstruction*>* cone) {
  if (!cone->insert(i).second) return;
  for (const HloInstruction* op : i->operands()) CollectCone(op, cone);
}

// One DUS arm of the cluster, in fused-computation instructions.
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

// Verifies the negative-preserving page/off index chain hanging off a DUS (or
// dynamic-slice) index operand: finds the single s32[1] parameter in its cone
// plus shift-right-logical by log2(block) and mask by block-1.
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

// Matches one root-tuple element: in-place DUS of select(pred, new, old).
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
  // The page-index operand's cone has only the shift; the in-page-offset
  // operand goes through the concat of both, so its cone has shift AND mask.
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
    // Raw arm (V): new value stored as-is.
    if (ShapeUtil::ElementsIn(on_true->shape()) != heads * hd) return false;
    arm->new_param = on_true;
    arm->has_rope = false;
    return true;
  }
  // Rope arm (K): concat(bf16(x1*cos - x2*sin), bf16(x1*sin + x2*cos)).
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
  // Each multiply pairs a half-slice of k_new with cos or sin of the angle.
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
    // Slice of bitcast(k_new): last-dim [0:hd/2] or [hd/2:hd].
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
  // x1*cos - x2*sin ; x1*sin + x2*cos (x1 = low half, x2 = high half).
  if (!(lo0 && cos0 && !lo1 && !cos1 && lo2 && !cos2 && !lo3 && cos3)) {
    return false;
  }
  // angle = mul(broadcast(f32(pos)), freq_table).
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

// Verifies F's pred operand chain: fusion(broadcast) <- fusion(and-reduce with
// bounds {P-1, B-1, 0, 0}) <- the same slot instruction F's DUS indices use.
bool MatchPredChain(const HloInstruction* pred_operand,
                    const HloInstruction* slot_instr, int64_t pages,
                    int64_t block) {
  // The pred[] -> pred[1,1,H,D] broadcast layer: at this pipeline point (end
  // of RunHloPasses) it is a BARE kBroadcast — FusionWrapper only wraps it
  // into the `wrapped_broadcast` kLoop fusion later, in RunBackend's
  // pre-scheduling passes. Accept both forms.
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
  // The reduce itself: and-reduce to pred[].
  const HloInstruction* root = reduce_f->fused_expression_root();
  if (root->opcode() != HloOpcode::kReduce ||
      root->to_apply()->root_instruction()->opcode() != HloOpcode::kAnd) {
    return false;
  }
  return saw_bounds && saw_slot;
}

// The old-slice producer: kLoop fusion rooted at dynamic-slice of the SAME
// cache with the SAME slot (so the pred=false write-back was a no-op).
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
      // Cache geometry from arm 0's target (validated per arm below).
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
      // Both arms share one pred and one slot.
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
    // The slice/pred/zero producer fusions are now dead.
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }
  return changed;
}

}  // namespace

MetalGpuCompiler::MetalGpuCompiler()
    : GpuCompiler(stream_executor::metal::kMetalPlatformId,
                  metal::TargetTriple(), metal::DataLayout()) {}

absl::StatusOr<std::unique_ptr<HloModule>> MetalGpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  // Force single-device so GpuCompiler::RunSPMDPasses takes the sharding-removal
  // branch (ShardingRemover + ShardyXLA + HloDCE) instead of the SPMD
  // partitioner, which on a single Metal device packs entry params into a temp
  // and makes multi-input fusions read zero. num_partitions==1
  // alone flips the `num_partitions>1 && use_spmd_partitioning` gate. See the
  // header for the full rationale.
  module->mutable_config().set_num_partitions(1);
  // Re-root past the ZML/Shardy result wrappers BEFORE the base pipeline, so the
  // real computation output is the entry result (not an interior temp). The base
  // sharding-removal pass alone doesn't re-root the gte-of-tuple wrapper.
  TF_RETURN_IF_ERROR(StripTransparentResultWrappers(module.get()));
  // Relax the flash-attn K/V operand layout so the per-token KV-cache transpose
  // folds to a bitcast (the decode seqlen-collapse cost). Before the base
  // pipeline / layout assignment so the relaxed constraint is honored.
  TF_RETURN_IF_ERROR(RelaxFlashAttnKVLayout(module.get()));
  // Disable DotMerger on Metal: it merges the per-projection dots that share an
  // activation lhs (q,k,v and gate,up in a Llama layer) into one GEMM by
  // CONCATENATING their weights, which materializes a ~100MB weight-concat COPY per
  // layer per prefill (~12.5ms, ~27% of prefill GPU time on Llama-3.2-3B; warm
  // prefill 63ms→50.5ms when disabled, verified bit-identical). For
  // skinny-batch prefill the separate per-projection dots — each its own metal$gemm
  // with the occupancy-optimized prefill tile — are cheaper than merge+concat.
  // Correctness-preserving: dot(A, concat(w_i)) == concat(dot(A, w_i)) element-wise.
  // Threshold 0 ⇒ no dot qualifies (dot_merger.cc:427 `bytes <= max_size_to_merge`).
  // Use set_config (fresh shared_ptr) rather than mutable_config() so the value
  // survives a later SPMD/Shardy pass re-sharing the config.
  {
    HloModuleConfig cfg = module->config();
    cfg.mutable_debug_options().set_xla_gpu_dot_merger_threshold_mb(0);
    module->set_config(std::move(cfg));
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> optimized,
      GpuCompiler::RunHloPasses(std::move(module), stream_exec, options));
  // The decode KV-write cluster only exists in its final 6-kernel form after
  // the whole fusion pipeline (incl. FusionWrapper), so rewrite it here — and
  // before RunBackend so copy insertion / buffer assignment see the custom
  // call's output aliasing.
  TF_RETURN_IF_ERROR(RewriteKvCacheWrites(optimized.get()).status());
  // Now every custom call's final form/layout is fixed and we still have a live
  // executor: precompile each lazily-JIT'd Metal pipeline so the first execute
  // is warm (flash-attn, paged-attn incl. all nsg decode variants, kv-write,
  // TopK).
  PrewarmMetalPipelines(optimized.get(), stream_exec);
  return optimized;
}

absl::Status MetalGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version, const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  // No cuDNN/MIOpen-style convolution canonicalization on Metal.
  return absl::OkStatus();
}

void MetalGpuCompiler::AddPaddingForGpublasGemms(
    HloPassPipeline& pipeline, const DebugOptions& debug_options,
    const se::GpuComputeCapability& gpu_version) {
  // No-op: MPSGraph handles any gemm shape, no cuBLAS-style padding needed.
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
  // No-op: add no autotuning pass. The base impl calls GetAutotunerBackends ->
  // PlatformObjectRegistry::FindObject<GetCodegenBackends>(PlatformId()), which
  // has no Metal registration (compile-time NotFound). Metal gemm = MPSGraph.
  return absl::OkStatus();
}

std::vector<std::string> MetalGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  // We never invoke an LLVM target backend; air-opt flags are passed on the
  // air-opt command line inside CompileMetalAirToMetallib.
  return {};
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
MetalGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const se::DeviceDescription& device_description, bool relocatable,
    const HloModule* debug_module, std::optional<int> shard_number) {
  if (relocatable) {
    // CanUseLinkModules() stays false (base default), so the split-module path
    // never requests a relocatable compile; mirror amdgpu and reject loudly.
    return absl::UnimplementedError(
        "Metal AIR backend does not support relocatable compilation.");
  }

  // Re-stamp the AIR module envelope (triple/datalayout + the air.* module flags
  // and air.version/language_version/compile_options named metadata) on the
  // whole linked module before printing. GpuCompiler may merge in a constants
  // module that carries only triple+datalayout, so the combined module needs the
  // envelope re-applied for air-as to accept it (the merge dedups the duplicate
  // module flags so air-as does not reject them).
  metal::StampAirModuleEnvelope(*llvm_module);

  std::string air_ll;
  llvm::raw_string_ostream os(air_ll);
  llvm_module->print(os, /*AAW=*/nullptr);
  os.flush();

  // Owning string: the ternary with a string literal would otherwise
  // materialize a temporary std::string and dangle a string_view into it.
  const std::string name =
      debug_module != nullptr ? debug_module->name() : "metal_air_module";
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> metallib,
                      CompileMetalAirToMetallib(air_ll, name));

  // asm_text is left empty on purpose: the base prepends a "// GPU Executable
  // PTX" marker to asm_text, which would contaminate the .ll if it were ever
  // re-parsed. The metallib bytes are the device binary.
  return BackendCompileResult{/*asm_text=*/"", /*binary=*/std::move(metallib)};
}

}  // namespace gpu
}  // namespace xla
