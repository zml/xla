# Qwen3.6-27B-FP8 on Metal (llmd) — bring-up + optimization report

Goal: make `~/models/Qwen/Qwen3.6-27B-FP8` run on Apple Metal in
`monorepo_metal-paged/llmd` (which uses `zml_raphael-metal` + the XLA Metal
plugin), then make it as fast as possible.

## 1. What the model is

Despite the "3.6" name, the checkpoint is architecturally **Qwen3.5**
(`model_type: "qwen3_5"`, `Qwen3_5ForConditionalGeneration`): a 64-layer hybrid
that alternates **Gated-DeltaNet linear attention** (3 of every 4 layers) with
**full attention** (every 4th), GQA with QK-norm, partial RoPE (64/256),
output-gated attention (`attn_output_gate`), and a dense SwiGLU MLP. It is
**multimodal** (vision tower + MTP head are present in the checkpoint but are
not needed for text generation, and llmd ignores them).

The new thing vs the Qwen3.5 already supported by llmd: the weights are
**block-wise FP8** — every big projection is stored as `F8_E4M3` with a
companion `*.weight_scale_inv` (bf16) holding one scale per **128×128** weight
block (DeepSeek-V3 style). On disk: 29 GB. Dequantized to bf16 it would be
~50 GB, which does **not** fit alongside the KV cache on this 64 GB
(≈39 GB usable working-set) M4 Max — so the weights must stay FP8 on device.

## 2. Making it work

The llmd `models/qwen3_5.zig` already implemented the exact architecture
(GDN `in_proj_qkv/z/a/b`, output-gated GQA, partial RoPE, q/k-norm). Four
changes were needed for this checkpoint:

1. **In-graph FP8 block dequant (`QuantLinear`)** — `models/qwen3_5.zig`.
   A drop-in replacement for `zml.nn.Linear` that, when a `weight_scale_inv`
   companion exists, keeps the weight as `f8e4m3fn` on device and dequantizes
   **inside the compiled graph**: `w_bf16 = f8_to_bf16(w) ⊙ scale`, with the
   128-block scale broadcast via `splitAxis`/`broad`/`merge`. Only a *transient*
   bf16 copy of the single weight being multiplied is ever materialized
   (freed after the matmul) — never the whole model. Verified by the loader
   reporting **model size = 27.4 GiB** (≈1 byte/param), not ~50 GB.
   - Works for both weight layouts (`{.dout,.d}` and down_proj's `{.d,.dout}`).
   - The XLA Metal backend lowers `convert(f8e4m3fn→bf16)` via its
     MLIR→LLVM→AIR pipeline (software bit-manipulation; `ExpandFloatOpsPass` /
     `FloatNormalization` with `GpuFloatSupport(F8E4M3FN, F16)`), so no native
     Metal fp8 type is required.

2. **`RmsNormGated` dtype fix** — it multiplied the f32-normalized GDN output by
   a bf16 `norm.weight` (mixed-dtype `mul` panic). Converted the weight to f32
   first, mirroring `RmsNorm.forward`. (Latent bug this checkpoint's bf16 norm
   weights + f32 GDN recurrence exposed.)

3. **GDN recurrent-state pool right-sizing** — the GDN cache reserves
   `managed_pages × 147 MiB`; at the default 64 pages that is ~9.7 GiB, which on
   top of the 27.4 GiB model left **0** pages for the full-attention KV cache
   (OOM at compile). Reduced to 16 pages (~2.9 GiB), leaving headroom for KV.

4. **macOS >2 GiB `pread` fix** — `zml/safetensors.zig`. The large-vocab
   `embed_tokens`/`lm_head` (248320×5120 bf16 = 2.37 GiB each) exceeded macOS's
   `pread`/`preadv` `INT_MAX` (~2 GiB) limit → `EINVAL` during weight load.
   Capped each positional read to 1 GiB (`streamRemaining` loops the rest).

Result: the model **loads, compiles, and serves coherent, correct text** on
Metal. E.g. *"Explain how a transformer neural network works."* →
*"At its core, a Transformer is a neural network architecture designed to handle
sequence data… Its breakthrough comes from one key idea: Attention."*

Run it:
```
cd monorepo_metal-paged
bazel run --@zml//platforms:metal=true //llmd:llmd -- \
  --model=$HOME/models/Qwen/Qwen3.6-27B-FP8 \
  --max-token-count=2048 --seq-len=4096 --batch-size=1 --pages=512 \
  --topk=20 --listen=127.0.0.1:8000
```

## 3. Baseline + profile

- **Baseline decode: ~7.4 tok/s** (single stream; two-point (160−32 tok)),
  ~6.3 tok/s end-to-end; prefill ~1 s for a short prompt. Plugin is
  `darwin_arm64-opt`.
- **`METAL_KPROF` profile** (decode-dominated run): **~85% of GPU time is in
  generic `input_reduce_fusion` GEMVs** (down_proj/gate/up/q/o/…) running at
  ~120 GB/s ≈ **30% of peak** bandwidth. `mpp_tensor_gemm` (metalBLAS) is used
  only for prefill; attention + GDN recurrence are <2%.
- **Root cause:** for M=1 decode the f8→bf16 dequant *prologue* makes the dot's
  weight operand a fusion (not a clean parameter), so the GemmRewriter does not
  reroute it to a fast thin-M GEMV; it falls back to the generic reduce fusion.
  Materializing bf16 then using the tuned bf16 GEMV is *worse* (135 GB vs 27 GB
  weight traffic/token). The only win is a GEMV that **reads f8 directly**.

## 4. Optimization #1 — fused FP8 block-scaled decode GEMV (`zml$fp8_gemv`)

A new custom-call-backed Metal kernel (mirrors the existing `zml$gdn` plumbing)
that reads the f8 weight + 128-block bf16 scales **inline** and dequantizes in
registers, so a decode matmul touches DRAM ~once over the 1-byte weight (≈half
a bf16 GEMV, and avoids the 30%-efficiency reduce path entirely).

- Kernel: `xla/service/gpu/metal_kernels/custom/fp8_gemv.metal` — one
  threadgroup (256 threads / 8 SIMD groups) per output row, reduces over K; a
  256-entry threadgroup LUT turns f8 decode into a single load; the tiny x row
  stays in L2 across rows, so it is ~weight-bandwidth-bound.
- Thunk: `xla/backends/gpu/runtime/metal_fp8_gemv_thunk.{h,cc}`; dispatch in
  `thunk_emitter.cc` on custom-call target `zml$fp8_gemv`.
- Model: `QuantLinear.forward` emits the custom call for **decode** (small
  batch, FP8 weight, no bias); **prefill keeps dequant+dot** (its bf16
  materialization amortizes over many tokens). **On by default**; set
  `METAL_FP8_GEMV_OFF` to fall back (used for the A/B below).

**Result** (single stream, M4 Max, same `darwin_arm64-opt` plugin, A/B via the
escape hatch; correctness verified each step — greedy output stays
coherent/correct, e.g. Rayleigh scattering, ocean tides, Paris answers):

Two iterations, both on the fused kernel:
1. **`zml$fp8_gemv`** (scalar): route decode GEMVs to the f8-reading kernel.
2. **Vectorized loads** (`uchar4`/`bfloat4`, 128/256-byte coalesced
   transactions): all contraction dims here are multiples of 1024, so 4-wide
   tiling is exact + aligned.

| metric | baseline (OFF) | #1 scalar | #2 vectorized |
|---|---|---|---|
| end-to-end decode, 160 tok | 6.4 tok/s | 8.3 tok/s (1.28×) | **9.3 tok/s (1.44×)** |
| per-GEMV (N=17408), KPROF | ~528 µs @ ~120 GB/s | ~296 µs @ ~300 GB/s (1.8×) | — |

**Total: 6.4 → 9.3 tok/s decode = 1.44×.** The per-GEMV win is larger (~1.8×+:
the kernel reaches ≥73% of peak bandwidth vs ~30% for the generic reduce
fusion). The end-to-end win is capped below that because single-stream decode
is **partly host-bound**: each token issues ~66 separate per-layer module
dispatches, and attention / the GDN recurrence / sampling / `lm_head` are
unchanged — so the GEMV speedup only applies to the GPU-GEMV fraction of
wall-clock.

## 5. Notes / further levers (not done)

- **Host dispatch overhead** is now the main decode cap (66 dispatches/token);
  fusing layers into fewer compiled modules would help more than further kernel
  tuning.
- **Kernel tuning:** vectorized (uchar4/bfloat4) loads + multiple output rows
  per threadgroup would push the small-K projections (o/q/k/v) closer to peak;
  the big MLP GEMVs are already ~73% BW.
- A **tiled f8 GEMM** would extend the fused-dequant idea to prefill.

## 6. Reliability fix (pre-existing tokenizer bug, NOT the FP8 work)

Symptom (before the fix): many back-to-back requests returned **0 completion
tokens** — the first sampled token was EOS (only ~3/10 generated). Diagnosis by
elimination, then the **real root cause and fix**:
- **NOT the FP8 kernel** — identical rate/pattern with `METAL_FP8_GEMV_OFF`
  (the first token is produced by prefill dequant+dot + bf16 `lm_head`, never
  the decode kernel).
- **NOT a GDN race or GDN state** — a prefill host-drain and a fresh-sequence
  h0-zero mask both left it unchanged (so both were reverted).
- **Greedy (`--topk=1`) → 0 tokens every time**: the model's first-token
  *argmax* is EOS, so it's not sampling either.
- **Root cause: llmd's IREE tokenizer mis-tokenizes `<think>`.** This model marks
  `<think>`(248068)/`</think>`(248069) (and `<tool_call>`, `<|fim_*|>`, …) as
  added tokens with **`special:false, normalized:false`**. HuggingFace/vLLM match
  *every* added token atomically (`normalized` only picks pre/post-norm;
  `special` is decode-only) → `<think>` = the single learned id **248068**. But
  IREE's `iree_tokenizer_huggingface_build_special_tokens`
  (`runtime/.../huggingface/tokenizer_json.c`) gated inclusion on
  `is_special || is_normalized`, **dropping** `special=false && normalized=false`
  tokens → they were BPE-split into `<`(27)`think`(26003)`>`(29). So the assistant
  turn opened with 3 out-of-distribution tokens instead of the learned 248068,
  and the model — never having seen that — emitted EOS. (vLLM doesn't hit this
  and does **not** use `min_tokens`; it just tokenizes correctly.)
- **Fix:** `zml_raphael-metal/third_party/iree/fix-added-token-matching.patch`
  (wired into `third_party/iree/repo.bzl`) makes the gate match all added tokens
  (`is_special || is_normalized || (!is_special && !is_normalized)`), matching HF.
  This repairs the whole `<think>`/`<tool_call>`/`<|fim_*|>` family for **every**
  model, not just this one. The `suppressFirstTokenEos` band-aid was **removed**.

Result: **12/12 requests generate full, coherent output; `reasoning_content`
(the `<think>` block) now parses correctly.** This is the proper root-cause fix.

## 7. Qwen3.6-35B-A3B-FP8 (the MoE sibling)

Same block-FP8 scheme, but a **256-expert top-8 MoE** (hidden 2048, moe_inter
512, 40 layers, hybrid GDN/full-attn). Experts ship as per-expert
`gate_up_proj`/`down_proj` + `*.weight_scale_inv`; they are **stacked in-loader**
(zml composite tensors, no offline conversion) and run via a model-emitted
`__metal$moe_gemm$f8` custom call after top-k routing (router logits padded to
1024 so TopK uses the Metal radix kernel; `x` repeated by `k`; expert id per
row). Weights stay FP8 on device (~33 GiB) — bf16 (~66 GiB) would not fit 64 GB.

### Decode + prefill GEMM perf

All numbers are per-seq tok/s at batch 16, warm, the 16-prompt bench
(`bench_batch.py`), same machine. The MoE expert GEMM is ~70% of decode GPU
time (KPROF), so it was the whole game.

| stage | tok/s/seq | change |
|---|---|---|
| unsorted MLX Steel gather (initial) | 3.67 | — |
| **P0** dispatch the per-row grouped GEMV instead | 6.92 | revert a regression |
| **P1** x-caching tiled GEMV (`fp8_moe_gemv`) | 8.52 | +23% |
| **P2** sorted-prefill gather | 9.88 | faster prefill lifts e2e |

- **The Steel gather regressed unsorted.** `fp8_gather_qmm_rhs` (vendored MLX)
  run-walks *contiguous same-expert rows* within a BM=16 tile — it is built for
  rows **pre-sorted by expert**. Fed our unsorted top-k rows it degenerates to 16
  length-1 runs/tile (~16× weight over-read), slower than a plain per-row GEMV.
- **P1 — x-caching tiled GEMV.** The per-`(n,row)` GEMV re-read x per output
  column, but that x re-read is absorbed by L2; the real cost was the `down`
  projection being **launch/overhead-bound** (~171 GB/s vs `gate_up`'s ~367,
  near the M4 Max BW peak). Computing `TN=8` columns per threadgroup (8× fewer
  threadgroups) amortizes the per-TG fixed cost: `down` 749→**310µs** (2.4×),
  `gate_up` 698→**529µs**.
- **P2 — sorted prefill.** Argsort-on-Metal is blocked (cub_sort has no Metal
  handler; TopK caps at k≤32), so the thunk does it internally for large R
  (≥1024): a single-threadgroup **counting sort** (`moe_argsort`) groups rows by
  expert, `gather_rows` permutes x into that order, the MLX gather q-GEMM runs
  (now reusing each expert's weight across its contiguous run — measured **13.8×**
  reuse: R=2048 → ~148 distinct experts), `scatter_rows` restores order. Decode
  (R=128, ~0.5 rows/expert: no reuse) stays on the P1 GEMV. Prefill MoE GEMM
  **2220→352ms (6.3×)**; warm TTFT (437-word prompt) **1.08s**; the graph is
  untouched (sort lives entirely in the thunk).

### vs vLLM-metal (the real target)

vLLM-metal has **no MoE kernel of its own** — it runs the model through Apple's
**MLX-LM** on the `mlx-community/Qwen3.6-35B-A3B-8bit` weights (MLX **8-bit
affine**, group_size 64; *not* bf16, *not* our block-FP8). MLX-LM's `SwitchGLU`
**sorts tokens by expert** (`indices.size≥64`) and runs `mx.gather_mm`. So both
sides are ~1 byte/weight — **byte parity** (we are ~6% leaner). Measured on the
same bench/machine: **vLLM 16.3 tok/s/seq** (the user's "~15"). We are at 9.88,
**1.65× behind** (was 1.91× before this work).

The decode gap is **not** one lever and not sorting (sorting helps `gate_up`'s
weight read, ~26% of decode GPU, ~1.1× e2e — it is mainly a *prefill* win): our
warm decode is ~85% GPU-bound and `gate_up` is already near BW peak, so closing
it means **broad MLX-quality kernel efficiency** (MMA-based MoE GEMM + faster
dense projections + lower dispatch overhead). Open, substantial, uncertain
ceiling.

New Metal kernels: `custom/fp8_moe_gemv.metal` (x-caching grouped GEMV),
`custom/moe_argsort.metal` (counting sort), `custom/permute_rows.metal`
(gather/scatter); the MoE thunk is `backends/gpu/runtime/metal_moe_gemv_thunk`.
