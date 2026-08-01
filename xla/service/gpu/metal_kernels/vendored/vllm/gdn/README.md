# Gated DeltaNet (GDN) kernel — vendored from vllm-metal

`gdn_linear_attention.metal`, copied VERBATIM from
`vllm-metal/vllm_metal/metal/kernels_v2/` (update = `cp`).

The recurrent delta-rule linear-attention kernel for Qwen3-Next style hybrid
models. Full templated `[[kernel]]` (f32/f16/bf16), one SIMD group of 32 threads
per (request, value-head, value-dim) cooperating over the key dim (Dk <= 256).
Handles both prefill (per-request sequence loop) and decode, updating the
recurrent state pool in place.

We vendor only this one. vllm-metal also ships lazy decode/prefill split
variants and a conv1d+SiLU kernel, but:
- the lazy `gdn_recurrent_{decode,prefill}` variants are the SAME recurrence with
  a deferred state-scatter for MLX's lazy graph — no speed gain outside MLX;
- the conv1d+SiLU is a separate preprocessing op, expressible as ordinary XLA
  fusions rather than a custom shader.

**NOT wired into the build yet** — there is intentionally no `embed_files`
target in ../../BUILD for this, so no `get_<stem>()` accessor is generated and no
thunk references it. Drop-only, pending integration.
