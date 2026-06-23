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

#include "xla/backends/gpu/runtime/metal_flash_attn_thunk.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/flash_attn_kernels.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_runtime.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

namespace {

// Exact replica of ggml_metal_kargs_flash_attn_ext_vec (passed as a device buffer
// at index 0 — the kernel's `constant fa_vec_args& [[buffer(0)]]` accepts one).
// logit_softcap mirrors ggml's field but is left 0: zml$flash_attn carries no
// softcap operand, so no logit capping is applied (correct for Llama; a softcap
// model such as Gemma-2 would need ZML to plumb the value and the kernel to
// apply it).
struct FaVecKArgs {
  int32_t ne01, ne02, ne03;
  uint64_t nb01, nb02, nb03;
  int32_t ne11, ne_12_2, ne_12_3;
  int32_t ns10;
  uint64_t nb11, nb12, nb13;
  int32_t ns20;
  uint64_t nb21, nb22, nb23;
  int32_t ne31, ne32, ne33;
  uint64_t nb31, nb32, nb33;
  int32_t ne1, ne2, ne3;
  float scale, max_bias, m0, m1;
  int32_t n_head_log2;
  float logit_softcap;
};
struct FaVecReduceKArgs {
  int32_t nrows;
};

}  // namespace

MetalFlashAttnThunk::MetalFlashAttnThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice q, Shape q_shape,
    BufferAllocation::Slice k, Shape k_shape, BufferAllocation::Slice v,
    Shape v_shape, BufferAllocation::Slice tok, Shape tok_shape,
    BufferAllocation::Slice out, Shape out_shape, int64_t n_kv,
    int64_t n_groups, int64_t seqlen, int64_t head_dim, bool kv_position_major,
    bool kv_full_cache, BufferAllocation::Slice layer, Shape layer_shape,
    BufferAllocation::Slice num_tokens, Shape num_tokens_shape,
    bool tok_host_coherent)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      q_(q),
      k_(k),
      v_(v),
      tok_(tok),
      out_(out),
      q_shape_(std::move(q_shape)),
      k_shape_(std::move(k_shape)),
      v_shape_(std::move(v_shape)),
      tok_shape_(std::move(tok_shape)),
      out_shape_(std::move(out_shape)),
      n_kv_(n_kv),
      n_groups_(n_groups),
      seqlen_(seqlen),
      head_dim_(head_dim),
      kv_position_major_(kv_position_major),
      kv_full_cache_(kv_full_cache),
      layer_(layer),
      layer_shape_(std::move(layer_shape)),
      num_tokens_(num_tokens),
      num_tokens_shape_(std::move(num_tokens_shape)),
      has_num_tokens_(num_tokens.allocation() != nullptr),
      tok_host_coherent_(tok_host_coherent) {}

void MetalFlashAttnThunk::PrewarmPipeline(se::StreamExecutor* executor,
                                         bool is_prefill, int64_t kv_pos_stride,
                                         int64_t seqlen) {
  // Best-effort. Compile + cache the metallib, then create the PSO(s) via the
  // compile-time executor (discarded — it warms Apple's driver pipeline cache, so
  // the thunk's first-execute LoadKernelWithConstants is a cache hit). FC values
  // must match what the thunk will request (see EnsurePrefill / EnsureFaVecMain).
  auto lib = CompileMetalSourceToMetallibCached(is_prefill ? get_flash_attn_prefill() : get_flash_attn_vec());
  if (!lib.ok()) return;
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  using FC = se::metal::MetalFunctionConstant;
  if (is_prefill) {
    // Warm both fa_ext variants: with the on-GPU row clamp (FC_tokclamp=1, the
    // num_tokens buffer at index 7, arity 8 — the prefill-with-num_tokens path) and
    // without (arity 7 — the legacy no-num_tokens path). We don't know has_num_tokens
    // here, so warming both keeps either first execute a pipeline-cache hit.
    for (int clamp = 0; clamp <= 1; ++clamp) {
      const FC fc[] = {{430, FC::Kind::kInt, kv_pos_stride},
                       {431, FC::Kind::kInt, kv_pos_stride},
                       {432, FC::Kind::kBool, clamp}};
      metal_exec
          ->LoadKernelWithConstants(*lib, "fa_ext", /*arity=*/clamp ? 8 : 7, fc)
          .IgnoreError();
    }
    return;
  }
  // Decode auto-ramps nsg ∈ {4,8,16} as the live KV length crosses 1024/2048
  // (ExecuteOnStream). Warm every nsg this seqlen can actually reach, so neither
  // the first token nor a ramp transition pays PSO creation.
  for (int nsg : {4, 8, 16}) {
    if (nsg == 8 && seqlen <= 1024) break;
    if (nsg == 16 && seqlen <= 2048) break;
    const FC fc[] = {{420, FC::Kind::kInt, kv_pos_stride},
                     {421, FC::Kind::kInt, kv_pos_stride},
                     {422, FC::Kind::kInt, nsg},
                     {423, FC::Kind::kInt, 1}};
    metal_exec->LoadKernelWithConstants(*lib, "fa_vec", /*arity=*/7, fc)
        .IgnoreError();
  }
}

// FATTN_SMEM(nsg) from ggml-metal-ops.cpp (half elements -> bytes). Linear in
// nsg: nsg=16 -> 16KB (under the 32KB threadgroup-memory ceiling).
static size_t FaVecSmem(int64_t head_dim, int nsg) {
  auto pad = [](int64_t x, int64_t n) { return ((x + n - 1) / n) * n; };
  return static_cast<size_t>(
      pad((pad(head_dim, 128) + 4 * 32 + 2 * pad(head_dim, 128)) * nsg * 2, 16));
}

absl::Status MetalFlashAttnThunk::EnsureFaVecMain(se::StreamExecutor* executor,
                                                  int idx) {
  if (fa_main_by_nsg_[idx] != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  using FC = se::metal::MetalFunctionConstant;
  const int nsg = kNsgVals[idx];
  const FC fc[] = {{420, FC::Kind::kInt, kv_pos_stride_},
                   {421, FC::Kind::kInt, kv_pos_stride_},
                   {422, FC::Kind::kInt, nsg},
                   {423, FC::Kind::kInt, 1}};
  TF_ASSIGN_OR_RETURN(fa_main_by_nsg_[idx],
                      metal_exec->LoadKernelWithConstants(
                          favec_lib_, "fa_vec", /*arity=*/7, fc));
  smem_by_nsg_[idx] = FaVecSmem(head_dim_, nsg);
  return absl::OkStatus();
}

absl::Status MetalFlashAttnThunk::EnsureFaVecHC(se::StreamExecutor* executor) {
  if (fa_main_hc_ != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  using FC = se::metal::MetalFunctionConstant;
  constexpr int kNwgHC = 32;  // = N_SIMDWIDTH (the reduce sums NWG partials via simd_sum)
  const int64_t HD = head_dim_;
  const int64_t NQH = n_kv_ * n_groups_;
  // NSG = n_kv: one simdgroup per KV head so the heads tile a contiguous load.
  const FC fc[] = {{420, FC::Kind::kInt, kv_pos_stride_},
                   {421, FC::Kind::kInt, kv_pos_stride_},
                   {422, FC::Kind::kInt, n_kv_},
                   {423, FC::Kind::kInt, kNwgHC}};
  TF_ASSIGN_OR_RETURN(fa_main_hc_,
                      metal_exec->LoadKernelWithConstants(
                          favec_lib_, "fa_vec_hc", /*arity=*/7, fc));
  smem_hc_ = FaVecSmem(HD, static_cast<int>(n_kv_));
  // Position-split (NWG=32) reduce + scratch (nrows=NQH), combining the per-
  // workgroup partials into bf16 out_.
  const FC reduce_fc[] = {{500, FC::Kind::kInt, HD},
                          {501, FC::Kind::kInt, kNwgHC}};
  TF_ASSIGN_OR_RETURN(fa_reduce_,
                      metal_exec->LoadKernelWithConstants(
                          favec_lib_, "fa_vec_reduce", /*arity=*/3, reduce_fc));
  const int64_t tmp_floats = NQH * HD * kNwgHC + NQH * 2 * kNwgHC;
  if (tmp_.opaque() == nullptr)
    tmp_ = executor->Allocate(tmp_floats * sizeof(float), 0);
  if (kargs_reduce_.opaque() == nullptr)
    kargs_reduce_ = executor->Allocate(sizeof(FaVecReduceKArgs), 0);
  if (tmp_.opaque() == nullptr || kargs_reduce_.opaque() == nullptr) {
    return absl::ResourceExhaustedError("Metal FA-vec HC: scratch alloc failed.");
  }
  FaVecReduceKArgs ar = {};
  ar.nrows = static_cast<int32_t>(NQH);
  TF_RETURN_IF_ERROR(
      executor->SynchronousMemcpy(&kargs_reduce_, &ar, sizeof(ar)));
  return absl::OkStatus();
}

absl::Status MetalFlashAttnThunk::EnsurePrefill(se::StreamExecutor* executor) {
  if (fa_prefill_ != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  const int64_t NQH = n_kv_ * n_groups_;
  const int64_t HD = head_dim_;
  const int64_t S = seqlen_;
  const int64_t q_len = q_shape_.dimensions(1);

  // Same K/V feed-layout derivation as the decode path: position-major
  // ([..,seqlen,n_kv,hd]) vs head-major ([n_kv,seqlen,hd]); strides swap.
  const int64_t kv_pos_stride = kv_position_major_ ? n_kv_ * HD : HD;   // elems
  const int64_t kv_head_stride = kv_position_major_ ? HD : S * HD;      // elems
  kv_pos_stride_ = kv_pos_stride;

  // Both the metallib and the PSO are normally already warm here (prewarmed at
  // HLO-compile time by MetalGpuCompiler::RunHloPasses → PrewarmPipeline); these
  // calls then hit the process metallib cache + Apple's driver pipeline cache.
  TF_ASSIGN_OR_RETURN(prefill_lib_, CompileMetalSourceToMetallibCached(get_flash_attn_prefill()));
  using FC = se::metal::MetalFunctionConstant;
  // FC_tokclamp (432) gates the on-GPU row clamp: when num_tokens is present the
  // kernel takes it at buffer(7) (arity 8) and early-returns Q-row blocks past the
  // real prompt length, so the host dispatches the full grid with no host-side
  // device read (the GDN-prefill race class). Without num_tokens: legacy full grid.
  const FC fc[] = {{430, FC::Kind::kInt, kv_pos_stride},
                   {431, FC::Kind::kInt, kv_pos_stride},
                   {432, FC::Kind::kBool, has_num_tokens_ ? 1 : 0}};
  TF_ASSIGN_OR_RETURN(fa_prefill_,
                      metal_exec->LoadKernelWithConstants(
                          prefill_lib_, "fa_ext",
                          /*arity=*/has_num_tokens_ ? 8 : 7, fc));
  prefill_smem_ = 10240;  // FATTN_SMEM(nsg=4) for hd128: 8*(128+2*128+2*128)*2

  if (!kv_full_cache_ && zero_layer_.opaque() == nullptr) {
    zero_layer_ = executor->Allocate(sizeof(uint32_t), 0);
    if (zero_layer_.opaque() == nullptr) {
      return absl::ResourceExhaustedError(
          "Metal FA prefill: zero-layer alloc failed.");
    }
    const uint32_t zero = 0;
    TF_RETURN_IF_ERROR(
        executor->SynchronousMemcpy(&zero_layer_, &zero, sizeof(zero)));
  }

  kargs_prefill_ = executor->Allocate(sizeof(FaVecKArgs), 0);
  if (kargs_prefill_.opaque() == nullptr) {
    return absl::ResourceExhaustedError("Metal FA prefill: kargs alloc failed.");
  }
  // ggml_metal_kargs_flash_attn_ext == FaVecKArgs layout. Prefill: ne01/ne1 = q_len
  // (the query rows), q strides for row-major [NQH, q_len, hd]; ne11 = S (KV loop
  // ceiling); causality keys off the on-device token_index. (FaVecKArgs is reused
  // because the struct layout matches the ext kargs exactly.)
  FaVecKArgs a = {};
  a.ne01 = q_len; a.ne02 = NQH; a.ne03 = 1;
  // Q byte-strides from the operand's ACTUAL layout: head-major {2,1,0}
  // (nb01=HD, nb02=q_len*HD), OR the relaxed token-major {2,0,1} that folds the
  // RoPE transpose to a bitcast (nb01=NQH*HD, nb02=HD — see RelaxFlashAttnKVLayout).
  // dims: 0=NQH(head) 1=q_len(token) 2=hd; hd (dim2) is always inner. The kernel
  // reads Q purely through these strides, so no kernel change is needed.
  int64_t qstride[3] = {1, 1, 1};  // element strides per logical dim
  if (q_shape_.dimensions().size() == 3) {
    int64_t acc = 1;
    for (int64_t phys : q_shape_.layout().minor_to_major()) {
      qstride[phys] = acc;
      acc *= q_shape_.dimensions(phys);
    }
  } else {
    qstride[1] = HD; qstride[0] = q_len * HD;  // fallback: head-major
  }
  a.nb01 = qstride[1] * 2; a.nb02 = qstride[0] * 2; a.nb03 = NQH * q_len * HD * 2;
  a.ne11 = S; a.ne_12_2 = n_kv_; a.ne_12_3 = 1;
  a.ns10 = kv_pos_stride; a.nb11 = kv_pos_stride * 2; a.nb12 = kv_head_stride * 2; a.nb13 = n_kv_ * S * HD * 2;
  a.ns20 = kv_pos_stride; a.nb21 = kv_pos_stride * 2; a.nb22 = kv_head_stride * 2; a.nb23 = n_kv_ * S * HD * 2;
  a.ne1 = q_len; a.ne2 = NQH; a.ne3 = 1;
  a.scale = 1.0f / std::sqrt(static_cast<float>(HD));
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&kargs_prefill_, &a, sizeof(a)));
  return absl::OkStatus();
}

absl::Status MetalFlashAttnThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  const int64_t NQH = n_kv_ * n_groups_;  // total query heads
  const int64_t HD = head_dim_;
  const int64_t S = seqlen_;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor) {
    kernel_ = nullptr;
    for (auto& k : fa_main_by_nsg_) k = nullptr;
    fa_main_hc_ = nullptr;
    fa_reduce_ = nullptr;
    fa_prefill_ = nullptr;

    const int64_t q_len = q_shape_.dimensions(1);
    // q_len>1 = prefill (the simdgroup-matrix kernel_flash_attn_ext, gated on
    // seqlen%64==0 so no kvpad pre-pass is needed). q_len==1 = decode (fa_vec).
    use_prefill_ = (q_len > 1 && head_dim_ == 128 && seqlen_ % 64 == 0 &&
                    q_shape_.element_type() == BF16 &&
                    out_shape_.element_type() == BF16);
    use_fa_vec_ = (q_len == 1 && head_dim_ == 128 && seqlen_ % 32 == 0 &&
                   q_shape_.element_type() == BF16 &&
                   out_shape_.element_type() == BF16);

    if (use_prefill_) {
      TF_RETURN_IF_ERROR(EnsurePrefill(executor));
      executor_ = executor;
    } else if (!use_fa_vec_) {
      TF_ASSIGN_OR_RETURN(
          std::vector<uint8_t> metallib,
          CompileMetalSourceToMetallib(get_flash_attn_serial(),
                               {{"__N_GROUPS__", absl::StrCat(n_groups_)},
                                {"__SEQLEN__", absl::StrCat(seqlen_)},
                                {"__HD__", absl::StrCat(head_dim_)}}));
      auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
          std::move(metallib), "flash_attn_vec", /*arity=*/5);
      TF_ASSIGN_OR_RETURN(kernel_, executor->LoadKernel(spec));
      executor_ = executor;
    } else {
      // K/V cache feed layout: head-major [n_kv,S,hd] (operand {2,1,0}) vs
      // position-major [S,n_kv,hd] (operand {2,0,1}, the cache-native slice — the
      // ZML transpose then folds to a free bitcast, no per-token KV copy). The
      // position stride and kv-head stride swap between the two layouts.
      const int64_t kv_pos_stride = kv_position_major_ ? n_kv_ * HD : HD;  // elems
      const int64_t kv_head_stride = kv_position_major_ ? HD : S * HD;     // elems
      kv_pos_stride_ = kv_pos_stride;

      // Compile the fa_vec metallib ONCE (the expensive xcrun step) and keep it;
      // pipeline variants are built lazily from it on first selection (see
      // EnsureFaVec*). Build only the nsg=4 nwg=1 variant now, so a small-context
      // decode pays the same single-PSO first-token cost as before — no startup
      // regression — while larger contexts amortize a variant's one-time build.
      TF_ASSIGN_OR_RETURN(favec_lib_, CompileMetalSourceToMetallibCached(get_flash_attn_vec()));
      TF_RETURN_IF_ERROR(EnsureFaVecMain(executor, /*idx=*/2));  // kNsgVals[2]=4

      if (!kv_full_cache_) {
        // Sliced fallback: the operand is a single layer, so bind a layer index
        // of 0 (the kernel's layer offset becomes a no-op).
        zero_layer_ = executor->Allocate(sizeof(uint32_t), 0);
        if (zero_layer_.opaque() == nullptr) {
          return absl::ResourceExhaustedError(
              "Metal FA-vec: zero-layer alloc failed.");
        }
        const uint32_t zero = 0;
        TF_RETURN_IF_ERROR(
            executor->SynchronousMemcpy(&zero_layer_, &zero, sizeof(zero)));
      }

      kargs_ = executor->Allocate(sizeof(FaVecKArgs), 0);
      if (kargs_.opaque() == nullptr) {
        return absl::ResourceExhaustedError("Metal FA-vec: kargs alloc failed.");
      }
      // kargs is fully static (strides/dims fixed; ne11=S is the loop ceiling,
      // causality keys off the device token_index, a separate operand).
      FaVecKArgs a = {};
      a.ne01 = 1; a.ne02 = NQH; a.ne03 = 1;
      a.nb01 = NQH * HD * 2; a.nb02 = HD * 2; a.nb03 = NQH * HD * 2;  // bf16 q
      a.ne11 = S; a.ne_12_2 = n_kv_; a.ne_12_3 = 1;
      a.ns10 = kv_pos_stride; a.nb11 = kv_pos_stride * 2; a.nb12 = kv_head_stride * 2; a.nb13 = n_kv_ * S * HD * 2;
      a.ns20 = kv_pos_stride; a.nb21 = kv_pos_stride * 2; a.nb22 = kv_head_stride * 2; a.nb23 = n_kv_ * S * HD * 2;
      a.ne1 = 1; a.ne2 = NQH; a.ne3 = 1;
      a.scale = 1.0f / std::sqrt(static_cast<float>(HD));
      TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&kargs_, &a, sizeof(a)));
      executor_ = executor;
    }
  }

  // PREFILL (q_len>1): single dispatch of the simdgroup-matrix kernel. Parallelism
  // is over the Q-row blocks (grid dim0 = ceil(q_len/8)) x heads x batch; the kernel
  // derives the 2D causal mask on-device. No host token read, no nsg ramp.
  if (use_prefill_) {
    const int64_t q_len = q_shape_.dimensions(1);
    // Clamp the query-row grid to the real prompt length ON-GPU. The host dispatches
    // the FULL static grid (ceil(q_len/8) Q-row blocks); when num_tokens is present
    // the kernel (compiled with FC_tokclamp) reads it from buffer(7) and early-returns
    // any block entirely past num_tokens, so the padded rows [num_tokens, q_len) skip
    // cheaply (they are pure waste — the model is causal + position-wise, so they never
    // feed the real output). This replaces the old host-side encode-time read of
    // num_tokens, which on Metal raced the GPU producer of that metadata (embed emits
    // it as a non-aliased copy = an async D2D blit; there is no totally-ordered host/GPU
    // path, so the host read could land before the blit and shrink the grid wrong,
    // leaving tail-sequence rows unwritten — the GDN-prefill garbage-token race class).
    // Reading num_tokens on the GPU makes it an ordinary buffer dependency, like CUDA.
    const se::DeviceAddressBase layer_arg =
        kv_full_cache_ ? allocs.GetDeviceAddress(layer_) : zero_layer_;
    se::KernelArgsPackedArray args(has_num_tokens_ ? 8 : 7);
    args.add_argument(kargs_prefill_);
    args.add_argument(allocs.GetDeviceAddress(q_));
    args.add_argument(allocs.GetDeviceAddress(k_));
    args.add_argument(allocs.GetDeviceAddress(v_));
    args.add_argument(allocs.GetDeviceAddress(tok_));
    args.add_argument(layer_arg);
    args.add_argument(allocs.GetDeviceAddress(out_));
    if (has_num_tokens_) {
      args.add_argument(allocs.GetDeviceAddress(num_tokens_));  // buffer(7)
    }
    args.add_shared_bytes(prefill_smem_);
    return fa_prefill_->Launch(se::ThreadDim(32, 4, 1),
                               se::BlockDim((q_len + 7) / 8, NQH, 1), stream, args);
  }

  if (!use_fa_vec_) {
    // The serial fallback is single-query; a prefill (q_len>1) that didn't qualify
    // for the matrix kernel is unsupported (loud, never silently wrong).
    if (q_shape_.dimensions(1) != 1) {
      return absl::UnimplementedError(
          "Metal FA: prefill (q_len>1) needs head_dim==128, bf16 q/out, "
          "seqlen%64==0.");
    }
    se::KernelArgsPackedArray kernel_args(/*num_args=*/5);
    kernel_args.add_argument(allocs.GetDeviceAddress(q_));
    kernel_args.add_argument(allocs.GetDeviceAddress(k_));
    kernel_args.add_argument(allocs.GetDeviceAddress(v_));
    kernel_args.add_argument(allocs.GetDeviceAddress(tok_));
    kernel_args.add_argument(allocs.GetDeviceAddress(out_));
    return kernel_->Launch(se::ThreadDim(32, 1, 1),
                           se::BlockDim(n_groups_, n_kv_, 1), stream,
                           kernel_args);
  }

  // LIVE KV length, to size the variant for this step's context (the HC gate + the
  // nsg ramp). This is PERF-ONLY: the kernel's KV-loop ceiling is the static seqlen
  // (kargs.ne11) and causality comes from the on-device tok[0], so the variant is
  // numerically identical regardless of kv — a wrong kv only mis-sizes it (slower),
  // never produces wrong output. We read tok host-side ONLY when it is host-coherent
  // (a host-staged entry parameter — RelaxFlashAttnKVLayout substituted the raw
  // token-position param for ZML's device convert). When it is NOT (the layout-only
  // relax fallback / no raw param), the operand is GPU-produced and a host deref
  // would race the producer on Metal (no totally-ordered host/GPU path — the
  // GDN-prefill race class), so we DON'T read it and fall back to the static kv = S
  // (the conservative max-context variant). Never host-deref a GPU-producible buffer.
  int64_t kv = S;
  if (tok_host_coherent_) {
    const int32_t* tok_host =
        static_cast<const int32_t*>(allocs.GetDeviceAddress(tok_).opaque());
    if (tok_host != nullptr) kv = static_cast<int64_t>(*tok_host) + 1;
  }
  if (kv < 1) kv = 1;
  if (kv > S) kv = S;

  // HEAD-CONTIGUOUS at large depth: one simdgroup per KV head so the n_kv heads of a
  // position are read as ONE contiguous run out of the position-major cache (the
  // per-head fa_vec reads each head strided by n_kv*hd -> ~15% lower KV bandwidth at
  // depth). Positions split across 32 workgroups; the unchanged fa_vec_reduce combines
  // the partials. grid (1, n_groups, 32), threads (32, n_kv, 1). Gated kv > 4096:
  // below that the per-head ramp is already at llama parity and the split-K + reduce
  // overhead isn't worth it; above, the contiguous read closes the gap (d11212 bf16
  // 56.2 -> 58.2 tok/s == llama).
  constexpr int64_t kHcMinKv = 4096;
  if (kv > kHcMinKv) {
    TF_RETURN_IF_ERROR(EnsureFaVecHC(executor));
    const se::DeviceAddressBase layer_arg =
        kv_full_cache_ ? allocs.GetDeviceAddress(layer_) : zero_layer_;
    {
      se::KernelArgsPackedArray args(7);
      args.add_argument(kargs_);
      args.add_argument(allocs.GetDeviceAddress(q_));
      args.add_argument(allocs.GetDeviceAddress(k_));
      args.add_argument(allocs.GetDeviceAddress(v_));
      args.add_argument(allocs.GetDeviceAddress(tok_));
      args.add_argument(layer_arg);
      args.add_argument(tmp_);
      args.add_shared_bytes(smem_hc_);
      TF_RETURN_IF_ERROR(fa_main_hc_->Launch(se::ThreadDim(32, n_kv_, 1),
                                             se::BlockDim(1, n_groups_, 32), stream,
                                             args));
    }
    {
      se::KernelArgsPackedArray args(3);
      args.add_argument(kargs_reduce_);
      args.add_argument(tmp_);
      args.add_argument(allocs.GetDeviceAddress(out_));
      TF_RETURN_IF_ERROR(fa_reduce_->Launch(se::ThreadDim(32 * 32, 1, 1),
                                            se::BlockDim(NQH, 1, 1), stream, args));
    }
    return absl::OkStatus();
  }

  // PER-EXECUTE nsg ramp (single dispatch, nwg=1): more simdgroups as context
  // grows. ne11=S (kargs) stays the loop ceiling. (kv > kHcMinKv took HC above.)
  int nsg = kv <= 1024 ? 4 : (kv <= 2048 ? 8 : 16);
  // Cap nsg by the device's threadgroup-memory budget instead of assuming 32 KB:
  // FA-vec smem is linear in nsg, so pick the largest nsg that fits. On every
  // shipping Apple GPU (32 KB) all of {4,8,16} fit -> byte-identical; on a
  // hypothetical smaller-smem part this self-limits rather than over-allocating
  // threadgroup memory. The host picker keys off the queried device limit
  // (set_shared_memory_per_block); the kernel stays function-constant driven.
  const int64_t tg_mem_limit =
      executor->GetDeviceDescription().shared_memory_per_block();
  while (nsg > 1 && tg_mem_limit > 0 &&
         static_cast<int64_t>(FaVecSmem(head_dim_, nsg)) > tg_mem_limit) {
    nsg /= 2;
  }

  const se::DeviceAddressBase layer_arg =
      kv_full_cache_ ? allocs.GetDeviceAddress(layer_) : zero_layer_;

  // Snap nsg to the nearest built variant (<= nsg), write bf16 directly to out_.
  // Grid (1, NQH, 1), threads (32, nsg, 1).
  int idx = 0;
  for (int i = 0; i < 5; ++i)
    if (kNsgVals[i] <= nsg) idx = i;
  TF_RETURN_IF_ERROR(EnsureFaVecMain(executor, idx));
  se::KernelArgsPackedArray args(7);
  args.add_argument(kargs_);
  args.add_argument(allocs.GetDeviceAddress(q_));
  args.add_argument(allocs.GetDeviceAddress(k_));
  args.add_argument(allocs.GetDeviceAddress(v_));
  args.add_argument(allocs.GetDeviceAddress(tok_));
  args.add_argument(layer_arg);
  args.add_argument(allocs.GetDeviceAddress(out_));
  args.add_shared_bytes(smem_by_nsg_[idx]);
  return fa_main_by_nsg_[idx]->Launch(se::ThreadDim(32, kNsgVals[idx], 1),
                                      se::BlockDim(1, NQH, 1), stream, args);
}

Thunk::BufferUses MetalFlashAttnThunk::buffer_uses() const {
  Thunk::BufferUses uses = {
      BufferUse::Read(q_, q_shape_),     BufferUse::Read(k_, k_shape_),
      BufferUse::Read(v_, v_shape_),     BufferUse::Read(tok_, tok_shape_),
      BufferUse::Write(out_, out_shape_),
  };
  if (kv_full_cache_) {
    uses.push_back(BufferUse::Read(layer_, layer_shape_));
  }
  if (has_num_tokens_) {
    uses.push_back(BufferUse::Read(num_tokens_, num_tokens_shape_));
  }
  return uses;
}

absl::StatusOr<ThunkProto> MetalFlashAttnThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalFlashAttnThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
