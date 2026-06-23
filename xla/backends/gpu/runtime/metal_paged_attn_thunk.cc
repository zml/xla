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

#include "xla/backends/gpu/runtime/metal_paged_attn_thunk.h"

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
#include "xla/service/gpu/metal_kernels/paged_attn_vec.h"
#include "xla/service/gpu/metal_kernels/pagedattention_tiled.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
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

// The exact source-concat preamble pagedattention_tiled.metal declares it needs
// (see its header comment "Requires utils.metal (DIVIDE_ROUND_UP, MIN, MAX),
// <metal_stdlib>, and `using namespace metal`"). Prepended to the VERBATIM
// vendored kernel; the kernel body itself is unmodified.
constexpr const char* kPreamble = R"PRE(
#include <metal_stdlib>
using namespace metal;
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
typedef bfloat bfloat16_t;
)PRE";

// FATTN_SMEM(nsg) (half elements -> bytes), same as the FA thunk's helper.
size_t PagedFaVecSmem(int64_t head_dim, int nsg) {
  auto pad = [](int64_t x, int64_t n) { return ((x + n - 1) / n) * n; };
  return static_cast<size_t>(
      pad((pad(head_dim, 128) + 4 * 32 + 2 * pad(head_dim, 128)) * nsg * 2, 16));
}

// (BQ, TILE_KV, NUM_THREADS) per head_size — must match select_tile_config in
// vllm's paged_ops.cpp and the instantiations at pagedattention_tiled.metal:543.
struct TileConfig {
  int bq, tile_kv, num_threads;
};
absl::StatusOr<TileConfig> TileConfigForHeadSize(int64_t head_size) {
  switch (head_size) {
    case 64:
    case 96:
    case 128:
      return TileConfig{32, 32, 128};
    case 256:
      return TileConfig{16, 16, 64};
    case 512:
      return TileConfig{8, 8, 32};
    default:
      return absl::UnimplementedError(absl::StrCat(
          "zml$paged_attn: head_size ", head_size,
          " unsupported (tiled kernel: 64/96/128/256/512)."));
  }
}

// Host mirror of the kernel's fa_vec_paged_args (constant buffer 0).
struct FaVecPagedKArgs {
  int32_t num_heads;
  int32_t n_kv;
  int32_t max_blocks;
  int32_t num_seqs;
  float scale;
};

absl::StatusOr<const char*> DtypeName(PrimitiveType t) {
  switch (t) {
    case BF16:
      return "bfloat16_t";
    case F16:
      return "half";
    default:
      return absl::UnimplementedError(
          "zml$paged_attn: only bf16/f16 q/k/v/out supported.");
  }
}

}  // namespace

MetalPagedAttnThunk::MetalPagedAttnThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice q, Shape q_shape,
    BufferAllocation::Slice k_cache, Shape k_cache_shape,
    BufferAllocation::Slice v_cache, Shape v_cache_shape,
    BufferAllocation::Slice block_table, Shape block_table_shape,
    BufferAllocation::Slice seq_lens, Shape seq_lens_shape,
    BufferAllocation::Slice query_start_len, Shape query_start_len_shape,
    BufferAllocation::Slice out, Shape out_shape, int64_t num_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t block_size, int64_t num_seqs,
    int64_t max_num_blocks_per_seq, int64_t total_q_tokens, float scale,
    float softcapping, int sliding_window, PrimitiveType element_type)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      q_(q),
      k_cache_(k_cache),
      v_cache_(v_cache),
      block_table_(block_table),
      seq_lens_(seq_lens),
      query_start_len_(query_start_len),
      out_(out),
      q_shape_(std::move(q_shape)),
      k_cache_shape_(std::move(k_cache_shape)),
      v_cache_shape_(std::move(v_cache_shape)),
      block_table_shape_(std::move(block_table_shape)),
      seq_lens_shape_(std::move(seq_lens_shape)),
      query_start_len_shape_(std::move(query_start_len_shape)),
      out_shape_(std::move(out_shape)),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      block_size_(block_size),
      num_seqs_(num_seqs),
      max_num_blocks_per_seq_(max_num_blocks_per_seq),
      total_q_tokens_(total_q_tokens),
      scale_(scale),
      softcapping_(softcapping),
      sliding_window_(sliding_window),
      element_type_(element_type) {}

void MetalPagedAttnThunk::Prewarm(se::StreamExecutor* executor,
                                  PrimitiveType dtype, int64_t head_dim,
                                  int64_t block_size, int64_t num_kv_heads) {
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  // Tiled prefill kernel — same metallib + kernel name EnsureLoaded builds.
  auto cfg = TileConfigForHeadSize(head_dim);
  auto dt = DtypeName(dtype);
  if (cfg.ok() && dt.ok() &&
      (block_size == 8 || block_size == 16 || block_size == 32)) {
    const std::string src = std::string(kPreamble) + get_pagedattention_tiled();
    auto lib = CompileMetalSourceToMetallibCached(src);
    if (lib.ok()) {
      std::string name = absl::StrCat(
          "paged_attention_tiled_", *dt, "_hs", head_dim, "_bs", block_size,
          "_bq", cfg->bq, "_tk", cfg->tile_kv, "_nt", cfg->num_threads);
      metal_exec->LoadKernelWithConstants(*lib, name, /*arity=*/22, {})
          .IgnoreError();
    }
  }
  // Decode vector kernel (when it qualifies): warm ALL THREE nsg variants.
  if (head_dim == 128 && dtype == BF16) {
    auto vec_lib = CompileMetalSourceToMetallibCached(get_paged_attn_vec());
    if (vec_lib.ok()) {
      using FC = se::metal::MetalFunctionConstant;
      for (int nsg : kVecNsgVals) {
        const FC fc[] = {{450, FC::Kind::kInt, nsg},
                         {451, FC::Kind::kInt, block_size},
                         {452, FC::Kind::kInt, num_kv_heads * head_dim}};
        metal_exec
            ->LoadKernelWithConstants(*vec_lib, "fa_vec_paged", /*arity=*/8, fc)
            .IgnoreError();
      }
    }
  }
}

absl::Status MetalPagedAttnThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (kernel_ != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);

  TF_ASSIGN_OR_RETURN(TileConfig cfg, TileConfigForHeadSize(head_dim_));
  TF_ASSIGN_OR_RETURN(const char* dt, DtypeName(element_type_));
  if (block_size_ != 8 && block_size_ != 16 && block_size_ != 32) {
    return absl::UnimplementedError(absl::StrCat(
        "zml$paged_attn: block_size ", block_size_, " unsupported (8/16/32)."));
  }
  bq_ = cfg.bq;
  num_threads_ = cfg.num_threads;
  total_q_blocks_ = total_q_tokens_ / bq_ + num_seqs_;

  // Kernel name: paged_attention_tiled_{dt}_hs{H}_bs{B}_bq{BQ}_tk{TK}_nt{NT}.
  std::string name = absl::StrCat("paged_attention_tiled_", dt, "_hs", head_dim_,
                                  "_bs", block_size_, "_bq", cfg.bq, "_tk",
                                  cfg.tile_kv, "_nt", cfg.num_threads);

  const std::string src = std::string(kPreamble) + get_pagedattention_tiled();
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib, CompileMetalSourceToMetallibCached(src));
  // 22 buffer args (indices 0..21, the verbatim sparse ABI's max + 1) + the
  // threadgroup-memory arg; arity counts buffer args only.
  TF_ASSIGN_OR_RETURN(
      kernel_, metal_exec->LoadKernelWithConstants(lib, name, /*arity=*/22, {}));

  // smem = (BQ + 2*TILE_KV) * (HEAD_SIZE + SMEM_PAD) * sizeof(T); SMEM_PAD =
  // 16/sizeof(T) = 8 elems for fp16/bf16. (pagedattention_tiled.metal:189-192.)
  const int elem_size = 2;  // bf16 / f16
  const int smem_pad = 16 / elem_size;
  const int ld = static_cast<int>(head_dim_) + smem_pad;
  shmem_bytes_ = static_cast<size_t>(cfg.bq + 2 * cfg.tile_kv) * ld * elem_size;

  // Stage the static scalar params into individual device buffers (the kernel
  // takes them as inline `constant&` args at fixed sparse buffer indices).
  auto stage = [&](se::DeviceAddressBase& dst, const void* val,
                   size_t n) -> absl::Status {
    dst = executor->Allocate(n, 0);
    if (dst.opaque() == nullptr) {
      return absl::ResourceExhaustedError("zml$paged_attn: scalar alloc failed.");
    }
    return executor->SynchronousMemcpy(&dst, val, n);
  };
  const int32_t v_num_kv_heads = static_cast<int32_t>(num_kv_heads_);
  const float v_scale = scale_;
  const float v_softcap = softcapping_;
  const int32_t v_max_blocks = static_cast<int32_t>(max_num_blocks_per_seq_);
  const int32_t v_q_stride = static_cast<int32_t>(num_heads_ * head_dim_);
  const int32_t v_kv_block_stride =
      static_cast<int32_t>(block_size_ * num_kv_heads_ * head_dim_);
  const int32_t v_kv_head_stride = static_cast<int32_t>(head_dim_);
  const int32_t v_num_seqs = static_cast<int32_t>(num_seqs_);
  const int32_t v_sliding_window = sliding_window_;
  TF_RETURN_IF_ERROR(stage(p_num_kv_heads_, &v_num_kv_heads, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(stage(p_scale_, &v_scale, sizeof(float)));
  TF_RETURN_IF_ERROR(stage(p_softcapping_, &v_softcap, sizeof(float)));
  TF_RETURN_IF_ERROR(stage(p_max_blocks_, &v_max_blocks, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(stage(p_q_stride_, &v_q_stride, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(
      stage(p_kv_block_stride_, &v_kv_block_stride, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(
      stage(p_kv_head_stride_, &v_kv_head_stride, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(stage(p_num_seqs_, &v_num_seqs, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(
      stage(p_sliding_window_, &v_sliding_window, sizeof(int32_t)));

  dummy_ = executor->Allocate(16, 0);
  if (dummy_.opaque() == nullptr) {
    return absl::ResourceExhaustedError("zml$paged_attn: dummy alloc failed.");
  }

  return absl::OkStatus();
}

absl::Status MetalPagedAttnThunk::EnsureVecVariant(
    se::StreamExecutor* executor, int idx) {
  if (vec_kernel_by_nsg_[idx] != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  using FC = se::metal::MetalFunctionConstant;
  const int nsg = kVecNsgVals[idx];
  const FC fc[] = {{450, FC::Kind::kInt, nsg},
                   {451, FC::Kind::kInt, block_size_},
                   {452, FC::Kind::kInt, num_kv_heads_ * head_dim_}};
  TF_ASSIGN_OR_RETURN(vec_kernel_by_nsg_[idx],
                      metal_exec->LoadKernelWithConstants(
                          vec_lib_, "fa_vec_paged", /*arity=*/8, fc));
  vec_smem_by_nsg_[idx] = PagedFaVecSmem(head_dim_, nsg);
  return absl::OkStatus();
}

absl::Status MetalPagedAttnThunk::EnsureVecDecode(
    se::StreamExecutor* executor) {
  TF_ASSIGN_OR_RETURN(vec_lib_, CompileMetalSourceToMetallibCached(get_paged_attn_vec()));
  TF_RETURN_IF_ERROR(EnsureVecVariant(executor, /*idx=*/0));  // nsg=4

  p_vec_args_ = executor->Allocate(sizeof(FaVecPagedKArgs), 0);
  if (p_vec_args_.opaque() == nullptr) {
    return absl::ResourceExhaustedError("zml$paged_attn: vec kargs alloc failed.");
  }
  FaVecPagedKArgs a = {};
  a.num_heads = static_cast<int32_t>(num_heads_);
  a.n_kv = static_cast<int32_t>(num_kv_heads_);
  a.max_blocks = static_cast<int32_t>(max_num_blocks_per_seq_);
  a.num_seqs = static_cast<int32_t>(num_seqs_);
  a.scale = scale_;
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&p_vec_args_, &a, sizeof(a)));
  return absl::OkStatus();
}

absl::Status MetalPagedAttnThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor) {
    kernel_ = nullptr;
    for (auto& k : vec_kernel_by_nsg_) k = nullptr;
    // Kernel selection, mirroring vllm's has_prefill = total_q_tokens > num_seqs
    // (paged_ops.cpp): a pure-decode executable has exactly one query token per
    // sequence, for which the tiled kernel's BQ=32 MMA query tile is 31/32
    // padding — dispatch the vector kernel instead. Static per thunk (shapes are
    // compile-time). NOTE the static shape test can also match a SMALL CHUNKED-
    // PREFILL executable (llmd's chunk ladder emits q_tokens == num_seqs shapes
    // that carry multi-token chunks); the vector kernel is varlen-causal via
    // cu_seqlens_q, so that routing is still CORRECT — just vector-shaped. The
    // vector kernel is bf16/hd128-specialized and applies scale but NOT
    // softcapping / sliding-window masking; anything else stays on the tiled
    // kernel, which handles all of them correctly, just slower.
    // TODO: extend fa_vec_paged with sliding-window masking (Mistral/Gemma
    // decode) and an hd=256 variant (Gemma-shaped decode).
    use_vec_decode_ = total_q_tokens_ == num_seqs_ && head_dim_ == 128 &&
                      element_type_ == BF16 && softcapping_ == 0.0f &&
                      sliding_window_ < 0;
    if (use_vec_decode_) {
      TF_RETURN_IF_ERROR(EnsureVecDecode(executor));
    } else {
      TF_RETURN_IF_ERROR(EnsureLoaded(executor));
    }
    executor_ = executor;
  }

  // DECODE: fa_vec_paged, one threadgroup per (query head, sequence). Every
  // dispatched output row is unconditionally written (a kv_len==0 slot writes
  // zeros), so no MemZero is needed.
  //
  // nsg (the per-threadgroup KV split) is sized from the model's STATIC max KV
  // capacity (max_num_blocks_per_seq * block_size), NOT a host-side read of the
  // seq_lens device buffer. This is correctness-irrelevant: the kernel reads the
  // real per-seq length on-device from seq_lens[seq_idx] (paged_attn_vec.metal — the
  // KV-loop bound + causal mask), so nsg only sets how many simdgroups split that
  // work; the output is identical for any nsg. The previous code peeked seq_lens
  // host-side at encode to pick the smallest sufficient nsg, but in llmd continuous
  // batching that buffer can be GPU-produced, and a host deref of a GPU-producible
  // buffer races the producer on Metal (no totally-ordered host/GPU path — the
  // GDN-prefill race class). Sizing nsg from the static capacity keeps long-context
  // decode well-parallelized (the ramp's whole purpose); a short context just
  // over-parallelizes slightly. Matches CUDA's static-grid dispatch.
  if (use_vec_decode_) {
    const int64_t kv_capacity = max_num_blocks_per_seq_ * block_size_;
    const int idx =
        kv_capacity <= 1024 ? 0 : (kv_capacity <= 2048 ? 1 : 2);
    TF_RETURN_IF_ERROR(EnsureVecVariant(executor, idx));
    se::KernelArgsPackedArray args(/*num_args=*/8);
    args.add_argument(p_vec_args_);                            // 0 args
    args.add_argument(allocs.GetDeviceAddress(q_));            // 1 q
    args.add_argument(allocs.GetDeviceAddress(k_cache_));      // 2 k_cache
    args.add_argument(allocs.GetDeviceAddress(v_cache_));      // 3 v_cache
    args.add_argument(allocs.GetDeviceAddress(block_table_));  // 4 block_tables
    args.add_argument(allocs.GetDeviceAddress(seq_lens_));     // 5 seq_lens
    args.add_argument(allocs.GetDeviceAddress(query_start_len_));  // 6 cu_seqlens_q
    args.add_argument(allocs.GetDeviceAddress(out_));          // 7 dst
    args.add_shared_bytes(vec_smem_by_nsg_[idx]);
    return vec_kernel_by_nsg_[idx]->Launch(
        se::ThreadDim(32, kVecNsgVals[idx], 1),
        se::BlockDim(1, num_heads_, total_q_tokens_), stream, args);
  }
  // Verbatim sparse buffer ABI (pagedattention_tiled.metal:116-138): bind args
  // positionally 0..21; the kernel reads only its declared slots, dummy fills
  // the gaps (0,1,6,7,14,18). Threadgroup memory goes to threadgroup(0) via
  // add_shared_bytes.
  // The tiled kernel early-returns for padding query-blocks (q_pos >= query_len),
  // so it leaves the OUTPUT rows of padding tokens UNWRITTEN. On the first request
  // that buffer is fresh-zero, but on later requests it holds stale activations
  // that — propagated through the transformer's residual stream and o_proj/MLP —
  // blow up to NaN within a couple layers and (via tiled GEMM row mixing) corrupt
  // real tokens too (garbage "!" output from the 2nd request on). Zero the output
  // up front so padding rows are deterministically 0, matching the first-request
  // behavior. (MetalStream::MemZero drains then memsets the unified buffer, so the
  // subsequent kernel write is correctly ordered after it.)
  se::DeviceAddressBase out_addr = allocs.GetDeviceAddress(out_);
  TF_RETURN_IF_ERROR(stream->MemZero(&out_addr, out_addr.size()));

  se::KernelArgsPackedArray args(/*num_args=*/22);
  args.add_argument(dummy_);                              // 0  (unused)
  args.add_argument(dummy_);                              // 1  (unused)
  args.add_argument(out_addr);                            // 2  out
  args.add_argument(allocs.GetDeviceAddress(q_));         // 3  q
  args.add_argument(allocs.GetDeviceAddress(k_cache_));   // 4  k_cache
  args.add_argument(allocs.GetDeviceAddress(v_cache_));   // 5  v_cache
  args.add_argument(dummy_);                              // 6  (unused)
  args.add_argument(dummy_);                              // 7  (unused)
  args.add_argument(p_num_kv_heads_);                     // 8  num_kv_heads
  args.add_argument(p_scale_);                            // 9  scale
  args.add_argument(p_softcapping_);                      // 10 softcapping
  args.add_argument(allocs.GetDeviceAddress(block_table_));  // 11 block_tables
  args.add_argument(allocs.GetDeviceAddress(seq_lens_));     // 12 context_lens
  args.add_argument(p_max_blocks_);                      // 13 max_num_blocks_per_seq
  args.add_argument(dummy_);                              // 14 (unused)
  args.add_argument(p_q_stride_);                        // 15 q_stride
  args.add_argument(p_kv_block_stride_);                 // 16 kv_block_stride
  args.add_argument(p_kv_head_stride_);                  // 17 kv_head_stride
  args.add_argument(dummy_);                              // 18 (unused)
  args.add_argument(allocs.GetDeviceAddress(query_start_len_));  // 19 cu_seqlens_q
  args.add_argument(p_num_seqs_);                       // 20 num_seqs
  args.add_argument(p_sliding_window_);                 // 21 sliding_window
  args.add_shared_bytes(shmem_bytes_);

  // grid (threadgroups) = (num_heads, total_q_blocks, 1); threads/tg =
  // (NUM_THREADS, 1, 1). (pagedattention_tiled.metal grid; vllm paged_ops.cpp:288.)
  return kernel_->Launch(se::ThreadDim(num_threads_, 1, 1),
                         se::BlockDim(num_heads_, total_q_blocks_, 1), stream,
                         args);
}

Thunk::BufferUses MetalPagedAttnThunk::buffer_uses() const {
  return {
      BufferUse::Read(q_, q_shape_),
      BufferUse::Read(k_cache_, k_cache_shape_),
      BufferUse::Read(v_cache_, v_cache_shape_),
      BufferUse::Read(block_table_, block_table_shape_),
      BufferUse::Read(seq_lens_, seq_lens_shape_),
      BufferUse::Read(query_start_len_, query_start_len_shape_),
      BufferUse::Write(out_, out_shape_),
  };
}

absl::StatusOr<ThunkProto> MetalPagedAttnThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalPagedAttnThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
