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

// Field-for-field replica of the kernel's fa_vec_args (buffer 0); the
// layout is the ABI.
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

static std::vector<std::pair<std::string, std::string>> HeadDimSubs(int64_t hd) {
  return {{"__DK__", absl::StrCat(hd)}, {"__DV__", absl::StrCat(hd)}};
}
static size_t PrefillSmem(int64_t hd) {
  const int64_t pv = ((hd + 63) / 64) * 64;  // PAD2(hd, 64)
  return static_cast<size_t>(8 * (hd + 2 * pv + 256) * 2);
}

void MetalFlashAttnThunk::PrewarmPipeline(se::StreamExecutor* executor,
                                         bool is_prefill, int64_t kv_pos_stride,
                                         int64_t seqlen, int64_t head_dim) {
  auto lib = is_prefill ? CompileMetalSourceToMetallibCached(
                              get_flash_attn_prefill(), HeadDimSubs(head_dim))
                        : CompileMetalSourceToMetallibCached(get_flash_attn_vec(),
                                                             HeadDimSubs(head_dim));
  if (!lib.ok()) return;
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  using FC = se::metal::MetalFunctionConstant;
  if (is_prefill) {
    const char* prefill_kernel = "fa_ext";
    for (int clamp = 0; clamp <= 1; ++clamp) {
      const FC fc[] = {{430, FC::Kind::kInt, kv_pos_stride},
                       {431, FC::Kind::kInt, kv_pos_stride},
                       {432, FC::Kind::kBool, clamp}};
      metal_exec
          ->LoadKernelWithConstants(*lib, prefill_kernel, /*arity=*/clamp ? 8 : 7,
                                    fc)
          .IgnoreError();
    }
    return;
  }
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
  const FC fc[] = {{420, FC::Kind::kInt, kv_pos_stride_},
                   {421, FC::Kind::kInt, kv_pos_stride_},
                   {422, FC::Kind::kInt, n_kv_},
                   {423, FC::Kind::kInt, kNwgHC}};
  TF_ASSIGN_OR_RETURN(fa_main_hc_,
                      metal_exec->LoadKernelWithConstants(
                          favec_lib_, "fa_vec_hc", /*arity=*/7, fc));
  smem_hc_ = FaVecSmem(HD, static_cast<int>(n_kv_));
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

  const int64_t kv_pos_stride = kv_position_major_ ? n_kv_ * HD : HD;   // elems
  const int64_t kv_head_stride = kv_position_major_ ? HD : S * HD;      // elems
  kv_pos_stride_ = kv_pos_stride;

  TF_ASSIGN_OR_RETURN(prefill_lib_, CompileMetalSourceToMetallibCached(
                                        get_flash_attn_prefill(), HeadDimSubs(HD)));
  using FC = se::metal::MetalFunctionConstant;
  const FC fc[] = {{430, FC::Kind::kInt, kv_pos_stride},
                   {431, FC::Kind::kInt, kv_pos_stride},
                   {432, FC::Kind::kBool, has_num_tokens_ ? 1 : 0}};
  TF_ASSIGN_OR_RETURN(fa_prefill_,
                      metal_exec->LoadKernelWithConstants(
                          prefill_lib_, "fa_ext",
                          /*arity=*/has_num_tokens_ ? 8 : 7, fc));
  prefill_smem_ = PrefillSmem(HD);  // hd128 -> 10240, hd64 -> 7168

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
  FaVecKArgs a = {};
  a.ne01 = q_len; a.ne02 = NQH; a.ne03 = 1;
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
    for (auto& k : fa_main_by_nsg_) k = nullptr;
    fa_main_hc_ = nullptr;
    fa_reduce_ = nullptr;
    fa_prefill_ = nullptr;

    const int64_t q_len = q_shape_.dimensions(1);
    use_prefill_ = (q_len > 1 && head_dim_ % 16 == 0 && seqlen_ % 64 == 0 &&
                    q_shape_.element_type() == BF16 &&
                    out_shape_.element_type() == BF16);
    use_fa_vec_ = (q_len == 1 && head_dim_ % 16 == 0 && seqlen_ % 32 == 0 &&
                   q_shape_.element_type() == BF16 &&
                   out_shape_.element_type() == BF16);

    if (use_prefill_) {
      TF_RETURN_IF_ERROR(EnsurePrefill(executor));
      executor_ = executor;
    } else if (!use_fa_vec_) {
      return absl::UnimplementedError(absl::StrCat(
          "Metal flash-attention: unsupported shape (head_dim=", head_dim_,
          ", seqlen=", seqlen_, ", q_len=", q_shape_.dimensions(1),
          ", q_dtype=", static_cast<int>(q_shape_.element_type()),
          "). Need head_dim%16==0, bf16, seqlen%64==0 (prefill) / %32==0 (decode)."));
    } else {
      const int64_t kv_pos_stride = kv_position_major_ ? n_kv_ * HD : HD;  // elems
      const int64_t kv_head_stride = kv_position_major_ ? HD : S * HD;     // elems
      kv_pos_stride_ = kv_pos_stride;

      TF_ASSIGN_OR_RETURN(favec_lib_, CompileMetalSourceToMetallibCached(
                                          get_flash_attn_vec(), HeadDimSubs(head_dim_)));
      TF_RETURN_IF_ERROR(EnsureFaVecMain(executor, /*idx=*/2));  // kNsgVals[2]=4

      if (!kv_full_cache_) {
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

  if (use_prefill_) {
    const int64_t q_len = q_shape_.dimensions(1);
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
    return absl::UnimplementedError(
        "Metal flash-attention: unsupported shape (need head_dim%16==0, bf16, "
        "aligned seqlen).");
  }

  int64_t kv = S;
  if (tok_host_coherent_) {
    const int32_t* tok_host =
        static_cast<const int32_t*>(allocs.GetDeviceAddress(tok_).opaque());
    if (tok_host != nullptr) kv = static_cast<int64_t>(*tok_host) + 1;
  }
  if (kv < 1) kv = 1;
  if (kv > S) kv = S;

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

  int nsg = kv <= 1024 ? 4 : (kv <= 2048 ? 8 : 16);
  const int64_t tg_mem_limit =
      executor->GetDeviceDescription().shared_memory_per_block();
  while (nsg > 1 && tg_mem_limit > 0 &&
         static_cast<int64_t>(FaVecSmem(head_dim_, nsg)) > tg_mem_limit) {
    nsg /= 2;
  }

  const se::DeviceAddressBase layer_arg =
      kv_full_cache_ ? allocs.GetDeviceAddress(layer_) : zero_layer_;

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
