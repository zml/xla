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

#include "xla/backends/gpu/runtime/metal_gdn_thunk.h"

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
#include "xla/service/gpu/metal_kernels/gdn_linear_attention.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

namespace {

// The vendored kernel includes <metal_stdlib> and `using namespace metal`
// itself, but instantiates a `bfloat16_t` template variant, so the source needs
// the `bfloat` -> `bfloat16_t` alias in scope (after `using namespace metal`).
// Prepended to the VERBATIM kernel; the body is unmodified.
constexpr const char* kPreamble = R"PRE(
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
)PRE";

absl::StatusOr<const char*> DtypeName(PrimitiveType t) {
  switch (t) {
    case F32:
      return "float";
    case F16:
      return "half";
    case BF16:
      return "bfloat16_t";
    default:
      return absl::UnimplementedError(
          "zml$gdn: only f32/f16/bf16 operands supported.");
  }
}

}  // namespace

MetalGdnThunk::MetalGdnThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice q, Shape q_shape,
    BufferAllocation::Slice k, Shape k_shape, BufferAllocation::Slice v,
    Shape v_shape, BufferAllocation::Slice g, Shape g_shape,
    BufferAllocation::Slice beta, Shape beta_shape, BufferAllocation::Slice h0,
    Shape h0_shape, BufferAllocation::Slice cu_seqlens, Shape cu_seqlens_shape,
    BufferAllocation::Slice slot_mapping, Shape slot_mapping_shape,
    BufferAllocation::Slice y, Shape y_shape, BufferAllocation::Slice ht,
    Shape ht_shape, int64_t num_seqs, int64_t hk, int64_t hv, int64_t dk,
    int64_t dv, PrimitiveType element_type)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      q_(q),
      k_(k),
      v_(v),
      g_(g),
      beta_(beta),
      h0_(h0),
      cu_seqlens_(cu_seqlens),
      slot_mapping_(slot_mapping),
      y_(y),
      ht_(ht),
      q_shape_(std::move(q_shape)),
      k_shape_(std::move(k_shape)),
      v_shape_(std::move(v_shape)),
      g_shape_(std::move(g_shape)),
      beta_shape_(std::move(beta_shape)),
      h0_shape_(std::move(h0_shape)),
      cu_seqlens_shape_(std::move(cu_seqlens_shape)),
      slot_mapping_shape_(std::move(slot_mapping_shape)),
      y_shape_(std::move(y_shape)),
      ht_shape_(std::move(ht_shape)),
      num_seqs_(num_seqs),
      hk_(hk),
      hv_(hv),
      dk_(dk),
      dv_(dv),
      element_type_(element_type) {}

absl::Status MetalGdnThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (kernel_ != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);

  TF_ASSIGN_OR_RETURN(const char* dt, DtypeName(element_type_));
  if (dk_ % 32 != 0 || dk_ > 256) {
    return absl::UnimplementedError(
        absl::StrCat("zml$gdn: Dk ", dk_, " must be a multiple of 32 and "
                                          "<= 256 (the kernel splits Dk over a "
                                          "32-thread SIMD group)."));
  }

  // Kernel name: gdn_linear_attention_{dtype}.
  const std::string name = absl::StrCat("gdn_linear_attention_", dt);
  const std::string src = std::string(kPreamble) + get_gdn_linear_attention();
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                      CompileMetalSourceToMetallibCached(src));
  // 14 buffer args: 0..8 device buffers (q,k,v,g,beta,state_pool,cu_seqlens,
  // slot_mapping,y) + 9..13 inline `constant int&` scalars.
  TF_ASSIGN_OR_RETURN(
      kernel_, metal_exec->LoadKernelWithConstants(lib, name, /*arity=*/14, {}));

  // Stage the five scalar params into individual device buffers.
  auto stage = [&](se::DeviceAddressBase& dst, const void* val,
                   size_t n) -> absl::Status {
    dst = executor->Allocate(n, 0);
    if (dst.opaque() == nullptr) {
      return absl::ResourceExhaustedError("zml$gdn: scalar alloc failed.");
    }
    return executor->SynchronousMemcpy(&dst, val, n);
  };
  const int32_t v_num_requests = static_cast<int32_t>(num_seqs_);
  const int32_t v_hk = static_cast<int32_t>(hk_);
  const int32_t v_hv = static_cast<int32_t>(hv_);
  const int32_t v_dk = static_cast<int32_t>(dk_);
  const int32_t v_dv = static_cast<int32_t>(dv_);
  TF_RETURN_IF_ERROR(stage(p_num_requests_, &v_num_requests, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(stage(p_hk_, &v_hk, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(stage(p_hv_, &v_hv, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(stage(p_dk_, &v_dk, sizeof(int32_t)));
  TF_RETURN_IF_ERROR(stage(p_dv_, &v_dv, sizeof(int32_t)));

  return absl::OkStatus();
}

absl::Status MetalGdnThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor) {
    kernel_ = nullptr;
    TF_RETURN_IF_ERROR(EnsureLoaded(executor));
    executor_ = executor;
  }

  // ht aliases h0, so the state-pool buffer already holds the gathered initial
  // state; the kernel reads it, runs the recurrence, and overwrites it in place.
  se::DeviceAddressBase state_pool = allocs.GetDeviceAddress(ht_);

  se::KernelArgsPackedArray args(/*num_args=*/14);
  args.add_argument(allocs.GetDeviceAddress(q_));             // 0  q
  args.add_argument(allocs.GetDeviceAddress(k_));             // 1  k
  args.add_argument(allocs.GetDeviceAddress(v_));             // 2  v
  args.add_argument(allocs.GetDeviceAddress(g_));             // 3  g (exp decay)
  args.add_argument(allocs.GetDeviceAddress(beta_));          // 4  beta
  args.add_argument(state_pool);                             // 5  state_pool
  args.add_argument(allocs.GetDeviceAddress(cu_seqlens_));    // 6  cu_seqlens
  args.add_argument(allocs.GetDeviceAddress(slot_mapping_));  // 7  slot_mapping
  args.add_argument(allocs.GetDeviceAddress(y_));             // 8  y
  args.add_argument(p_num_requests_);                        // 9  num_requests
  args.add_argument(p_hk_);                                  // 10 Hk
  args.add_argument(p_hv_);                                  // 11 Hv
  args.add_argument(p_dk_);                                  // 12 Dk
  args.add_argument(p_dv_);                                  // 13 Dv

  // Grid (threadgroups) = (Dv, 1, num_requests * Hv); one SIMD group of 32
  // threads per group cooperating over Dk. (gid.x=dv, gid.z=req*Hv+hv.)
  return kernel_->Launch(se::ThreadDim(32, 1, 1),
                         se::BlockDim(static_cast<uint64_t>(dv_), 1,
                                      static_cast<uint64_t>(num_seqs_ * hv_)),
                         stream, args);
}

Thunk::BufferUses MetalGdnThunk::buffer_uses() const {
  return {
      BufferUse::Read(q_, q_shape_),
      BufferUse::Read(k_, k_shape_),
      BufferUse::Read(v_, v_shape_),
      BufferUse::Read(g_, g_shape_),
      BufferUse::Read(beta_, beta_shape_),
      // h0 (operand) and ht (output) alias the same buffer: read initial state,
      // write final state in place.
      BufferUse::Read(h0_, h0_shape_),
      BufferUse::Read(cu_seqlens_, cu_seqlens_shape_),
      BufferUse::Read(slot_mapping_, slot_mapping_shape_),
      BufferUse::Write(y_, y_shape_),
      BufferUse::Write(ht_, ht_shape_),
  };
}

absl::StatusOr<ThunkProto> MetalGdnThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalGdnThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
