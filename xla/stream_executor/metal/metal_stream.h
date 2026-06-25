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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor::metal {

class MetalExecutor;

class MetalStream : public StreamCommon {
 public:
  explicit MetalStream(
      MetalExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority);
  ~MetalStream() override;

  absl::Status WaitFor(Stream* other) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status WaitFor(Event* event) override;

  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* device_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& device_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* device_dst,
                      const DeviceAddressBase& device_src,
                      uint64_t size) override;

  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;
  absl::Status BlockHostUntilDone() override;
  absl::Status FlushBatchedWork() override;
  absl::Status CommitBatchedWorkNoWait() override;

  Stream::PlatformSpecificHandle platform_specific_handle() const override;

  absl::Status LaunchMetalKernel(
      const ThreadDim& thread_dims, const BlockDim& block_dims,
      const std::optional<ClusterDim>& cluster_dims, void* pipeline,
      void* function, bool use_argument_buffer, absl::string_view name,
      void** args, int64_t shmem_bytes, bool use_pdl);

  // Force-commits this stream's OPEN command buffer iff it carries the awaited
  // signal `value` (encoded but not yet committed). Called by
  // MetalExecutor::CommitOpenBufferThrough so a cross-stream GPU waiter on
  // `value` is committed only AFTER its signaler (the single-queue ordering
  // rule, metal_runtime.h). No-op when no open buffer carries `value` — which
  // is always, until per-execute commits are deferred.
  void FlushOpenBufferIfCarrying(uint64_t value);

 private:
  absl::Status LaunchKernel(const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const std::optional<ClusterDim>& cluster_dims,
                            void* function, absl::string_view name,
                            void** args, int64_t shmem_bytes,
                            bool use_pdl) override;
  // Ensures `command_buffer_` holds an open MPSCommandBuffer.
  void EnsureOpenCommandBuffer();
  // Commits the open command buffer WITHOUT waiting — submits batched GPU work so
  // its signals / PJRT definition events can fire and (eventually) complete — and
  // advances the signal bookkeeping. No-op when nothing is open. Does NOT touch the
  // adaptive-K / executes_this_token_ state. Called by FlushBatchedWork at the token
  // boundary.
  void CommitOpenBufferNoWait();

  MetalExecutor* executor_;
  // The open (uncommitted) MPSCommandBuffer for the current execution, or null
  // between executions. Committed + waited by BlockHostUntilDone, committed +
  // recorded by RecordEvent.
  void* command_buffer_ = nullptr;
  // The most recent shared-event value this stream has signaled (0 = none yet).
  // RecordEvent's else branch (command_buffer_ == nullptr) is reached when this
  // execute has no open buffer of its own to signal into — e.g. the previous
  // batch already committed at an adaptive-K / token boundary, or the execute
  // encoded no GPU work. Such an event still lets a consumer order against this
  // stream's work by referencing the highest value already signaled (committed or
  // pending), not 0. (PJRT coalesces all sibling outputs of one execute onto a
  // single definition event, so there is one RecordEvent per execute, not per
  // output.) Single compute stream → no atomics needed.
  uint64_t last_signaled_value_ = 0;
  // Highest shared-event value ENCODED into the current OPEN command buffer but
  // not yet committed (0 when no open buffer / nothing signaled into it). Lets
  // FlushOpenBufferIfCarrying tell whether the open buffer carries a given
  // awaited value. Nonzero once RecordEvent defers commits (commit_every_ > 1).
  uint64_t pending_signal_high_ = 0;
  // Partial command-buffer batching. The GPU pays a ~0.1ms scheduling + cross-cb
  // shared-event gap BETWEEN command buffers; with ~30 executes/token that is
  // ~18% GPU idle. Keeping the command buffer OPEN across K executes (RecordEvent
  // signals but does not commit; same-stream consumers order in-buffer by program
  // order, see WaitFor) amortizes that gap over K executes — like CUDA's single
  // stream. But the sweet spot is per-MODEL: amortization peaks where it balances
  // lost host/GPU overlap (K=8 for 3B's ~30-execute token, i.e. ~4 commits/token);
  // a model with far fewer executes/token would over-batch at a fixed 8. So the
  // portable knob is commits-PER-TOKEN, and K is derived per token (adaptive_k_).
  //
  // commit_every_ is the CEILING on the adaptive K (default 8, tuned to the 3B
  // decode peak); K=1 reproduces the per-execute baseline exactly.
  int commit_every_ = 8;
  // Target voluntary commits per decode token (default 4 → K≈8 on 3B). The real,
  // model-portable cadence constant.
  int target_commits_per_token_ = 4;
  // Commit-eligible executes (signals) since the last token boundary
  // (FlushBatchedWork = the logits readback). Recomputes adaptive_k_ each token.
  int executes_this_token_ = 0;
  // Current per-token commit threshold: clamp(round(executes_per_token /
  // target_commits_per_token_), 1, commit_every_). Initialized to the ceiling in
  // the constructor; recomputed at every token boundary, so it auto-scales to the
  // model with no fixed K. K=1 (ceiling) still reproduces the baseline bit-for-bit.
  int adaptive_k_ = 8;
  int signals_since_commit_ = 0;
  // Highest shared-event value already encoded as a GPU WAIT into the current
  // OPEN command buffer. A later consumer in the SAME buffer needing
  // value <= this is ordered by program order behind that earlier wait, so
  // its encodeWaitForEvent (a ~0.1 ms firmware stall EACH, even when the
  // value is long signaled) is elided. Dominant case: every per-layer execute
  // waits the SAME per-token H2D-staged input (block tables etc.) — one wait
  // per buffer instead of one per layer. Reset whenever the open buffer
  // closes: a wait in a PREVIOUS buffer does not order a new one (buffers on
  // the queue can overlap execution — proven by the reverted commit-order
  // elision corrupting CLI decode).
  uint64_t waited_value_high_ = 0;
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_
