# 11. Local collective permute peer-copy path

## Executive recommendation

Keep the current SYCL host synchronization as a correctness workaround until XLA has timeout-protected evidence that Level Zero/SYCL cross-device event barriers make progress across the exact context and topology matrix XLA supports. The workaround is not just a performance preference: it protects the two ordering edges in the peer-copy protocol that prevent reading a source buffer before the producer stream is ready and prevent reusing a source buffer before the receiver has finished its peer read.

Do not remove host synchronization outright for the local SYCL collective-permute path in the next patch. Instead, replace the broad `BlockHostUntilDone()` drains with event-specific host waits or a small host waiter/aggregator, then feature-gate a fully device-side event-barrier protocol after the repro matrix passes. This preserves correctness while reducing the largest avoidable cost: queue-wide host drains at both protocol boundaries.

Top classification: **Must keep for correctness today; should replace the implementation because it is a performance issue once the replacement is validated**.

## Must/Should/Could classification

Must:

- Keep a SYCL-specific fallback that does not depend on cross-device `ext_oneapi_submit_barrier(waitList)` until same-context, per-device-context, root-device, and subdevice tests prove it cannot stall.
- Preserve the two ordering edges in `RunPeerAccessPermute()`: source ready before receiver D2D read, and receiver done before sender source reuse.
- Treat peer USM accessibility as conditional. XLA enables peer access with `zeDeviceCanAccessPeer`, `ext_oneapi_can_access_peer`, and `ext_oneapi_enable_peer_access`, but core SYCL still requires USM explicit-memory operations to use the allocation context, and the oneAPI peer-access extension describes peer USM device access in the same context.
- Add timeout-protected multi-GPU tests before flipping SYCL local collective-permute back to cross-device event waits.

Should:

- Replace `BlockHostUntilDone()` in the SYCL peer path with event-specific host waits on the recorded `ready` and `done` markers. This keeps host coordination but avoids draining unrelated later stream work.
- Add a protocol enum or debug option for `host_event_wait`, `device_event_barrier`, and `queue_drain` so tests can compare the current workaround with proposed replacements.
- Store SYCL event provenance in `SyclEvent` and validate cross-context/cross-device waits before submitting a barrier with a wait list.
- Add topology logging and tests for same-root tile subdevices, separate root devices, and multiple Level Zero platforms.

Could:

- Add an opt-in shared same-root subdevice context policy for tile ordinals. This can make USM and events same-context for tile-to-tile collective paths, but it should not become the multi-root default.
- Evaluate native Level Zero events for a same-context protocol after event-pool/context ownership is explicit.
- Evaluate `ext_oneapi_set_external_event()` only as an experimental optimization; do not make it a correctness dependency.
- Keep the existing queue-drain path as an emergency fallback behind a debug option while rolling out event-specific host waits.

## XLA change candidates with concrete files/functions

- `~/github/openxla/xla/xla/backends/gpu/runtime/collective_permute_thunk.cc`
  - `IsSyclExecutor()` lines 70-76: keep as the dispatch gate, but move the protocol choice behind a named SYCL peer-copy sync policy.
  - `EnablePeerAccessForLocalClique()` lines 95-121: keep pairwise enablement, and log/report the peer matrix before the path commits to D2D copies.
  - `UsesLocalSyclPeerAccessPath()` and `Prepare()` lines 206-230: keep local SYCL avoiding oneCCL communicator acquisition, but add a clear fallback/error if peer access is unsupported for any local pair.
  - `PrepareCollective()` lines 254-271: keep requesting peer memory for both source and destination buffers.
  - `RunPeerAccessCollective()` lines 469-495: keep enabling peer access before running the peer-copy protocol; add topology and protocol logging.
  - `RunPeerAccessPermute()` lines 570-672: replace the boolean `use_host_sync` with a protocol enum. For the SYCL default, record `ready`, wait for that event on host rather than draining the queue, enqueue copies/zeroing, record `done`, wait for the `done` event on host, and set the done future only after completion.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.h` and `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc`
  - `SyclEvent` lines 45-78: represent unrecorded versus recorded events explicitly and attach queue/context/device metadata when recording.
  - `WaitStreamOnEvent()` lines 62-91: validate target queue and recorded event metadata before submitting `ext_oneapi_submit_barrier({event})`; return OK for unrecorded events.
  - Add or expose an event-specific host wait using `wait_and_throw()` for recorded events. This is the primitive the host waiter protocol needs.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - `RecordEvent()` lines 198-203: keep using `SyclSubmitBarrierEvent()` as the marker, but call a metadata-preserving setter.
  - `WaitFor(Event*)` lines 206-210: route through validated SYCL event waits.
  - `BlockHostUntilDone()` lines 309-310: use an async-error-reporting wait path; do not use it as the peer-copy boundary if event-specific waits are available.
  - `Memcpy(DeviceAddressBase*, const DeviceAddressBase&, uint64_t)` lines 269-278: add tests or debug validation that cross-ordinal D2D copies are legal under the selected context and peer-access policy.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc`
  - `SyclDevicePool::GetDeviceContext()` lines 166-187: keep per-ordinal contexts by default; log visible-device/root/subdevice identity.
  - `SyclStreamPool::InitStreamPool()` and `GetOrCreateStream()` lines 227-318: queues are in-order and bound to the per-ordinal context; expose enough metadata for event validation tests.
  - `SyclSubmitBarrierEvent()` lines 433-444: keep as the recorded-marker primitive.
  - `SyclMemcpyDeviceToDevice*()` lines 514-590: add cross-context pointer/peer-access validation or document the Level Zero dependency explicitly.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc`
  - `LevelZeroCanAccessPeer()` lines 277-297 and `CanEnablePeerAccessTo()` / `EnablePeerAccessTo()` lines 876-965: keep the Level Zero query before SYCL fallback; add asymmetric and repeated-enable tests.

- Tests:
  - `~/github/openxla/xla/xla/backends/gpu/runtime/collective_permute_thunk_multigpu_test.cc`
  - `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event_test.cc`
  - `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc`
  - `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime_test.cc`

## Evidence with code references and spec/oneAPI references

### XLA source evidence

- `collective_permute_thunk.cc` lines 70-76 detects the SYCL platform by `stream_executor::sycl::kSyclPlatformId`.
- `collective_permute_thunk.cc` lines 206-230 opts local SYCL collective-permute into the peer-copy path and skips communicator acquisition because the path only needs peer-memory exchange.
- `collective_permute_thunk.cc` lines 254-271 requests peer memory for source and destination buffers when `use_peer_memory()` is set or when local SYCL peer access is selected.
- `collective_permute_thunk.cc` lines 452-495 dispatch local SYCL directly to `RunPeerAccessCollective()`, enables peer access for the local clique, and then runs `RunPeerAccessPermute()`.
- `collective_permute_thunk.cc` lines 566-672 implement the peer D2D protocol. For SYCL, lines 582-595 record a ready event and immediately call `BlockHostUntilDone()`. Lines 615-617 skip the normal receiver-side `stream.WaitFor(source_events.ready)`. Lines 648-655 call `BlockHostUntilDone()` after the receiver copy/zeroing and set the done promise. Lines 666-668 skip the normal sender-side wait on the target done event.
- `sycl_event.cc` lines 62-91 show the non-SYCL peer-copy path shape that SYCL avoids: wait by submitting `ext_oneapi_submit_barrier(std::vector<sycl::event>{event})`.
- `sycl_stream.cc` lines 190-210 build stream/event waits from recorded barrier events. `RecordEvent()` stores the event returned from `SyclSubmitBarrierEvent()`.
- `sycl_stream.cc` lines 269-278 enqueue D2D copies through `SyclMemcpyDeviceToDeviceAsync()`. `sycl_gpu_runtime.cc` lines 79-94 and 575-590 implement that as `queue::memcpy()`.
- `sycl_stream.cc` lines 309-310 implement `BlockHostUntilDone()` as `SyclStreamSynchronize()`, and `sycl_gpu_runtime.cc` lines 422-430 implements that as `queue::wait()`.
- `sycl_gpu_runtime.cc` lines 166-187 cache one `sycl::context` per visible device ordinal. Lines 227-318 create in-order, profiling-enabled queues using that ordinal's context and device.
- `sycl_executor.cc` lines 876-965 use `zeDeviceCanAccessPeer`, fall back to `ext_oneapi_can_access_peer`, and enable one-way peer access with `ext_oneapi_enable_peer_access`.
- `collective_memory_requests.h` lines 51-55 define peer memory as address exchange between local ranks for pointer-based peer access. `collective_memory.cc` lines 461-535 exchange peer allocation addresses through in-process rendezvous, and lines 135-179 look up peer addresses by clique, rank, and allocation/slice.
- `collective_permute_thunk_multigpu_test.cc` lines 198-214 has a two-device collective-permute execution test, but the file header says the tests are through NCCL and CUDA-oriented command-buffer paths. It does not cover SYCL root/subdevice topology or event-barrier stall behavior.
- `sycl_gpu_runtime_test.cc` lines 126-146 verify per-ordinal context caching and distinct contexts for ordinals 0 and 1. Lines 596-665 test two-device D2D copies, but without topology classification, peer-access assertions, or cross-device event barriers.

### SYCL 2020 evidence

- `~/sycl/sycl-2020-map.md` maps the cited sections: event dependencies `S|5|3002-3023`, USM `S|3|23305-24820`, and explicit memory operations `S|5|30753-31085`.
- `~/sycl/sycl-2020.html` lines 3002-3019 say events can coordinate host/device execution and can define prerequisites between command groups.
- `~/sycl/sycl-2020.html` lines 11460-11478 say `property::queue::in_order` only orders commands in the same queue and gives no ordering for commands submitted to other queues.
- `~/sycl/sycl-2020.html` lines 23584-23637 summarize USM accessibility. Device USM access from another device is optional P2P; shared USM access from another device is optional.
- `~/sycl/sycl-2020.html` lines 23640-23649 state the critical context rule: each USM allocation has an associated SYCL context; kernels and explicit memory commands using the pointer must use the same context, and violating this is undefined behavior.
- `~/sycl/sycl-2020.html` lines 30753-31035 define explicit memory operations. Lines 30946-30951 require `memcpy` USM pointers to be accessible on the handler's device and, if USM, allocated from the same context as the handler's queue. Lines 30979-31029 make the same-context/accessibility rule explicit for `memset`, `fill`, `prefetch`, and `mem_advise`.

### oneAPI and Level Zero evidence

- The local oneAPI docs are present under `~/sycl/oneapi`, which is a symlink to `~/github/intel/llvm/sycl/doc/`. Default `rg --files` did not show them because ignore handling skipped the symlink; `rg --files -u ~/sycl/oneapi` did.
- `~/sycl/oneapi/MultiTileCardWithLevelZero.md` lines 30-32 say multiple GPUs appear as multiple SYCL root devices, on one Level Zero platform on Linux and potentially multiple Level Zero platforms on Windows.
- The same file lines 43-55 say tile-capable roots can be partitioned into subdevices corresponding to physical tiles, with persistent order from `create_sub_devices()` and `ZE_AFFINITY_MASK` controlling exposed subdevices.
- Lines 64-67 say a SYCL context may contain one or multiple root devices or subdevices from the same SYCL platform.
- Lines 74-90 say Level Zero `malloc_device` is accessible only by the specified device, `malloc_host` by host and any device in the context, `malloc_shared` by host and the specified device, and root-device allocations are accessible by all subdevices of that root.
- Lines 105-168 describe the context alternatives. Option A is one context per subdevice/tile, no data sharing across queues, and best per-tile performance. Option B is a shared same-root subdevice context with data sharing. Option D is a multi-root context with sharing at the cost of slow host access or explicit copies.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc` lines 76-99 define enqueued barriers as host-nonblocking dependency edges. Lines 288-300 state there are no context constraints on queues participating in a `waitList`. This supports a device-side barrier protocol in principle, but it does not override XLA's observed stall comment or remove the need for runtime tests.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_in_order_queue_events.asciidoc` lines 44-63 mark the in-order queue events extension experimental and not recommended for general usage. Lines 99-154 define `ext_oneapi_get_last_event()` and `ext_oneapi_set_external_event()`, including the requirement that the external event not reach complete before completion of the most recent command submitted to the queue.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_peer_access.asciidoc` lines 49-60 say P2P memory access is available only on CUDA, HIP, and Level Zero backends and applies to USM device allocations, not USM shared allocations. Lines 89-153 say peer access allows direct access to USM device allocations for a peer device in the same context and remains subject to normal context rules.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md` lines 72-113 map SYCL devices, contexts, queues, and events to Level Zero handles. Lines 391-410 say `make_queue()` requires the device to be in the context and `make_event()` requires a Level Zero event from an event pool created in the same context.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_intel_event_mode.asciidoc` lines 53-74 and 125-137 describe an experimental low-power event mode for waits, currently effective for Level Zero barrier and partial-barrier commands. This is a CPU-cost optimization, not a correctness primitive.

## Findings

1. The current SYCL workaround is protecting real correctness edges. The ready edge prevents the receiver from reading a source peer buffer before the source stream's prior writes complete. The done edge prevents the sender from reusing or overwriting that source buffer before the receiver's D2D copy has completed.

2. The workaround avoids the exact operation suspected of stalling: submitting a target-queue barrier that waits on a `sycl::event` produced by another device queue. XLA records the event, then for SYCL drains the recording stream on the host and skips `stream.WaitFor(source_events.ready)` and `stream.WaitFor(target_done)`.

3. Current XLA multi-ordinal SYCL execution normally uses different SYCL contexts. `SyclDevicePool::GetDeviceContext()` caches one context per visible ordinal, and all queues for that ordinal use that context. Therefore cross-device collective-permute event waits are cross-context unless a future context policy changes this.

4. oneAPI enqueue barriers appear to allow wait-list events across contexts, but XLA source already records that SYCL cross-device event barriers can stall. Without runtime evidence, the source comment must be treated as the stronger project-local evidence.

5. Peer memory exchange only gives XLA the peer pointer. It does not by itself prove SYCL-level legality of submitting a USM `queue::memcpy()` on one context with a pointer allocated in another context. XLA partially mitigates with Level Zero peer access checks, but the spec evidence still requires targeted tests.

6. Same-root tile subdevices are a special case. A shared same-root subdevice context can make data sharing more natural, but XLA's current per-ordinal context policy matches oneAPI option A and does not share data across tile queues by default.

7. The current host-sync implementation is stronger than necessary. `BlockHostUntilDone()` waits for the whole queue up to that point. The protocol only needs to wait for two marker events: ready and done. Replacing queue drains with event-specific host waits should preserve ordering and reduce collateral synchronization.

8. Fully avoiding host synchronization is a should-replace target, not a safe immediate change. It needs a validated event-barrier protocol and a fallback path for topologies where cross-device barriers stall.

## Protocol alternatives

### Current queue-drain host synchronization

Protocol: record `ready`; for SYCL, call `BlockHostUntilDone()`; rendezvous; receiver skips waiting on source ready because the source host already drained; receiver enqueues D2D copies or memzero; for SYCL, call `BlockHostUntilDone()`; receiver sets the done promise; sender awaits the promise and skips waiting on the done event.

Correctness: strong. Source prior work is complete before any peer read. Receiver copy/zeroing is complete before returning and before sender source reuse.

Cost: high. It blocks the rank host thread twice and drains all prior work in the stream rather than only the relevant marker. It also prevents overlap between the peer copy and useful host-side scheduling for the same collective.

Recommendation: keep as fallback, but replace as default once event-specific host waits are implemented and tested.

### Event-specific host waiter or aggregator

Protocol:

1. Record a `ready` marker event on each rank's stream.
2. Wait on the local `ready` event on the host, preferably through a small waiter/aggregator using `wait_and_throw()` or a `SyclEvent::Synchronize()` override. Do not call `queue::wait()`.
3. Rendezvous after the ready event is complete, or rendezvous first and have receivers await the source ready-complete future before enqueueing the peer read.
4. Receiver enqueues the D2D copies or destination memzero.
5. Receiver records a `done` marker after the copy/zeroing and waits on that marker on the host.
6. Receiver resolves the done promise only after the done marker completes.
7. Sender returns only after the target done promise resolves.

Correctness proof:

- Source readiness: `ready` is a barrier event recorded after all prior source stream work. Waiting for the ready event to complete before enqueueing or executing the receiver copy means the receiver cannot read stale source data.
- Destination readiness: `done` is recorded after the receiver's copy/zeroing in the receiver's in-order queue. Waiting for `done` means the destination write has completed before the receiver returns.
- Source reuse: the sender's done future resolves only after the receiver's done event completes, so the sender cannot return and enqueue later source-buffer reuse before the receiver has finished reading.
- No source: the receiver records and waits on `done` after `MemZero`, so the destination is still ready before return.

Cost: still host-synchronized, but it waits only on marker events. It avoids cross-device event barriers and avoids queue-wide drain semantics.

Recommendation: best near-term replacement.

### Device-side explicit dependency barrier

Protocol: use the existing non-SYCL path. Record source `ready`; receiver calls `stream.WaitFor(source_ready)` which submits `ext_oneapi_submit_barrier({source_ready})`; receiver enqueues copy/zeroing and records `done`; sender calls `stream.WaitFor(target_done)`.

Correctness proof: same edges as above, but enforced by target-queue dependency barriers rather than host waits.

Required changes:

- `SyclEvent` must carry recorded-event metadata and validate queue/event compatibility.
- Tests must show no stalls and correct data for same context, per-device contexts, same-root tile subdevices, separate roots, and multi-platform Level Zero cases.
- Add a timeout and fallback to the host-wait protocol when a topology is not certified.

Recommendation: long-term performance target, not immediate default.

### Shared same-root subdevice context

Protocol: for tile ordinals from one root device, create one context containing the selected subdevices, allocate collective buffers in that context, and use device-side event barriers inside the shared context.

Benefits: aligns with oneAPI option B and removes the core SYCL same-context issue for tile-to-tile USM operations.

Costs: changes allocator ownership, module cache identity, context-level allocation behavior, and root/tile policy. It does not solve multi-root or multi-platform systems.

Recommendation: optional experiment for tile systems only.

### `ext_oneapi_set_external_event`

Protocol: set the source ready event as an external event on the receiver in-order queue before submitting the copy, and set the target done event as an external event on the sender queue before subsequent work.

Problems: the extension is experimental, not recommended for general use, requires in-order queues, and says behavior is undefined if the external event reaches complete before completion of the queue's most recent command. Ready events can be complete before the receiver sets them, especially after host fallback or fast source streams.

Recommendation: do not use for correctness. Benchmark only after the barrier protocol is validated.

### Native Level Zero events

Protocol candidate: create native Level Zero events and use Level Zero command-list event waits/signals, wrapping events with `sycl::make_event()` only where needed.

Problems: the local Level Zero backend extension only establishes handle mapping and says `make_event()` requires an event pool from the same context. It does not provide enough local evidence for a portable cross-context event protocol.

Recommendation: investigate only after collecting Level Zero runtime behavior and context ownership constraints. Same-context native events may help the shared same-root subdevice policy; they do not automatically solve current per-ordinal contexts.

## Proposed patch plan

1. Add protocol selection without changing behavior.
   - Introduce `SyclPeerCopySyncMode` in `collective_permute_thunk.cc`: `kQueueDrain`, `kHostEventWait`, `kDeviceEventBarrier`.
   - Default SYCL to `kQueueDrain` initially.
   - Add VLOG output showing selected mode, clique, source/target rank, device ordinal, and whether peer access was enabled.

2. Implement event-specific host waits.
   - Add a recorded-event host wait API in `SyclEvent`, using `wait_and_throw()` once async-error propagation is available.
   - In `RunPeerAccessPermute()`, record `ready`, host-wait `ready`, rendezvous, enqueue copy/zeroing, record `done`, host-wait `done`, then set the done promise after completion.
   - Keep `kQueueDrain` as fallback.

3. Add correctness tests for host-event wait mode.
   - Two-rank local collective-permute exchange.
   - One-way pair and no-source memzero.
   - Repeated invocations reusing the same source/destination buffers.
   - Multi-buffer collective-permute.
   - Delayed source producer and delayed receiver copy to stress the ready/done edges.

4. Harden SYCL event waits before trying device-side mode.
   - Add explicit unrecorded/recorded state and event metadata in `SyclEvent`.
   - Validate target queue backend/context/device before `ext_oneapi_submit_barrier(waitList)`.
   - Add timeout-protected tests for cross-device waits. The test should fail fast if the barrier stalls.

5. Enable device-side barrier mode only behind a flag.
   - Gate by runtime test coverage and, if needed, by topology: same context, same-root shared subdevice context, per-device root contexts, etc.
   - On unsupported or untested topologies, fall back to host-event wait mode.

6. Investigate context policy separately.
   - Keep per-ordinal contexts by default.
   - Prototype shared same-root subdevice context only under an explicit experimental flag and include context policy in allocator/module/event diagnostics.

## Test/benchmark coverage

Repro matrix, because no runtime was available in this research pass:

- Same device, same context:
  - Two in-order queues from the same `SyclDevicePool::GetDeviceContext(0)`.
  - Record ready on queue A, submit barrier wait on queue B, enqueue copy/fill, record done.
  - Expected: no stall; this is the baseline.

- Same device, different contexts:
  - Two independent `sycl::context(device0)` objects and queues.
  - Same barrier wait protocol.
  - Expected: this probes whether the enqueue-barrier "no context constraints" rule works in the adapter.

- Current XLA per-device contexts, two root devices:
  - Use executor 0 and executor 1 streams.
  - Run a synthetic ready/copy/done protocol and the full collective-permute path.
  - Include `CanEnablePeerAccessTo` both directions, `EnablePeerAccessTo`, and pointer-query diagnostics.

- Shared multi-root context:
  - Manually create `sycl::context({device0, device1})` when both roots are on the same platform.
  - Compare event barriers, D2D copy correctness, and allocation behavior against current XLA per-device contexts.

- Same-root subdevices, per-tile contexts:
  - If `create_sub_devices(partition_by_affinity_domain(next_partitionable))` is available, create one context per subdevice and run the event barrier and peer copy tests.
  - This matches XLA's current per-ordinal model if tiles are exposed as ordinals.

- Same-root subdevices, shared context:
  - Create one context containing the subdevices and queues attached to each subdevice.
  - Run the device-side barrier protocol and compare with host-event waits.

- Multi-platform Level Zero:
  - On Windows-style multiple Level Zero platforms or emulation if available, verify that shared context is not attempted and that peer/event behavior is classified per platform.

Collective tests to add:

- `CollectivePermuteThunkSyclPeerCopyTest.ExchangeTwoRanksHostEventWait`
- `CollectivePermuteThunkSyclPeerCopyTest.ExchangeTwoRanksDeviceBarrier`
- `CollectivePermuteThunkSyclPeerCopyTest.OneWayAndNoSourceMemZero`
- `CollectivePermuteThunkSyclPeerCopyTest.RepeatedInvocationsReuseSource`
- `CollectivePermuteThunkSyclPeerCopyTest.MultiBuffer`
- `CollectivePermuteThunkSyclPeerCopyTest.UnsupportedPeerAccessFailsBeforeCopy`
- `CollectivePermuteThunkSyclPeerCopyTest.CrossDeviceBarrierTimeoutFallback`

Metrics:

- Latency per collective for queue drain, host-event wait, and device barrier.
- Host blocked time and CPU utilization.
- D2D bandwidth for each topology.
- Number and duration of queue waits or event waits.
- Overlap with independent work on other streams.
- Failure/stall rate under repeated invocations.

## Rollout risk

- Correctness risk is high if host synchronization is simply removed. A missed ready edge can read stale source data; a missed done edge can allow source reuse while a peer read is still in flight.
- Performance risk is high if queue drains stay as the default. The current path blocks the host and drains the whole stream at both protocol boundaries.
- Event-specific host waits are medium risk. They preserve ordering but require reliable `SyclEvent` host synchronization and async-error propagation.
- Device-side event barriers are medium to high risk until tested. oneAPI documentation permits them in principle, but XLA source says cross-device barriers can stall.
- Shared same-root subdevice contexts are high risk as a default. They affect allocator isolation, module cache keys, memory visibility, and performance assumptions.
- Native Level Zero events are high risk without a complete context/event-pool ownership design.

## Evidence gaps

- No SYCL runtime or Intel multi-GPU hardware tests were run in this pass. The stall is reproduced by code analysis and XLA's local comment, not by a fresh measurement.
- No local result proves which topologies actually stall: same context, per-device contexts, root devices, same-root subdevices, multi-root contexts, or multi-platform Level Zero.
- No local measurement proves that current cross-context D2D copies are valid on all target Level Zero systems. The code works through peer-access checks, but core SYCL and the oneAPI peer-access extension still leave a same-context evidence gap.
- No local evidence defines a safe native Level Zero cross-context event protocol. The Level Zero backend doc only proves handle mapping and same-context `make_event()` requirements.
- No benchmark data quantifies how much `BlockHostUntilDone()` costs versus event-specific host waits.
- No tests found that validate SYCL local collective-permute peer-copy mode across root/subdevice topologies with timeout protection.
