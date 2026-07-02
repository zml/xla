# SYCL Research Overview

This overview synthesizes the 11 SYCL research notes in this directory into an execution order for XLA SYCL backend work. The ranking favors correctness, status visibility, and API contract fidelity before performance tuning.

## Prioritization Principles

1. Correctness and observability first: failures must become `absl::Status` values at status-returning boundaries, and stream/event/callback contracts must match StreamExecutor semantics.
2. Preserve current safe defaults until tests prove a replacement: keep independent in-order queues, per-ordinal contexts, and the SYCL collective-permute host synchronization fallback.
3. Make unsupported behavior fail explicitly: unsupported priority, cluster launch, cross-context pointer use, and non-Level Zero discovery should not be silently accepted.
4. Optimize after instrumentation: queue caps, profiling properties, immediate command lists, hidden waits, and callback latency all need target hardware measurements before default changes.

## Highest Priority Changes

### 1. Make SYCL asynchronous errors status-visible

Classification: Core fixed by `441ce43c24`; remaining runtime fault-injection coverage still recommended.

Research sources: `5-async-error-propagation.md`, `6-event-semantics-and-stream-waits.md`, `7-host-callbacks.md`, `8-synchronization-costs.md`.

Status:

- `441ce43c24` replaced the log-only async handler with per-device async error storage and drains that state through status-returning synchronization paths.
- SYCL contexts and queues now install async handlers that record device ordinal, source, SYCL error code/category, and message.
- `SyclStreamSynchronize()`, `SyclStreamPool::SynchronizeStreamPool()`, `SyclStreamPool::DestroyStream()`, synchronous copy/memset/fill waits, and `SyclEvent::Wait()`/`Event::Synchronize()` now use `wait_and_throw()` plus async-error draining.
- Stream destruction and stream-pool reset now fail fast if they observe unconsumed async errors on no-return teardown paths.
- Remaining coverage gaps: real runtime fault-injection tests for kernel, copy, graph, and barrier execution where the target runtime can safely produce async failures.

Why this was first:

- `SyclAsyncHandler()` currently logs and consumes async SYCL exceptions, so execution failures can disappear instead of becoming XLA statuses.
- Status-returning wait paths use `queue::wait()` or `event.wait()` in important places, which do not guarantee async error delivery.
- This blocks reliable fixes for events, callbacks, command graphs, copies, memset/fill, and synchronization.

Initial patch direction:

- Done: add per-device async error storage that records async exceptions with device ordinal, source metadata, SYCL code/category, and message.
- Done: install async handlers on both contexts and queues.
- Done: replace status-returning waits with centralized helpers using `wait_and_throw()` and an async-state drain.
- Done: update `SyclStreamSynchronize()`, `SyclStreamPool::SynchronizeStreamPool()`, synchronous copy/memset/fill waits, and `SyclEvent::Wait()`/`Synchronize()`.
- Done: treat async failures found only in stream destruction or stream-pool reset as fatal, because there is no normal status consumer left.
- Remaining: add operation-specific queue/source tags if finer attribution becomes necessary.

Minimum tests:

- Done: inject stored async errors and verify stream sync/`BlockHostUntilDone()`, pool sync, event wait, and base `Event::Synchronize()` return non-OK.
- Remaining: add failure-injection tests for kernel, copy, graph, and barrier execution where the target runtime can safely produce them.
- Add regression coverage for the existing null packed kernel argument path, which currently logs and can return OK.

### 2. Make host callbacks actually occupy the stream

Classification: Core fixed by `f2fee4b7f641576d3acfe7a3db28a6b4e42936dc`; remaining hardening and tests still recommended.

Research source: `7-host-callbacks.md`.

Status:

- `f2fee4b7f641576d3acfe7a3db28a6b4e42936dc` replaced the event-plus-worker-thread callback model with queued SYCL `handler::host_task` commands.
- Host callbacks now participate in the in-order stream dependency chain, so later same-stream work waits for the callback body to return.
- `BlockHostUntilDone()` and stream destruction now naturally drain accepted callbacks through queue synchronization.
- The commit added a regression test showing that a blocking callback prevents later same-stream copy work from observing stale host state.
- Remaining coverage gaps: explicit `BlockHostUntilDone()` blocking test, stream-destruction drain test, and failed-dependency-to-`error_cb` test.

Why this is second:

- StreamExecutor requires host callbacks to block later work on the same stream.
- Before `f2fee4b7f641576d3acfe7a3db28a6b4e42936dc`, the SYCL implementation recorded a barrier event, then ran the callback on a side host thread. Later same-stream work could run before the callback returned.
- Before `f2fee4b7f641576d3acfe7a3db28a6b4e42936dc`, `BlockHostUntilDone()` could return after queue work completed but before an accepted callback body completed.
- XLA uses callbacks for lifetime and completion protocols, including host execute and outfeed.

Initial patch direction:

- Done: replace the event-plus-worker-thread execution model with queued SYCL `handler::host_task` commands.
- Done: make `BlockHostUntilDone()` drain accepted callbacks naturally through the queue.
- Done: preserve drain-on-destroy for callbacks that were successfully accepted.
- Remaining: call `error_cb` for all failure cases, especially failed dependency events, and keep scheduling rejection coverage explicit.

Minimum tests:

- Remaining: blocking callback keeps `BlockHostUntilDone()` blocked until the callback is released.
- Done: blocking callback prevents later same-stream copy work from completing.
- Remaining: accepted callbacks drain during stream destruction.
- Remaining: callback dependency failure reaches `error_cb` and skips the user callback.

### 3. Rebuild `SyclEvent` around explicit event state and validation

Classification: Core fixed by `4fbee7aa7c`; cross-device topology validation still required before broader event waits.

Research sources: `6-event-semantics-and-stream-waits.md`, `11-local-collective-permute-peer-copy-path.md`.

Status:

- `4fbee7aa7c` replaced the default-constructed `sycl::event` sentinel with explicit recorded/unrecorded state using `std::optional<sycl::event>`.
- Unrecorded events now preserve StreamExecutor compatibility: they poll complete and host/device waits are no-ops, while raw SYCL event access fails through `GetRecordedEvent()`.
- `Event::Synchronize()` now works for recorded SYCL events and routes through `wait_and_throw()` plus async-error draining.
- Recording now stores provenance metadata: executor/device ordinal, queue identity, context, device, backend, and barrier-marker classification.
- Stream waits now validate backend, context, device, and executor ordinal before submitting `ext_oneapi_submit_barrier({event})`; cross-device waits return `Unimplemented` until timeout-protected topology tests prove them safe.
- `WaitForEventOnExternalStream()` rejects null external streams and validates same-context/same-device queue handles before submitting a wait.

Why this is third:

- A default `sycl::event` is complete by spec, but XLA needs to distinguish "never recorded" from "recorded and complete".
- Done: `SyclEvent` now has a real `Event::Synchronize()` override.
- Done: `WaitForEventOnExternalStream()` rejects null handles and validates the target queue before waiting.
- Cross-device and cross-context event waits are not proven on the Level Zero topology XLA uses.
- Event metadata is needed before replacing broad collective-permute queue drains with event-specific waits.

Initial patch direction:

- Done: store `std::optional<sycl::event>` plus recording metadata instead of a default event sentinel.
- Done: preserve StreamExecutor compatibility: unrecorded events poll complete and waits are no-ops.
- Done: add `GetRecordedEvent()` so callers cannot accidentally use an unrecorded event as a SYCL marker.
- Done: add `Synchronize()` and route recorded-event waits through `wait_and_throw()` plus async-error draining.
- Done: store provenance on record: executor/device ordinal, queue identity, context, device, backend, and whether the event is an XLA barrier marker.
- Done: validate target queue/backend/context/device policy before submitting `ext_oneapi_submit_barrier({event})`.
- Remaining: add timeout-protected cross-device topology tests before enabling cross-device event barriers.

Minimum tests:

- Done: create-event is unrecorded, polls complete, synchronizes OK, and `GetRecordedEvent()` fails.
- Done: recording transitions to recorded state and orders same-context queues.
- Done: base-pointer `Event::Synchronize()` works for SYCL events.
- Done: external stream null is rejected; valid same-context external queue succeeds.
- Remaining: cross-device waits get timeout-protected tests before any broader enablement.

### 4. Lock down device discovery and ordinal identity

Classification: Must.

Research source: `1-device-identity-ordering-and-tile-exposure.md`.

Why this is high priority:

- The backend later uses Level Zero native interop unconditionally, so non-Level Zero devices must never enter the pool.
- Current discovery filters platform names by substring rather than checking the backend enum.
- Ordinal behavior needs to be explicit for multi-GPU, selector-filtered, and tile-exposed systems.

Initial patch direction:

- In `SyclDevicePool::InitDevicePool()`, select platforms with `platform.get_backend() == sycl::backend::ext_oneapi_level_zero`.
- Get GPU devices from Level Zero platforms and defensively verify each device backend before adding it.
- Document XLA ordinal `N` as the visible Level Zero SYCL GPU root-device list index after oneAPI/SYCL filtering.
- Preserve SYCL returned order by default. Do not sort across `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, or other user visibility controls.
- Log enough identity fields to distinguish root devices, tiles exposed as roots, and selector-filtered views.

Minimum tests:

- Every XLA SYCL device is a Level Zero GPU.
- XLA count matches Level Zero SYCL GPU discovery.
- Ordinal roundtrip preserves visible runtime order.
- Add manual/integration probe coverage for `ONEAPI_DEVICE_SELECTOR` tile exposure.

### 5. Enforce USM, memory-space, and peer-copy contracts

Classification: Core fixed by `c92737a2b5`; remaining peer-enable and collective-permute coverage still recommended.

Research sources: `2-context-policy-and-usm-isolation.md`, `3-memory-allocation-address-spaces-and-data-movement.md`, `11-local-collective-permute-peer-copy-path.md`.

Status:

- `c92737a2b5` added `SyclExecutor::GetPointerMemorySpace()` through SYCL USM pointer classification.
- Device, host, shared, pageable/unknown, null, collective, and cross-context pointer behavior is now covered by SYCL executor tests.
- `kCollective` allocations are now explicitly tested as ordinary device USM.
- D2D copy helpers now validate that device pointers are device/shared USM in the submitting queue context, and reject cross-context pointers with a clear failure instead of enqueueing undefined SYCL work.
- H2D/D2H async wrappers now preserve asynchronous stream semantics for pageable host memory by staging through host USM and freeing the staging allocation with a stream-ordered host task.
- Remaining coverage gaps: peer enablement should still get explicit directional/idempotent tests, and local collective-permute peer-copy tests should prove fallback/failure behavior for unsupported cross-context peer paths.

Why this is high priority:

- XLA uses one SYCL context per visible ordinal. Core SYCL requires USM operations to use pointers allocated for the queue context and accessible on the handler device.
- Current D2D paths can submit a copy on one ordinal while using pointers allocated through another ordinal/context.
- `kCollective` is ordinary device USM plus peer-access requirements; it is not inherently peer-visible.
- Before `c92737a2b5`, `GetPointerMemorySpace()` was missing even though SYCL can classify USM pointers.
- Before `c92737a2b5`, H2D/D2H "async" wrappers blocked for non-host-USM host pointers, which leaked a CUDA-shaped assumption.

Initial patch direction:

- Keep per-ordinal contexts as the default. Do not switch to a shared multi-root context.
- Done: implement `SyclExecutor::GetPointerMemorySpace()` using `sycl::get_pointer_type()`.
- Done: add pointer/context tests for device, host, shared, unknown/pageable, null, and cross-context pointers.
- Done: make `kCollective -> device USM` visible in tests and documentation.
- Done: validate D2D copies against the submitting queue's SYCL context and return clear cross-context failures.
- Done: preserve H2D/D2H async behavior for pageable host pointers via host-USM staging.
- Remaining: add targeted local collective-permute coverage for unsupported cross-context peer paths and any future peer-copy fallback.

Minimum tests:

- Done: USM pointer classification for all supported allocation kinds.
- Done: collective allocator returns device USM.
- Remaining: peer enablement is directional and idempotent.
- Done: cross-ordinal D2D copy is rejected when per-ordinal context checks show the path is invalid.
- Done: pageable host H2D/D2H behavior is explicitly tested.

### 6. Improve the SYCL local collective-permute peer-copy protocol without removing the fallback

Classification: Core implemented; keep queue-drain fallback and add broader topology coverage.

Research sources: `8-synchronization-costs.md`, `11-local-collective-permute-peer-copy-path.md`.

Status:

- Local SYCL collective-permute now bypasses the oneCCL communicator path when peer access is available, using XLA's peer-memory exchange and SYCL D2D copies instead.
- The peer-copy protocol has an explicit sync selector: `device_event_barrier`, `host_event_wait`, and `queue_drain`.
- SYCL defaults to `host_event_wait`, recording ready/done marker events and waiting for those marker events on the host instead of draining the whole queue.
- `queue_drain` remains available through `XLA_SYCL_COLLECTIVE_PERMUTE_SYNC_PROTOCOL=queue_drain` as the conservative fallback.
- `device_event_barrier` remains opt-in through `XLA_SYCL_COLLECTIVE_PERMUTE_SYNC_PROTOCOL=device_event_barrier` pending timeout-protected cross-device topology tests.
- Unsupported local SYCL peer access fails clearly before copying rather than silently falling back to oneCCL.

Why this comes after events and async errors:

- The current SYCL path uses host synchronization because cross-device event barriers can stall.
- Those waits protect two required protocol edges: source ready before peer read, and receiver done before source reuse.
- The previous implementation drained whole queues with `BlockHostUntilDone()`, which was stronger and costlier than necessary.

Initial patch direction:

- Done: add a protocol enum for `kQueueDrain`, `kHostEventWait`, and `kDeviceEventBarrier`.
- Done: route local SYCL collective-permute through SYCL peer-access D2D copies instead of oneCCL when the clique is local.
- Done: request peer memory for source and destination buffers used by the local peer-copy path.
- Done: implement `kHostEventWait`: record ready/done marker events, host-wait those events specifically, and avoid draining unrelated later stream work.
- Done: keep `kQueueDrain` as a runtime-selectable fallback.
- Done: keep device-side barrier mode opt-in until timeout-protected tests prove it does not stall for the supported topology matrix.
- Remaining: promote the sync selector from environment variable to a debug option if this needs stable user-facing control.

Minimum tests:

- Two-rank exchange, one-way pair, no-source zeroing, repeated source reuse, and multi-buffer cases.
- Unsupported peer access fails before copying.
- Cross-device event-barrier tests use watchdogs and fall back rather than hanging.

## Next Priority Hardening

### 7. Reject unsupported stream and command-buffer priorities consistently

Classification: Must for consistency; Should for oneAPI priority hints.

Research source: `9-stream-priority.md`.

Initial patch direction:

- Accept only `std::nullopt`, `StreamPriority::Default`, and raw integer `0` for SYCL streams until priority mapping exists.
- Reject raw nonzero integer priorities and typed `Highest`/`Lowest` with `Unimplemented`.
- Reject non-default priority in SYCL command-buffer `CreateLaunch()` and `CreateEmptyCmd()` instead of silently ignoring it.
- Add a stream-priority capability query so generic callers degrade intentionally instead of hard-coding `kSyclPlatformId`.
- Later, map `Highest`/`Default`/`Lowest` to supported oneAPI queue-priority hints and key `SyclStreamPool` by normalized priority.

### 8. Reject non-default cluster/cooperative launch in command-buffer recording

Classification: Must.

Research source: `10-cluster-cooperative-launch.md`.

Initial patch direction:

- Direct SYCL launch already rejects non-`1x1x1` clusters before queue submission.
- Add record-time rejection in `KernelThunk::Record()` and `CustomKernelThunk::Record()` before command-buffer launch creation drops cluster dimensions.
- Add earlier SYCL diagnostics for serialized/proto/cache inputs carrying non-default cluster dims.
- Do not lower clusters to ordinary SYCL `nd_range`.

### 9. Preserve independent in-order queues, then instrument and tune

Classification: Must preserve model; Should tune cap/properties.

Research source: `4-queue-and-stream-model.md`.

Initial patch direction:

- Keep production streams as independent in-order SYCL queues. Do not collapse streams onto one queue.
- Keep a hard stream cap, but make it configurable with telemetry and a warning threshold.
- Factor queue property construction so profiling, priority, and immediate/batched experiments are controlled in one place.
- Keep profiling enabled by default until timer/autotuning coverage proves selective profiling is safe.
- Add queue live-count, high-water mark, creation latency, destruction count, and cap-hit telemetry.

### 10. Remove hidden synchronization only after correctness scaffolding exists

Classification: Should.

Research sources: `3-memory-allocation-address-spaces-and-data-movement.md`, `8-synchronization-costs.md`.

Initial patch direction:

- Split copy/memset/fill helpers into enqueue helpers returning `sycl::event` plus explicit synchronous wrappers.
- Replace synchronous `event.wait()` with `wait_and_throw()` once async-error state exists.
- For H2D/D2H async calls with non-host-USM pointers, enforce the StreamExecutor host allocation contract or stage through host USM.
- Change constants from "upload and immediately block" to a cached allocation plus ready event. Consuming streams wait on the ready event when needed.
- Keep timer waits and explicit user synchronization blocking by contract.

### 11. Add topology and context diagnostics before changing context policy

Classification: Should.

Research sources: `1-device-identity-ordering-and-tile-exposure.md`, `2-context-policy-and-usm-isolation.md`, `11-local-collective-permute-peer-copy-path.md`.

Initial patch direction:

- Keep one context per visible XLA ordinal by default.
- Clarify comments so "physical device" means "visible SYCL device/XLA ordinal".
- Record/log topology metadata: backend, platform, native Level Zero handles, PCI/BDF, optional UUID, tile/subdevice information, context native handle, and peer matrix.
- Only consider shared same-root subdevice contexts behind an explicit experimental flag. Do not use a shared multi-root context as the default.

## Things Not To Do Yet

- Do not silently accept non-Level Zero devices.
- Do not sort or reinterpret visible device order by default.
- Do not automatically expose tiles by calling `create_sub_devices()` during default discovery.
- Do not switch to one shared context across multiple root devices.
- Do not collapse XLA streams to one SYCL queue.
- Do not remove SYCL collective-permute host synchronization without timeout-protected replacement tests.
- Do not map non-default clusters to plain SYCL `nd_range`.
- Do not claim oneAPI priority hints provide guaranteed scheduling priority; they are best-effort hints.

## Suggested Implementation Order

1. Backend enum discovery and ordinal documentation. This is low-risk and prevents invalid Level Zero interop.
2. Done in `441ce43c24`: async error state plus centralized `wait_and_throw()` helpers. This is the foundation for reliable status handling.
3. Done in `4fbee7aa7c`: explicit `SyclEvent` recorded/unrecorded state, event metadata, same-context wait validation, and base `Synchronize()` support.
4. Host callbacks as stream-ordered commands. This fixes a direct StreamExecutor contract violation.
5. Done in `c92737a2b5`: USM pointer introspection, memory-space contract tests, cross-context D2D rejection, and pageable-host async staging.
6. Collective-permute `kHostEventWait` mode, keeping `kQueueDrain` as fallback.
7. Priority and cluster explicit rejection patches. These are contained hardening changes that prevent silent semantic loss.
8. Hidden wait reductions for H2D/D2H and constants.
9. Queue cap/property telemetry and performance experiments.
10. Optional oneAPI priority hints, device-side collective barriers, shared same-root tile contexts, and immediate/batched queue modes, only after hardware measurements.

## Cross-Cutting Test Matrix

The research repeatedly points to missing coverage in the same areas. Build the following matrix before changing defaults:

- Devices: one device, two root devices, same-root subdevices/tiles, multi-platform Level Zero where available.
- Contexts: current per-ordinal contexts, same device different contexts, manual same-platform multi-device context, optional shared same-root subdevice context.
- Events: same queue, same context different queues, different contexts same device, different root devices, subdevices.
- Memory: device USM, host USM, shared USM, pageable host pointers, collective device USM, peer-accessible pointers.
- Streams: default priority, rejected non-default priority, many streams, callback streams, command-buffer tracing streams.
- Synchronization: queue wait, event wait, pool sync, callback drain, collective ready/done, constants ready event.

Hardware/runtime tests should use watchdogs for cross-device event barriers and should log visible device identity, context identity, and peer-access results before running the protocol.
