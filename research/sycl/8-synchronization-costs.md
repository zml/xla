# 8. Synchronization costs

## Executive recommendation

Treat the SYCL backend's blocking points as three different classes, not one generic "wait problem".

Must: preserve the current local collective-permute host-sync fallback until a deadlock-tested replacement exists. The source says SYCL cross-device event barriers can stall, and the current host synchronization is a correctness workaround at protocol boundaries, not just a slow path.

Should: remove hidden waits from paths whose XLA surface is asynchronous or stream-ordered. The highest-payoff candidates are non-host-USM H2D/D2H "async" copies and constant uploads that immediately call `BlockHostUntilDone()`.

Could: optimize measurement, teardown, and callback waits after the correctness-sensitive paths are stable. Timer waits are measurement-only; stream destruction and reset waits are mostly lifetime/cleanup costs.

## Must/Should/Could classification

- Must: local SYCL collective-permute synchronization protocol in `~/github/openxla/xla/xla/backends/gpu/runtime/collective_permute_thunk.cc::RunPeerAccessPermute`. Do not reintroduce cross-device event waits unless they are proven non-stalling on the target Level Zero/SYCL stack. Add a watchdog multi-GPU test before any replacement.
- Must: keep explicit user-requested synchronization APIs blocking: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc::BlockHostUntilDone`, `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc::SyclStreamSynchronize`, and `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc::SynchronizeAllActivity`.
- Should: split enqueue from wait in `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` copy/memset helpers so asynchronous wrappers can retain events instead of losing them.
- Should: replace blocking non-host-USM H2D/D2H paths in `SyclMemcpyHostToDeviceAsync` and `SyclMemcpyDeviceToHostAsync` with either a hard precondition that host pointers come from `HostMemoryAllocate`, or a staging mechanism that preserves async stream semantics.
- Should: replace the constant-upload wait in `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc::CreateOrShareConstant` with a ready-event model so later streams wait only when they first consume a not-yet-ready constant.
- Could: move callbacks from a dedicated host thread waiting on `SyclEvent::Wait()` to a SYCL `host_task` or another stream-ordered host callback mechanism, if error propagation and teardown semantics remain equivalent.
- Could: reduce CPU cost for measurement/debug waits using SYCL/oneAPI profiling or low-power event mode where available, but do not expect timer reads to become non-blocking when callers ask for elapsed duration.
- Could: make reset/destructor waits cheaper by waiting outside global locks and narrowing ownership, after the stream-pool lifetime model is made explicit.

## XLA change candidates with concrete files/functions

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc`
  - `MemcpyDeviceToHost`, `MemcpyHostToDevice`, `MemcpyDeviceToDevice`, `MemsetDevice`, `MemfillDevice`: split into `Enqueue*` helpers returning `StatusOr<::sycl::event>` plus synchronous wrappers that wait. Replace `event.wait()` with `event.wait_and_throw()` where synchronous behavior remains required.
  - `SyclMemcpyDeviceToHostAsync`, `SyclMemcpyHostToDeviceAsync`: remove hidden device-transfer waits for ordinary host pointers. Preferred policy is to honor the common `Stream::Memcpy` contract that host pointers are allocated by `StreamExecutor::HostMemoryAllocate`; defensive fallback can use host-USM staging plus a stream callback to copy/free staging.
  - `SyclStreamSynchronize`, `SyclStreamPool::SynchronizeStreamPool`, `SyclStreamPool::Reset`: keep blocking semantics, but use `wait_and_throw()` when changing behavior is acceptable, and avoid using pool-wide waits for narrower internal ordering.
  - `SyclStreamPool::GetOrCreateStream`: benchmark, do not assume, queue properties for `ext::intel::property::queue::immediate_command_list` and `no_immediate_command_list`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - `BlockHostUntilDone`: keep as explicit API synchronization.
  - `DoHostCallbackWithStatus` / `RunCallbackTask`: evaluate replacing worker-thread `event.wait_and_throw()` with a stream-ordered host task. Preserve callback FIFO behavior, status propagation, and destructor behavior.
  - `~SyclStream`: keep lifetime safety; only optimize after shared default-stream ownership is clarified.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_timer.cc`
  - `SyclTimer::GetElapsedDuration` / `GetEventElapsedTime`: keep measurement blocking, but compare Level Zero timestamp queries with SYCL `event::get_profiling_info` and oneAPI low-power barrier events for CPU-use reduction.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc`
  - `CreateOrShareConstant`: cache a ready event with the constant allocation. Same-stream users rely on queue order; other streams call `WaitFor(ready)` before first use. Keep the current blocking path behind a fallback flag until multi-stream tests pass.
  - `SynchronizeAllActivity`: keep draining all streams for the public all-activity API; do not use it as an internal shortcut for one stream's dependency.
- `~/github/openxla/xla/xla/backends/gpu/runtime/collective_permute_thunk.cc`
  - `RunPeerAccessPermute`: replace host boundary drains only with a protocol that proves source-buffer readiness and target read completion without cross-device stalls. Candidate mechanisms are a host aggregator that waits on per-rank events without draining unrelated stream work, Level Zero events created in a compatible context, or oneAPI in-order external events, gated by feature tests and a watchdog.

## Evidence

Code references:

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:45-126` submits `memcpy`, `memset`, and `fill`, then calls `event.wait()` when the helper is not in async mode.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:535-572` makes `SyclMemcpyDeviceToHostAsync` and `SyclMemcpyHostToDeviceAsync` non-blocking only when the host pointer is SYCL host USM.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:320-336` waits every queue in a device ordinal's stream pool; `~/github/openxla/xla/xla/stream_executor/sycl/sycl_context.cc:34-35` and `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:815-817` route context/all-activity sync through that pool drain.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:373-385` resets the global stream pool by waiting all queue handles for all devices before clearing them.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:309-311` implements `BlockHostUntilDone()` as `SyclStreamSynchronize`; `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:422-430` implements that with `queue::wait()`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:345-373` joins the callback thread, then blocks the stream before destroying/removing the queue handle.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:397-409` runs host callbacks by waiting on a recorded `SyclEvent` in a callback worker thread.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_timer.cc:30-77` gets native Level Zero event handles, calls `zeEventHostSynchronize(end_event, UINT64_MAX)`, and then queries timestamps; `:89-99` records the stop event before querying elapsed duration.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:702-705` uploads a constant with `stream->Memcpy(...)` and immediately calls `stream->BlockHostUntilDone()`.
- `~/github/openxla/xla/xla/backends/gpu/runtime/collective_permute_thunk.cc:582-596` records a ready event and then blocks the stream for SYCL; `:648-652` blocks again before publishing done.
- `~/github/openxla/xla/xla/stream_executor/stream.h:146-155` says H2D/D2H `Stream::Memcpy` host pointers must be allocated by `StreamExecutor::HostMemoryAllocate`; SYCL implements host allocation with `aligned_alloc_host` in `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:684-696` and routes `SyclExecutor::HostMemoryAllocate` through it in `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:409-430` and `:865-867`.

SYCL 2020 references:

- `~/sycl/sycl-2020-map.md:48` maps scheduling to `S|4|4246-4262`; `~/sycl/sycl-2020.html:4246-4259` says command-group calls enqueue work and are host non-blocking, with queued commands ordered by dependencies.
- `~/sycl/sycl-2020-map.md:591-593` maps queue wait APIs; `~/sycl/sycl-2020.html:10890-10918` says `queue::wait` and `queue::wait_and_throw` block until prior commands complete, with `wait_and_throw` additionally consuming asynchronous errors.
- `~/sycl/sycl-2020-map.md:610` maps event wait; `~/sycl/sycl-2020.html:11660-11669` says `event::wait` blocks until associated and dependent commands complete.
- `~/sycl/sycl-2020-map.md:613-617` and `:121` map profiling descriptors; `~/sycl/sycl-2020.html:11743-11760` says `event::get_profiling_info` blocks until requested profiling information is available, and `~/sycl/sycl-2020.html:11853-11977` defines submit/start/end timestamp descriptors.
- `~/sycl/sycl-2020-map.md:206` maps explicit memory operations; `~/sycl/sycl-2020.html:30753-31085` describes explicit copy/fill/prefetch/mem-advise operations as queued memory commands, not host waits.

oneAPI references:

- `~/sycl/oneapi.md:7-12` confirms the local oneAPI docs tree and extension status directories; `:32` routes Level Zero docs to `oneapi/MultiTileCardWithLevelZero.md` and `oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md`.
- `~/sycl/oneapi/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc:45-58` says the immediate-command-list extension is supported but mainly tested on Intel Data Center Max Series GPUs and meaningful on the Level Zero backend. Lines `62-78` distinguish immediate submission from standard queue batching; lines `117-126` define the `immediate_command_list` and `no_immediate_command_list` queue hints.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_intel_event_mode.asciidoc:65-74` says some event waits can busy-wait and low-power mode reduces CPU checking at latency cost. Lines `127-136` say low-power mode affects `event::wait()` / `wait_and_throw()` only when supported, currently for barrier and partial-barrier commands on Level Zero queues.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_in_order_queue_events.asciidoc:55-63` describes APIs for the last event and external event dependencies on in-order queues, but warns it is solving a specific problem and is not recommended for general usage. Lines `116-154` define `ext_oneapi_get_last_event()` and `ext_oneapi_set_external_event()`.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc:79-98` says enqueued barriers are non-blocking from the host perspective, add dependency edges, and can return an event.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md:297-316` defines `get_native` for Level Zero handles and notes queue native handles can be command queues or command lists. Lines `391-410` define `make_queue` and `make_event` constraints, including same-context Level Zero event-pool requirements.

## Findings

1. The explicit sync APIs are doing what their names require. `BlockHostUntilDone`, `SynchronizeAllActivity`, and `SyclStreamPool::SynchronizeStreamPool` are blocking by contract; the risk is accidental reuse of these APIs where an event dependency would suffice.
2. The H2D/D2H async wrappers are the clearest hidden serialization. They block for non-host-USM pointers even though the `Stream::Memcpy` interface already documents host memory as coming from `HostMemoryAllocate`. The likely policy fix is to reject or stage non-host-USM host pointers, not drain the device transfer in an async API.
3. Constant upload blocks even though the copy is on a stream. This makes the constant globally ready before caching, but it also destroys overlap. A ready-event cache can preserve correctness with finer-grained waits.
4. The SYCL collective-permute local peer path intentionally uses host synchronization because cross-device event barriers can stall. This is a workaround, not an accidental wait. Removing it without an equivalent readiness/done protocol risks deadlock or source-buffer reuse races.
5. Timer waits are measurement-only. SYCL profiling APIs are also allowed to block when profiling data is unavailable, so timer optimization should target CPU overhead and implementation clarity, not full non-blocking elapsed-duration reads.
6. Queue waits use `wait()` in several places. This is cheaper to reason about for synchronization costs, but `wait_and_throw()` is the SYCL mechanism that also flushes async errors; changing it may expose latent failures and should be tested with the error-propagation work.
7. oneAPI immediate command-list properties are relevant to enqueue overhead and overlap, but local docs explicitly say applications should usually rely on the default and that the hint has hardware/backend limits. Treat it as a benchmark axis, not a default change.

## Wait-site inventory

| ID | Site | Current wait | Category | Replacement candidate | Priority |
|---|---|---|---|---|---|
| W1 | `sycl_gpu_runtime.cc:45-126` helper `Memcpy*`, `MemsetDevice`, `MemfillDevice` | `event.wait()` when `async=false` | Required for synchronous wrappers; accidental if helper reuse hides async intent | Split enqueue helpers returning events from synchronous wrappers; use `wait_and_throw()` in sync wrappers | Should |
| W2 | `sycl_gpu_runtime.cc:535-572` `SyclMemcpyDeviceToHostAsync`, `SyclMemcpyHostToDeviceAsync` | Calls W1 with `async=false` unless host pointer is SYCL host USM | Workaround / accidental serialization | Enforce `HostMemoryAllocate` precondition, or host-USM staging plus stream callback/free | Should |
| W3 | `sycl_stream.cc:309-311`, `sycl_gpu_runtime.cc:422-430` `BlockHostUntilDone` / `SyclStreamSynchronize` | `queue::wait()` | Required for explicit host blocking | Keep blocking; consider `wait_and_throw()` and tests for async errors | Must keep |
| W4 | `sycl_gpu_runtime.cc:320-336`, `sycl_context.cc:34-35`, `sycl_executor.cc:815-817` | Waits all queues in the ordinal's stream pool | Required for all-activity semantics; overdrain if used internally | Keep for all-activity API; avoid calling for single-stream ordering | Must keep |
| W5 | `sycl_gpu_runtime.cc:373-385` `SyclStreamPool::Reset` | Waits every stream for every device while resetting pool | Required cleanup/debug path | Wait outside broad locks if possible; keep as test cleanup | Could |
| W6 | `sycl_stream.cc:345-373` `~SyclStream` | Joins callback thread, then blocks stream before destroy | Required lifetime cleanup; destructor latency | Clarify queue ownership/refcounts before narrowing wait | Could |
| W7 | `sycl_stream.cc:397-409` `RunCallbackTask` | Callback worker waits on event | Required callback ordering, but not caller-blocking | SYCL `host_task` or stream-ordered callback command | Could |
| W8 | `sycl_timer.cc:30-77`, `:89-99` `GetElapsedDuration` | Records stop, then `zeEventHostSynchronize` | Measurement-only | Compare SYCL profiling info and low-power barrier events; keep blocking semantics | Could |
| W9 | `sycl_executor.cc:702-705` `CreateOrShareConstant` | `stream->Memcpy`, then `BlockHostUntilDone` | Accidental serialization for globally cached constant readiness | Cache ready event; wait only on consuming streams | Should |
| W10 | `collective_permute_thunk.cc:582-596`, `:648-652` `RunPeerAccessPermute` | Host sync at ready and done protocol boundaries | Workaround for cross-device event stalls | Host aggregator waiting on events without draining unrelated work; Level Zero same-context events; gated oneAPI in-order external events | Must |

## Ranked improvement plan

| Rank | Class | Wait/site | Replacement mechanism | Expected payoff | Correctness risk | Test/benchmark |
|---|---|---|---|---|---|---|
| 1 | Must | `RunPeerAccessPermute` SYCL host boundary sync | Keep fallback; prototype a feature-gated event protocol using a host aggregator or native Level Zero events. Only use `ext_oneapi_set_external_event` after proving its in-order and completion preconditions fit XLA's event lifecycle. | Restores overlap for local multi-GPU collective permute without relying on stalled cross-device barriers | Highest: source buffer may be reused before receiver reads, or sender may proceed before target read completion | Multi-GPU collective-permute watchdog test, repeated 2/4/8 GPU ring and bidirectional cases, stress with unrelated work on same streams |
| 2 | Should | H2D/D2H async wrappers for non-host-USM host pointers | Enforce `Stream::Memcpy` host allocation contract or stage through host USM and attach cleanup/copyback to stream callback | Removes host blocking from common transfer APIs and enables copy/compute overlap | Medium: non-host pointer lifetime and D2H visibility semantics must be explicit | Copy microbench with malloc, `HostMemoryAllocate`, and host USM; unit tests for correctness after delayed stream sync |
| 3 | Should | `CreateOrShareConstant` upload wait | Store `{DeviceMemoryBase, ready_event}` in constant cache and make consuming streams wait on the ready event if needed | Avoids first-use host stalls and preserves stream overlap during constant materialization | Medium-high: constants can be shared across streams; same-stream order and cross-stream readiness must be modeled | Multi-stream constant-use test; benchmark first-use latency and overlapped kernel launch |
| 4 | Should | Copy/memset helper waits | Refactor helpers to return events, with explicit synchronous wrappers waiting via `wait_and_throw()` | Reduces accidental serialization and gives later changes the event needed for callbacks/barriers | Low-medium: error propagation may change | Existing `sycl_gpu_runtime_test.cc` plus async-error injection from synchronization/error work |
| 5 | Could | Callback worker event waits | Replace worker wait with stream-ordered `host_task` when available, or keep worker but use lower-overhead event waiting | Improves callback throughput and destructor latency | Medium: callback order, status propagation, and shutdown behavior are subtle | `sycl_stream_test.cc` callback FIFO/failure/destructor tests; callback microbench with one/multiple streams |
| 6 | Could | Timer host sync | Use SYCL profiling info for comparison, or low-power event mode on barrier commands if feature macro/backend support is present | Reduces CPU spin and possibly code complexity for measurement paths | Low-medium: timestamp domains and wrap handling must stay correct | `sycl_timer_test.cc` plus timer microbench for in-flight vs completed events |
| 7 | Could | Reset/destructor/global pool waits | Keep semantics; move waits outside global locks where safe and document production vs test usage | Reduces teardown tail latency and contention | Medium: stream-pool ownership is global/shared | Stream-pool concurrency tests; destructor/reset latency microbench |
| 8 | Could | Queue submission mode | Feature-gated benchmark of immediate vs batched command-list queue properties | May improve short-kernel/copy submission behavior on supported Level Zero GPUs | Low if flag-gated; hardware-specific default would be risky | Same copy/callback/timer microbenches with default/immediate/no-immediate queue properties |

## Test/benchmark coverage

No measurements were collected for this report. Proposed coverage:

- Copy microbench: H2D, D2H, D2D; sizes 4 KiB, 1 MiB, 64 MiB; host malloc vs `StreamExecutor::HostMemoryAllocate`; one stream vs N streams; record enqueue latency, time to stream completion, and overlap with a long kernel. Run default queue mode plus immediate/no-immediate command-list variants where the feature macro is available.
- Callback microbench: enqueue N callbacks after short kernels and after long kernels; measure enqueue latency, callback completion latency, callback worker utilization, and destructor latency. Include one stream and multiple streams.
- Timer microbench: compare `SyclTimer::GetElapsedDuration` on completed vs in-flight work; compare Level Zero timestamp path with SYCL profiling queries if implemented; record CPU time while waiting, not just wall time.
- Collective microbench: local SYCL peer collective permute on 2, 4, and 8 GPUs if available; one-way ring, bidirectional pairs, and no-source zeroing; run current host-sync protocol and any prototype event protocol under a watchdog. Do not report speedups until deadlock-free repeated iterations pass.
- Unit tests: extend `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime_test.cc` for async copy pointer classes; `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc` for callback and destructor ordering; `~/github/openxla/xla/xla/stream_executor/sycl/sycl_timer_test.cc` for timer correctness; `~/github/openxla/xla/xla/backends/gpu/runtime/collective_permute_thunk_multigpu_test.cc` for SYCL local peer protocol stress.

## Rollout risk

- The collective-permute replacement is the riskiest change. The current host sync is expensive but simple: it proves source readiness before peer reads and proves copy completion before publishing done. Any event-based protocol must prove both properties.
- Removing hidden H2D/D2H waits can expose callers that pass ordinary host memory despite the `Stream::Memcpy` contract. A strict precondition is cleaner; staging is more compatible but adds host memory pressure and callback complexity.
- Constant-cache ready events can introduce cross-stream ordering bugs if the event lifetime is shorter than the cached constant or if a consumer stream skips the wait.
- Switching `queue::wait()` to `wait_and_throw()` can surface previously hidden asynchronous failures. This is desirable for correctness but may change failure timing.
- oneAPI immediate command-list behavior is backend and hardware sensitive. Local docs say the hint is meaningful on Level Zero and mainly well-tested on PVC, so default changes need data and a rollback flag.
- Intel event mode is experimental and currently limited to barrier/partial-barrier commands on Level Zero queues. It is not a general fix for every blocking event wait.

## Evidence gaps

- No runtime measurements were collected; all payoff statements are expected effects from code structure and API semantics.
- The source comment that SYCL cross-device event barriers can stall was not independently reproduced on hardware in this pass.
- Local oneAPI extension docs for immediate command lists, in-order queue events, enqueue barriers, Intel event mode, and Level Zero backend interop are present. However, `~/sycl/oneapi.md` itself only routes Level Zero interop, not immediate/event-mode docs; those were found by searching `oneapi/extensions`.
- No native Level Zero API reference for `zeEventHostSynchronize` or `zeEventQueryKernelTimestamp` was found under `~/sycl/oneapi`; only SYCL Level Zero interop docs and XLA's use of the native APIs were available locally.
- The exact production frequency of non-host-USM async copy calls and constant-cache first-use waits was not measured.
- Queue immediate vs batched mode should be tested on the target Intel GPU generation; the local doc specifically cautions against broad assumptions outside Intel Data Center Max Series GPUs.
