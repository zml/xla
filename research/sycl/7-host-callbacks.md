# 7. Host callbacks

## Executive recommendation

**Must fix before treating SYCL host callbacks as CUDA-equivalent.** The current SYCL implementation orders a callback after prior work by recording a barrier event, but it runs the callback on an out-of-band stream-owned host thread. That does not emulate the StreamExecutor contract that host callbacks occupy the stream: later work submitted to the same SYCL in-order queue can execute before the callback returns, and `BlockHostUntilDone()` can return after the queue barrier completes but before the callback body has completed.

Accepted callbacks should not be canceled during normal stream destruction. They are used for lifetime and completion protocols in XLA GPU code, so shutdown must close admission and then drain all callbacks that were successfully accepted. Cancellation/status delivery is still needed for callbacks that cannot be scheduled or whose dependency event fails.

Recommended direction: enqueue the callback as an actual SYCL command, preferably `handler::host_task`, or implement an equivalent stream-stalling protocol that makes callback execution part of the queue dependency chain. Pair this with explicit async-error status propagation, stronger shutdown/drain semantics, and stress tests for callback ordering, `BlockHostUntilDone()`, destruction, and failed dependencies.

## Must/Should/Could classification

**Must**

- Make `SyclStream::DoHostCallbackWithStatus()` occupy the stream, not just wait for a recorded event from a side thread. Later same-stream work must not begin until the callback returns.
- Make `SyclStream::BlockHostUntilDone()` wait for all callbacks entrained before the call, either naturally through queued `host_task` commands or by an explicit callback-drain mechanism.
- Preserve drain-on-destroy semantics for all callbacks that returned `OkStatus()` from scheduling. Stream destruction must not silently skip accepted callbacks.
- Invoke `error_cb` when callback scheduling fails, including shutdown races and barrier/event submission failures, as required by `stream_executor::Stream`.
- Propagate failed dependency/event status into the callback `error_cb` instead of only logging through `SyclAsyncHandler()`.

**Should**

- Add diagnostics: callback IDs, stream pointer/device ordinal, queue ordering property checks, scheduling/drain logs at `VLOG(2)`, and warnings when callback rejection happens after a marker event was submitted.
- Add deterministic callback-ordering tests in `sycl_stream_test.cc` and failure-injection coverage for async errors.
- Reconcile this topic with the async-error workstream: `queue::wait()` paths and an async handler that only logs are not enough for callback failure contracts.

**Could**

- Remove the per-stream callback thread if callbacks move to SYCL `host_task` plus a small registry for error observation.
- Benchmark `host_task` callbacks against the current event-waiting thread for empty callbacks, callback latency after kernels, and many-stream/many-callback stress.
- Consider cancellation only for stream-poisoned or never-accepted callbacks. Do not add normal shutdown cancellation for accepted callbacks unless XLA callers are audited and opt in.

## XLA change candidates with concrete files/functions

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - `SyclStream::DoHostCallbackWithStatus()`: replace the event-plus-host-thread scheduling model with a queued command, likely `stream_handle_->submit([&](sycl::handler& cgh) { cgh.host_task(...); })`, or add an equivalent queue-side gate. Catch scheduling exceptions and call `error_cb` before returning failure.
  - `SyclStream::BlockHostUntilDone()`: use a synchronization path that includes callback commands. If the thread model remains temporarily, add a callback-drain barrier up to the call point.
  - `SyclStream::~SyclStream()`: close callback admission, drain accepted callbacks, then synchronize/destroy the queue. Keep drain semantics; add status logging for callback drain failures.
  - `SyclStream::CallbackWorkLoop()` / `RunCallbackTask()`: retire or narrow to monitoring failed scheduled callback events. If retained, it must not be the mechanism that allows later stream work to pass a running callback.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.h`
  - Replace `callback_tasks_` thread state with queued callback bookkeeping, or add explicit in-flight accepted callback counters/notifications for `BlockHostUntilDone()` and destruction.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc`
  - `SyclEvent::Wait()`: ensure failed command/dependency status can be returned to callback scheduling, not swallowed by the queue async handler.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc`
  - `SyclAsyncHandler()`: stop treating async errors as log-only for stream/callback paths. Store status per queue/stream or rethrow through `wait_and_throw()` paths consistently.
  - `SyclStreamSynchronize()`: evaluate `wait_and_throw()` or a status-store check for stream operations whose callers need failure status.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc`
  - Add deterministic callback ordering, blocking, error callback, destruction, and stress tests.

## Evidence with code references and spec/oneAPI references where available

### XLA contract and CUDA comparison

- `stream_executor::Stream` says host callbacks "block/occupy the stream just as device functions" and execute one at a time while blocking later stream operations: `~/github/openxla/xla/xla/stream_executor/stream.h:247-250`.
- CUDA enqueues callbacks into the CUDA stream with `cuLaunchHostFunc`/`cuLaunchHostFunc_v2`: `~/github/openxla/xla/xla/stream_executor/cuda/cuda_stream.cc:467-507`.
- CUDA's callback registry cancels or fails callbacks on closed/poisoned streams and invokes `error_cb` on callback failure: `~/github/openxla/xla/xla/stream_executor/cuda/host_callback_registry.cc:53-78`, `:214-236`, `:253-261`.
- CUDA tests cover callback success/error and cross-stream ordering through stream waits: `~/github/openxla/xla/xla/stream_executor/cuda/cuda_stream_test.cc:191-203`, `:262-312`, `:314-356`.

### Current SYCL implementation

- SYCL queues are created with `property::queue::in_order()`, which is necessary for marker ordering: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:239-258`, `:309-316`.
- `RecordEvent()` records an `ext_oneapi_submit_barrier()` event, so the marker represents prior queue work on the in-order queue: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:198-203`, `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:433-444`.
- `DoHostCallbackWithStatus()` records that event, captures the callback and optional error callback, and pushes a `CallbackTask` to a deque serviced by a host thread: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:281-307`.
- The callback worker drains FIFO, waits the recorded event, then runs the callback: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:379-409`.
- `BlockHostUntilDone()` only calls `SyclStreamSynchronize()`, and that uses `queue::wait()`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:309-310`, `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:422-430`.
- The destructor sets callback shutdown, joins the callback thread, then waits for the stream and destroys the queue: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:345-373`.
- `SyclEvent::Wait()` uses `event::wait_and_throw()`, but the queue async handler currently catches and logs async exceptions: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc:98-106`, `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:216-225`.
- Existing SYCL callback tests cover only a simple callback after `BlockHostUntilDone()` and explicitly omit deterministic cross-stream callback ordering: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc:217-230`, `:340-370`.

### XLA callers that make this correctness-critical

- Host execute moves backing buffers into a callback to keep them alive after H2D copies: `~/github/openxla/xla/xla/backends/gpu/runtime/host_execute_thunk.cc:321-327`.
- Host execute start schedules host work from a stream callback and later records an event on the H2D stream for the done thunk: `~/github/openxla/xla/xla/backends/gpu/runtime/host_execute_thunk.cc:566-613`, `:674-689`.
- Outfeed signals completion from a host callback after D2H transfer: `~/github/openxla/xla/xla/backends/gpu/runtime/outfeed_thunk.cc:81-117`.

### SYCL and oneAPI references

- SYCL in-order queues guarantee same-queue commands execute in submission order and do not order other queues: `~/sycl/sycl-2020.html:11460-11478`.
- SYCL events can coordinate host/device execution; `event::wait()` blocks for associated commands and dependencies, while `wait_and_throw()` also processes async errors: `~/sycl/sycl-2020.html:3002-3019`, `:11660-11699`.
- SYCL queue destruction does not block; submitted commands continue and queue resources are released after the last command completes: `~/sycl/sycl-2020.html:10330-10339`. XLA therefore must explicitly synchronize/drain before destroying queue objects.
- SYCL host/device coordination says `queue::wait()` blocks the calling thread until submitted command groups finish, and object lifetime rules put responsibility on explicit waits for queue/context destruction: `~/sycl/sycl-2020.html:4134-4164`, `:4263-4333`.
- SYCL `host_task` is a native C++ callable scheduled by the SYCL runtime; its returned event completes once the callable returns, and the handler API enqueues it according to the dependency model: `~/sycl/sycl-2020.html:31682-31723`, `:32151-32191`.
- `interop_handle` represents queue/device/context state at the point a host task is invoked and can expose native backend objects only in that scope: `~/sycl/sycl-2020.html:31878-32150`.
- oneAPI enqueue-barrier docs exist locally and state barriers are enqueued, non-blocking from the host perspective, and prevent later same-queue command groups until wait conditions are satisfied: `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc:76-98`.
- oneAPI in-order queue events docs exist but are experimental and explicitly not recommended for general use: `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_in_order_queue_events.asciidoc:44-63`.
- oneAPI Level Zero environment docs mention host-visible/device-scope event modes and barrier command-list behavior that may affect host event waits and performance: `~/sycl/oneapi/EnvironmentVariables.md:286-288`.

## Findings

1. **Prior-work ordering is mostly satisfied today.** For an in-order queue, `RecordEvent()` submits a barrier after already enqueued work. The callback worker waits that event before invoking the callback. This is enough to avoid running the callback before prior same-queue work completes, assuming the barrier event reports failures correctly.

2. **Later-work ordering is not satisfied.** The callback body is not a SYCL command and is not part of the queue. Once the recorded barrier completes, the queue is free to execute later commands while the callback thread is still running the callback. This violates the StreamExecutor contract and differs from CUDA's stream callback path.

3. **`BlockHostUntilDone()` can race callbacks.** It waits only for the SYCL queue. Because the callback is outside the queue, a blocking callback submitted before `BlockHostUntilDone()` is not guaranteed to finish before `BlockHostUntilDone()` returns.

4. **Destructor behavior drains accepted callback tasks but in the wrong execution model.** The current destructor sets shutdown and joins the worker. The worker exits only after the deque is empty, so callbacks already in the deque are drained. However, the drain is independent of later queue work ordering, and scheduling failures during shutdown do not invoke `error_cb`.

5. **Failed dependencies are under-specified in the current path.** `RunCallbackTask()` intends to call `error_cb` if the event wait fails, but async failures can be consumed by `SyclAsyncHandler()` as log-only, while `SyclStreamSynchronize()` uses `queue::wait()` instead of `wait_and_throw()`. This can turn a dependency failure into a log without a callback status.

6. **General callback cancellation is not desirable.** Accepted callbacks are used for memory lifetime, outfeed completion, and host execute coordination. Normal shutdown should drain accepted callbacks. Cancellation is appropriate only for callbacks that were not accepted, cannot be scheduled, or are behind a poisoned stream.

7. **SYCL `host_task` is the closest standard semantic match.** It is a runtime-scheduled host callable with an event that completes after the callable returns. It naturally blocks later work on an in-order queue. The implementation still needs careful status handling because a host task that throws becomes an async error.

## Proposed patch plan

1. Add a failing regression test first: enqueue a callback that blocks on a notification, then call `BlockHostUntilDone()` from another thread. The test should assert that `BlockHostUntilDone()` does not return until the callback is released.

2. Add a same-stream later-work test: enqueue a blocking callback, enqueue a later device write/copy on the same stream, and verify the later work does not complete until the callback returns.

3. Change `DoHostCallbackWithStatus()` to submit a queue command for the callback. Preferred shape:
   - Move callback state into a heap/shared task object.
   - Submit `handler::host_task` on `stream_handle_`.
   - Inside the host task, invoke the callback exactly once, catch exceptions, and call `error_cb` on non-OK callback status.
   - Return an `InternalError` and invoke `error_cb` if the `submit` itself throws.

4. Preserve explicit admission/drain state:
   - Reject callbacks after shutdown starts.
   - Invoke `error_cb(CancelledError(...))` for rejected callbacks.
   - Track accepted callbacks if any path remains outside the queue; otherwise rely on `queue::wait_and_throw()` to drain queued `host_task` commands.

5. Upgrade error propagation:
   - Use `wait_and_throw()` or a stored async-status check in `BlockHostUntilDone()`, callback-event monitoring, and destruction paths that need status.
   - Convert prior dependency failures into `error_cb` and skip the user callback.
   - Do not throw callback-return statuses from the host task unless XLA intentionally wants callback failures to poison the SYCL queue.

6. Update destructor ordering:
   - Set shutdown to prevent new callbacks.
   - Drain accepted callbacks/queued host tasks through the stream synchronization path.
   - Destroy the queue only after drain/sync. Log any ignored destructor status with callback counts and the stream/device identity.

7. Once correctness tests pass, benchmark the new path and decide whether the callback thread can be removed entirely.

## Test/benchmark coverage

**Unit/regression tests**

- `DoHostCallbackBlocksBlockHostUntilDone`: callback waits on a notification; `BlockHostUntilDone()` must remain pending until release.
- `DoHostCallbackBlocksLaterSameStreamWork`: later same-stream device write/copy must not complete while callback is blocked.
- `DoHostCallbackRunsAfterPriorWork`: prior device write/copy must be visible to callback.
- `DoHostCallbackWithStatusSuccessAndError`: mirror CUDA success/error tests and assert `error_cb` receives callback-returned status.
- `DoHostCallbackSchedulingFailureCallsErrorCb`: force rejection after shutdown or through a test hook and assert `error_cb` is called.
- `DestroyStreamDrainsPendingCallback`: destroy stream while a callback is blocked; destructor must wait, then complete after release.
- `FailedDependencyCallsErrorCb`: enqueue a failing command before callback and assert the user callback is skipped and `error_cb` receives a failure status.
- `ManyCallbacksFifoSameStream`: submit many callbacks on one stream and assert FIFO order.
- `MultipleStreamsNoGlobalOrder`: retain no deterministic order across independent streams, but use locking to avoid test data races.

**Stress tests**

- Repeated stream create/destroy with pending callbacks.
- Concurrent callback submission and stream destruction, verifying no accepted callback is lost and rejected callbacks report failure.
- Multi-GPU stress with callbacks on independent devices and with host execute/outfeed style callback bodies.

**Benchmarks**

- Empty callback latency after an empty barrier, after a short kernel, and after a D2H copy.
- Throughput for many callbacks per stream and many streams per device.
- `host_task` callback path under Level Zero immediate command lists versus batched command lists.
- Callback overhead impact on host execute and outfeed microbenchmarks.

## Rollout risk

- Making callbacks occupy the stream can reduce overlap that the current implementation accidentally allowed. That overlap is not contract-compliant, but workloads may see timing/performance changes.
- `host_task` runs on SYCL runtime-managed host execution resources. XLA callback bodies that block for long periods can occupy those resources; this matches stream semantics but should be measured.
- Changing `queue::wait()`/async-error handling can surface failures that were previously logged and ignored. This is a correctness improvement with possible test fallout.
- If callback status is represented by throwing from `host_task`, it may poison the queue differently from CUDA. Prefer catching callback status and calling `error_cb` unless a separate XLA decision says callback failure should fail stream synchronization.
- Capturing or using SYCL reference-semantics objects inside a host task has SYCL lifetime restrictions. The patch should capture plain callback state and avoid capturing `sycl::queue`, `sycl::event`, or accessors unless required and verified.

## Evidence gaps

- No runtime experiment was run in this pass; conclusions are from local XLA source and local SYCL/oneAPI documents.
- Need empirical confirmation on DPC++/Level Zero for a host task submitted after a failed dependency: whether it is skipped, whether its event fails, and how `wait_and_throw()` plus XLA's async handler reports the failure.
- Need a direct audit of XLA callback bodies to ensure running them from a SYCL `host_task` thread does not violate SYCL reference-semantics capture restrictions or create new deadlocks.
- The oneAPI docs referenced by `oneapi.md` are present locally for enqueue barriers, in-order queue events, Level Zero backend, and environment variables. I did not find a oneAPI-specific document that defines callback cancellation, stream-destruction drain semantics, or a Level Zero host-callback equivalent for SYCL queues; those remain implementation behavior to test.
- Need Level Zero measurements under `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS`, immediate command lists, and multiple-command-list barriers because host-visible event/proxy behavior may change callback latency and failure timing.
