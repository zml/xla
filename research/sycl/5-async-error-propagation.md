# 5. Async error propagation

## Executive recommendation

XLA should make SYCL asynchronous execution failures observable through XLA status-returning synchronization APIs, and should reserve process termination for failures discovered only on no-return paths such as stream destruction, pool reset, queue/context teardown, or clearly unrecoverable device/context loss.

The must-fix design is not just replacing `queue::wait()` with `queue::wait_and_throw()`. XLA's current async handler catches `sycl::exception` and only logs it, so any async error passed to that handler is consumed without becoming an `absl::Status`. XLA needs a per-device/context async error state, preferably with queue tags, captured by queue and context async handlers. Every status-returning wait path should call `wait_and_throw()` or `event::wait_and_throw()`, then drain the captured async errors into a non-OK `absl::Status`.

Recommended policy:

- Synchronous API/setup failures remain recoverable statuses.
- Async kernel, copy, memset/fill, barrier, graph execution, and event dependency failures become recoverable `absl::Status` values when observed by `BlockHostUntilDone()`, `SynchronizeAllActivity()`, `SyclStreamSynchronize()`, `SyclStreamPool::SynchronizeStreamPool()`, `SyclEvent::Wait()`, or host-callback `error_cb`.
- Async failures first discovered in destructors, `SyclStreamPool::Reset()`, or queue/context destruction are fatal or at least executor-poisoning, because there is no safe status consumer left.
- Error states that SYCL marks undefined, such as waiting on an event after its originating queue has been destroyed, are unsupported to recover.

## Must/Should/Could classification

### Must

- Replace all status-returning queue synchronization calls that currently use `queue::wait()` with a helper that calls `queue::wait_and_throw()` and drains XLA's stored async errors.
- Ensure the same helper catches synchronous wait exceptions and converts them to statuses; `SyclStreamPool::SynchronizeStreamPool()` currently calls `wait()` directly inside a status-returning function.
- Replace synchronous helper `event.wait()` calls in copy/memset/fill completion paths with `event.wait_and_throw()` plus the same async-error drain.
- Change `SyclAsyncHandler()` from log-only consumption to durable recording of each async exception into XLA-owned state. Plain logging loses the only spec-guaranteed report point.
- Add a context-level async handler as well as queue-level handlers, or otherwise guarantee context-associated async errors are captured before context teardown.
- Make `SyclEvent::Wait()` drain stored async errors after `event::wait_and_throw()`. Current event waiting uses the right SYCL primitive, but the current handler still only logs.
- Fix no-return cleanup paths so unconsumed async errors are not silently ignored: `SyclStream::~SyclStream()` currently ignores `BlockHostUntilDone()` status, and `SyclStreamPool::Reset()` is `void`.
- Add failure-injection tests that prove async failures reach XLA statuses or intentionally terminate on no-return paths.

### Should

- Tag stored async errors with device ordinal, queue pointer, operation kind if known, and SYCL error code/category. The minimum viable state can be per device/context, but queue tags make diagnosis and stream-specific synchronization less surprising.
- Mark an executor or device context unhealthy after severe async execution faults, especially device loss or context loss, so later launches do not proceed after a known poisoned state.
- Prevalidate kernel argument pointer slots before entering the `queue::submit()` lambda. The current null argument path logs inside the command group and returns OK without launching a kernel.
- Aggregate multiple async exceptions deterministically. Return the first error as the primary status and attach/update with a count and summarized follow-on messages.
- Ensure async-handler storage is lock-safe and does not take `SyclStreamPool::stream_pool_mu_`, because handlers can run during waits and queue/context destruction.

### Could

- Add a non-consuming diagnostic poll API that reports whether a device/context has pending async errors without forcing `throw_asynchronous()`.
- Keep the event returned by `queue::ext_oneapi_graph()` in `SyclCommandBuffer::Submit()` for profiling or more precise graph-execution attribution. This is not required if stream synchronization reliably drains queue async errors.
- Explore oneAPI low-power event mode for barrier/event waits if `wait_and_throw()` overhead or CPU spin becomes measurable.

## XLA change candidates with concrete files/functions

### `xla/stream_executor/sycl/sycl_gpu_runtime.{h,cc}`

- Add `SyclAsyncErrorState`, for example a mutex-protected object with `Record(exception_list, source)` and `Consume()`/`ConsumeForDevice()` methods returning `absl::Status`.
- Replace `void SyclAsyncHandler(::sycl::exception_list)` with `MakeSyclAsyncHandler(std::shared_ptr<SyclAsyncErrorState>, AsyncErrorSource)` or equivalent. The handler should record `sycl::exception::what()`, `code()`, and source metadata, and may log after recording.
- Construct contexts in `SyclDevicePool::GetDeviceContext()` with an async handler, or store context and error state together in a per-device record. Current contexts are constructed without a handler.
- Construct queues in `SyclStreamPool::InitStreamPool()` and `GetOrCreateStream()` with handlers that capture the same per-device/context state, optionally tagged by queue pointer after construction.
- Add a queue-pointer to async-state registry, or change stream pool return types so `SyclStream` and wait helpers can find the state for a queue.
- Change `SyclStreamSynchronize()` from `stream_handle->wait()` to `stream_handle->wait_and_throw()` plus state drain.
- Change `SyclStreamPool::SynchronizeStreamPool()` from per-stream `wait()` to per-stream `wait_and_throw()` plus aggregate drain.
- Change synchronous helper waits in `MemcpyDeviceToHost()`, `MemcpyHostToDevice()`, `MemcpyDeviceToDevice()`, `MemsetDevice()`, and `MemfillDevice()` from `event.wait()` to `event.wait_and_throw()` plus queue-state drain.
- Change `SyclStreamPool::Reset()` to either return `absl::Status` or call a status-returning internal reset helper and `LOG(FATAL)`/poison the executor if async errors are found on this no-return path.
- Consider making `SyclStreamPool::DestroyStream()` synchronize/drain before queue reset when it is called directly, not only through `SyclStream::~SyclStream()`.

### `xla/stream_executor/sycl/sycl_event.cc`

- Update `SyclEvent::Wait()` to call `event_.wait_and_throw()` and then drain the device/context async error state associated with `executor_->device_ordinal()`.
- Keep `PollForStatus()` non-consuming unless XLA intentionally wants polling to consume async handler state. It currently maps `get_info()` exceptions to `Event::Status::kError`, which is reasonable for a poll API.
- `WaitStreamOnEvent()` should continue returning submission-time barrier errors, but later dependency failures must be observed by the stream/event wait that waits on the barrier.

### `xla/stream_executor/sycl/sycl_stream.cc`

- `SyclStream::BlockHostUntilDone()` should inherit the fixed `SyclStreamSynchronize()` behavior.
- `SyclStream::~SyclStream()` should not call `.IgnoreError()` on async execution failures. If no status consumer exists, log fatally or mark executor/device unhealthy before destruction completes.
- `SyclStream::RunCallbackTask()` should continue sending event-wait or callback failures to `error_cb`; after `SyclEvent::Wait()` is fixed, async event failures will flow to the callback's error path.
- `LaunchSyclKernel()` should return an error for null packed argument pointer slots before or during submit instead of only logging inside the command group lambda.

### `xla/stream_executor/sycl/sycl_command_buffer.cc`

- Submission/finalization/update paths already catch synchronous SYCL exceptions and return statuses. Execution failures in graph nodes should surface at later stream/event waits after the queue async-error state is fixed.
- Optionally retain the event from `sycl_stream->stream_handle()->ext_oneapi_graph(...)` for finer graph execution attribution. Today the return event is discarded.

## Evidence with code references and spec/oneAPI references where available

### XLA code evidence

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:216-225`: `SyclAsyncHandler()` rethrows each `exception_ptr`, catches `sycl::exception`, and only `LOG(ERROR)`. It does not store or rethrow into an XLA status.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:257-258` and `:315-316`: queues are constructed with `SyclAsyncHandler` and in-order/profiling properties.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:166-186`: per-device contexts are constructed without an async handler.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:320-336`: pool synchronization calls `stream->wait()` directly in a status-returning function, so async errors are missed and synchronous wait exceptions are not converted locally to `absl::Status`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:373-385`: pool reset calls `stream_handle->wait()` in a `void` function, then resets queues.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:422-430`: `SyclStreamSynchronize()` calls `queue::wait()`, catches only synchronous `sycl::exception`, and returns OK otherwise.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:45-53`, `:62-70`, `:79-87`, `:96-104`, and `:113-119`: synchronous copy/memset/fill helpers wait with `event.wait()`, not `event.wait_and_throw()`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:535-590` and `:610-654`: async copy/memset/fill wrappers return after enqueue for true async paths, so later synchronization must surface execution errors.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc:98-106`: `SyclEvent::Wait()` uses `event_.wait_and_throw()`, but current handler semantics still make async errors log-only unless state is drained.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc:62-90`: stream-on-event waits submit an `ext_oneapi_submit_barrier()` and return submission errors; they do not wait for barrier completion.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:118-167`: kernel launch catches submission-time exceptions. Execution-time kernel errors are deferred to later waits.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:120-124`: a null packed kernel argument pointer is only logged inside the submit lambda and returns from the lambda, leaving `LaunchSyclKernel()` to return OK.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:190-210`: stream/event waits record or submit barrier events rather than blocking immediately.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:345-367`: stream destruction shuts down callbacks, calls `BlockHostUntilDone().IgnoreError()`, then destroys the queue and only logs destroy failures.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:397-409`: host callback tasks wait on the marker event, run the callback only if the event wait succeeds, and report non-OK status to `error_cb` or `LOG(WARNING)`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_context.cc:34-35` and `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:815-817`: executor-wide synchronization inherits `SyclStreamPool::SynchronizeStreamPool()` and its current plain-wait behavior.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_command_buffer.cc:1153-1179`: graph submit catches synchronous submission exceptions but discards the returned graph event, so execution errors must surface at later stream waits.

### SYCL 2020 evidence

- `~/sycl/sycl-2020-map.md:255-257` maps async handler sections to `~/sycl/sycl-2020.html:35221-35294`.
- `~/sycl/sycl-2020.html:35221-35233`: async handlers may be triggered by `queue::wait_and_throw()`, `queue::throw_asynchronous()`, `event::wait_and_throw()`, or queue/context destruction, and receive unconsumed async errors.
- `~/sycl/sycl-2020.html:35236-35238`: once passed to an async handler, an async error instance is consumed.
- `~/sycl/sycl-2020.html:35248-35256`: without a user async handler, the default handler must report errors when possible and terminate.
- `~/sycl/sycl-2020.html:11486-11515`: queue errors split into synchronous C++ exceptions and asynchronous errors reported through async handlers at specified times.
- `~/sycl/sycl-2020.html:10890-10900`: `queue::wait()` blocks and reports synchronous errors through SYCL exceptions. It does not promise async-handler delivery.
- `~/sycl/sycl-2020.html:10903-10918`: `queue::wait_and_throw()` blocks and passes at least all unconsumed queue/context async errors to the async handler.
- `~/sycl/sycl-2020.html:10921-10935`: `queue::throw_asynchronous()` checks for unconsumed async errors and reports them to the queue or context async handler.
- `~/sycl/sycl-2020.html:11684-11699`: `event::wait_and_throw()` blocks and passes async errors held by queues or contexts associated with the event and dependent events to the appropriate handler.
- `~/sycl/sycl-2020.html:35283-35291`: if an event's originating queue has been destroyed before `event::wait_and_throw()`, behavior is undefined.

### oneAPI evidence

- `~/sycl/oneapi.md:7-12`, `:29-33`: the local route map points to oneAPI docs for command graphs, Level Zero, enqueue barriers, and in-order queue events. The `oneapi` path is a symlink to `~/github/intel/llvm/sycl/doc/`; the relevant files below exist when searched with `find -L`.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc:76-80`: enqueued barriers are non-blocking from the host perspective and operate as part of the asynchronous execution graph.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc:89-98`: `queue::ext_oneapi_submit_barrier()` returns an event, and that event completes when the barrier's dependencies complete.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc:238-240`: the returned barrier event enters complete state when implicit or explicit dependencies have completed.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_graph.asciidoc:1966-1984`: `queue::ext_oneapi_graph()` returns an event representing the graph submission and throws synchronous `invalid` exceptions for specified invalid cases.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_graph.asciidoc:1909-1916`: graph submission to an in-order queue executes in order with respect to other command groups.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_graph.asciidoc:2238-2244`: graph recording mode does not generate async exceptions because no device execution occurs; synchronous errors still throw.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_intel_event_mode.asciidoc:127-136`: low-power event mode can affect `event::wait()` and `event::wait_and_throw()` for Level Zero barrier/partial-barrier events. This is performance-relevant, not a substitute for async-error propagation.
- `~/sycl/oneapi/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc:117-130`: immediate command-list properties are Level Zero hints and conflicting properties throw synchronous `errc::invalid`.

## Findings

### Kernel failure path

Kernel submission failures are mostly status-returning today: null kernel function, null stream handle, invalid packed metadata, unsupported cluster dimensions, reflected arity mismatch, and `queue::submit()` exceptions return non-OK statuses.

Execution-time kernel failures are not status-safe today. After a kernel is submitted, XLA relies on later stream or pool synchronization. Those paths call `queue::wait()`, which the SYCL spec only ties to synchronous errors. If an async kernel failure is reported later through XLA's handler, the handler only logs and consumes it. The error can therefore be lost until queue/context teardown and still not become an XLA status.

There is also a synchronous lost-error bug in the launch path: a null packed argument pointer is detected inside the command group lambda, logged, and the lambda returns without launching. `LaunchSyclKernel()` then returns OK.

### Copy and memset/fill failure path

Immediate argument failures, such as null pointers and invalid copy kind, already return statuses. Synchronous copy wrappers submit a copy and then call `event.wait()`. That wait can miss async copy/memset/fill failures for the same reason as `queue::wait()`.

True async H2D/D2H/D2D, memset, and fill wrappers return after enqueue. Their failures must surface at a later stream/event/pool synchronization point. With current plain queue waits and log-only async handler, this is a must-fix lost-error path.

The H2D/D2H async wrappers may convert pageable or non-host-USM transfers into blocking transfers by choosing `async=false`; those blocking waits still use plain `event.wait()`, so this is also a must-fix status propagation path.

### Host callback path

Host callback user failures are not SYCL async errors. `RunCallbackTask()` reports a non-OK callback status to `error_cb` when provided, otherwise it logs a warning. That policy is acceptable for callback-returned statuses.

Event failures before the callback should become `error_cb` statuses. The structure is already present: `RunCallbackTask()` waits on the marker event and only runs the callback if the event wait succeeds. After `SyclEvent::Wait()` drains stored async errors, barrier or prior-work failures should flow to `error_cb`.

### Barrier and event wait path

Barrier submission errors are already status-returning. Completion errors are deferred: `RecordEvent()` and `WaitFor()` submit barriers/events, and errors from prior work or dependencies are observed only when an event or stream is later waited.

`SyclEvent::Wait()` uses `event::wait_and_throw()`, which is the right SYCL primitive, but the current XLA handler consumes async errors by logging. Event wait must drain stored handler state to become status-safe.

`PollForStatus()` should not be the main async-error propagation mechanism. It is a poll API, and consuming async errors there would make later blocking waits less deterministic.

### Plain wait, event wait, stream destruction, and pool reset comparison

- Plain `queue::wait()` waits for completion and reports synchronous errors. It does not guarantee async-handler delivery. Current XLA queue synchronization uses it in `SyclStreamSynchronize()`, `SyclStreamPool::SynchronizeStreamPool()`, and `SyclStreamPool::Reset()`; the pool-level paths also do not locally catch wait exceptions.
- Plain `event.wait()` waits for completion but does not guarantee async-handler delivery. Current synchronous copy/memset/fill helpers use it.
- `queue::wait_and_throw()` and `event::wait_and_throw()` trigger handler delivery of unconsumed async errors, but XLA must record and drain the handler state because current handler code catches and logs.
- Stream destruction currently ignores `BlockHostUntilDone()` status, then destroys the queue. Any async errors reported at that point have no status consumer.
- Pool reset currently calls plain `wait()` in a void function while holding the pool mutex, then destroys queues. It cannot currently return async failures as statuses.

## Proposed patch plan

1. Introduce async error storage.
   - Add `SyclAsyncErrorState` in `sycl_gpu_runtime.{h,cc}`.
   - Store errors as `absl::Status` records with SYCL `what()`, `code()`, device ordinal, queue pointer if known, and source.
   - Provide `Record(::sycl::exception_list, source)`, `Consume()`, and aggregate helpers.

2. Install handlers at context and queue construction.
   - Change `SyclDevicePool::GetDeviceContext()` to construct contexts with an async handler or return a context record containing both context and state.
   - Change queue construction in `SyclStreamPool` to capture the per-device/context state.
   - Register queue pointer to async state so `SyclStreamSynchronize(::sycl::queue*)` can drain without changing every caller immediately.

3. Centralize wait behavior.
   - Add `SyclWaitAndDrain(::sycl::queue*)` and `SyclWaitAndDrain(::sycl::event, ::sycl::queue*)` helpers.
   - Update `SyclStreamSynchronize()`, `SynchronizeStreamPool()`, synchronous copy/memset/fill helpers, and `SyclEvent::Wait()`.
   - Catch synchronous `sycl::exception` from wait calls and merge it with any recorded async errors.

4. Fix no-return cleanup paths.
   - Make `SyclStream::~SyclStream()` call the new wait helper and treat any newly discovered async execution failure as fatal or executor-poisoning instead of `.IgnoreError()`.
   - Add a status-returning reset helper, e.g. `ResetAndReport()`, and keep `Reset()` as a testing convenience that fails fast if unconsumed async errors exist.
   - Avoid blocking waits while holding `stream_pool_mu_` if practical: snapshot stream handles under lock, release, wait/drain, then reacquire to clear.

5. Fix launch-path lost synchronous errors.
   - Pre-scan `arg_ptrs` before `queue::submit()` or throw a controlled exception from the lambda and convert it to `InvalidArgument`.
   - Do not let a logged null argument slot produce an OK launch status.

6. Keep command-buffer execution tied to stream synchronization.
   - Leave synchronous graph build/update/submit exception handling as status-returning.
   - Optionally retain graph submit events for attribution, but rely on fixed stream/event waits for execution failures.

## Test/benchmark coverage

No runtime failure injection was executed in this pass. The following tests should be added rather than claiming runtime results.

### Deterministic unit tests with injected async errors

- `SyclStreamSynchronizeReturnsStoredAsyncError`: inject a synthetic async exception/status into the queue's async state, call `SyclStreamSynchronize()`, expect non-OK and state drained.
- `SynchronizeStreamPoolAggregatesAsyncErrors`: create multiple streams, inject errors on more than one queue, call `SyclStreamPool::SynchronizeStreamPool()`, expect a non-OK aggregate with both sources represented.
- `SyclEventWaitReturnsStoredAsyncError`: record an event, inject an async error associated with the device/context, call `SyclEvent::Wait()`, expect non-OK.
- `HostCallbackErrorCbReceivesEventFailure`: enqueue a host callback with `error_cb`, inject an event/queue async error before the marker completes, verify the callback body is skipped and `error_cb` receives non-OK status.
- `StreamDestructorDiesOrPoisonsOnUnconsumedAsyncError`: inject an async error with no later status consumer and destroy the stream. Expect the selected fatal/poison policy.
- `ResetDiesOrReportsOnUnconsumedAsyncError`: inject an async error before `SyclStreamPool::Reset()` or the new `ResetAndReport()` and verify the chosen no-return policy.
- `LaunchKernelNullPackedArgReturnsStatus`: pass a null packed argument pointer slot and verify `LaunchKernel()` returns `InvalidArgument` instead of OK.

### Runtime integration tests where the backend can reliably produce failures

Gate these behind an opt-in environment variable such as `XLA_SYCL_RUN_UNSAFE_ASYNC_ERROR_TESTS=1`, because some failures may poison the device context.

- Kernel failure: submit a test-only raw SYCL kernel on an XLA queue that writes through an invalid USM pointer or otherwise triggers a backend device fault. Then call `BlockHostUntilDone()` and assert either a non-OK status or the documented fatal policy for context-loss errors.
- Copy failure: enqueue a copy using a deliberately invalid or freed USM pointer. Accept an immediate enqueue status or a later sync status, but require that the failure is not only logged.
- Barrier failure: submit a failing command, record a barrier event with `RecordEvent()` or `SyclSubmitBarrierEvent()`, make another stream wait on it, and verify the waiting stream's later synchronization reports the dependency failure.
- Command graph failure: create a command buffer with a failing kernel or invalid memory operation, submit it, and verify `BlockHostUntilDone()` reports execution failure after graph submission succeeds.
- Host callback failure: enqueue a callback returning a non-OK status and verify `error_cb`; separately enqueue a callback after a failing event and verify the event status reaches `error_cb`.

### Benchmarks

- Empty `queue::wait()` versus `queue::wait_and_throw()` plus empty-state drain.
- Copy H2D/D2H/D2D plus synchronization for host USM and pageable host memory.
- Event wait latency for barrier events with and without oneAPI low-power event mode where available.
- Pool synchronization over 1, 2, 8, and 64 streams with no errors.
- Error-heavy microbenchmark that drains many recorded async exceptions to bound aggregation overhead.

## Rollout risk

- Behavior risk: existing code may implicitly rely on async errors being logged and computation continuing. Surfacing non-OK statuses can expose latent device faults earlier.
- Fatal-policy risk: destructors and reset paths need a clear policy. Fatal is safer than silent continuation after unconsumed device execution failure, but it can make tests fail more loudly.
- Attribution risk: per-device/context storage can report an error on a later wait that is not the exact stream the caller expected. Queue tags mitigate this; per-queue state plus context state is better but more invasive.
- Concurrency risk: async handlers can run during waits and object destruction. The error state must avoid deadlocks with stream-pool locks and must tolerate handler calls after partial teardown.
- Performance risk: `wait_and_throw()` plus state drain may add overhead to hot synchronization paths. The benchmark plan should quantify this before broad rollout.
- Backend variability risk: actual device fault behavior differs across Level Zero drivers and hardware. Runtime failure tests should be opt-in or carefully isolated.

## Evidence gaps

- I did not execute runtime failure-injection tests, and no local results prove how the installed Level Zero backend reports invalid kernel memory access, invalid copy pointers, or graph execution faults.
- The local oneAPI docs relevant to barriers, graph submission, event mode, immediate command lists, and Level Zero backend are present through the `oneapi` symlink when followed with `find -L`. I did not find a oneAPI document that changes the core SYCL async-error propagation rules; the core SYCL 2020 spec is the authority used here.
- The exact propagation behavior when an XLA async handler itself throws was not validated. The proposed design avoids relying on handler-thrown exceptions by recording handler input and draining it explicitly.
- The recoverability boundary for device loss, context loss, or backend reset needs empirical Level Zero testing. The proposed classification treats first observation as status-capable when a status consumer exists, then marks later use as poisoned or fatal.
- Command graph execution failure attribution is inferred from oneAPI graph submission returning an event and in-order queue semantics; runtime tests should confirm failures are delivered through the same queue/context async handler state as non-graph submissions.
