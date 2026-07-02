# 6. Event semantics and stream waits

## Executive recommendation

Treat `SyclEvent` as an XLA event state machine, not as a raw always-present `sycl::event`.

The current wrapper is close for recorded stream markers, but it conflates "never recorded" with "recorded and complete" by storing a default-constructed `sycl::event`. SYCL 2020 explicitly says a default `sycl::event` is immediately ready, has no dependencies or commands, waits immediately, and reports `complete`; that is a valid SYCL object but it is not a recorded XLA stream marker. XLA should preserve StreamExecutor's no-op semantics for waits on unrecorded events, but it should model that as an explicit wrapper state.

Top classification:

- **Must**: represent default-created `SyclEvent` as unrecorded, with unrecorded waits as no-ops and unrecorded polling as `kComplete` for StreamExecutor compatibility.
- **Must**: validate external queue waits enough to reject null, wrong-backend, and known-incompatible queue/event combinations before submitting a barrier; record event provenance when `RecordEvent()` stores a SYCL event.
- **Must**: add a real `Event::Synchronize()` override and define failed-event behavior through `wait_and_throw()` plus async-error capture. Polling alone does not surface SYCL async failures.
- **Should**: gate cross-device or cross-context event waits on targeted Level Zero tests. The oneAPI barrier extension says wait-list queues have no context constraints, but XLA currently creates one SYCL context per device, and the multi-GPU route map already suspects cross-device barrier stalls.
- **Could**: evaluate the experimental in-order queue event and Intel event-mode extensions after correctness is locked down. Do not use them as correctness dependencies.

## Must/Should/Could classification

**Must**

- Replace the raw default `sycl::event` sentinel in `SyclEvent` with explicit unrecorded state, most likely `std::optional<::sycl::event>` plus recording metadata. This matches the TODO in `sycl_event.h` and prevents default-selector backend/context details from leaking into XLA state.
- Keep StreamExecutor compatibility: an unrecorded event is considered complete for `Stream::WaitFor(Event*)` and host wait APIs. `WaitFor(unrecorded)` returns OK and enqueues nothing.
- Provide a recorded-only accessor, for example `GetRecordedEvent()` returning `absl::StatusOr<::sycl::event>`, so timer/profiling paths cannot silently use an unrecorded event.
- Add `SyclEvent::Synchronize() override`, implemented as the same policy as `Wait()`: unrecorded returns OK; recorded uses `event.wait_and_throw()` and returns an error if SYCL reports one.
- Stop swallowing asynchronous errors at the XLA boundary. `SyclAsyncHandler()` currently logs async exceptions and consumes them. Failed recorded events need a path to become an `absl::Status` and, once known, `PollForStatus()` should be able to return `kError`.
- Validate `WaitForEventOnExternalStream(std::intptr_t)` inputs. The API can still require the integer to encode a valid `sycl::queue*`, but after casting it should validate null, backend, and, where available from recorded metadata, context/device compatibility. Invalid or unsupported combinations should return `InvalidArgument` or `FailedPrecondition`, not attempt a blind barrier.

**Should**

- Store event provenance on `SetEvent()`/`RecordEvent()`: recording executor, device ordinal, queue pointer or queue identity, `sycl::context`, `sycl::device`, backend, and whether the event was produced by XLA's barrier marker.
- Change `WaitStreamOnEvent()` to accept a `SyclEvent` or recorded-event metadata wrapper rather than a bare `sycl::event`. Bare SYCL events are insufficient for validation and for unrecorded handling.
- Add targeted cross-queue and cross-device tests before declaring cross-device waits supported. Same executor/same context should be required to pass. Different contexts and different devices should either pass with `ext_oneapi_submit_barrier(waitList)` or fall back to a host wait protocol.
- Use `queue::wait_and_throw()` or `queue::throw_asynchronous()` at synchronization boundaries outside the event wrapper, especially `BlockHostUntilDone()` and stream-pool synchronization. This overlaps the async-error research section, but event failure tests depend on it.
- Keep the existing "already complete, submit no barrier" optimization only after async-error policy is defined. A complete status is a readiness fact, not proof that no asynchronous error exists.

**Could**

- Evaluate `sycl_ext_oneapi_in_order_queue_events` for in-order queue event plumbing (`ext_oneapi_get_last_event()` and `ext_oneapi_set_external_event()`), but the extension is experimental and explicitly not recommended for general use.
- Evaluate `sycl_ext_intel_event_mode` low-power barrier events for host callback and teardown waits after correctness tests pass.
- Add benchmark-only instrumentation for barrier event status polling, host fallback waits, and low-power event waits.

## XLA change candidates with concrete files/functions

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.h`
  - Lines 45-57: change `Create()` to create an unrecorded wrapper, not a wrapper around `::sycl::event()`.
  - Lines 49-56: replace `GetEvent()` or add `GetRecordedEvent()` so callers must handle unrecorded state.
  - Lines 69-78: replace `::sycl::event event_` with `std::optional<::sycl::event>` and an `EventRecordMetadata` struct.
  - Add `absl::Status Synchronize() override`.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc`
  - Lines 29-60: update `PollForStatus()` to branch on unrecorded first. Unrecorded returns `Event::Status::kComplete`; recorded polls `command_execution_status`; known async failure returns `kError`.
  - Lines 62-91: change `WaitStreamOnEvent()` to take a wrapper/metadata object, return OK for unrecorded, validate the queue, and submit `ext_oneapi_submit_barrier({event})` only for recorded pending events.
  - Lines 93-96: make external-stream wait validate the cast `sycl::queue*` before use. At minimum: non-null, backend query succeeds, backend is the expected SYCL backend, and recorded-event metadata is compatible with the queue policy.
  - Lines 98-106: rename or reuse `Wait()` as the implementation of `Synchronize()`, but make async-error propagation explicit.
  - Lines 108-114: `Create()` should not allocate a default SYCL event as the persistent state.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - Lines 190-210: update stream-to-stream and stream-to-event waits to pass `SyclEvent` metadata rather than `GetEvent()`.
  - Lines 198-203: `RecordEvent()` should call a `SyclEvent::SetRecordedEvent(barrier_event, metadata)` helper.
  - Lines 281-307 and 397-409: host callback waits should call the synchronized event API and propagate failures.
  - Lines 309-311: `BlockHostUntilDone()` should use a queue wait path that surfaces async errors.
  - Lines 375-376: `RecordCompletedEvent()` remains the stream-to-stream marker point.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc`
  - Lines 216-224: `SyclAsyncHandler()` should record/report an `absl::Status` to XLA instead of only logging.
  - Lines 227-318: queue creation already uses `enable_profiling` and `in_order`; keep same-context stream tests tied to these queues.
  - Lines 320-336 and 422-430: stream-pool and stream synchronization should use `wait_and_throw()` or explicit async error reporting.
  - Lines 433-444: keep `SyclSubmitBarrierEvent()` as the recorded-marker primitive.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc`
  - Lines 854-856: `CreateEvent()` should return an unrecorded `SyclEvent`.
  - Lines 876-965: peer-access checks are memory-access checks, not event-wait checks, but cross-device wait tests should report peer-access capability alongside barrier behavior.

- Tests in `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event_test.cc` and `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc`
  - Update the current default-created event test so it asserts wrapper state (`!IsRecorded()`), not that a stored default SYCL event is the implementation.
  - Add unrecorded, recorded, failed, external-stream, and cross-device cases described below.

## Evidence with code references and spec/oneAPI references

XLA contracts and current implementation:

- `Event` is meaningful after `Stream::RecordEvent()`: `~/github/openxla/xla/xla/stream_executor/event.h` lines 25-28 say an event is inserted with `RecordEvent()` and "from then on" status can be monitored. Lines 31-39 define `kUnknown`, `kError`, `kPending`, and `kComplete`.
- `Stream::WaitFor(Event*)` explicitly permits unrecorded waits as complete/no-op: `~/github/openxla/xla/xla/stream_executor/stream.h` lines 133-136.
- `SyclEvent::Create()` currently stores a default-constructed `::sycl::event`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.h` lines 45-47 and `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc` lines 108-114.
- The source already flags the modeling issue: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.h` lines 75-78 has a TODO to use `std::optional<::sycl::event>` for unrecorded events.
- `PollForStatus()` only queries `info::event::command_execution_status` and maps submitted/running to pending, complete to complete, SYCL exceptions to error: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc` lines 29-60.
- `WaitStreamOnEvent()` checks status first and returns immediately if complete; otherwise it submits `ext_oneapi_submit_barrier({event})`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc` lines 62-91.
- External waits blindly bit-cast the integer to `sycl::queue*`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event.cc` lines 93-96.
- Stream-to-stream and stream-to-event waits use recorded barrier events: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc` lines 190-210.
- `RecordEvent()` stores the event returned from `SyclSubmitBarrierEvent()`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc` lines 198-203 and `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 433-444.
- XLA SYCL queues are in-order and profiling-enabled: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 239-258 and 309-316.
- XLA creates one `sycl::context` per device ordinal: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 166-186. Therefore current multi-device waits are normally different-context waits.
- The async handler only logs: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 216-224.
- `BlockHostUntilDone()` uses `queue::wait()`, not `wait_and_throw()`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc` lines 309-310 and `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 422-430.
- Generic callers can call `Event::Synchronize()`: `~/github/openxla/xla/xla/service/gpu/gpu_executable_buffer_allocator.cc` lines 346-348. `SyclEvent` currently has `Wait()` but does not override `Synchronize()`.
- Existing tests encode current behavior: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_event_test.cc` lines 42-50 expects a newly-created event to poll complete; `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc` lines 320-335 waits on an unrecorded event as a no-op; lines 393-399 verify `RecordEvent()` replaces the default event and then waits on it.

SYCL 2020 evidence from local `~/sycl/sycl-2020.html` and `~/sycl/sycl-2020-map.md`:

- Route-map anchors used: event class `S|4|11516-12032`, event constructor `A|11609`, event wait `A|11660`, event status `A|11818`, complete status `A|12014`, execution order `S|5|2834-3001`, event dependencies `S|5|3002-3023`.
- A default `sycl::event` is immediately ready, has no dependencies or associated commands, waits immediately, and returns complete status: `~/sycl/sycl-2020.html` lines 11616-11620.
- The default event is constructed as though from a default queue, so its backend follows the default selector: `~/sycl/sycl-2020.html` lines 11622-11626. This is another reason not to use it as an XLA executor-bound recorded marker.
- `event::wait()` blocks until associated commands and dependencies complete: `~/sycl/sycl-2020.html` lines 11660-11669.
- `event::wait_and_throw()` also passes unconsumed async errors for queues/contexts used by associated commands to async handlers: `~/sycl/sycl-2020.html` lines 11672-11699.
- Event status returns submitted/running/complete: `~/sycl/sycl-2020.html` lines 11818-11844. Complete means the command finished and waiting will not block; `get_info()` returning complete has synchronization equivalent to `event::wait()`: lines 12014-12024.
- Events are the standard way to order command groups: `~/sycl/sycl-2020.html` lines 3002-3019 and `handler::depends_on()` lines 29560-29618.
- `property::queue::in_order` orders only commands in the same queue and gives no ordering across other queues: `~/sycl/sycl-2020.html` lines 11460-11478.
- `queue::wait_and_throw()` and `queue::throw_asynchronous()` are the queue async-error reporting paths: `~/sycl/sycl-2020.html` lines 10903-10918 and 10921-10935.
- Async handlers are invoked by `queue::wait_and_throw`, `queue::throw_asynchronous`, `event::wait_and_throw`, or destruction: `~/sycl/sycl-2020.html` lines 35221-35229.

oneAPI evidence from local docs:

- The relevant oneAPI docs are present locally despite the route-map caveat: `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc`, `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_in_order_queue_events.asciidoc`, and `~/sycl/oneapi/extensions/experimental/sycl_ext_intel_event_mode.asciidoc`.
- `sycl_ext_oneapi_enqueue_barrier` defines non-blocking queue barriers; later commands in the same queue cannot execute until implicit or explicit wait conditions are met: `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc` lines 76-99.
- The same extension states the `waitList` barrier has no context constraints for participating queues: `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc` lines 288-300.
- The in-order queue events extension adds `ext_oneapi_get_last_event()` and `ext_oneapi_set_external_event()`, but is experimental and "not recommended for general usage": `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_in_order_queue_events.asciidoc` lines 44-63 and 118-154.
- The Intel event-mode extension is experimental and affects low-power waits mainly for Level Zero barrier/partial-barrier commands: `~/sycl/oneapi/extensions/experimental/sycl_ext_intel_event_mode.asciidoc` lines 53-74 and 127-136.
- The peer-access extension is memory-oriented: it says P2P memory access is only possible between devices from the same backend and USM device peer allocations are for the same context: `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_peer_access.asciidoc` lines 47-53 and 87-93. This does not prove event wait behavior, but it is useful metadata for cross-device tests.

## Findings

1. **The current default-created wrapper follows SYCL default-event semantics but not an explicit XLA unrecorded state.** A default `sycl::event` is complete by spec, so the current `CreateEvent` and existing tests are understandable. The problem is that XLA cannot tell whether an event was never recorded or was recorded and completed.

2. **Unrecorded waits should stay no-op.** StreamExecutor explicitly says `WaitFor(Event*)` before `RecordEvent()` considers the event complete and does nothing. The fix should not turn unrecorded waits into errors.

3. **Recorded stream waits are modeled as barrier events.** `RecordEvent()` records an `ext_oneapi_submit_barrier()` marker, and `WaitFor()` submits another barrier with that marker in its wait list. This is the right basic shape for same-device/same-context queues.

4. **External waits are under-validated.** `WaitForEventOnExternalStream()` accepts an integer and bit-casts to `sycl::queue*`. Unlike CUDA/ROCm raw stream handles, a bogus SYCL queue pointer can be an unsafe C++ object pointer. XLA needs a documented external stream contract and runtime validation for any pointer it dereferences.

5. **Cross-device waits are not proven.** oneAPI's barrier spec permits wait-list events from queues in any context, but XLA's per-device context policy means multi-device stream waits are cross-context by default. This is exactly the class of behavior that needs runtime tests on Level Zero root devices and subdevices.

6. **Polling hides asynchronous failures unless there is separate async-error state.** `PollForStatus()` only queries event readiness. SYCL async errors are reported through `wait_and_throw`, `throw_asynchronous`, or destruction. Current `SyclAsyncHandler()` logs and consumes exceptions, so even `wait_and_throw()` may return OK unless XLA records or rethrows the async error.

7. **`SyclEvent` is missing the generic synchronization override.** It has `Wait()`, but generic XLA code calls `Event::Synchronize()`. Today that falls back to the base unimplemented method for SYCL events.

## Proposed patch plan

1. Add explicit state to `SyclEvent`.
   - `Create()` returns `{state = unrecorded, event_ = nullopt}`.
   - `SetRecordedEvent(event, metadata)` transitions to recorded and stores queue/device/context/backend provenance.
   - Move construction and assignment preserve state and metadata.

2. Define the wrapper API contract in comments near `SyclEvent`.
   - **Unrecorded**: created but never recorded. `PollForStatus()` returns `kComplete`; `Wait()`, `Synchronize()`, stream waits, and external waits return OK without enqueuing work; `GetRecordedEvent()` returns `FailedPrecondition`.
   - **Recorded**: contains a SYCL event returned by a stream command or barrier. `PollForStatus()` reports readiness. Stream waits submit a queue barrier unless the event is already complete.
   - **Failed**: a recorded event whose queue/context async handler has reported an error. `PollForStatus()` returns `kError` after the error is known. `Wait()`/`Synchronize()` return the stored error.
   - **Cross-device**: allowed only for validated queue/event combinations. Unsupported combinations return a status or use an explicit host fallback; they must not silently rely on untested barrier behavior.

3. Change `WaitStreamOnEvent()`.
   - Accept `const SyclEvent&` or a `RecordedEventRef` containing the event and metadata.
   - Return OK immediately for unrecorded.
   - Validate `stream_handle != nullptr`.
   - For recorded events, query backend/context/device from the target queue inside a `try` block. If target queue metadata is incompatible with the event policy, return an error.
   - Submit `stream_handle->ext_oneapi_submit_barrier({recorded_event})` for pending recorded events.

4. Fix synchronization and async-error propagation.
   - Add `SyclEvent::Synchronize() override`.
   - Have `Wait()` call `Synchronize()` or vice versa.
   - Change `SyclAsyncHandler()` and queue construction so async errors are available to XLA as an `absl::Status`, not only as logs.
   - Update `SyclStreamSynchronize()` and stream-pool synchronization to use a wait path that processes async errors.

5. Preserve and update current tests.
   - Keep the no-op wait test, but assert wrapper unrecorded state.
   - Update `RecordEvent` tests to assert transition to recorded rather than comparison with a default SYCL event.

6. Add cross-device policy tests before enabling different-context waits by default.
   - If Level Zero supports the wait reliably, allow it and document the tested matrix.
   - If it stalls or fails, implement a host-side fallback: wait on the source event with `wait_and_throw()` from a helper host thread or synchronization point, then enqueue a target-queue barrier/marker.

## Test/benchmark coverage

Unit tests to add or update:

- `SyclEventTest.CreateEventIsUnrecorded`: `Create()` yields `!IsRecorded()`, `PollForStatus() == kComplete`, `Synchronize()` OK, and `GetRecordedEvent()` fails with `FailedPrecondition`.
- `SyclStreamTest.WaitForUnrecordedEventIsNoOp`: waiting on a fresh event returns OK and does not change recorded state.
- `SyclStreamTest.RecordEventTransitionsToRecorded`: after `RecordEvent()`, `IsRecorded()` is true and `GetRecordedEvent()` succeeds.
- `SyclEventTest.SynchronizeOverrideFromBasePointer`: create `std::unique_ptr<Event>`, record it, call `event->Synchronize()`, and expect OK. This guards the current missing override.
- `SyclStreamTest.RecordedEventOrdersQueuesSameContext`: two XLA streams on one executor; stream A writes/fills a buffer, records an event, stream B waits and reads the value.
- `SyclEventTest.ExternalStreamRejectsNull`: `WaitForEventOnExternalStream(0)` returns an error.
- `SyclEventTest.ExternalStreamValidatesQueue`: pass a valid `sycl::queue*` from the same backend/context and assert a recorded wait succeeds. Add wrong-backend or wrong-device/context cases when such queues are available.

Failed-event tests:

- Add a test-only async handler or queue error sink that records the first async exception as `absl::Status`.
- Submit a kernel or command designed to produce an async device error on Level Zero, record a barrier event after it, then verify:
  - `PollForStatus()` may report pending/complete before async error consumption.
  - `Synchronize()` or `BlockHostUntilDone()` returns non-OK.
  - Once the error is recorded by XLA, `PollForStatus()` returns `kError`.
- If reliable device-fault injection is unavailable, add a lower-level fake async-error sink test for wrapper state transitions, and keep the real device-fault test as hardware/manual.

Cross-device and subdevice tests:

- `SameDeviceDifferentQueuesSameContext`: direct SYCL queues constructed from the same `SyclDevicePool::GetDeviceContext(0)` and device.
- `SameDeviceDifferentContexts`: two independent `sycl::context(device)` objects and queues for the same root device; source event in one context, target barrier in the other.
- `DifferentRootDevicesDifferentContexts`: XLA executor 0 stream event waited on by executor 1 stream. Skip if fewer than two Level Zero GPUs. Include a timeout to catch stalls.
- `DifferentRootDevicesSharedContext`: construct a manual `sycl::context({device0, device1})` if supported and compare with XLA's per-device context behavior.
- `Subdevices`: if `create_sub_devices` is supported, test same root/different subdevices and different root/subdevice combinations. Skip with a clear reason otherwise.

Benchmarks:

- Barrier wait latency for same queue, same device/different queues, different contexts, different devices, root/subdevice.
- Polling overhead and completion latency for recorded barrier events.
- Host fallback latency versus device-side `ext_oneapi_submit_barrier(waitList)`.
- CPU cost of host waits with default events versus Intel low-power event mode where supported.

## Rollout risk

- Explicit unrecorded state is a low-risk correctness change if no-op wait and `PollForStatus() == kComplete` are preserved for unrecorded events.
- Changing async-error handling is higher risk because current behavior logs and continues. Returning non-OK or terminating earlier can expose existing latent failures.
- Cross-context and cross-device event waits can deadlock or stall on some Level Zero configurations. They need timeout-protected tests before default enablement.
- External queue validation cannot make an arbitrary integer safe to dereference. The API contract must say the caller supplies a valid `sycl::queue*`; XLA can validate semantic compatibility only after that pointer is safe to use.
- Replacing `GetEvent()` with `GetRecordedEvent()` touches timer and profiling code. Tests around `SyclTimer` should run with event changes.
- The in-order queue events and Intel event-mode extensions are experimental; using them for required correctness would add API stability risk.

## Evidence gaps

- No SYCL runtime tests were run in this research pass. Cross-context, cross-root-device, and subdevice event-barrier behavior still needs empirical validation on the target Intel Level Zero systems.
- The local oneAPI docs exist for enqueue barriers, in-order queue events, and Intel event mode, but the enqueue-barrier document still labels itself "Final Draft" and warns preview interfaces were not intended for shipping software. XLA already uses `ext_oneapi_submit_barrier()`, so this is a stability note, not a blocker by itself.
- I did not inspect native Level Zero event interop or `ext_oneapi_set_external_event` implementation behavior. Those are candidates only if barrier wait tests fail.
- The exact behavior of `event::get_info(command_execution_status)` after a device-side failure is not established here. The spec does not make it an async-error reporting path, so failed-event propagation should not depend on polling.
- Root-device versus subdevice enumeration policy in `platform.get_devices()` was not proven from the source alone. Tests should log whether each SYCL device is a root device or subdevice before running cross-device cases.
