# 4. Queue and stream model

## Executive recommendation

Model XLA streams as independent in-order SYCL queues. This is the right semantic match for XLA's stream contract: work submitted to one stream is FIFO, while work on separate streams can overlap until an explicit stream/event wait creates an ordering edge. A single shared in-order SYCL queue would preserve FIFO but would serialize independent XLA streams and defeat the intended kernel/copy and compute/communication overlap.

Top classification: **Should change the stream-pool policy, Must preserve the independent-queue model**. The current implementation is semantically defensible, but the pool defaults are too blunt: every production `CreateStream()` allocates an independent profiling-enabled queue, and the cap is a hard-coded 4096 queues per device. Keep the hard cap, add stream-count/profiling benchmarks, make the cap and queue properties configurable, then lower or specialize defaults only when measurements justify it.

## Must/Should/Could classification

| Class | Recommendation | Rationale |
| --- | --- | --- |
| Must | Keep production XLA streams as independent in-order SYCL queues. | SYCL `property::queue::in_order` orders only commands submitted to the same queue; it explicitly gives no ordering guarantee against other queues. That matches XLA streams plus explicit waits. |
| Must | Do not collapse all XLA streams onto the SYCL default queue. | One in-order queue would serialize unrelated XLA streams and break intended overlap for async compute, communication, D2H/D2D helper streams, sub-streams, command-buffer tracing, and callback streams. |
| Must | Keep a hard maximum stream count and keep returning `ResourceExhausted` when exceeded. | Removing the cap would allow unbounded queue creation from PJRT helper streams, service stream pools, sub-streams, and async executable streams. |
| Must | Add overlap and queue-pressure benchmarks before changing default cap, profiling, or immediate/batched command-list policy. | This worker environment has no runnable SYCL runtime (`sycl-ls` and `icpx` were not found), so there is no local measurement basis for changing performance defaults. |
| Should | Replace `kMaxStreamsPerDevice = 4096` with a configurable SYCL queue cap and warning threshold. | 4096 is a safety valve, not a measured operating point. A candidate default of 256 or 512 should be accepted only if benchmark and workload runs show no `ResourceExhausted` and no throughput regression. |
| Should | Factor SYCL queue property construction and benchmark profiling-on versus profiling-off queues. | XLA currently enables `property::queue::enable_profiling` on every queue, but timers/profilers are not active on every stream. Keep profiling default-on until benchmarks prove selective profiling is safe. |
| Should | Make the `enable_multiple_streams=false` path explicitly non-production or non-owning. | The public SYCL executor override always requests multiple streams, but internal/test paths can create wrappers around the default queue. That path should not accidentally become a shared owning stream model. |
| Should | Add live queue telemetry: per device current queues, high-water mark, queue creation latency, and pool cap hits. | This provides evidence for tuning the cap and diagnosing scheduler pressure without changing semantics. |
| Could | Expose an XLA debug option to add oneAPI `immediate_command_list` or `no_immediate_command_list` queue properties. | oneAPI already provides environment controls, and the extension docs say applications should usually rely on the default. Hard-coding a policy in XLA should wait for device-specific wins. |
| Could | Experiment with `sycl_ext_oneapi_in_order_queue_events` to reduce marker/barrier overhead. | The extension can expose the last event of an in-order queue and set an external event dependency, but it is experimental and not recommended for general use. |
| Could | Split pools by queue purpose, such as compute, copy, callback, profiling, and tracing. | This may help if Level Zero copy-engine or profiling behavior differs by queue purpose, but it requires benchmark evidence. |

## XLA change candidates with concrete files/functions

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.h`
  - `kMaxStreamsPerDevice` at lines 83-91: replace the fixed constant with a defaulted runtime/debug option plus a hard upper bound.
  - `SyclStreamPool::GetOrCreateStream` declaration at lines 105-114: consider accepting a small queue-options struct if profiling or immediate/batched policy becomes configurable.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc`
  - `SyclStreamPool::InitStreamPool` at lines 227-264 and `GetOrCreateStream` at lines 284-318: factor `property_list` creation, add queue creation timing/logging, use the configurable cap, and optionally include oneAPI command-list properties behind a debug option.
  - `SyclStreamPool::SynchronizeStreamPool`, `DestroyStream`, and `Reset` at lines 320-384: add high-water mark and live-count accounting around queue creation/destruction.
  - `SyclSubmitBarrierEvent` at lines 433-444: keep this as the explicit marker event used to preserve cross-stream waits unless a benchmarked in-order queue event replacement proves better.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.h`
  - `SyclExecutor::CreateStream(priority)` at lines 137-141: the public override currently maps all production calls to `CreateStream(/*enable_multiple_streams=*/true, priority)`. Keep that behavior; clarify that `false` is not the normal XLA stream path.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc`
  - `SyclExecutor::CreateStream` at lines 844-852: add queue creation telemetry and keep `alive_gpu_streams_` keyed by raw queue handle.
  - `SyclExecutor::DeallocateStream` at lines 870-874: update telemetry on stream destruction.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - `SyclStream::Create` at lines 313-343: keep independent queue creation for public streams, but make the default-queue path visibly restricted.
  - `SyclStream::~SyclStream` at lines 345-373: if the default-queue path remains, avoid treating shared default-queue wrappers as unique queue owners.
  - `WaitFor`, `RecordEvent`, and `WaitFor(Event*)` at lines 190-210: preserve explicit barrier/event dependencies across independent queues.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_timer.cc`
  - `SyclTimer::Create` and `GetElapsedDuration` at lines 89-108: any profiling-off queue experiment must prove these timer paths still work or must force profiling-capable queues where timers can be created.

- Tests and benchmarks:
  - Extend `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime_test.cc` around `TestMaxStreamsPerDevice` lines 212-228 for configurable caps.
  - Extend `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc` around `MultipleStreams` lines 345-371 and `RecordEvent` starting at line 373 for independent-stream ordering and wait correctness.
  - Add a SYCL queue benchmark target, for example under `xla/stream_executor/sycl/` or `xla/tools/`, that sweeps queue count, profiling, copy/compute overlap, and immediate/batched modes.

## Evidence

### XLA stream-pool behavior

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.h:79-91` defines `StreamPtr` as `std::shared_ptr<::sycl::queue>`, stores pools as vectors per device ordinal, and sets `kMaxStreamsPerDevice = 4096`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.h:105-114` documents the current split: without multiple streams, return the default queue; with multiple streams, create a new queue up to the cap.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:227-264` initializes a per-device pool with one `sycl::queue` constructed from the device/context and a property list containing `enable_profiling` and `in_order`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:284-318` returns the default stream when `enable_multiple_streams` is false; otherwise it creates and appends a new profiling-enabled in-order queue until `kMaxStreamsPerDevice`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:313-343` validates priority, calls `SyclStreamPool::GetOrCreateStream`, creates a completed event, and wraps the queue in `SyclStream`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.h:137-141` makes the public SYCL `CreateStream(priority)` override call `CreateStream(/*enable_multiple_streams=*/true, priority)`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:844-852` stores live streams in `alive_gpu_streams_` keyed by the underlying raw queue handle.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:190-210` implements stream waits and event waits by recording an event on the source queue and submitting a barrier on the target queue.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:433-444` uses `queue::ext_oneapi_submit_barrier()` to record a marker event for prior queue work.

### XLA callers that request multiple streams

Production code does not call the SYCL-only `CreateStream(false, priority)` overload. It calls the generic `StreamExecutor::CreateStream()`, which routes to the SYCL override and therefore requests `enable_multiple_streams=true`.

- `~/github/openxla/xla/xla/stream_executor/stream_executor.h:101-106` defines the generic `CreateStream()` interface.
- `~/github/openxla/xla/xla/service/backend.cc:145-151` creates one stream per executor for GPU allocator setup.
- `~/github/openxla/xla/xla/service/backend.cc:209-238` exposes `BorrowStream` and `BorrowStreams`; `~/github/openxla/xla/xla/service/stream_pool.cc:56-64` creates a new stream with `executor_->CreateStream(priority)` when the service pool is empty.
- `~/github/openxla/xla/xla/pjrt/local_device_state.cc:100-118` creates compute and host-to-device streams; lines 124-149 create device-to-host, device-to-device, fixed-size-pool, and external-ready-event helper streams. Defaults in `~/github/openxla/xla/xla/pjrt/local_device_state.h:265-268` are four each for D2H, D2D, fixed-size-pool, and external-ready-event streams.
- `~/github/openxla/xla/xla/pjrt/local_device_state.cc:263-282` lazily creates callback streams with `executor_->CreateStream()`.
- `~/github/openxla/xla/xla/pjrt/local_device_state.cc:372-377` creates pooled usage streams on demand.
- `~/github/openxla/xla/xla/pjrt/gpu/se_gpu_pjrt_client.cc:1330-1343` builds `LocalDeviceState` with default stream options for each addressable GPU executor.
- `~/github/openxla/xla/xla/service/gpu/gpu_executable.cc:246-274` computes additional runtime stream counts from debug options and async thunks; `AsyncStreamKind` has four slots in `~/github/openxla/xla/xla/xla_data.proto:177-185`.
- `~/github/openxla/xla/xla/service/gpu/gpu_executable.cc:669-701` borrows command-buffer trace, communication, and compute streams at execution time when a stream borrower is available.
- `~/github/openxla/xla/xla/backends/gpu/runtime/async_thunk.cc:82-109` dispatches nested async thunks on additional compute or communication streams.
- Additional generic call sites include sub-stream creation in `~/github/openxla/xla/xla/stream_executor/stream_common.cc:100-105`, address allocator streams in `~/github/openxla/xla/xla/stream_executor/stream_executor_address_allocator.cc:106-114`, command-buffer tracing in `~/github/openxla/xla/xla/stream_executor/trace_command_buffer_factory.cc:31-39`, and GPU infeed in `~/github/openxla/xla/xla/service/gpu/infeed_manager.cc:43-46`.

### XLA stream-count knobs

- `~/github/openxla/xla/xla/debug_options_flags.cc:453-454` sets `xla_gpu_executable_num_compute_streams` and `xla_gpu_executable_num_communication_streams` defaults to zero.
- `~/github/openxla/xla/xla/debug_options_flags.cc:2758-2770` describes those flags as the number of additional compute or communication streams to allocate for a GPU executable.
- Compile-time execution stream assignment uses four default compute stream IDs and one or two communication stream IDs: `~/github/openxla/xla/xla/service/gpu/compile_module_to_llvm_ir.cc:95-106`, `~/github/openxla/xla/xla/service/gpu/gpu_compiler.cc:3353-3362`, and `~/github/openxla/xla/xla/backends/gpu/runtime/execution_stream_id.h:135-138`.
- Runtime stream allocation still clamps communication stream count to at least four because `kAsyncStreamTotal` is computed from `ASYNC_STREAM_KIND_MEMCPYP2P + 1`: `~/github/openxla/xla/xla/service/gpu/gpu_executable.cc:246-258`.

### SYCL 2020 references

- `~/sycl/sycl-2020-map.md:109-115` maps the queue class, constructors, member functions, properties, and error-handling sections.
- `~/sycl/sycl-2020-map.md:573-606` maps queue constructors, `queue::is_in_order`, waits, shortcuts such as `memcpy`, and queue properties including `enable_profiling` and `in_order`.
- `~/sycl/sycl-2020.html:2834-2910` describes default out-of-order execution based on dependencies and says that work submitted to different queues or from multiple threads is ordered by the runtime.
- `~/sycl/sycl-2020.html:10303-10330` defines `queue` as scheduling kernels on a device and says `queue::submit` schedules command groups asynchronously.
- `~/sycl/sycl-2020.html:11435-11457` describes `property::queue::enable_profiling`: the implementation captures profiling information for command groups submitted to that queue, and queue construction can fail if the device lacks `aspect::queue_profiling`.
- `~/sycl/sycl-2020.html:11462-11480` describes `property::queue::in_order`: commands submitted to that queue execute in submission order as if dependent on the previous command, but this property gives no ordering guarantee relative to other queues.
- `~/sycl/sycl-2020.html:11743-11774` says `event::get_profiling_info` can block until profiling information is available and throws if the submitting queue was not constructed with `enable_profiling`.

### oneAPI references

The oneAPI route-map files needed for this section are present locally. No absence gap was found for the immediate command-list or in-order queue events documents.

- `~/sycl/oneapi.md:32` routes Level Zero documentation to `oneapi/MultiTileCardWithLevelZero.md` and `oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md`.
- `~/sycl/oneapi/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc:44-56` says the immediate command-list extension is supported by DPC++, but specifying immediate command lists is not recommended for non-Data-Center-Max Intel GPUs.
- `~/sycl/oneapi/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc:60-76` describes Level Zero queue submission as either immediate command lists or standard command queues, with default behavior chosen by the implementation for most workloads.
- `~/sycl/oneapi/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc:119-128` defines `immediate_command_list` as immediate submission and `no_immediate_command_list` as standard queue submission where commands may be batched.
- `~/sycl/oneapi/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc:153-158` says queue properties take precedence over the immediate-command-list environment variable.
- `~/sycl/oneapi/EnvironmentVariables.md:160-165` documents `SYCL_UR_USE_LEVEL_ZERO_V2`, which is aimed at queue modes including immediate/batched and in-order/out-of-order and currently supports immediate command lists.
- `~/sycl/oneapi/EnvironmentVariables.md:275-294` documents copy offload, V2 batched forcing, legacy batch sizes, copy/compute engine controls, immediate command-list mode, event reuse, command-list cleanup, and immediate-command-list event cleanup thresholds.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_in_order_queue_events.asciidoc:56-64` says in-order queue events provide APIs to get the last event and set an external event dependency, but the extension solves a specific problem and is not recommended for general usage.
- `~/sycl/oneapi/extensions/experimental/sycl_ext_oneapi_in_order_queue_events.asciidoc:116-154` describes `ext_oneapi_get_last_event()` and `ext_oneapi_set_external_event()`, both restricted to queues with `property::queue::in_order`.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc:76-98` documents non-blocking enqueued barriers and returned events. XLA currently uses `queue::ext_oneapi_submit_barrier()` for marker/wait behavior.

## Findings

1. The current production SYCL stream model is independent in-order queues, not one shared queue. The public override in `SyclExecutor` always enables multiple streams.

2. This model matches XLA stream semantics better than one in-order queue. In-order preserves per-stream FIFO; separate queues preserve possible overlap; `WaitFor` and `RecordEvent` add explicit barrier dependencies when XLA needs ordering.

3. A single in-order queue would be simpler but too conservative. It would serialize host-to-device, device-to-host, device-to-device, async compute, async collective, command-buffer trace, sub-stream, callback, and pooled usage streams even when XLA intentionally separates them.

4. The current cap of 4096 streams per device is a hard safety guard, not an evidence-based target. Fixed PJRT setup creates far fewer queues by default: compute, host-to-device, four D2H, four D2D, four fixed-size-pool, four external-ready-event streams, plus lazy callback and usage streams. GPU executable runtime can add command-buffer trace, communication, and compute streams depending on async thunks and debug options.

5. All current SYCL queues are profiling-enabled. That is robust for timers and event profiling, but it may add queue creation, event, or submission overhead. No local measurement is available.

6. XLA does not currently set oneAPI immediate or no-immediate queue properties. DPC++/Level Zero default or environment variables decide immediate versus batched command-list behavior.

7. The immediate command-list extension is not a universal performance recommendation. The local oneAPI doc says to rely on implementation defaults in most cases and warns against explicitly requesting immediate command lists outside the tested Data Center Max GPU scope.

8. Existing tests cover creation, max-stream exhaustion, basic multi-stream independence, event recording, and copies. They do not measure overlap, queue scheduler pressure, profiling overhead, immediate versus batched modes, or high stream-count behavior in realistic XLA workloads.

## Proposed patch plan

1. Keep semantics unchanged first. Preserve `SyclExecutor::CreateStream(priority) -> CreateStream(true, priority)` and the use of in-order queues.

2. Add a small queue-options helper in `sycl_gpu_runtime.cc`, for example `BuildQueueProperties(const SyclQueueOptions&)`, initially returning the existing `{enable_profiling, in_order}` property list. Route both `InitStreamPool` and `GetOrCreateStream` through it.

3. Add instrumentation in `SyclStreamPool`: live queue count, high-water mark, queue creation duration, queue destruction count, and cap-hit count per device ordinal. Log at VLOG and expose enough state for tests.

4. Make the stream cap configurable, with a hard upper bound. Keep 4096 as a temporary compatibility default if necessary, but add a lower warning threshold such as 256. After benchmark coverage lands, lower the default to 256 or 512 only if large workloads and stress tests pass.

5. Clarify the default-queue path. Either restrict `enable_multiple_streams=false` to raw non-owning default queue access or document/test that multiple `SyclStream` wrappers cannot concurrently own the same default queue.

6. Add a profiling property experiment behind a debug option. The safe initial mode is current behavior: all queues profiling-enabled. The experiment should allow non-profiling queues for non-timer streams and force profiling-capable queues when `CreateEventBasedTimer` can be called.

7. Add optional immediate/batched queue properties only behind an off-by-default debug option. Prefer existing DPC++ environment variables for first measurements.

8. Update tests:
   - Cap configuration and cap-hit error.
   - Independent queue handles for two production `CreateStream()` calls.
   - Cross-stream wait correctness with `WaitFor(Stream*)` and `WaitFor(Event*)`.
   - Default-queue path ownership behavior if retained.
   - Queue property selection helper.

## Test/benchmark coverage

No runtime benchmark was run in this worker environment because `sycl-ls` and `icpx` were not found. The following benchmark should be added and run on the target Intel GPU stack.

### Queue model benchmark

Variants:

- One in-order queue for all work.
- Independent in-order queues, one queue per XLA stream.
- Independent in-order queues with explicit cross-queue waits at XLA dependency points.

Workloads:

- Long compute kernel plus H2D copy on another stream.
- Long compute kernel plus D2H copy on another stream.
- Long compute kernel plus D2D copy on another stream.
- Async compute and communication-like kernels with explicit waits.
- Many tiny kernels across 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512 queues.

Metrics:

- Wall time versus serialized time.
- Host enqueue time per operation.
- Device execution overlap, using event or Level Zero timestamps.
- Queue creation/destruction latency.
- CPU utilization and queue-thread activity.
- Errors and async exception behavior.

Acceptance threshold:

- Independent queues should show meaningful overlap: for long compute plus copy, wall time should be no more than `max(compute_time, copy_time) + 15%` when hardware copy/compute overlap is available.
- One in-order queue is expected to approach serialized time and should not be selected as the production model unless independent queues are broken on a target runtime.
- Increasing queue count beyond the chosen cap must not improve representative workload p95 latency by more than 1% or throughput by more than 1%; otherwise the cap is too low.

### Profiling property benchmark

Variants:

- All queues constructed with `enable_profiling` plus `in_order`.
- Only timer/profiler-capable queues constructed with `enable_profiling`.
- No profiling property on ordinary queues, with a fallback or error for `CreateEventBasedTimer`.

Workloads:

- Queue creation stress.
- Empty barrier markers via `ext_oneapi_submit_barrier`.
- Tiny kernel submit loop.
- H2D/D2H/D2D copy loop.
- XLA autotuning or BLASLt timer path using `SyclTimer`.

Acceptance threshold:

- Keep all queues profiling-enabled if profiling-off improves host submit time by less than 3% and memory/queue overhead by less than 5%.
- Make profiling selective only if non-profiling queues improve host submit time by at least 3% or queue memory by at least 5%, and timer/autotuning correctness remains unchanged.

### Immediate versus batched command-list benchmark

Run the queue model benchmark under:

- Runtime default.
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`.
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`.
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2`.
- `SYCL_UR_USE_LEVEL_ZERO_V2=0/1` where supported.
- `UR_L0_V2_FORCE_BATCHED=1` where supported.
- Optional queue properties `immediate_command_list` and `no_immediate_command_list` only in an experimental XLA build.

Acceptance threshold:

- Keep relying on runtime defaults unless one mode improves representative XLA throughput by at least 5% or p95 latency by at least 3% with no more than 1% regression on copy/compute overlap and tiny-kernel workloads.
- Do not make immediate command lists the XLA default for non-PVC or unknown Intel GPU targets without target-specific evidence.

### Stream cap benchmark

Sweep cap candidates 64, 128, 256, 512, 1024, and 4096.

Acceptance threshold:

- A lower cap is acceptable only if default PJRT initialization, callback streams, pooled usage streams, command-buffer tracing, async compute/communication, and sub-stream workloads never hit `ResourceExhausted`.
- Prefer 256 if it passes large XLA workloads and queue pressure degrades beyond that count.
- Prefer 512 if 256 is too tight for real models but 512 shows no scheduler pressure.
- Keep 4096 only as an override or compatibility ceiling, not as the evidence-based default.

## Rollout risk

- Queue model risk: collapsing to one queue is high risk because it removes intended overlap. Keep independent queues.
- Cap risk: lowering the cap can fail workloads with many clients, sub-streams, callback streams, or async thunks. Roll out with telemetry and an override.
- Profiling risk: removing `enable_profiling` can break `event::get_profiling_info` users and may affect `SyclTimer` or autotuning. Keep profiling default-on until timer paths are validated.
- Immediate/batched risk: oneAPI documents platform-specific caveats. Hard-coding immediate command lists can improve tiny submissions on one target and regress batching or copy behavior on another.
- Test risk: current unit tests do not prove overlap or scheduler behavior. Performance changes require target hardware benchmarks.
- Multi-worker risk: this report intentionally does not edit XLA source. Patch ownership should be coordinated by the parent agent.

## Evidence gaps

- No local runtime measurement: `sycl-ls` and `icpx` were not found in this environment.
- No target hardware evidence for copy/compute overlap, queue creation cost, queue memory footprint, or queue-thread pressure.
- No measurement of `enable_profiling` overhead on XLA's actual SYCL queues.
- No measurement of Level Zero immediate versus batched command-list behavior for XLA workloads.
- No proof that all queues require profiling. Current XLA code makes that conservative choice, but the cost/benefit is unknown.
- No proof that 4096 is the right stream cap. It is an upper bound with headroom, not a measured default.
- The requested oneAPI immediate-command-list and in-order queue event documents are present locally, so there is no document-absence gap for those references. The remaining gap is that extension documentation does not establish XLA workload performance.
