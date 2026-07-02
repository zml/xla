# 9. Stream priority

## Executive recommendation

Classify SYCL stream priority handling as a **Must fix for consistency and capability reporting**, with a **Should fix to map oneAPI queue-priority hints after the stream pool is priority-aware**.

XLA callers do request non-default priority in production, mainly for communication and device-to-device streams, but the current SYCL paths partially work around this with platform checks. The unsafe part is not that SYCL lacks true priority today; it is that XLA can still carry priority intent into SYCL command buffers and raw integer stream creation, where the backend either rejects, ignores, or records metadata without applying it.

Recommended policy:

- **Must:** SYCL must reject every unsupported non-default priority consistently, including raw nonzero `int` priorities, `StreamPriority::Highest`, `StreamPriority::Lowest`, and command-buffer per-node priority parameters. `0`, `std::nullopt`, and `StreamPriority::Default` remain accepted.
- **Must:** Add explicit capability reporting so XLA callers degrade intentionally instead of hard-coding `kSyclPlatformId` checks or relying on backend-specific failure.
- **Should:** Map `StreamPriority::{Highest,Lowest,Default}` to `sycl_ext_oneapi_queue_priority` high/low/normal hints when built against an implementation that exposes the supported extension, but only after `SyclStreamPool` can create/cache queues by normalized priority.
- **Could:** Support raw numeric priorities only if the numeric oneAPI priority property and device `priority_range` descriptor become supported in the local dependency. In the local docs, numeric priority is in the proposed extension, not the supported one.

## Must/Should/Could classification

### Must

- Reject raw nonzero integer stream priority in `SyclStream::Create()`. Today `std::variant<StreamPriority, int>` accepts `int`, stores it in `StreamCommon`, and never applies it to queue creation.
- Reject non-default command-buffer priority at the SYCL backend boundary until implemented. Callers may downgrade before reaching SYCL, but only through an explicit capability decision. `SyclCommandBuffer::CreateEmptyCmd()` and `CreateLaunch()` currently accept a `StreamPriority priority` parameter and ignore it, while `SetPriority(non-default)` returns `Unimplemented`.
- Add a stream-priority support/capability query in stream executor or GPU runtime code, and use it at the existing high-priority call sites.
- Replace SYCL-specific caller workarounds with the capability query, especially in GPU executable communication streams and PJRT device-to-device streams.

### Should

- Implement best-effort oneAPI priority hints for `StreamPriority::Highest`, `Default`, and `Lowest` by appending `sycl::ext::oneapi::property::queue::{priority_high,priority_normal,priority_low}` to the queue property list when available.
- Refactor `SyclStreamPool` so queues are keyed by priority as well as device ordinal and stream multiplicity. A high-priority request must not reuse an already-created normal-priority queue.
- Preserve an explicit "hint may be ignored by runtime/backend" status in capability reporting, because the supported oneAPI extension describes queue priority as a hint.

### Could

- Add raw numeric priority mapping later, using `sycl::ext::oneapi::property::queue::priority(int)` and `sycl::ext::oneapi::info::device::priority_range`, if/when the numeric revision is supported by the local SYCL implementation. The local file with numeric priority is under `oneapi/extensions/proposed/`, not `supported/`.
- Add performance benchmarks that compare default/high/low oneAPI queues under copy/compute contention on Level Zero.
- Document unsupported priority behavior for downstream users after the Must fix, but documentation alone is not enough because current code can silently ignore priority.

## XLA change candidates with concrete files/functions

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - `SyclStream::Create()`: normalize priority in one helper. Accept only `std::nullopt`, `StreamPriority::Default`, and raw `int{0}` until oneAPI mapping is implemented. Return `Unimplemented` for raw nonzero integers and typed non-default priorities.
  - When oneAPI mapping is implemented, pass the normalized priority to `SyclStreamPool::GetOrCreateStream()` instead of computing an unused local integer.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.{h,cc}`
  - `SyclStreamPool::GetOrCreateStream()` and `InitStreamPool()`: add a normalized priority argument and make the internal pool/cache key include priority. Build `sycl::property_list` with `enable_profiling`, `in_order`, and, conditionally, oneAPI priority hints.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_command_buffer.cc`
  - `SyclCommandBuffer::CreateEmptyCmd()` and `CreateLaunch()`: return `Unimplemented` when `priority != StreamPriority::Default` until SYCL graph/command-buffer priority semantics are actually implemented.
  - Keep `SetPriority(non-default)` returning `Unimplemented`, but align the error text with stream priority errors and mention unsupported command-buffer priority.

- `~/github/openxla/xla/xla/stream_executor/stream_executor.h`
  - Add a capability API, for example `StreamPrioritySupport GetStreamPrioritySupport()` or `bool SupportsStreamPriority(StreamPriority, PrioritySemantics*)`, rather than overloading `GetGpuStreamPriority()`. The current default `GetGpuStreamPriority()` returns `0`, which is not enough to distinguish unsupported from default numeric priority.

- `~/github/openxla/xla/xla/service/gpu/gpu_executable.cc`
  - Replace the `kSyclPlatformId` special case around `xla_gpu_enable_highest_priority_async_stream` with the capability query.

- `~/github/openxla/xla/xla/pjrt/local_device_state.cc`
  - Replace the `kSyclPlatformId` special case for device-to-device streams with the capability query.
  - Validate `StreamOptions::priority` against backend support before passing the raw integer into `CreateStream()`.

- `~/github/openxla/xla/xla/backends/gpu/runtime/traced_command_buffer.cc`
  - Apply priority consistently when replacing an evicted traced command-buffer cache entry. The empty-slot path calls `SetPriority(priority)`; the eviction path currently does not.

## Evidence with code references and spec/oneAPI references where available

### SYCL stream creation

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:313-343`: `SyclStream::Create()` accepts `std::optional<std::variant<StreamPriority, int>>`. It copies raw `int` into `stream_priority`, maps `StreamPriority::Default` to `0`, rejects typed non-default priority, then calls `SyclStreamPool::GetOrCreateStream()` without passing the normalized priority.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:844-852`: `SyclExecutor::CreateStream()` funnels all SYCL stream creation through `SyclStream::Create()`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:240-258` and `:309-317`: SYCL queues are constructed with `property::queue::enable_profiling` and `property::queue::in_order`; no priority property is used.
- `~/github/openxla/xla/xla/stream_executor/stream_common.cc:48-55` and `stream_common.h:75-76`: `StreamCommon` records the requested priority variant, so a raw integer can be reported as stream metadata even when SYCL did not apply it.

### XLA stream-priority callers

- `~/github/openxla/xla/xla/debug_options_flags.cc:331` and `:2203-2207`; `~/github/openxla/xla/xla/xla.proto:699`: `xla_gpu_enable_highest_priority_async_stream` defaults true and is described as enabling highest-priority async streams.
- `~/github/openxla/xla/xla/service/gpu/gpu_executable.cc:572-588`: GPU executable chooses `StreamPriority::Highest` for communication streams when the debug option is enabled, but forcibly disables that path for `kSyclPlatformId`.
- `~/github/openxla/xla/xla/service/gpu/gpu_executable.cc:682-685`: communication streams are borrowed with `communication_stream_priority`.
- `~/github/openxla/xla/xla/service/backend.cc:209-237`: `Backend::BorrowStream(s)` accepts a typed `StreamPriority`.
- `~/github/openxla/xla/xla/service/stream_pool.cc:30-64`: `StreamPool::BorrowStream()` pools streams by typed `StreamPriority` and creates streams through `executor_->CreateStream(priority)`.
- `~/github/openxla/xla/xla/pjrt/local_device_state.cc:100-110`: PJRT local device state passes `stream_options->priority` as a raw integer into `CreateStream()`.
- `~/github/openxla/xla/xla/pjrt/local_device_state.h:102-107`: `StreamOptions::priority` is an `int` defaulting to `0`.
- `~/github/openxla/xla/xla/pjrt/local_device_state.cc:130-138`: device-to-device streams use `StreamPriority::Highest` except on SYCL, where the code forces default priority.

### Search results for requested priority forms

- `StreamPriority::Highest` production hits are communication streams, PJRT device-to-device streams, CUDA/ROCm backend mapping, and GPU command-buffer collective paths. Relevant production references: `gpu_executable.cc:587`, `local_device_state.cc:135`, `collective_thunk.cc:427`.
- `StreamPriority::Lowest` has no production call-site found in XLA beyond enum/string handling (`platform.cc:26-27`).
- Raw integer stream priority production surface is `LocalDeviceState::StreamOptions::priority` flowing into `executor->CreateStream(stream_options->priority)` (`local_device_state.h:104`, `local_device_state.cc:109`). The generic stream test also exercises `CreateStream(1)` at `~/github/openxla/xla/xla/stream_executor/stream_test.cc:56-59`.
- SYCL has a specific test expecting typed non-default priority to fail: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc:72-79`.

### Command-buffer priority

- `~/github/openxla/xla/xla/backends/gpu/runtime/command.h:74-92` and `:183-219`: runtime commands store a `se::StreamPriority`, with default/lowest/highest as the intended levels.
- `~/github/openxla/xla/xla/backends/gpu/runtime/kernel_thunk.cc:277-282`: `KernelThunk::Record()` passes `priority()` to `command_buffer->CreateLaunch()`.
- `~/github/openxla/xla/xla/backends/gpu/runtime/custom_kernel_thunk.cc:174-178`: custom kernel command recording also passes `priority()` to `CreateLaunch()`.
- `~/github/openxla/xla/xla/backends/gpu/runtime/async_thunk.cc:145-151`: async-done recording passes `priority()` to `CreateEmptyCmd()`.
- `~/github/openxla/xla/xla/backends/gpu/runtime/traced_command_buffer.h:53-57` and `.cc:60-106`: traced command-buffer creation accepts priority and calls `SetPriority()` for a non-default priority when filling an empty cache entry.
- `~/github/openxla/xla/xla/backends/gpu/runtime/traced_command_buffer.cc:120-128`: the eviction/replacement path recreates a traced command buffer but does not reapply non-default priority.
- `~/github/openxla/xla/xla/backends/gpu/runtime/traced_command.cc:74-78`: traced commands pass their priority into `GetOrTraceCommandBuffer()`.
- `~/github/openxla/xla/xla/backends/gpu/runtime/collective_thunk.cc:415-428`: collective command-buffer recording creates a nested command buffer and unconditionally sets `StreamPriority::Highest`.
- `~/github/openxla/xla/xla/backends/gpu/runtime/ragged_all_to_all_thunk.cc:666-668`: ragged all-to-all propagates non-default priority to nested command buffers.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_command_buffer.cc:717-733`: SYCL command buffer `CreateEmptyCmd()` and `CreateLaunch()` accept priority parameters but ignore them.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_command_buffer.cc:987-992`: SYCL command buffer `SetPriority(non-default)` returns `Unimplemented`.
- Cross-backend comparison: CUDA maps graph kernel priority via `CU_LAUNCH_ATTRIBUTE_PRIORITY` in `~/github/openxla/xla/xla/stream_executor/cuda/cuda_command_buffer.cc:592-600` and applies it to all kernel nodes in `SetPriority()` at `:741-765`; ROCm documents graph priority as silently ignored in `~/github/openxla/xla/xla/stream_executor/rocm/rocm_command_buffer.h:154-160`.

### SYCL and oneAPI references

- `sycl-2020-map.md` points queue constructors to `S|5|10541-10762` and queue properties to `S|5|11427-11485`.
- `~/sycl/sycl-2020.html:10541-10548`: SYCL queue constructors accept a `property_list` and refer to core queue properties.
- `~/sycl/sycl-2020.html:11427-11485`: core queue properties listed locally are `enable_profiling` and `in_order`; no core SYCL queue priority property appears in that section.
- `~/sycl/oneapi.md:41`: the route map lists `oneapi/extensions/supported/sycl_ext_oneapi_queue_priority.asciidoc`.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_queue_priority.asciidoc:42-56`: the local supported oneAPI extension is implemented and fully supported by DPC++, used on Level Zero/CUDA/HIP, and says priority is a hint that may be ignored by unsupported underlying backends.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_queue_priority.asciidoc:81-108`: the supported extension defines `priority_normal`, `priority_low`, and `priority_high` queue properties, and multiple different priority hints on one queue are invalid.
- `~/sycl/oneapi/extensions/proposed/sycl_ext_oneapi_queue_priority.asciidoc:84-87` and `:90-119`: the proposed revision adds numeric `property::queue::priority(int)` and `info::device::priority_range`, with clamping and lower numerical values implying higher urgency.
- `~/sycl/oneapi/extensions/proposed/sycl_ext_oneapi_queue_priority.asciidoc:153-160` and `:181-186`: the proposed revision maps normal/low/high to numeric values and defines the priority range descriptor. Because this is under `proposed/`, do not treat raw integer priority as currently supported evidence.

## Findings

1. **SYCL stream priority behavior is internally inconsistent.** Typed non-default `StreamPriority` is rejected, but raw nonzero integer priority is accepted, stored, and ignored.

2. **Current production stream-priority callers mostly avoid SYCL by platform check.** GPU executable communication streams and PJRT D2D streams request highest priority on other GPU backends but explicitly force default priority on SYCL.

3. **Command-buffer priority still reaches SYCL and has mixed behavior.** Per-node priority arguments to `CreateEmptyCmd()` and `CreateLaunch()` are ignored, while whole-command-buffer `SetPriority(non-default)` fails.

4. **XLA callers use priority as a performance/latency hint, not as a correctness contract.** That matches oneAPI's supported extension semantics, but XLA still needs an explicit unsupported/best-effort signal to avoid silently losing user or runtime intent.

5. **oneAPI gives a plausible Should-level implementation route, not a Must-level correctness dependency.** The supported extension exposes high/normal/low queue-priority hints. Numeric raw integer priority is only documented in the local proposed extension.

6. **The current SYCL queue pool is not priority-aware.** Even if priority properties are added to queue construction, `SyclStreamPool` must avoid returning a preexisting default-priority queue for a high- or low-priority request.

7. **There is a generic traced-command-buffer priority propagation bug.** The empty-cache path applies non-default priority, but the cache eviction path recreates a command buffer without calling `SetPriority()`.

## Call-site impact

- `GpuExecutable` communication streams: With capability reporting, SYCL should deliberately choose default priority until oneAPI mapping is enabled. CUDA remains highest priority when supported; ROCm behavior remains backend-defined.
- `LocalDeviceState` D2D streams: SYCL should continue defaulting intentionally, but without a hard-coded platform check. Raw nonzero `StreamOptions::priority` should fail early on SYCL until numeric priority support exists.
- `Backend::BorrowStream(s)` and `StreamPool`: No semantic change for default-priority callers. If SYCL oneAPI hints are implemented, stream pooling must keep separate queues per priority.
- Runtime command buffers: SYCL command-buffer recording with non-default priority should fail predictably or downgrade through an explicit capability decision. It should not silently drop per-node priority while rejecting whole-buffer priority.
- Tests that assume `CreateStream(1)` always succeeds may need to become backend-specific. Generic interfaces allow raw integer priorities, but SYCL should not accept unsupported raw nonzero integers.

## Proposed patch plan

1. Add a SYCL-local priority normalization helper:
   - input: `std::optional<std::variant<StreamPriority, int>>`;
   - output: normalized `StreamPriority::Default` for `std::nullopt`, `StreamPriority::Default`, and raw `0`;
   - error: `Unimplemented` for typed highest/lowest and raw nonzero integers.

2. Use that helper in `SyclStream::Create()` and delete the unused `stream_priority` local.

3. Add SYCL tests:
   - `Create(std::nullopt)` succeeds;
   - `Create(StreamPriority::Default)` succeeds;
   - `Create(0)` succeeds;
   - `Create(1)` and `Create(-1)` return `Unimplemented`;
   - `Create(StreamPriority::Highest)` and `Create(StreamPriority::Lowest)` return `Unimplemented`.

4. Update `SyclCommandBuffer::{CreateEmptyCmd,CreateLaunch}` to reject non-default priority with `Unimplemented` until implemented.

5. Add command-buffer tests that exercise non-default priority through `CreateLaunch()`, `CreateEmptyCmd()`, and `SetPriority()`.

6. Add a stream-priority capability API to `StreamExecutor` and wire it through CUDA, ROCm, SYCL, host, and TPU conservatively:
   - CUDA: supported with concrete priority mapping;
   - ROCm: streams supported, graph node priority best-effort/ignored where documented;
   - SYCL: unsupported initially, then best-effort hints when oneAPI mapping lands;
   - host/TPU/interpreter: unsupported or default-only.

7. Replace platform checks in `GpuExecutable` and `LocalDeviceState` with capability checks. On unsupported platforms, use default priority and optionally VLOG once per executor when a high-priority optimization is downgraded.

8. Fix `TracedCommandBuffer::GetOrTraceCommandBuffer()` to apply non-default priority after cache eviction/replacement as well as after empty-slot creation.

9. Should-level follow-up: implement oneAPI priority hints behind `#ifdef SYCL_EXT_ONEAPI_QUEUE_PRIORITY` and update `SyclStreamPool` to key queues by normalized priority. Treat the resulting support as best-effort, not guaranteed scheduling priority.

## Test/benchmark coverage

- Unit tests:
  - Extend `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc` for raw integer priority rejection/acceptance.
  - Add SYCL command-buffer tests for non-default priority rejection in `CreateLaunch()`, `CreateEmptyCmd()`, and `SetPriority()`.
  - Add a generic traced-command-buffer test that fills the cache, triggers eviction, and verifies `SetPriority()` is applied on the replacement path.
  - Add capability-query tests for CUDA/ROCm/SYCL/host defaults.

- Integration tests:
  - Run a SYCL GPU executable with `xla_gpu_enable_highest_priority_async_stream=true` and verify it uses default priority through capability downgrade instead of backend-specific platform checks.
  - Run a PJRT local-device-state construction path with raw nonzero `StreamOptions::priority` on SYCL and verify it fails clearly.

- Benchmarks:
  - If oneAPI mapping lands, benchmark default/high/low queue hints on Level Zero with concurrent compute, D2D copy, H2D/D2H copy, and collective-like traffic.
  - Include a fallback benchmark where priority hints are ignored to make sure the capability/reporting path labels support as best-effort.

## Rollout risk

- Rejecting raw nonzero integers on SYCL is a behavior change for any downstream user passing `LocalDeviceState::StreamOptions::priority != 0`, but the current behavior is misleading because the priority is not applied.
- Capability API changes touch common stream-executor interfaces and need conservative defaults to avoid affecting host/TPU/interpreter backends.
- oneAPI hint mapping may create more queues or separate pools per priority, increasing queue count and resource usage.
- oneAPI priority hints are not guaranteed scheduling priority. Enabling them without clear best-effort reporting could create false performance expectations.
- Command-buffer priority rejection may expose existing high-priority command-buffer paths on SYCL. This is preferable to silently recording a different priority policy than the command sequence requested.

## Evidence gaps

- The oneAPI queue-priority extension file exists locally under `oneapi/extensions/supported/`; there is no evidence gap for the high/normal/low hint API. The route-map warning about possibly absent docs does not apply to this extension in the current workspace.
- The numeric raw integer priority API exists only in the local proposed oneAPI extension file. I did not find local evidence that the numeric property and `priority_range` descriptor are supported by the current dependency.
- I did not run compile-time checks against installed DPC++ headers for `SYCL_EXT_ONEAPI_QUEUE_PRIORITY`; the recommendation relies on local docs and should be verified in the build environment before implementation.
- I did not run hardware measurements showing that Level Zero honors high/low queue hints or quantifying their scheduling effect.
- I did not inspect native Level Zero command queue priority APIs in local oneAPI docs beyond the SYCL extension route; implementation should confirm the exact backend behavior before claiming more than best-effort support.
- I did not run XLA tests; this note is a source-evidence research output only.
