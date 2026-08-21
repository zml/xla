# ROCm 7.14 Performance Plan

## Objective

Make the ROCm StreamExecutor and XLA GPU runtime as fast as the ROCm 7.14
runtime and AMD hardware allow, with particular emphasis on:

1. ordinary kernel enqueue latency;
2. command-buffer construction, update, and replay latency;
3. BLAS, BLASLt, DNN, and FFT call overhead;
4. allocation, initialization, autotuning, and teardown overhead; and
5. end-to-end decode throughput and time to first token in ZML.

This plan intentionally targets **ROCm 7.14.0 / HIP 7.14.60850 only**.
Compatibility code for older ROCm releases is out of scope. Each action item
must still be feature-gated until its correctness and performance gates pass,
because several HIP 7.14 APIs used below remain documented as beta.

The plan is based on XLA commit
`50572259f75b54b78370cee79cc9c67e8dca1bf4` on branch `rocm`.

### ROCm 7.14 API references

Use the installed 7.14 headers as the build contract and the matching official
AMD documentation when validating semantics:

- [HIP 7.14 event management](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/reference/hip_runtime_api/modules/event_management.html)
- [HIP 7.14 graph management](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/reference/hip_runtime_api/modules/graph_management.html)
- [HIP 7.14 stream memory operations](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/reference/hip_runtime_api/modules/stream_memory_operations.html)
- [HIP 7.14 execution control](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/reference/hip_runtime_api/modules/execution_control.html)
- [HIP 7.14 virtual-memory management](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/reference/hip_runtime_api/modules/memory_management/virtual_memory_reference.html)
- [HIP 7.14 stream-ordered allocator](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/doxygen/html/group___stream_o.html)
- [HIP 7.14 porting guide, including module launches](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/how-to/hip_porting_guide.html)
- [HIP 7.14 performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/how-to/performance_guidelines.html)

Header presence is not proof of implementation. In particular, AMD documents
some graph flags as ignored and several graph/stream-memory APIs as beta. Every
such feature below has a target-GPU behavior test before integration.

## Non-negotiable development rules

1. Implement one independently measurable optimization per commit. Do not mix
   refactors, formatting changes, or unrelated fixes into a performance commit.
2. Before every patch, record the current commit, ROCm/HIP versions, GPU model,
   clock/power state, build command, run command, command-buffer mode, and all
   `XLA_FLAGS`/environment variables.
3. Build the plugin after every patch:

   ```bash
   cd /home/steeve/xla
   bazel build --spawn_strategy=local \
     --config=rocm_ci_hermetic \
     --config=baseline_x86_64 \
     //xla/pjrt/c:pjrt_c_api_gpu_plugin
   ```

4. Run the large-model correctness smoke test after every patch that touches
   launches, graphs, streams, events, libraries, or allocation:

   ```bash
   cd /home/steeve/zml
   bazel run //examples/llm \
     --@zml//platforms:rocm=true \
     --run_under="CUDA_VISIBLE_DEVICES=7 " -- \
     --model=/var/models/Qwen/Qwen3.8-27B/ \
     --prompt="tell me a story"
   ```

5. Run the release throughput benchmark after every performance patch:

   ```bash
   cd /home/steeve/zml
   bazel run //examples/llm --config=release \
     --@zml//platforms:rocm=true \
     --run_under="CUDA_VISIBLE_DEVICES=7 " -- \
     --model=/var/models/meta-llama/Llama-3.1-8B-Instruct/ \
     --prompt="tell me a story"
   ```

6. Commit only after the build and required tests pass. Put measured before and
   after results in the commit message. If a patch fails or regresses, commit a
   documented revert separately; do not erase the experiment from history.
7. A throughput change smaller than normal run-to-run variation is not a win.
   Use at least one warm-up plus five measured runs, report median, minimum,
   maximum, and median absolute deviation. Interleave baseline and candidate
   runs when thermals or shared-machine activity could bias the result.
8. Preserve exact numerical output for deterministic tests. For model tests,
   compare token IDs or output hashes through a fixed generation length, not
   merely whether execution completed.
9. Any optimization that weakens a memory fence, changes stream ordering, or
   retains a library stream handle must ship with a targeted lifetime/ordering
   test and a runtime rollback flag for its first landing.

## Measurement foundation

Complete this section before judging any hot-path patch.

### M0. Add ROCm host-overhead microbenchmarks

Create a ROCm benchmark target near the StreamExecutor ROCm tests, using a
preloaded no-op or minimal kernel and a pre-created stream. Measure host time
without synchronizing after each operation; synchronize once after each batch.
At minimum measure:

- `hipModuleLaunchKernel` with 0, 1, 4, 16, and 64 arguments;
- pointer-array arguments versus `HIP_LAUNCH_PARAM_BUFFER_POINTER`;
- default versus `hipStreamNonBlocking` streams;
- dependency-event record/wait with system- versus device-scope release, plus a
  separate timing-event benchmark with `hipEventDisableSystemFence`;
- `rocblas_set_stream` when changing and not changing streams;
- per-node graph executable updates for 1, 8, 32, 128, and 512 nodes;
- graph replay before and after explicit `hipGraphUpload`;
- first replay separately from steady replay;
- `hipGraphExecUpdate` versus individual executable-node updates;
- `hipMalloc`/`hipFree`, BFC reuse, and HIP async-pool allocate/free; and
- one VMM timeline marker per teardown versus one coalesced high-watermark
  marker; test `hipStreamBatchMemOp` separately only if nonredundant adjacent
  operations remain afterward.

Report API-call host latency, batch enqueue rate, final synchronization time,
and total wall time separately. A change that merely moves work from an API
call into the final synchronization is not an enqueue improvement.

### M1. Add lightweight runtime counters

Behind VLOG or a disabled-by-default diagnostic flag, count per executable:

- ordinary kernel launches;
- graph launches, instantiations, uploads, and node updates by node type;
- event records, waits, synchronizations, and event visibility class;
- `rocblas_set_stream`, `miopenSetStream`, and FFT stream-binding calls;
- VMM map/unmap, peer-access queries, and timeline writes; and
- device-count, stream-priority-range, and device-property queries.

The counters determine whether a micro-optimization is exercised by the Llama
decode path. They must not remain enabled in release builds.

### M2. Standard benchmark matrix

For each relevant patch, test the following separately:

| Axis | Required values |
|---|---|
| Execution | eager thunks; FUSION command buffers |
| Update mode | `ALWAYS_UPDATE`; stable-address non-`ALWAYS_UPDATE` modes |
| Model phase | compile/load; prefill; first decode token; steady decode |
| Workload | launch microbench; small GEMM loop; Llama 8B; Qwen 27B smoke |
| Streams | default semantics; nonblocking candidate |
| GPUs | selected GPU 7; at least one two-GPU peer/VMM test |

Store raw measurements under a non-source results directory or external
artifact store; do not commit generated profiles or model output to XLA.

## Phase 1: low-risk cleanup and instrumentation

### P1. Fix unconditional ROCm occupancy logging

**Problem.** `RocmKernel::GetMaxOccupiedBlocksPerCore` logs occupancy at
`VLOG(0)`, unlike CUDA's `VLOG(3)` diagnostic.

**Files.**

- `xla/stream_executor/rocm/rocm_kernel.cc`
- `xla/stream_executor/rocm/rocm_kernel_test.cc` or a new focused test

**Implementation.** Change only the verbosity level. Do not remove occupancy
validation or the HIP occupancy call.

**Tests and gate.** Verify no occupancy line appears at default verbosity and
that it appears at the chosen debug verbosity. Benchmark an autotuning-heavy
compile. This is expected to improve logs/autotuning only, not steady decode.

### P2. Cache visible device count

**Problem.** `ROCmPlatform::VisibleDeviceCount` calls `hipGetDeviceCount` on
every query, while the CUDA platform memoizes the result.

**Files.**

- `xla/stream_executor/rocm/rocm_platform.cc`
- `xla/stream_executor/rocm/rocm_platform_test.cc`

**Implementation.** Cache the successful result with `absl::call_once` or an
equivalent process-lifetime initialization primitive. Match CUDA's handling of
initialization failure: do not accidentally convert a transient initialization
error into a permanent successful zero-device result.

**Tests and gate.** Add concurrent callers and failure-path tests where the
wrapper can be mocked. Confirm one runtime query per process. Startup-only win.

### P3. Cache the HIP stream-priority range per executor

**Problem.** every nondefault-priority stream creation calls
`hipDeviceGetStreamPriorityRange`, including attempts that ultimately reuse a
cached ROCm stream handle.

**Files.**

- `xla/stream_executor/rocm/rocm_executor.h`
- `xla/stream_executor/rocm/rocm_executor.cc`
- ROCm executor/stream tests

**Implementation.** Initialize the least/greatest priority range once after
executor activation. Cache an `absl::StatusOr`-equivalent state so an error is
reported, not silently replaced by default priority. Keep the mapping from XLA
priority enums to HIP integer priorities centralized.

**Tests and gate.** Concurrent high/low-priority stream creation, injected
query failure, and reuse of the stream cache. Startup/stream-creation win only.

### P3A. Cache per-device HIP properties centrally

ROCm repeats `hipGetDeviceProperties` through GPU ISA/GCN-name queries, device
description construction, and rocBLAS initialization. Build one immutable
per-ordinal `hipDeviceProp_t` cache after platform/executor initialization and
derive those values from it. Prefer the executor's completed `DeviceDescription`
inside rocBLAS instead of querying HIP again. Cache only successful results and
retain precise initialization errors. This is initialization-only but also gives
P15, P21, and P26 one authoritative capability source.

### P4. Use the existing peer-access cache in `RocmVmmAllocator`

**Problem.** the VMM allocator and the lower-level memory-reservation access
path resolve a peer executor and call the `StreamExecutor*` peer-access overload
for each peer. That overload repeats `hipDeviceCanAccessPeer` and bypasses
ROCm's existing ordinal cache. The access descriptors are then installed one at
a time even though HIP accepts an array.

**Files.**

- `xla/stream_executor/rocm/rocm_vmm_allocator.cc`
- `xla/stream_executor/rocm/rocm_memory_reservation.cc`
- `xla/stream_executor/rocm/rocm_executor.cc`
- `xla/stream_executor/rocm/rocm_vmm_allocator_test.cc`

**Implementation.** Use the cached ordinal-based lookup in both access paths.
Reuse the cached visible-device count where possible. Build every accessible
`hipMemAccessDesc` first, then make one `hipMemSetAccess` call with the descriptor
array and count. Do not change which accessible peers receive access or their
read/write flags.

**Tests and gate.** On two or more GPUs, verify access descriptors are identical
before/after and count one capability query per device pair. Exercise asymmetric
or denied peer access. Allocation/startup win only.

### P5. Retry HIP graph instantiation after graph-memory trim

**Problem.** CUDA retries resource-exhausted graph instantiation after trimming
the device graph-memory pool; ROCm returns the first OOM.

**Files.**

- `xla/stream_executor/rocm/rocm_command_buffer.cc`
- `xla/stream_executor/rocm/rocm_command_buffer_test.cc`

**Implementation.** On `hipErrorOutOfMemory` only, call
`hipDeviceGraphMemTrim(device_ordinal)` and retry `hipGraphInstantiate` once.
Preserve the original error if trimming fails and include both statuses in the
diagnostic. Never trim or retry other failures.

**Tests and gate.** Wrapper-injected first-OOM/second-success, trim failure,
second OOM, and non-OOM failure. This is graph creation reliability, not steady
performance.

### P6. Correct ROCm timer error propagation

**Problem.** `GetEventElapsedTime` returns `false` into `StatusOr<float>` after
a failed `hipEventSynchronize`, which becomes a successful zero measurement.

**Files.**

- `xla/stream_executor/rocm/rocm_timer.cc`
- `xla/stream_executor/rocm/rocm_timer_test.cc`

**Implementation.** Return the HIP synchronization error as `absl::Status`.
Keep this correctness fix separate from timer-performance patches so benchmark
changes remain attributable.

## Phase 2: steady-state library overhead

### P7. Cache rocBLAS stream state with lifetime tracking

**Priority: highest-confidence steady-state candidate.**

**Problem.** each rocBLAS operation calls `rocblas_set_stream` before the
operation and resets the handle to the null stream afterward. The reset was an
intentional correctness workaround: GemmAlgorithmPicker used a temporary stream,
and rocBLAS retained the destroyed stream. It must not simply be deleted.

**Files.**

- `xla/stream_executor/rocm/rocm_blas.h`
- `xla/stream_executor/rocm/rocm_blas.cc`
- `xla/stream_executor/rocm/rocm_executor.cc`
- `xla/stream_executor/blas.h` if a generic notification hook is required
- ROCm BLAS and executor tests

**Design.**

1. Add `std::optional<hipStream_t> current_stream_` protected by the existing
   rocBLAS mutex.
2. Skip `rocblas_set_stream` when the requested handle already matches.
3. Add `NotifyStreamDestroyed(Stream*)` or a ROCm-private equivalent wired from
   `RocmExecutor::DeallocateStream`, following the established MIOpen pattern.
4. If the destroyed stream matches the cache, invalidate the optional under the
   BLAS mutex. Do not call rocBLAS or choose an arbitrary live stream from
   `DeallocateStream`; that adds activation and lock-order risk during teardown.
   Reset to null only if targeted handle-destruction tests prove it necessary.
5. After invalidation, the next operation must always rebind, including when HIP
   has reused the same raw stream-handle value for a new wrapper.
6. Keep the mutex held across state selection and the rocBLAS operation; a
   shared stateful handle cannot safely serve two streams concurrently.

**Tests.** Same-stream repetition causes one set call; A/B/A stream switching
causes three; destroying the bound wrapper invalidates cached state so the next
operation rebinds even after raw-handle reuse; destroying an unrelated stream
does not; temporary autotune-stream capture no longer reproduces
`Stream Capture Check Failed`; concurrent calls remain serialized correctly.

**Performance gate.** Count and time `rocblas_set_stream`; run small GEMMs and
Llama decode. Land only if call count falls and there is no throughput or graph
capture regression.

### P8. Revisit rocBLAS workspace binding safely

**Problem.** the CUDA backend binds a caller-owned workspace per call. ROCm's
equivalent block is disabled because `rocblas_set_workspace` may free/reallocate
handle-managed memory using capture-unsafe blocking operations.

**Files.**

- `xla/stream_executor/rocm/rocm_blas.cc`
- `xla/stream_executor/rocm/rocm_blas.h`
- rocBLAS tests and graph-capture tests

**Design experiments, in order.**

1. Confirm ROCm 7.14's exact `rocblas_set_workspace` behavior with a standalone
   capture microbenchmark.
2. Test binding one stable workspace once during handle initialization, before
   any capture, rather than resetting it per operation.
3. If one global handle cannot satisfy concurrent streams, evaluate a small
   handle-per-stream pool with the same stream-lifetime notifications as P7.
4. Never call `rocblas_set_workspace` inside stream capture unless the standalone
   test proves it capture-safe.

The stable workspace must be a handle-owned allocation with an explicit budget
and destruction order that dominates the rocBLAS handle. Never retain a pointer
borrowed from a per-call `ScopedWorkspace` or scratch allocator. If that memory
cost is unacceptable, pair the experiment with a bounded per-stream handle pool
rather than weakening ownership.

**Gate.** Must reduce internal allocation/capture overhead without increasing
reserved memory excessively. Otherwise retain the current disabled block and
document the ROCm 7.14 limitation.

### P8A. Cache rocBLAS atomics mode transitions

Deterministic rocBLAS calls currently set `rocblas_atomics_not_allowed` on each
operation. Query and store the initial atomics mode during handle initialization,
derive the desired mode from the call's determinism requirement, and invoke
`rocblas_set_atomics_mode` only when the desired value changes. Keep the cached
mode under the existing rocBLAS mutex and update it only after a successful API
call. Exercise deterministic/non-deterministic/deterministic transitions and
verify deterministic output. This is a small library-call win that should land
separately from P7.

### P9. Cache FFT plan stream state

**Problem.** ROCm calls `hipfftSetStream` for each plan execution. CUDA does the
same, so this is a HIP-side experiment rather than a parity fix.

**Files.**

- `xla/stream_executor/rocm/rocm_fft.cc`
- ROCm FFT plan types and tests

**Implementation.** Store the last bound `hipStream_t` in each plan and skip
the setter if unchanged. Cache the last work-area pointer and size as well, and
skip `hipfftSetWorkArea` only when the newly obtained live per-execution scratch
pointer matches the cached value; the plan must not retain ownership of scratch.
Keep required scratch size immutable in the plan. Stream caching still requires
retirement invalidation or proof that the stream dominates the plan.
Do not remove the current out-of-place input copy: the ROCm 7.14 rocFFT contract
permits input overwrite even for out-of-place execution.

**Gate.** Microbenchmark small repeated FFTs. Do not land if setter cost is
below noise or lifetime plumbing outweighs the benefit.

### P10. Reuse MIOpen algorithm-query descriptors

**Problem.** MIOpen stream state is already cached, but algorithm discovery
still has a TODO to create descriptors once and reuse them between methods.

**Files.**

- `xla/stream_executor/rocm/rocm_dnn.cc`
- ROCm DNN tests

**Implementation.** Start locally rather than with a global cache. The two passes
inside `GetMIOpenConvolveAlgorithms` should acquire the MIOpen handle lock once
and share one RAII descriptor bundle between find-database population and the
immediate-solution query. Reset every shape/stride/padding field before reuse;
never rely on old descriptor contents. Do not share mutable descriptors
concurrently without locking. Only after this low-risk refactor should a global
keyed cache or ROCm 7.14's generic `miopenProblem`/`miopenFindSolutions` APIs be
benchmarked behind a flag.

**Gate.** Measure convolution algorithm discovery/autotuning and model compile
time. This is not expected to change steady decode.

### P11. Expand hipBLASLt autotuning without adding stream state

hipBLASLt receives the stream as an operation argument rather than retaining it,
and XLA already has a persistent executor handle, plan cache, and stable
workspace. There is no missing set-stream cache to add.

ROCm 7.14 exposes StreamK tile scheduling and SM/CU-count target attributes.
Extend the autotune candidate identity to include StreamK OFF/ON (and AUTO only
if it is distinguishable and stable) plus a small set of CU targets intended to
leave room for overlap. Encode every knob in the algorithm/cache key; otherwise
a cached winner can be replayed with the wrong descriptor state. Under the
BLASLt mutex, cache the last descriptor attribute values to avoid redundant
setters. Retain full-CU scheduling as the safe baseline.

Benchmark isolated GEMMs and the whole concurrent model. Drop candidates that
only improve standalone GEMMs while hurting overlap, determinism, memory use, or
end-to-end throughput.

## Phase 3: streams, events, and ordinary launches

### P12. A/B test nonblocking HIP streams

**Problem.** CUDA creates nonblocking streams; ROCm creates `hipStreamDefault`
streams and retains a source TODO about changing this.

**Files.**

- `xla/stream_executor/rocm/rocm_stream.cc`
- `xla/stream_executor/rocm/rocm_stream_test.cc`
- tests for libraries, callbacks, graph capture, and cross-stream events

**Implementation.** Use `hipStreamNonBlocking` for ordinary XLA-created streams.
The existing ROCm stream cache already keys handles by flags; verify this rather
than changing cache identity. Initially place the behavior behind a ROCm debug
option or environment-controlled experiment.

**Correctness matrix.** Null-stream interaction, explicit events, D2D/H2D/D2H
copies, callbacks, graph capture, rocBLAS, hipBLASLt, MIOpen, FFT, collectives,
and stream destruction/reuse. Tests must prove that all ordering required by XLA
is explicit and not inherited from legacy null-stream semantics.

**Gate.** Measure isolated launch latency and full inference. This likely helps
only when null-stream work would otherwise serialize execution; no improvement
for an isolated `hipModuleLaunchKernel` is expected.

### P13. Introduce explicit HIP event visibility classes

**Problem.** every non-timing ROCm event is created with
`hipEventDisableTiming | hipEventReleaseToSystem`. Many XLA events only order
GPU streams and do not require host or peer-device visibility.

**Files.**

- `xla/stream_executor/event.h`
- `xla/stream_executor/stream_executor.h`
- `xla/stream_executor/rocm/rocm_event.h`
- `xla/stream_executor/rocm/rocm_event.cc`
- event-pool and runtime users under `xla/backends/gpu/runtime`

**Design.** Add an explicit creation intent, not a global flag replacement:

- host-visible completion: `hipEventDisableTiming | hipEventReleaseToSystem`;
- same-device stream dependency: `hipEventDisableTiming |
  hipEventReleaseToDevice`; and
- timing-only: handled by P14.

Classify every event-pool caller. Any event that may be polled/synchronized to
make device writes visible to the CPU, or consumed by another GPU, remains
system scope. `RocmStream::WaitFor(Stream*)` currently uses the other stream's
private completion event without proving both streams belong to one device. Keep
that event at system scope until `WaitFor` enforces same-device use or selects a
separate system-scope event for cross-device waits. Only strictly proven
same-executor internal dependencies may start with device scope. Keep public
`CreateEvent`, transfer completion, host callbacks, and cross-device users at
system scope. The type system or distinct factory methods should make accidental
weakening difficult.
Store the chosen visibility in the ROCm wrapper for tests and diagnostics,
because HIP does not expose a general event-get-flags API.

**Tests.** Device-stream handoff, host polling followed by reading pinned memory,
peer-device visibility, collectives, event reuse, and concurrent event pools.
Use memory-content assertions, not only completion status.

**Gate.** Count event types in Llama decode and measure event record/wait host
and device cost. Land only if a meaningful hot fraction can use device scope.

### P14. Disable system fences for timing-only HIP events

**Problem.** ROCm timers use ordinary timing events. AMD documents
`hipEventDisableSystemFence` specifically to avoid cache writeback/invalidation
that can perturb timing and following work.

**Files.**

- `xla/stream_executor/rocm/rocm_event.cc`
- `xla/stream_executor/rocm/rocm_timer.cc`
- ROCm event/timer tests

**Implementation.** Add a timing-only event flag that preserves timing but adds
`hipEventDisableSystemFence`. It is valid only when event synchronization is
used for completion/timestamp availability and not to establish host memory
visibility. Keep system-scope ordinary events unchanged.
This patch can use the existing timing/non-timing creation branch and does not
need to wait for P13's broader public event-intent refactor.

**Gate.** Compare timer accuracy, variance, and autotuner algorithm choices over
many repetitions. Reject if it changes correctness or increases selection
variance even if it is faster.

### P15. Eliminate per-argument allocations and benchmark ABI-packed HIP kernargs

**Priority: best experiment that might affect plain launch latency.**

**Problem.** the current ordinary path constructs packed arguments per launch,
and `KernelArgsPackedArray` owns a separately allocated `PackedArg` for each
argument before ROCm passes a `void** kernelParams` array. HIP 7.14's own typed
GGL path instead builds one contiguous kernarg segment and submits it through
`HIP_LAUNCH_PARAM_BUFFER_POINTER` and `HIP_LAUNCH_PARAM_BUFFER_SIZE`.

**Files.**

- `xla/stream_executor/kernel_args.h`
- `xla/stream_executor/kernel_args_packing_spec.*`
- `xla/service/gpu/stream_executor_util.cc`
- `xla/stream_executor/rocm/rocm_kernel.*`
- `xla/stream_executor/rocm/rocm_stream.cc`
- kernel loading/metadata code in `rocm_executor.cc`
- focused kernel argument and launch tests

**Stage A: standalone proof.** Use known AMDGPU argument layouts to compare
pointer-array and packed-buffer launches for 0/1/4/16/64 arguments. Confirm
whether HIP's host time or deferred synchronization time changes.

**Stage B: remove generic heap churn independently.** Give common simple kernels
one contiguous or inlined owned host representation instead of one allocation
per argument. Benchmark argument construction separately from the HIP call so a
generic XLA allocation win is not confused with a HIP ABI-path win.

**Stage C: represent exact HIP layout.** ROCm 7.14 exposes
`hipKernelGetParamInfo` with per-parameter offsets and sizes, but only for a
`hipKernel_t` obtained through `hipLibraryLoadData`/`hipLibraryGetKernel`; XLA
currently loads `hipModule_t`/`hipFunction_t`. Pilot the HIP library loader for
normal HSACO kernels and cache reflected offsets/sizes in `RocmKernel`. Preserve
the module loader for in-process symbol/custom kernels that cannot use the
library API. Never load one HSACO independently through both APIs: module and
library instances can have distinct globals. A library-loaded HSACO record must
route kernel lookup, global-symbol lookup, and unload through the same library
record; choose module loading only as a per-HSACO fallback. Expose actual host
argument sizes from `KernelArgsPackedArrayBase` and require exact equality with
the reflected size unless a specific packing kind defines a tested conversion.
Do not infer ABI layout from C++ host types.

**Stage D: fast ABI packing.** Reuse inline storage sized for common kernarg
segments, initially around 256 bytes, with an owned fallback for larger ones.
Zero-fill the complete segment, including padding gaps, then copy dynamic scalar
values and device addresses into reflected offsets. Pass
`kernelParams=nullptr` and the `extra` configuration. Only after ordinary
launches pass should graph nodes use their `extra` field; graph storage must own
the byte segment, the `size_t` buffer-size value, and the complete `void* extra[]`
configuration for the executable lifetime. Add source-mutation/lifetime tests
after graph add and executable-node update to learn exactly what HIP deep-copies.

**Correctness.** Cover mixed pointer/scalar sizes, alignment gaps, zero-argument
kernels, dynamic shared memory, TMA/custom packing paths, command-buffer nodes,
and large kernarg segments. Byte-compare the packed segment against compiler ABI
expectations.

**Gate.** Require at least a 5% enqueue improvement or approximately 1 us per
launch without shifting cost to synchronization. Fall back for any mixed
scalar/custom mismatch. If HIP internally converges both inputs to the same path,
retain only the independently useful no-heap generic packing from Stage B.

### P16. Remove heap allocation from nested ROCm activation scopes

`GpuExecutable::ExecuteAsyncOnStreamImpl` already activates the executor once
around the entire execution, and nested StreamExecutor operations use thread-local
activation depth to avoid repeated `hipSetDevice`. However, the virtual
`RocmExecutor::Activate` interface still returns a freshly heap-allocated
`ScopedActivateContext`, and ordinary ROCm stream helpers create/destroy that
object around each API call.

Add a ROCm-private stack-scoped activation helper or a safe concrete context
accessor for the ROCm launch/copy/event helpers. It must use the existing
`ScopedActivateContext` logic: at nested depth it is cheap, while at outermost
depth it still calls `hipSetDevice` and distrusts any external HIP code that may
have changed the active device. Do not cache a device across outer scopes or
remove activation entirely.

Benchmark object allocation and complete StreamExecutor launch separately. This
is likely sub-microsecond but is one of the few XLA-owned costs paid by ordinary
launches. Convert call sites gradually; do not redesign the generic virtual
interface until the platform-local proof shows a measurable win.

### P17. Keep `hipExtAnyOrderLaunch` off the ordinary path

Same-stream ordering is part of StreamExecutor semantics. Do not set
`hipExtAnyOrderLaunch` globally. ROCm 7.14 also documents it as unsupported on
AMD GFX9xx, including the target gfx942/MI300 class. It may be evaluated only on
a future supported architecture with a scheduler that proves launches
independent; explicit HIP graphs already express that independence more safely.
This is a documented rejection, not an optimization to implement now.

## Phase 4: command buffers and stable addresses

### P18. Verify and likely close explicit `hipGraphUpload`

**Problem.** ROCm instantiates an executable graph and launches it without an
explicit upload/preparation step.

**Files.**

- `xla/stream_executor/rocm/rocm_command_buffer.h`
- `xla/stream_executor/rocm/rocm_command_buffer.cc`
- generic command-buffer lifecycle interfaces if preparation needs a stream

**Design.** AMD CLR source corresponding to this API currently validates the
graph and stream but leaves preparatory work as a TODO. Confirm the installed
ROCm 7.14 behavior with the M0 first-launch benchmark. If it is a no-op, close
this item and do not add lifecycle/API overhead to XLA. Only if the installed
runtime shows a real benefit should XLA add an idempotent `Prepare(Stream*)`
state. Invoke it early enough to overlap with other setup; calling it immediately
before launch on the same critical path is not a win. Track whether executable
updates invalidate upload state and re-upload only if measurement requires it.

**Tests and gate.** First launch, repeated launch, update-after-upload, launch on
a different compatible stream, destruction during/after upload, and upload
failure. Target time-to-first-token, not warm decode throughput.

### P19. Benchmark whole-graph `hipGraphExecUpdate`

**Problem.** XLA performs one HIP executable-node setter call per changed node.
HIP 7.14 offers whole-graph executable update.

**Design experiment.** Compare:

1. current individual `hipGraphExec*NodeSetParams` calls;
2. updating template-node parameters plus one `hipGraphExecUpdate`; and
3. stable-address VMM with no pointer-driven node updates.

Use several update densities and graph sizes. Whole-graph update performs
topology validation and may lose badly for sparse updates. It also requires the
template graph to be kept fully current, which the present executable-only
setters do not do.

**Landing rule.** Implement an adaptive threshold only if a stable crossover is
demonstrated across representative graphs. Otherwise prefer VMM stability and
retain per-node updates. Never choose the strategy solely by total node count;
use changed-node count and node types.

### P20. Coalesce VMM deferred deallocations under one timeline marker

**Problem.** the generic device-address VMM allocator currently enqueues a
timeline write for each `Deallocate`/`Unmap` operation. A burst of buffer teardown
therefore pays many `hipStreamWriteValue64` calls even though one later marker
can retire all earlier entries in stream order.

**Reference.** Inspect and adapt upstream commit `108e424570`, which is not an
ancestor of this checkout and explicitly includes ROCm. Its policy groups a
bounded burst under one sequence number/marker, with limits such as 64 entries
or 64 MiB to prevent unbounded delayed reclamation.

**Files.**

- `xla/stream_executor/device_address_vmm_allocator.cc`
- CUDA and ROCm device-address allocator tests
- ROCm timeline-write instrumentation from M1

**Implementation invariants.** Every object attached to a marker must have been
last used before that marker on the same stream. Flush a group before switching
streams/executors, before a lifetime-dependent remap, at threshold, and during
shutdown. The reclaimer may free all entries whose shared marker has completed,
but never an entry from a later group. Preserve bounded retained bytes as well
as bounded entry count. A failed marker submission must leave the group retryable.
If an allocation is reused before an open group is flushed, cancellation must
remove exactly that entry without corrupting byte/accounting totals. Destruction
must synchronize every executor whose peer work may still reference a mapping,
and must not invoke virtual backend operations from a partially destroyed base
object.
`WaitUntilSeqno` and every reclamation path that polls a sequence number must
flush an open group before polling, or it can wait forever for a marker that was
never submitted. Preserve ordering between paired stale reservation/allocation
entries sharing a sequence number so cancellation or reuse cannot release
virtual or physical memory early.

**Gate.** Count timeline writes during executable teardown/update and compare
VMM throughput and retained memory. This is preferable to
`hipStreamBatchMemOp`: eliminating redundant writes is better than batching the
same redundant sequence. Retain `hipStreamBatchMemOp` only as a later experiment
if multiple nonredundant wait/write operations remain naturally adjacent. Do not
use unsupported batch remote-flush or memory-barrier entries.

### P21. Add an explicit VMM stream-write self-test and diagnostics

The ROCm stable-address allocator now uses `hipStreamWriteValue64`, but unlike
CUDA it does not validate that the selected target can perform a coherent 64-bit
stream write. ROCm 7.14 has no exact “can write value 64” attribute; its
wait-value attribute is only a conservative proxy.

Preserve the current coherent host-memory timeline. At allocator creation,
optionally perform a one-time self-test: write a sentinel with
`hipStreamWriteValue64`, synchronize only the private allocator stream, verify
CPU visibility, restore zero, and return an actionable failure before reserving
large virtual-address ranges. Document this initialization-only synchronization.
Update stale header comments that still describe a signal-memory allocation.
This is robustness, not a speedup, but it is required before making
stable-address modes the default.

### P22. Roll out non-`ALWAYS_UPDATE` modes by measurement

The PJRT VMM allocator is no longer CUDA-only; ROCm has
`RocmDeviceAddressVmmAllocator` and can use non-`ALWAYS_UPDATE` modes. Do not
assume these modes are faster: earlier local measurements showed stable-address
VMM could regress throughput.

For each mode, report:

- node updates per token;
- VMM mapping/remapping and timeline writes;
- command-buffer update host time;
- graph replay time;
- allocation/reservation footprint; and
- Llama/Qwen throughput and first-token latency.

Select the default separately for ROCm rather than copying CUDA. Keep
`ALWAYS_UPDATE` if VMM bookkeeping costs more than avoided HIP graph updates.

### P23. Re-enable FUSION command buffers only through staged gates

FUSION is intentionally opt-in on ROCm at the baseline commit. Rollout order:

1. correctness with static shared memory;
2. dynamic shared-memory regression reproducer that previously returned
   `hipError_t(98)`;
3. graph create/replay with no updates;
4. sparse and dense updates;
5. library calls under capture;
6. Qwen 27B prefill and decode; and
7. Llama throughput versus eager thunks.

Do not restore `hipFuncSetAttribute` caching. History shows that the cached call
itself caused the dynamic-shared-memory graph failure. ROCm should continue to
pass `sharedMemBytes` in ordinary and graph-node launch parameters unless AMD
documents and tests a need for the attribute on the target hardware.
The regression must exercise create/update shared-memory sequences
`0 -> small -> large -> small`, not merely one fixed size.

### P24. Re-investigate the three reverted command-buffer micro-optimizations

The branch contains backports followed by same-day reverts for:

- direct kernel lookup during command recording;
- inline storage for changed allocation indices; and
- packing simple command-buffer kernel arguments without heap allocation.

Do not reapply them as a group. For each original commit:

1. cherry-pick onto a temporary branch;
2. build and run focused unit tests;
3. run the dynamic shared-memory FUSION reproducer;
4. run Qwen correctness and Llama throughput;
5. collect allocation profiles and command-buffer update timing; and
6. record the original reason for revert from available local benchmark notes or
   reproduce the regression.

Only reland an item after its failure mechanism is understood. The no-heap
argument patch is especially relevant to P15, but it packs StreamExecutor's
pointer-array representation, not necessarily the AMDGPU ABI buffer; keep those
experiments distinct.

### P24A. Instrument and optimize the traced-command cache

**Problem.** `traced_command_buffer.cc` caches traced graphs by allocation
signature with a default capacity of 16. Dynamic address patterns can retrace
and evict graphs even when most of the graph is reusable.

**First step.** Add counters for hit, miss, retrace, and eviction, then sweep
capacities 16/32/64/128 while recording memory use. A larger cache is useful only
if the workload cycles through a bounded signature set; it can otherwise retain
large graphs without reducing retracing.

**Experiments to mine, not blindly backport.** Non-ancestor commits
`da7699f75c` (graph flattening/update-throughput experiment), `2acd22f44d`
(in-place traced node address patch), `3700911598` (memcpy/memset extension),
and `bc2b27f191` (owned extra-style kernarg buffers) contain relevant prototypes.
Reconstruct their invariants and results before reuse. Preserve every dependency
edge when flattening children. Never scan opaque kernel-argument bytes for
pointer-looking values.
Any in-place patching must use compiler-provided structured argument metadata,
own storage for the full graph-executable lifetime, and fall back to retracing
when an argument cannot be proven patchable.
Opaque traced nodes from MIOpen, rocBLAS/hipBLASLt, RCCL, custom calls, and
runtime-internal captured kernels must force retracing or stable-address
execution unless XLA owns trustworthy metadata for that exact node.

**Gate.** Measure trace-cache behavior, create/update time, retained graph memory,
and steady throughput. Prior experiments reportedly improved synthetic update
throughput much more than end-to-end inference, so require an end-to-end win or
a demonstrated retrace reduction before landing.

Before attempting patching or flattening, add regression tests for direct capture
into a nonempty parent graph, repeated traces, empty traces, concurrent capture
streams, and callback/trace failure. Failure cleanup must still terminate capture
with `hipStreamEndCapture` and leave the stream usable.

### P24B. Measure and, if necessary, remove command-buffer lock contention

`CommandBufferThunk` holds a per-executor mutex through allocation comparison,
graph update, and submission. Add lock wait/hold histograms split into update and
submit phases, and test concurrent executions on multiple streams. Do not merely
narrow the lock: one graph executable has mutable node/update state and cannot
be concurrently updated/launched without a proven HIP contract.

If contention is material, design a bounded per-stream or concurrent-execution
pool of graph executables. Each entry needs independent executable-node state,
allocation-signature tracking, upload/preparation state, and stable-address/VMM
bookkeeping. Select an idle entry, update and submit it under its own lock, and
retire it only after launch completion. Bound memory growth and fall back to
serialization when the pool is full. Require multithreaded determinism and an
end-to-end throughput win before landing.

### P25. Graph priorities remain blocked on HIP runtime implementation

ROCm 7.14 headers expose node-priority symbols, but AMD documents
`hipGraphInstantiateWithFlags` as ignoring all flags and behaving as plain
instantiation. Without functional use-node-priority instantiation, setting node
attributes does not provide CUDA-equivalent scheduling. Do not implement this
in XLA until an AMD runtime test proves end-to-end priority behavior.

Also leave graph conditionals, moved-child-graph ownership, cuDNN raw graph
integration, CUDA PDL, and CUDA cluster launch on the unsupported watchlist.
Recheck them only when ROCm release notes claim implementation, not merely when
symbols appear in headers.

## Phase 5: memory allocation and host transfer

### P26. Port the async device-memory pool allocator to HIP 7.14

**Problem.** XLA's stream-ordered pool allocator and PJRT selection are CUDA-only,
although HIP 7.14 implements async allocation and memory-pool APIs.

**Files.**

- use `xla/stream_executor/gpu/gpu_cudamallocasync_allocator.*` as a behavioral
  reference, not as a name/API abstraction to extend blindly;
- add ROCm allocator implementation and tests under StreamExecutor;
- update `xla/pjrt/gpu/se_gpu_pjrt_client.cc` allocator selection;
- update BUILD targets and configuration plumbing.

**Design.**

1. Check `memoryPoolsSupported`/the ROCm 7.14 device attribute.
2. Support the default pool and an explicitly created pool. Zero-initialize
   `hipMemPoolProps`; set pinned allocation type, no exported handle type, device
   location, and `maxSize=allocator_memory` so PJRT's budget is a hard cap rather
   than a system-dependent zero/default.
3. Configure a release threshold so the pool does not return reserved memory on
   every synchronization.
4. Initially bind allocate/free exactly once to PJRT's compute stream, matching
   the CUDA allocator. The generic allocator interface has no consumer-stream
   argument and cannot safely maintain a separate allocation stream. A dedicated
   stream is a later API/lifetime redesign requiring explicit dependencies for
   every consumer.
5. Set peer access on the pool for all accessible GPUs.
6. Preserve allocation-size statistics and deterministic-allocation behavior.
   In deterministic mode disable opportunistic and internal-dependency reuse;
   retain event-dependency reuse only when ordering is explicit.
7. On OOM, optionally synchronize the bound stream and retry once, matching
   the proven CUDA allocator policy; do not add unconditional synchronization.
8. Decide how graph capture owns allocation nodes and ensure pointer-lifetime
   expectations remain compatible with PJRT donation and command buffers.
9. If preallocating by allocate/free, synchronize the bound stream before
   claiming pages are resident or clearing allocator statistics.

Initially expose this allocator only with `ALWAYS_UPDATE`. PJRT deliberately
selects the stable-address VMM allocator for other command-buffer update modes,
so replacing that allocator would invalidate their address-stability contract.
Make allocation statistics optional: always maintaining a pointer-size hash map
and mutex can erase the hot allocation win. Protect any process-wide multi-device
pool registry with a mutex.

**Tests.** Pool support failure, alignment, reuse, cross-stream lifetime,
cross-device access, capture, OOM retry, release threshold, stats, shutdown,
and allocation after a stream has been retired.

**Gate.** Compare BFC, ROCm VMM, and async pool on model load, peak reserved
memory, fragmentation, dynamic-shape execution, first token, and steady decode.
Do not make it default solely because allocation microbenchmarks improve.

### P27. Treat pinned-host NUMA policy as a measured configuration

Current ROCm allocation uses `hipHostMallocPortable`. HIP normally places pinned
memory near the active GPU; `hipHostMallocNumaUser` instead follows the caller's
NUMA policy. Therefore, blindly adding `NumaUser` is not equivalent to CUDA's
explicit NUMA allocator and may make placement worse.

**Experiment.** On a multi-socket host, bind the launch thread to the GPU-local
NUMA node and compare:

- current `hipHostMallocPortable`;
- `Portable | NumaUser` with explicit thread/memory policy;
- explicit NUMA allocation followed by `hipHostRegister`; and
- coherent versus noncoherent pinned allocation for bulk copies and small flags.

For the explicit path, mirror CUDA's tagged fallback design: `NUMAMalloc` on the
device's node with suitable alignment, `hipHostRegisterPortable`, paired
`hipHostUnregister`/`NUMAFree`, and fallback to `hipHostMalloc`/`hipHostFree`.
Use `hipHostGetDevicePointer` for zero-copy consumers rather than assuming host
and device pointer equality. AMD documents write-combined host allocation as
nonfunctional on AMD, so do not add it as a candidate.

Use system-scope events before the CPU reads noncoherent memory. Choose separate
allocators for bulk transfer buffers and small host/device synchronization words
if measurements justify it. Validate access from every context because current
portable semantics are intentional.

### P28. Separate bulk-transfer and synchronization host-memory coherence

ROCm exposes coherent and noncoherent host allocations. Coarse/noncoherent
storage can offer better bulk-transfer behavior, while fine/coherent storage is
appropriate for timeline words and CPU/GPU atomics. Audit all
`HostMemoryAllocate` users and create explicit intents rather than changing the
global default. Pair this work with P13's event visibility classes; memory
coherence and event fence scope must be designed together.

## Phase 6: autotuning, callbacks, and teardown

### P29. Port and benchmark the delay-kernel timer

CUDA optionally launches a semaphore-controlled delay kernel before timed work;
ROCm ignores `use_delay_kernel`. Hand-port upstream commit
`d6ae7162b0a1f8be57427b948083f8890d1b4550`, which already contains a ROCm
implementation and tests but is not an ancestor of this checkout. Retain its
portable/coherent host semaphore, `wall_clock64`,
`hipDeviceAttributeWallClockRate`, approximately 100 ms timeout, and disablement
when HIP launch/copy serialization environment settings could deadlock a held
delay kernel. Keep enablement target-specific and benchmark-driven. Validate
that it improves warm-up/clock-state consistency on the target AMD GPU rather
than copying NVIDIA behavior.

Tests must cover semaphore release, timeout, stream failure, teardown while held,
and timer destruction. Compare algorithm choices, variance, compile time, and
resulting model throughput. Land only if selected algorithms or reproducibility
improve materially.

### P30. Evaluate `hipExtModuleLaunchKernel` for directly profiled kernels

HIP's extension can associate start/stop events with one kernel launch. Build a
standalone benchmark comparing the current start-record, launch, stop-record
sequence with the extension. `hipExtModuleLaunchKernel` takes global work sizes,
not grid counts, so compute `blocks * threads` with checked overflow and respect
HIP's dimension limit. If useful, add a ROCm-specific profiled-launch path for
autotuners that time a single kernel. Do not replace the generic event timer,
which must also time BLAS, DNN, FFT, and multi-operation regions.

Do not set `hipExtAnyOrderLaunch`. Validate the extension's documented event
semantics on ROCm 7.14 before relying on elapsed time: the header's start-event
wording is nonintuitive and says timing excludes the system-release flush.

### P31. Improve callback failure handling without adding hot overhead

CUDA has a host-callback registry with explicit lifetime/error handling; ROCm
allocates one callback object and submits `hipLaunchHostFunc`. The current ROCm
path releases ownership before checking whether enqueue succeeded, so the error
path can leak the callback, and the error-callback overload is not fully honored.

First fix ownership on enqueue failure. If callbacks appear in profiles or
teardown failures require stronger handling, move the API-independent portions
of CUDA's registry into common GPU code or add a ROCm equivalent. Add poisoned
stream detection with `hipStreamQuery`, exclude capturing streams from an
out-of-band monitor, and invoke success/error callbacks exactly once. Compare
`hipLaunchHostFunc` with `hipStreamAddCallback`, whose callback receives a HIP
status, before choosing a primitive. Goals are correctness, safe poisoning, and
deferred cleanup—not assumed per-callback speed. Preserve HIP's prohibition on
calling HIP APIs from inside a host callback.

### P32. Replace unconditional stream-destructor synchronization only with safe retirement

ROCm synchronizes a stream before returning its handle to the stream cache. This
is intentional: reusing a still-active handle would violate ordering and lifetime
invariants. Never simply remove the synchronization.

If stream churn is measurable, implement deferred retirement:

1. record a completion event on the retiring stream;
2. put the handle and event on a retirement queue;
3. return the handle to the cache only after nonblocking completion polling; and
4. drain safely during executor shutdown.

This item depends on P31 or equivalent retirement-owned callback state. It is
not safe to destroy the wrapper asynchronously while in-flight callback
ownership remains tied to it, even when callbacks are not performance hot.

On a stream error, destroy/cancel rather than caching the handle. Callback state
must survive wrapper destruction and run exactly once. Keep retirement owned by
the executor and drain it before the executor/context disappears; alternatively,
a global queue must store only a device ordinal and other self-contained state,
never a dangling executor pointer. Bound the pending-retirement list.

Update stream-cache tests so immediate recreation while old work is pending gets
a different handle, while a completed retired stream becomes reusable later.
Add callback-survives-wrapper-destruction, poisoned-stream, and 1000-stream-churn
tests. Measure client creation/destruction and temporary autotuning streams. This
is unlikely to change steady decode when streams are long lived.

### P33. Preserve eager rocBLAS initialization

ROCm eagerly initializes BLAS so hipBLASLt/rocBLAS initialization cannot occur
inside graph capture. This is an intentional correctness workaround. Do not make
it lazy merely to improve startup. If startup becomes important, split pure
library loading from handle/workspace initialization and prove every capture path
performs the necessary initialization before capture begins.

## Already-complete optimizations to protect with regression tests

These are not implementation tasks, but the plan must not regress them:

1. ROCm captures traces directly into an existing graph with
   `hipStreamBeginCaptureToGraph`; it no longer creates a temporary graph.
2. ROCm MIOpen caches its bound stream and receives stream-destruction
   notifications.
3. ROCm caches peer-access capability in `RocmExecutor`.
4. PJRT supports `RocmDeviceAddressVmmAllocator` for stable addresses and uses
   `hipStreamWriteValue64` for deferred deallocation.
5. The historical per-update `hipFuncSetAttribute` call is gone. Dynamic shared
   memory is passed through launch/node parameters, avoiding `hipError_t(98)`.
6. Generic command-buffer update tracking already skips commands whose relevant
   allocations did not change.
7. ROCm stream handles are cached with flags/priority identity and are only
   reused after completion.
8. `GpuExecutable` activates the ROCm device once around a whole execution,
   avoiding an outermost `hipSetDevice` for each nested launch.

Add or retain focused tests for these behaviors before changing adjacent code.

## Master action tracker

The `Status` column describes the intended disposition at the start of this
plan, not implementation progress.

| ID | Action | Primary metric | Risk | Status / dependency |
|---|---|---|---|---|
| M0 | Host-overhead microbenchmarks | API host time and batch rate | Low | Do first |
| M1 | Disabled runtime counters | Hot-path call counts | Low | Do first |
| M2 | Reproducible benchmark matrix | Variance and end-to-end metrics | Low | Do first |
| P1 | Occupancy `VLOG(0)` | Autotune wall time | Low | Ready |
| P2 | Device-count cache | Startup/query count | Low | Ready |
| P3 | Priority-range cache | Stream creation | Low | Ready |
| P3A | Device-property cache | Startup/query count | Low | Ready |
| P4 | VMM cached peer lookup | VMM allocation | Low | Ready |
| P5 | Graph trim/retry | OOM recovery | Low | Ready |
| P6 | Timer error propagation | Correct failure reporting | Low | Ready |
| P7 | rocBLAS stream cache | Calls/GEMM and tok/s | Medium | Highest steady-state priority |
| P8 | Stable rocBLAS workspace | Capture and GEMM overhead | High | After P7; prove 7.14 behavior |
| P8A | rocBLAS atomics-mode cache | Library-call overhead | Low | Separate transition cache |
| P9 | FFT stream cache | Small FFT latency | Medium | Benchmark first |
| P10 | MIOpen descriptor reuse | Autotune/compile time | Medium | Benchmark first |
| P11 | BLASLt StreamK/CU autotuning | GEMM/overlap throughput | Medium | ROCm 7.14 gated candidates |
| P12 | Nonblocking streams | Launch overlap and tok/s | Medium | Feature-gated A/B |
| P13 | Event visibility classes | Event cost and tok/s | High | Required before P28 |
| P14 | Fence-free timer events | Timer variance/autotune | Medium | Independent timing-only patch |
| P15 | Packed HIP kernargs | Plain launch host time | High | Standalone proof before XLA design |
| P16 | Stack-scoped nested activation | Plain launch host time | Medium | Benchmark platform-local proof |
| P17 | Any-order launch | N/A | Critical | Do not enable without dependency proof |
| P18 | Graph upload | First graph launch/first token | Low | Verify likely 7.14 no-op, then close |
| P19 | Whole-graph exec update | Command-buffer update | High | Benchmark update-density crossover |
| P20 | Coalesced VMM deferred teardown | Timeline writes/teardown | Medium | Adapt upstream bounded-marker design |
| P21 | VMM stream-write self-test | Early diagnostics | Low | Before stable-address default |
| P22 | Update-mode selection | Update time and tok/s | Medium | After M0/M1 and P4/P20/P21 |
| P23 | FUSION rollout | Correctness and tok/s | High | Last graph-default decision |
| P24 | Reverted micro-optimizations | Update allocations/host time | Medium | One-at-a-time root-cause work |
| P24A | Traced-command cache/patching | Retraces and update time | High | Instrument first; structured metadata only |
| P24B | Command-buffer exec pool | Lock wait/hold and concurrency | High | Instrument contention before design |
| P25 | Graph priorities/watchlist | N/A until runtime support | Critical | Blocked by HIP implementation |
| P26 | HIP async allocator | Allocation time/fragmentation | High | Independent feature-gated allocator |
| P27 | NUMA host-memory policy | H2D/D2H bandwidth | Medium | Multi-socket measurement required |
| P28 | Host-memory coherence intents | Transfer/signal cost | High | Depends on P13 visibility model |
| P29 | HIP delay-kernel timer | Autotune variance/results | Medium | Benchmark target GPU behavior |
| P30 | Fused profiled launch | Single-kernel autotune time | Medium | Direct kernels only |
| P31 | Callback registry | Callback/teardown correctness | Medium | Profile or failure-driven |
| P32 | Deferred stream retirement | Stream teardown | High | Depends on P31 callback ownership |
| P33 | Eager BLAS initialization | Capture correctness | High | Preserve; optimize only with proof |

## Patch ordering and dependencies

Recommended landing sequence:

1. **Measurement:** M0-M2.
2. **Independent cleanup:** P1-P6/P3A, one commit each or a small query-cache series.
3. **Timing foundation:** P14 and P29 after P6, before judging autotuned library
   candidates in P8/P10/P11.
4. **Strong steady-state candidate:** P7 rocBLAS stream caching, then P8A.
5. **Launch experiments:** P12 nonblocking streams and P15 packed kernargs,
   independently feature-gated.
6. **Event model:** P13; P28 depends on it.
7. **Command-buffer correctness:** validate P23's FUSION + `ALWAYS_UPDATE`, then
   P5/P18/P19/P24/P24A/P24B experiments.
8. **VMM rollout:** P4 and P21, then P20, then P22 stable-address measurements;
   only afterward choose P23's FUSION/update-mode defaults.
9. **Allocation:** P26, followed by P27/P28.
10. **Library cold paths and tuning:** P8-P11 after the timing foundation.
11. **Callbacks/lifecycle:** P31 before P32; retain P33.

Do not combine P7, P12, P13, P15, P19, P22, P23, or P26. Each changes a major
state/lifetime model and needs an isolated revert boundary.

## Promotion criteria

An optimization may become the ROCm default only when all applicable criteria
hold:

- plugin builds in the hermetic ROCm 7.14 configuration;
- focused unit and integration tests pass;
- Qwen 27B completes prefill and decode without asynchronous PJRT errors;
- output tokens/hashes match the baseline;
- no new HIP errors occur during graph create, update, submit, or teardown;
- median relevant microbenchmark improves by at least 5%, or the end-to-end
  metric improves by at least 1% with effect larger than measured noise;
- final synchronization time does not reveal that host work was merely deferred;
- peak memory does not regress by more than an explicitly accepted amount;
- multi-GPU peer and stream-lifetime tests pass; and
- the feature has a documented rollback switch during its first release cycle.

For an optimization aimed at correctness/reliability rather than speed, replace
the performance threshold with a deterministic reproducer that fails before and
passes after without a measurable steady-state regression.

## Expected impact by path

| Path | Most relevant work | Important limitation |
|---|---|---|
| Plain approximately 20 us kernel launch | P15 packed kernargs/allocation removal; P16 stack activation; P12 only if legacy-stream synchronization is involved | No known cache currently explains the isolated HIP enqueue latency. If P15/P16 are neutral, focus below XLA in HIP runtime/driver dispatch. |
| Command-buffer update | P20 VMM teardown; P22 stable addresses; P24/P24A traced/update packing; P24B contention; P19 whole-graph experiment | HIP node-update/retrace cost may dominate even after XLA skips unchanged commands. |
| Library-call overhead | P7 rocBLAS stream cache; P8 workspace; P9 FFT stream cache | MIOpen stream caching is already implemented; BLASLt takes a direct stream. |
| Initialization/teardown | P2-P5, P10, P18, P26-P27, P31-P33 | These may improve first token or process churn without changing warm tok/s. |
| Autotuning | P1, P6, P14, P29-P30 | Faster timing is valuable only if measurements remain accurate and algorithm selection improves. |

## Final priority shortlist

If engineering capacity requires a strict initial cut, execute these first:

1. M0 launch/event/library microbenchmarks and M1 counters.
2. P7 safe rocBLAS stream cache.
3. P15 packed HIP kernarg proof of concept and P16 stack-activation microproof.
4. P14/P29 timing foundation before advanced library autotuning.
5. P13 event-scope specialization and P12 nonblocking streams.
6. P20 coalesced VMM deferred teardown.
7. P22/P24A stable-address and traced-command update strategy measurements.
8. P26 HIP async allocator.

The rest remain in scope, but these eight efforts have the best combination of
hot-path relevance, plausible payoff, and ability to produce a decisive result.
