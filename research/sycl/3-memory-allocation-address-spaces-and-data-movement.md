# 3. Memory allocation, address spaces, and data movement

## Executive recommendation

Treat the SYCL backend memory contract as explicit SYCL USM semantics, not CUDA-compatible semantics by assumption. The backend currently provides:

- Device memory as 64-byte aligned device USM.
- Host-pinned/registered memory as 64-byte aligned host USM.
- Unified memory as shared USM only through `CreateMemoryAllocator(MemoryType::kUnified)`.
- Collective memory as ordinary device USM with a separate peer-access requirement.
- Constants as cached device USM plus a blocking upload path, not SYCL constant address-space storage.
- Dynamic local/shared memory as a SYCL `local_accessor<int8_t, 1>` plus SPIR-V Workgroup address space materialization.

The must-fix decision is to make the silent differences visible and tested: H2D/D2H "async" copies block the host unless the host pointer is SYCL host USM; `GetPointerMemorySpace()` is still unimplemented even though the runtime already relies on SYCL pointer queries; collective memory is device USM and must not imply peer visibility without successful peer enablement; and unsupported memory-space paths must fail predictably instead of being inferred from CUDA behavior.

No runtime measurements were performed for this record. Measurement-heavy items below are phrased as proposed tests or benchmarks, not observed results.

## Must/Should/Could classification

| Priority | Item | Decision |
| --- | --- | --- |
| Must | Host-pinned async copy contract | Document and test that SYCL H2D/D2H async wrappers are non-blocking only for host USM in the queue context. Fix misleading stream logs or add diagnostics for pageable-host fallback. |
| Must | Pointer memory-space introspection | Implement `SyclExecutor::GetPointerMemorySpace()` for SYCL USM or explicitly reject unsupported pointer classes with tests. Use `sycl::get_pointer_type()` and map `host/device/shared/unknown` deliberately. |
| Must | Memory-space support contract | Publish and test the split contract: raw `Allocate(size, memory_space)` supports `kDevice`, `kCollective`, and `kHost`; allocator creation supports `kUnified`, `kCollective`, and `kHost`; `kCollective` is device USM. |
| Must | Peer access correctness for collective memory | Keep collective allocations as device USM only if peer enablement and peer-address copy paths are tested across device pairs. Cache peer enablement with enough context to avoid stale ordinal-only assumptions. |
| Must | Allocation and free edge behavior | Keep or change the internal behavior intentionally: zero-size allocation returns null; null `SyclFree()` currently errors even though raw SYCL `free(nullptr, ...)` is specified as no-op. This needs caller-facing documentation and tests. |
| Should | Constants cache hardening | Keep device-USM constants, but harden cache identity with size/content validation or a collision-safe key, document the blocking upload, and address the deleter's executor-lifetime dependency. |
| Should | Device memory usage reporting | Continue reporting free memory as unknown unless a documented Level Zero-backed implementation is added. Add tests that scheduling/diagnostics tolerate `free_bytes == -1`. |
| Should | Local/shared memory limits and lowering | Validate launch-time `shared_mem_bytes` against device limits where possible, and add SPIR-V/FileCheck or runtime tests for Workgroup address space, trailing local-accessor args, and sub-byte lowering. |
| Should | Pageable-vs-host-USM copy benchmarks | Add benchmarks that quantify the fallback cost and identify pageable host-buffer paths in XLA, especially constants and host-side literal uploads. |
| Could | Pageable staging pool | Add an internal host-USM staging pool for pageable H2D/D2H callers if benchmarks show significant blocking or bandwidth loss. |
| Could | Level Zero memory telemetry | Add optional native memory-property or free-memory telemetry only after local oneAPI/Level Zero documentation is available or an official source is added to the research inputs. |
| Could | Peer-copy synchronization optimization | Replace the SYCL local collective-permute host-sync workaround only after proving a correct cross-device event protocol for the selected context policy. |
| Could | Constants cache eviction/prefetch | Add cache-size limits, eviction, and batched uploads if workload traces show constant-cache growth or repeated blocking uploads matter. |

## XLA change candidates with concrete files/functions

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc`
  - `SyclMemcpyDeviceToHostAsync()` and `SyclMemcpyHostToDeviceAsync()`: expose the host-USM-only async condition, add tests for `usm::alloc::host` vs `usm::alloc::unknown`, and consider returning structured metadata or VLOG text that says when the wrapper waited.
  - `MemcpyDeviceToHost()`, `MemcpyHostToDevice()`, `MemcpyDeviceToDevice()`: keep the event-wait behavior explicit; do not call these "async" unless the caller passed `async=true`.
  - `SyclMallocDevice()`, `SyclMallocHost()`, `SyclMallocShared()`, `SyclFree()`: document zero-size/null behavior and the 64-byte alignment contract; add wrong-context and aspect-missing tests if test hardware can cover them.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.h`
  - Allocation comments around `SyclMallocHost()` and `SyclFree()`: say "host USM" explicitly and state that null free is an internal error, not raw SYCL no-op behavior.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc`
  - `Allocate()`: document `kCollective -> kDevice USM`; guard unsupported memory spaces in a way that is visible in tests.
  - `CreateMemoryAllocator()`: keep `kUnified -> SyclMallocShared`, `kCollective -> SyclMallocDevice`, `kHost -> SyclMallocHost`; explain why `kDevice` allocator creation is unsupported while raw `Allocate(kDevice)` works.
  - `HostMemoryAllocate()`: document that this returns SYCL host USM suitable for non-blocking H2D/D2H wrappers.
  - `CreateOrShareConstant()`: add cache-key hardening, document that constants are device USM, and consider reducing blocking by staging through host USM.
  - `DeviceMemoryUsage()`: add a regression test for `free_bytes == -1` and total memory reporting.
  - `EnablePeerAccessTo()` and `CanEnablePeerAccessTo()`: make the peer-access cache robust to context/device identity, not just ordinal strings, if context policy can change.
  - Add `GetPointerMemorySpace()` implementation next to the existing stream-executor overrides.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.h`
  - Replace the `GetPointerMemorySpace()` TODO with an override declaration and a comment that maps SYCL USM kinds to `MemorySpace`.
- `~/github/openxla/xla/xla/backends/gpu/runtime/collective_permute_thunk.cc`
  - Keep the local SYCL peer-copy path guarded by peer access. Add SYCL-specific tests for the host-sync boundary behavior before changing it.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - `SyclStream::Memcpy()`: update VLOG text so H2D/D2H does not always claim an async enqueue when the lower wrapper may have waited.
  - `LaunchSyclKernel()`: add a device-limit check for `shared_mem_bytes` if the launch path can access `DeviceDescription`; otherwise test that oversized local memory fails cleanly.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_command_buffer.cc`
  - `AddLaunchNode()`: keep local-accessor argument ordering consistent with `sycl_stream.cc`; add a graph-command-buffer local-memory test if command graphs are expected to support shared memory.
- `~/github/openxla/xla/xla/service/gpu/llvm_gpu_backend/spirv_backend.cc`
  - `CompileToSPIRV()` and `MaterializeWorkgroupSlm()`: add lowering tests for kernel pointer arg address-space casts, Workgroup SLM (`addrspace(3)`), globals preservation, and sub-byte operations touching shared/local memory.

## Evidence

### XLA code evidence

- H2D/D2H/D2D copy primitives enqueue `queue::memcpy()` and wait on the returned event when `async` is false: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:45`, `:62`, `:79`.
- `SyclMemcpyDeviceToHostAsync()` queries the destination host pointer with `sycl::get_pointer_type()` and sets `async` only for `sycl::usm::alloc::host`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:535`.
- `SyclMemcpyHostToDeviceAsync()` does the same for the source host pointer: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:555`.
- `SyclMemcpyDeviceToDeviceAsync()` always calls the D2D helper with `async=true`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:575`.
- `SyclStream::Memcpy()` always logs "async memcpy" after calling the async wrappers, even though the H2D/D2H wrapper may have waited for pageable host pointers: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:246` and `:257`.
- Device, host, and shared allocations use `sycl::aligned_alloc_device`, `aligned_alloc_host`, and `aligned_alloc_shared` with alignment 64 through a default stream for the ordinal: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:657`, `:684`, `:711`.
- `SyclFree()` rejects null, validates the ordinal, calls `sycl::free(ptr, *stream_handle)`, and nulls the caller's reference on success: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:738`.
- `DeviceAllocate()`, `AllocateHostMemory()`, and the host allocation deleter wrap the runtime helpers: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:363`, `:396`, `:410`.
- `MemorySpace` is `kDevice=0`, `kUnified=1`, `kCollective=2`, `kHost=5`: `~/github/openxla/xla/xla/stream_executor/memory_space.h:25`.
- `SyclExecutor::Allocate()` maps `kCollective` and `kDevice` to device USM, maps `kHost` to host USM, and fatals for other memory spaces: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:727`.
- `CreateMemoryAllocator()` supports `kUnified` via shared USM, `kCollective` via device USM, and `kHost` via host USM; it returns unimplemented for other types, including `kDevice`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:754`.
- The generic stream-executor API says host allocations are required for async memcpy and `GetPointerMemorySpace()` is virtual but defaults to unimplemented: `~/github/openxla/xla/xla/stream_executor/stream_executor.h:181` and `:187`.
- The SYCL executor header still has a TODO for `GetPointerMemorySpace()` and does not override it: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.h:112`.
- `CreateOrShareConstant()` caches by 128-bit fingerprint, allocates with `Allocate(content.size(), 0)`, uploads with `stream->Memcpy()`, blocks with `BlockHostUntilDone()`, and captures `this` in the deleter: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:670`.
- `DeviceMemoryUsage()` reports `global_mem_size` as total and sets free memory to `-1` because SYCL has no standard free-memory query: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:967`; the same contract is documented in the header at `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.h:164`.
- Peer access first tries native Level Zero `zeDeviceCanAccessPeer`, then falls back to `ext_oneapi_can_access_peer`; enablement uses `ext_oneapi_enable_peer_access` and caches ordinal-pair strings: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc:277` and `:876`.
- Local collective permute enables peer access for every ordered device pair, then the SYCL peer-copy protocol uses host synchronization at ready/done boundaries because cross-device event barriers can stall: `~/github/openxla/xla/xla/backends/gpu/runtime/collective_permute_thunk.cc:111` and `:582`.
- Kernel launch reflects argument count, sets packed args, fills trailing scratch args with null, and passes dynamic local memory as `sycl::local_accessor<int8_t, 1>` at the trailing reflected argument: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:78` and `:118`.
- SYCL command-buffer launch repeats the local-accessor argument pattern: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_command_buffer.cc:389`.
- The SPIR-V backend explicitly runs address-space inference, rewrites SPIR kernel pointer arguments from addrspace(0) to addrspace(1) with address-space casts, materializes Workgroup SLM in addrspace(3), preserves globals with a fake use, and expands sub-byte bitreverse intrinsics: `~/github/openxla/xla/xla/service/gpu/llvm_gpu_backend/spirv_backend.cc:81`, `:98`, `:141`, `:158`, `:247`, `:288`, `:322`.
- Device description uses native Level Zero interop to get device, memory, cache, and compute properties, and sets shared-memory limits from `maxSharedLocalMemory`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_device_description.cc:156` and `:256`.

### SYCL 2020 local spec evidence

- SYCL does not require global shared backend state; two queues without a common explicit context do not have to share a context: `~/sycl/sycl-2020.html:4263`.
- Unified addressing gives stable pointer values across host/devices but does not make every address accessible everywhere: `~/sycl/sycl-2020.html:23471`.
- SYCL USM defines host, device, and shared allocation kinds; device USM is not host-accessible, host USM is device-accessible, and shared USM is host/device-accessible: `~/sycl/sycl-2020.html:23482` and `:23552`.
- USM support is optional and devices may not support all USM allocation kinds: `~/sycl/sycl-2020.html:23552`.
- Device, host, and shared allocation functions return null on insufficient resources; zero-size allocation follows `std::malloc` style unspecified return and non-null must be freed: `~/sycl/sycl-2020.html:24046`, `:24216`, `:24361`.
- Device/shared allocations require the device to be in the context or a descendant of a device in the context; otherwise they throw `errc::invalid`: `~/sycl/sycl-2020.html:24086` and `:24401`.
- `sycl::free()` permits null pointers as no-op, requires no in-progress/enqueued commands using the memory, and is not specified as blocking or non-blocking: `~/sycl/sycl-2020.html:24705`.
- USM pointer queries exist only on the host; `get_pointer_type()` returns `unknown` outside valid USM for the context, and `get_pointer_device()` throws invalid for non-USM in the context: `~/sycl/sycl-2020.html:24765`.
- Explicit pointer memory operations require USM pointers to come from the same context as the handler queue and be accessible on the handler device: `~/sycl/sycl-2020.html:30753`.
- SYCL has five address spaces: global, local, constant, private, and generic; direct declaration of address-spaced pointers is discouraged and `multi_ptr` is the portable boundary: `~/sycl/sycl-2020.html:20536`.
- SYCL provides explicit pointer aliases for global, local, constant, private, and generic address spaces; `constant_ptr` is deprecated in SYCL 2020: `~/sycl/sycl-2020.html:22923`.
- `local_accessor` allocates device local memory shared by work-items in a work-group, with an independent copy per simultaneously executing work-group: `~/sycl/sycl-2020.html:18868`.
- `local_accessor` exposes `multi_ptr<..., local_space, ...>` and constructors tied to a command-group handler: `~/sycl/sycl-2020.html:19003`.
- `handler::set_arg()` is backend-specific interop; the precise semantics are defined by each backend: `~/sycl/sycl-2020.html:29625`.

### oneAPI/Level Zero local evidence

- `~/sycl/oneapi.md:32` routes Level Zero evidence to `oneapi/MultiTileCardWithLevelZero.md` and `oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md`.
- Targeted local searches found no `oneapi/` documentation tree under `~/sycl` or `~/github/openxla/xla`. The commands `rg --files ~/sycl | rg '/oneapi/'`, `rg --files ~/github/openxla/xla | rg '/oneapi/'`, and `find ... -name oneapi` produced no documentation paths.
- Therefore this record uses only Level Zero facts directly evidenced by XLA source: `sycl::get_native<backend::ext_oneapi_level_zero>()`, `zeDeviceCanAccessPeer`, `zeDeviceGetMemoryProperties`, `zeDeviceGetComputeProperties`, and `ext_oneapi_*peer_access` calls. It does not claim MultiTileCard USM sharing/isolation semantics.

## Findings

1. H2D/D2H "async" is conditional. The stream API path can block the host inside the async wrapper when the host pointer is pageable or otherwise not `sycl::usm::alloc::host` in the stream context. This is a CUDA-shaped assumption leak because XLA callers may see `Stream::Memcpy()` as asynchronous.
2. Host memory allocated through `HostMemoryAllocate()` is SYCL host USM and is the safe path for non-blocking H2D/D2H submission. Existing async copy tests use host USM helpers, so they do not cover pageable fallback.
3. `SyclFree()` is stricter than raw SYCL free. SYCL specifies null free as no-op, but the XLA helper returns `InvalidArgument` and tests expect that. Keep it if desired, but treat it as an XLA helper contract.
4. `kCollective` is not a special allocation kind in SYCL. It is device USM, and collective correctness depends on peer access and peer-address bookkeeping outside the allocator.
5. `kUnified` works through `CreateMemoryAllocator(MemoryType::kUnified)` only. Raw `Allocate(size, kUnified)` is unsupported and currently falls into `LOG(FATAL)`.
6. `GetPointerMemorySpace()` is missing despite direct evidence that SYCL can query USM allocation kind. This leaves generic stream-executor paths without a backend answer and forces callers to rely on backend-specific knowledge.
7. Constants are not in SYCL constant memory. They are device USM allocations cached by content fingerprint, uploaded through the stream, and synchronized before return. The cache has a documented collision assumption and a deleter that depends on executor lifetime.
8. Free-memory reporting is intentionally incomplete. Total memory is reported, but free memory is `-1`; no local oneAPI docs were available to justify a Level Zero free-memory implementation.
9. Peer access is directional, capability-gated, and cached globally by ordinal pair. That may be too weak if the context policy or device enumeration changes. The collective-permute SYCL path also uses host synchronization for correctness around cross-device event stalls.
10. Local/shared memory lowering exists but lacks enough guardrails. Runtime launch passes a SYCL local accessor, SPIR-V lowering materializes Workgroup SLM in addrspace(3), and device description records a max local-memory value, but this record found no launch-time limit check or focused lowering tests.

## Proposed patch plan

1. Contract documentation and low-risk tests.
   - Update comments and VLOG text in `sycl_gpu_runtime.h`, `sycl_gpu_runtime.cc`, `sycl_stream.cc`, and `sycl_executor.cc`.
   - Add tests that distinguish host USM from pageable host pointers for H2D/D2H wrappers. Do not assert performance in unit tests; assert classification and use benchmarks for timing.
   - Add tests that lock in zero-size allocation, null free, invalid ordinal, and wrong-context behavior where hardware/runtime support allows.

2. Pointer introspection.
   - Add `SyclExecutor::GetPointerMemorySpace(const void*)` in `sycl_executor.h/cc`.
   - Proposed mapping: `sycl::usm::alloc::device -> MemorySpace::kDevice`, `host -> MemorySpace::kHost`, `shared -> MemorySpace::kUnified`, `unknown -> InvalidArgument` or `Unimplemented` with a clear message.
   - Add tests for device USM, host USM, shared USM, null, pageable host pointer, cross-context pointer, and peer-accessible pointer.

3. Memory-space behavior and constants.
   - Add explicit tests/documentation for raw `Allocate()` vs `CreateMemoryAllocator()` support.
   - Keep `kCollective` allocation as device USM, but make that visible in tests using `get_pointer_type()`.
   - Change constant-cache key to include size at minimum; optionally add debug content verification on weak-cache hits.
   - Document that `CreateOrShareConstant()` blocks and uses device USM, not SYCL constant address space.

4. Peer access and collective memory.
   - Add multi-device tests that call `CanEnablePeerAccessTo()`, `EnablePeerAccessTo()`, repeated enablement, and D2D copy through peer-visible collective allocations.
   - Include skipped tests for fewer than two devices, but log topology/device identities so missing coverage is visible.
   - Revisit ordinal-only peer cache if context policy research decides that ordinals can be reused across distinct contexts.

5. Memory usage and local/shared memory.
   - Add `DeviceMemoryUsage()` tests for total positive and free unknown.
   - Add optional native-memory telemetry only after official docs or checked local references are available.
   - Add local-memory launch tests for within-limit and over-limit shared memory.
   - Add SPIR-V backend tests for Workgroup SLM, kernel arg addrspace casts, globals preservation, and sub-byte operations in shared/local memory paths.

## Test/benchmark coverage

Existing local coverage found:

- `//xla/stream_executor/sycl:sycl_gpu_runtime_test`
  - Async H2D/D2H tests use `SyclMallocHost()` host USM helpers: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime_test.cc:31`, `:419`, `:445`.
  - Allocation tests cover shared allocation, invalid ordinals, zero-size allocation, null free, and double free: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime_test.cc:684`, `:690`, `:700`, `:721`.
- `//xla/stream_executor/sycl:sycl_executor_test`
  - `CreateMemoryAllocator()` tests cover `kUnified`, `kHost`, `kCollective`, and unsupported `kDevice`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor_test.cc:140`.
- `//xla/stream_executor/sycl:sycl_device_description_test`
  - Sanity checks include positive `shared_memory_per_block()` and `shared_memory_per_core()`: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_device_description_test.cc:27`.

Proposed additions:

- Unit tests in `sycl_gpu_runtime_test.cc`
  - `MemcpyH2DAsyncHostUsmDoesNotWaitForUnrelatedWork`
  - `MemcpyH2DAsyncPageableHostFallsBackToWait`
  - `MemcpyD2HAsyncHostUsmDoesNotWaitForUnrelatedWork`
  - `MemcpyD2HAsyncPageableHostFallsBackToWait`
  - `SyclFreeNullDocumentsInternalContract`
  - `SyclMallocWrongContextOrUnsupportedAspectReturnsStatus`
- Unit tests in `sycl_executor_test.cc`
  - `GetPointerMemorySpaceDeviceUsm`
  - `GetPointerMemorySpaceHostUsm`
  - `GetPointerMemorySpaceSharedUsm`
  - `GetPointerMemorySpacePageableUnknown`
  - `CollectiveAllocatorReturnsDeviceUsm`
  - `DeviceMemoryUsageReportsUnknownFreeMemory`
  - `CreateOrShareConstantReusesLiveConstant`
  - `CreateOrShareConstantRecreatesExpiredConstant`
- Multi-device tests, likely a new `sycl_executor_multigpu_test.cc`
  - `CanEnablePeerAccessToIsDirectionalAndStable`
  - `EnablePeerAccessToIsIdempotent`
  - `CollectiveDeviceUsmPeerD2DCopyRoundTrip`
  - `PeerAccessSkippedWhenLessThanTwoDevices`
- Runtime/codegen tests
  - Local-memory launch within and above `shared_memory_per_block`.
  - Command-buffer launch with `shared_mem_bytes > 0`.
  - SPIR-V lowering FileCheck for addrspace(1) kernel pointer args, addrspace(3) Workgroup SLM, and sub-byte shared/local operations.
- Benchmarks
  - H2D/D2H throughput and host-call latency for pageable host memory vs `SyclMallocHost()` host USM, by size.
  - Allocation/free latency and fragmentation for `aligned_alloc_device`, `aligned_alloc_host`, and `aligned_alloc_shared` with 64-byte alignment.
  - Constant creation latency for cold cache, warm live cache, and expired weak-cache entry.
  - Peer D2D copy bandwidth/latency for same-root tiles vs separate root devices, if the hardware topology is available.

## Rollout risk

- Making pageable H2D/D2H behavior explicit can surface latent callers that relied on async-looking APIs while passing ordinary host pointers. The least risky first change is diagnostic/logging and tests; enforcing host-USM-only async semantics should wait for caller audit.
- Implementing `GetPointerMemorySpace()` can change generic stream-executor behavior. Unknown/pageable pointers should return a clear error instead of being guessed as `kHost` unless XLA explicitly wants pageable host pointers in this API.
- Changing `SyclFree(nullptr)` to match raw SYCL no-op semantics would break existing SYCL tests and may hide double-free mistakes. Keep current behavior unless all callers are audited.
- Peer-access cache changes can affect collective startup. Strengthening the key to include native device/context identity is safer than weakening capability checks.
- Replacing local collective-permute host synchronization is correctness-sensitive. Keep the host-sync workaround until cross-device event behavior is proven on target hardware and context policy.
- Free-memory estimates from native APIs can be worse than unknown if they are per-tile, per-root, cached, or not comparable to XLA allocation visibility. Treat any implementation as advisory until benchmarked.
- Local-memory limit checks may reject kernels earlier than today. This is desirable for diagnostics, but tests must account for backend-specific reported limits.

## Evidence gaps

- The oneAPI docs referenced by `~/sycl/oneapi.md` were not present locally. In particular, this record could not read `oneapi/MultiTileCardWithLevelZero.md`, `oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md`, or Level Zero memory API documentation.
- The route-map reference for "address spaces S|4|14741-15172" does not match the local `sycl-2020-map.md`; that range lands in host-memory/data-management text. The local map places address-space classes at `S|4|20536-23101`.
- No runtime measurements were run for allocation latency, fragmentation, pageable-vs-host-USM copy behavior, peer bandwidth, constant upload latency, or local-memory occupancy.
- No hardware topology evidence was collected. Same-root tile, multi-root tile, and separate-card behavior remain unproven from local sources.
- No full caller audit was performed for every `SyclStream::Memcpy()` host pointer source. Constants are one proven pageable-looking path, but literals, host callbacks, and higher-level XLA buffer transfers need follow-up.
- No XLA source files were modified for this record.
