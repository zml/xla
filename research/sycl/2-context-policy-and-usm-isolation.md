# 2. Context policy and USM isolation

## Executive recommendation

Keep XLA's current default of one cached SYCL context per visible XLA ordinal. Do not replace it with a single shared multi-root context for multi-card or multi-root systems.

The current policy is a good default for XLA's independent per-device BFC arenas: it keeps device USM allocation and free paths bound to the executor ordinal, avoids merging unrelated root-device allocation budgets into one context, and matches the oneAPI guide's highest-performance "single sub-device in its own context" model when ordinals are tiles.

The required follow-up is not a broad context-ownership rewrite. XLA should add topology-aware checks/tests around USM visibility and cross-ordinal D2D copies. Core SYCL says USM allocations are associated with a context and kernels or explicit memory operations that use a USM pointer must use that same context; violations are undefined behavior. oneAPI peer access can enable access to another device's USM device allocation, but the peer extension still describes peer device allocations "in the same context" and says access remains subject to normal SYCL context rules. XLA currently has copy helpers and tests that submit a D2D copy on one ordinal's queue while using a destination allocation made through a different ordinal/context; that path needs explicit capability coverage and a fallback policy.

Recommended policy:

- Root devices: keep one context per visible root ordinal. Use peer access only after `CanEnablePeerAccessTo` succeeds; otherwise fail the peer D2D path or use a host-staged fallback where correctness requires a copy.
- Tiles/subdevices exposed as XLA ordinals: keep one context per visible tile by default for isolation and per-tile performance. Treat "physical device" in existing comments as "visible SYCL device/XLA ordinal" unless XLA explicitly records root/tile topology. If XLA later needs tile-to-tile shared device USM on the same card, add an opt-in same-root shared-subdevice context policy, not a global default.
- Multi-card/multiple root devices: do not create one shared context containing all roots as the default. The oneAPI guide says that model offers sharing at a cost and can require slow access through host memory or explicit copies. It also undermines XLA's per-device arena isolation.

## Must/Should/Could classification

Must:

- Keep allocation and free on the same SYCL context. `sycl::free(ptr, context/queue)` requires `ptr` to have been allocated against that context, and XLA's current `SyclFree(device_ordinal, ptr)` relies on that property.
- Treat cross-context USM pointer use as non-portable unless proven by a Level Zero/oneAPI capability path for the exact device pair and context policy. Core SYCL makes cross-context kernel dereference and explicit memory operations undefined.
- Avoid a shared multi-root Level Zero context as the default context policy. It changes USM visibility, context-level allocation behavior, native interop handles, and the accounting assumptions behind independent XLA BFC arenas.

Should:

- Keep the per-ordinal context cache, but clarify the comment in `SyclDevicePool::GetDeviceContext()` so "physical device" does not imply "Level Zero root device" when subdevices/tiles are exposed.
- Add topology discovery/logging for root devices versus subdevices, including native Level Zero handles where available, so tests and diagnostics can classify same-root tiles versus separate roots.
- Add peer/copy checks that ensure cross-ordinal D2D paths only run when the peer pair is enabled and the queue/context rules are satisfied or the implementation has a documented native Level Zero path.
- Add regression tests for same-root tile, multi-root same-platform, and multi-platform cases. Current tests cover per-ordinal contexts and some two-device copies, but they do not assert topology, pointer visibility, peer enablement, or fallback behavior.

Could:

- Add an opt-in same-root tile policy that groups subdevices from one root into a shared context for workloads that explicitly want tile-to-tile sharing. This should be separate from the default per-ordinal policy and should not include multiple root devices.
- Add a benchmarking mode that compares option A (single tile contexts), option B (shared same-root subdevice context), option C (root-device implicit scaling), and option D (multi-root context) using the oneAPI guide's taxonomy.
- Expose diagnostic metadata in XLA device descriptions: Level Zero backend, root/subdevice classification, native handle identity, parent/root id, `ZE_AFFINITY_MASK`, and peer access matrix.

## XLA change candidates with concrete files/functions

No XLA source was edited for this note. Candidate changes:

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc`
  - `SyclDevicePool::InitDevicePool()` lines 132-163: add topology capture when enumerating Level Zero GPU devices. Today it pushes every `platform.get_devices()` GPU into `device_pool_` without recording whether the device is a root device or subdevice.
  - `SyclDevicePool::GetDeviceContext()` lines 166-187: keep the per-ordinal cache, but change the comment at lines 179-181 from "physical device" to "visible SYCL device/XLA ordinal", or add a second sentence explaining that root/tile meaning depends on discovery.
  - `SyclStreamPool::InitStreamPool()` lines 227-265 and `GetOrCreateStream()` lines 284-318: queues already bind to the same per-ordinal context and device. Add an invariant check that the queue's `get_context()` contains the selected device and record context/device topology in VLOG output.
  - `SyclMemcpyDeviceToDevice()` lines 514-532 and `SyclMemcpyDeviceToDeviceAsync()` lines 575-590: add a validated cross-ordinal API or a pointer/context validation helper. The current functions accept raw pointers and a queue/ordinal, so they cannot prove that both pointers are valid in the queue context.
  - `SyclMallocDevice()`, `SyclMallocHost()`, `SyclMallocShared()`, `SyclFree()` lines 657-755: keep allocation/free by default stream/context; add tests that `get_pointer_type` and `get_pointer_device` return expected results for the allocating context and `unknown`/invalid for unrelated contexts.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_context.cc`
  - `SyclContext::Create()` lines 22-27: no default policy change. If an opt-in same-root tile context is later added, it should be selected here through a topology-aware policy object, not by changing every ordinal to a global context.
  - `SyclContext::Synchronize()` lines 34-36: keep ordinal-scoped stream-pool synchronization. A shared context must not accidentally widen synchronization to unrelated ordinals.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc`
  - `SyclExecutor::Init()` lines 477-481: record executor topology metadata after `GetDevice()` and `SyclContext::Create()`.
  - `DeviceAllocate()`, `DeviceDeallocate()`, `HostAllocate()`, `AllocateHostMemory()` lines 362-430 and `SyclExecutor::Allocate()` / `CreateMemoryAllocator()` lines 727-813: keep ordinal-bound allocator behavior. Add allocation diagnostics that include context id and device/root id.
  - `LevelZeroCanAccessPeer()` lines 277-297, `CanEnablePeerAccessTo()` lines 927-965, `EnablePeerAccessTo()` lines 876-925: keep the Level Zero native query before SYCL fallback. Add tests for asymmetric peer enablement and repeated enablement; consider implementing the ordinal overload `CanEnablePeerAccessTo(int)` so callers can check before they only have an ordinal.
  - `LoadLevelZeroModule()` and `GetModuleFunction()` lines 123-146 and 238-243: native modules/kernels are created against the executor's SYCL/Level Zero context. A shared-context policy would affect module cache identity and must be included in the module cache key if ever added.

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - `SyclStream::Memcpy(DeviceAddressBase*, const DeviceAddressBase&, uint64_t)` lines 269-278: enforce the base `Stream` contract from `~/github/openxla/xla/xla/stream_executor/stream.h` lines 184-186, which says peer access must be enabled between owning executors for GPU-to-GPU copies.

## Evidence

### XLA code evidence

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 132-163 initializes the device pool by selecting platforms whose name contains "Level-Zero" and pushing every GPU from `platform.get_devices()`. There is no root/subdevice classification in this code.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 166-187 caches one `sycl::context` per `device_ordinal`. The comment at lines 179-181 says this keeps USM scoped to one physical device and avoids shared context-level allocation budget competition.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_context.cc` lines 22-27 delegates context creation to `SyclDevicePool::GetDeviceContext(device_ordinal)`. Lines 34-36 synchronize the stream pool for that ordinal.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 227-265 and 284-318 create SYCL queues from the ordinal's context and device. This means kernels and explicit memory operations submitted on those queues inherit the per-ordinal context.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 657-755 allocate device, host, and shared USM through the default stream for the ordinal, and free through the default stream for the same ordinal.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc` lines 362-430 wrap allocation/free through `context->device_ordinal()`, and lines 727-813 bind device, host, unified, and collective memory allocators to `device_ordinal()`.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc` lines 277-297 implement `LevelZeroCanAccessPeer()` using `zeDeviceCanAccessPeer`; lines 927-965 use it before falling back to `device.ext_oneapi_can_access_peer`; lines 876-925 enable one-way peer access and cache enabled ordinal pairs.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc` lines 514-532 and 575-590 perform D2D copies using one selected queue/context but accept raw pointers without checking the allocating context or peer status.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime_test.cc` lines 126-146 already tests cached per-ordinal contexts and that ordinal 0 and ordinal 1 contexts differ. Lines 596-621 and 631-665 test two-device D2D copies, but they do not assert topology or peer enablement.

### SYCL 2020 spec evidence

- `~/sycl/sycl-2020-map.md` maps context class to `S|4|7075-7451|sec:context-class`, USM to `S|3|23305-24820|sec:usm`, USM allocation restrictions to `S|4|23783-24764|_usm_allocations`, and backend interop to `S|4|5179-5541|sec:backend-interoperability`.
- `~/sycl/sycl-2020.html` lines 7075-7079 define a SYCL context as backend state used to interact with a group of devices associated with a platform.
- `~/sycl/sycl-2020.html` lines 7168-7174 define construction from a single device; lines 7185-7196 define construction from a device list and require all devices to be associated with the same platform.
- `~/sycl/sycl-2020.html` lines 7224-7233 and 7306-7322 expose the devices associated with a context.
- `~/sycl/sycl-2020.html` lines 23552-23557 state USM support is optional and should be queried through device aspects.
- `~/sycl/sycl-2020.html` lines 23584-23637 summarize USM visibility: device allocation is not host-accessible, another-device access is optional P2P, host allocation is accessible by any device, and shared allocation has optional another-device behavior.
- `~/sycl/sycl-2020.html` lines 23640-23649 are the key correctness rule: each USM allocation has an associated context, and any kernel or explicit memory command using that pointer must use the same context; violating this is undefined behavior.
- `~/sycl/sycl-2020.html` lines 24582-24588 and 24645-24652 require non-host USM allocation device arguments to be contained by the context or be a descendent device of a contained device.
- `~/sycl/sycl-2020.html` lines 24720-24731 require `sycl::free(ptr, ctxt)` to receive a pointer allocated against that context and have no in-progress/enqueued commands using it.
- `~/sycl/sycl-2020.html` lines 24787-24813 define pointer queries: `get_pointer_type(ptr, context)` returns `unknown` if the pointer is not valid for that context, and `get_pointer_device(ptr, context)` throws for a pointer outside that context.
- `~/sycl/sycl-2020.html` lines 5179-5228 describe backend interop as optional and non-generic SYCL. Lines 5325-5366 define `get_native`, including backend mismatch behavior. Lines 5370-5537 define `make_*` interop functions and require matching backend for context-taking overloads.

### oneAPI/Level Zero evidence

- `~/sycl/oneapi.md` lines 7-9 and 32 confirm the local route map points to `oneapi/MultiTileCardWithLevelZero.md` and `oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md`. Both files exist locally.
- `~/sycl/oneapi/MultiTileCardWithLevelZero.md` lines 30-32 describe multiple GPUs as multiple SYCL root devices; on Linux they can be root devices of the same Level Zero platform, while on Windows they can appear as multiple Level Zero platforms.
- `~/sycl/oneapi/MultiTileCardWithLevelZero.md` lines 43-55 describe multi-tile GPUs: a root device can be partitioned into subdevices corresponding to physical tiles, `create_sub_devices` returns persistent order, and `ZE_AFFINITY_MASK` can control exposed subdevices.
- `~/sycl/oneapi/MultiTileCardWithLevelZero.md` lines 64-67 state contexts are used for resource isolation and sharing; a context may contain one or multiple root devices/subdevices from the same SYCL platform.
- `~/sycl/oneapi/MultiTileCardWithLevelZero.md` lines 74-90 describe Level Zero USM behavior: device USM is accessible only by the specified device, host USM by host and any device in the context, shared USM by host and the specified device, and root-device allocations are accessible by all its subdevices.
- `~/sycl/oneapi/MultiTileCardWithLevelZero.md` lines 98-101 say buffers in same-root subdevice contexts are allocated on the root and accessible to devices in that context, while buffers for contexts with different root devices are allocated on host.
- `~/sycl/oneapi/MultiTileCardWithLevelZero.md` lines 105-168 define context options A-D. Option A is one context per subdevice/tile, with execution/visibility limited to that subdevice and no data sharing across queues. Option B is a shared same-root subdevice context. Option C uses one root-device context for implicit driver scaling. Option D is a multi-root context with sharing at a cost and explicit copies/slow host access.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md` lines 72-113 map SYCL device/context to `ze_device_handle_t` and `ze_context_handle_t` and require a `DeviceList` when constructing a SYCL context from a native Level Zero context.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md` lines 309-313 say only full-platform root-device contexts are guaranteed to reuse the Level Zero driver default context; subset, subdevice, and custom contexts are not guaranteed to map that way.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md` lines 370-380 say `make_device` copies one of the fixed enumerated devices/subdevices, and `make_context` requires at least one device, all from the same SYCL platform/Level Zero driver.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md` lines 391-410 require `make_queue`'s device to be in the provided context and Level Zero events to come from an event pool created in the same context.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_peer_access.asciidoc` lines 49-60 say peer access is implemented for GPU devices but only CUDA, HIP, and Level Zero allow peer-to-peer memory access, and the extension applies to USM device allocations, not USM shared allocations.
- `~/sycl/oneapi/extensions/supported/sycl_ext_oneapi_peer_access.asciidoc` lines 89-153 define peer access as access to USM device allocations for a peer device in the same context and subject to normal context rules.

## Findings

1. XLA intentionally uses one SYCL context per visible ordinal today. The implementation is not accidental: the context cache is keyed by ordinal, queues are created from that context and ordinal's device, allocators allocate/free through that ordinal, and tests assert different contexts for different ordinals.

2. "Physical device" in the current comment is imprecise. In current code it means `device_pool_[device_ordinal]`, which is the visible SYCL device returned by `platform.get_devices()`. The code does not prove that this is a Level Zero root device. If subdevices are exposed or XLA later partitions root devices into tile ordinals, the same comment would refer to the visible tile/subdevice, not the physical card/root.

3. Per-ordinal contexts are the safest default for XLA root-device and multi-card execution. A shared multi-root context would widen resource sharing and can change buffer/USM placement. The oneAPI guide explicitly describes multi-root contexts as the least performant option in its list and notes slow host access or explicit copies.

4. Same-root tile execution has two valid models, but they optimize different things. Per-tile contexts match oneAPI option A: best per-tile performance and no sharing across queues. A shared same-root subdevice context matches option B: explicit scaling with data sharing across queues. XLA should not silently switch models because the allocator and peer-copy contracts differ.

5. Device USM visibility does not become global just because pointer values are unified. Core SYCL unified addressing preserves pointer values, but accessibility is still constrained by allocation type, context, and optional P2P support.

6. The current cross-device D2D wrappers need stronger correctness checks. `SyclMemcpyDeviceToDevice(kDevice0, dst_on_device1, src_on_device0, ...)` submits on device 0's context while `dst_on_device1` was allocated through device 1's context. Core SYCL marks cross-context explicit memory operations undefined. If the implementation happens to accept it, XLA still needs tests that prove this through the Level Zero peer path and document the dependency.

7. Native Level Zero interop is context-sensitive. Modules, queues, events, and buffers/images are tied to native contexts. A future shared-context policy would affect module cache keys, event/queue construction, and native handle ownership; it is not just a `sycl::context` constructor change.

## Proposed patch plan

1. Clarify the existing policy without changing ownership.
   - Update the `GetDeviceContext()` comment to say "visible SYCL device/XLA ordinal" and mention that root/tile meaning depends on discovery.
   - Add VLOG diagnostics for each ordinal: backend, `device.is_gpu()`, platform, native Level Zero device handle, context native handle, and whether the device has a parent/root if that can be queried.

2. Add a topology helper.
   - In `SyclDevicePool`, store per-ordinal metadata: platform id/backend, native `ze_device_handle_t`, root/subdevice classification, parent/root handle where available, and same-root grouping.
   - Keep this helper read-only at first so it does not perturb device enumeration.

3. Harden peer-copy capability.
   - Add a SYCL test helper that checks `get_pointer_type` and `get_pointer_device` for source and destination pointers under source and destination contexts.
   - Add an executor-level validated D2D path that requires either same context or an explicitly enabled/supported peer path for the source/destination ordinals.
   - Decide the fallback behavior for unsupported pairs: return `FailedPrecondition` at the stream executor layer, or add an explicit host-staged copy for paths that require correctness over performance.

4. Preserve allocator isolation.
   - Keep `kDevice`, `kCollective`, `kHost`, and `kUnified` allocations bound to the executor ordinal/context.
   - Add allocation debug checks in tests, not hot-path production code initially, to avoid runtime overhead.

5. Only then consider an opt-in same-root tile shared context.
   - Gate it behind a clearly named experimental flag or policy enum.
   - Restrict it to subdevices of the same root device and same SYCL platform.
   - Include context policy in module cache identity and allocator diagnostics.
   - Do not enable it for multi-root or multi-platform systems.

## Test/benchmark coverage

Current coverage found:

- `SyclGpuRuntimeTest.TestStaticDeviceContext` verifies repeated calls for one ordinal return the same context.
- `SyclGpuRuntimeTest.TestDeviceContextsArePerOrdinal` verifies ordinals 0 and 1 have different contexts when at least two SYCL devices are visible.
- `SyclGpuRuntimeTest.TestMultiDeviceAllocationAndSyncCopy` and `TestMultiDeviceAllocationAndAsyncCopy` perform two-device D2D copies, but they do not classify topology, validate peer enablement, or assert pointer/context legality.

Recommended tests:

- Context identity:
  - Assert per-ordinal contexts stay distinct by default.
  - Assert each context contains the ordinal's SYCL device.
  - Assert `get_native<level_zero>(context)` identity differs for per-ordinal contexts unless the runtime documents aliasing.

- USM visibility:
  - Allocate device, host, and shared USM on ordinal A.
  - Query `get_pointer_type`/`get_pointer_device` in context A and context B.
  - Expect valid results in the allocating context; expect `unknown` or `errc::invalid` for unrelated contexts unless a documented shared-context policy applies.
  - Repeat for same-root tiles and multi-root devices.

- Peer access:
  - Test `CanEnablePeerAccessTo` both directions for same-root tiles and separate roots.
  - Test repeated enablement does not surface an unhandled `errc::invalid`.
  - Test copy behavior when peer access is false: failure or explicit host-staged fallback, whichever policy is chosen.

- D2D copy correctness:
  - Same ordinal: device-to-device copy remains valid.
  - Same-root tiles, per-tile contexts: copy must be gated by peer/context validation.
  - Same-root tiles, opt-in shared context: copy should pass without cross-context pointer use.
  - Multi-root same-platform: copy only after peer capability says yes; otherwise fallback/fail.
  - Multi-platform: no shared SYCL context; peer query should fail or return false.

- Benchmarks/experiments:
  - Compare oneAPI options A-D: per-tile contexts, shared same-root tile context, root-device implicit scaling, and shared multi-root context.
  - Measure max allocatable device USM per ordinal, total allocation pressure across ordinals, allocation/free latency, H2D/D2H/D2D bandwidth, and kernel overlap.
  - Record `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, `CreateMultipleRootDevices`, `CreateMultipleSubDevices`, Level Zero adapter mode, device names, native handles, and platform/backend ids.

No hardware/runtime measurements were made in this research pass. The above experiments are required before claiming performance or allocation-budget numbers.

## Rollout risk

- Low risk: documentation/comment changes and VLOG-only topology diagnostics.
- Medium risk: adding pointer/context validation to tests and debug builds may expose existing undefined behavior in current two-device D2D tests.
- Medium risk: changing D2D copy behavior to fail when peer/context checks are missing could surface latent callers that rely on implementation-defined cross-context behavior.
- High risk: switching default context ownership to a shared same-root or multi-root context. It can change allocation budget contention, memory placement, module/cache identity, event/queue interop constraints, and synchronization scope.
- Highest risk: using one shared context across multiple root devices/cards by default. It conflicts with the current allocator isolation rationale and oneAPI's guidance that multi-root sharing trades performance for sharing convenience.

## Evidence gaps

- No gap for the requested oneAPI docs: `oneapi/MultiTileCardWithLevelZero.md`, `oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md`, and `oneapi/extensions/supported/sycl_ext_oneapi_peer_access.asciidoc` are present under `~/sycl/oneapi`.
- No local measurement proves that the current per-ordinal contexts avoid Level Zero context-level allocation budget competition; this is an implementation rationale in the XLA comment that still needs measurement on real hardware.
- No local measurement proves whether current cross-context D2D copies are implemented through a valid Level Zero peer path or are relying on undefined-but-working behavior.
- No local evidence maps XLA visible ordinals to Level Zero root devices versus subdevices under `ZE_AFFINITY_MASK`; XLA currently does not log or persist that topology.
- No tests found that cover same-root tile subdevices, multiple root devices on Linux, multiple Level Zero platforms on Windows, or asymmetric peer access.
- No evidence yet that a shared same-root tile context improves end-to-end XLA workloads enough to justify the allocator and interop complexity.
