# 1. Device identity, ordering, and tile exposure

## Executive recommendation

XLA should define a SYCL device ordinal as the index of a visible Level Zero SYCL GPU device in the process-local SYCL enumeration after oneAPI/SYCL filtering has been applied. This means the ordinal is not intrinsically a physical card, a Level Zero root handle, or a tile. It is whatever DPC++ exposes as a SYCL root device for the active environment, restricted by XLA to the Level Zero backend and GPU devices.

The current XLA behavior is close to this policy but is not explicit enough for multi-GPU reliability. XLA must replace platform-name substring matching with `get_backend() == sycl::backend::ext_oneapi_level_zero` because the rest of the SYCL backend immediately uses Level Zero native interop. XLA should document and test that ordinals preserve SYCL/oneAPI visible-device order, including `ONEAPI_DEVICE_SELECTOR` tile exposure. XLA should not silently call `create_sub_devices()` during default discovery; tiles should become XLA ordinals only when the SYCL runtime exposes them as root devices, or through a future explicit XLA opt-in.

## Must/Should/Could classification

- Must: change discovery in `SyclDevicePool::InitDevicePool()` to check the backend enum, not `info::platform::name` containing `"Level-Zero"`.
- Must: reject or skip any device whose `device.get_backend()` is not `backend::ext_oneapi_level_zero` before later code can call Level Zero `get_native`.
- Should: document the ordinal policy in `sycl_gpu_runtime.h` and log enough identity fields to make root-vs-tile visible.
- Should: add regression tests for Level Zero-only discovery and for ordinal preservation under `ONEAPI_DEVICE_SELECTOR`.
- Should: preserve SYCL/oneAPI order by default. Do not unconditionally sort if it would override user selector order.
- Could: add opt-in stable sorting for unfiltered discovery, gated so `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, and other explicit visibility controls keep user order.
- Could: add an explicit XLA tile-exposure mode that calls `create_sub_devices()` only when requested, and never mixes parent root devices with child tile devices unless the user explicitly asks for that mixed topology.
- Could: enrich diagnostics with DPC++ `ext_oneapi_index_within_platform`, Intel PCI/BDF, UUID, and stack/tile-count queries when the corresponding extension aspects are present.

## XLA change candidates

- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc`
  - `SyclDevicePool::InitDevicePool()` lines 132-163: replace platform-name substring filtering with `platform.get_backend() == ::sycl::backend::ext_oneapi_level_zero`, use `platform.get_devices(::sycl::info::device_type::gpu)`, and also verify each device backend.
  - `SyclDevicePool::GetDeviceCount()`, `GetDeviceOrdinal()`, and `GetDevice()` lines 189-210: keep ordinal-as-vector-index semantics, but document that the vector is the visible Level Zero SYCL GPU root-device list.
  - `SyclGetTimerProperties()` lines 387-420: no direct policy change, but this code shows why non-Level Zero devices must never enter the pool.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime.h`
  - Lines 43-73: update comments from "all Level-Zero backend GPUs" to the precise visible-device policy, including environment filtering and tile exposure.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_platform.cc`
  - `VisibleDeviceCount()` and `ExecutorForDevice()` lines 43-61: no algorithm change, but tests should assert these are driven by the same visible ordinal set.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_executor.cc`
  - `SyclExecutor::Init()` lines 477-481: consumes the ordinal-selected device and context. No direct change, but this is the boundary where wrong ordinal identity becomes executor identity.
  - `CreateDeviceDescription()` lines 839-842 delegates to `CreateOneApiDeviceDescription()` and should continue using the same ordinal.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_device_description.cc`
  - `CreateOneApiDeviceDescription()` lines 155-230 already obtains Level Zero native device and driver handles and sets PCI bus ID. Add optional diagnostic fields/logging here if needed.
- Tests:
  - `~/github/openxla/xla/xla/stream_executor/sycl/sycl_gpu_runtime_test.cc`
  - `~/github/openxla/xla/xla/stream_executor/sycl/sycl_platform_test.cc`

## Evidence

### XLA source

- `SyclDevicePool::InitDevicePool()` enumerates `sycl::platform::get_platforms()`, reads `info::platform::name`, selects platforms whose name contains `"Level-Zero"`, and pushes `device.is_gpu()` devices from `platform.get_devices()` in returned order (`sycl_gpu_runtime.cc:132-163`). This is name-based, not backend-based.
- Ordinals are exactly `device_pool_` vector indices. `GetDeviceCount()` returns `device_pool_.size()`, `GetDeviceOrdinal()` uses `std::find`, and `GetDevice()` indexes the vector (`sycl_gpu_runtime.cc:189-210`).
- XLA platform visibility and executor caching depend directly on that count and ordinal: `VisibleDeviceCount()` returns `SyclDevicePool::GetDeviceCount()`, and `ExecutorForDevice()` caches by ordinal (`sycl_platform.cc:43-61`).
- `SyclExecutor::Init()` binds an executor to `SyclDevicePool::GetDevice(device_ordinal())` and `SyclContext::Create(device_ordinal())` (`sycl_executor.cc:477-481`).
- `SyclContext::Create()` retrieves the per-ordinal SYCL context from `SyclDevicePool::GetDeviceContext()` (`sycl_context.cc:22-26`). `GetDeviceContext()` creates a one-device `sycl::context` for the ordinal (`sycl_gpu_runtime.cc:166-187`).
- `CreateDeviceDescription()` delegates to `CreateOneApiDeviceDescription()` (`sycl_executor.cc:839-842`). That function obtains `ze_device_handle_t` and `ze_driver_handle_t` with `get_native<backend::ext_oneapi_level_zero>`, queries Level Zero properties, and sets PCI bus ID from `zeDevicePciGetPropertiesExt` (`sycl_device_description.cc:141-148`, `155-230`).
- Other runtime paths also assume Level Zero native interop after ordinal lookup: module loading uses Level Zero device/context handles (`sycl_executor.cc:123-147`), peer access uses `zeDeviceCanAccessPeer` (`sycl_executor.cc:277-292`), and timer properties use `zeDeviceGetProperties` (`sycl_gpu_runtime.cc:387-420`).
- Existing tests cover count, ordinal roundtrip, per-ordinal contexts, and platform executor caching, but not backend enum filtering or root/tile policy (`sycl_gpu_runtime_test.cc:89-147`, `sycl_platform_test.cc:29-47`).

### SYCL 2020 spec

- Device discovery returns all devices from all platforms exposed by all supported SYCL backends (`sycl-2020.html:4348-4349`).
- Device selectors query root devices from all backends; tied selection can depend on enumeration order outside SYCL runtime control (`sycl-2020.html:6404-6415`).
- A platform is associated with a single backend, and `platform::get_backend()` returns it (`sycl-2020.html:6745-6754`, `6851-6860`).
- `platform::get_platforms()` returns all available platforms from all backends (`sycl-2020.html:6967-6978`).
- `platform::get_devices()` returns root devices associated with that platform for the requested type (`sycl-2020.html:6938-6962`).
- `device::get_backend()` returns the backend associated with a device (`sycl-2020.html:7572-7582`).
- `device::get_devices()` returns root devices from all backends for the requested type (`sycl-2020.html:7850-7874`).
- `device::create_sub_devices()` exists for partitioning, including `partition_by_affinity_domain` and `next_partitionable` (`sycl-2020.html:7704-7847`), but XLA does not call it during discovery.
- Backend interop `get_native` must throw `errc::backend_mismatch` when the SYCL object backend does not match the target backend (`sycl-2020.html:5325-5366`). This supports checking the backend before calling Level Zero native interop.

### oneAPI local docs

- `oneapi/extensions/supported/sycl_ext_oneapi_backend_level_zero.md` adds `backend::ext_oneapi_level_zero` and says the serving backend for a SYCL platform can be queried with `platform::get_backend()` (lines 31-61).
- The Level Zero backend interop spec maps `platform` to `ze_driver_handle_t`, `device` to `ze_device_handle_t`, and `context` to `ze_context_handle_t` (lines 72-114), and `get_native` is the API to obtain those handles (lines 297-307).
- The same Level Zero backend spec says `make_device` does not create a new device; it returns a copy of a device from the fixed Level Zero device or subdevice enumerations (lines 342-370).
- `oneapi/EnvironmentVariables.md` says `ONEAPI_DEVICE_SELECTOR` limits available devices and can expose GPU subdevices or sub-subdevices as individual devices (lines 36-39). Its dot syntax exposes subdevices as SYCL root devices (`<num>.<num>`, `<num>.*`, `*.*`) (lines 54-58), and `level_zero:*,*.*` exposes both devices/cards and subdevices as SYCL root devices (lines 73-87).
- The environment docs warn that Level Zero `ZE_FLAT_DEVICE_HIERARCHY` affects how root devices map to tiles or CCSs (lines 92-104). The linked Level Zero environment-variable document is not present locally.
- `oneapi/MultiTileCardWithLevelZero.md` says Intel GPUs are represented as SYCL GPU root devices, multiple GPUs appear as multiple root devices, and tile-capable roots can be partitioned into subdevices corresponding to physical tiles (lines 3-58). It says repeated `create_sub_devices` calls return the same subdevices in persistent order (line 54) and that `ZE_AFFINITY_MASK` can control subdevice exposure (line 55), but this local file does not define exact `ZE_AFFINITY_MASK` syntax.
- The same multi-tile guide distinguishes root-device implicit scaling from explicit tile submission: a queue on a root device may be distributed across tiles by the driver, while queues on tile subdevices target those tiles (lines 103-149).
- `oneapi/extensions/experimental/sycl_ext_oneapi_composite_device.asciidoc` is experimental and Level Zero-specific. It says current DPC++ can expose each PVC tile as a separate root device by default, and composite devices represent cards (`lines 44-87`). Because this conflicts with older root-then-partition wording, actual runtime behavior must be probed for the target driver.
- `oneapi/extensions/supported/sycl_ext_oneapi_platform_device_index.asciidoc` provides `device::ext_oneapi_index_within_platform()` and guarantees Level Zero index compatibility with `zeDeviceGet` only when `ONEAPI_DEVICE_SELECTOR` is not set (lines 160-180). This is useful for diagnostics and maybe unfiltered sorting, but it is not a universal ordinal contract.
- `oneapi/extensions/supported/sycl_ext_intel_device_info.asciidoc` exposes optional PCI address, UUID, and tile-count queries via aspects (`pci_address` lines 82-130, `uuid` lines 460-507, `xe_stack_count` lines 1115-1155).

## Findings

1. The current XLA ordinal is a SYCL enumeration index, not a documented physical-device identifier. It is the index in `SyclDevicePool::device_pool_` after XLA filters platforms by name and devices by `is_gpu()`.
2. The platform-name substring check is too weak for code that later unconditionally uses Level Zero native interop. SYCL and oneAPI both expose backend enum checks, and `get_native<ext_oneapi_level_zero>` is only valid for Level Zero objects.
3. XLA does not create tile subdevices internally today. Therefore tiles are hidden or exposed entirely according to the SYCL runtime's visible root-device list.
4. Under plain environment, XLA sees whatever DPC++ exposes through the Level Zero platform's GPU root devices. Local docs are version-sensitive: one guide describes card roots with tile subdevices, while the experimental composite-device doc says current DPC++ can expose PVC tiles as root devices by default.
5. Under `ONEAPI_DEVICE_SELECTOR`, XLA should see the post-filtered root-device list. Selectors such as `level_zero:*.*` can expose tiles as root devices, so XLA ordinals may become tile ordinals without XLA calling `create_sub_devices`.
6. Under `ZE_AFFINITY_MASK`, local docs only establish that it can control Level Zero subdevice exposure. They do not define exact ordering or syntax. Treat this as a runtime-controlled visibility input that must preserve user intent, not as something XLA can reinterpret without measurement.
7. PCI/BDF is available today in XLA through Level Zero `zeDevicePciGetPropertiesExt`, and oneAPI also exposes optional SYCL-level PCI/UUID/tile-count queries. These are good diagnostics, but BDF alone may not distinguish multiple tiles on the same card.
8. `ext_oneapi_index_within_platform()` is promising for diagnostics and unfiltered stable ordering, but the local spec only guarantees Level Zero backend-index compatibility when `ONEAPI_DEVICE_SELECTOR` is not set.

## Proposed patch plan

1. Backend checks:
   - In `SyclDevicePool::InitDevicePool()`, replace `platform_name.find("Level-Zero")` with `platform.get_backend() == ::sycl::backend::ext_oneapi_level_zero`.
   - Retrieve `platform.get_devices(::sycl::info::device_type::gpu)` instead of all devices plus `device.is_gpu()`, while keeping `device.is_gpu()` or `device.has(aspect::gpu)` as a defensive check.
   - Verify `device.get_backend() == ::sycl::backend::ext_oneapi_level_zero` before pushing into `device_pool_`.
   - Keep a clear error if no Level Zero GPU is visible, mentioning `ONEAPI_DEVICE_SELECTOR` and Level Zero runtime visibility.
2. Ordinal policy:
   - Document in `sycl_gpu_runtime.h` that XLA ordinal `N` means `device_pool_[N]`, where `device_pool_` is the visible Level Zero SYCL GPU root-device list after process-level oneAPI/SYCL filtering.
   - State explicitly that "root device" is the SYCL runtime concept after environment filtering. It may represent a card, a tile exposed as a root device, or another Level Zero-visible root according to DPC++ runtime settings.
3. Stable ordering:
   - Preserve SYCL returned order by default. This respects `ONEAPI_DEVICE_SELECTOR` order and avoids changing user-selected topologies.
   - Add optional diagnostic logging per ordinal: platform backend/name, device name, native `ze_device_handle_t`, PCI/BDF, `ext_oneapi_index_within_platform()` when available, partition info, UUID when available, and tile-count query when available.
   - If maintainers require unfiltered deterministic ordering across platforms, add an opt-in stable-sort mode only when no explicit visibility/order environment variables are set. Candidate key: Level Zero driver/platform identity, PCI/BDF, `ext_oneapi_index_within_platform()`, and native handle as last-resort diagnostic only. Do not sort mixed parent/root and child/tile lists unless the policy explicitly defines that topology.
4. Tile exposure:
   - Default: do not call `create_sub_devices()` in XLA discovery. Surface tiles as XLA ordinals only when they are in the SYCL root-device list supplied by DPC++ (`ONEAPI_DEVICE_SELECTOR`, `ZE_FLAT_DEVICE_HIERARCHY`, driver default, or `ZE_AFFINITY_MASK`).
   - Future opt-in: add an XLA flag such as `xla_sycl_expose_tiles=true` that partitions each selected root with `partition_by_affinity_domain(next_partitionable)`. This should be mutually exclusive with selector strings that already expose subdevices, or it should detect and avoid duplicating the same hardware.
   - Do not mix parent card roots and tile roots by default. If a user requests `ONEAPI_DEVICE_SELECTOR=level_zero:*,*.*`, preserve the runtime list but warn/log that parent and child devices may alias hardware.
5. Regression tests:
   - Add tests that every `SyclDevicePool::GetDevice(i)` has Level Zero backend and GPU aspect.
   - Add process-env integration tests for `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `level_zero:0` when available, and `level_zero:*.*` on partitionable GPUs.
   - Add a helper/probe binary used by tests and manual debugging to print the full identity tuple for every ordinal.

## Test/benchmark coverage

- Unit/regression:
  - `SyclGpuRuntimeTest.GetDeviceCount`: extend or add a sibling test that compares XLA count to the count from Level Zero backend GPU platforms only.
  - `SyclGpuRuntimeTest.GetDeviceOrdinal`: keep the current roundtrip test and add assertions that `GetDeviceOrdinal(GetDevice(i)) == i` for all visible ordinals.
  - New `SyclGpuRuntimeTest.DevicesAreLevelZeroGpus`: assert `device.get_backend() == backend::ext_oneapi_level_zero` and `device.is_gpu()` for every ordinal.
  - New platform test: `VisibleDeviceCount()` equals `SyclDevicePool::GetDeviceCount()` and every executor ordinal is valid under the same pool.
- Environment integration:
  - Run the same small discovery binary under:
    - no selector;
    - `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`;
    - `ONEAPI_DEVICE_SELECTOR=level_zero:0`;
    - `ONEAPI_DEVICE_SELECTOR=level_zero:*.*` on a partitionable GPU;
    - `ONEAPI_DEVICE_SELECTOR=level_zero:*,*.*` to expose duplicate parent/child topology intentionally.
  - Add `ZE_AFFINITY_MASK` cases once exact local target syntax is confirmed from installed Level Zero docs or hardware-owner guidance.
- Probe output should include:
  - XLA ordinal;
  - platform index, backend, name, and native driver handle;
  - device name, backend, `is_gpu`, SYCL platform index if available;
  - native `ze_device_handle_t`;
  - PCI/BDF from Level Zero and/or `ext_intel_pci_address`;
  - UUID if `aspect::ext_intel_device_info_uuid`;
  - `partition_type_property`, `partition_type_affinity_domain`, `partition_max_sub_devices`, and subdevice count from `create_sub_devices(partition_by_affinity_domain(next_partitionable))`;
  - stack/tile count if `aspect::ext_intel_xe_stack_count`.
- Benchmark:
  - Discovery cost is not expected to matter, but record startup time before/after backend checks and any optional identity-query logging.
  - If stable sorting is added, benchmark startup on systems with many GPUs/tiles and verify selector order is unchanged when selectors are active.

## Rollout risk

- Backend enum filtering is low risk and correctness-positive. The current implementation already requires Level Zero native interop later, so accepting non-Level Zero devices would be a latent bug.
- Changing ordinal ordering is high risk. PJRT clients, caches, logs, and multi-device tests may assume stable ordinal identity within a run. Preserve SYCL/selector order unless an explicit XLA flag changes it.
- Automatic tile exposure is high risk. It can change visible device count, memory visibility, peer-access behavior, collective topology, and user expectations. Keep it out of default discovery.
- Parent-plus-child exposure can alias the same hardware. If allowed through `ONEAPI_DEVICE_SELECTOR=level_zero:*,*.*`, XLA should preserve user intent but surface diagnostics because scheduling both parent and tile ordinals independently may oversubscribe or duplicate hardware.
- Multi-platform Level Zero behavior, especially on Windows, needs care. The multi-tile doc says multiple Windows GPUs may appear as root devices of multiple Level Zero platforms, while Linux may put multiple roots in one platform.

## Evidence gaps

- No Intel multi-GPU or multi-tile runtime measurement was performed here. `sycl-ls` and `icpx` are not available in this environment, so all runtime behavior above is inferred from local docs and XLA source.
- The local oneAPI route-map files are present under `~/sycl/oneapi`, including `MultiTileCardWithLevelZero.md`, `EnvironmentVariables.md`, filter selector, and Level Zero backend extension docs. The external Level Zero environment-variable document linked by `EnvironmentVariables.md` is not present locally.
- `ZE_AFFINITY_MASK` is mentioned locally as controlling subdevice exposure, but exact syntax, interaction with `ZE_FLAT_DEVICE_HIERARCHY`, and ordering effects are not specified in the local docs consulted.
- Local oneAPI docs are version-sensitive on root-vs-tile defaults. `MultiTileCardWithLevelZero.md` describes card roots partitioned into tile subdevices, while the experimental composite-device extension says current DPC++ can expose PVC tiles as root devices by default. The proposed probe must resolve behavior on the target driver.
- PCI/BDF support exists in XLA through Level Zero and in oneAPI through an optional SYCL extension, but BDF may identify the card rather than individual tiles. UUID, native handle, and partition metadata should be included in probes before using BDF for ordering.
