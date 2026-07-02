# Local device, context, stream, and lifetime identity

## Scope

This note reviews XLA's local oneCCL/SYCL identity path for one-process,
multi-GPU Intel execution: SYCL device enumeration, Level Zero identity,
oneCCL device/context/stream creation, rank-to-device mapping, stream cache
lifetime, communicator teardown, and process shutdown. It focuses on
correctness and reliability for PVC/BMG-style multi-GPU hosts, multi-tile
devices exposed as SYCL devices, visibility masks, and integrated plus discrete
Intel GPU mixes.

`research/oneccl/intel-map.md` was not present. Source review started from
`research/oneccl/plan.md` and `~/github/uxlfoundation/oneCCL/map.md`.

## XLA needs

XLA needs the oneCCL rank for each local participant to use the same
`sycl::device`, `sycl::context`, and `sycl::queue` that the corresponding
`stream_executor::sycl::SyclExecutor` and `SyclStream` use. For visible-device
masks and subdevice/tile exposure, XLA needs rank order to be the visible SYCL
device order, not an inferred physical-card order. XLA also needs the
`ccl::stream` object used for a collective to remain tied to the live XLA
stream queue and to stop being used when that stream is destroyed.

The local Intel defaults must be applied only for device sets where XLA intends
to use oneCCL's local Intel GPU paths, and diagnostics must make ambiguous
sets visible: mixed integrated/discrete devices, mixed PVC/BMG families, and
root-device plus subdevice exposure in the same process.

## oneCCL behavior

oneCCL's C++ device path accepts caller-supplied native SYCL objects:
`ccl::create_device(sycl_device)`, `ccl::create_context(sycl_context)`, and
`ccl::create_stream(sycl_queue)`. The public docs describe the same model: a
device-memory example creates oneCCL device/context from SYCL device/context
and then creates a oneCCL stream from a user-supplied `sycl::queue`
(`deps/libccl/doc/rst/source/programming-model/device-communication.rst`,
published at `https://uxlfoundation.github.io/oneCCL/api/concepts/stream.html`
and `https://uxlfoundation.github.io/oneCCL/api/concepts/communicator.html`).

The legacy implementation stores native SYCL objects by value in oneCCL wrapper
objects. `stream::create_stream` routes through `stream_selector::create`; the
`ccl_stream` constructor copies the queue, creates worker queues from the same
queue context/device, and records Level Zero device/context handles for Level
Zero streams (`deps/libccl/src/stream_impl.hpp`,
`deps/libccl/src/common/stream/stream_selector_impl.hpp`,
`deps/libccl/src/common/stream/stream.cpp`). Communicator construction stores
the supplied `ccl::device` and `ccl::context`, then uses them for topology,
NUMA grouping, IPC exchange, comm environment selection, and SYCL kernel
initialization (`deps/libccl/src/comm/comm.cpp`, `comm.hpp`). Split
communicators inherit the parent device/context.

The v2 legacy plugin has a separate C API path: `onecclSetDevice(index)` picks
`l0_platform.get_devices()[index]` and a platform default context, and
`onecclCommInitRank` then uses those thread-local selections
(`plugins/legacy/ccl_legacy.cpp`). XLA's communicator path does not use that
API; it uses the legacy C++ API directly. The plugin's `get_stream` also has a
thread-local `std::unordered_map<sycl::queue, ccl::stream>`, but that is not
the cache used by XLA's C++ collective launches.

SYCL and Level Zero make visibility and subdevice identity environment
dependent. `ONEAPI_DEVICE_SELECTOR` can expose subdevices as SYCL root devices,
and Intel's docs note that `ZE_FLAT_DEVICE_HIERARCHY` affects exact tile/CCS
mapping. Level Zero documents root devices and subdevices as implementation
specific, with subdevice handles queryable through `zeDeviceGetSubDevices`,
UUID/subdevice properties available through `zeDeviceGetProperties`, and
subdevice allocations scoped to the subdevice hierarchy.

## Must fix

- Replace `ToOnecclStream`'s process-lifetime
  `absl::flat_hash_map<sycl::queue*, shared_ptr<ccl::stream>>` with a lifetime
  tied cache. The safest shape is a `Stream::Resource` attached to the
  `se::Stream`, or an equivalent `SyclStream`-owned generation/erase hook.
  The current raw pointer key can survive `SyclStream` destruction, grow
  without bound, and return a `ccl::stream` wrapping an old copied SYCL queue if
  a new queue is allocated at the same address. That violates XLA's stream
  ordering contract and can launch oneCCL work on the wrong queue/device.

- Add a launch-time identity check before creating or reusing a oneCCL stream:
  the SYCL queue's `get_device()` and `get_context()` must match the
  communicator's `SyclExecutor::GetDevice()` and `GetContext()`, and the queue
  must be in-order. `VerifyStreamExecutor` only checks the parent
  `StreamExecutor`; it does not protect against stale `ccl::stream` cache
  entries or a future bad platform-specific handle.

- Detect and reject ambiguous local cliques that contain both a Level Zero root
  device and one of its subdevices, or duplicate/overlapping subdevice
  identities. This can happen with selectors such as root plus tile exposure.
  Until XLA has explicit support for that mode, rank mapping is ambiguous and
  oneCCL topology/IPC setup can observe a physical layout that does not match
  XLA's logical rank set.

- For supported subdevice/tile-only masks, make rank-to-device identity
  explicit: collect and compare Level Zero UUID, root/subdevice flag, and
  subdevice id for every local rank before communicator creation. XLA should
  fail closed if two ranks map to the same Level Zero identity or if a device
  cannot report enough identity to prove uniqueness.

## Should fix

- Tighten the Intel local default predicates. `VisibleDevicesAreMultiGpuIntelDevices`
  and `StreamExecutorsAreMultiGpuIntelDevices` currently check only "at least
  two visible SYCL Level Zero GPUs, all Intel vendor id". That includes
  integrated plus discrete mixes, mixed Intel product families, and tile
  exposure. Prefer actual clique devices over the global visible-device
  fallback, and log/guard product family, integrated flag, PCI bus id, UUID,
  root/subdevice state, and subdevice id when defaults are applied.

- Keep the conservative defaults for unclassified all-Intel multi-GPU cliques
  initially, but warn when the clique is not a tested homogeneous class
  (for example PVC-only cards, BMG-only cards, or tile-only masks). Narrowing
  too aggressively risks re-exposing the oneCCL Intel GPU path issues that the
  defaults are intended to avoid.

- Add structured VLOG diagnostics for every oneCCL communicator:
  XLA device ordinal, global rank, SYCL backend, device name, PCI bus id,
  Level Zero UUID, subdevice id, integrated/discrete flag, native context
  handle fingerprint, queue pointer/generation, and oneCCL runtime version.

- Make communicator teardown ordering explicit. oneCCL communicators should be
  destroyed before their stream resources, and stream resources before SYCL
  queues are removed from `SyclStreamPool`. A `Stream::Resource` cache gives
  that lifecycle naturally; a global leaked cache does not.

- Audit `OnecclCollectives::Allocate`/`Deallocate`. They use the v2 C API
  `onecclMemAlloc`/`onecclMemFree`, while the legacy plugin allocates from a
  thread-local device/default stream set by `onecclSetDevice`. XLA's main
  communicator path does not call `onecclSetDevice`; leave these APIs unused,
  route them through the executor's SYCL allocator, or make device selection
  explicit before enabling them.

## Could fix

- Use oneCCL's `create_communicators` local rank-to-device mapping overload for
  all-local communicator creation if it proves clearer than parallel
  per-rank `create_communicatorExt` calls. This is not required if XLA keeps
  passing exact `(size, rank, device, context, kvs)` and validates identity.

- Extend `Stream::PlatformSpecificHandle` or add a SYCL-specific accessor that
  exposes a queue generation or `shared_ptr`-backed identity, reducing the need
  for collector-specific dynamic casts.

- Add user docs with supported oneCCL/SYCL visibility patterns:
  root devices only, tile/subdevice devices only, no mixed root+subdevice
  exposure, and tested PVC/BMG/integrated-discrete combinations.

- Track oneCCL version behavior around stream construction and topology
  initialization. XLA pins oneCCL v1 to commit
  `4ceafd15c03ce46f11eeaf91781a92afebd3cecf` and v2 to `master-v2`; the
  published 2021.17 docs are useful but the source routes are the authority.

## Affected files/call sites

- `xla/backends/gpu/collectives/oneccl_communicator.cc`:
  `ToOnecclStream`, `VerifyStreamExecutor`, `LaunchOnStream`,
  `OnecclCommunicator::~OnecclCommunicator`, `Abort`, `Split`.
- `xla/backends/gpu/collectives/oneccl_collectives.cc`:
  `IsIntelGpuDevice`, `VisibleDevicesAreMultiGpuIntelDevices`,
  `StreamExecutorsAreMultiGpuIntelDevices`, `CreateOnecclDeviceContext`,
  `CreateOnecclCommunicator`, `CreateOnecclCommunicatorForRank`,
  `PrepareOnecclCommunicatorCreation`, `OnecclCollectives::Allocate`.
- `xla/stream_executor/sycl/sycl_gpu_runtime.{h,cc}`:
  `SyclDevicePool`, `GetDeviceContext`, `SyclStreamPool`,
  `GetOrCreateStream`, `DestroyStream`, `Reset`.
- `xla/stream_executor/sycl/sycl_stream.{h,cc}`:
  `platform_specific_handle`, `stream_handle`, `SyclStream::~SyclStream`.
- `xla/stream_executor/sycl/sycl_executor.cc`:
  `Init`, `CreateStream`, `DeallocateStream`, `CreateMemoryAllocator`,
  `EnablePeerAccessTo`.
- `xla/stream_executor/sycl/sycl_device_description.cc`:
  `CreateOneApiDeviceDescription`.
- `xla/backends/gpu/collectives/gpu_cliques.cc`,
  `gpu_clique_key.{h,cc}`, `xla/backends/gpu/runtime/collective_thunk.cc`:
  clique rank mapping and communicator creation rendezvous.
- oneCCL source routes:
  `deps/libccl/src/ccl_api_functions.cpp`,
  `deps/libccl/src/ccl_cpp_environment.cpp`,
  `deps/libccl/src/{device_impl.hpp,context_impl.hpp,stream_impl.hpp}`,
  `deps/libccl/src/common/{device,context,stream}`,
  `deps/libccl/src/comm/{comm.cpp,comm.hpp}`,
  `deps/libccl/src/native_device_api/sycl/export.cpp`,
  `plugins/legacy/ccl_legacy.cpp`, `include/oneapi/ccl.h`.
- Dependency routes:
  `third_party/oneccl/workspace.bzl`,
  `third_party/oneccl/oneccl_v1.BUILD`,
  `third_party/oneccl/oneccl_v2.BUILD`, `MODULE.bazel`.

## Evidence to cite

- XLA creates oneCCL device/context directly from `SyclExecutor`:
  `xla/backends/gpu/collectives/oneccl_collectives.cc:537`
  `CreateOnecclDeviceContext`; communicator creation calls
  `ccl::create_communicatorExt` at `:559`.
- XLA's oneCCL stream cache is raw-pointer keyed and process-lifetime:
  `xla/backends/gpu/collectives/oneccl_communicator.cc:126`
  `ToOnecclStream`.
- XLA only checks stream parent executor before creating/reusing oneCCL stream:
  `xla/backends/gpu/collectives/oneccl_communicator.cc:112`
  `VerifyStreamExecutor`, `:684` `LaunchOnStream`.
- `SyclStream` exposes only `stream_handle_.get()` as the generic platform
  handle and destroys the queue through `SyclStreamPool::DestroyStream`:
  `xla/stream_executor/sycl/sycl_stream.h:87`,
  `xla/stream_executor/sycl/sycl_stream.cc:336`, `:362`.
- `SyclDevicePool` enumerates all Level Zero GPU devices and creates
  per-device SYCL contexts for process lifetime:
  `xla/stream_executor/sycl/sycl_gpu_runtime.cc:132`, `:166`.
- `SyclStreamPool` creates in-order profiling queues and has an XLA-side max
  stream bound of 4096 per device:
  `xla/stream_executor/sycl/sycl_gpu_runtime.h:84`,
  `xla/stream_executor/sycl/sycl_gpu_runtime.cc:227`, `:284`.
- XLA rank IDs are the clique device order; `GpuCliqueKey` stores ordered
  `GlobalDeviceId`s and `gpu_cliques.cc` sorts `DeviceRank`s by rank before
  communicator creation.
- oneCCL C++ docs/source show user-supplied SYCL device/context/queue wrapping:
  `deps/libccl/doc/rst/source/programming-model/device-communication.rst`,
  `deps/libccl/src/stream_impl.hpp`,
  `deps/libccl/src/common/stream/stream.cpp`.
- oneCCL `ccl_comm::init` uses device/context for topology, NUMA, IPC, and
  SYCL kernel setup: `deps/libccl/src/comm/comm.cpp:186`, `:212`, `:235`,
  `:409`, `:568`, `:581`.
- oneCCL scaleout device buffers assert repeated queue/device identity:
  `deps/libccl/src/comm/comm.cpp:904`.
- oneCCL v2 legacy plugin `onecclSetDevice` selects
  `l0_platform.get_devices()[index]` and a default context:
  `plugins/legacy/ccl_legacy.cpp:489`.
- oneCCL C API docs require device selection before communicator creation:
  `include/oneapi/ccl.h:163`.
- Official docs:
  `https://intel.github.io/llvm/EnvironmentVariables.html` for
  `ONEAPI_DEVICE_SELECTOR`, subdevice selector syntax, and
  `ZE_FLAT_DEVICE_HIERARCHY` interaction;
  `https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html`
  for Level Zero root/subdevice identity, context isolation, and subdevice
  allocation scope;
  `https://github.khronos.org/SYCL_Reference/iface/queue.html` for queue
  context/device info and in-order semantics;
  `https://github.khronos.org/SYCL_Reference/iface/context.html` for context
  device membership.

## Test coverage plan

- Add unit coverage for a `OnecclStreamResource` or equivalent cache:
  repeated `ToOnecclStream` calls on one live stream reuse the resource, stream
  destruction releases it, and a recreated stream at a reused queue address
  does not observe the old oneCCL stream.
- Add identity validation tests with fake or injectable SYCL identity metadata:
  duplicate Level Zero UUID/subdevice id fails, root plus subdevice overlap
  fails, tile-only unique subdevices pass, and queue context/device mismatch
  fails before oneCCL launch.
- Add SYCL integration coverage, gated on hardware availability, for
  `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `level_zero:0,1`,
  `level_zero:0.*`, `level_zero:*,*.*`, and representative
  `ZE_AFFINITY_MASK` tile masks. Verify rank logs, default application, and
  fail-closed behavior for mixed root+subdevice exposure.
- Add mixed-device coverage for integrated plus discrete Intel GPUs and mixed
  PVC/BMG families. Verify XLA either applies conservative defaults with a
  warning or refuses unsupported local cliques according to the final policy.
- Add stream create/destroy stress coverage around communicator use: create
  streams, launch oneCCL collective, destroy streams, recreate streams, launch
  again, then verify launch happens on the current queue identity.
- Add communicator teardown coverage for normal destruction, `Abort`, split
  communicator destruction, and process shutdown ordering. Ensure no oneCCL
  stream resource outlives the XLA stream that created it.
- Add registered/symmetric memory coverage under visibility masks and
  subdevices, because oneCCL communicator registration relies on the same
  device/context identity even though allocation itself comes from XLA's SYCL
  allocator.

## Rollout risk

Changing the stream cache can affect collective launch overhead if XLA creates
a fresh `ccl::stream` per call. A per-`se::Stream` resource cache should keep
the current reuse behavior while removing stale pointer risk. Gate the first
change with a debug flag if needed, but the raw pointer cache should not remain
the default.

Stricter subdevice validation can reject existing user environments, especially
selectors that expose both cards and tiles. Start with clear diagnostics and a
documented escape hatch only for known-good internal testing; do not silently
run ambiguous root+subdevice cliques.

Narrowing Intel default predicates may improve mixed-device behavior but can
re-enable oneCCL paths that the current defaults were avoiding. Keep the
conservative behavior for unknown all-Intel multi-GPU cliques until the
visibility-mask and product-family tests are in place.

oneCCL version behavior is a moving target: XLA's pinned v1 commit and v2
master-v2 snapshot may not match the published docs. Treat source routes as
primary evidence and add version-gated warnings where behavior depends on
oneCCL internals rather than API guarantees.
