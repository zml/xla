# XLA oneCCL single-machine multi-GPU research plan

Audience: XLA/SYCL runtime work for multiple Intel GPUs in one host. XLA source
is assumed at `~/github/openxla/xla`. oneCCL source context is
assumed at `~/github/uxlfoundation/oneCCL`; start with its
`map.md` before chasing oneCCL internals.

Goal: produce XLA-facing research documents, not experiments. Each topic should
answer: based on oneCCL implementation details and XLA's actual use, what must
be fixed for correctness/reliability, what should be improved for performance
or maintainability, and what could be improved when capacity allows.

Non-goals: running benchmarks, writing reproducers, designing multi-node
transport, defining MPI/rendezvous policy, full oneCCL tuning, general SYCL
portability, and broad compiler/codegen audit.

## Action rubric

Every topic document must classify recommendations as one of: Must / Should /
Could fix.

- Must fix: XLA can hang, corrupt data, violate StreamExecutor/SYCL ordering
  contracts, leak or deadlock on expected local failures, or rely on a oneCCL
  behavior that the implementation does not guarantee.
- Should fix: XLA is correct but leaves material overlap/performance on the
  table, has fragile env or lifetime behavior, has poor diagnostics, or depends
  on oneCCL details that should be guarded/documented.
- Could fix: optional cleanup, docs, future version tracking, broader device
  coverage, or changes that should wait for oneCCL/oneAPI maturity.

Each topic document should contain:

- Scope: the XLA use pattern and the oneCCL implementation area being reviewed.
- XLA needs: the contract XLA requires from oneCCL and SYCL/Level Zero.
- oneCCL behavior: what the implementation appears to guarantee, not guarantee,
  or leave version/env-dependent.
- Must / Should / Could changes: concrete XLA-side changes, or explicit
  no-change decisions.
- Affected files/call sites: exact XLA files/functions and oneCCL source routes.
- Evidence to cite: code references, oneCCL docs/spec references, comments,
  build flags, env variable definitions, and version pins.
- Test coverage plan: tests XLA should add if it accepts the recommendation.
- Rollout risk: user env override risk, oneCCL version risk, performance risk,
  and fallback/flag plan.

## Source routing

Use static source review and documentation only. Do not run tests, examples,
benchmarks, profilers, or ad hoc experiments for this plan.

Read in this order for each topic:

1. XLA map/context: `intel-map.md`, this file, and the relevant XLA anchors.
2. XLA oneCCL integration:
   - `xla/backends/gpu/collectives/oneccl_collectives.{h,cc}`
   - `xla/backends/gpu/collectives/oneccl_communicator.{h,cc}`
   - `xla/backends/gpu/collectives/oneccl_registered_memory.{h,cc}`
   - `xla/backends/gpu/collectives/oneccl_symmetric_memory.{h,cc}`
   - `xla/backends/gpu/runtime/collective_permute_thunk.cc`
   - GPU collective thunk call sites in `xla/backends/gpu/runtime`
   - `xla/service/gpu/gpu_executable.cc` for barrier-after-executable
3. XLA SYCL runtime:
   - `xla/stream_executor/sycl/sycl_stream.{h,cc}`
   - `xla/stream_executor/sycl/sycl_gpu_runtime.{h,cc}`
   - `xla/stream_executor/sycl/sycl_event.{h,cc}`
   - `xla/stream_executor/sycl/sycl_executor.{h,cc}`
4. oneCCL source map: `~/github/uxlfoundation/oneCCL/map.md`.
5. oneCCL implementation routes from the map:
   - v2 API/plugin adapter: `src/api.cpp`, `plugins/legacy/ccl_legacy.cpp`
   - legacy public C++ API: `deps/libccl/include/oneapi/ccl`
   - event/stream/device/context: `deps/libccl/src/common`
   - collectives: `deps/libccl/src/coll`,
     `deps/libccl/src/coll/algorithms/*/sycl`
   - scheduler/Level Zero: `deps/libccl/src/sched`,
     `deps/libccl/src/sched/entry/ze`, `deps/libccl/src/sched/ze`
   - transport/env: `deps/libccl/src/atl`,
     `deps/libccl/src/common/env`
   - memory/window: `deps/libccl/src/comm`,
     `deps/libccl/src/common`, `deps/libccl/src/umf`
   - native SYCL export: `deps/libccl/src/native_device_api/sycl`
6. Pinned XLA dependency routes:
   - `third_party/oneccl/workspace.bzl`
   - `third_party/oneccl/oneccl_v1.BUILD`
   - `third_party/oneccl/oneccl_v2.BUILD`
   - `workspace2.bzl`, `MODULE.bazel`
7. External references:
   - oneCCL C++ and C API docs for event, stream, group, split, point-to-point,
     memory registration, and window semantics.
   - oneCCL env reference for SYCL algorithms, thresholds, OFI/SHM, worker
     threads, topology, and debug variables.
   - SYCL 2020 queue/event/context/native-handle rules and oneAPI Level Zero
     interop extensions.
   - Level Zero command queue/list/event, memory visibility, copy engine, peer
     access, and event docs.
   - libfabric `shm` provider docs only for local OFI behavior.

## 1. Completion, queue ordering, barrier, and memory visibility

Merged topics: event completion contract, same-queue ordering, barrier ordering,
memory visibility, and cross-device event hazards.

XLA needs: when oneCCL is launched on an XLA `se::Stream`, XLA needs prior
kernels/copies on that stream to be visible to oneCCL, oneCCL writes to be
visible to later kernels, and `ccl::event::wait()` to mean device completion
where XLA currently blocks the host. For barriers, XLA needs the operation to
order device memory across ranks, not only synchronize host code.

XLA anchors: `WaitForOnecclEvent`, `LaunchOnecclAndWait`,
`OnecclCommunicator::LaunchOnStream`, `ToOnecclStream`, `Barrier`,
`SyclStream`, `SyclEvent`, `SyclStreamPool`,
`xla/backends/gpu/runtime/collective_permute_thunk.cc`.

oneCCL routes: event wrapper and request completion, `ccl::stream` construction
from SYCL queue, scheduler completion, SYCL interop event path, Level Zero
schedule entries, local collective algorithms, barrier implementation, and any
cross-device event waits.

Document output:

- Must/Should/Could changes for whether XLA can keep treating oneCCL event
  wait as full device completion.
- Must/Should/Could changes for whether XLA must add explicit SYCL queue/event
  dependencies around oneCCL calls.
- Must/Should/Could changes for whether XLA should avoid oneCCL paths that
  introduce cross-device SYCL event waits similar to the collective-permute
  special case.
- A test coverage plan for producer-kernel -> oneCCL -> consumer-kernel
  ordering, barrier-after-executable, same-stream and cross-stream dependency
  cases.

## 2. Host asynchrony, grouped collectives, and launch sequencing

Merged topics: host wait semantics/overlap, oneCCL group API mismatch, grouped
HLO collectives, and P2P launch ordering constraints.

XLA needs: XLA should know whether oneCCL work can be enqueued asynchronously
and represented by XLA futures/dependencies, or whether oneCCL requires the
current immediate host wait. For grouped collectives, XLA needs a safe grouped
event model or a justified no-change decision. For P2P, XLA needs launch
ordering that cannot deadlock for cycles, fanout, self edges, and mixed
send/recv groups.

XLA anchors: `OnecclCommunicator::Execute`,
`OnecclCommunicator::Execute<T>`, `OnecclCommunicator::GroupExecute`,
`OnecclCommunicator::GroupExecuteCounted`, `ValidateCreateCommunicatorConfig`,
`LaunchCollectivePermute`, `LaunchSend`, `LaunchRecv`, grouped collective thunk
call sites in `all_reduce_thunk.cc`, `all_gather_thunk.cc`,
`collective_broadcast_thunk.cc`, `collective_group_thunk.cc`,
`collective_permute_thunk.cc`, and `ragged_all_to_all_thunk.cc`.

oneCCL routes: event lifetime and ownership, group start/end implementation,
grouped event behavior, scheduler batching, P2P send/recv matching, communicator
ordering rules, cancellation behavior with pending requests, and v2 legacy
plugin forwarding.

Document output:

- Must/Should/Could changes for replacing already-completed futures with real
  async completion, or a no-change decision if oneCCL implementation cannot
  support it safely.
- Must/Should/Could changes for using or continuing to avoid oneCCL groups.
- Must/Should/Could changes for P2P launch ordering and validation in XLA.
- A test coverage plan for grouped HLO collectives, send/recv cycles, fanout,
  self edges, missing peer matches, cancellation, and async error propagation.

## 3. Local bootstrap, env defaults, and init timing

Merged topics: required local defaults, env initialization timing, user override
policy, OFI SHM local path, BF16/multi-GPU threshold defaults, and local build
knobs that affect runtime behavior.

XLA needs: XLA needs one-process N-GPU bootstrap to be deterministic without
requiring MPI/PMIx launch semantics. XLA also needs to know which oneCCL env
variables must be set before `ccl::init`, which may be set before communicator
creation, and which defaults are unsafe to partially override.

XLA anchors: `InitOnecclOnce`, `SetOnecclEnvDefaultGroupIfUnset`,
`SetOnecclSingleProcessBootstrapEnvDefaults`,
`SetOnecclIntelGpuCollectiveEnvDefaultsIfNeeded`,
`VisibleDevicesAreMultiGpuIntelDevices`, `third_party/oneccl/oneccl_v1.BUILD`.

oneCCL routes: env variable definitions and parsing, global init, process
manager selection, ATL OFI and SHM provider setup, algorithm selector
construction, SYCL allgather/allgatherv and allreduce threshold handling,
BF16/FP16 vector paths, and compile-time feature macros such as
`CCL_ENABLE_OFI_HMEM`, `CCL_ENABLE_SYCL_INTEROP_EVENT`,
`CCL_SYCL_ENABLE_ARCB`, vector BF16/FP16, and UMF.

Document output:

- Must/Should/Could changes for XLA's bootstrap defaults:
  `CCL_PROCESS_LAUNCHER=none`, `CCL_LOCAL_SIZE=1`, `CCL_LOCAL_RANK=0`,
  `CCL_ATL_TRANSPORT=ofi`, `CCL_ATL_SHM=1`, and `FI_PROVIDER=shm`.
- Must/Should/Could changes for default-group override policy when users set
  only one variable in a related group.
- Must/Should/Could changes for setting or warning about env variables after
  oneCCL has already initialized.
- Must/Should/Could changes for BF16/allgather/allreduce threshold defaults:
  `CCL_SYCL_ALLGATHERV_LL_THRESHOLD=1GiB` and
  `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=32MiB`.
- Must/Should/Could changes for documenting or changing local oneCCL build
  macros that XLA depends on.
- A test coverage plan for env grouping, late init, partial overrides, local
  bootstrap without external launchers, and threshold-sensitive BF16 paths.

## 4. Local device, context, stream, and lifetime identity

Merged topics: local device discovery, device/context identity, subdevice/tile
mapping, `ccl::stream` cache lifetime, queue pointer reuse, and shutdown order.

XLA needs: XLA needs oneCCL communicators to use the same SYCL devices,
contexts, queues, and rank mapping that StreamExecutor uses. XLA also needs
cached oneCCL streams to remain valid for the lifetime of the underlying
`sycl::queue`, without stale pointer reuse or unbounded growth becoming a
long-running process problem.

XLA anchors: `IsIntelGpuDevice`, `VisibleDevicesAreMultiGpuIntelDevices`,
`CreateOnecclDeviceContext`, `ToOnecclStream`, `SyclDevicePool`,
`SyclStreamPool`, `SyclExecutor`, `CreateOneApiDeviceDescription`.

oneCCL routes: `ccl::create_device`, `ccl::create_context`, `ccl::stream`
construction, native SYCL handle ownership, communicator creation, device
topology, and destructor/shutdown paths.

Document output:

- Must/Should/Could changes for when XLA applies Intel local multi-GPU defaults,
  especially PVC/BMG, multi-tile devices, visibility masks, and integrated plus
  discrete mixes.
- Must/Should/Could changes for asserting or documenting device/context
  identity between XLA and oneCCL.
- Must/Should/Could changes for subdevice/tile rank mapping assumptions.
- Must/Should/Could changes for replacing or bounding the raw `sycl::queue*`
  `ccl::stream` cache.
- A test coverage plan for visibility masks, subdevices, mixed devices, stream
  create/destroy, communicator teardown, and process shutdown.

## 5. Local transport, topology, and copy path selection

Merged topics: OFI SHM local transport, Level Zero copy engines/topology,
peer-access path comparison, and oneCCL-vs-XLA local D2D routing decisions.

XLA needs: XLA needs to know whether local oneCCL collectives and P2P use the
right local transport and Level Zero paths for one-process N-GPU execution, and
whether any XLA operation should prefer StreamExecutor peer D2D copies over
oneCCL for correctness or maintainability.

XLA anchors: `SetOnecclSingleProcessBootstrapEnvDefaults`,
`EnablePeerAccessTo`, `SyclMemcpyDeviceToDevice`,
`xla/backends/gpu/runtime/collective_permute_thunk.cc`,
`CreateOneApiDeviceDescription`.

oneCCL routes: ATL OFI, SHM provider path, IPC/peer access utilities, topology
discovery, Level Zero copy entries, local collective algorithm selection, and
SYCL/ZE send/recv paths.

Document output:

- Must/Should/Could changes for relying on `ofi` + `shm` even in local mode.
- Must/Should/Could changes for detecting or reporting missing local libfabric
  SHM support.
- Must/Should/Could changes for choosing oneCCL vs XLA D2D paths for local
  collective permute and P2P-like shapes.
- Must/Should/Could changes for topology-aware defaults or diagnostics.
- A test coverage plan for local transport configuration, peer access fallback,
  cross-tile/cross-root local shapes, and direct D2D fallback paths.

## 6. Collective operation contracts

Merged topics: allreduce, broadcast, reduce-scatter, allgather, alltoall,
collective-permute/P2P, send/recv, and barrier op-specific contracts.

XLA needs: XLA needs each `Launch*` wrapper to pass counts, datatypes, roots,
in-place aliases, pointer vectors, and peer matching in the form oneCCL expects.
The document should focus on API contract mismatches and implementation-derived
risks, not performance measurement.

XLA anchors: `LaunchAllReduce`, `LaunchBroadcast`, `LaunchReduceScatter`,
`LaunchAllGather`, `LaunchAllToAll`, `LaunchCollectivePermute`, `LaunchSend`,
`LaunchRecv`, `Barrier`, `ToOnecclCount`, and the GPU runtime collective thunk
files that call these methods.

oneCCL routes: public API parameter contracts, v2 legacy adapter forwarding,
datatype and reduction mapping, allreduce algorithm selection, broadcast root
handling, reduce-scatter count checks, allgather/allgatherv selector, alltoall
contiguous and pointer-vector paths, P2P matching, and barrier implementation.

Document output:

- Must/Should/Could changes for AllReduce reduction coverage, BF16/FP16 paths,
  complex handling, in-place aliasing, and threshold-sensitive algorithm
  selection.
- Must/Should/Could changes for Broadcast root-rank and in-place/out-of-place
  behavior.
- Must/Should/Could changes for ReduceScatter per-rank output count,
  divisibility, and dtype edge cases.
- Must/Should/Could changes for AllGather fixed-size assumptions and whether
  uneven gathers can reach the oneCCL path.
- Must/Should/Could changes for AllToAll contiguous fast path vs pointer-vector
  API, noncontiguous buffers, alignment assumptions, and rank-count scaling
  risks visible from implementation.
- Must/Should/Could changes for CollectivePermute/P2P cycles, self edges,
  fanout, peer matching, and deadlock risk.
- Must/Should/Could changes for Send/Recv tagless matching, same-rank behavior,
  cancellation, and launch ordering constraints.
- Must/Should/Could changes for Barrier semantics if not fully covered by
  section 1.
- A test coverage plan for every accepted XLA contract or validation change.

## 7. Pointer, registered memory, and symmetric memory contracts

Merged topics: raw pointer legality, `ToOnecclCount`, registered memory,
symmetric memory/window, USM address spaces, alignment, zero-size, aliasing,
lifetime, and destructor safety.

XLA needs: XLA passes `DeviceAddressBase::opaque()` pointers into oneCCL and
wraps oneCCL memory registration/window APIs. XLA needs to know which pointer
kinds and lifetimes are legal, how zero-size/null/aliasing cases behave, and
whether symmetric memory ABI assumptions are stable enough for kernels.

XLA anchors: `OnecclRegisteredMemory`, `OnecclSymmetricMemory`,
`ToOnecclCount`, `Launch*` pointer conversions, `SyclExecutor` allocation
paths, and StreamExecutor memory-space handling.

oneCCL routes: v2 memory registration/window APIs, legacy `register_buffer`,
window/symmetric allocation, memory handle ownership, deregistration, UMF/IPC
paths, datatype count handling, and public parameter checks.

Document output:

- Must/Should/Could changes for accepting or rejecting USM device/shared/host
  pointers, non-USM host pointers, null pointers, and zero counts.
- Must/Should/Could changes for overlapping send/recv buffers, aliased
  in-place operation, and complex dtype count conversion.
- Must/Should/Could changes for registered-memory alignment, repeated
  registration, per-communicator/per-context validity, and deregistration
  lifetime.
- Must/Should/Could changes for symmetric-memory ordering, all-rank
  requirements, local address requirements, `ccl::window*` kernel ABI stability,
  and destructor safety.
- A test coverage plan for pointer validation, memory registration lifetime,
  symmetric memory teardown, and kernel ABI guard coverage.

## 8. Communicator lifecycle, KVS, split, abort, and failure cleanup

Merged topics: local KVS deadlock, communicator creation cost/semantics,
thread-safety, split semantics, parent/child lifetime, abort/destruction, host
wait during failure, and repeated create/destroy.

XLA needs: XLA creates local per-rank oneCCL communicators concurrently, may
split them for cliques, and uses communicator reset as the local abort path.
XLA needs failures to unblock local ranks and not leave permanent waits in KVS,
oneCCL events, or scheduler threads.

XLA anchors: `InMemoryOnecclKvs`, `CreateOnecclCommunicatorsForRanks`,
`SplitOnecclCommunicatorsForRanks`, `OnecclCollectives::CreateCommunicators`,
`OnecclCommunicator::Abort`, `~OnecclCommunicator`, and GPU clique abort paths.

oneCCL routes: `kvs_interface` get/set contract, communicator creation,
communicator split, communicator destructor, request/event cancellation,
scheduler thread shutdown, and error behavior when peers are missing.

Document output:

- Must/Should/Could changes for making `InMemoryOnecclKvs::get`
  cancellation-aware or timeout/failure-aware.
- Must/Should/Could changes for prevalidating local ranks and clique sizes
  before communicator creation.
- Must/Should/Could changes for failure cleanup after partial communicator
  creation.
- Must/Should/Could changes for split parent/child lifetime and stale clique
  invalidation.
- Must/Should/Could changes for abort behavior while a host wait is in
  progress or while a matching peer operation is missing.
- A test coverage plan for creation failure, cancellation, split teardown,
  abort while waiting, repeated create/destroy, and stale clique cleanup.

## 9. Version pins, v1/v2 wrapper, and upgrade risk

Merged topics: pinned oneCCL v1, v2 wrapper source, build reproducibility,
version-specific reliability/performance issues, and source compatibility with
upstream oneCCL.

XLA needs: XLA needs a reproducible oneCCL dependency story and a clear mapping
between observed implementation risks and the pinned version used by XLA.
Recommendations should identify whether a fix belongs in XLA integration,
XLA's oneCCL BUILD/workspace pins, or upstream oneCCL.

XLA anchors: `third_party/oneccl/workspace.bzl`,
`third_party/oneccl/oneccl_v1.BUILD`, `third_party/oneccl/oneccl_v2.BUILD`,
`workspace2.bzl`, `MODULE.bazel`.

oneCCL routes: pinned `uxlfoundation/oneCCL` commit, v2 wrapper source,
`ze_loader.patch`, local oneCCL checkout, upstream release notes/issues, and
implementation areas cited by the other topic documents.

Document output:

- Must/Should/Could changes for replacing non-reproducible pins, especially
  `master-v2`, with immutable references.
- Must/Should/Could changes for upgrading oneCCL when a topic's Must/Should
  issue is already fixed upstream.
- Must/Should/Could changes for carrying XLA-local patches vs upstreaming them.
- Must/Should/Could changes for documenting oneAPI, Level Zero, libfabric, and
  oneCCL version assumptions in XLA.
- A test coverage plan for version-sensitive behavior that XLA should preserve
  across oneCCL upgrades.

## 10. Documentation set and triage order

Create one document per topic section above. Each document should be concise
enough to review independently, but should use shared citations where multiple
topics depend on the same oneCCL source path.

Recommended document names:

- `completion-ordering-visibility.md`
- `async-group-p2p-sequencing.md`
- `bootstrap-env-init.md`
- `device-context-stream-lifetime.md`
- `transport-topology-copy-path.md`
- `collective-op-contracts.md`
- `memory-contracts.md`
- `communicator-lifecycle-failure.md`
- `version-upgrade-risk.md`

Triage order:

1. Completion, queue ordering, barrier, and memory visibility.
2. Communicator lifecycle, KVS, split, abort, and failure cleanup.
3. Local bootstrap, env defaults, and init timing.
4. Collective operation contracts.
5. Pointer, registered memory, and symmetric memory contracts.
6. Host asynchrony, grouped collectives, and launch sequencing.
7. Local device, context, stream, and lifetime identity.
8. Local transport, topology, and copy path selection.
9. Version pins, v1/v2 wrapper, and upgrade risk.
