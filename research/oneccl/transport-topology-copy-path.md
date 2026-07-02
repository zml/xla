# Local transport, topology, and copy path selection

## Scope

This note covers single-host Intel GPU execution where XLA may either create
oneCCL communicators or use StreamExecutor/SYCL peer device-to-device copies for
local traffic. The focus is the topic-5 path in `research/oneccl/plan.md`:
OFI SHM bootstrap, missing local libfabric SHM handling, Level Zero topology and
copy engines, and the routing choice between oneCCL send/recv and XLA's local
collective-permute D2D path.

`research/oneccl/intel-map.md` was not present in this checkout, so the XLA
anchors from `plan.md` and the oneCCL routes in
`~/github/uxlfoundation/oneCCL/map.md` are the starting map.

## XLA needs

XLA needs local communication to satisfy three contracts:

- Bootstrap for one-process multi-GPU oneCCL use must not depend on MPI, PMI,
  or a network provider. For any all-local oneCCL communicator that XLA creates,
  the local transport defaults must be set before `ccl::init()`.
- Local collective-permute must keep StreamExecutor ordering and peer-memory
  semantics. If the fast path uses peer addresses and `MemcpyD2D`, unsupported
  peer access must fail clearly rather than silently changing to a different
  send/recv protocol.
- Routing must be explainable. XLA should be able to tell whether an operation
  used XLA's D2D copy path, oneCCL topo/Level Zero IPC, or oneCCL direct OFI,
  and failures should point to the transport or topology assumption that failed.

The current XLA shape is already split:

- `CollectivePermuteThunk::UsesLocalSyclPeerAccessPath` selects XLA D2D for
  local SYCL collective-permute when symmetric memory is not used.
- `CollectivePermuteThunk::RunPeerAccessCollective` enables peer access for the
  local clique and calls `RunPeerAccessPermute`.
- `RunPeerAccessPermute` exchanges peer addresses through XLA rendezvous and
  performs `stream.MemcpyD2D`; for SYCL it uses host synchronization at protocol
  boundaries because cross-device event barriers can stall.
- oneCCL `LaunchCollectivePermute`, `LaunchSend`, and `LaunchRecv` are still
  available for non-local or non-D2D-routed paths, and they synchronously wait on
  oneCCL events.

## oneCCL behavior

oneCCL does not enable OFI SHM by default in the legacy implementation. In
`deps/libccl/src/common/env/env.cpp`, `enable_shm` defaults to `0`; the ATL
transport default is build-dependent in source and documented as MPI in the
legacy environment reference. XLA's
`SetOnecclSingleProcessBootstrapEnvDefaults` compensates for single-process
local use by defaulting:

- `CCL_PROCESS_LAUNCHER=none`
- `CCL_LOCAL_SIZE=1`
- `CCL_LOCAL_RANK=0`
- `CCL_ATL_TRANSPORT=ofi`
- `CCL_ATL_SHM=1`
- `FI_PROVIDER=shm`

The oneCCL OFI path then treats SHM as a real provider, not just a hint:

- `atl_ofi_adjust_env` appends `shm` to `FI_PROVIDER` when SHM is enabled and
  the user supplied a provider list without SHM.
- `atl_ofi::open_providers` rejects `FI_PROVIDER=shm` for non-local process
  sets and rejects `FI_PROVIDER=shm` when `CCL_ATL_SHM` is not enabled.
- If SHM is requested but unavailable, oneCCL logs that the SHM provider is not
  available and disables SHM. With XLA's `FI_PROVIDER=shm`, there is no network
  provider to fall back to, so communicator creation should fail rather than
  silently using TCP/sockets.
- `atl_ofi_get_prov` chooses the SHM provider only when the peer is local and
  the message fits the provider's max message size; otherwise it chooses a
  network provider by endpoint index. Under XLA's single-process all-local
  `FI_PROVIDER=shm` setup, local traffic should therefore remain on SHM if the
  provider opened successfully.

For GPU-buffer point-to-point operations, oneCCL has a separate Level Zero
topology path:

- `topo_manager::ze_base_init` gathers Level Zero device information, builds a
  `zeDeviceCanAccessPeer` matrix, checks fabric connectivity, and records
  whether all relevant local peers have P2P access.
- `selector_send.cpp` and `selector_recv.cpp` default SYCL+Level Zero GPU
  send/recv to `topo`, but reject it through `ccl_can_use_topo_algo` and the
  SYCL-kernel capability checks when the peer is not local, buffers/backend are
  unsuitable, the stream is unsuitable, or topology requirements are not met.
- `ccl_coll_build_send` and `ccl_coll_build_recv` dispatch `topo` to
  `ccl_coll_build_topo_send` and `ccl_coll_build_topo_recv`; otherwise direct
  send/recv uses the ATL transport.
- Topo send/recv uses Level Zero IPC handle exchange plus D2D `copy_entry`
  operations and explicit ack entries. Copy queue selection is controlled by
  `CCL_ZE_*_COPY_ENGINE` modes, with fallback to compute queues when requested
  copy queues are not usable.

This means oneCCL can do local GPU P2P-like traffic without OFI SHM for the data
movement when the topo path is selected, but the communicator/bootstrap and
direct fallback still depend on the OFI/MPI ATL configuration.

## Must fix

- Preserve the current XLA D2D route for local SYCL collective-permute by
  default. Do not reroute local collective-permute through oneCCL send/recv
  merely because oneCCL has topo send/recv. XLA's path already encodes a SYCL
  host-sync workaround and avoids oneCCL communicator acquisition for the local
  case; changing this can violate ordering or rely on oneCCL stream/topology
  behavior that XLA has not validated.
- Keep oneCCL single-process local bootstrap defaults for all-local oneCCL
  communicator creation, and set them before `ccl::init()`. Removing
  `CCL_ATL_TRANSPORT=ofi`, `CCL_ATL_SHM=1`, or `FI_PROVIDER=shm` would make
  local one-process operation depend on oneCCL's build defaults and launcher
  assumptions.
- Do not silently fall back from the XLA local D2D collective-permute path to
  oneCCL when `EnablePeerAccessForLocalClique` fails. Peer-access failure means
  XLA cannot satisfy the selected peer-memory plan; any oneCCL fallback would
  need an explicit, tested routing decision with diagnostics and ordering
  coverage.

## Should fix

- Add actionable diagnostics around missing or overridden local SHM support.
  When all-local oneCCL communicator creation fails after XLA requested the
  single-process bootstrap defaults, report the effective values of
  `CCL_ATL_TRANSPORT`, `CCL_ATL_SHM`, `FI_PROVIDER`,
  `CCL_PROCESS_LAUNCHER`, `CCL_LOCAL_SIZE`, and `CCL_LOCAL_RANK`, and include a
  hint that local oneCCL bootstrap expects the libfabric SHM provider.
- Tighten the partial-override warning for the bootstrap default group. Today
  `SetOnecclEnvDefaultGroupIfUnset` leaves the whole group untouched when any
  variable is preset. For this topic, the risky case is a user-supplied
  `FI_PROVIDER` or `CCL_ATL_TRANSPORT` that prevents XLA from setting
  `CCL_ATL_SHM=1` and the no-launcher local rank defaults. Keep respecting user
  overrides, but make the warning identify the local transport risk.
- Add a small routing policy comment or helper around
  `CollectivePermuteThunk::UsesLocalSyclPeerAccessPath`: local SYCL
  collective-permute uses XLA D2D; oneCCL send/recv remains the communicator
  path for actual send/recv and non-local collective-permute. This avoids future
  cleanups accidentally "simplifying" the paths into oneCCL.
- Improve topology diagnostics at XLA decision points. Log, at `VLOG`, whether
  local collective-permute chose XLA D2D, whether peer access was enabled for
  each source/target executor pair, and when oneCCL communicator initialization
  is entered for an all-local clique. XLA cannot currently query oneCCL's
  internal `topo_manager`, so XLA-side peer-access diagnostics are the practical
  source of truth.
- Document that `CreateOneApiDeviceDescription` does not currently surface a
  complete topology model for routing. Until XLA owns root/tile/fabric
  metadata, topology-aware defaults should be diagnostics and explicit route
  guards, not automatic copy-engine or oneCCL algorithm tuning.

## Could fix

- Add an opt-in debug flag that forces local collective-permute through oneCCL
  for comparison or bisecting, disabled by default and guarded by clear
  unsupported-performance wording.
- Track oneCCL public API maturity for provider/topology reporting. If oneCCL
  exposes active ATL provider, topo algorithm, P2P matrix, or selected copy
  engine in a stable API, surface that in XLA communicator initialization logs.
- Add XLA device-description fields for local topology once the SYCL backend can
  populate them portably. Those fields could later support route choices for
  cross-tile or cross-root-device shapes without relying on oneCCL internals.
- Document advanced oneCCL tuning variables such as `CCL_TOPO_P2P_ACCESS`,
  `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK`, `CCL_ZE_COPY_ENGINE`, and
  `CCL_ZE_D2D_COPY_ENGINE` for support engineers, but do not set them by
  default from XLA without a separate tuning decision.

## Affected files/call sites

XLA:

- `xla/backends/gpu/collectives/oneccl_collectives.cc`
  - `SetOnecclEnvDefaultGroupIfUnset`
  - `SetOnecclSingleProcessBootstrapEnvDefaults`
  - `InitOnecclOnce`
  - `GetOnecclSingleProcessCliqueIds`
  - `OnecclCollectives::CreateCommunicatorsWithCancel`
  - `OnecclCollectives::InitializeTopology`
  - `LogOnecclCommunicatorInitialization`
- `xla/backends/gpu/collectives/oneccl_communicator.cc`
  - `OnecclCommunicator::LaunchCollectivePermute`
  - `OnecclCommunicator::LaunchSend`
  - `OnecclCommunicator::LaunchRecv`
  - `LaunchOnecclAndWait`
- `xla/backends/gpu/runtime/collective_permute_thunk.cc`
  - `EnablePeerAccessForLocalClique`
  - `CollectivePermuteThunk::UsesLocalSyclPeerAccessPath`
  - `CollectivePermuteThunk::Prepare`
  - `CollectivePermuteThunk::ExecuteOnStream`
  - `CollectivePermuteThunk::RunPeerAccessCollective`
  - `RunPeerAccessPermute`
- `xla/stream_executor/sycl/sycl_executor.cc`
  - `SyclExecutor::EnablePeerAccessTo`
  - `SyclExecutor::CanEnablePeerAccessTo`
  - `LevelZeroCanAccessPeer`
- `xla/stream_executor/sycl/sycl_stream.cc`
  - `SyclStream::Memcpy`
- `xla/stream_executor/sycl/sycl_gpu_runtime.{h,cc}`
  - `SyclMemcpyDeviceToDevice`
  - `SyclMemcpyDeviceToDeviceAsync`
- `xla/stream_executor/sycl/sycl_device_description.cc`
  - `CreateOneApiDeviceDescription`
- `third_party/oneccl/workspace.bzl`
  - XLA oneCCL pinning routes for v1 and v2.

oneCCL:

- `~/github/uxlfoundation/oneCCL/map.md`
- `deps/libccl/src/common/env/env.cpp`
  - `env_data::env_data`
  - parsing/logging for `CCL_ATL_TRANSPORT`, `CCL_ATL_SHM`,
    `CCL_TOPO_*`, `CCL_SYCL_PT2PT_*`, and `CCL_ZE_*_COPY_ENGINE`
- `deps/libccl/src/atl/atl_base_comm.cpp`
  - ATL default attributes and transport selection
- `deps/libccl/src/atl/ofi/atl_ofi.cpp`
  - `atl_ofi::init`
  - `atl_ofi::open_providers`
  - OFI send/recv provider selection call sites
- `deps/libccl/src/atl/ofi/atl_ofi_helper.cpp`
  - `atl_ofi_get_prov`
  - `atl_ofi_adjust_env`
  - `atl_ofi_set_env`
- `deps/libccl/src/topology/topo_manager.{hpp,cpp}`
  - `build_p2p_matrix`
  - `check_p2p_access`
  - `ze_base_init`
  - `to_string`
- `deps/libccl/src/coll/selection/selector_send.cpp`
- `deps/libccl/src/coll/selection/selector_recv.cpp`
- `deps/libccl/src/coll/selection/selection.cpp`
  - `ccl_can_use_topo_algo`
- `deps/libccl/src/coll/algorithms/utils/sycl_selection.cpp`
  - `can_use_sycl_kernels`
- `deps/libccl/src/coll/coll.cpp`
  - `ccl_coll_build_send`
  - `ccl_coll_build_recv`
- `deps/libccl/src/coll/algorithms/send/send.cpp`
  - `ccl_coll_build_direct_send`
  - `ccl_coll_build_topo_send`
- `deps/libccl/src/coll/algorithms/recv/recv.cpp`
  - `ccl_coll_build_direct_recv`
  - `ccl_coll_build_topo_recv`
- `deps/libccl/src/sched/entry/ze/ze_primitives.cpp`
  - `get_queue_group_type`
- `deps/libccl/src/sched/ze/ze_list_manager.cpp`
  - `queue_factory::can_use_queue_group`
  - copy queue selection
- `deps/libccl/src/sched/ze/ze_handle_manager.cpp`
  - Level Zero IPC handle open/cache paths

## Evidence to cite

- XLA sets local one-process bootstrap defaults in
  `xla/backends/gpu/collectives/oneccl_collectives.cc`:
  `SetOnecclSingleProcessBootstrapEnvDefaults` defaults OFI, SHM, and
  `FI_PROVIDER=shm`; `CreateCommunicatorsWithCancel`,
  `GetOnecclSingleProcessCliqueIds`, and `InitializeTopology` call it for
  local/single-process paths before communicator work.
- XLA local collective-permute route is in
  `xla/backends/gpu/runtime/collective_permute_thunk.cc`:
  `UsesLocalSyclPeerAccessPath`, `RunPeerAccessCollective`, and
  `RunPeerAccessPermute`. The implementation explicitly uses host sync for
  SYCL cross-device protocol boundaries and performs `stream.MemcpyD2D`.
- XLA SYCL peer access is checked/enabled in
  `xla/stream_executor/sycl/sycl_executor.cc` through
  `zeDeviceCanAccessPeer`, `ext_oneapi_can_access_peer`, and
  `ext_oneapi_enable_peer_access`.
- oneCCL environment defaults and parsing are in
  `~/github/uxlfoundation/oneCCL/deps/libccl/src/common/env/env.cpp`.
  Legacy oneCCL environment docs in
  `~/github/uxlfoundation/oneCCL/deps/libccl/doc/rst/source/env-variables.rst`
  document `CCL_ATL_TRANSPORT`, `CCL_ATL_SHM`,
  `CCL_PROCESS_LAUNCHER`, `CCL_SEND`, and `CCL_RECV`.
- oneCCL OFI SHM provider logic is in
  `deps/libccl/src/atl/ofi/atl_ofi.cpp` and
  `deps/libccl/src/atl/ofi/atl_ofi_helper.cpp`. `atl_ofi::open_providers`
  handles `FI_PROVIDER=shm` and missing SHM; `atl_ofi_get_prov` selects SHM for
  local peers when available.
- oneCCL topology and P2P matrix logic is in
  `deps/libccl/src/topology/topo_manager.cpp`, especially
  `build_p2p_matrix`, `check_p2p_access`, and `ze_base_init`.
- oneCCL send/recv topo selection and build paths are in
  `deps/libccl/src/coll/selection/selector_send.cpp`,
  `deps/libccl/src/coll/selection/selector_recv.cpp`,
  `deps/libccl/src/coll/selection/selection.cpp`,
  `deps/libccl/src/coll/algorithms/utils/sycl_selection.cpp`,
  `deps/libccl/src/coll/coll.cpp`,
  `deps/libccl/src/coll/algorithms/send/send.cpp`, and
  `deps/libccl/src/coll/algorithms/recv/recv.cpp`.
- XLA pins oneCCL sources in `third_party/oneccl/workspace.bzl`: v1 is pinned
  to `uxlfoundation/oneCCL` commit
  `4ceafd15c03ce46f11eeaf91781a92afebd3cecf`; v2 uses the `master-v2`
  archive route.
- Official libfabric SHM provider documentation:
  https://ofiwg.github.io/libfabric/main/man/fi_shm.7.html.
- oneCCL's legacy docs refer to Level Zero IPC memory documentation:
  https://spec.oneapi.io/level-zero/latest/core/PROG.html#memory-1.
- The reviewed oneCCL checkout was
  `~/github/uxlfoundation/oneCCL` at
  `1318c3aaf8a67536ff538331c418c8cea3114f33`; this may differ from the XLA
  pinned v1 archive.

## Test coverage plan

Static review only was performed for this document. If XLA accepts the
recommendations, add tests in these layers:

- Bootstrap configuration tests:
  - Verify all-local/single-process topology calls
    `SetOnecclSingleProcessBootstrapEnvDefaults` before oneCCL initialization.
  - Verify the complete default group values for local bootstrap.
  - Verify partial user overrides produce a warning that names local OFI SHM
    risk and lists the unset companion variables.
- Missing SHM/provider diagnostics tests:
  - Mock a local oneCCL communicator creation failure after XLA applies
    `FI_PROVIDER=shm` and assert the error includes the relevant env values and
    a libfabric SHM hint.
  - Cover the case where a user preset `FI_PROVIDER` prevents XLA from applying
    the local bootstrap group.
- Local collective-permute routing tests:
  - For local SYCL clique keys with no symmetric memory, assert
    `UsesLocalSyclPeerAccessPath` selects the XLA D2D route and does not acquire
    a oneCCL communicator.
  - With a fake executor that rejects peer access, assert local
    collective-permute fails clearly instead of falling back to oneCCL.
  - Cover no-source zero-fill, one-source/one-target, self-edge, and cyclic
    source-target plans on the XLA D2D path.
- Peer-access and D2D copy integration tests:
  - On multi-GPU Intel systems, run opt-in tests for same-tile, cross-tile, and
    cross-root/local-device shapes where topology labels are available.
  - Validate the direct D2D fallback behavior for unsupported peer access:
    either skip at setup when the hardware cannot access peers, or assert the
    specific XLA error when peer access is expected but unavailable.
- oneCCL send/recv routing tests:
  - Keep separate coverage for actual `Send`/`Recv` and non-local
    collective-permute paths that use oneCCL, so tests do not conflate XLA D2D
    collective-permute with oneCCL topo send/recv.
  - Add a diagnostic-only test hook, not a performance test, that records
    whether XLA entered all-local oneCCL communicator initialization.

## Rollout risk

- User env override risk: XLA currently leaves the whole default group alone if
  any one variable is preset. Better diagnostics can expose existing fragile
  setups, but should not override user intent without a separate compatibility
  decision.
- oneCCL version risk: XLA's pinned oneCCL v1 archive may not match the local
  oneCCL checkout reviewed here. Recommendations that depend on internals such
  as `topo_manager` or `atl_ofi::open_providers` should be guarded by source
  version checks or treated as diagnostics rather than stable API contracts.
- Performance risk: XLA's local D2D collective-permute path uses host sync for
  SYCL protocol boundaries. oneCCL topo send/recv may be faster for some local
  P2P-like shapes, but changing the default route is higher risk than adding a
  guarded debug path and tests first.
- Failure-mode risk: proactive SHM diagnostics should make missing libfabric
  SHM support fail clearer, not later or differently. Avoid adding a fallback
  from `FI_PROVIDER=shm` to a network provider for single-process local use
  unless XLA explicitly supports that mode.
