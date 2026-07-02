# Communicator lifecycle, KVS, split, abort, and failure cleanup

## Scope

This reviews XLA's oneCCL communicator lifecycle for local Intel GPU execution:
`InMemoryOnecclKvs`, concurrent communicator creation, communicator split,
parent/child clique lifetime, communicator reset as abort, host waits on oneCCL
events, and stale clique cleanup. The review used static source and
documentation only. No tests, examples, benchmarks, profilers, or ad hoc
experiments were run.

oneCCL source review used `~/github/uxlfoundation/oneCCL`, starting
from `map.md`. In this checkout, `research/oneccl/intel-map.md` was not present.

## XLA needs

XLA creates local per-rank oneCCL communicators concurrently for one process
with multiple GPU ranks. All local ranks must either finish communicator
creation/split together or all unblock and fail together. XLA cannot allow one
rank to return an error while another local rank remains permanently blocked in
KVS, oneCCL communicator creation, oneCCL split, `ccl::event::wait()`, or
transport-level peer matching.

XLA also needs cached cliques to be invalidated as a unit. A split child clique
must not outlive, or keep a raw stale reference to, an invalidated parent
communicator set. If a task incarnation fails, both already-published cliques
and cliques still pending construction must receive a cancellation signal that
is capable of waking local creation/split waits.

## oneCCL behavior

The public legacy C++ KVS interface is synchronous and status-less:
`deps/libccl/include/oneapi/ccl/kvs.hpp` defines `kvs_interface::get`,
`set`, and `get_id`, with no timeout or cancellation argument. oneCCL wraps a
user-supplied KVS in `users_kvs`; `users_kvs::kvs_get_value_by_name_key`
concatenates the oneCCL namespace/key and calls `kvs->get()` directly
(`deps/libccl/src/atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.cpp`).
The oneCCL process-manager layer has `CCL_KVS_GET_TIMEOUT`, but that timeout is
checked after `users_kvs` returns. If XLA's `get()` blocks forever, the oneCCL
timeout cannot fire.

XLA's current local KVS blocks forever. `InMemoryOnecclKvs::get` waits in a
condition-variable loop until the key exists and does not observe
`CancellationToken`, a deadline, a failure status, or KVS destruction
(`xla/backends/gpu/collectives/oneccl_collectives.cc:432`). XLA passes the same
KVS to all local rank-creation tasks in `CreateOnecclCommunicatorsForRanks`.
If one local rank fails before setting a key that another rank needs, the
waiting rank can block the whole `JoinFutures(...).Await()`.

oneCCL communicator creation validates only part of what XLA needs. In
`ccl_comm::init`, oneCCL rejects `rank >= size` and `size <= 0`, but the source
does not reject a negative rank before constructing internal communicator state
(`deps/libccl/src/comm/comm.cpp`). XLA's
`ValidateCreateCommunicatorInputs` currently checks non-empty ranks, local rank
count, and `CliqueIds` count, but it does not verify rank uniqueness, rank
range, rank-to-clique membership, or duplicate local devices before entering
oneCCL.

oneCCL split is collective over the parent communicator. `ccl_comm::split`
rejects negative color/key and, with `split_external_use=true`, calls
`create_subcomm_split_independent` (`deps/libccl/src/comm/comm.cpp`). The OFI
split path constructs a new `atl_ofi_comm` from the parent, exchanges split
metadata with `allgatherv`, and waits for it; the MPI path calls
`MPI_Comm_split`. Missing parent ranks can therefore block inside oneCCL or the
transport. XLA's split path performs size checks but does not prevalidate color,
key range/uniqueness, parent rank coverage, or parent/child lifetime before
entering oneCCL.

oneCCL event cancellation is not usable for XLA abort today. The public
`event` API exposes `cancel()`, but `host_event_impl::cancel()` and
`native_event_impl::cancel()` throw "not implemented"
(`deps/libccl/src/common/event/impls/host_event.cpp`,
`deps/libccl/src/common/event/impls/native_event.cpp`). XLA calls
`event.wait()` synchronously in `WaitForOnecclEvent`; that blocks in oneCCL
request wait and then waits the native SYCL event when present. The v2 C API
declares `onecclCommAbort`, but the legacy plugin used for the legacy C++ route
wires `destroy` and `split` and does not install an `abort` callback
(`plugins/legacy/ccl_legacy.cpp:709`). XLA's `OnecclCommunicator::Abort` only
cancels XLA's token and resets the `ccl::communicator` object; it is not a
documented oneCCL operation abort.

The official oneCCL specification describes communicators as rank groups wired
with KVS and events as operation-progress objects. It specifies blocking
`event::wait()` and non-blocking `event::test()`, but it does not provide a
communicator-abort or KVS-cancellation contract for the legacy C++ API:

- https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/main_objects
- https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/operation_progress

XLA has a process-global clique cache and stale-incarnation invalidation, but
the pending path is incomplete. `ProcessGpuCliques` stores `pending_cliques`,
yet `UpdateGlobalProcessInfo` aborts only already-published cliques. In the
non-split initialization path, a pending cancellation token is inserted before
`CreateCommunicatorsWithCancel` and erased after it returns. In the split path,
the pending token is inserted and then erased before
`SplitCommunicatorsWithCancel` starts, so a task-failure update cannot reach an
in-progress split.

## Must fix

- Make `InMemoryOnecclKvs::get` cancellation-aware and timeout/failure-aware.
  The KVS should share the clique `CancellationToken`, support an explicit
  failure state, signal all waiters on cancel/failure/destruction, and throw a
  caught exception from `get()` rather than waiting forever. Because oneCCL's
  `kvs_interface::get` has no status return, the practical XLA-side contract is
  to throw `std::runtime_error` or `ccl::exception` and rely on the existing
  `OnecclValue` exception wrapper around communicator creation.

- Cancel the local KVS and clique token on the first communicator creation or
  split failure. `CreateOnecclCommunicatorsForRanks` and
  `SplitOnecclCommunicatorsForRanks` should not wait for all futures to finish
  after the first hard error without first waking peer ranks blocked in KVS or
  split. Any successfully constructed communicators from a failed set should be
  aborted/reset as a group before the error is returned.

- Wire pending clique cancellation into stale-task handling. `UpdateGlobalProcessInfo`
  must cancel both `state.cliques` and `state.pending_cliques`; split
  initialization must keep the pending token registered until
  `SplitCommunicatorsWithCancel` has reached a definitive state. Otherwise task
  failure can invalidate published cliques while creation/split threads remain
  blocked below the cache.

- Prevalidate local ranks and clique sizes before oneCCL creation. Before
  launching the rank thread pool, XLA should verify `num_devices > 0`, every
  rank is unique, every rank is in `[0, num_devices)`, local rank count matches
  `GpuCliqueKey::num_local_participants()`, local ranks are members of the
  clique key, devices/stream executors are non-null and unique as required, and
  `num_devices` fits oneCCL's `int` size. Do this before any oneCCL rank enters
  KVS.

- Prevalidate split inputs before oneCCL split. XLA should verify non-null
  oneCCL parent communicators, parent rank coverage, key uniqueness and range,
  non-negative split color, and rank/key vector consistency before launching
  split tasks. `GetCommSplitColor` should avoid the `abs(INT_MIN)` negative
  corner case rather than relying on oneCCL to reject it after entering split.

- Make `OnecclCommunicator` abort/thread state safe. `comm_` and `aborted_`
  are currently accessed without a mutex, while health checks or stale-clique
  abort can race with a host thread blocked in `LaunchOnecclAndWait`.
  Introduce explicit operation state so `Abort()` cannot reset the communicator
  while a launch is still using it, and so new launches fail immediately after
  cancellation.

- Make host waits abort-aware. XLA cannot rely on oneCCL `event.cancel()` or
  legacy plugin abort. `WaitForOnecclEvent` should be able to poll `event.test`
  with cancellation/deadline checks, return a cancellation status, and mark the
  communicator/clique poisoned. If oneCCL cannot safely destroy an incomplete
  operation, XLA should quarantine the communicator rather than reset it under
  an in-flight oneCCL request.

- Invalidate split children when a parent clique is aborted or becomes stale.
  `GpuClique` stores a raw `const GpuClique* parent`. A child whose key does
  not contain a failed incarnation can survive erasure of a larger parent whose
  key does contain that incarnation, leaving `IsParentSupersetOf` with a
  dangling parent pointer. Store parent lifetime/generation explicitly or erase
  all descendants when a parent is invalidated.

## Should fix

- Add concise diagnostics for KVS wait failures: clique key, local ranks,
  missing key fingerprint or sanitized key suffix, cancellation reason, and
  elapsed wait. This should be available without enabling full oneCCL debug
  logging.

- Add a creation/split cleanup helper shared by `CreateCommunicatorsWithCancel`
  and `SplitCommunicatorsWithCancel`. It should own the token, KVS failure
  state, partial communicator vector, and concurrent abort/reset policy so the
  two paths cannot diverge.

- Record a per-clique lifecycle generation and include it in logs. That makes
  repeated create/destroy, abandon-and-resplit, stale invalidation, and parent
  generation mismatches diagnosable.

- Keep the existing stale-parent compatibility logic, but document it next to
  `IsParentSupersetOf`. The current logic avoids one deadlock class by
  abandoning cliques created from incompatible parents; reviewers need to know
  that parent identity is part of split safety, not just a cache optimization.

- Add an upstream oneCCL issue or version-tracking note for legacy C++ abort
  semantics. If a future oneCCL release implements communicator abort or event
  cancellation for the SYCL path, XLA should gate use of that behavior by
  version and plugin capability rather than assuming it from the C API name.

## Could fix

- Bound or garbage-collect completed local KVS entries after communicator
  creation if repeated create/destroy shows unbounded memory growth. This is
  secondary after cancellation because local KVS objects are per creation today.

- Expose a debug-only fault-injection hook for oneCCL communicator creation and
  split so lifecycle tests can deterministically fail one rank before KVS set,
  after KVS set, after oneCCL creation, and after wrapper creation.

- Track migration to oneCCL v2 communicator APIs if the v2 legacy plugin grows
  a real `onecclCommAbort` implementation. XLA currently uses the legacy C++
  communicator route directly.

- Add developer documentation explaining why XLA reset is not equivalent to a
  oneCCL operation abort for in-flight waits.

## Affected files/call sites

XLA:

- `xla/backends/gpu/collectives/oneccl_collectives.cc`:
  `InMemoryOnecclKvs::get`, `InMemoryOnecclKvs::set`,
  `CreateOnecclCommunicatorsForRanks`,
  `SplitOnecclCommunicatorsForRanks`,
  `ValidateCreateCommunicatorInputs`, `ValidateSplitCommunicatorInputs`,
  `CheckOnecclCommunicatorNotCancelled`,
  `OnecclCollectives::GetOrCreateKvs`,
  `OnecclCollectives::CreateCommunicatorsWithCancel`, and
  `OnecclCollectives::SplitCommunicatorsWithCancel`.
- `xla/backends/gpu/collectives/oneccl_communicator.{h,cc}`:
  `WaitForOnecclEvent`, `LaunchOnecclAndWait`,
  `OnecclCommunicator::~OnecclCommunicator`, `OnecclCommunicator::Abort`,
  `CheckReady`, `LaunchOnStream`, `Execute`, and
  `OnecclCommunicator::Split`.
- `xla/backends/gpu/collectives/gpu_cliques.{h,cc}`:
  `ProcessGpuCliques::pending_cliques`, both `InitializeGpuClique` overloads,
  `AbortCliquesWithIncarnations`, `UpdateGlobalProcessInfo`,
  `DestroyAcquiredCliques`, and `LockableGpuClique::IsParentSupersetOf`.
- `xla/backends/gpu/collectives/gpu_clique.{h,cc}`:
  `GpuClique::Cancel`, `GpuClique::Abort`, parent storage, and child lifetime.
- `xla/backends/gpu/runtime/collective_cliques.{h,cc}`:
  `AcquireCollectiveCliques` and its all-ranks-must-enter contract.
- `xla/core/collectives/communicator.h`: `Communicator::Abort` contract.

oneCCL:

- `deps/libccl/include/oneapi/ccl/kvs.hpp`: `kvs_interface` contract.
- `deps/libccl/src/atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.cpp`:
  user KVS forwarding.
- `deps/libccl/src/atl/util/pm/pmi_resizable_rt/pmi_resizable_simple.cpp`:
  `CCL_KVS_GET_TIMEOUT` loop that cannot fire while user `get()` blocks.
- `deps/libccl/src/communicator_impl.hpp`: legacy communicator factory
  forwarding to `comm_interface::create_comm_impl*`.
- `deps/libccl/src/comm/comm.cpp`: communicator init, KVS wrapping,
  `ccl_comm::split`, `create_subcomm`, and
  `create_subcomm_split_independent`.
- `deps/libccl/src/atl/ofi/atl_ofi_comm.cpp` and
  `deps/libccl/src/atl/ofi/atl_ofi.cpp`: OFI split metadata exchange and wait.
- `deps/libccl/src/atl/mpi/atl_mpi_comm.cpp` and
  `deps/libccl/src/atl/mpi/atl_mpi.cpp`: MPI split path.
- `deps/libccl/include/oneapi/ccl/event.hpp`,
  `deps/libccl/src/ccl_app_api_event.cpp`,
  `deps/libccl/src/common/event/impls/host_event.cpp`, and
  `deps/libccl/src/common/event/impls/native_event.cpp`: wait/test/cancel
  behavior.
- `src/api.cpp` and `plugins/legacy/ccl_legacy.cpp`: v2 `onecclCommAbort`
  dispatch and legacy plugin lack of abort callback.

## Evidence to cite

- `research/oneccl/plan.md`, topic 8 and the Must/Should/Could rubric.
- `~/github/uxlfoundation/oneCCL/map.md`, especially the v2
  plugin, legacy C++ API, communicator, scheduler, transport, and docs routes.
- XLA `InMemoryOnecclKvs::get`: indefinite condvar wait with no token/deadline.
- XLA `CreateOnecclCommunicatorsForRanks` and
  `SplitOnecclCommunicatorsForRanks`: per-rank thread pools plus
  `JoinFutures(...).Await()`.
- XLA `gpu_cliques.cc`: pending token insertion/erasure, stale-incarnation
  checks, published-clique abort, and raw parent pointer compatibility checks.
- oneCCL `kvs_interface` and `users_kvs`: no status/timeout in user KVS API,
  direct forwarding to user `get()`.
- oneCCL `pmi_resizable_simple.cpp`: `CCL_KVS_GET_TIMEOUT` is ineffective if
  user `get()` never returns.
- oneCCL `comm.cpp` and `atl_ofi_comm.cpp`: split can block in transport-level
  collective exchange and wait.
- oneCCL event implementations: `wait()` blocks, `cancel()` is not implemented.
- oneCCL v2 C API and legacy plugin: `onecclCommAbort` exists in the dispatcher
  but the legacy plugin does not install `comm->abort`.
- oneAPI spec:
  - https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/main_objects
  - https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/operation_progress

## Test coverage plan

- KVS creation failure: use a fake/fault-injected oneCCL creation wrapper or
  KVS to make one local rank fail before publishing expected keys; verify all
  local creation futures return promptly and the KVS reports cancellation.
- KVS timeout/cancellation: call `InMemoryOnecclKvs::get` for a missing key,
  cancel/fail the KVS, and verify `get` wakes and throws/returns through the
  XLA wrapper as a non-OK status.
- Rank prevalidation: unit-test duplicate ranks, negative ranks, rank equal to
  clique size, mismatched local participant count, non-member local ranks,
  empty ranks, and oversized `num_devices` before any oneCCL call is made.
- Partial creation cleanup: inject failure after one rank creates a
  communicator and another rank is still waiting; verify the created
  communicator is aborted/reset and the clique is not cached.
- Split teardown: inject split failure before and after a child communicator is
  returned; verify child comms are cleaned up, parent remains usable only when
  oneCCL guarantees it, and failed children are not inserted into the cache.
- Parent/child stale cleanup: create a parent clique and split child, mark an
  incarnation present only in the parent as failed, and verify both the parent
  and all descendants with that parent generation are invalidated or safely
  detached.
- Abort while waiting: start a host wait on a fake oneCCL event that never
  completes, call clique/communicator abort, and verify the wait returns a
  cancellation status without resetting the communicator under the waiter.
- Missing peer operation: use fake send/recv or collective-permute launch
  plumbing to simulate unmatched peer work; verify cancellation unwinds the host
  waiter and poisons the clique.
- Repeated create/destroy: create, abort/destroy, and recreate the same local
  clique repeatedly with fault injection enabled; verify no stale cached clique,
  stale KVS, or stale parent generation is reused.
- Pending stale task: update global process info while communicator creation
  and split are pending; verify pending tokens are cancelled, waiters wake, and
  no pending entry is leaked.

## Rollout risk

The main behavior change is that local oneCCL creation/split failures will
return errors instead of hanging. That can expose previously latent rank or
clique bugs earlier in execution, but this is the desired failure mode.

Timeouts must be conservative and configurable. A hard default that is too low
could turn slow communicator creation into false failures on loaded systems.
Prefer explicit cancellation on known failure and use timeouts primarily for
diagnostics/watchdog-style protection.

Abort cleanup is sensitive because oneCCL does not provide a legacy C++
operation-abort primitive. If an event is in flight and cannot be cancelled,
resetting the communicator may be unsafe. The safer rollout is to poison and
quarantine the communicator/clique, wake XLA waiters, and only destroy when no
operation is in progress or when oneCCL documents that destruction is safe.

Split parent/child invalidation can reduce communicator-cache reuse. That is a
reasonable tradeoff for correctness; performance-sensitive reuse can be rebuilt
with generation-aware parent ownership after the lifetime rules are explicit.

User environment overrides such as `CCL_KVS_GET_TIMEOUT` should not be treated
as sufficient protection for XLA's in-memory KVS. XLA must own local KVS
cancellation because oneCCL's timeout is downstream of the blocking user
callback.
