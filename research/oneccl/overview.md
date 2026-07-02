# oneCCL research overview

## Sources read

This overview synthesizes all notes currently in `research/oneccl`:

- `async-group-p2p-sequencing.md`
- `bootstrap-env-init.md`
- `collective-op-contracts.md`
- `communicator-lifecycle-failure.md`
- `completion-ordering-visibility.md`
- `device-context-stream-lifetime.md`
- `memory-contracts.md`
- `transport-topology-copy-path.md`
- `version-upgrade-risk.md`

Priority is based on correctness first: deadlock, use-after-free, wrong-device
launch, data corruption, and process-global initialization hazards outrank
performance, cleanup, and future feature work.

## Executive priority list

### P0: prevent hangs, stale launches, and silent corruption

1. Make communicator creation, split, KVS, abort, and host waits
   cancellation-aware.

   Status: fixed by this change.

   The implementation now threads the shared cancellation token through the
   all-local oneCCL KVS, creation, split, launch, and host wait paths. Local KVS
   waits can fail on explicit failure, token cancellation, or destruction; the
   first local creation/split failure cancels peer ranks and aborts partial
   communicators; split pending tokens remain registered until split completion;
   stale-task updates cancel pending clique tokens and invalidate children of
   stale parent cliques; host waits poll `event.test()` so they can observe
   cancellation before the final completion `wait()`; and communicator abort no
   longer resets an underlying oneCCL handle while another host thread is using
   it. Communicators that see failed or cancelled oneCCL work are poisoned and
   quarantined if unfinished oneCCL operations make destruction unsafe.

   XLA's in-memory oneCCL KVS can block forever because `get()` waits for a key
   without observing cancellation, failure, timeout, destruction, or stale-task
   invalidation. oneCCL's user KVS API has no status or cancellation return, so
   XLA must own the failure path. On the first local rank creation/split failure,
   XLA should cancel the shared token, fail the KVS, wake all local waiters,
   clean up any partial communicator set, and avoid caching failed cliques.

   This should also fix pending-clique cancellation during stale-task handling,
   keep split pending tokens registered until split completes, poison or
   quarantine communicators with in-flight uncancellable oneCCL operations, and
   make `OnecclCommunicator::Abort()` safe against concurrent launches.

   Main files:
   `xla/backends/gpu/collectives/oneccl_collectives.cc`,
   `xla/backends/gpu/collectives/oneccl_communicator.{h,cc}`,
   `xla/backends/gpu/collectives/gpu_cliques.{h,cc}`,
   `xla/backends/gpu/collectives/gpu_clique.{h,cc}`.

2. Replace the process-lifetime oneCCL stream cache with stream-lifetime
   ownership and launch-time identity checks.

   Status: fixed by `1da3722637` (`Fix oneCCL stream identity handling`).

   `ToOnecclStream()` previously keyed a process-lifetime global cache by raw
   `sycl::queue*`. After a `SyclStream` was destroyed, that pointer could become
   stale or reused, letting XLA launch oneCCL work on an old copied queue or
   wrong device. The fix now stores the oneCCL stream cache as a
   `Stream::Resource` attached to the XLA stream, so the oneCCL stream lifetime
   follows the XLA stream lifetime.

   Before reusing or creating the cached oneCCL stream, the launch path now
   verifies the communicator executor identity and the SYCL queue's device,
   context, and in-order property. These checks run on each oneCCL launch before
   cache lookup, protecting the completion contract and catching stale or
   mismatched queue handles before launch.

   Main files:
   `xla/backends/gpu/collectives/oneccl_communicator.cc`.

3. Fail closed for unsafe oneCCL P2P shapes and keep local SYCL
   collective-permute on XLA's D2D path.

   oneCCL send/recv can require grouped concurrent progress, but XLA cannot use
   oneCCL groups as a completion primitive, and the current XLA oneCCL wrapper
   waits every send/recv immediately. That can deadlock cycles, all-rank
   send-first patterns, ragged all-to-all send/recv loops, missing peer matches,
   and other unmatched P2P configurations.

   Add oneCCL-specific validation before `LaunchSend`, `LaunchRecv`, and
   oneCCL-routed `LaunchCollectivePermute`. Reject cycles or unsupported fanout,
   duplicate peers, missing matches, out-of-range peers, and any shape that
   needs nonblocking grouped P2P progress. Bypass oneCCL for self edges with a
   local copy/no-op because oneCCL's SYCL self P2P queues are process-global and
   unkeyed. Do not fall back from XLA's local D2D collective-permute path to
   oneCCL if peer access fails; fail clearly instead.

   Main files:
   `xla/backends/gpu/collectives/oneccl_communicator.cc`,
   `xla/backends/gpu/runtime/collective_permute_thunk.cc`,
   `xla/backends/gpu/runtime/p2p_thunk_common.{h,cc}`,
   `xla/backends/gpu/runtime/send_thunk.cc`,
   `xla/backends/gpu/runtime/recv_thunk.cc`,
   `xla/backends/gpu/runtime/ragged_all_to_all_thunk.cc`.

4. Split and validate oneCCL environment defaults before any oneCCL init path.

   The current all-or-nothing default group is too coarse. A user setting only
   `FI_PROVIDER`, only `CCL_ATL_SHM`, or only one SYCL threshold can suppress
   unrelated mandatory defaults. For all-local single-process XLA, the local
   launcher coordinates and OFI SHM transport policy must be complete before
   oneCCL parses env.

   Split defaults into process-coordinate, transport/provider, and threshold
   groups. Complete compatible partial overrides, fail incompatible ones before
   `ccl::init()`, and validate `FI_PROVIDER=shm` with
   `CCL_ATL_TRANSPORT=ofi` and `CCL_ATL_SHM=1`. Apply the same pre-init policy
   to every XLA-controlled path that can initialize oneCCL, including the v2
   `Allocate`/`Deallocate` entry points or gate those APIs as unsupported.

   Keep `CCL_SYCL_ALLGATHERV_LL_THRESHOLD=1GiB` as an XLA-owned safety default
   until the BF16 event-completion issue is fixed and version-gated.

   Main files:
   `xla/backends/gpu/collectives/oneccl_collectives.cc`,
   `third_party/oneccl/oneccl_v1.BUILD`,
   `third_party/oneccl/oneccl_v2.BUILD`.

5. Add oneCCL-side operation, pointer, count, and alias validation before
   launch.

   XLA currently passes raw addresses directly to oneCCL and relies too much on
   oneCCL's internal validation, which is incomplete or environment-gated for
   XLA's needs. Add explicit validation for:

   - complex reductions: allow complex `SUM` only; reject complex `PRODUCT`,
     `MIN`, and `MAX` because the real/imag doubled-count mapping is not the
     HLO arithmetic semantics;
   - pointer legality: non-zero GPU operations should accept only pointers that
     oneCCL can validate in the communicator SYCL context, conservatively USM
     device/shared pointers;
   - zero counts: short-circuit or explicitly define behavior before oneCCL,
     especially because SYCL zero-byte allocation can return null;
   - count conversion: check overflow before doubling complex element counts;
   - aliasing: allow only documented exact or segmented in-place forms, reject
     partial overlap, and reject oneCCL SYCL all-to-all in-place unless a safe
     copy-through-scratch path is implemented.

   Also gate oneCCL out of symmetric-kernel, multimem, `Put`, `Signal`, and
   `WaitSignal` paths. A `ccl::window*` host wrapper is not a stable device
   kernel ABI.

   Main files:
   `xla/backends/gpu/collectives/oneccl_communicator.cc`,
   `xla/backends/gpu/collectives/oneccl_registered_memory.{cc,h}`,
   `xla/backends/gpu/collectives/oneccl_symmetric_memory.{cc,h}`,
   `xla/stream_executor/sycl/sycl_executor.cc`,
   `xla/stream_executor/sycl/sycl_gpu_runtime.cc`,
   GPU collective thunk files.

6. Preserve the current oneCCL completion boundary and make cross-stream
   ordering explicit.

   Status: fixed by this change.

   The implementation now documents the synchronous oneCCL completion contract
   at the launch and Future-construction points: XLA returns a ready Future only
   after the returned `ccl::event` has been waited. It also makes explicit that
   same-stream ordering relies on the in-order SYCL queue and oneCCL front
   barrier, while cross-stream producer/consumer dependencies must be expressed
   with StreamExecutor `RecordEvent()`/`WaitFor()` barriers. The SYCL event path
   documents the same-device event-barrier contract and continues to reject
   cross-device SYCL event waits. `BarrierAfterExecutable()` is documented as a
   stronger module-exit stream completion and host rendezvous boundary than a
   communicator barrier.

   For the current integration, `ccl::event::wait()` is the completion
   boundary, and `OnecclCommunicator::Execute()` should keep returning an
   already-completed `Future` only after the wait succeeds. Same-stream ordering
   is supported by XLA's in-order SYCL queues and oneCCL's front barrier, but
   cross-stream producer/consumer ordering must be materialized by XLA
   StreamExecutor events and waits before or after the oneCCL launch.

   Do not weaken this into enqueue-only completion, completed futures before
   `event.wait()`, or oneCCL group completion. Do not replace
   `BarrierAfterExecutable()` with a streamed oneCCL barrier; it is a stronger
   module-exit stream completion and host rendezvous boundary.

   Main files:
   `xla/backends/gpu/collectives/oneccl_communicator.cc`,
   `xla/stream_executor/sycl/sycl_stream.cc`,
   `xla/stream_executor/sycl/sycl_event.cc`,
   `xla/service/gpu/gpu_executable.cc`.

### P1: make support boundaries explicit and upgrade-safe

7. Pin the oneCCL v2 archive by immutable commit and add a version manifest.

   `@oneccl_v1` is pinned to an immutable commit, but `@oneccl` uses a
   `master-v2` archive URL with a fixed SHA. The SHA prevents silent content
   substitution, but the branch URL makes source identity hard to audit and can
   break clean fetches after branch movement. Replace it with the reviewed
   commit archive and encode the commit in `strip_prefix`.

   Add a checked-in manifest near `third_party/oneccl` that records the v2
   wrapper commit, v1/libccl commit, SHA256 values, upstream tag/release note,
   `ze_loader.patch` status, oneAPI version, Level Zero version, and
   libfabric/OFI expectations. Treat v2 wrapper and v1/libccl upgrades as an
   atomic pair unless an intentional mismatch is documented.

   Main files:
   `third_party/oneccl/workspace.bzl`,
   `third_party/oneccl/oneccl_v1.BUILD`,
   `third_party/oneccl/oneccl_v2.BUILD`,
   `third_party/oneccl/ze_loader.patch`.

8. Enforce oneCCL API-surface boundaries.

   XLA's communicator path uses the legacy C++ API, while allocation and error
   helpers call v2 C API entry points. The v2 legacy plugin allocation path
   depends on v2 `onecclSetDevice` thread-local state that XLA's C++ path does
   not establish. Either remove/gate v2 allocation use, move it to a path that
   shares the existing C++ device/context setup, or add an explicit v2
   initialization wrapper and version gate.

   Main files:
   `xla/backends/gpu/collectives/oneccl_collectives.cc`,
   `xla/backends/gpu/collectives/oneccl_errors.cc`,
   `xla/backends/gpu/collectives/oneccl_communicator.cc`,
   `third_party/oneccl/oneccl_v2.BUILD`.

9. Validate local rank, device, and subdevice identity before communicator
   creation.

   Before launching local rank creation, check rank uniqueness, rank range,
   local participant count, local-rank membership, non-null and unique
   executors/devices, and `int` size limits. For SYCL/Level Zero identity,
   reject ambiguous cliques that mix a root device with one of its subdevices or
   contain duplicate/overlapping subdevice identities. For supported tile-only
   masks, log and compare UUID, root/subdevice flag, and subdevice id.

   Keep conservative Intel defaults for unclassified all-Intel multi-GPU
   cliques initially, but warn when the clique is mixed integrated/discrete,
   mixed product family, or otherwise outside the tested homogeneous classes.

   Main files:
   `xla/backends/gpu/collectives/oneccl_collectives.cc`,
   `xla/stream_executor/sycl/sycl_gpu_runtime.{h,cc}`,
   `xla/stream_executor/sycl/sycl_device_description.cc`,
   `xla/backends/gpu/collectives/gpu_cliques.cc`.

10. Improve diagnostics around transport, routing, versions, and rejected
    configs.

    Failures should name the effective `CCL_ATL_TRANSPORT`, `CCL_ATL_SHM`,
    `FI_PROVIDER`, `CCL_PROCESS_LAUNCHER`, `CCL_LOCAL_SIZE`, and
    `CCL_LOCAL_RANK` values for local oneCCL bootstrap. P2P rejections should
    include HLO name, channel/communication id, current rank, source/target
    ranks, selected route, transport, and reason.

    Communicator initialization logs should include XLA ordinal, global rank,
    SYCL backend, device name, PCI bus id, Level Zero UUID/subdevice id,
    integrated/discrete flag, queue generation, native context fingerprint,
    legacy C++ oneCCL version, v2 wrapper version, and build macros where
    available.

### P2: performance and future feature work after the correctness boundary

11. Add all-to-all and all-gather route guards without changing semantics.

    Keep XLA's oneCCL all-gather as fixed-size only until an explicit
    allgatherv wrapper exists. For all-to-all, diagnose pointer-vector lowering,
    rank-count limits, alignment-sensitive ARC/low-latency paths, and in-place
    rejection. This is mostly about making restricted oneCCL selector behavior
    visible rather than changing the default route.

12. Make registered memory support honest.

    In the reviewed legacy C++ route, oneCCL `buffer_register` returns a null
    handle and deregistration is empty. XLA should either report registered
    memory as unsupported/no-op for oneCCL or put it behind a feature/version
    check with precise diagnostics. Deduplicate registrations by allocation and
    communicator before relying on future real handles.

13. Track upstream oneCCL fixes and revisit disabled paths only with version
    gates.

    Future work may re-enable async futures, native SYCL event integration,
    oneCCL local SYCL P2P, oneCCL groups, registered memory, or symmetric memory
    only after upstream behavior is documented, source-verified, and tied to a
    oneCCL version/build guard.

## Recommended implementation order

1. Add tests and seams for failures that currently risk hanging:
   cancellation-aware KVS, first-rank failure cleanup, pending clique
   cancellation, abort while waiting, rank prevalidation, and P2P rejection.

2. Implement the lifecycle fixes: KVS failure/cancel state, first-error
   cancellation, pending-clique stale cancellation, split pending-token
   lifetime, communicator operation state, and safer abort/quarantine behavior.

3. Fix stream lifetime and identity: per-stream oneCCL stream resource,
   launch-time device/context/in-order checks, and stream destroy/recreate
   stress coverage.

4. Split env defaults and close init-order holes, including v2 allocation
   gating. Add diagnostics for partial overrides and local OFI SHM failures.

5. Add validation guards before launch: dtype/reduction, pointer kind,
   zero-count, count overflow, aliasing, all-to-all in-place, P2P shape, and
   symmetric-memory kernel ABI rejection.

6. Make versioning reproducible: immutable v2 pin, version manifest, v1/v2 pair
   check, build macro logging, and API-surface boundary enforcement.

7. Fill the integration matrix: same-stream and cross-stream ordering tests,
   local D2D collective-permute route tests, BF16 threshold tests, all-to-all
   diagnostics, subdevice visibility tests, and guarded hardware tests.

## Changes not to make yet

- Do not make oneCCL communicator futures truly async until event ownership,
  cancellation, abort, teardown, native SYCL output events, and non-OFI behavior
  are designed and tested.
- Do not use `ccl::group_start/end` as XLA's grouped collective primitive.
  Grouped events are not a waitable completion object in the reviewed source,
  explicit dependencies are unsupported for groups, and OFI grouped P2P is not
  a safe route.
- Do not reroute local SYCL collective-permute through oneCCL send/recv by
  default. Keep XLA's D2D path and host-sync workaround until oneCCL P2P is
  proven safe for XLA's ordering model.
- Do not replace `BarrierAfterExecutable()` with oneCCL barrier alone.
- Do not pass `ccl::window*` to XLA device kernels or enable oneCCL symmetric
  kernel paths until oneCCL exposes a stable device ABI.
- Do not treat user environment overrides as equivalent to a supported XLA
  configuration. Log and validate them against the exact oneCCL version and
  transport policy.

## Cross-note dependency map

- Lifecycle and cancellation are prerequisites for any future async completion
  or safe P2P fallback.
- Stream lifetime and queue identity are prerequisites for trusting the current
  synchronous `event.wait()` completion boundary.
- Env/init policy is prerequisite to reproducible communicator creation,
  threshold behavior, and transport diagnostics.
- P2P validation, local D2D routing, and collective operation validation should
  land before broadening oneCCL coverage for ragged all-to-all, send/recv, or
  collective-permute.
- Version pinning and the manifest should land before any oneCCL upgrade or
  upstream-fix-based relaxation of the guards above.
