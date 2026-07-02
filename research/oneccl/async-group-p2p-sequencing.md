# Host asynchrony, grouped collectives, and launch sequencing

## Scope

This note reviews XLA's host-initiated oneCCL path for local Intel GPU
collectives: the completed-`Future` execution model in
`xla/backends/gpu/collectives/oneccl_communicator.cc`, XLA grouped collective
call sites, and point-to-point launch sequencing for collective-permute,
send/recv, and ragged all-to-all. oneCCL source review uses the local
`~/github/uxlfoundation/oneCCL` tree, starting from `map.md`;
`research/oneccl/intel-map.md` was not present in this checkout.

The relevant oneCCL implementation is the legacy C++ library under
`deps/libccl`: XLA pins `@oneccl_v1` to commit
`4ceafd15c03ce46f11eeaf91781a92afebd3cecf` and builds v2 as a thin loader plus
legacy plugin in `third_party/oneccl/{workspace.bzl,oneccl_v1.BUILD,oneccl_v2.BUILD}`.

## XLA needs

XLA needs communicator methods to return a `Future` that accurately represents
collective completion. If the future is ready, downstream XLA code can treat
the collective as complete and errors as already reported. If the future is
made asynchronous, it must own the oneCCL event/request lifetime, surface
`ccl::event::wait()` failures, and interact safely with clique cancellation and
`OnecclCommunicator::Abort()`.

Grouped HLO collectives need all nested launches to be issued in an order that
cannot deadlock and a completion object that becomes ready only after all nested
work is complete. P2P paths additionally need a launch protocol that handles
cycles, fanout, self edges, and missing peer matches without relying on an
unmatched `send` or `recv` completing before its peer has been posted.

## oneCCL behavior

XLA currently waits immediately. `WaitForOnecclEvent()` calls
`ccl::event::wait()`, `LaunchOnecclAndWait()` waits each returned event, and
`OnecclCommunicator::Execute()` returns an already-ready `Future` from the
synchronous status. `GroupExecute()` intentionally does not call
`ccl::group_start/end`; it executes the body directly so each XLA launch can
wait its own event.

oneCCL does not provide a safe grouped event model in this tree.
`ccl::group_end()` is documented in the C++ headers as returning after grouped
operations are enqueued, not necessarily complete
(`deps/libccl/include/oneapi/ccl/api_functions.hpp`). In implementation,
grouped collective calls return placeholder events and store closures in
`coll/group/group.cpp`; `host_event_impl` warns that `wait()`, `test()`, and
`get_native()` are unsupported for collectives inside the group API
(`deps/libccl/src/common/event/impls/host_event.cpp`). `coll.cpp` also logs
that explicit dependencies are unsupported for group calls. This conflicts with
XLA's need for a waitable per-group completion value.

oneCCL also forces synchronous execution for the OFI transport:
`deps/libccl/src/coll/coll.cpp` sets `attr.synchronous = 1` when
`atl_transport == ccl_atl_ofi` due to an asynchronous OFI failure workaround,
then starts the schedule and immediately calls `ccl_wait_impl()`. XLA's local
single-process defaults select OFI/SHM (`CCL_ATL_TRANSPORT=ofi`,
`FI_PROVIDER=shm`), so replacing ready futures with background event waits
would usually not add true launch asynchrony.

P2P is especially fragile. oneCCL's v2 C header says multiple
`onecclSend/onecclRecv` operations that need concurrent progress should be
placed in a group, but the legacy implementation rejects OFI group usage in
selection (`deps/libccl/src/coll/selection/selection.cpp`) and XLA must avoid
the oneCCL group API for completion reasons. The SYCL P2P path also handles
self send/recv with process-global FIFO queues (`q_self_send`, `q_self_recv`)
in `send_sycl.cpp` and `recv_sycl.cpp`, not keyed by communicator, stream, or
datatype.

## Must fix

- Keep oneCCL communicator futures already-completed for now. Do not replace
  `OnecclCommunicator::Execute()` with real async completion until oneCCL
  non-OFI async behavior, event lifetime, cancellation, and communicator
  teardown are explicitly designed. `host_event_impl::cancel()` and
  `native_event_impl::cancel()` throw "not implemented", while
  `Abort()` currently cancels XLA's token and resets the communicator.

- Continue avoiding `ccl::group_start/end` in XLA. This is a no-change decision
  but should be documented in code: grouped oneCCL calls do not return waitable
  events, explicit deps are not supported in groups, and OFI group P2P is not a
  supported route.

- Add oneCCL-specific P2P gating before using `LaunchSend`, `LaunchRecv`, or
  multi-edge `LaunchCollectivePermute` for patterns that require concurrent
  unmatched operations. Current XLA oneCCL `LaunchSend()` and `LaunchRecv()`
  wait immediately. `ragged_all_to_all_thunk.cc` issues `send` then `recv` for
  each peer inside `GroupExecute()`, but oneCCL `GroupExecute()` does not batch
  them. That can deadlock for all-rank send-first patterns.

- Bypass oneCCL for self P2P edges. Use a local stream copy/no-op for
  `peer == current_rank` instead of oneCCL `send/recv`, because oneCCL's
  self-edge FIFO is global and unkeyed.

- Fail early on unsupported or unverifiable oneCCL P2P configurations: cycles
  if the selected oneCCL path cannot enqueue all peers without blocking,
  duplicate/fanout edges not representable by `P2PConfig`, missing peer
  matches, and peer ranks outside the communicator. HLO verifier catches many
  cases for normal scheduled modules, but thunk proto deserialization and
  backend call paths should not be able to hang oneCCL.

## Should fix

- If future oneCCL versions support safe non-OFI async, add an XLA-owned
  non-group batching layer: launch operations, collect individual waitable
  `ccl::event`s, and join them at `GroupExecute()` completion. Do not implement
  this via oneCCL groups unless grouped events become waitable and dependency
  semantics are specified.

- Add diagnostics when oneCCL P2P is rejected or downgraded: include HLO name,
  channel/communication id, current rank, source/target ranks, transport, and
  the reason (`OFI synchronous`, `self edge`, `cycle`, `fanout`, or
  `missing match`).

- Add a oneCCL `GpuCollectives::GroupLaunch()` override only if it can share
  the same XLA-owned event batching and can join all involved communicators.
  The current inherited default just executes the body.

## Could fix

- Track oneCCL changes around `CCL_SYCL_OUTPUT_EVENT`, group event support,
  OFI asynchronous failure fixes, and event cancellation, then revisit the
  async/no-group decisions by version.

- Add XLA-facing comments near `GroupExecute()` and P2P launches explaining why
  oneCCL differs from NCCL/RCCL group launch semantics.

## Affected files/call sites

- XLA: `xla/backends/gpu/collectives/oneccl_communicator.cc`
  (`WaitForOnecclEvent`, `LaunchOnecclAndWait`, `Execute`,
  `GroupExecute`, `GroupExecuteCounted`, `LaunchCollectivePermute`,
  `LaunchSend`, `LaunchRecv`, `Abort`).
- XLA: `xla/backends/gpu/collectives/oneccl_collectives.cc`
  (`ValidateCreateCommunicatorConfig`, oneCCL env defaults and cancellation
  setup).
- XLA grouped call sites:
  `all_reduce_thunk.cc`, `all_gather_thunk.cc`,
  `collective_broadcast_thunk.cc`, `collective_group_thunk.cc`,
  `collective_permute_thunk.cc`, `ragged_all_to_all_thunk.cc`.
- XLA P2P validation/runtime:
  `p2p_thunk_common.{h,cc}`, `send_thunk.cc`, `recv_thunk.cc`,
  `collective_permute_thunk.cc`, `xla/service/hlo_verifier.cc`.
- oneCCL: `deps/libccl/include/oneapi/ccl/api_functions.hpp`,
  `include/oneapi/ccl.h`, `deps/libccl/src/coll/coll.cpp`,
  `deps/libccl/src/coll/group/group.cpp`,
  `deps/libccl/src/common/event/impls/host_event.cpp`,
  `deps/libccl/src/common/event/impls/native_event.cpp`,
  `deps/libccl/src/coll/selection/selection.cpp`,
  `deps/libccl/src/coll/algorithms/{send,recv}/sycl/*.cpp`,
  `deps/libccl/src/coll/algorithms/utils/sycl_pipe_send.cpp`.

## Evidence to cite

- XLA waits immediately and returns ready futures:
  `oneccl_communicator.cc::WaitForOnecclEvent`,
  `LaunchOnecclAndWait`, and `Execute`.
- XLA avoids oneCCL groups by design:
  `oneccl_communicator.cc::GroupExecute` comment.
- oneCCL group API contract:
  `deps/libccl/include/oneapi/ccl/api_functions.hpp::group_end` says enqueue,
  not completion.
- oneCCL grouped events are not composable:
  `host_event_impl::wait/test/get_native` warn for group API;
  `coll.cpp` returns empty/placeholder events and warns explicit deps are not
  supported in group calls.
- OFI disables async:
  `deps/libccl/src/coll/coll.cpp` sets `attr.synchronous = 1` for
  `ccl_atl_ofi` and immediately waits synchronous schedules.
- Event cancellation is unavailable:
  `host_event_impl::cancel()` and `native_event_impl::cancel()` throw.
- P2P group mismatch:
  `include/oneapi/ccl.h` recommends grouping multiple send/recv operations,
  while `selection.cpp` rejects OFI group API and XLA defaults to OFI/SHM.
- Self-edge hazard:
  process-global `q_self_send`/`q_self_recv` in oneCCL SYCL P2P sources.
- Existing XLA validation:
  `hlo_verifier.cc` rejects duplicate collective-permute sources/targets and
  scheduled send/recv cycles/nonmatches, but tests show unscheduled modules and
  pipelined annotations are special cases.

## Test coverage plan

- Unit tests for oneCCL `GroupExecute()` semantics with a fake/wrapped launcher:
  verify grouped HLO collectives do not call `ccl::group_start/end`, errors from
  a nested launch propagate, and a single-buffer `GroupExecuteCounted` remains
  synchronous.

- HLO verifier/thunk-construction tests for P2P configs: duplicate source,
  duplicate target/fanout, cycles, self edges, peer outside communicator,
  missing send/recv match, and thunk proto deserialization that bypasses normal
  HLO parsing.

- Runtime-level oneCCL gating tests using a fake `GpuCommunicator`: ensure
  ragged all-to-all self peers are copied locally or rejected before oneCCL,
  and send/recv cycles or all-rank send-first patterns produce a deterministic
  XLA error rather than entering `LaunchSend()`.

- If async futures are later implemented, add cancellation and async error
  propagation tests: future is not ready before event completion, event wait
  exception becomes the future status, abort prevents new launches, pending
  waiters cannot outlive communicator/request state, and cancellation does not
  report success for unfinished oneCCL work.

## Rollout risk

The safest near-term rollout is conservative: preserve synchronous oneCCL
completion and reject or bypass unsafe oneCCL P2P shapes. This may reduce
overlap and may disable some ragged P2P paths on Intel GPUs, but it avoids
deadlocks and silent self-edge corruption.

User environment overrides are a major risk. `CCL_ATL_TRANSPORT`, `FI_PROVIDER`,
`CCL_SEND`, `CCL_RECV`, `CCL_SYCL_OUTPUT_EVENT`, and worker/thread settings can
change whether oneCCL is synchronous, grouped, or using SYCL P2P paths. XLA
should treat unrecognized async/group-capable env combinations as unsupported
unless explicitly allowlisted by oneCCL version and transport.

Version risk is high because XLA builds a pinned legacy oneCCL plus v2 loader,
while the local source tree and public docs may move. Any future relaxation
should be guarded by runtime version logging and a feature flag, with the
default remaining synchronous until grouped events, cancellation, and OFI async
behavior are verified from source and documented contract.
