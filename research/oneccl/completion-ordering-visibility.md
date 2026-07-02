# Completion, queue ordering, barrier, and memory visibility

## Scope

This reviews XLA's oneCCL completion boundary for local Intel GPU collectives:
`WaitForOnecclEvent()`, `LaunchOnecclAndWait()`,
`OnecclCommunicator::LaunchOnStream()`, `ToOnecclStream()`, streamed oneCCL
`Barrier()`, SYCL `Stream`/`Event` ordering, and the local
collective-permute peer-copy special case. The review is static source and
documentation review only; no tests or experiments were run.

oneCCL source review used `~/github/uxlfoundation/oneCCL`, starting
from `map.md`. That map shows that the relevant implementation is the legacy
C++ library under `deps/libccl`, with v2 dispatch and the legacy plugin as the
adapter layer. `research/oneccl/intel-map.md` was not present in this checkout.

## XLA needs

For a oneCCL call launched on an XLA `se::Stream`, XLA needs:

- work already enqueued on that stream to happen before oneCCL reads its input
  buffers;
- oneCCL writes to be visible to later kernels and copies on that stream;
- `ccl::event::wait()` to be a host-visible full completion boundary for the
  device work when XLA returns a completed `Future`;
- any cross-stream producer or consumer dependency to be materialized in SYCL
  queue/event terms before or after the oneCCL call; and
- barrier-after-executable to wait for device work and memory accesses, not only
  for host code to rendezvous.

For local collective-permute/P2P paths, XLA also needs to avoid cross-device
SYCL event waits that can stall. XLA already has a local SYCL special case in
`collective_permute_thunk.cc` that uses host synchronization around peer-copy
protocol boundaries rather than waiting on another device's SYCL event.

## oneCCL behavior

XLA currently waits immediately. `WaitForOnecclEvent()` calls
`ccl::event::wait()` and comments that XLA marks the collective complete only
after the returned oneCCL event is complete
(`xla/backends/gpu/collectives/oneccl_communicator.cc:77`). All regular
collectives go through `LaunchOnecclAndWait()` or explicitly wait each send/recv
event in `LaunchCollectivePermute()`, and `Execute()` returns an already-ready
`Future` from the synchronous status.

The same-stream case is supported by the current XLA and oneCCL sources. XLA's
SYCL stream pool creates queues with `sycl::property::queue::in_order`
(`xla/stream_executor/sycl/sycl_gpu_runtime.cc:240` and `:309`), and
`ToOnecclStream()` wraps the same native `sycl::queue` in `ccl::create_stream`
(`oneccl_communicator.cc:126`). oneCCL preserves the in-order property for its
worker queues (`deps/libccl/src/common/stream/stream.cpp:48`) and tests stream
ordering with `is_queue_in_order()` (`deps/libccl/src/coll/coll_util.cpp:644`).
The v2 legacy plugin also rejects out-of-order queues as unsupported
(`plugins/legacy/ccl_legacy.cpp:209`).

oneCCL inserts a front queue barrier for in-order streams before collectives:
`deps/libccl/src/coll/coll.cpp:188` says the barrier synchronizes the user's
in-order queue with oneCCL's out-of-order queue so oneCCL does not execute
before user tasks are complete. Later, for asynchronous SYCL/topology paths,
oneCCL sets the request native event to a barrier on the user queue dependent on
the internal sync event (`coll.cpp:460`). `host_event_impl::wait()` waits the
oneCCL request through `ccl_wait_impl()` and then waits the native SYCL event
when present (`deps/libccl/src/common/event/impls/host_event.cpp:103`). This is
the key implementation reason the current host-blocking XLA path can treat
`event.wait()` as device completion.

The official oneCCL spec also supports this model at the API level: operations
return an event for operation progress, `event::wait()` blocks for operation
completion, and `event::get_native()` returns a native object signaled when the
operation completes. The spec separately notes that support for input event
handling is optional, which is important because XLA does not pass oneCCL event
dependency vectors for cross-stream ordering. See:

- https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/operation_progress
- https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/main_objects

The current behavior is still conditional. XLA's pinned build enables
`CCL_ENABLE_SYCL_INTEROP_EVENT`
(`third_party/oneccl/oneccl_v1.BUILD:144`), and oneCCL documents
`CCL_SYCL_OUTPUT_EVENT=1` as the default that enables retrieving a SYCL output
event from the oneCCL event:
https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-sycl-output-event.
XLA does not currently retrieve the native event, so this is not required for
the current synchronous wait path, but it becomes required if XLA later tries to
represent oneCCL completion as a StreamExecutor event instead of blocking the
host.

oneCCL's barrier is a communicator operation, not a replacement for XLA's
module-exit stream synchronization. The public header says barrier completes
only after all ranks have called it and returns a `ccl::event`
(`deps/libccl/include/oneapi/ccl/api_functions.hpp:1557`). The implementation
may run a SYCL barrier kernel or fall back to scheduled barrier work
(`deps/libccl/src/coll/coll.cpp:1624`, `:1696`). XLA's
`BarrierAfterExecutable()` currently blocks the main stream on the host before
entering a host rendezvous (`xla/service/gpu/gpu_executable.cc:960`), which is a
different and stronger local stream-completion boundary.

The oneCCL local SYCL P2P paths do use cross-queue/cross-device `sycl::event`
objects. `send_sycl.cpp` waits on receiver readiness events and uses them as
memcpy dependencies; `recv_sycl.cpp` stores receiver-ready and sender-copy
events in shared process state and submits waits on them. The helper path also
submits `host_task`s and immediately waits on their SYCL events for P2P
handshakes (`deps/libccl/src/coll/algorithms/utils/sycl_coll_base.cpp:1249` and
`:1318`). This is very close to the hazard XLA already avoids in
`collective_permute_thunk.cc:582`, where the comment says SYCL cross-device
event barriers can stall.

## Must fix

- Keep treating `ccl::event::wait()` as the oneCCL completion boundary only for
  the current synchronous XLA model. This is a no-change decision for the
  current code: same in-order XLA queue, oneCCL stream wrapping that queue, and
  immediate host wait. Do not replace this with enqueue-only completion,
  completed futures before `event.wait()`, or oneCCL group completion.

- Materialize cross-stream dependencies in XLA before and after oneCCL launches.
  Same-stream ordering does not need extra oneCCL dependency vectors, because
  the XLA queue is in-order and oneCCL inserts a front barrier. Cross-stream
  producer or consumer cases must use StreamExecutor event/stream waits
  (`SyclStream::RecordEvent()`, `SyclStream::WaitFor()`,
  `SyclEvent::WaitStreamOnEvent()`) so the oneCCL launch queue sees the producer
  completion and the consumer queue sees oneCCL completion. Passing no deps to
  oneCCL is not a cross-stream ordering contract.

- Preserve the local SYCL collective-permute peer-copy special case and avoid
  routing local SYCL P2P through oneCCL send/recv paths that exchange
  cross-device `sycl::event`s. This is a correctness/reliability no-change
  decision until the oneCCL P2P implementation is changed upstream or XLA has
  explicit validation that those waits do not stall.

- Do not replace `BarrierAfterExecutable()` with a streamed oneCCL barrier alone.
  The existing XLA path blocks the stream before host rendezvous. A oneCCL
  barrier may be useful as a communicator synchronization operation, but the
  oneCCL docs reviewed here do not state a broad device-memory visibility
  contract that would make it a drop-in replacement for XLA's module-exit
  stream completion.

## Should fix

- Add an in-order queue assertion or diagnostic in `ToOnecclStream()`. XLA's
  current stream pool creates in-order queues, and oneCCL rejects out-of-order
  queues in the legacy plugin, but failing early in XLA would make the required
  contract explicit.

- Document the completion assumptions next to `WaitForOnecclEvent()` and
  `LaunchOnStream()`: same SYCL queue, in-order queue, oneCCL event waited on
  the host, and no implicit cross-stream dependencies. This is fragile enough to
  deserve local comments because future async work will be tempted to return an
  unready `Future` or use `event.get_native()`.

- If XLA later wants async oneCCL completion, gate that work on native SYCL
  output event support. The pinned build has `CCL_ENABLE_SYCL_INTEROP_EVENT`,
  and oneCCL defaults `CCL_SYCL_OUTPUT_EVENT=1`, but an async integration should
  detect or reject `CCL_SYCL_OUTPUT_EVENT=0` rather than silently losing the
  native device event needed for stream dependency plumbing.

- Add barrier-specific documentation in XLA distinguishing
  `OnecclCommunicator::Barrier()` from `BarrierAfterExecutable()`. The former is
  a oneCCL communicator operation launched on a stream and waited by event; the
  latter is a stream host sync plus local host rendezvous for module teardown
  and symmetric-buffer safety.

## Could fix

- Version-gate a future attempt to re-enable oneCCL local SYCL P2P only after
  upstream removes or proves safe the cross-device event waits in
  `send_sycl.cpp` and `recv_sycl.cpp`.

- Expose oneCCL native SYCL events through a StreamExecutor event wrapper for
  overlap work. This should wait for the host-asynchrony design because the
  current integration intentionally returns completed futures.

- Add a short developer note in the oneCCL integration docs explaining why XLA
  uses oneCCL for collectives but keeps a custom local SYCL collective-permute
  peer-copy path.

## Affected files/call sites

XLA:

- `xla/backends/gpu/collectives/oneccl_communicator.cc`:
  `WaitForOnecclEvent()`, `LaunchOnecclAndWait()`, `ToOnecclStream()`,
  `OnecclCommunicator::Barrier()`, `LaunchCollectivePermute()`,
  `LaunchSend()`, `LaunchRecv()`, `LaunchOnStream()`, and `Execute()`.
- `xla/stream_executor/sycl/sycl_gpu_runtime.cc`: in-order queue creation,
  `SyclStreamSynchronize()`, and `SyclSubmitBarrierEvent()`.
- `xla/stream_executor/sycl/sycl_stream.cc`: `RecordEvent()`, `WaitFor()`, and
  stream completion behavior.
- `xla/stream_executor/sycl/sycl_event.cc`: `WaitStreamOnEvent()` and
  `SyclEvent::Wait()`.
- `xla/backends/gpu/runtime/collective_permute_thunk.cc`:
  `RunPeerAccessPermute()` local SYCL host-sync path.
- `xla/service/gpu/gpu_executable.cc`: `BarrierAfterExecutable()` and the
  post-executable barrier request path.
- `third_party/oneccl/workspace.bzl`, `third_party/oneccl/oneccl_v1.BUILD`,
  `third_party/oneccl/oneccl_v2.BUILD`: oneCCL v1/v2 pins and SYCL interop
  event build define.

oneCCL:

- `map.md`: v2 plugin shape and `deps/libccl` implementation routes.
- `plugins/legacy/ccl_legacy.cpp`: v2 legacy plugin stream handling and
  in-order queue requirement.
- `deps/libccl/include/oneapi/ccl/event.hpp` and
  `deps/libccl/include/oneapi/ccl/api_functions.hpp`: event and barrier API.
- `deps/libccl/src/common/event/impls/host_event.cpp`,
  `deps/libccl/src/common/event/impls/native_event.cpp`,
  `deps/libccl/src/common/request/request.{hpp,cpp}`,
  `deps/libccl/src/exec/exec.cpp`: request and event wait completion.
- `deps/libccl/src/common/stream/stream.cpp`,
  `deps/libccl/src/coll/coll_util.cpp`,
  `deps/libccl/src/common/utils/sycl_utils.cpp`: stream wrapping, in-order
  detection, output-event selection, and SYCL barrier helpers.
- `deps/libccl/src/coll/coll.cpp` and `deps/libccl/src/sched/sched.cpp`:
  front queue barrier, output event creation, and barrier implementation.
- `deps/libccl/src/coll/algorithms/send/sycl/send_sycl.cpp`,
  `deps/libccl/src/coll/algorithms/recv/sycl/recv_sycl.cpp`,
  `deps/libccl/src/coll/algorithms/utils/sycl_coll_base.cpp`: local SYCL P2P
  event exchange and host-task handshakes.
- `deps/libccl/src/sched/entry/ze` and `deps/libccl/src/sched/ze`: Level Zero
  event and command-list scheduler machinery.

## Evidence to cite

- XLA waits oneCCL events immediately:
  `xla/backends/gpu/collectives/oneccl_communicator.cc:77` and `:105`.
- XLA wraps the native SYCL queue in a oneCCL stream:
  `oneccl_communicator.cc:126`.
- XLA avoids oneCCL grouping because grouped events are not waitable:
  `oneccl_communicator.cc:364`.
- XLA SYCL queues are in-order:
  `xla/stream_executor/sycl/sycl_gpu_runtime.cc:240` and `:309`.
- XLA stream/event waits submit SYCL barriers:
  `xla/stream_executor/sycl/sycl_stream.cc:190` and
  `xla/stream_executor/sycl/sycl_event.cc:62`.
- XLA's local SYCL collective-permute path avoids cross-device event waits:
  `xla/backends/gpu/runtime/collective_permute_thunk.cc:582`.
- Barrier-after-executable blocks the stream before rendezvous:
  `xla/service/gpu/gpu_executable.cc:960`.
- oneCCL event API defines blocking wait and native event retrieval:
  `deps/libccl/include/oneapi/ccl/event.hpp:69`.
- oneCCL front-barriers in-order queues before collectives:
  `deps/libccl/src/coll/coll.cpp:188`.
- oneCCL sets a native event from the sync event or a user-queue barrier:
  `deps/libccl/src/coll/coll.cpp:442`.
- `host_event_impl::wait()` waits the request and native event:
  `deps/libccl/src/common/event/impls/host_event.cpp:103`.
- oneCCL barrier API and implementation:
  `deps/libccl/include/oneapi/ccl/api_functions.hpp:1557`,
  `deps/libccl/src/coll/coll.cpp:1624`, and `:1696`.
- oneCCL local SYCL P2P exchanges cross-device/cross-queue events:
  `deps/libccl/src/coll/algorithms/send/sycl/send_sycl.cpp:230` and
  `deps/libccl/src/coll/algorithms/recv/sycl/recv_sycl.cpp:231`.
- oneCCL official docs for event progress and native event completion:
  https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/operation_progress
- oneCCL official docs for streams, events, and optional input event handling:
  https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/spec/main_objects
- oneCCL env docs for `CCL_SYCL_OUTPUT_EVENT`:
  https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-sycl-output-event
- SYCL queue docs for in-order queues and queue wait:
  https://github.khronos.org/SYCL_Reference/iface/queue.html
- Level Zero synchronization docs for events/fences and memory dependencies:
  https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#synchronization-primitives

## Test coverage plan

- Same-stream producer to oneCCL to consumer: launch a producer kernel that
  writes collective input, run oneCCL all-reduce/all-gather/reduce-scatter on
  the same XLA stream, then launch a consumer kernel on the same stream that
  validates the collective output. The test should not insert host sync between
  the producer, oneCCL call, and consumer.

- Same-stream barrier sequencing: enqueue a producer kernel, call
  `OnecclCommunicator::Barrier()` on the same stream, then enqueue a consumer
  kernel. The test should verify that barrier completion does not allow the
  consumer to observe stale local state and should cover both single-rank and
  multi-rank local communicators where available.

- Cross-stream producer dependency: enqueue the producer on an auxiliary stream,
  record an event, make the oneCCL launch stream wait on it, run oneCCL, then
  validate from a consumer on the oneCCL stream. This verifies that XLA-side
  stream waits, not oneCCL empty deps, provide cross-stream input ordering.

- Cross-stream consumer dependency: run oneCCL on the launch stream, record a
  completion event after the oneCCL call, make a consumer stream wait on it, and
  validate the output from the consumer stream. This covers oneCCL to later
  non-launch-stream visibility.

- Barrier-after-executable: add a module that requests the post-executable
  barrier after writing/reading symmetric or collective buffers, and verify the
  teardown path keeps the existing `BlockHostUntilDone()` before host rendezvous.

- Local collective-permute SYCL path: cover producer to local peer-copy
  collective-permute to consumer for same-stream and cross-stream dependency
  cases, and assert the SYCL host-sync path remains selected for local peer-copy
  rather than oneCCL send/recv.

- Configuration guards: add tests that fail or skip clearly if an out-of-order
  SYCL queue reaches oneCCL or if a future async native-event integration is run
  with `CCL_SYCL_OUTPUT_EVENT=0`.

## Rollout risk

The lowest-risk rollout is mostly documentation, diagnostics, and tests, because
the current same-stream synchronous wait path is supported by both source review
and the oneCCL event API. The main user-visible risk is env drift: users can set
oneCCL variables such as `CCL_SYCL_OUTPUT_EVENT=0`, and future async event
plumbing would need to reject that configuration or fall back to host waits.

The highest correctness risk is accidentally weakening the completion boundary:
returning an XLA `Future` before `ccl::event::wait()`, switching to oneCCL group
completion, or relying on oneCCL input dependency vectors for XLA cross-stream
ordering would violate assumptions in this review. The second major risk is
reusing oneCCL local SYCL P2P paths for collective-permute and reintroducing
cross-device SYCL event waits that XLA already avoids.

Performance risk is that preserving immediate waits leaves overlap on the table.
That is intentional for this topic: async oneCCL completion should be designed
with native SYCL event ownership, cancellation, and teardown semantics rather
than introduced as a queue-ordering change.
