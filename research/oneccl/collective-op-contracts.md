# Collective operation contracts

## Scope

This note covers XLA GPU collective operation contracts for the oneCCL backend: all-reduce, broadcast, reduce-scatter, all-gather, all-to-all, collective-permute, explicit send/recv, and barrier. It is based on static source review only.

The XLA entry points reviewed are `xla/backends/gpu/collectives/oneccl_communicator.{h,cc}` and the GPU runtime thunk implementations for all-reduce, broadcast, reduce-scatter, all-gather, all-to-all, collective-permute, send, recv, and barrier. The oneCCL source routes reviewed start from `~/github/uxlfoundation/oneCCL/map.md`, then the public API headers, the v2 legacy adapter, legacy collective parameter validation, collective selectors, SYCL algorithm implementations, P2P implementations, and barrier implementations under `deps/libccl`.

`research/oneccl/intel-map.md` was not present, so no Intel-specific map was available for this topic.

## XLA needs

XLA needs a stable per-op contract that is stricter than "the call compiles":

- Reduction collectives must preserve HLO reduction semantics for every accepted dtype and reduction kind, including BF16, FP16, integer, predicate, and complex inputs.
- Count parameters must match oneCCL's per-rank contracts and must not silently reinterpret unsupported layouts.
- In-place and aliased buffers must either match the exact oneCCL in-place form or be rejected/de-aliased before launch.
- Broadcast must use the correct root rank in the oneCCL communicator, not just the first global replica id.
- All-gather and all-to-all must distinguish fixed-size contracts from uneven/vector contracts.
- P2P-based collectives must have explicit peer matching, launch ordering, self-edge behavior, and cancellation expectations, because oneCCL send/recv has no XLA-visible tag parameter.
- Barrier must be treated as stream-ordering plus a waited oneCCL event, not as a data collective.

## oneCCL behavior

- XLA maps oneCCL calls in `OnecclCommunicator::Launch*`. `ToOnecclDataType` maps `C64` to `float32`, `C128` to `float64`, and `ToOnecclCount` doubles the element count for complex types. That representation is only semantically safe for additive complex reductions, not for complex product/min/max.
- XLA maps `ReductionKind::{SUM, PRODUCT, MIN, MAX}` to oneCCL predefined reductions in `ToOnecclReduction`. There is no oneCCL-specific dtype/reduction compatibility check before launch; XLA's generic collective gate in `collective_thunk.cc` is still NCCL-shaped and allows complex reduction operands.
- oneCCL legacy datatype support includes `int16`/`uint16` in `deps/libccl/include/oneapi/ccl/types.hpp`, while the v2 C API datatype enum in `include/oneapi/ccl/v2/types.h` does not. XLA currently uses the legacy C++ API via `oneapi/ccl.hpp`.
- oneCCL public API contracts in `include/oneapi/ccl.h` define all-reduce and broadcast in-place as `sendbuff == recvbuff`; reduce-scatter in-place as `recvbuff == sendbuff + rank * recvcount`; all-gather in-place as `sendbuff == recvbuff + rank * sendcount`; all-to-all as a fixed count per peer. The reviewed public header does not define an all-to-all in-place contract.
- `coll_param::validate` in `deps/libccl/src/coll/coll_param.cpp` checks reduce-scatter as `send_count == recv_count * comm->size()`. XLA's `RunReduceScatter` divides the source element count by `comm.NumRanks()` before calling `LaunchReduceScatter`, so the basic per-rank count contract is aligned.
- XLA `RunAllGather` calls fixed-size `ccl::allgather(send, recv, count, dtype, ...)`. The v2 legacy adapter implements v2 all-gather through legacy `ccl::allgatherv` with uniform recv counts. oneCCL SYCL allgatherv code rejects unequal recv counts on the fast path, so uneven gather must not reach this wrapper.
- XLA `LaunchAllToAll` chooses contiguous `ccl::alltoall(base_send, base_recv, count, ...)` when per-peer buffers are contiguous, otherwise the pointer-vector overload with one pointer per rank. oneCCL marks vector-buffer all-to-all with `is_vector_buf=1`; selectors restrict available algorithms for vector buffers, and SYCL kernel selection rejects vector buffers in `can_use_sycl_kernels`.
- oneCCL all-to-all SYCL paths contain alignment and scaling assumptions: ARC requires `(count * dtype_size)` alignment to the line size and power-of-two world size, and several kernels allocate arrays sized by `MAX_NODE_RANKS` from `deps/libccl/src/coll/algorithms/utils/consts.hpp`.
- oneCCL all-reduce, reduce-scatter, all-gatherv, and all-to-all SYCL implementations use threshold-sensitive algorithm selection. Relevant knobs include `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD`, `CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD`, `CCL_SYCL_ALLGATHERV_LL_THRESHOLD`, and all-to-all low-latency/chunk thresholds from `deps/libccl/src/common/env`.
- oneCCL P2P send/recv matching is peer/count/dtype based in the public API. There is no XLA-provided tag in `ccl::send`/`ccl::recv`. SYCL P2P creates internal tags from communicator and scheduler ids, and self-communication uses process-global FIFO queues in `send_sycl.cpp`/`recv_sycl.cpp` with comments assuming one send per recv.
- oneCCL public send/recv docs warn that operations may block GPU progress and that multiple sends/recvs requiring concurrent progress should be enclosed in `group_start`/`group_end`. XLA's `GroupExecute` intentionally does not use oneCCL groups because oneCCL group event waiting is not suitable for the current XLA async model; each oneCCL call is waited directly.
- oneCCL barrier is available through legacy `ccl::barrier`; SYCL barrier may run as a GPU barrier or CPU host task, and fallback barrier uses scheduler send/recv/ATL paths. XLA waits the returned event in `OnecclCommunicator::Barrier`.

## Must fix

- Add oneCCL-specific reduction/type validation before launch. At minimum, reject complex `PRODUCT`, `MIN`, and `MAX` because XLA's current complex mapping reduces real/imag lanes independently; only complex `SUM` has the intended arithmetic semantics under the doubled-count mapping. The validator should also make accepted BF16/FP16/integer/predicate reductions explicit, because oneCCL SYCL fast paths and ESIMD paths support different dtype subsets depending on algorithm selection thresholds.
- Validate or de-alias in-place/overlapping buffers per collective. Accepted aliases should be limited to oneCCL's documented forms: all-reduce and broadcast exact `send == recv`, reduce-scatter rank-offset receive, and all-gather rank-offset send. All-to-all should reject or copy through non-aliased scratch for in-place/overlap because the reviewed public contract does not define all-to-all in-place behavior and oneCCL SYCL all-to-all has unsupported in-place paths.
- Prevalidate collective-permute source-target pairs before converting them to XLA's single optional source/target map. Fanout, multiple sources for one target, duplicate pairs, and self edges need explicit handling. Otherwise invalid or unsupported HLO shapes can be collapsed or launched as tagless P2P with silent peer loss or mismatched waits.
- Do not rely on standalone oneCCL P2P for cycles or same-rank communication until launch ordering is proven safe. XLA currently cannot use oneCCL group semantics for collectives, while oneCCL docs require grouping for multiple concurrent nonblocking P2P and the SYCL self path uses global FIFO queues. For oneCCL fallback, reject such shapes or route self/local edges through explicit local copy/memzero behavior with deterministic ordering.

## Should fix

- Add an XLA-side broadcast root assertion tying `RankId(0)` to the first replica id in each clique. `RunCollectiveBroadcast` always passes root rank 0, which is correct only if communicator rank ordering is built from the replica group in that order.
- Make the fixed-size all-gather contract explicit in oneCCL lowering. Uneven gather should either be rejected before this path or lowered to a future allgatherv wrapper with oneCCL-specific tests.
- Add preflight diagnostics for oneCCL all-to-all when XLA chooses the pointer-vector path. Noncontiguous per-rank buffers are semantically supported by the API but lose the contiguous SYCL fast path and enter more restricted selector behavior.
- Guard all-to-all rank-count and alignment-sensitive paths. At least report clear errors or fallback decisions for communicator sizes beyond oneCCL kernel assumptions such as `MAX_NODE_RANKS`, and for ARC/low-latency alignment constraints.
- Replace NCCL-specific validation error text in generic GPU collective checks with backend-aware wording, or add oneCCL-specific errors near the oneCCL lowering boundary.
- Document that reduce-scatter count handling is currently correct only for divisible, uniform per-rank outputs; dtype/reduction validation should share the all-reduce validator.
- Keep barrier ordering assumptions tied to the broader oneCCL stream/event contract. If topic 1 does not fully cover it, add a oneCCL barrier note that XLA must wait the returned barrier event before exposing completion.

## Could fix

- Add a future oneCCL allgatherv/alltoallv abstraction if XLA needs uneven gather or truly variable per-peer all-to-all contracts on this backend.
- Add a debug or tuning flag to force contiguous all-to-all only, useful when diagnosing pointer-vector fallback performance or selector coverage.
- Track the legacy C++ API versus v2 C API datatype mismatch for `int16`/`uint16` so a future v2 migration does not silently drop supported legacy routes.
- Add structured logging for threshold-selected oneCCL algorithms when debug logging is enabled, especially for BF16/FP16 all-reduce and reduce-scatter.

## Affected files/call sites

- `xla/backends/gpu/collectives/oneccl_communicator.h`
- `xla/backends/gpu/collectives/oneccl_communicator.cc`
- `xla/backends/gpu/runtime/collective_thunk.cc`
- `xla/backends/gpu/runtime/all_reduce_thunk.cc`
- `xla/backends/gpu/runtime/collective_broadcast_thunk.cc`
- `xla/backends/gpu/runtime/all_gather_thunk.cc`
- `xla/backends/gpu/runtime/all_to_all_thunk.cc`
- `xla/backends/gpu/runtime/collective_permute_thunk.cc`
- `xla/backends/gpu/runtime/send_thunk.cc`
- `xla/backends/gpu/runtime/recv_thunk.cc`
- `xla/backends/gpu/runtime/p2p_thunk_common.{h,cc}`
- `third_party/oneccl/oneccl_v1.BUILD`
- `third_party/oneccl/oneccl_v2.BUILD`

## Evidence to cite

- XLA oneCCL wrapper: `xla/backends/gpu/collectives/oneccl_communicator.cc`, especially `ToOnecclCount`, `ToOnecclDataType`, `ToOnecclReduction`, `LaunchAllReduce`, `LaunchBroadcast`, `LaunchReduceScatter`, `LaunchAllGather`, `LaunchAllToAll`, `LaunchCollectivePermute`, `LaunchSend`, `LaunchRecv`, `Barrier`, and `GroupExecute`.
- XLA all-reduce/reduce-scatter: `xla/backends/gpu/runtime/all_reduce_thunk.cc`, `RunAllReduce`, `RunReduceScatter`, and `CheckImplementableInst`.
- XLA broadcast: `xla/backends/gpu/runtime/collective_broadcast_thunk.cc`, `RunCollectiveBroadcast`.
- XLA all-gather: `xla/backends/gpu/runtime/all_gather_thunk.cc`, `RunAllGather`.
- XLA all-to-all: `xla/backends/gpu/runtime/all_to_all_thunk.cc`, `CheckImplementable`, `RunAllToAll`.
- XLA collective-permute and P2P: `xla/backends/gpu/runtime/collective_permute_thunk.cc`, `RunCollectivePermute`, `RunPeerAccessPermute`, `GetP2PConfig`; `xla/backends/gpu/runtime/send_thunk.cc`; `xla/backends/gpu/runtime/recv_thunk.cc`; `xla/backends/gpu/runtime/p2p_thunk_common.cc`.
- oneCCL repository map: `~/github/uxlfoundation/oneCCL/map.md`.
- oneCCL public API contracts: `~/github/uxlfoundation/oneCCL/include/oneapi/ccl.h`.
- oneCCL v2 datatype/reduction enums: `~/github/uxlfoundation/oneCCL/include/oneapi/ccl/v2/types.h`.
- oneCCL v2 legacy forwarding: `~/github/uxlfoundation/oneCCL/plugins/legacy/ccl_legacy.cpp`.
- oneCCL legacy datatypes: `~/github/uxlfoundation/oneCCL/deps/libccl/include/oneapi/ccl/types.hpp` and `deps/libccl/src/common/datatype/datatype.cpp`.
- oneCCL parameter validation and in-place checks: `~/github/uxlfoundation/oneCCL/deps/libccl/src/coll/coll_param.cpp` and `coll_check.cpp`.
- oneCCL collective selectors: `deps/libccl/src/coll/selection/selector_allreduce.cpp`, `selector_reduce_scatter.cpp`, `selector_allgather.cpp`, `selector_allgatherv.cpp`, `selector_alltoall.cpp`, `selector_barrier.cpp`, and `selection.cpp`.
- oneCCL SYCL selection and thresholds: `deps/libccl/src/coll/algorithms/utils/sycl_selection.cpp`, `deps/libccl/src/common/env/env.cpp`, and `deps/libccl/src/common/env/vars_experimental.hpp`.
- oneCCL SYCL algorithms: `deps/libccl/src/coll/algorithms/allreduce/sycl`, `reduce_scatter/sycl`, `allgatherv/sycl`, `alltoall/sycl`, `send/sycl`, `recv/sycl`, and `barrier/sycl`.
- oneCCL rank/kernel constants: `~/github/uxlfoundation/oneCCL/deps/libccl/src/coll/algorithms/utils/consts.hpp`.

## Test coverage plan

- Add focused XLA unit tests for oneCCL dtype/reduction validation: complex `SUM` accepted, complex `PRODUCT`/`MIN`/`MAX` rejected, BF16/FP16 reductions accepted only for supported reductions, integer and predicate behavior explicit.
- Add aliasing contract tests around each oneCCL launch wrapper: all-reduce/broadcast exact in-place allowed; reduce-scatter and all-gather rank-offset aliases allowed; arbitrary overlap rejected; all-to-all in-place rejected or copied through scratch.
- Add reduce-scatter tests for divisible and nondivisible source element counts, plus dtype/reduction edge cases shared with all-reduce.
- Add broadcast tests that verify root rank mapping for replica groups where the global source id is not numerically zero but is communicator rank 0.
- Add all-gather tests proving only fixed-size all-gather reaches the oneCCL path; add an uneven-shape rejection test until allgatherv is implemented.
- Add all-to-all tests for split and non-split forms, contiguous fast-path detection, noncontiguous pointer-vector lowering, per-peer count uniformity, alias rejection, and rank/alignment guard diagnostics.
- Add collective-permute tests for cycles, self edges, duplicate targets, fanout, no-source memzero, no-target skip, and local-copy handling versus oneCCL fallback rejection.
- Add send/recv contract tests for peer matching, count/dtype mismatch rejection, same-rank behavior, and launch ordering diagnostics. These tests should use fakes or static validation; they must not depend on running real oneCCL P2P.
- Add barrier ordering coverage only if not already covered by the oneCCL stream/event topic: producer work before barrier, barrier wait, and consumer visibility after barrier through a fake communicator.
- Add threshold-sensitive validation tests by injecting selector/tuning metadata or environment parsing fakes for BF16/FP16 all-reduce and reduce-scatter. The tests should verify XLA's accepted contract does not change silently when oneCCL thresholds select a different algorithm family.

## Rollout risk

The main risk is that stricter validation can reject programs that currently reach oneCCL and sometimes work. That is preferable for complex non-sum reductions, unsupported P2P shapes, and aliasing forms that oneCCL does not guarantee, because those cases can corrupt data or hang.

Performance risk is concentrated in all-to-all and BF16/FP16 reductions. Adding diagnostics or fallbacks for pointer-vector all-to-all, in-place all-to-all, high rank counts, and threshold-selected algorithms can change which oneCCL path is used. Rollout should start with validation and clear errors, then add optimized safe alternatives only where needed.

Compatibility risk is moderate for future oneCCL v2 migration because v2 C datatype coverage differs from the legacy C++ API XLA currently calls. Keep the accepted XLA contract tied to the actual API in use.
