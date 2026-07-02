# Pointer, registered memory, and symmetric memory contracts

## Scope

This note covers topic 7 from `research/oneccl/plan.md`: raw pointer legality, zero-count/null-buffer handling, overlapping buffers, complex dtype count conversion, registered-memory lifetime, and symmetric-memory/window contracts for XLA's oneCCL GPU backend.

Review sources were static only. `research/oneccl/intel-map.md` was not present, so the oneCCL route started from `~/github/uxlfoundation/oneCCL/map.md`.

## XLA needs

XLA's oneCCL communicator currently passes `DeviceAddressBase::opaque()` directly into oneCCL collectives and P2P calls from `xla/backends/gpu/collectives/oneccl_communicator.cc`. It also exposes memory-registration wrappers through `OnecclRegisteredMemory::Create` and `OnecclSymmetricMemory::Create`, and `OnecclSymmetricMemory::PackKernelArg` returns a `ccl::window*`.

For correctness, XLA needs an explicit contract for:

- Which pointers are legal for non-zero GPU collectives: SYCL USM device/shared/host pointers, non-USM host/system pointers, null pointers, and wrong-context pointers.
- Which zero-count operations may pass null buffers and which should short-circuit before oneCCL.
- Which aliasing patterns are valid: exact in-place, operation-specific segmented in-place, and invalid partial overlap.
- Whether `ToOnecclCount` can safely multiply complex element counts by two before passing a real oneCCL datatype.
- Whether registered memory is a real performance feature, what alignments and repeated registrations are supported, and how deregistration is ordered against communicator teardown.
- Whether symmetric memory is collective, local-only, same-size/all-rank, and whether any oneCCL object is a stable kernel ABI.

## oneCCL behavior

The v2 C API in `~/github/uxlfoundation/oneCCL/src/api.cpp` checks `comm` for collectives but does not validate send/recv pointers or counts there. Registration and window entry points do reject null `buff` pointers via `ONECCL_CHECK_PTR`, even if size is zero.

The legacy plugin route in `plugins/legacy/ccl_legacy.cpp` decides CPU vs GPU handling with `is_host_pointer(buf)`, implemented with an `msync` probe on the page-aligned address. That makes passing null or unknown pointers through a non-short-circuited zero-count collective unsafe as a frontend contract. GPU-buffer execution expects a `sycl::queue*` stream; oneCCL docs in `docs/config/plugins.md` and `docs/config/examples.md` state that GPU buffers should pass a SYCL queue pointer and examples use USM device allocations.

The legacy C++ validator in `deps/libccl/src/coll/coll_check.cpp::ccl_check_usm_pointers` validates only non-zero buffers from `coll_param.cpp::get_all_non_zero_bufs`. When stream/device/context validation is active, all non-zero buffers must be the same SYCL USM allocation type and must belong to the communicator context. It rejects host USM for GPU/accelerator devices and rejects `unknown`; shared USM is not rejected. Zero-count buffers are intentionally skipped by this validator.

oneCCL has no complex datatype in `deps/libccl/include/oneapi/ccl/types.hpp`. XLA maps `C64` to `float32` and `C128` to `float64` and doubles the count in `ToOnecclCount`.

oneCCL in-place and overlap behavior is operation-specific. `coll_util.cpp::is_allgatherv_inplace` and `is_reduce_scatter_inplace` detect exact segmented in-place layouts, but partial-alias rejection is gated by the `check_inplace_aliasing` environment setting. The SYCL all-to-all path in `coll/algorithms/alltoall/sycl/alltoall_sycl.cpp` explicitly does not support in-place all-to-all in the multi-node path.

Registered memory is not a meaningful backend feature in the reviewed legacy C++ route: `deps/libccl/src/comm/comm.hpp::buffer_register` sets `*handle = NULL`, and `buffer_deregister` is empty. The v2/legacy plugin still rejects null registration buffers and forwards deregistration to this no-op implementation.

Symmetric memory is implemented through oneCCL window wrappers. `deps/libccl/src/common/window/window_impl.cpp` only creates an internal window for `CCL_WIN_COLL_SYMMETRIC` on supported SYCL/Level Zero Arc paths; otherwise it logs that window registration is not supported and ignores the flag. `deps/libccl/src/comm/windows.cpp::ccl_window::register_buf` requires a pointer recognized by oneCCL's Level Zero context/device tracking, obtains the allocation base and size, rejects ranges outside the allocation, and exchanges IPC handles across the node communicator. `coll/algorithms/utils/simple_ipc.cpp::do_simple_ipc_exchange` requires all ranks to exchange matching sizes. The public `ccl::window` wrapper in `deps/libccl/include/oneapi/ccl/window.hpp` is a host C++ object, not a documented device-kernel ABI.

## Must fix

- Add XLA-side pointer validation before oneCCL GPU collectives, P2P, registration, and window creation. For non-zero GPU operations, accept only pointers that oneCCL can validate in the communicator/stream SYCL context. USM device and shared pointers are the conservative candidates for normal collectives; host USM, non-USM host/system pointers, `unknown`, wrong-context pointers, and null pointers with non-zero counts must be rejected with an XLA error.
- Short-circuit or otherwise explicitly define zero-count operations before entering oneCCL. SYCL allocation in `xla/stream_executor/sycl/sycl_gpu_runtime.cc` returns null for zero bytes, while oneCCL's v2 registration/window APIs reject null buffers and the legacy collective plugin can probe the buffer address before deeper validation.
- Add overflow checks to `ToOnecclCount` in `oneccl_communicator.cc`. Complex counts are doubled because oneCCL lacks complex datatypes; `count > max_size_t / 2` must fail before multiplication.
- Reject partial buffer overlap in XLA unless the exact operation-specific in-place layout is supported and documented. Do not rely on oneCCL's alias diagnostics because the relevant allgather/reduce-scatter overlap checks are environment-gated. All-to-all in-place should be rejected for oneCCL SYCL.
- Stop treating `ccl::window*` as an XLA device-kernel ABI. `OnecclSymmetricMemory::PackKernelArg` currently returns `window_.get()`, but oneCCL documents no stable kernel ABI for this host wrapper, and XLA's current symmetric kernels and barrier names are NCCL-specific. Gate oneCCL out of symmetric-kernel, multimem, `Put`, `Signal`, and `WaitSignal` paths until a oneCCL-compatible device ABI exists.
- Fix registered and symmetric memory lifetime relative to communicator lifetime. `OnecclRegisteredMemory` and `OnecclSymmetricMemory` store a raw `const ccl::communicator*` and dereference it in destructors. They must hold or be tied to communicator state so deregistration cannot run after communicator destruction.
- Require symmetric-memory all-rank ordering, local-support checks, and same-size registration before calling oneCCL windows. XLA's `CollectiveMemoryCache` and `AcquireSymmetricMemory` already rely on deterministic all-rank ordering, but oneCCL's IPC exchange also requires matching sizes and local Level Zero support. Fail early with a clear unsupported error when those preconditions are not met.

## Should fix

- Make registered-memory support honest. Since the reviewed oneCCL route sets the registration handle to null and deregistration is empty, XLA should either report this as unsupported/no-op for oneCCL or keep it behind a feature/version check with diagnostics.
- Deduplicate registered-memory registration per allocation and communicator/clique in `allocator_memory_registration.cc`. Repeated registration is currently possible if the same recorded allocation is registered with the same clique more than once; future oneCCL implementations with real handles could leak or over-register.
- Validate known buffer sizes when `DeviceAddressBase::size()` is available. Non-zero collective byte ranges should fit within the passed address range, and registration/window size should be non-zero and within the allocation returned by the SYCL/Level Zero query.
- Centralize oneCCL pointer classification in the SYCL stream executor layer instead of duplicating backend-specific pointer probes in each collective call site.
- Add operation-specific diagnostics that include operation name, dtype, logical element count, converted oneCCL count, pointer kind, address size, and expected stream/context.

## Could fix

- Support shared USM, system allocations, or non-device pointer classes for oneCCL only if upstream documentation and validation guarantee the behavior for GPU collectives.
- Coalesce symmetric-memory registrations after the correctness contract is enforced. `collective_memory.cc` already has a TODO for coalescing symmetric allocations.
- Add a future oneCCL device-window ABI only if upstream exposes a stable struct or kernel interface. Until then, compile-time and runtime guards should prevent accidental use of `ccl::window*` in kernels.
- Add version/build guards for oneCCL memory features because XLA currently references both a v1 commit and a v2 tarball route, and the memory/window APIs are not equally mature across routes.

## Affected files/call sites

- `xla/backends/gpu/collectives/oneccl_communicator.cc`: `ToOnecclCount`, `LaunchAllReduce`, `LaunchBroadcast`, `LaunchReduceScatter`, `LaunchAllGather`, `LaunchAllToAll`, `LaunchCollectivePermute`, `LaunchSend`, `LaunchRecv`, `CreateRegisteredMemory`, `CreateSymmetricMemory`, `Put`, `Signal`, and `WaitSignal`.
- `xla/backends/gpu/collectives/oneccl_registered_memory.{cc,h}`: registration creation, raw communicator pointer storage, destructor deregistration.
- `xla/backends/gpu/collectives/oneccl_symmetric_memory.{cc,h}`: window creation, raw communicator pointer storage, destructor deregistration, and `PackKernelArg`.
- `xla/backends/gpu/collectives/allocator_memory_registration.{cc,h}`: allocation tracking, repeated registration, and deregistration through `TiedRef`.
- `xla/stream_executor/sycl/sycl_executor.cc` and `xla/stream_executor/sycl/sycl_gpu_runtime.cc`: USM allocation kind, zero-byte allocation behavior, and potential pointer-query helpers.
- `xla/backends/gpu/runtime/collective_memory.{cc,h}`, `collective_memory_cache.{cc,h}`, and `collective_memory_requests.{cc,h}`: symmetric allocation request ordering, cache identity, and clique lifetime ties.
- `xla/backends/gpu/runtime/all_gather_thunk.cc`, `collective_kernel_thunk.cc`, `ragged_all_to_all.cc`, `ragged_all_to_all_thunk.cc`, `collective_kernel_api.{cc,h}`, and `xla/stream_executor/gpu/multi_gpu_barrier_kernel.h`: symmetric/kernel paths that assume NCCL-style device support.
- `~/github/uxlfoundation/oneCCL/src/api.cpp`, `plugins/legacy/ccl_legacy.cpp`, `deps/libccl/src/coll/*`, `deps/libccl/src/comm/*`, and `deps/libccl/src/common/window/*`: upstream behavior to track.

## Evidence to cite

- XLA `oneccl_communicator.cc::ToOnecclDataType` maps `C64` to `ccl::datatype::float32` and `C128` to `float64`; `ToOnecclCount` doubles counts for complex types without an overflow check.
- XLA `oneccl_communicator.cc::Launch*` methods pass `send_buffer.opaque()` and `recv_buffer.opaque()` directly to oneCCL calls.
- XLA `oneccl_registered_memory.cc::OnecclRegisteredMemory::Create` calls `ccl::comm_register(comm, addr.opaque(), addr.size(), &handle)` and its destructor calls `ccl::comm_deregister(*comm_, handle_)`.
- XLA `oneccl_symmetric_memory.cc::OnecclSymmetricMemory::Create` calls `ccl::comm_window_register(..., CCL_WIN_COLL_SYMMETRIC)`, its destructor calls `ccl::comm_window_deregister(*comm_, *window_)`, and `PackKernelArg` returns `window_.get()`.
- XLA `gpu_communicator.h` documents registered memory as a local RAII optimization and symmetric memory as a collective operation that all clique ranks must call.
- XLA `sycl_gpu_runtime.cc::SyclMallocDevice`, `SyclMallocHost`, and `SyclMallocShared` return null for zero bytes and allocate 64-byte-aligned USM otherwise.
- oneCCL `src/api.cpp` checks null buffers for registration/window calls but not collective buffers; `plugins/legacy/ccl_legacy.cpp::is_host_pointer` probes the buffer address before choosing CPU vs GPU execution.
- oneCCL `deps/libccl/src/coll/coll_check.cpp::ccl_check_usm_pointers` validates same-kind USM pointers in the communicator context and rejects host USM for GPU/accelerator devices and `unknown` pointers.
- oneCCL `deps/libccl/src/coll/coll_param.cpp::get_all_non_zero_bufs` skips zero-count buffers for validation.
- oneCCL `deps/libccl/src/coll/coll_util.cpp::is_allgatherv_inplace` and `is_reduce_scatter_inplace` make partial-alias rejection depend on `check_inplace_aliasing`; `deps/libccl/src/coll/algorithms/alltoall/sycl/alltoall_sycl.cpp` rejects in-place all-to-all in the multi-node path.
- oneCCL `deps/libccl/src/comm/comm.hpp::buffer_register` sets `*handle = NULL`, while `buffer_deregister` is empty.
- oneCCL `deps/libccl/src/comm/windows.cpp::ccl_window::register_buf` requires Level Zero allocation metadata, checks the requested range against the allocation base/size, and exchanges IPC handles; `deregister_buf` closes remote IPC handles.
- oneCCL `deps/libccl/src/common/window/window_impl.cpp` only creates an internal symmetric window on supported SYCL/Level Zero Arc paths and otherwise logs that the flag is ignored.
- SYCL 2020 defines USM allocation kinds and pointer queries in the official specification: <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm>.

## Test coverage plan

- Add pure unit tests for the new oneCCL pointer-validation helper using mocked SYCL pointer classification: device/shared accepted for normal non-zero GPU collectives, host USM rejected, non-USM/unknown rejected, null with non-zero count rejected, wrong-context rejected, and zero-count behavior short-circuited.
- Add `ToOnecclCount` tests for complex dtypes, including overflow rejection and unchanged real/integer counts.
- Add overlap tests per collective: exact allowed in-place, segmented allgather/reduce-scatter in-place, all-to-all in-place rejection, partial-overlap rejection, and P2P send/recv alias rejection where applicable.
- Add registered-memory tests for null/zero-size rejection, non-USM host rejection, repeated registration behavior, and destructor safety when allocator records are released before/after clique teardown.
- Add symmetric-memory tests for all-rank same-order registration, mismatched-size failure without hang, local-clique requirement, unsupported oneCCL window support returning a clear error, cache hit/miss behavior, and teardown after outstanding work is synchronized.
- Add kernel ABI guard coverage proving that oneCCL does not enter NCCL symmetric-kernel paths: one-sided all-gather, collective-kernel multimem, ragged all-to-all with symmetric memory, and `MultiGpuBarrierWithNcclKernel`.

## Rollout risk

Stricter validation may reject workloads that previously relied on accidental host pointers, null zero-count buffers, or partial aliasing. That is a correctness improvement, but diagnostics should be precise enough to identify the offending collective and pointer.

Zero-count short-circuiting must preserve collective/P2P ordering expectations. For collectives, all ranks must agree on the short-circuit condition; for P2P, matched zero-sized sends/recvs should be specified before skipping oneCCL.

Disabling oneCCL symmetric-kernel paths may remove experimental functionality or performance opportunities, but the current `ccl::window*` packing has no stable ABI basis and risks device-side misuse.

Treating registered memory as unsupported or no-op may reduce advertised feature coverage. It is safer than implying registration lifetime or performance semantics that the reviewed oneCCL implementation does not provide.
