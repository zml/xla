# 10. Cluster/cooperative launch

## Executive recommendation

Keep non-default cluster/cooperative launch dimensions unsupported for SYCL until there is a real lowering to a SYCL extension or backend-specific launch API. Direct SYCL kernel launch already fails before queue submission for non-`1x1x1` `ClusterDim`, so there is no direct-launch semantic weakening today. The must-fix issue is the command-buffer record path: `KernelThunk::Record()` and `CustomKernelThunk::Record()` drop stored cluster dimensions because `CommandBuffer::CreateLaunch()`/`UpdateLaunch()` cannot represent them.

Decision: no current HLO path was found that intentionally creates non-default cluster dimensions for SYCL. Non-default values can still enter shared runtime objects through `KernelThunk`/`CustomKernel` construction or proto/cache deserialization, and direct SYCL execution rejects them late at runtime. Add earlier diagnostics for SYCL, and add explicit record-time rejection before any command-buffer launch is built.

## Must/Should/Could classification

- **Must fix: command-buffer silent semantic weakening.** `KernelThunk::Record()` and `CustomKernelThunk::Record()` record only thread/block dimensions. If a thunk carries non-default cluster dims, recording silently loses them. Until the command-buffer API supports cluster dims, reject non-default cluster dims in these `Record()` methods, at least for SYCL and preferably generically.
- **Should fix: earlier SYCL diagnostics.** Direct launch rejection happens in `SyclStream::LaunchKernel()` after kernel args have been packed and the kernel object loaded. Add a platform-aware validation earlier in thunk execution/AOT load/custom-kernel load so users see a named unsupported-feature error before launch setup.
- **Should fix: serialized input validation.** `KernelThunkProto`, `CustomKernelProto`, and `KernelReuseCache` can carry cluster dims. SYCL executable/proto loading should reject non-default cluster dims with a clear message rather than deferring to the stream launch path.
- **Could support: future backend-specific lowering.** A future CUDA-SYCL path could map `ClusterDim` to the proposed Codeplay CUDA `cluster_size` launch property and capability query. No Intel-specific lowering was found; do not map clusters to ordinary `nd_range` work-groups.

## XLA change candidates with concrete files/functions

- `~/github/openxla/xla/xla/backends/gpu/runtime/kernel_thunk.cc`
  - `KernelThunk::ExecuteOnStream()`: direct launch already forwards `cluster_dim_`; optional place for a named early SYCL rejection.
  - `KernelThunk::Record()`: must reject non-default `cluster_dim_` before `command_buffer->CreateLaunch()`/`UpdateLaunch()`, or extend `CommandBuffer` to carry cluster dims.
  - `KernelThunk::FromProto()`: should validate non-default cluster dims when loading for a SYCL target if platform context is available elsewhere in the load path.
- `~/github/openxla/xla/xla/backends/gpu/runtime/custom_kernel_thunk.cc`
  - `CustomKernelThunk::ExecuteOnStream()`: direct launch forwards `custom_kernel_.cluster_dims()`.
  - `CustomKernelThunk::Record()`: must reject non-default `custom_kernel_.cluster_dims()` before command-buffer create/update.
- `~/github/openxla/xla/xla/stream_executor/command_buffer.h`
  - `CommandBuffer::CreateLaunch()` and `UpdateLaunch()` have no `ClusterDim` parameter. Longer-term support would require an API extension plus backend implementations.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc`
  - `SyclStream::LaunchKernel()` is the current direct-launch rejection point. Improve message context if earlier rejection is not added.
- `~/github/openxla/xla/xla/stream_executor/sycl/sycl_command_buffer.cc`
  - `LaunchParams` and `AddLaunchNode()` store only threads, blocks, kernel, args, and shared memory. If command-buffer API gains cluster dims, this file needs an implementation or explicit unsupported check.
- `~/github/openxla/xla/xla/pjrt/triton_oneapi.cc`
  - If oneAPI Triton ever needs CUDA-style clusters, this path must explicitly lower or reject `ttg.num-ctas` semantics; today it does not extract cluster dims like the CUDA path.

## Evidence with code references and spec/oneAPI references where available

- `ClusterDim` defaults to `1,1,1` in `~/github/openxla/xla/xla/stream_executor/launch_dim.h:81`; the non-cluster `Kernel::Launch()` overload forwards `std::nullopt` in `~/github/openxla/xla/xla/stream_executor/kernel.h:160`.
- Direct SYCL launch rejects non-default clusters before calling the non-cluster launch helper: `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:170` checks `cluster_dim_x/y/z` and returns `UnimplementedError` at lines 179-181. The actual `queue::submit()` is in the lower helper at `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:118`, so the cluster rejection is pre-submit.
- `SyclStream::LaunchKernel()` forwards optional cluster dims into that rejecting helper at `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream.cc:412`; `SyclKernel::Launch()` forwards the optional value unchanged at `~/github/openxla/xla/xla/stream_executor/sycl/sycl_kernel.cc:47`.
- `KernelThunk` stores, serializes, deserializes, and launches cluster dims in `~/github/openxla/xla/xla/backends/gpu/runtime/kernel_thunk.cc:60`, `:99`, `:117`, and `:259`. Tests prove non-default values survive construction/proto round trip in `~/github/openxla/xla/xla/backends/gpu/runtime/kernel_thunk_test.cc:121`, `:181`, and `:277`.
- `CustomKernel` can carry cluster dims and serializes them in `~/github/openxla/xla/xla/backends/gpu/codegen/kernels/custom_kernel.cc:44`, `:65`, `:92`, and `:110`. `CustomKernelThunk::ExecuteOnStream()` forwards them at `~/github/openxla/xla/xla/backends/gpu/runtime/custom_kernel_thunk.cc:154`.
- Command-buffer recording drops cluster dims: `KernelThunk::Record()` calls `CreateLaunch()`/`UpdateLaunch()` with only thread/block dims in `~/github/openxla/xla/xla/backends/gpu/runtime/kernel_thunk.cc:277`; `CustomKernelThunk::Record()` does the same in `~/github/openxla/xla/xla/backends/gpu/runtime/custom_kernel_thunk.cc:174`.
- The base command-buffer API has no cluster parameter in `~/github/openxla/xla/xla/stream_executor/command_buffer.h:164` and `:170`. SYCL command-buffer `LaunchParams` has no cluster field in `~/github/openxla/xla/xla/stream_executor/sycl/sycl_command_buffer.cc:148`, and `AddLaunchNode()` builds only an `nd_range` from thread/block dims at lines 364-401.
- Current emitted XLA kernels do not create non-default clusters: `ThunkEmitter` stores `cluster_dim=std::nullopt` in `~/github/openxla/xla/xla/service/gpu/thunk_emitter.cc:1354`, Triton fusion does the same in `~/github/openxla/xla/xla/backends/gpu/codegen/triton/fusion.cc:238`, and `CollectiveKernelThunk` launches with `std::nullopt` in `~/github/openxla/xla/xla/backends/gpu/runtime/collective_kernel_thunk.cc:489`.
- SYCL PTX custom-call emission is explicitly unimplemented in `~/github/openxla/xla/xla/service/gpu/custom_kernel_emitter_sycl_stub.cc:27`; the CUDA emitter uses the no-cluster custom-kernel helper in `~/github/openxla/xla/xla/service/gpu/custom_kernel_emitter_cuda.cc:59`.
- CUDA PjRt Triton extracts `ttg.num-ctas` into cluster dims in `~/github/openxla/xla/xla/pjrt/triton_cuda.cc:165`. The oneAPI Triton compiler returns SPIR-V/shared-memory/scratch data without extracting cluster dims in `~/github/openxla/xla/xla/pjrt/triton_oneapi.cc:114`; the PjRt C Triton args expose no output cluster fields in `~/github/openxla/xla/xla/pjrt/c/pjrt_c_api_triton_extension.h:36`.
- SYCL 2020 describes ordinary work-items/work-groups and `nd_range` execution in `~/sycl/sycl-2020.html:3024` and kernel invocation forms in `~/sycl/sycl-2020.html:29625`; local `rg` found no `cluster`/`cooperative` hits in `sycl-2020.html` or `sycl-2020-map.md`.
- The local proposed oneAPI/Codeplay extension `~/sycl/oneapi/extensions/proposed/sycl_ext_codeplay_cuda_cluster_group.asciidoc` is CUDA-oriented: it describes CUDA cc 9.0 thread block clusters at lines 78-90, requires an `ext_codeplay_cuda_cluster_group` aspect at lines 119-130, defines a `cluster_size` launch property at lines 133-168, and adds device-code `cluster_group` access at lines 197-214. `oneapi.md` says `proposed` extensions are not yet implemented/stable at `~/sycl/oneapi.md:39`.

## Findings

1. Direct SYCL execution is semantically safe for non-default clusters: it returns `UnimplementedError` before queue submission.
2. Direct SYCL rejection is late. The error is raised inside the stream launch helper, not when a SYCL executable is compiled, loaded, deserialized, or recorded.
3. Command-buffer recording is unsafe for clustered thunks. Since command-buffer launches cannot carry `ClusterDim`, record paths silently degrade a clustered launch into an ordinary launch.
4. No current Intel-specific cluster launch lowering was found. XLA SYCL code has no hits for the proposed extension names (`cluster_size`, `ext_codeplay_cuda_cluster_group`, `max_cluster_group_size`), and the oneAPI Triton path does not mirror CUDA cluster extraction.

## Affected HLO/custom-call paths

- Regular SYCL HLO codegen: currently no non-default cluster producer found; emitted entries use `std::nullopt`.
- Triton fusion through XLA GPU codegen: currently emits `std::nullopt` cluster dims.
- PjRt Triton CUDA helper: extracts cluster dims from Triton IR, but the PjRt C API does not expose those fields and this is not a SYCL lowering path.
- PjRt Triton oneAPI helper: no non-default cluster extraction found.
- PTX custom call: CUDA-only emitter path exists and SYCL stub rejects PTX custom calls. The programmatic `CustomKernel`/proto path can still carry cluster dims into `CustomKernelThunk`.
- AOT/proto/cache paths: `KernelThunkProto`, `CustomKernelProto`, and `KernelReuseCache` can persist non-default cluster dims. Loading such artifacts for SYCL should fail explicitly.
- Command-buffer execution: affected for both `KernelThunk` and `CustomKernelThunk` because recording drops cluster dims before SYCL command-graph construction.

## Proposed patch plan

1. Add a small helper such as `IsDefaultClusterDim(std::optional<se::ClusterDim>)` in an appropriate runtime utility location.
2. In `KernelThunk::Record()` and `CustomKernelThunk::Record()`, return `UnimplementedError` when cluster dims are present and not `1x1x1`, before `CreateLaunch()` or `UpdateLaunch()`.
3. Add a SYCL-specific earlier validation point for direct execution, preferably before packed args and kernel launch. The diagnostic should name the kernel/thunk and the requested cluster dims.
4. Add load/deserialization validation for SYCL executables if platform context is available at `GpuExecutable`/AOT load time. If not available there, validate during thunk initialization on a SYCL executor.
5. Do not implement fallback lowering to ordinary SYCL `nd_range`; that would erase CUDA cluster scheduling, local-memory, and synchronization semantics.
6. Future support path: extend StreamExecutor command-buffer launch APIs with optional cluster dims, add SYCL capability checks for a concrete implemented extension, then lower `ClusterDim` to a backend property such as `cluster_size` only when the device reports support.

## Test/benchmark coverage

- Add a SYCL direct-launch unit/integration test in `~/github/openxla/xla/xla/stream_executor/sycl/sycl_stream_test.cc` or `~/github/openxla/xla/xla/stream_executor/sycl/restricted/sycl_kernel_test.cc` that launches a loaded test kernel with `se::ClusterDim(2,1,1)` and expects `kUnimplemented` before `BlockHostUntilDone()`.
- Add runtime unit tests in `~/github/openxla/xla/xla/backends/gpu/runtime/kernel_thunk_test.cc` and `~/github/openxla/xla/xla/backends/gpu/runtime/custom_kernel_thunk_test.cc` proving `Record()` rejects non-default cluster dims instead of calling `CreateLaunch()`.
- Add proto/AOT validation tests that deserialize `KernelThunkProto` and `CustomKernelProto` with `cluster_dim { x: 2 }` for a SYCL target and assert an explicit unsupported-feature error.
- Keep existing serialization tests for cluster dims because CUDA support still needs the shared representation.
- No benchmark is needed for the rejection patch. If future cluster support is implemented, add backend-specific correctness tests for cluster synchronization/local-memory semantics before performance benchmarks.

## Rollout risk

Risk is low for direct execution because SYCL already rejects non-default clusters. The main behavior change is that command-buffer capture/recording will fail early instead of building an ordinary non-cluster launch. That is a correctness-preserving breaking change for unsupported programs. Generic command-buffer API changes would have broader backend risk and should be separate from the short-term rejection patch.

Future support has high semantic risk: CUDA clusters require concurrent work-group scheduling, cluster-scope synchronization, and cross-work-group local-memory access. These cannot be inferred from SYCL 2020 `nd_range` alone.

## Evidence gaps

- No local evidence proves an implemented Intel SYCL cluster launch API equivalent to CUDA thread block clusters.
- The local Codeplay CUDA cluster-group document is in `oneapi/extensions/proposed`, so it is evidence of a possible future CUDA-oriented API shape, not proof of supported Intel lowering.
- I did not run XLA tests or hardware experiments; this note is based on local source and local documentation inspection.
- I did not audit downstream users outside `~/github/openxla/xla` that might construct `KernelThunk` or `CustomKernel` with non-default cluster dims.
