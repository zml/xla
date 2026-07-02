# Local bootstrap, env defaults, and init timing

## Scope

This reviews XLA's oneCCL local single-process, multi-GPU bootstrap policy and
the oneCCL env/init paths it depends on. The focus is the XLA defaults for:

- `CCL_PROCESS_LAUNCHER=none`, `CCL_LOCAL_SIZE=1`, `CCL_LOCAL_RANK=0`
- `CCL_ATL_TRANSPORT=ofi`, `CCL_ATL_SHM=1`, `FI_PROVIDER=shm`
- `CCL_SYCL_ALLGATHERV_LL_THRESHOLD=1073741824`
- `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=33554432`

The review is static source/doc review only. No tests or experiments were run.

## XLA needs

XLA needs all-local oneCCL communicators to initialize deterministically in one
host process with N GPU ranks and without MPI/PMIx launcher semantics. For that
case, `CCL_LOCAL_SIZE` and `CCL_LOCAL_RANK` describe process-local launcher
coordinates, not GPU rank count, so XLA's single process should be
`CCL_LOCAL_SIZE=1` and `CCL_LOCAL_RANK=0`.

XLA also needs oneCCL env defaults to be applied before the first oneCCL API
that constructs the legacy C++ environment singleton. After that point, oneCCL
has parsed env into `global_data::env()`, initialized ATL internal env, and
built algorithm selectors; later `setenv` calls are not a reliable way to
change bootstrap or threshold behavior.

For BF16 Intel multi-GPU paths, XLA needs the conservative SYCL thresholds to be
present before oneCCL parses env. The current XLA comment says oneCCL's default
BF16 allgatherv path can return events that do not reliably complete, which is
a StreamExecutor correctness concern, not only a tuning concern.

## oneCCL behavior

`ccl::init()` in `deps/libccl/src/ccl_api_functions.cpp` only forces
`ccl::detail::environment::instance()`. That singleton is constructed once in
`deps/libccl/src/ccl_cpp_environment.cpp`, where `global_data::get().init()`
runs. `global_data::init()` in `deps/libccl/src/common/global/global.cpp`
parses env, sets local process coordinates, initializes wrappers, calls
`env_data::set_internal_env()`, and initializes the algorithm selector.

The relevant env defaults are not neutral for XLA:

- `env_data` defaults `process_launcher` to `hydra`; `process_launcher=none`
  makes `global_data::set_local_coord()` read `CCL_LOCAL_RANK` and
  `CCL_LOCAL_SIZE`.
- With MPI enabled, `env_data` defaults `atl_transport` to MPI, although
  `env_parser::env_2_atl_transport()` switches to OFI when no MPI launcher env
  is detected and `CCL_ATL_TRANSPORT` is unset. This does not change the
  process launcher default.
- `CCL_ATL_SHM` defaults off. OFI reads `FI_PROVIDER` directly. If
  `FI_PROVIDER=shm` is set but `CCL_ATL_SHM=1` is not, `atl_ofi::open_providers`
  rejects the configuration as "shm provider is requested through FI_PROVIDER
  but not requested from CCL level".
- `CCL_ATL_SHM=1` lets `atl_ofi_adjust_env()` add `shm` to an existing
  `FI_PROVIDER`; setting `FI_PROVIDER=shm` selects only the SHM provider for
  local process communication.
- oneCCL source defaults are much lower than XLA's conservative values:
  `sycl_allgatherv_ll_threshold=2048` and
  `sycl_allreduce_simple_threshold=4194304`.

XLA's current communicator path usually applies defaults before init:
`CreateCommunicatorsWithCancel()` calls
`SetOnecclSingleProcessBootstrapEnvDefaults()` for all-local cliques and then
`PrepareOnecclCommunicatorCreation()` calls
`SetOnecclIntelGpuCollectiveEnvDefaultsIfNeeded()` before
`EnsureOnecclInitialized()`. `InitializeTopology()` also applies bootstrap
defaults for single-process topology. However, XLA's `Allocate()` and
`Deallocate()` call the oneCCL v2 C API directly; `onecclMemAlloc()` initializes
the v2 plugin and the legacy plugin expects `onecclSetDevice()` before
allocation.

Official oneCCL docs document `CCL_ATL_TRANSPORT`, `CCL_ATL_SHM`,
`CCL_PROCESS_LAUNCHER`, `CCL_LOCAL_SIZE`, and `CCL_LOCAL_RANK` at
https://uxlfoundation.github.io/oneCCL/env-variables.html, and document that
`FI_PROVIDER` controls the libfabric provider for direct OFI transport at
https://uxlfoundation.github.io/oneCCL/general-configuration/transport-selection.html.
The exact `CCL_SYCL_ALLGATHERV_LL_THRESHOLD` and
`CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD` variables were source-only in this
checkout.

## Must fix

- Split and validate XLA's env default groups. The current
  `SetOnecclEnvDefaultGroupIfUnset()` all-or-nothing policy is too coarse:
  setting only `FI_PROVIDER`, only `CCL_ATL_SHM`, or only one threshold prevents
  XLA from setting unrelated mandatory defaults. That can leave all-local
  bootstrap on oneCCL's launcher defaults or disable the allgatherv BF16 safety
  threshold. At minimum, split the current six bootstrap defaults into
  process-coordinate and transport/provider groups, and split threshold handling
  so `CCL_SYCL_ALLGATHERV_LL_THRESHOLD` is not suppressed by a user setting only
  the allreduce threshold.
- Fail fast or complete missing values for partial local bootstrap overrides.
  For all-local single-process XLA, partial `CCL_PROCESS_LAUNCHER`,
  `CCL_LOCAL_SIZE`, `CCL_LOCAL_RANK` settings should not silently disable
  defaults. Compatible partials can be completed to `none/1/0`; incompatible
  partials should produce an actionable error before `ccl::init()`.
- Validate transport/provider compatibility before oneCCL init. In particular,
  `FI_PROVIDER=shm` requires `CCL_ATL_TRANSPORT=ofi` and `CCL_ATL_SHM=1`.
  `CCL_ATL_SHM=1` without `FI_PROVIDER=shm` may still open network providers,
  so XLA should either fill `FI_PROVIDER=shm` for all-local single-process
  bootstrap or clearly respect a complete user transport policy.
- Ensure every XLA-controlled oneCCL entry point that can initialize oneCCL runs
  the same pre-init default policy first. The communicator path mostly does
  this today; `OnecclCollectives::Allocate()` and `Deallocate()` are the notable
  v2 C API paths to gate or make unsupported until device/init ordering is
  defined.
- Keep `CCL_SYCL_ALLGATHERV_LL_THRESHOLD=1GiB` as an XLA-owned safety default
  for multi-GPU Intel SYCL until the oneCCL BF16 allgatherv event-completion
  issue is fixed and version-gated.

## Should fix

- Add a best-effort late-init diagnostic. XLA can track whether its own
  `EnsureOnecclInitialized()` has run and warn/error if a later path attempts
  to set bootstrap or threshold defaults. External pre-init by another oneCCL
  user may not be detectable through public APIs, so the message should explain
  that oneCCL env must be configured before any oneCCL call in the process.
- Treat `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=32MiB` as a conservative paired
  default, but make its safety/performance rationale explicit. It should not
  suppress the allgatherv threshold when users tune only allreduce.
- Extend the partial-override warning to print the missing variables in the
  related group and the exact values XLA would have used. The current warning
  says XLA is leaving remaining values unset, but not why that can make oneCCL
  fall back to launcher defaults.
- Document XLA's local oneCCL build macros near `third_party/oneccl/oneccl_v1.BUILD`
  and log them with the existing oneCCL version log: `CCL_ENABLE_OFI_HMEM`,
  `CCL_ENABLE_SYCL_INTEROP_EVENT`, `CCL_SYCL_ENABLE_ARCB`,
  `CCL_SYCL_VEC_SUPPORT_BF16`, `CCL_SYCL_VEC_SUPPORT_FP16`, and
  `CCL_ENABLE_UMF`.
- Add a version-gated escape hatch for the Intel threshold defaults once the
  oneCCL source/release note proving the BF16 event issue is fixed is available.

## Could fix

- Add a user-facing debug flag that prints the final XLA oneCCL env policy
  before init without requiring `CCL_LOG_LEVEL=info`.
- Track the upstream oneCCL env docs gap for LL/simple SYCL thresholds and link
  the source-only defaults in XLA comments until official docs cover them.
- Refine Intel device detection for threshold defaults by GPU family once the
  affected oneCCL paths are narrowed to PVC, BMG/ARCB, or another family.
- Revisit `FI_PROVIDER=shm` if XLA later supports multi-process local or
  multi-node oneCCL via this path; it is appropriate for the current
  one-process local bootstrap target, not a general distributed transport
  policy.

## Affected files/call sites

- XLA:
  - `xla/backends/gpu/collectives/oneccl_collectives.cc`:
    `SetOnecclEnvDefaultGroupIfUnset`,
    `SetOnecclSingleProcessBootstrapEnvDefaults`,
    `SetOnecclIntelGpuCollectiveEnvDefaultsIfNeeded`, `InitOnecclOnce`,
    `EnsureOnecclInitialized`, `PrepareOnecclCommunicatorCreation`,
    `OnecclCollectives::CreateCliqueIds`,
    `OnecclCollectives::CreateCommunicatorsWithCancel`,
    `OnecclCollectives::CreateUniqueCliqueId`,
    `OnecclCollectives::Allocate`, `OnecclCollectives::Deallocate`,
    `OnecclCollectives::InitializeTopology`.
  - `third_party/oneccl/oneccl_v1.BUILD`: local legacy oneCCL build macros.
  - `third_party/oneccl/oneccl_v2.BUILD`: v2 wrapper plus legacy plugin linkage.
  - `third_party/oneccl/workspace.bzl`: pinned v1 commit
    `4ceafd15c03ce46f11eeaf91781a92afebd3cecf` and v2 `master-v2` archive.
- oneCCL:
  - `deps/libccl/src/ccl_api_functions.cpp`: `ccl::init`.
  - `deps/libccl/src/ccl_cpp_environment.cpp`: environment singleton and
    `global_data::get().init()`.
  - `deps/libccl/src/common/global/global.cpp`: `global_data::init`,
    `set_local_coord`, `getenv_local_coord`.
  - `deps/libccl/src/common/env/env.cpp`: env defaults, `env_data::parse`,
    `env_data::set_internal_env`.
  - `deps/libccl/src/common/env/env_parser.cpp`: automatic OFI fallback when no
    MPI launcher env is detected.
  - `deps/libccl/src/common/env/vars.hpp` and `vars_experimental.hpp`: env
    variable definitions.
  - `deps/libccl/src/exec/exec.cpp`: `ccl_executor::generate_atl_attr`.
  - `deps/libccl/src/atl/ofi/atl_ofi.cpp` and
    `deps/libccl/src/atl/ofi/atl_ofi_helper.cpp`: `FI_PROVIDER`, SHM provider,
    and OFI env setup.
  - `deps/libccl/src/coll/algorithms/allgatherv/sycl/*` and
    `deps/libccl/src/coll/algorithms/allreduce/sycl/*`: threshold-sensitive
    SYCL algorithm branches.
  - `plugins/legacy/ccl_legacy.cpp`: v2 legacy plugin `onecclMemAlloc`,
    `onecclSetDevice`, and legacy C++ forwarding.
  - `deps/libccl/CMakeLists.txt`, root `CMakeLists.txt`, and
    `deps/libccl/cmake/helpers.cmake`: CMake feature macro defaults.

## Evidence to cite

- `research/oneccl/plan.md`, topic 3 rubric and focus points.
- `~/github/uxlfoundation/oneCCL/map.md`, env/config and build
  source routes.
- XLA `oneccl_collectives.cc`: current default values, group-default policy,
  and init order.
- oneCCL `ccl_cpp_environment.cpp` and `global.cpp`: one-time environment
  parsing during singleton initialization.
- oneCCL `env.cpp`: source defaults for `process_launcher=hydra`,
  `CCL_ATL_SHM=0`, `sycl_allgatherv_ll_threshold=2048`, and
  `sycl_allreduce_simple_threshold=4194304`.
- oneCCL `vars.hpp`: docs/source comments for `CCL_PROCESS_LAUNCHER`,
  `CCL_LOCAL_SIZE`, `CCL_LOCAL_RANK`, and `CCL_ATL_SHM`.
- oneCCL `atl_ofi.cpp` and `atl_ofi_helper.cpp`: direct `FI_PROVIDER` reads,
  SHM provider validation, and `FI_PROVIDER` mutation when `CCL_ATL_SHM=1`.
- oneCCL `allgatherv_sycl.cpp`, `allgatherv_pcie.cpp`,
  `allreduce_sycl.cpp`, and `allreduce_pcie.cpp`: launch-time checks of the
  parsed threshold fields.
- oneCCL docs:
  - https://uxlfoundation.github.io/oneCCL/env-variables.html
  - https://uxlfoundation.github.io/oneCCL/general-configuration/transport-selection.html
- libfabric SHM provider docs:
  - https://ofiwg.github.io/libfabric/main/man/fi_shm.7.html

## Test coverage plan

- Add hermetic unit coverage for the XLA env-policy helper without calling
  real oneCCL:
  - no bootstrap vars set: all local process and transport defaults are set;
  - compatible partial process vars: missing values are completed or the chosen
    policy is reported deterministically;
  - incompatible partial process vars: fail before `ccl::init`;
  - `FI_PROVIDER=shm` only, `CCL_ATL_SHM=1` only, and conflicting transport
    settings: validate or fail with actionable diagnostics;
  - threshold partial overrides: setting only allreduce does not disable the
    allgatherv safety default, and setting allgatherv explicitly is respected.
- Add an init-order unit seam around `EnsureOnecclInitialized()` so tests can
  assert defaults are applied before the first init call and late attempts emit
  the expected warning/error. This should use a fake init hook, not oneCCL.
- Add a static/build test that verifies the oneCCL Bazel target still defines
  the macros XLA relies on, especially `CCL_ENABLE_SYCL_INTEROP_EVENT` and the
  BF16/FP16 vector macros.
- Add guarded Intel SYCL multi-GPU integration coverage for local all-local
  communicator creation with no MPI/PMIx launcher env, checking that XLA uses
  in-memory KVS and reaches oneCCL with the expected env policy.
- Add guarded BF16 allgather/allreduce coverage around the XLA threshold
  boundaries. The assertions should cover completion and data correctness, not
  performance.

## Rollout risk

Changing partial override behavior can surprise users who rely on setting one
variable to suppress all XLA oneCCL defaults. The rollout should log the old and
new policy at `VLOG(1)` first, then move incompatible partial settings to a
clear error once tests cover the common cases.

The main correctness risk is oneCCL version drift: XLA pins oneCCL v1 by commit
but v2 by the `master-v2` branch archive. Threshold semantics and feature macro
requirements should be tied to a specific oneCCL revision in comments or tests.

The main performance risk is the conservative BF16 threshold policy. It may
avoid faster default oneCCL paths on some Intel GPU families, so keep the
override user-controllable and version-gate removal when oneCCL provides a
source/release-note fix.

External pre-initialization remains a gap: if another library calls oneCCL
before XLA, public oneCCL APIs do not appear to expose a clean "already parsed
env" query. XLA should own all init paths it controls and make the process-wide
env timing requirement visible in diagnostics.
