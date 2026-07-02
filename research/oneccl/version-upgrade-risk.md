# Version pins, v1/v2 wrapper, and upgrade risk

## Scope

This note reviews XLA's vendored oneCCL dependency shape and the upgrade
process needed for single-process multi-GPU Intel SYCL collectives. It covers
the Bazel pins in `third_party/oneccl`, the v2 C wrapper plus legacy v1
implementation split, XLA call sites that use C++ and C oneCCL APIs, and the
external oneAPI/Level Zero/libfabric assumptions that should travel with any
oneCCL upgrade.

`research/oneccl/intel-map.md` was not present, so the review started from
`research/oneccl/plan.md` and `~/github/uxlfoundation/oneCCL/map.md`.
The local oneCCL checkout is `master-v2` at
`1318c3aaf8a67536ff538331c418c8cea3114f33`, tagged `2022.0.0`. Its
`deps/libccl` submodule is `4ceafd15c03ce46f11eeaf91781a92afebd3cecf`,
the same commit currently pinned as XLA's `@oneccl_v1`.

## XLA needs

XLA needs a reproducible oneCCL source identity, not only a content hash, so
correctness findings from the other oneCCL topic documents can be tied to a
specific upstream revision and retested after upgrades.

XLA also needs an explicit policy for whether the integration is a legacy C++
API integration, a v2 C API integration, or a deliberately mixed integration.
Today collective communicator operations use `oneapi/ccl.hpp` and the legacy
C++ API, while `OnecclCollectives::Allocate` and `Deallocate` use v2 C API
entry points (`onecclMemAlloc` and `onecclMemFree`). The upgrade process must
preserve the behavior of both surfaces or remove the mixed surface.

For rollout, XLA needs a documented version matrix covering:

- oneCCL v2 wrapper commit and oneCCL v1/libccl commit.
- oneAPI compiler/runtime versions used by hermetic and system SYCL builds.
- Level Zero loader/header/runtime assumptions.
- libfabric/OFI assumptions for local `FI_PROVIDER=shm` use.
- XLA-local oneCCL patches and their upstream status.

## oneCCL behavior

XLA currently pins `@oneccl_v1` to an immutable commit:
`third_party/oneccl/workspace.bzl:repo_v1` downloads
`uxlfoundation/oneCCL/archive/4ceafd15c03ce46f11eeaf91781a92afebd3cecf.tar.gz`,
uses a fixed SHA256, and applies `third_party/oneccl/ze_loader.patch`.

XLA currently pins `@oneccl` less clearly:
`third_party/oneccl/workspace.bzl:repo_v2` downloads
`https://github.com/uxlfoundation/oneCCL/archive/refs/heads/master-v2.tar.gz`
with `strip_prefix = "oneCCL-master-v2"` and a SHA256. The SHA256 prevents
silent content substitution, but the URL and strip prefix name a moving branch,
so a clean fetch can fail once the branch changes and the source identity is not
auditable from the file itself. The local matching immutable source is
`1318c3aaf8a67536ff538331c418c8cea3114f33`.

XLA's v2 BUILD file does not build upstream oneCCL's whole superproject. It
compiles only `src/api.cpp`, `src/debug.cpp`, and
`plugins/legacy/ccl_legacy.cpp` from the v2 source archive, then links those
files against the separate `@oneccl_v1` archive. Upstream CMake instead uses
the v2 repository's own `deps/libccl` submodule unless
`ONECCL_USE_SYSTEM_LIBCCL` is set. For the local `2022.0.0` checkout that
submodule matches XLA's v1 commit, but XLA should make that pairing explicit.

The v2 wrapper is plugin based. `src/api.cpp` loads a selected plugin through
`onecclPluginCall`, and its plugin search table contains an XLA/Bazel-specific
legacy plugin name, `libexternal_Soneccl_Slibccl_Ulegacy.so`. The legacy plugin
in `plugins/legacy/ccl_legacy.cpp` translates v2 C API calls back to legacy
`ccl::` C++ API calls. Its memory allocation path is thread-local device
selection based: `oneccl_mem_alloc_impl` requires `oneccl_set_device_impl` to
have populated `selected_device`, `selected_context`, and `default_stream`.

The official oneCCL v2 docs describe the new C API as plugin based and
document `CCL_PLUGIN` overrides. The official upgrade docs distinguish the
legacy C++ API (`oneapi/ccl.hpp`, `libccl.so.1`) from the new C API
(`oneapi/ccl.h`, `libccl.so.2`). Intel's 2022.0 release notes say the new
NCCL-like C API becomes default in oneAPI 2026.0 and state that applications
cannot use both the C and C++ APIs simultaneously.

XLA logs the legacy C++ oneCCL compiled and runtime versions through
`ccl::get_library_version`, but it does not log the v2 C wrapper version from
`onecclGetVersion`. The v1 Bazel BUILD also injects version values manually
(`2022.1.0`, build date `2026-05-13`), which do not match the local upstream
libccl CMake values (`2022.0.0` in the checked-out submodule).

## Must fix

- Replace the `master-v2` archive in `third_party/oneccl/workspace.bzl` with
  an immutable commit archive. For the reviewed local source, the coherent v2
  commit is `1318c3aaf8a67536ff538331c418c8cea3114f33`, whose
  `deps/libccl` submodule is
  `4ceafd15c03ce46f11eeaf91781a92afebd3cecf`. Keep the SHA256, but make the
  URL and `strip_prefix` encode the commit, not `refs/heads/master-v2`.

- Treat oneCCL v2 wrapper upgrades and v1/libccl upgrades as an atomic pair.
  If XLA updates `repo_v2`, it must also verify the upstream v2 superproject's
  `deps/libccl` commit and either update `repo_v1` to that exact commit or
  document why the pair is intentionally different. A mismatched wrapper and
  legacy implementation can change plugin ABI, datatype enums, communicator
  config layout, memory APIs, and group behavior without an XLA source change.

- Decide and enforce the supported API boundary before enabling more v2 C API
  calls. Current communicator collectives use the legacy C++ API directly, but
  `OnecclCollectives::Allocate` and `Deallocate` call v2 `onecclMemAlloc` and
  `onecclMemFree`. Because the v2 legacy plugin's allocation path requires
  prior v2 `onecclSetDevice` state, XLA should either move these allocations to
  a C++ API path that shares the existing `ccl::stream`/device setup, or add an
  explicit v2 initialization wrapper and version gate for all v2 calls.

- For any issue classified Must in the other topic documents, prefer upgrading
  or backporting the upstream oneCCL fix before carrying an XLA workaround that
  relies on unspecified implementation behavior. If an upstream fix is already
  present after the pinned commits, the rollout gate should be "upgrade oneCCL
  or cherry-pick the oneCCL fix" before enabling the affected path by default.

## Should fix

- Add a checked-in oneCCL version manifest near `third_party/oneccl` recording
  the v2 wrapper commit, v1/libccl commit, archive SHA256 values, upstream tag
  or release note, `ze_loader.patch` status, oneAPI compiler/runtime version,
  Level Zero version, and libfabric/OFI expectation. This should be the source
  of truth for future oneCCL upgrades.

- Align the manually generated legacy version in
  `third_party/oneccl/oneccl_v1.BUILD` with the actual pinned upstream
  revision, or explain why XLA intentionally reports a different oneCCL version.
  Misleading `ccl::get_library_version` output makes field debugging and
  upgrade bisection harder.

- Log both oneCCL surfaces at communicator initialization: the legacy C++
  library version from `ccl::get_library_version` and the v2 C wrapper version
  from `onecclGetVersion`, including the v2 wrapper commit when available.
  `src/CMakeLists.txt` already supports v2 build metadata in upstream CMake;
  XLA's Bazel build should provide comparable metadata or document why it
  cannot.

- Keep `third_party/oneccl/ze_loader.patch` small, but do not let it remain an
  anonymous downstream delta. Either upstream the `libze_loader.so.1` default or
  document the runtime packaging reason, affected oneAPI/Level Zero versions,
  and the condition under which the patch can be dropped.

- Maintain a per-topic upstream-fix map. For each Must or Should issue in the
  other oneCCL research docs, record the oneCCL source route, the pinned
  behavior, the first upstream commit or release believed to fix it, and
  whether XLA should upgrade, cherry-pick, or keep an XLA-side guard.

- Document supported user override policy for version-sensitive environment
  variables. `CCL_PLUGIN`, `CCL_*`, and `FI_PROVIDER` can move XLA onto paths
  not covered by the pinned-source assumptions. XLA should make clear which
  overrides are supported, unsupported, or diagnostic-only.

- Reconcile XLA's oneAPI version signals. `third_party/gpus/sycl_configure.bzl`
  and `third_party/gpus/sycl/sycl_dl_essential.bzl` currently describe a
  hermetic oneAPI `2025.1` path, while
  `xla/pjrt/gpu/se_gpu_pjrt_client.cc:platform_version` reports
  `oneapi 2026.0`. That mismatch should be resolved or explicitly documented
  in the version matrix.

## Could fix

- Move duplicated `oneccl_v1()` and `oneccl_v2()` calls in `workspace2.bzl`
  into a single keep-sorted location if repository-rule idempotence is the only
  reason this is harmless.

- Add a small Bazel/Starlark helper for oneCCL archive declarations so v1 and
  v2 commits, SHA256 values, and BUILD files are updated together.

- Upstream XLA's Bazel-specific plugin name support if it is not already
  accepted upstream as part of `src/api.cpp`. This reduces the chance that a
  future v2 wrapper refactor drops XLA's plugin discovery path.

- Consider a future migration to a single oneCCL API surface after oneCCL's C
  API is mature enough for all XLA requirements, including stream/event
  ordering, communicator split, memory registration, symmetric memory, and
  failure cleanup.

## Affected files/call sites

- `third_party/oneccl/workspace.bzl`: `repo_v1`, `repo_v2`, archive URLs,
  `strip_prefix`, SHA256 values, and `ze_loader.patch`.
- `third_party/oneccl/oneccl_v1.BUILD`: `_CMAKE_COMMON_LIST` version
  substitutions, legacy source glob, compile defines such as
  `CCL_ENABLE_OFI_HMEM`, `CCL_ENABLE_SYCL_INTEROP_EVENT`,
  `CCL_ENABLE_UMF`, `CCL_SYCL_ENABLE_ARCB`, BF16/FP16 vector support.
- `third_party/oneccl/oneccl_v2.BUILD`: `:oneccl`, `:ccl_legacy`, and `:libs`
  targets that combine v2 wrapper sources with `@oneccl_v1`.
- `third_party/oneccl/ze_loader.patch`: local Level Zero loader SONAME patch.
- `workspace2.bzl`, `third_party/extensions/third_party.bzl`, `MODULE.bazel`:
  oneCCL repository registration.
- `xla/backends/gpu/collectives/BUILD`: oneCCL targets linking `@oneccl//:libs`.
- `xla/backends/gpu/collectives/oneccl_collectives.cc`:
  `CompiledOnecclVersion`, `GetOnecclLibraryVersion`,
  `LogOnecclCommunicatorInitialization`, `Allocate`, `Deallocate`,
  `SetOnecclSingleProcessBootstrapEnvDefaults`,
  `SetOnecclIntelGpuCollectiveEnvDefaultsIfNeeded`.
- `xla/backends/gpu/collectives/oneccl_communicator.cc`:
  `ToOnecclStream`, `LaunchOnecclAndWait`, `WaitForOnecclEvent`,
  `GroupExecute`, `Launch*`, `CreateRegisteredMemory`,
  `CreateSymmetricMemory`.
- `xla/backends/gpu/collectives/oneccl_errors.cc`: v2 C API error string
  wrappers.
- `xla/backends/gpu/collectives/oneccl_registered_memory.cc` and
  `xla/backends/gpu/collectives/oneccl_symmetric_memory.cc`: legacy C++ memory
  registration/window surfaces.
- `third_party/gpus/sycl_configure.bzl`,
  `third_party/gpus/sycl/sycl_dl_essential.bzl`,
  `third_party/gpus/sycl/level_zero.bzl`,
  `third_party/gpus/find_sycl_config.py`,
  `xla/pjrt/gpu/se_gpu_pjrt_client.cc`: oneAPI, Level Zero, and CCL discovery
  assumptions.
- oneCCL source routes:
  `~/github/uxlfoundation/oneCCL/src/api.cpp`,
  `~/github/uxlfoundation/oneCCL/plugins/legacy/ccl_legacy.cpp`,
  `~/github/uxlfoundation/oneCCL/include/oneapi/ccl.h`,
  `~/github/uxlfoundation/oneCCL/include/oneapi/ccl/v2/types.h`,
  `~/github/uxlfoundation/oneCCL/src/internal/api/comm.h`,
  `~/github/uxlfoundation/oneCCL/src/internal/api/plugin.h`,
  `~/github/uxlfoundation/oneCCL/CMakeLists.txt`,
  `~/github/uxlfoundation/oneCCL/deps/libccl/CMakeLists.txt`,
  `~/github/uxlfoundation/oneCCL/deps/libccl/include/oneapi/ccl/config.h.in`,
  `~/github/uxlfoundation/oneCCL/deps/libccl/src/common/utils/version.cpp`.

## Evidence to cite

- XLA `repo_v1` is immutable:
  `third_party/oneccl/workspace.bzl:repo_v1` uses commit
  `4ceafd15c03ce46f11eeaf91781a92afebd3cecf`, fixed SHA256, and
  `ze_loader.patch`.
- XLA `repo_v2` uses a moving branch URL:
  `third_party/oneccl/workspace.bzl:repo_v2` uses
  `refs/heads/master-v2.tar.gz` and `strip_prefix = "oneCCL-master-v2"`.
- XLA v2 BUILD compiles only wrapper/plugin sources and depends on
  `@oneccl_v1`: `third_party/oneccl/oneccl_v2.BUILD`.
- XLA legacy C++ oneCCL use:
  `xla/backends/gpu/collectives/oneccl_communicator.cc` includes
  `oneapi/ccl.hpp` and calls `ccl::allreduce`, `ccl::broadcast`,
  `ccl::reduce_scatter`, `ccl::allgatherv`, `ccl::alltoall`,
  `ccl::send`, `ccl::recv`, and `ccl::barrier`.
- XLA v2 C API use:
  `xla/backends/gpu/collectives/oneccl_collectives.cc:Allocate` calls
  `onecclMemAlloc`; `Deallocate` calls `onecclMemFree`;
  `xla/backends/gpu/collectives/oneccl_errors.cc` calls
  `onecclGetErrorString` and `onecclGetLastError`.
- oneCCL v2 plugin loader:
  `~/github/uxlfoundation/oneCCL/src/api.cpp` has
  `kPluginPaths`, `load_plugin`, `init_library`, `onecclGetVersion`, and
  forwarding wrappers.
- oneCCL legacy plugin:
  `~/github/uxlfoundation/oneCCL/plugins/legacy/ccl_legacy.cpp`
  has `onecclPluginCall`, `execute_collective`, `get_stream`,
  `oneccl_init_communicator_impl`, `oneccl_set_device_impl`, and
  `oneccl_mem_alloc_impl`.
- oneCCL upstream superproject pairing:
  local `oneCCL` HEAD is `1318c3aaf8a67536ff538331c418c8cea3114f33`
  (`2022.0.0`); `git ls-tree HEAD deps/libccl` reports submodule
  `4ceafd15c03ce46f11eeaf91781a92afebd3cecf`.
- Upstream CMake pairing:
  `~/github/uxlfoundation/oneCCL/CMakeLists.txt` uses
  `SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/libccl` for the embedded legacy build
  unless `ONECCL_USE_SYSTEM_LIBCCL` is set.
- Official oneCCL C API docs:
  https://uxlfoundation.github.io/oneCCL/v2/index.html
- Official oneCCL plugin docs:
  https://uxlfoundation.github.io/oneCCL/v2/plugins.html
- Official oneCCL C API upgrade docs:
  https://uxlfoundation.github.io/oneCCL/v2/upgrade.html
- Official oneCCL C API env docs:
  https://uxlfoundation.github.io/oneCCL/v2/env.html
- Official legacy oneCCL env docs:
  https://uxlfoundation.github.io/oneCCL/env-variables.html
- Intel oneCCL 2022.0 release notes and system requirements:
  https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-collective-communication-library-ccl/2022-0.html
- Level Zero versioning and ABI reference:
  https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html
- libfabric SHM provider reference:
  https://ofiwg.github.io/libfabric/main/man/fi_shm.7.html

## Test coverage plan

- Add a source identity presubmit or repository-rule check that verifies the
  v2 wrapper commit and v1/libccl commit are the intended upstream pair. The
  check should fail if `repo_v2` uses a branch URL or if the documented v2
  superproject submodule commit differs from `repo_v1` without an explicit
  override.

- Add a build/analysis test that compiles all XLA oneCCL targets with the
  pinned v1/v2 pair and asserts that the expected oneCCL symbols are resolved
  from the intended API surface. This should cover `onecclMemAlloc`,
  `onecclMemFree`, `onecclGetErrorString`, legacy `ccl::communicator`, and
  memory registration/window APIs.

- Add a version-reporting test for `CompiledOnecclVersion`,
  `ccl::get_library_version`, and `onecclGetVersion` formatting. The test
  should catch mismatches caused by manual Bazel version substitutions.

- Add an upgrade checklist test suite that is run for every oneCCL bump:
  producer-kernel to oneCCL to consumer-kernel ordering, barrier ordering,
  local bootstrap env defaults, communicator create/split/destroy/failure
  cleanup, grouped collectives, P2P cycles, BF16/allgather/allreduce threshold
  behavior, pointer registration, symmetric memory window lifecycle, and
  `CCL_PLUGIN` override diagnostics. These tests preserve the version-sensitive
  behavior identified by the other topic documents.

- Add a small negative test around v2 memory allocation initialization if XLA
  continues to call `onecclMemAlloc`: verify the wrapper either initializes the
  v2 device state explicitly or returns an XLA diagnostic that tells the user
  which oneCCL API path is unsupported.

## Rollout risk

Replacing `master-v2` with a commit archive is low runtime risk but moderate
repository risk: all mirrors and SHA256 values must be updated together, and
stale caches may mask missing mirror coverage.

Upgrading oneCCL is high behavioral risk until the other topic documents have
tests. oneCCL version changes can alter event completion, group behavior,
communicator split, algorithm thresholds, BF16/FP16 paths, Level Zero command
submission, OFI/SHM transport, and plugin selection. Keep an XLA flag or build
pin rollback path for the first upgrade after this audit.

The mixed v1/v2 API surface is the main compatibility risk. Intel's 2022.0
release notes say applications cannot use both C and C++ APIs simultaneously,
while XLA currently links both wrapper and legacy C++ call paths. Even if this
works for the current vendored pair, future oneCCL releases can make the mix
less stable as the C API becomes default.

`ze_loader.patch` is a runtime packaging risk. Dropping it may regress systems
that ship `libze_loader.so.1` but not an unversioned `libze_loader.so`; carrying
it forever without upstreaming increases rebase friction.

User environment overrides are a rollout risk. `CCL_PLUGIN`, `CCL_*`, and
`FI_PROVIDER` can select code paths outside XLA's tested defaults. The rollout
should log enough version, plugin, and env information to make unsupported
overrides obvious.

oneAPI/Level Zero/libfabric assumptions are not centralized today. Hermetic XLA
metadata mentions oneAPI `2025.1` and Level Zero `1.21.10`; PJRT reports
`oneapi 2026.0`; Intel oneCCL 2022.0 release notes target oneAPI 2026.0. A
oneCCL upgrade should not be accepted until this matrix is made explicit.
