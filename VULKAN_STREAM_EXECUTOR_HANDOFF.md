# Vulkan StreamExecutor handoff

## Current state

The worktree contains an uncommitted, minimal Vulkan StreamExecutor runtime and
the PJRT/XLA registrations needed to expose it as platform `"vulkan"`.

Implemented under `xla/stream_executor/vulkan/`:

- Vulkan platform and executor registration.
- Dynamic loading of `libvulkan.so.1` (there is no link-time Vulkan loader
  dependency).
- Vulkan 1.2 instance creation, physical-device discovery, compute queue and
  logical-device setup.
- Host-visible storage-buffer allocation and synchronous host/device and
  device/device copies.
- Synchronous StreamExecutor streams and events backed by one-time Vulkan
  command buffers.
- Loading `VulkanSpirvInMemory`, creating shader modules, descriptor layouts,
  descriptor sets, pipeline layouts, and compute pipelines.
- Dispatch, memory barriers, and queue synchronization.

Also added:

- The hermetic Bzlmod Vulkan headers dependency in `MODULE.bazel`.
- Vulkan compiler registration for the generic StreamExecutor GPU PJRT client.
- Vulkan transfer-manager registration.
- Vulkan default computation-placer registration.
- Linkage from `//xla/pjrt/c:pjrt_c_api_gpu_plugin.so` to the Vulkan platform.
- Propagation of the requested PJRT platform name (`"vulkan"`) instead of
  reporting only a compile-time CUDA/ROCm/SYCL name.

## What was verified

These builds completed successfully before the final computation-placer edit:

```bash
cd /home/reesechong/xla
bazel --batch build --config=bzlmod \
  //xla/stream_executor/vulkan:vulkan_platform
bazel --batch build --config=bzlmod \
  //xla/pjrt/c:pjrt_c_api_gpu_plugin
```

The ZML test then progressed through two useful checkpoints:

1. Vulkan `PlatformManager` registration worked. The old error saying no
   platform named `vulkan` disappeared.
2. After adding the Vulkan transfer manager, the missing-transfer-manager error
   disappeared.

The most recent ZML error was:

```text
Could not find registered computation placer for platform VULKAN
```

The computation-placer registration was then added in
`xla/service/computation_placer.cc` and its Bazel dependency was added, but the
large plugin rebuild was deliberately interrupted at approximately
19,352/19,813 actions. No compile error had occurred.

## Resume here

First finish the XLA plugin build:

```bash
cd /home/reesechong/xla
bazel --batch build --config=bzlmod \
  //xla/pjrt/c:pjrt_c_api_gpu_plugin
```

Then run the existing ZML integration test. The ZML local repository rule
already stages the XLA artifact at
`/home/reesechong/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so`.

```bash
cd /home/reesechong/zml
bazel --batch run \
  --noexperimental_collect_system_network_usage \
  --//platforms:vulkan=true \
  //examples/llm:vulkan_test
```

The network-usage flag avoids an unrelated Bazel sandbox failure seen in this
environment.

## Likely next integration gaps

Treat the next ZML failure as the next missing PJRT/StreamExecutor contract,
not as evidence that ZML failed to load the plugin. The log already proves the
shared object loads and the `vulkan` platform registers.

Check these likely assumptions in this order:

1. **Allocator selection.** The generic GPU PJRT client may select CUDA/ROCm
   virtual-memory allocation. Vulkan currently implements ordinary
   StreamExecutor address allocation only. If client creation reports a VMM or
   allocator error, force the Vulkan platform to use the address/platform
   allocator in `xla/pjrt/gpu/se_gpu_pjrt_client.cc`.
2. **Plugin topology metadata.** Some C API topology paths still choose a
   compile-time CUDA/ROCm/SYCL identity. If the error mentions topology or a
   mismatched platform ID, add Vulkan handling in the corresponding
   `xla/pjrt/c` GPU topology code.
3. **Unsupported generic GPU services.** This minimal executor intentionally
   has no BLAS, DNN, FFT, collectives, peer access, command graphs, priorities,
   or asynchronous staging. Keep the first test to buffer allocation/copy and a
   simple elementwise compute kernel.
4. **Vulkan availability.** The runtime expects `libvulkan.so.1` and at least
   one physical device with a compute-capable queue. Loader/device failures are
   returned explicitly during Vulkan executor initialization.
5. **Memory model.** Allocations currently require host-visible Vulkan memory
   and prefer host-coherent memory. This is suitable for initial buffer tests,
   but it is not a production discrete-GPU allocator.

## Runtime assumptions

- SPIR-V is Vulkan compute SPIR-V targeting Vulkan 1.2, not OpenCL/SYCL
  `SPIR_KERNEL` SPIR-V.
- Kernels use descriptor set 0 with storage-buffer descriptors. Descriptor
  binding metadata comes from `VulkanSpirvInMemory`.
- Dispatch is synchronous: commands are submitted and followed by queue-idle
  synchronization. This is intentional for the minimum buffer-test milestone.
- `DeviceAddressBase::opaque()` is a mapped host pointer. Allocation lookup
  translates that pointer (including sliced offsets) back to a `VkBuffer` and
  Vulkan byte offset when building descriptors.
- The compiler fixes local workgroup size in SPIR-V; runtime dispatch uses XLA
  block dimensions as Vulkan workgroup counts.

## Before committing

Run formatting and focused builds/tests, inspect all uncommitted changes, and
do not discard the earlier Vulkan compiler work in the same worktree:

```bash
cd /home/reesechong/xla
git status --short
git diff --check
bazel --batch build --config=bzlmod \
  //xla/stream_executor/vulkan:vulkan_platform \
  //xla/pjrt/c:pjrt_c_api_gpu_plugin
```

Add a device-independent platform-registration unit test and, where a Vulkan
device is available, a small allocation/copy/dispatch test before considering
the runtime complete.
