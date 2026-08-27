# XLA GPU command buffers: code map and implementation notes

This document is a source-oriented handoff for an agent investigating or changing
XLA's command-buffer implementation. It describes the repository at commit
`e8ac27565277eb5bf0b2f12a45cf63fe2863470a` plus the documented working-tree
observability, VMM/PJRT integration, and ZML validation changes (2026-08-27). NVIDIA/CUDA is the
primary focus; ROCm and the platform-neutral boundaries are included so that a
change is not accidentally designed around a CUDA-only property.

In this code, an XLA "command buffer" is a reusable, explicitly constructed GPU
execution graph: a CUDA Graph on NVIDIA and a HIP Graph on AMD. It is not a
serialized command list. XLA records a stable graph topology once, updates node
parameters when device addresses or other dynamic inputs change, and submits the
whole graph with one host launch.

## Executive summary

- The implementation has three layers: GPU thunk conversion and lifetime
  management, a platform-neutral StreamExecutor API plus shared GPU graph logic,
  and CUDA/HIP driver adapters.
- The main path is
  `ThunkSequence -> CommandBufferConversionPass -> CommandSequence ->
  CommandExecutor -> stream_executor::CommandBuffer -> CUgraphExec/hipGraphExec`.
- Graph topology is immutable after creation. An update must replay the same
  command structure and update existing node parameters; it cannot add, remove,
  or reorder nodes.
- NVIDIA is the complete implementation. It supports explicit conditionals and
  while loops on CUDA 12.3+, Programmatic Dependent Launch (PDL) edges on CUDA
  12.3+, and ownership-moving child graphs on CUDA 12.9+.
- ROCm supports explicit kernels, D2D copies, memsets, barriers, cloned child
  graphs, graph updates, and stream tracing. It does not implement graph
  conditionals/while, moved child graphs, explicit DNN graph nodes, PDL edges,
  or per-node priorities.
- Operations are recorded in one of three ways: explicit node construction,
  tracing a stream into a nested graph, or explicit nested control-flow/child
  graphs. Explicit construction is preferred because its parameters can be
  updated in place and it composes predictably.
- XLA infers graph dependencies from buffer read/write conflicts and explicit
  resource tokens. The default scheduling mode is `LHS`, which preserves the
  latency-hiding scheduler's intended overlap.
- The default address policy is `ALWAYS_UPDATE`. Two VMM-backed modes can give
  selected allocations stable virtual addresses and avoid many graph updates.
- An experimental primary-graph LRU can instead retain several finite,
  recurring device-address specializations. It is controlled by
  `--xla_gpu_command_buffer_cache_size`; the default of 1 preserves the former
  update-in-place behavior.
- Unsupported or too-small regions stay as ordinary thunks. During some
  profiling and warm-up paths, the command-buffer thunk deliberately executes
  its retained sequential fallback.

## End-to-end map

```text
HLO scheduling and buffer assignment
  xla/service/gpu/{gpu_hlo_ordering,compile_module_to_llvm_ir}.cc
                         |
                         v
GPU thunk sequence produced by lowering
                         |
                         v
CommandBufferConversionPass groups eligible consecutive thunks
  xla/backends/gpu/runtime/command_buffer_conversion_pass.cc
             | unsupported / too small              | converted region
             v                                      v
       ordinary thunks                    CommandBufferThunk
                                           + fallback SequentialThunk
                                                   |
                                      CommandBufferCmdEmitter
                                                   |
                                          CommandSequence
                                                   |
                              CommandExecutor builds dependency DAG
                                                   |
                                 explicit Record(Create or Update)
                                                   |
                          stream_executor::CommandBuffer interface
                                                   |
                              GpuCommandBuffer shared graph logic
                                      /                    \
                        CudaCommandBuffer            RocmCommandBuffer
                        CUgraph/CUgraphExec           hipGraph/hipGraphExec
                                      \                    /
                                       one graph submission
```

There is also a trace path for library calls or external implementations that
cannot explicitly add nodes:

```text
TracedCommand -> TracedCommandBuffer address-keyed cache
              -> TraceCommandBufferFactory
              -> capture a borrowed stream into a nested GPU graph
              -> clone that graph into the primary graph
```

## Source index

### Configuration, compilation, and conversion

| File | Why it matters |
| --- | --- |
| [`xla/xla.proto`](xla/xla.proto) | Defines command categories, scheduling modes, update modes, and their intended semantics. |
| [`xla/debug_options_flags.cc`](xla/debug_options_flags.cc) | Sets defaults and parses `--xla_gpu_*` command-buffer flags. |
| [`xla/service/gpu/gpu_executable.cc`](xla/service/gpu/gpu_executable.cc) | Installs `CommandBufferConversionPass` in the GPU thunk-pass pipeline and supplies a separate borrowed stream for capture. |
| [`xla/backends/gpu/runtime/command_buffer_conversion_pass.h`](xla/backends/gpu/runtime/command_buffer_conversion_pass.h) | Conversion-pass contract and options. |
| [`xla/backends/gpu/runtime/command_buffer_conversion_pass.cc`](xla/backends/gpu/runtime/command_buffer_conversion_pass.cc) | Eligibility, grouping, async-region balancing, version/platform feature gates, and recursive control-flow conversion. Start here to answer "why was this thunk not captured?" |
| [`xla/backends/gpu/runtime/command_buffer_cmd_emitter.cc`](xla/backends/gpu/runtime/command_buffer_cmd_emitter.cc) | Translates converted thunks into `Command` objects and nested executors; flattens async starts, filters async-done markers, and implements concurrent-region two-lane scheduling. |

### Runtime command model and graph lifetime

| File | Why it matters |
| --- | --- |
| [`xla/backends/gpu/runtime/command.h`](xla/backends/gpu/runtime/command.h) | Base `Command`, `CommandSequence`, create/update `RecordAction`, buffer/resource usage, warm-up/update requirements, and stateless-command invariant. |
| [`xla/backends/gpu/runtime/command_state.h`](xla/backends/gpu/runtime/command_state.h) and [`.cc`](xla/backends/gpu/runtime/command_state.cc) | Per-command, per-command-buffer typed mutable state. State lives with the executable graph instead of leaking through cached `Command` objects. |
| [`xla/backends/gpu/runtime/command_executor.h`](xla/backends/gpu/runtime/command_executor.h) and [`.cc`](xla/backends/gpu/runtime/command_executor.cc) | Builds dependencies, records the graph, selects create versus update, and skips updates whose referenced allocations did not change. |
| [`xla/backends/gpu/runtime/command_buffer_thunk.h`](xla/backends/gpu/runtime/command_buffer_thunk.h) and [`.cc`](xla/backends/gpu/runtime/command_buffer_thunk.cc) | Owns graph state per `StreamExecutor`, retains the sequential fallback, performs warm-up/profiling fallback, compares allocation addresses, updates, submits, and globally evicts old instantiated graphs. |
| [`xla/backends/gpu/runtime/traced_command.h`](xla/backends/gpu/runtime/traced_command.h) and [`.cc`](xla/backends/gpu/runtime/traced_command.cc) | Base for commands implemented by stream capture. |
| [`xla/backends/gpu/runtime/traced_command_buffer.h`](xla/backends/gpu/runtime/traced_command_buffer.h) and [`.cc`](xla/backends/gpu/runtime/traced_command_buffer.cc) | MRU cache of traced nested graphs keyed by relevant device addresses. |
| [`xla/backends/gpu/runtime/conditional_thunk.cc`](xla/backends/gpu/runtime/conditional_thunk.cc) | Device-side graph conditional recording and ordinary-thunk host-synchronized fallback. |
| [`xla/backends/gpu/runtime/while_thunk.cc`](xla/backends/gpu/runtime/while_thunk.cc) | Device-side graph while recording and optional known-trip-count unrolling. |
| [`xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.cc`](xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.cc) | Recomputes dynamic embedded-buffer addresses and records/updates the CUDA 12.9 moved-child graph. |
| [`xla/backends/gpu/runtime/cudnn_thunk.cc`](xla/backends/gpu/runtime/cudnn_thunk.cc) | Selects native cuDNN graph populate/update or the generic traced-command path. |
| [`xla/backends/gpu/runtime/custom_call_thunk.cc`](xla/backends/gpu/runtime/custom_call_thunk.cc) | FFI record API path, capture fallback, and custom-call eligibility traits. |
| [`xla/backends/gpu/runtime/record_ffi.h`](xla/backends/gpu/runtime/record_ffi.h) and [`.cc`](xla/backends/gpu/runtime/record_ffi.cc) | Bridges the FFI C record extension to explicit StreamExecutor graph commands. The public ABI is in [`xla/ffi/api/record_c_api.h`](xla/ffi/api/record_c_api.h). |
| [`xla/backends/gpu/runtime/collective_thunk.cc`](xla/backends/gpu/runtime/collective_thunk.cc) | Collective-specific synchronized tracing, initialization updates, and warm-up behavior. |

Concrete command implementations live beside their thunks. Useful examples are
[`kernel_thunk.cc`](xla/backends/gpu/runtime/kernel_thunk.cc),
[`device_to_device_copy_thunk.cc`](xla/backends/gpu/runtime/device_to_device_copy_thunk.cc),
[`memset_thunk.cc`](xla/backends/gpu/runtime/memset_thunk.cc), and
[`replica_id_thunk.cc`](xla/backends/gpu/runtime/replica_id_thunk.cc).

### StreamExecutor abstraction and shared GPU implementation

| File | Why it matters |
| --- | --- |
| [`xla/stream_executor/command_buffer.h`](xla/stream_executor/command_buffer.h) | Platform-neutral API, opaque node handles, primary/nested modes, create/update methods, state machine, resources, control flow, and submission contract. |
| [`xla/stream_executor/command_buffer.cc`](xla/stream_executor/command_buffer.cc) | Type-keyed resources attached to a graph and their locking. |
| [`xla/stream_executor/stream_executor.h`](xla/stream_executor/stream_executor.h) | `CreateCommandBuffer`; the default implementation returns `Unimplemented`. |
| [`xla/stream_executor/trace_command_buffer_factory.h`](xla/stream_executor/trace_command_buffer_factory.h) and [`.cc`](xla/stream_executor/trace_command_buffer_factory.cc) | Creates and finalizes a nested graph by stream capture. Direct `Trace` is private because a traced graph cannot participate in ordinary parameter updates; callers compose it as a child. |
| [`xla/stream_executor/gpu/gpu_command_buffer.h`](xla/stream_executor/gpu/gpu_command_buffer.h) and [`.cc`](xla/stream_executor/gpu/gpu_command_buffer.cc) | Shared node bookkeeping, dependency conversion, control-flow graph construction, child-graph policies, finalization, graph counters, and graph dump hooks. |
| [`xla/stream_executor/gpu/gpu_command_buffer_listener.h`](xla/stream_executor/gpu/gpu_command_buffer_listener.h) and [`.cc`](xla/stream_executor/gpu/gpu_command_buffer_listener.cc) | Global/thread-local telemetry-listener registration for graph/node/annotation lifecycle events. At this snapshot only its tests call the API; graph construction is not yet wired to emit these events. |
| [`xla/stream_executor/gpu/scoped_command_buffer_annotation.h`](xla/stream_executor/gpu/scoped_command_buffer_annotation.h) and [`.cc`](xla/stream_executor/gpu/scoped_command_buffer_annotation.cc) | Thread-local annotation stack intended to associate HLO/thunk labels with graph nodes. It is likewise scaffolding with no non-test call sites at this snapshot. |

### NVIDIA/CUDA implementation

| File | Why it matters |
| --- | --- |
| [`xla/stream_executor/cuda/cuda_executor.cc`](xla/stream_executor/cuda/cuda_executor.cc) | Creates `CudaCommandBuffer`. |
| [`xla/stream_executor/cuda/cuda_command_buffer.h`](xla/stream_executor/cuda/cuda_command_buffer.h) and [`.cc`](xla/stream_executor/cuda/cuda_command_buffer.cc) | `CUgraph`/`CUgraphExec` adapter: nodes, parameter updates, capture, instantiate, launch, error recovery, conditional nodes, priorities, cluster dimensions, and PDL edges. |
| [`xla/stream_executor/cuda/cuda_command_buffer_12_9.cc`](xla/stream_executor/cuda/cuda_command_buffer_12_9.cc) | CUDA 12.9 ownership-moving child-graph node and version eligibility. |
| [`xla/stream_executor/cuda/command_buffer_kernels.h`](xla/stream_executor/cuda/command_buffer_kernels.h) and [`.cc`](xla/stream_executor/cuda/command_buffer_kernels.cc) | Auxiliary kernels that set CUDA conditional handles for `case`/`while` and provide no-op workarounds. |
| [`xla/stream_executor/cuda/cuda_dnn.cc`](xla/stream_executor/cuda/cuda_dnn.cc) | cuDNN frontend native CUDA Graph populate/update bridge. |
| [`xla/stream_executor/dnn.h`](xla/stream_executor/dnn.h) | Platform-neutral DNN graph command-buffer hooks and capability query. |

### ROCm and other platforms

| File | Why it matters |
| --- | --- |
| [`xla/stream_executor/rocm/rocm_executor.cc`](xla/stream_executor/rocm/rocm_executor.cc) | Creates `RocmCommandBuffer`. |
| [`xla/stream_executor/rocm/rocm_command_buffer.h`](xla/stream_executor/rocm/rocm_command_buffer.h) and [`.cc`](xla/stream_executor/rocm/rocm_command_buffer.cc) | HIP Graph adapter and the authoritative list of currently unimplemented ROCm graph features. |
| [`xla/stream_executor/cuda/cuda_device_address_vmm_allocator.cc`](xla/stream_executor/cuda/cuda_device_address_vmm_allocator.cc) | NVIDIA virtual-memory reservation/map/unmap implementation used by stable-address update modes. |
| [`xla/stream_executor/rocm/rocm_device_address_vmm_allocator.cc`](xla/stream_executor/rocm/rocm_device_address_vmm_allocator.cc) | AMD counterpart, showing that VA-remap policy is intentionally cross-GPU. |

At this snapshot, CUDA and ROCm are the only production `StreamExecutor`
implementations in the tree that override `CreateCommandBuffer`. Other
StreamExecutor platforms inherit the default `Unimplemented` result. The thunk
conversion/lifetime subsystem is under the GPU backend, so CPU and TPU do not
use this mechanism merely because the abstract API exists. The shared GPU test's
BUILD target is explicitly disabled for oneAPI with a TODO to implement its GPU
command buffer, which is evidence of intended extension rather than present
support.

### Address stability, scheduling, and tests

| File | Why it matters |
| --- | --- |
| [`xla/service/gpu/gpu_executable_buffer_allocator.h`](xla/service/gpu/gpu_executable_buffer_allocator.h) and [`.cc`](xla/service/gpu/gpu_executable_buffer_allocator.cc) | Finds command-buffer-referenced allocations and selects the base or VA-remapping allocator. |
| [`xla/service/gpu/gpu_executable_va_remap_allocator.h`](xla/service/gpu/gpu_executable_va_remap_allocator.h) and [`.cc`](xla/service/gpu/gpu_executable_va_remap_allocator.cc) | Fixed virtual-address reservations, profiling of stable buffers, execution-time aliases, and update filtering. |
| [`xla/stream_executor/device_address_vmm_allocator.h`](xla/stream_executor/device_address_vmm_allocator.h) and [`.cc`](xla/stream_executor/device_address_vmm_allocator.cc) | Shared VMM allocation/reclamation state, physical-memory accounting, and delegation for memory spaces such as pinned host memory that device VMM cannot back. |
| [`xla/pjrt/gpu/se_gpu_pjrt_client.cc`](xla/pjrt/gpu/se_gpu_pjrt_client.cc) | Creates the GPU allocator, joins VMM device allocation with the pinned-host allocator, and exposes VMM physical-memory statistics through PJRT. |
| [`xla/service/gpu/gpu_hlo_ordering.cc`](xla/service/gpu/gpu_hlo_ordering.cc) | Defines concurrent-region boundaries and large-operation heuristics. |
| [`xla/service/gpu/compile_module_to_llvm_ir.cc`](xla/service/gpu/compile_module_to_llvm_ir.cc) | Connects scheduling mode to HLO ordering and therefore to buffer reuse. |
| [`xla/stream_executor/gpu/gpu_command_buffer_test.cc`](xla/stream_executor/gpu/gpu_command_buffer_test.cc) | Cross-GPU low-level graph API tests: explicit nodes, tracing, nesting, control flow, barriers, and resources. |
| [`xla/stream_executor/cuda/cuda_command_buffer_test.cc`](xla/stream_executor/cuda/cuda_command_buffer_test.cc) | CUDA-specific PDL, clusters, cuDNN graph, and capture behavior. |
| [`xla/backends/gpu/runtime/command_buffer_conversion_pass_test.cc`](xla/backends/gpu/runtime/command_buffer_conversion_pass_test.cc) | Conversion categories, filters, minimum sizes, async regions, and control flow. |
| [`xla/backends/gpu/runtime/command_executor_test.cc`](xla/backends/gpu/runtime/command_executor_test.cc) | Dependency DAGs and selective allocation updates. |
| [`xla/backends/gpu/runtime/command_buffer_thunk_test.cc`](xla/backends/gpu/runtime/command_buffer_thunk_test.cc) | Create/update/skip, persistence, fallback, profiling, GEMM, kernels, and control flow. |
| [`xla/backends/gpu/tests/command_buffer_test.cc`](xla/backends/gpu/tests/command_buffer_test.cc) | End-to-end GPU coverage across scheduling modes, fusions, libraries, custom calls, control flow, async execution, collectives, and profiling. |

## What is converted

`CommandBufferConversionPass` scans a thunk sequence, groups consecutive
eligible thunks, and replaces a group with a `CommandBufferThunk`. It flushes a
group at an unsupported thunk. A group must contain at least
`max(1, --xla_gpu_graph_min_graph_size)` thunks; the default is 5. This threshold
counts thunks before graph lowering, not final CUDA/HIP graph nodes.

The category list can be specified absolutely or adjusted relative to defaults
with `+TYPE` and `-TYPE`. Defaults come from `debug_options_flags.cc`.

| Category | Representative thunk kinds | Default | Main recording strategy and restrictions |
| --- | --- | --- | --- |
| `FUSION` | Kernel, custom kernel, replica/partition ID, D2D copy | On | Explicit kernel/copy/memset nodes. Host/device copies are not eligible. |
| `CUBLAS` | GEMM | On | Usually a `TracedCommand`; cached by relevant buffer addresses. |
| `CUBLASLT` | cuBLASLt matmul | On | Stream tracing. |
| `CUDNN` | `CuDnnThunk` | On | Native cuDNN CUDA Graph populate/update when supported, otherwise tracing. ROCm's explicit DNN graph hook is unimplemented. |
| `CONVOLUTION` | Convolution thunk | Off | Stream tracing. |
| `CUSTOM_CALL` | Typed FFI custom call | On | Only GPU FFI handlers marked command-buffer-compatible. Prefer the FFI record extension; it may request tracing. Legacy custom calls are rejected. |
| `DYNAMIC_SLICE_FUSION` | Dynamic-slice fusion v2 | On | Callback-owned, ownership-moving child graph; CUDA 12.9+ only. Loop-dependent slices need loop unrolling and a known trip count. |
| `CONDITIONAL` | Conditional thunk | On | Explicit device-side graph conditional; CUDA 12.3+ only. |
| `WHILE` | While thunk | Off | Explicit graph while, or repeated child commands when unrolled; CUDA 12.3+ for the graph conditional form. |
| `COLLECTIVES` | All-gather/reduce/to-all, broadcast, permute, ragged all-to-all, reduce-scatter, send/recv | Off | Collective-specific capture and synchronization. Also filtered by `--xla_gpu_enable_collectives_command_buffer_filter` (default `ALLCOLLECTIVES`). Ragged all-to-all has additional one-shot/intra-node checks. |

`MemzeroCommand` and `Memset32Command` implement the explicit command API and are
used by individual runtime components/tests, but the conversion pass does not
currently classify the ordinary memset thunk kinds as `FUSION`. Existence of a
`Command` implementation therefore does not by itself mean automatic conversion
will select it.

### Async and nested regions

- An async start/done region is convertible only if it is balanced and every
  covered operation is convertible. The pass groups the shortest enclosing
  region when async intervals overlap or cross.
- The command emitter flattens an async-start nested sequence into the graph.
  Async-done is a scheduling marker and emits no graph node.
- The pass recursively converts convertible regions inside otherwise
  non-convertible conditional branches and while bodies.
- A dynamic-slice-fusion-v2 nested sequence is eligible only when the backend can
  move ownership of a child graph; today that means CUDA 12.9+.
- On CUDA older than 12.3, the pass removes categories that need tracing and
  graph conditionals. On ROCm, it removes conditional/while conversion. These
  are conversion-time gates in addition to backend methods returning an error.

## Runtime lifecycle

### 1. Command objects describe stable work

A runtime `Command` is the command-buffer counterpart of a thunk. It has three
major stages:

1. `Prepare`: host-side preparation.
2. `Initialize`: executor-specific setup, potentially synchronized across ranks.
3. `Record`: add a node during `RecordAction::Create`, or update the corresponding
   node during `RecordAction::Update`.

Commands must be thread-safe and effectively stateless. Mutable state that is
specific to a command and one instantiated graph belongs in
`CommandStateManager`, keyed by `(Command*, CommandBuffer*, TypeId)`. This is an
important lifetime rule: storing per-graph state in a long-lived command would
leak it when XLA evicts graphs.

Each command reports:

- device buffers it reads or writes;
- resource tokens it reads or writes;
- whether it needs an update after initialization;
- whether it needs a warm-up execution;
- whether it needs an update on every execution; and
- whether it is trace-based or supports loop unrolling.

`CommandSequence` owns converted commands but may also borrow command/thunk
hybrids owned by its retained fallback `SequentialThunk`.

### 2. CommandExecutor derives a DAG

Except in `SERIALIZE` mode, `CommandExecutor` creates an `ExecutionGraph` from
buffer and resource uses. A write conflicts with earlier reads/writes to the
same allocation slice; explicit resource tokens encode non-buffer ordering such
as scheduler or collective constraints. Every command also writes its own token
so additional control edges can name it.

When creating the StreamExecutor graph, the executor passes the recorded handles
of dependency commands to the next command. A legitimate no-op (for example a
zero-byte copy) may return a null command handle. Dependency resolution walks
through such null nodes to their predecessors, preserving ordering without
inventing a GPU node.

The executor records the first time with `Create`, saves each returned opaque
handle as a resource attached to the command buffer, and finalizes it. Later it
replays the same command sequence with `Update` and finalizes the update.

### 3. CommandBufferThunk owns and submits graphs

One `CommandBufferThunk` contains:

- the `CommandExecutor` and its `CommandSequence`;
- the original sequential thunk sequence as a fallback;
- a mutex-protected state map with one primary command-buffer cache per
  `StreamExecutor`;
- per-entry command state, a primary graph, its recorded allocation signature,
  and an execution counter; and
- a cache-wide warm-up state.

The mutex matters because graph mutation and submission are not treated as
thread-safe. The per-executor map lets one compiled executable be used with more
than one device executor.

On execution, the thunk follows this decision sequence:

1. Run the sequential path when profiling is active and
   `--xla_enable_command_buffers_during_profiling=false` (the default).
2. Run a one-time sequential warm-up when any command requires it.
3. Look up the current non-persistent allocation signature in the primary-graph
   LRU.
4. Reuse a hit, create a graph on a miss with room, or retarget the least
   recently used graph on a full miss.
5. Update a hit only when a command explicitly requires per-execution mutation
   or an initialization/persistent-allocation transition requires it.
6. Submit the primary graph as one launch.

Initialization tries to instantiate/record graphs before execution, notably to
avoid graph-allocation deadlocks when NCCL graphs are built concurrently across
ranks. Some collective initialization updates must also be recorded by all
ranks. When persistent-allocation information is temporarily unavailable in the
`SKIP_PROFILED` observation window, execution intentionally uses the fallback.

Creating a new `CommandBufferThunk` evicts previously instantiated command
buffers through a process-wide weak-state registry. The rationale in the source
is that higher-level caches can retain roughly thousands of executables, while
GPU graph definitions cost memory (the shared implementation gives a rough
rule-of-thumb of 8 KiB per node). Reconstructing the active few graphs is cheaper
than keeping all of them resident.

## Scheduling and dependencies

| Mode | Graph ordering | Buffer-assignment implication |
| --- | --- | --- |
| `SERIALIZE` | Every command depends on its immediate predecessor. | Maximum reuse, no graph-level overlap. |
| `CONCURRENT` | Only inferred buffer/resource conflicts impose edges. | Uses dependency-based HLO ordering and keeps independent buffers distinct; can increase memory, cause OOMs, and add performance variability. |
| `LHS` (default) | Reconstructs latency-hiding-scheduler intent. Normal/start commands depend on the nearest prior non-async-start; async done depends on its matching start and, when distinct, its immediate predecessor. | Compilation uses the ordinary sequential HLO ordering while command tokens preserve intended async overlap. |
| `CONCURRENT_REGIONS` | Splits execution into regions. Small latency-bound regions use a two-lane list scheduler; every source in the next region depends on every sink in the prior region. | Region-aware HLO ordering prevents unsafe reuse only where overlap is planned. |

The two-lane limit in concurrent regions is deliberate: it reduces jitter from
excessive concurrency. `gpu_hlo_ordering.cc` treats custom calls and large fusions
as region boundaries. A fusion is considered large when it contains a dot or
convolution, has more than 100 fused instructions, or accounts for more than
20,000,000 bytes under the file's size estimate.

The important cross-layer consequence is that scheduling is not only a graph
edge choice. `CONCURRENT` and `CONCURRENT_REGIONS` can alter buffer assignment so
independent work has distinct storage. Changing graph scheduling without
checking HLO ordering and buffer reuse is incomplete.

## Recording strategies

### Explicit nodes: preferred

Explicit commands call `CreateLaunch`, `CreateMemset`, `CreateMemcpyDeviceToDevice`,
and corresponding `Update*` APIs. The first recording adds stable graph nodes;
subsequent recordings set executable-node parameters in place.

Use this path for new operations whenever the implementation owns the kernel or
can express the library graph explicitly. It gives XLA exact dependencies,
avoids capture restrictions, and allows parameter updates without retracing.

The typed FFI record extension is the external analogue. A custom call can add
or update kernels, add or update D2D copies, emit an empty join node, or request
stream capture. The implementation stores its returned handles in per-graph
command state. At this snapshot the C API permits at most 16 commands, and the
last returned command must be a sink representing all emitted work; clients
with independent tails insert an empty join node.

### Stream tracing: compatibility path

`TracedCommand` is used when XLA cannot explicitly construct the underlying
nodes, especially for library calls. It captures work on a borrowed stream into
a finalized nested graph, then clones that child into the primary graph.

A traced graph cannot update its internals through the generic command-buffer
API. `TracedCommandBuffer` therefore keeps an MRU cache keyed by the exact device
addresses of buffers relevant to that command. A cache hit reuses a previously
captured nested graph; a miss retraces. The default cache capacity is 16
(`--xla_cmd_buffer_trace_cache_size`). This makes accurate buffer-use reporting
both a correctness and performance requirement.

Collectives use a specialized capture path rather than the general address-keyed
cache. Capture occurs inside communicator coordination because participating
ranks must enter graph recording consistently. Collective commands also request
initialization updates and warm-up where required.

Tracing should be treated as an opaque compatibility layer:

- any first-call initialization should happen in `Prepare`, `Initialize`, or a
  warm-up rather than unexpectedly during capture;
- operations forbidden by CUDA/HIP stream capture will fail;
- dynamic scalar or non-address state may require explicit
  `requires_update_on_execute` handling; and
- a topology change means retracing or a different explicit child graph, not a
  normal node-parameter update.

### Child graphs and control flow

There are two child-graph ownership models:

1. **Cloned child:** `CreateChildCommand(nested)` clones the nested graph into the
   parent. Updating replaces the child graph as a unit. This is the portable
   CUDA/ROCm form and is what tracing uses.
2. **Moved child:** a callback creates a nested graph whose ownership is moved
   into the parent. Individual inner nodes can later be updated through the
   parent's executable graph. This is available only on CUDA 12.9+ and powers
   dynamic-slice-fusion-v2 and unrolled child sequences.

For a `case`, the shared layer creates CUDA conditional handles, a kernel that
sets those handles from the device selector, and an IF conditional node per
branch. The auxiliary setter kernel handles at most eight branches, so larger
cases are emitted in batches of eight. An out-of-range selector chooses the last
branch, implemented only in the final batch. Dependencies on the logical case
command expand to all of its branch conditional nodes. Boolean conditionals need
one extra translation detail: `ConditionalThunk` stores false then true, while
the graph case API expects the true branch first, so the command emitter reverses
the two branch executors.

For a `while`, the parent first records commands that compute the condition,
then an auxiliary kernel transfers the device predicate to a conditional handle,
then a WHILE conditional node owns a nested body graph. The nested graph ends by
recomputing and resetting the condition. No device-to-host predicate transfer is
needed. The ordinary non-graph `WhileThunk`/`ConditionalThunk` fallback does use
host-side predicate handling and synchronization.

When `--xla_gpu_command_buffer_unroll_loops=true`, a loop with a known trip count
and fully supported nested commands can instead record repeated condition/body
commands into a moved child graph. Each iteration gets its own `RecordId` so
per-iteration state does not collide.

## NVIDIA/CUDA inner workings

### Definition versus executable

`CudaCommandBuffer` owns a `CUgraph` definition. A primary buffer also owns a
`CUgraphExec`; a nested buffer does not. The abstract state machine is:

```text
kCreate --Finalize--> kFinalized --Update--> kUpdate --Finalize--> kFinalized
```

During the first primary `Finalize`, CUDA instantiates the executable graph with
`CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY`. A nested `Finalize` only seals
the definition. During update finalization XLA does not instantiate again: each
`Update*` call has already changed parameters in the top-level `CUgraphExec`
using APIs such as `cuGraphExecKernelNodeSetParams`,
`cuGraphExecMemcpyNodeSetParams`, `cuGraphExecMemsetNodeSetParams`, or the child
graph setter. A nested command finds the top-level executable through its parent
pointer.

Submission is `cuGraphLaunch`. Nested command buffers cannot be submitted.

If instantiation fails with resource exhaustion, the CUDA implementation calls
`cuDeviceGraphMemTrim` and retries once. The shared layer reports an error with
guidance about GPU graph memory pressure. CUDA DOT rendering is available via the
driver debug API and XLA's dump hooks.

### CUDA feature gates

| Feature | Minimum CUDA/tooling condition | Implementation detail |
| --- | --- | --- |
| Explicit kernels, copies, memsets, empty nodes, cloned children | Baseline supported CUDA in the build | Direct CUDA Graph node APIs. |
| Stream capture into an existing graph | Runtime and driver path at least 12.3 | Uses thread-local capture mode to avoid a process-wide driver stall. An empty capture gets an explicit empty node because an empty child graph can fail during instantiation. |
| IF/WHILE conditional nodes | Runtime and driver path at least 12.3 | `cuGraphConditionalHandleCreate` plus conditional-node parameters; device helper kernels set the handles. |
| PDL graph edges | Driver/toolkit support for the CUDA 12.3 node API | For a PDL-enabled kernel, an edge from a predecessor kernel uses programmatic edge ports/types. Older drivers warn and use an ordinary dependency. |
| Kernel node priority | CUDA implementation | Per-node priority attribute; graph instantiation opts into priorities. |
| Kernel cluster dimensions | CUDA implementation | Cluster-dimension node attribute. |
| Ownership-moving child graph | Compile-time toolkit, runtime, and driver all at least 12.9 | `CU_GRAPH_NODE_TYPE_GRAPH` with `CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE`; implemented separately in `cuda_command_buffer_12_9.cc`. |
| Native cuDNN graph populate/update | cuDNN frontend reports `SUPPORTS_CUDA_GRAPH_NATIVE_API` | cuDNN populates or updates the raw CUDA graph; otherwise `CuDnnThunk` uses tracing. |

The moved-child eligibility check intentionally uses the minimum of compile-time
toolkit support, runtime version, and driver version. Do not gate only on the
headers used to build XLA.

For CUDA drivers before 12.8, the shared/CUDA finalization path inserts a no-op
kernel into an otherwise empty conditional body, because those driver versions
do not support an empty conditional graph. The helper kernels in
`command_buffer_kernels.cc` are consequently part of correctness, not merely an
optimization detail.

One shared low-level test for a conditional nested in a while loop is disabled
with a TODO referencing `b/339653343`. Treat this specific composition as a test
coverage/driver-integration caveat when modifying nested control flow.

## Address changes and graph updates

An executable can receive different device addresses on different invocations.
`CommandBufferThunk` stores the allocation addresses used for the last recording.
`CommandExecutor::RecordUpdate` receives the changed allocation indices and skips
a command when all of the following hold:

- the changed set is known;
- it has no intersection with that command's referenced allocations;
- the command does not require an update on every execution; and
- no initialization event forces an update.

Constants are always persistent. Commands whose referenced allocations are all
persistent can also skip otherwise requested initialization updates. If the
changed set is unknown, XLA conservatively updates all commands.

The selectable update policies are:

| Mode | Behavior | Trade-off |
| --- | --- | --- |
| `ALWAYS_UPDATE` (default) | Uses ordinary dynamically allocated addresses and updates commands affected by address changes. | Simplest and broadly applicable; repeated graph-node update cost remains. |
| `SKIP_TEMP` | Gives command-buffer-referenced preallocated temp buffers fixed virtual addresses through a VMM reservation. Other dynamic buffers still cause updates. | Avoids updates for a common allocation class; requires the platform VMM allocator. |
| `SKIP_PROFILED` | Executes the fallback while observing the first three invocations, automatically includes temps, selects other stable nonconstant/non-thread-local allocations, then transitions once to a fixed reservation. | Can suppress more updates; adds profiling/fallback startup cost, serializes remap state, and assumes symmetric behavior across ranks. |

The VMM allocator reserves a deterministic virtual-address range per executable
and executor. Temp physical allocations are mapped directly into assigned
offsets. Selected input/output buffers remain owned by the ordinary allocation
table but are aliased into the reservation immediately before command-buffer
execution; the graph sees an execution-only address table containing the stable
addresses. Aliases are unmapped afterward. A mutex is held for the execution
using that remap state.

`SKIP_PROFILED` considers a candidate stable only if its non-null address remains
unchanged during the observation window, it is an exact allocation from the VMM
allocator, and it does not ambiguously share the same address. A wrong stability
prediction costs an extra remap rather than producing an incorrect graph. If VMM
is unavailable or no useful allocations are selected, the factory falls back to
the base allocation behavior.

This framework has both CUDA and ROCm `DeviceAddressVmmAllocator`
implementations. It is therefore a shared GPU optimization even though advanced
graph features are CUDA-only.

The top-level PJRT allocator must still support every advertised GPU memory
space. Device VMM cannot back pinned-host buffers: those buffers are CPU-written
staging memory. The working-tree integration delegates `MemorySpace::kHost` to
the normal host allocator and records the owning delegate by `(device ordinal,
address)` so PJRT's later top-level `Deallocate` call returns to the same
allocator. Without the allocation-time delegation, a CPU copy faults while
writing a device-only VA; without the deallocation routing, teardown reports
the host pointer as an unknown VMM address.

`DeviceAddressVmmAllocator::GetAllocatorStats` now reports physical allocation
usage and budget rather than the much larger reserved virtual-address space.
This makes PJRT `Device_MemoryStats` and `ClearMemoryStats` usable with VMM.
It is not graph-memory accounting: CUDA driver storage for `CUgraphExec` remains
opaque and is not included as a separately measurable pool.

### Primary address-specialization LRU (experimental, 2026-08-27)

`--xla_gpu_command_buffer_cache_size=N` sets the maximum number of primary
graphs retained by each `CommandBufferThunk` for each `StreamExecutor`. Values
below 1 are rejected. The upstream-compatible default is 1; use 64 for the
initial finite-set LLM experiment. Although the owner is platform-neutral GPU
code, the current performance and acceptance target is CUDA.

Each entry owns its primary `se::CommandBuffer` (and therefore its
`CUgraphExec` on CUDA), `CommandStateManager`, and a sparse, allocation-indexed
signature. The signature contains the pointer and size of every referenced
non-persistent allocation. Persistent allocations remain governed by the
existing address-policy contract and are excluded from the signature.

The cache vector is ordered most-recently-used first:

- An exact signature hit moves the entry to the front and submits it without an
  address-driven record/update.
- A miss with free capacity allocates and records a new primary graph, then
  leaves it at the front.
- A miss at capacity moves the least-recently-used entry to the front and
  retargets it with the existing selective update path. Capacity 1 is therefore
  the previous update-in-place behavior.
- A failed create or update discards the selected entry. Its proposed signature
  is committed only after recording succeeds, so partially updated state cannot
  become a future cache hit.

Warm-up remains once per thunk/executor cache, rather than once per address
specialization. Command-buffer regions containing a collective have an
effective capacity of 1: instantiating an additional graph during execution can
require cross-rank coordination and would risk a deadlock in this first version.
`requires_update_on_execute`, initialization updates, and persistent-allocation
policy transitions still update a cached graph for correctness and are reported
with a distinct update reason.

This primary cache is separate from `TracedCommandBuffer`, whose MRU cache holds
opaque nested graphs produced by stream capture. Increasing one cache does not
increase the other. It is also independent of VMM: VMM makes selected virtual
addresses stable, whereas this LRU keeps multiple executable graphs specialized
to ordinary addresses.

Capacity is per converted region and executor, not per executable or process.
An executable with many command-buffer regions can therefore retain
`regions * executors * capacity` primary graphs. Driver/GPU graph storage grows
accordingly; the existing rough estimate is 8 KiB per graph node. The current
process-wide policy is unchanged: constructing a new `CommandBufferThunk` can
clear all populated primary LRUs. There is no global byte budget, host-RAM graph
offload, serialization of `CUgraphExec`, or cold graph-definition tier. CUDA
owns opaque executable graph storage and XLA currently cannot page it to host
RAM; eviction followed by reconstruction is the available cold path.

## Platform comparison

| Capability | NVIDIA CUDA | AMD ROCm/HIP | Other StreamExecutor platforms |
| --- | --- | --- | --- |
| Primary executable graph | `CUgraph` + `CUgraphExec` | `hipGraph_t` + `hipGraphExec_t` | No production override in this snapshot |
| Explicit kernel node/create-update | Yes | Yes | Default `Unimplemented` |
| D2D copy/create-update | Yes | Yes | Default `Unimplemented` |
| Memset/create-update and empty node | Yes | Yes | Default `Unimplemented` |
| Stream tracing | Yes, driver 12.3+ | Yes | No implementation here |
| Cloned child graph/create-update | Yes | Yes | No implementation here |
| Moved child graph/in-place inner updates | CUDA 12.9+ | No | No |
| Device-side case/while | CUDA 12.3+ | No | No |
| PDL/programmatic edges | CUDA 12.3-capable path | No | No |
| Per-node priority | Yes | Ignored because HIP lacks an equivalent path here | No |
| Kernel cluster dimensions | Yes | Argument is currently ignored | No |
| Explicit DNN graph populate/update | cuDNN capability-dependent | Unimplemented | No |
| Fixed-VA update modes | CUDA VMM allocator | ROCm VMM allocator | Platform-dependent; not wired to this GPU path |
| Graph DOT/debug string | CUDA driver DOT output | `ToString` reports unsupported; a file helper calls HIP debug DOT print | No common backend |

ROCm capture replaces the implementation's initially empty owned graph with the
captured HIP graph and adds an empty node for an empty capture, paralleling CUDA.
It can instantiate, launch, and update the portable node subset. Conversion
removes conditional/while categories before reaching the unimplemented methods,
but new code should still return a clear capability error rather than assume the
pass is the only caller.

For portable feature work, target the explicit kernel/copy/memset/empty/cloned
child subset and keep priority, cluster, PDL, conditional, DNN-native, and
moved-child behavior behind capability checks.

## Invariants and common traps

1. **Do not change graph topology in update mode.** Create and update must visit
   the same logical commands in the same shape. Changing node count,
   dependencies, or child structure requires a new graph or traced-cache entry.
2. **Report exact buffer uses.** They drive dependency correctness, selective
   updates, traced-cache keys, referenced-allocation discovery, and sometimes
   buffer assignment. Omitting a use can cause a race or stale node parameters;
   over-reporting loses concurrency and causes needless updates/retraces.
3. **Keep `Command` objects stateless.** Put graph-local mutable data in
   `CommandStateManager` or attach a typed `CommandBuffer::Resource` so eviction
   destroys it with the graph.
4. **Handle no-op/null nodes.** A command may legitimately emit no GPU node.
   Dependency propagation must bypass it to its recorded predecessors.
5. **Distinguish primary and nested graphs.** Only a primary graph owns an
   executable and may be submitted. Nested updates reach the parent's executable.
6. **Treat tracing as address-specialized.** A cloned captured graph does not
   expose its internal nodes to generic update. Keep cache keys complete and
   retrace on relevant dynamic state.
7. **Coordinate collectives across ranks.** Initialization, warm-up, and capture
   can require every rank to participate even when local addresses look stable.
8. **Remember the sequential fallback.** It is required for unsupported gaps,
   profiling behavior, VMM profiling, and warm-up. A command-buffer-only change
   must not silently diverge from thunk semantics.
9. **Check build, runtime, and driver versions.** CUDA headers alone do not prove
   a deployed driver supports conditionals, capture-into-graph, or moved children.
10. **Minimum graph size counts thunks.** Do not infer the threshold from final
    graph node count, especially for library captures and batched conditionals.
11. **Scheduling affects memory planning.** A new concurrency edge policy may
    require corresponding HLO ordering/buffer-assignment changes.
12. **Per-execution state still needs update hooks.** Stable addresses do not
    eliminate updates for changing scalars, offsets, device predicates, or other
    parameters represented outside base allocation addresses.

## Decision guide for new work

When adding a command-buffer-capable GPU operation:

1. Implement an explicit `Command` with matching create/update paths if the work
   can be represented by known nodes. Use tracing only for an opaque library or
   external handler.
2. Define buffer and resource uses before deciding dependency edges manually.
   Let `CommandExecutor` derive ordinary hazards.
3. Store opaque node handles and other graph-local state in
   `CommandStateManager`; keep the command reusable across graphs/executors.
4. Add the thunk-kind/category mapping to `CommandBufferConversionPass` and
   decide deliberately whether it belongs in the default category list.
5. If the operation contains independent tail nodes, return/create a join node
   so the command's handle is a true sink.
6. Decide whether initialization, warm-up, or per-execution update is required.
7. For nested dynamic work, choose cloned children for portability or a moved
   child only when CUDA 12.9 is an acceptable hard feature gate.
8. Implement or explicitly reject ROCm behavior. Do not let a CUDA-only node
   attribute silently become a correctness requirement.
9. Test first record, address-changing update, address-stable skip, graph
   eviction/recreation, sequential fallback, nested use, and platform/version
   gates.

For performance decisions:

- Prefer `LHS` unless measurements and buffer-memory analysis justify another
  scheduling mode.
- Use `CONCURRENT` only with explicit memory-headroom validation; it changes
  buffer assignment, not just launch overlap.
- Use `CONCURRENT_REGIONS` for bounded overlap of small kernels; its two-lane
  policy is intentionally conservative.
- Evaluate `SKIP_TEMP` before `SKIP_PROFILED` when temp-address churn dominates;
  it has less startup complexity.
- Inspect update skips and trace-cache misses separately. A stable primary graph
  can still pay for repeated traced-child replacement if a library command's
  address key changes.

## Diagnostics and validation

- The conversion pass logs grouping and rejection decisions at verbose logging
  levels. The runtime emits `command_buffer::initialize`, `record`, `update`,
  `execute`, `cache_hit`, `cache_miss`, and `cache_evict` tracing scopes. Cache
  logs include entry count/capacity and cumulative hit/miss/eviction counts;
  record/update events include their reason.
- With HLO dumping enabled, the thunk pipeline can emit
  `thunk_sequence_after_thunk_passes.txt`, which shows the actual converted
  regions.
- `GpuCommandBuffer::Finalize` has high-verbosity graph dump hooks (VLOG 9/10).
  CUDA's `ToString` uses a temporary DOT dump; ROCm's string rendering is much
  less useful, although it has a HIP DOT-file helper.
- The low-level shared test suggests `--vmodule=gpu_command_buffer=100` when a
  graph visualization is needed during a test run.
- Start focused validation with the conversion-pass, command-executor,
  command-buffer-thunk, shared StreamExecutor GPU, and CUDA-specific tests listed
  in the source index. Use the end-to-end GPU command-buffer test for scheduling
  and fallback interactions.

For the finite-address-set experiment:

```bash
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_MAX_VLOG_LEVEL=0 \
TF_CPP_VMODULE=command_buffer_thunk=6 \
XLA_FLAGS='--xla_gpu_command_buffer_cache_size=64 \
  --xla_gpu_command_buffer_update_mode=SKIP_TEMP' \
  <model command>
```

`TF_CPP_VMODULE` must spell `command_buffer_thunk` completely. Setting the
global maximum to zero does not disable this module-specific override; it keeps
unrelated VLOG output quiet. Cache logs distinguish `cache hit`, `cache miss
(empty slot)`, and `cache miss (LRU replacement)`. Recording logs distinguish
first records from updates and include `new_cache_entry`, `cache_eviction`,
`persistent_allocation_policy`, `requires_update_on_initialize`,
`requires_update_on_execute`, or `allocation_addresses` as the reason.

The first traversal should report empty-slot misses and `Recorded` graphs.
Later traversals should report cache hits and executions only: no misses, LRU
replacements, or updates, except for commands explicitly reported with a
correctness-required update reason. Count both execution-time and
initialization-time updates; PJRT calls thunk initialization before each run, so
address-driven graph retargeting can appear as `Updated command buffer during
initialization` even when there is no execution-time `Updated` line.

The ZML Llama validation command was:

```bash
cd ~/github/zml/zml
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_MAX_VLOG_LEVEL=0 \
TF_CPP_VMODULE=command_buffer_thunk=6 \
XLA_FLAGS='--xla_gpu_command_buffer_cache_size=64 \
  --xla_gpu_command_buffer_update_mode=SKIP_TEMP' \
CUDA_VISIBLE_DEVICES=0 \
bazel run --config=alldebug --@zml//platforms:cuda=true //examples/llm -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct/ \
  --prompt=hi --seqlen=64 --topk=1
```

For a driver-level count, profile the already-built binary so Bazel is outside
the captured process:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/llama-cmdbuffer \
  bazel-bin/examples/llm/llm \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct/ \
  --prompt=hi --seqlen=64 --topk=1

nsys stats --report cuda_api_sum /tmp/llama-cmdbuffer.nsys-rep
```

The decisive no-update check is that
`cuGraphExecKernelNodeSetParams_v2` (and the corresponding memcpy/memset/child
setters, if present in a workload) is absent from the CUDA API summary after the
initial graph records.

Focused source validation used for this change:

```bash
USE_BAZEL_VERSION=7.7.0 bazel test --features=-module_maps \
  //xla/backends/gpu/runtime:command_buffer_conversion_pass_test_nvgpu_any \
  //xla/backends/gpu/runtime:command_buffer_thunk_test_nvgpu_any

USE_BAZEL_VERSION=7.7.0 bazel build //xla/pjrt/c:pjrt_c_api_gpu_plugin
```

### Experiment results: 2026-08-27, RTX 5090

- Focused CUDA conversion/thunk tests and the CUDA end-to-end
  `command_buffer_test_nvgpu_any` suite: passed. `--features=-module_maps` was
  needed locally to avoid duplicate `crosstool` module declarations in gRPC.
- Plugin build: the exact command above failed in the external gRPC `uuid_v4`
  compile because the configured host and CUDA toolchains both declare the
  Clang module `crosstool`. Adding `--features=-module_maps` built
  `bazel-bin/xla/pjrt/c/libpjrt_c_api_gpu_plugin.so` successfully. This is a
  local toolchain/module-map issue rather than an LRU source failure.
- The final 403 MiB plugin was packaged as `libpjrt_cuda.so` in
  `bazel-bin/xla/pjrt/c/pjrt-cuda_linux-amd64.tar.gz`. The deterministic archive
  used by the current working tree, including the later VMM/PJRT fixes, has
  SHA-256
  `48aacdb6b2326359bfc34ba5840eb3342cda5994e800a3bee3767a3b4b4213f4`;
  `~/github/zml/zml/platforms/cuda/cuda.bzl` points its amd64 CUDA asset at that
  local archive. The model loaded 14.96 GiB of weights and completed normally.
- Correct VLOG settings did work through the dynamically loaded plugin. The
  earlier conclusion that plugin VLOG output was unavailable was caused by the
  logging invocation, not the PJRT boundary.
- Address-index logging plus an HLO buffer-assignment dump isolated all churn.
  In `llm_llama_decode_layer`, allocation 11 is the `bf16[1,4096]` hidden-state
  parameter, allocation 23 is the scalar `u32` token index, and allocation 34
  is a 33,621,008-byte `preallocated-temp`. With the original ZML example and
  `SKIP_TEMP`, changing allocations 11 and 23 still produced 639 LRU
  replacements in a 757-execution run. With stable ZML input addresses but
  `ALWAYS_UPDATE`, every steady-state replacement changed only allocation 34.
- ZML was changed to keep one session-owned prefill hidden buffer and one
  session-owned decode hidden buffer, donate them through embed/layer execution,
  and keep one donated scalar token-index buffer for the decode turn. The LM
  head advances that scalar in place. This preserves the original token-index
  sequence (the first decode index is initialized one past the pre-loop token
  count) while avoiding a new hidden/index allocation signature for each call.
- Combining those ZML changes with `SKIP_TEMP` passed the acceptance criterion.
  The final plugin smoke run reported 86 empty-slot records, 1,428 cache hits,
  757 executions, zero LRU replacements, zero initialization updates, and zero
  execution-time updates. It produced a normal response at 94.2 tokens/s.

The following are single-run measurements, not a throughput benchmark. The two
profiled generations stopped after slightly different token counts, so raw
launch totals differ; the update count per launch is still unambiguous.

| Configuration | Empty records | Cache hits | LRU replacements | CUDA graph instantiations | CUDA kernel-node setters | CUDA graph launches | Peak process GPU memory | Decode rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable ZML inputs, `ALWAYS_UPDATE`, cache 64 | 117 | 724 | 607 | 117 | 9,712 | 724 | 29,514 MiB | 91.0 tok/s |
| Stable ZML inputs, `SKIP_TEMP`, cache 64 | 86 | 1,428 | 0 | 86 | **0** | 757 | 17,760 MiB | 92.9 tok/s |

`nvidia-smi` sampled process GPU memory every 100 ms during each Nsight run.
Separate ordinary (non-Nsight) `/usr/bin/time -v` runs measured peak host RSS of
9,642,356 KiB for `ALWAYS_UPDATE` and 8,723,420 KiB for `SKIP_TEMP`, with decode
rates of 93.2 and 94.6 tokens/s respectively. These samples show no reproduction
of the previously observed VMM slowdown on this RTX 5090; repeat a longer,
fixed-token benchmark before treating the small speed difference as real.
`CUgraphExec` driver storage remains opaque, so the GPU/RSS numbers are whole
process peaks rather than graph-only byte counts. In the one-shot profiles, the
117 and 86 instantiations also bound the maximum resident executable-graph count
before process teardown.

If a larger model or sequence encounters graph-instantiation OOM, reduce
`--xla_gpu_command_buffer_cache_size`. For this Llama shape, the largest decode
layer cache stabilized at 32 entries, so capacity 64 had sufficient headroom.

### llmd Qwen3.5 follow-up: 2026-08-27, RTX 5090

The follow-up used `~/github/zml/monorepo` with the local ZML checkout and the
same PJRT plugin archive:

```bash
cd ~/github/zml/monorepo
ZML_AUTOTUNE_CACHE_DIR=/tmp/brabier/xla_cache \
CUDA_VISIBLE_DEVICES=0 \
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_MAX_VLOG_LEVEL=0 \
TF_CPP_VMODULE=command_buffer_thunk=6 \
XLA_FLAGS='--xla_gpu_command_buffer_cache_size=64 \
  --xla_gpu_command_buffer_update_mode=SKIP_TEMP' \
bazel run --config=alldebug --@zml//platforms:cuda=true //llmd -- \
  --model=/var/models/Qwen/Qwen3.5-9B/ \
  --batch-size=1 --bench-prompt=hi
```

The original allocation behavior did not form a finite recurring signature
set. A 16-token `SKIP_TEMP` diagnostic still produced 301 first records and 864
LRU replacements. The dominant linear-attention signature changed the hidden
state, device token positions, attention/GDN parameters and GDN metadata on
every model invocation in addition to cycling through layer weights. The
full-attention signature similarly changed its token-position/attention inputs.
Because embed created those device outputs anew, every decode step produced a
new family of layer signatures; increasing the LRU alone could not converge.

`llmd/models/qwen3_5.zig` now gives every compiled prefill/decode runner one
persistent embed scratch set. The embed executable copies host-staged token,
attention, and GDN state into donated device buffers and returns hidden state,
token positions, slot mappings, attention parameters, GDN parameters, and GDN
metadata in those same buffers. Subsequent layer calls donate the same hidden
buffer in place. Different prefill shapes keep separate scratch sets, while the
repeated decode executable reuses one stable set. The remaining variation is
the finite set of layer parameter addresses, which fits in the primary LRU.

The uncapped 256-token acceptance run exited normally and reported:

| Empty records | Cache hits | LRU replacements | Initialization updates | Execution updates | Decode rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 71 | 35,913 | **0** | **0** | **0** | 88.93 tok/s |

The run also had zero VMM deallocation errors. A short 16-token run reported 71
records, 2,313 hits, zero replacements/updates, and 88.19 tok/s. The pre-change
`ALWAYS_UPDATE` baseline had 455 records, 18,015 hits, 17,514 LRU replacements,
and 17,514 initialization-time updates, demonstrating that the improvement is
address stability rather than a logging or workload-count artifact.

A separate uncapped run sampled the llmd PID every 100 ms and measured peak
process GPU memory of 29,392 MiB and peak host RSS of 10,602,028 KiB while
producing 256 tokens at 89.46 tok/s. These are whole-process peaks, not graph-only
or driver-only accounting. The 71 initial records bound the maximum resident
primary executable-graph count for that run.

Two integration defects appeared only when llmd first exercised `SKIP_TEMP`:

- PJRT memory statistics were unimplemented for the VMM allocator, causing
  llmd's memory planning to reject the platform. VMM now reports physical-memory
  usage/budget and supports clearing cumulative statistics.
- The VMM allocator initially handled pinned-host requests as device VMM
  allocations, which faulted when `HostStagedBuffer` CPU-wrote them. Host memory
  is now delegated to the normal pinned allocator, and delegated addresses are
  routed back to it during deallocation.

The earlier benchmark teardown segfault was independent of CUDA graphs. A
request queue lived in an HTTP handler stack frame while the model service was
still producing into it. `llmd/main.zig` now stops and joins the model service
before stopping/joining the HTTP server; both short and full benchmarks shut
down cleanly.

Nsight Systems 2025.6.3 generated a report for the 16-token run, but that report
contained neither CUDA nor NVTX trace tables on this host, so it could not
provide a driver setter count for llmd. The prior Llama profile did capture CUDA
and showed zero graph-node setters. For llmd, the zero-update conclusion is
based on the complete `command_buffer_thunk=6` log: every post-record lookup was
a hit and neither initialization nor execution emitted an update. Re-run Nsight
after resolving the local injection/CUPTI issue.

#### llmd input-copy baseline: 2026-08-28, RTX 5090

`~/github/zml/monorepo/report16.nsys-rep` is a successful CUDA/NVTX capture of
Qwen3.5-9B with batch size 1, a 128-token warm-up plus measured generation, and
the default 1,024-token batch and 128,000-token context capacities. It used
`SKIP_TEMP` and a primary cache capacity of 64. Unlike the earlier failed
capture, this report contains CUDA API, GPU activity, graph, and NVTX tables.

The 18,075.6 MB total H2D traffic is not steady-state graph churn. 266 copies
larger than 1 MiB account for 18,045.7 MB and 368.9 ms of GPU copy time; they
occur once while the model weights are loaded. After 46.5 seconds, the two
generation traversals contain 258 decode iterations and the following exact
per-iteration transfer pattern:

| Direction and size | Copies per decode iteration | llmd source |
| --- | ---: | --- |
| H2D 32,000 bytes | 1 | CUDA FlashAttention block table: 128,000 context tokens / 16 tokens per page * 4 bytes |
| H2D 4,096 bytes | 3 | 1,024-entry token, token-position, and slot-mapping staging buffers |
| H2D 4 bytes | 6 | sampling temperature/top-p, FlashAttention used length, and three GDN scalar arrays |
| H2D 8 bytes | 2 | FlashAttention and GDN cumulative sequence-length arrays |
| D2H 4,096 bytes | 1 | sampled token returned through the full 1,024-entry host-pinned token buffer |
| D2D 4 bytes | approximately 1 | device-side scalar state |

The steady window therefore has 3,615 copies and 12.49 MB of traffic. Their GPU
activity totals only 2.50 ms, but the correlated CUDA API calls total 21.85 ms.
The 3,096 `copyFromSlice` CPU ranges used to populate the twelve H2D inputs total
another 9.91 ms. Amortized over 258 iterations, transfer activity is about 9.7
microseconds and CUDA enqueue time about 84.7 microseconds per iteration, before
counting the host staging copies. The copies are visually prominent and impose
roughly a tenth of a millisecond of CPU-side work per token, but they are not
CUDA graph updates and the primary LRU cannot remove them.

The immediate no-code byte reduction is to set realistic
`--max-context-len` and `--token-batch-size` values. The block table scales with
`max_context_len / page_chunk_size`; the other three capacity-sized inputs scale
with `token_batch_size`. This reduces bytes but leaves the twelve-copy launch
pattern intact and can reduce long-prefill throughput if the token batch is set
too low.

The preferred implementation order for reducing steady-state overhead is:

1. Give the decode runner compact staging shapes (decode token count rather
   than the global prefill capacity) and return sampled tokens through a
   batch-sized host buffer instead of the full token staging buffer.
2. Pack the remaining decode metadata into one pinned-host allocation and
   unpack it in the embed program. This should collapse twelve H2D submissions
   to one while preserving stable addresses and the existing command-buffer
   LRU behavior.
3. For a larger change, keep the attention/GDN tables and counters resident on
   device and update only changed entries. In batch-1 decode, most metadata is
   constant or increments predictably, and the block table changes only at a
   page boundary. This needs explicit stream ordering and a host-visible token
   result path.

Capturing these transfers inside the primary CUDA graph is not currently a
small XLA change. `HostToDeviceCopyThunk` and `DeviceToHostCopyThunk` implement
only ordinary stream execution. The command-buffer conversion pass accepts only
`DeviceToDeviceCopyThunk`, and the StreamExecutor command-buffer interface
exposes `CreateMemcpyD2D`/`UpdateMemcpyD2D`, not host/device graph memcpy nodes.
Adding H2D graph capture would require StreamExecutor API and CUDA backend work,
plus pinned-host lifetime and multi-stream dependency guarantees. Compact or
packed ZML staging is the lower-risk first experiment.

The runtime also logs that cuDNN 9.22.0 is loaded while this local plugin was
compiled against 9.24.0. The model completes because the affected paths do not
fail the run, but the package and XLA build versions should be aligned before
using cuDNN behavior or timings as release-quality data.

#### llmd Gemma 4 packed decode inputs: 2026-08-28, RTX 5090

Gemma 4 had the same address-stability and small-copy problems as Qwen3.5, but
with two full-attention states: one global and one sliding-window state. Pure
decode staged twelve host inputs per invocation: tokens, token positions, the
global slot mapping and three semantic attention fields, the sliding slot
mapping and three semantic attention fields, temperature, and top-p.

`llmd/models/gemma4_text.zig` now uses the backend-neutral attention metadata
adapter introduced for Qwen3.5. Its packet contains semantic `page_table`,
`sequence_lengths`, and `query_offsets` fields for both attention states; no
FlashAttention field name appears in the packet contract. The packed branch is
selected only for pure CUDA decode, independent of whether the active CUDA
attention backend is FA2, FA3, Triton, or another backend supported by the
adapter. Prefill and non-CUDA execution retain the unpacked path.

For token capacity `T`, batch size `B`, global page-table width `P_full`, and
sliding page-table width `P_sliding`, the packet contains

```
4*T + B*(P_full + P_sliding) + 6*B + 2
```

32-bit words. The default batch-1 configuration (`T=1024`, `P_full=8000`,
`P_sliding=81`) is therefore 48,740 bytes. Host preparation writes all twelve
fields into one pinned allocation and submits one H2D copy. Embed consumes
tokens directly from the device packet and fans the other eleven fields out to
the existing device tensors. The implementation intentionally keeps the
transformer-layer interfaces unchanged. Although these are eleven logical
fan-out operations in the ZML program, the CUDA profile below shows that XLA
optimized or fused them: they did not become CUDA D2D memcpy activities.

Each compiled prefill/decode runner now owns one donated embed scratch set:
hidden state, token positions, both slot mappings, both attention parameter
sets, sampling temperature/top-p, and an optional decode packet. Embed and all
remaining layer programs return into these buffers. Consequently each layer
type observes only its finite set of weight/cache addresses instead of a fresh
family of embed outputs on every request. Gemma DFlash's shared prefill embed
uses the same scratch ownership but stays unpacked; its separate speculative
mixed-decode embed is unchanged.

Tests cover the CUDA/decode activation predicate, all supported attention
backend tags, backend-independent layout equality, multi-batch sizing,
non-overlap through sequential offsets, and byte-exact packing including f32
bit patterns. Both ordinary and CUDA-configured `//llmd:test` passed. The local
`~/github/zml/zml` checkout currently contains an unrelated untracked,
syntactically incomplete `pjrt-execution-batch.patch`; validation used an
isolated copy with only its hunk count corrected, leaving that worktree intact.

The end-to-end single-GPU command was:

```bash
cd ~/github/zml/monorepo
CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS=--xla_gpu_command_buffer_cache_size=64 \
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_MAX_VLOG_LEVEL=0 \
TF_CPP_VMODULE=command_buffer_thunk=6 \
bazel run --config=alldebug --@zml//platforms:cuda=true //llmd -- \
  --model=/var/models/RedHatAI/gemma-4-12B-it-NVFP4/ \
  --batch-size=1 --bench-prompt=hi
```

The acceptance run exited normally after 83 completion tokens at 137.47 tok/s
with 11.84 ms TTFT. A second targeted VLOG run completed 125 tokens at 136.77
tok/s with 11.60 ms TTFT. Counting only events after the prompt-benchmark
warm-up marker gave:

| Cache hits | Empty misses | LRU replacements | Records | Updates | Executions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12,348 | **0** | **0** | **0** | **0** | 6,174 |

Thus the measured traversal has one packet H2D per decode and no steady-state
command-buffer update. The unquantized
`/var/models/google/gemma-4-12B-it/` checkpoint did not fit on one 32 GiB 5090:
weight loading filled the 28.22 GiB BFC pool and failed a subsequent 30 MiB
allocation. That failure occurred before model execution and is independent of
the graph LRU or packet. Use the NVFP4 checkpoint, multiple GPUs, or a smaller
cache allocation for repeat measurements. The cuDNN 9.22/9.24 mismatch noted
above was also present in these runs.

An Nsight Systems 2025.6.3 capture of the same NVFP4 benchmark is stored at
`/tmp/gemma4-packed.nsys-rep`. Across the warm-up and measured requests it
recorded 214 pure-decode iterations and exactly 214 H2D copies of 48,740 bytes,
confirming one packet transfer per decode. It recorded **zero D2D memcpy
activities**, so the eleven logical fan-out operations add no CUDA memcpy
submissions. It also recorded 215 D2H copies of 4,096 bytes from the existing
sampled-output path. The profiled request completed 131 tokens at 133.20 tok/s
with 11.50 ms TTFT. The CUDA API trace contained graph launches but no graph
executable node-setter calls, consistent with the command-buffer VLOG result.

Remaining Gemma-specific follow-up is three unprofiled fixed-token runs for a
median throughput comparison. DFlash mixed decode should be profiled separately
because it does not call the normal Gemma decode embed.

Validation after these fixes:

```bash
USE_BAZEL_VERSION=7.7.0 bazel test --features=-module_maps \
  //xla/stream_executor:device_address_vmm_allocator_test \
  //xla/pjrt/gpu:se_gpu_pjrt_client_test \
  //xla/backends/gpu/runtime:command_buffer_thunk_test_nvgpu_any \
  //xla/backends/gpu/runtime:command_buffer_conversion_pass_test_nvgpu_any

cd ~/github/zml/monorepo
bazel test --config=alldebug --@zml//platforms:cuda=true \
  //llmd:test
```

All focused targets passed. The XLA test command still needs
`--features=-module_maps` with this local gRPC/toolchain configuration.

Useful repository searches:

```bash
# Every automatic conversion decision.
rg -n "IsCommandTypeEnabled|IsConvertible|CommandBufferCmdType" \
  xla/backends/gpu/runtime/command_buffer_conversion_pass.cc

# Explicit create/update pairs.
rg -n "Create(Launch|Memset|Memcpy|Child|Case|While)|Update(Launch|Memset|Memcpy|Child|Case|While)" \
  xla/backends/gpu/runtime xla/stream_executor

# Backend capability boundaries.
rg -n "CreateCommandBuffer|UnimplementedError|Supports.*CommandBuffer|Is.*CommandBuffer.*Supported" \
  xla/stream_executor xla/backends/gpu/runtime

# Update and fallback decisions.
rg -n "updated_allocs|persistent_alloc|requires_update|requires_warmup|fallback" \
  xla/backends/gpu/runtime/command_{executor,buffer_thunk}* \
  xla/service/gpu/gpu_executable_*allocator*
```

## Working mental model

The most useful way to reason about the subsystem is as a cached program with a
fixed control/data-dependency skeleton and a mutable parameter block:

- `CommandBufferConversionPass` chooses the program boundary.
- `CommandExecutor` defines the skeleton from semantic resource uses.
- the CUDA/HIP graph definition stores that skeleton;
- the executable graph is the backend's optimized instance;
- create/update command pairs maintain the parameter block;
- traced nested graphs are address-specialized opaque subprograms;
- VMM modes reduce parameter churn by stabilizing virtual addresses; and
- the retained thunk sequence is the semantic reference and escape hatch.

That model explains most design choices: why topology cannot change during an
update, why address tracking is central, why explicit nodes are superior to
tracing, why nested graph ownership matters, and why scheduling and buffer
assignment must be analyzed together.
