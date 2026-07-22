# MUSA XLA-to-bridge shim ABI v1

The MUSA PJRT plugin and current XLA LLVM never declare vendor-reserved
`llvm.musa.*` intrinsics or calling convention 102. Current XLA emits ordinary
external calls under the symbol-level versioned prefix
`__xla_musa_v1_`; the isolated vendor-LLVM bridge is the only component that
may replace those declarations with registered MUSA intrinsics and attach the
native kernel ABI.

`musa_shim_abi.def` is the single mapping source. Each enabled row freezes the
XLA symbol, exact function type, memory effects, convergence, required
attributes, vendor intrinsic, minimum mapping revision, and `mp_21` target.
Current XLA and the isolated bridge consume the generated runtime table; the
bridge validates its canonical fingerprint before translation.
Required attributes match the pinned vendor declarations literally; coordinate
reads are `nounwind` and `memory(none)` but do not claim `willreturn`. Shim ABI
v1 currently uses mapping version 3 and the canonical mapping SHA-256 in
`musa_shim_abi.h`.

The base mapping admits the compiler-source-verified register reads for thread,
block, block-size, and grid-size coordinates, 32/64-bit clocks, and the
workgroup barrier. The clock calls intentionally have inaccessible read/write
effects so optimizers cannot common-subexpression-eliminate changing values.
The `llvm.musa.barrier0` mapping is available on `mp_21`; it is convergent and
has unrestricted read/write effects so memory operations cannot move across
it.

The interchange admits generic AS0, global AS1, constant AS2,
workgroup/shared AS3, and private/local/scratch AS5. The pinned SDK emits AS5
for local pointers and defines it as 64-bit. AS4 is a 32-bit reserved space
with no public semantic contract and is rejected.

Mapping version 2 adds a logical 32-lane shuffle while preserving S80's
physical 128-thread hardware warp. Shared GPU scheduling uses the logical
subgroup size only for subgroup algorithms; `threads_per_warp` and the target
device contract remain physical. Current LLVM derives the logical lane as
`thread_id_x & 31`, lowers all four `gpu.shuffle` modes with constant width 32,
uses the caller's value for an invalid source lane, decomposes values into i32
words, and calls `__xla_musa_v1_subgroup_read_lane_i32`.

Only the isolated vendor bridge knows the SDK-specific adapter. It validates
the registered `llvm.musa.shfl.idx.sync.fake` signature and attributes, adds a
private aligned `[128 x i32]` AS3 scratch object, and supplies the mask,
logical source lane, width, predicate, and scratch arguments. Direct S80
source and LLVM probes establish the selected fake-intrinsic path; the
similarly named wave-lane/read-lane intrinsics are not used because live
probes did not implement the required logical shuffle semantics.

Mapping version 2 also admits only the `nsz` fast-math flag needed by shared
reduction emitters. Every other fast-math relaxation is rejected independently
by current LLVM validation and vendor LLVM 14 validation.

Mapping version 3 adds exactly one atomic instruction contract: a strong,
non-volatile scalar i32 `cmpxchg` on global AS1, system/default sync scope,
monotonic success and failure orderings, and alignment 4. The canonical
mapping text binds every one of those properties. Current XLA routes only
scalar 32-bit integer (s32/u32) and f32 add reductions through the shared
compare-and-swap loop;
f32 values are compared and exchanged through their bitwise i32
representation. The lowering explicitly casts tensor-buffer addresses to AS1
and emits alignment 4.

The contract was qualified against MUSA 4.0.1 on S80 `mp_21`: vendor LLVM 14
accepted the exact interchange form, `mcc` produced a MUBIN, and a driver
launch changed an initialized i32 from 0 to 1. Current-LLVM validation and the
isolated vendor-LLVM validator independently enforce the same exact allowlist.
Atomic loads/stores, `atomicrmw`, fences, weak or volatile cmpxchg, other
types/address spaces/scopes/orderings/alignments, and every unrecognized
atomic form remain fail-closed.

Subgroup synchronization, vote, and non-generic math remain named unsupported
capability categories. Nonconstant or non-32 shuffle widths also fail with a
specific diagnostic. Unknown shims and silent CUDA/NVVM fallback are never
permitted.

The native target contract is:

- triple `mtgpu-mt-musa`;
- architecture `mp_21`;
- exact data layout
  `e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128`;
- native kernel calling convention 102, installed only after the bridge parses
  and validates the interchange module.

Changing a row, address-space rule, or unsupported classification requires a
mapping-version update and refreshed canonical SHA-256. An incompatible shim
type or symbol namespace requires a shim-ABI version update.

## Textual bridge protocol v1

`protocol.proto` is serialized with protobuf TextFormat behind distinct
request/response magic lines. The decoder accepts only the canonical printer's
byte representation, rejects unknown fields recursively, and applies bounds
before parsing. Request identity covers the exact canonical wire bytes.

A request binds normalized LF-terminated LLVM and its SHA-256/byte size; the
protocol, shim, and mapping versions; mapping fingerprint; sorted kernels and
typed exported globals; triple, architecture, layout, opaque 64-bit pointer
model, byte order, numerical flags, and optimization mode; XLA/current-LLVM
revisions; and expected provider, bridge, and toolchain fingerprints. It has no
path, working-directory, environment, arbitrary-argument, shell, or output-name
field.

A response is bound to the request digest and echoes the actual mapping,
provider, bridge, and toolchain identities. It contains bounded structured
diagnostics and statistics plus either an explicit MUBIN and SHA-256 or a
failure status with no binary. Cross-message validation rejects identity,
count, size, symbol, or digest disagreement. Compilation errors remain data;
the future sidecar cannot terminate the PJRT host through this interface.
