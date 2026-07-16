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
Current XLA consumes the generated runtime table; the C06 bridge will consume
the same library and validate its canonical fingerprint before translation.
Required attributes match the pinned vendor declarations literally; coordinate
reads are `nounwind` and `memory(none)` but do not claim `willreturn`.

ABI v1 admits only the compiler-source-verified register reads for thread,
block, block-size, and grid-size coordinates, 32/64-bit clocks, and the
workgroup barrier. The clock calls intentionally have inaccessible read/write
effects so optimizers cannot common-subexpression-eliminate changing values.
The `llvm.musa.barrier0` mapping is available on `mp_21`; it is convergent and
has unrestricted read/write effects so memory operations cannot move across
it.

The interchange admits generic AS0, global AS1, constant AS2,
workgroup/shared AS3, and private/local/scratch AS5. The pinned SDK emits AS5
for local pointers and defines it as 64-bit. AS4 is a 32-bit reserved space
with no public semantic contract and is rejected. Admitting AS5 does not admit
shuffle: its scratch layout and subgroup semantics remain unqualified.

Subgroup synchronization, shuffle, vote, atomics, and non-generic math are
named unsupported capability categories in mapping version 1. They fail with a
specific diagnostic until compiler-source and execution probes establish their
signatures and semantics. Unknown shims and silent CUDA/NVVM fallback are never
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
