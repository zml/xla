# MUSA target and runtime contract

This file records the C01 contract used by XLA's MUSA backend. It supports the
initial S80 (`mp_21`) contract with MUSA 4.0.1 and the S4000 (`mp_22`) build
contract with MUSA 5.1.0. These values are target-specific and are not fallback
defaults. See [S4000_BUILD.md](S4000_BUILD.md) for the reproducible build.

## Compilation target

| Field | Qualified value |
| --- | --- |
| LLVM target triple | `mtgpu-mt-musa` |
| Target CPU | S80: `mp_21`; S4000: `mp_22` |
| Pointer model | 64-bit generic (AS0), global (AS1), constant (AS2), and private/local/scratch (AS5) pointers; 32-bit workgroup/shared pointers (AS3); reserved AS4 is forbidden |
| Data layout | `e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128` |
| Vendor kernel calling convention | LLVM calling convention 102 |
| Device binary | ELF64 little-endian shared object (`ET_DYN`), called MUBIN |
| ELF machine | `EM_MTGPU`, numeric value 253 (`0xfd`) |
| Vendor note | `MTGPU`, note type `0x40` |

MUBIN is a first-class MUSA binary kind. It must not be labelled or serialized
as CUDA CUBIN. Calling convention 102 and vendor intrinsics are installed only
inside the isolated vendor-LLVM process planned by C06; they must not cross the
current-XLA LLVM boundary. The C05 boundary is frozen in the
[MUSA shim ABI](../../service/gpu/musa/MUSA_SHIM_ABI.md): symbol-versioned
ordinary calls, a mapping fingerprint, and a bounded canonical textual
protocol.

## Device and launch contract

The live device is authoritative. `MUSA_GPU_ARCHS` selects offline compiler
inputs only and must not identify a live device. C01 obtains the device name,
PCI identity, compute capability, dimensions, memory limits, registers,
clocks, cache, and alignment through `libmusart` discovery APIs.

The S80 driver reports compute capability 2.1, mapped only to `mp_21`; the
S4000 contract reports 2.2, mapped only to `mp_22`. Other compute capabilities
fail closed. Grid and block dimensions retain XLA's X/Y/Z order when passed to
the MUSA runtime.

The hardware scheduling width and compiler-visible logical subgroup width are
different facts:

| Width | S80/S4000 value | Use |
| --- | ---: | --- |
| Hardware warp/scheduling width | 128 | Preserved in `MusaComputeCapability` and `DeviceDescription::threads_per_warp`. |
| Compiler logical subgroup width | 32 | Provisional vendor-LLVM ABI input, preserved separately in `MusaComputeCapability`; C07/C08 must qualify subgroup-operation semantics before lowering relies on it. |

Neither value may overwrite the other. Subgroup lowering and numerical
conformance remain subject to the later C07, C08, and C12 qualification gates.

The driver's reported maximum blocks per multiprocessor is recorded as a raw
fact, but is deliberately not exported to the shared `DeviceDescription`
scheduling field until independently validated.

## Version identities

MUSA runtime and driver API integers use:

~~~text
raw = major * 10000 + minor * 100 + patch
~~~

Thus raw `10504` is API version 1.5.4. S4000's qualified toolkit value is raw
`50100`, or 5.1.0. These are distinct from:

- the target-specific SDK release, 4.0.1 or 5.1.0;
- the user-mode `libmusa` and `libmusart` SONAME, 1.5;
- the Linux driver package version, 3.0.0;
- the compiler, linker, bundler, and `libdevice` identities recorded in the
  external reproducer manifest.

These identities must remain separate in diagnostics and artifact provenance.
Strong executable compatibility and the complete serialized executable
envelope are deferred to C14.

The S80's raw maximum-resident-block value and floating-point execution units
per multiprocessor are not yet semantically qualified. C01 leaves its shared
`max_blocks_per_multiprocessor` and `fpus_per_core` cost-model inputs unknown;
C10 must either qualify them or disable consumers before reconnecting the
shared GPU compiler. S4000 `fpus_per_core` is independently qualified as 128:
the published 25 TFLOP/s FP32 rating agrees with the live 56-MP device's
reported 1.791 GHz maximum clock (`56 * 128 * 1.791 GHz * 2 = 25.68 TFLOP/s`,
where an FMA counts as two operations). Maximum resident blocks remains
unknown on both targets.

## Explicitly deferred contracts

This file does not define:

- vendor-LLVM translation and MUBIN production (C06);
- module, function, global, or launch ownership (C02-C04);
- executable serialization compatibility (C14);
- vendor-library ABIs (C15-C21).

Unsupported or unknown values must fail before compilation or launch rather
than selecting CUDA/NVVM behavior or fabricating a device property.

## Default unqualified muFFT discovery

SDK 5.1's muFFT 1.12 does not satisfy the existing FFT contract. A well-formed
but unqualified default-discovered FFT shim is omitted from available optional
ABIs so it cannot block loading an unrelated GEMM executable. FFT execution
remains rejected. Explicit `XLA_MUSA_MUFFT_SHIM_PATH` selection and malformed
tables remain errors. No FFT ABI fingerprint or support claim is expanded.
