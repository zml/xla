# Metal AIR Matmul Experiment

This is a feasibility spike for using Apple AIR as a backend handoff point.
The script emits textual AIR LLVM IR from MSL, assembles it with `air-as`, links
it with `metallib`, then validates and benchmarks an FP32 matrix multiplication.

Run:

```sh
bash xla/service/gpu/metal_air_experiment/run_matmul_air.sh
```

The important generated files are under `/private/tmp/xla_metal_air_matmul`:

- `matmul_air.ll`: textual AIR-flavored LLVM IR.
- `matmul_air.air`: binary AIR assembled from the textual IR.
- `matmul_air.metallib`: library consumed by the Metal runtime.
- `matmul_air_direct.air`: binary AIR assembled directly from
  `matmul_air_direct.ll`.
- `matmul_air_direct.metallib`: optimized direct-AIR library consumed by the
  Metal runtime.

The fast kernel is `matmul_simdgroup_8x8`. It uses Metal simdgroup matrix
multiply-accumulate operations and currently requires `M % 16 == 0`,
`N % 32 == 0`, and `K % 8 == 0`.

On an Apple M3 Max, a 2048x2048x2048 FP32 multiply validates and runs at about
2.9 TFLOP/s through both MSL-generated AIR and the hand-authored textual AIR
path after `air-opt --O3`.
