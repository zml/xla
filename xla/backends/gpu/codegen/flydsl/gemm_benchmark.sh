#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  runfiles_root="${RUNFILES_DIR}"
elif [[ -n "${RUNFILES_MANIFEST_FILE:-}" ]]; then
  runfiles_root="${RUNFILES_MANIFEST_FILE%.runfiles_manifest}.runfiles"
else
  runfiles_root="$0.runfiles"
fi

workspace_name="${TEST_WORKSPACE:-xla}"
autotuner="${runfiles_root}/${workspace_name}/xla/backends/gpu/autotuner/autotuner_main"
template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/gemm_benchmark.hlo.tpl"
rocm_lib="${runfiles_root}/local_config_rocm/rocm/rocm_dist/lib"

if [[ ! -x "${autotuner}" || ! -f "${template}" || ! -d "${rocm_lib}" ]]; then
  echo "Could not locate benchmark runfiles." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${rocm_lib}:${rocm_lib}/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# TheRock multi-architecture redists keep each architecture's hipBLASLt
# database and code objects in a subdirectory. Bazel packages only the
# configured architectures, so select it automatically when it is unique.
if [[ -z "${HIPBLASLT_TENSILE_LIBPATH:-}" ]]; then
  hipblaslt_arch_dirs=("${rocm_lib}"/hipblaslt/library/gfx*)
  if [[ "${#hipblaslt_arch_dirs[@]}" -eq 1 &&
        -d "${hipblaslt_arch_dirs[0]}" ]]; then
    export HIPBLASLT_TENSILE_LIBPATH="${hipblaslt_arch_dirs[0]}"
  fi
fi

output_dir="${FLY_GEMM_BENCHMARK_OUTPUT_DIR:-/tmp/fly_gemm_benchmark}"
mkdir -p "${output_dir}"

rhs_layout="${FLY_GEMM_BENCHMARK_RHS_LAYOUT:-1,0}"
if [[ "${rhs_layout}" != "1,0" && "${rhs_layout}" != "0,1" ]]; then
  echo "Invalid FLY_GEMM_BENCHMARK_RHS_LAYOUT='${rhs_layout}'; expected 1,0 or 0,1." >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  set -- 256x256x256 512x512x512 1024x1024x1024
fi

for shape in "$@"; do
  IFS=x read -r m n k extra <<<"${shape}"
  if [[ -n "${extra:-}" || -z "${m}" || -z "${n}" || -z "${k}" ||
        ! "${m}" =~ ^[0-9]+$ || ! "${n}" =~ ^[0-9]+$ ||
        ! "${k}" =~ ^[0-9]+$ ]]; then
    echo "Invalid shape '${shape}'; expected MxNxK." >&2
    exit 1
  fi

  hlo="${output_dir}/bf16_gemm_${shape}.hlo"
  selected="${output_dir}/bf16_gemm_${shape}.autotuned.hlo"
  log="${output_dir}/bf16_gemm_${shape}.autotune.textproto"

  sed -e "s/__M__/${m}/g" -e "s/__N__/${n}/g" -e "s/__K__/${k}/g" \
    -e "s/__RHS_LAYOUT__/${rhs_layout}/g" \
    "${template}" >"${hlo}"
  rm -f "${log}"

  XLA_FLAGS="${XLA_FLAGS:-} \
--xla_gpu_autotune_level=3 \
--xla_gpu_enable_flydsl_gemm=true \
--xla_gpu_experimental_autotune_backends=triton,fly,hipblaslt_fission \
--xla_gpu_dump_autotune_logs_to=${log}" \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    HIPBLASLT_PRELOAD_KERNELS="${HIPBLASLT_PRELOAD_KERNELS:-0}" \
    HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL="${HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL:-1}" \
    "${autotuner}" --hlo_files="${hlo}" >"${selected}"

  echo "${shape}:"
  echo "  selected config: ${selected}"
  echo "  candidate timings: ${log}"
done
