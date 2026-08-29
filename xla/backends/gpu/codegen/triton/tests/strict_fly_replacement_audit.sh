#!/usr/bin/env bash

set -uo pipefail

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  runfiles_root="${RUNFILES_DIR}"
elif [[ -n "${RUNFILES_MANIFEST_FILE:-}" ]]; then
  runfiles_root="${RUNFILES_MANIFEST_FILE%.runfiles_manifest}.runfiles"
else
  runfiles_root="$0.runfiles"
fi
runfiles_root="$(cd "${runfiles_root}" && pwd -P)"

workspace_name="${TEST_WORKSPACE:-xla}"
workspace_root="${runfiles_root}/${workspace_name}"
runner="${STRICT_FLY_RUN_HLO_MODULE:-${workspace_root}/xla/tools/run_hlo_module}"
corpus_root="${STRICT_FLY_CORPUS_ROOT:-${workspace_root}/xla/backends/gpu/codegen/triton/tests}"
rocm_lib="${STRICT_FLY_ROCM_LIB:-${runfiles_root}/local_config_rocm/rocm/rocm_dist/lib}"
audit_root="${STRICT_FLY_AUDIT_ROOT:-$(mktemp -d /tmp/strict-fly-corpus.XXXXXX)}"
mkdir -p "${audit_root}/logs"
summary="${audit_root}/summary.tsv"

if [[ ! -x "${runner}" || ! -d "${corpus_root}" ||
      ! -d "${rocm_lib}" ]]; then
  echo "Could not locate strict Fly audit runfiles." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${rocm_lib}:${rocm_lib}/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

declare -A expected_nonpass=(
  [collectives/all_gather.hlo]=multi_device
  [collectives/all_reduce.hlo]=multi_device
  [dot/layout_021_201_210.hlo]=invalid_hlo
  [dot/layout_201_012_201.hlo]=invalid_hlo
  [dot/layout_210_210_102.hlo]=invalid_hlo
  [dot/layout_210_210_210.hlo]=invalid_hlo
  [elementwise/nested_reducer_fusion.hlo]=invalid_hlo
  [reduce/intra_warp_reduce_of_reduce.hlo]=invalid_hlo
)

if [[ "$#" -eq 0 ]]; then
  mapfile -t hlo_files < <(find -L "${corpus_root}" -type f -name '*.hlo' | sort)
else
  hlo_files=("$@")
fi
if ((${#hlo_files[@]} == 0)); then
  echo "Strict Fly audit corpus is empty." >&2
  exit 1
fi

printf 'file\trc\tclassification\n' >"${summary}"
unexpected_failures=0
for hlo in "${hlo_files[@]}"; do
  if [[ "${hlo}" != /* ]]; then
    hlo="${corpus_root}/${hlo}"
  fi
  relative="${hlo#${corpus_root}/}"
  key="${relative//\//_}"
  log="${audit_root}/logs/${key}.log"
  timeout "${STRICT_FLY_TIMEOUT_SECONDS:-60}" env \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    XLA_FLAGS="${XLA_FLAGS:-} \
--xla_gpu_flydsl_replace_triton=true \
--xla_gpu_autotune_level=${STRICT_FLY_AUTOTUNE_LEVEL:-0} \
--xla_gpu_experimental_disable_binary_libraries=true" \
    "${runner}" \
      --platform=gpu \
      --reference_platform='' \
      --run_test_hlo_passes=true \
      --iterations=1 \
      "${hlo}" >"${log}" 2>&1
  rc=$?

  classification=pass
  if ((rc != 0)); then
    if ((rc == 124)); then
      classification=timeout
    elif grep -q 'FlyDSL replacement mode found' "${log}"; then
      classification=triton_escape
    elif [[ "${relative}" == collectives/* ]]; then
      classification=multi_device
    elif grep -q -E \
      'Failed to parse|INVALID_ARGUMENT|does not match|Expected array|layout' \
      "${log}"; then
      classification=invalid_hlo
    else
      classification=compile_or_runtime_failure
    fi
    if [[ "${expected_nonpass[${relative}]:-}" != "${classification}" ]]; then
      ((unexpected_failures += 1))
    fi
  fi
  printf '%s\t%d\t%s\n' "${relative}" "${rc}" "${classification}" |
    tee -a "${summary}"
done

awk -F '\t' \
  'NR > 1 {count[$3]++} END {for (kind in count) print kind, count[kind]}' \
  "${summary}" | sort
printf 'audit_root=%s\n' "${audit_root}"
exit "${unexpected_failures}"
