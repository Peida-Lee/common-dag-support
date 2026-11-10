#!/bin/bash
# Execute the final experiments in a fresh directory prepared by the cloning script.
TASK_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
TASK_PY=${DAG_PYTHON:-python3.11}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JULIA_NUM_THREADS=1
export DAG_JULIA=${DAG_JULIA:-$(command -v julia)}
export DAG_JULIA_DEPOT=${DAG_JULIA_DEPOT:-$TASK_ROOT/environment/julia_depot}
TASK_FAILURES=0

run_stage() {
    "$TASK_PY" "$TASK_ROOT/$1" "$2"
    TASK_CODE=$?
    if [ "$TASK_CODE" -ne 0 ]; then
        printf 'Stage requires attention. %s %s exit %s\n' "$1" "$2" "$TASK_CODE"
        TASK_FAILURES=$((TASK_FAILURES+1))
    fi
}

mapfile -t TASK_CORE_GROUPS < <("$TASK_PY" - <<'PY'
from pathlib import Path
import os
cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    topology = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
    key = tuple((topology / field).read_text().strip() for field in ('physical_package_id', 'core_id'))
    cores.setdefault(key, cpu)
cpus = list(cores.values())
baseline_count = min(64, max(1, len(cpus)//4))
science = cpus[:-baseline_count] or cpus
baseline = cpus[-baseline_count:]
print(','.join(map(str, science)))
print(','.join(map(str, baseline)))
print(','.join(map(str, baseline[-4:])))
PY
)
taskset -c "${TASK_CORE_GROUPS[1]}" "$TASK_PY" "$TASK_ROOT/run_final376_v1.py" baseline_cpu & TASK_CPU_PID=$!
taskset -c "${TASK_CORE_GROUPS[2]}" "$TASK_PY" "$TASK_ROOT/run_final376_v1.py" baseline_gpu & TASK_GPU_PID=$!
for TASK_STAGE in power connected joint wind light camera; do
    taskset -c "${TASK_CORE_GROUPS[0]}" "$TASK_PY" "$TASK_ROOT/run_final376_v1.py" "$TASK_STAGE"
    if [ "$?" -ne 0 ]; then
        TASK_FAILURES=$((TASK_FAILURES+1))
    fi
done
for TASK_STAGE in power connected joint; do
    run_stage verify_final376_v1.py "$TASK_STAGE"
done
wait "$TASK_CPU_PID" || TASK_FAILURES=$((TASK_FAILURES+1))
wait "$TASK_GPU_PID" || TASK_FAILURES=$((TASK_FAILURES+1))
run_stage verify_final376_v1.py measured
run_stage verify_final376_v1.py frozen_files
"$TASK_PY" "$TASK_ROOT/summarize_final376_v1.py"
if [ "$?" -ne 0 ]; then
    TASK_FAILURES=$((TASK_FAILURES+1))
fi
"$TASK_PY" "$TASK_ROOT/plot_final376_v1.py"
"$TASK_PY" "$TASK_ROOT/evaluate_density_reference378_v1.py" "$TASK_ROOT" --output "$TASK_ROOT/outputs/final_tables/density_reference_results378_v1.json"
if [ "$?" -ne 0 ]; then
    TASK_FAILURES=$((TASK_FAILURES+1))
fi
"$TASK_PY" "$TASK_ROOT/verification/check_root_target_design381_v1.py" --output "$TASK_ROOT/outputs/verification/root_target_design381_v1.json" --workers 8
if [ "$?" -ne 0 ]; then
    TASK_FAILURES=$((TASK_FAILURES+1))
fi
printf 'Final replay finished. Driver-level issues %s\n' "$TASK_FAILURES"
exit "$TASK_FAILURES"
