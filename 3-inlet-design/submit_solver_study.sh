#!/bin/bash
# Solver stability study: one job per pressure-velocity coupling variant (SOLVER_VARIANTS in run_inlet.py),
# same geometry and boundary conditions. Each job gets its own case directory and log.
#   bash submit_solver_study.sh            -> T1 T2 T3
#   bash submit_solver_study.sh T2 T3      -> only those

cd "$(dirname "$0")"
variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(T1 T2 T3)

for v in "${variants[@]}"; do
    sbatch --job-name="inlet_${v}" --output="inlet_${v}.log" run_inlet.sh "./inlet_${v}" "${v}"
done
