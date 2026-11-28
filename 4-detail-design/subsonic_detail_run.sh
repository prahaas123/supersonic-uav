#!/bin/bash

# Job array: one 100-core job per Mach number, run in parallel (sbatch subsonic_detail_run.sh)
# The first job builds the subsonic mesh (detail_mesh_subsonic_half), the others wait for it and reuse it.
#SBATCH --job-name=subsonic_detail_run
#SBATCH --account=def-jphickey
#SBATCH --time=1-00:00:00
#SBATCH --array=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --output=subsonic_detail_run_%a.log
#SBATCH --open-mode=append
#SBATCH --mail-user=kprahaask@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

# Cases: one entry per array index (--array picks the indices to run; keep the lists the same length)
MACHS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
ALPHAS=(0 0 0 0 0 0 0 0 0)
BETAS=(0 0 0 0 0 0 0 0 0)
ALTITUDES=(0 1375 2750 4125 5500 6875 8250 9625 11000)
# engine-face recovery for the back pressure: measured at Mach 0.1-0.3 and 0.9 (Nov 2026 runs at 0.97),
# linear between Mach 0.3 and 0.9 for the others
RECOVERIES=(0.9064 0.9182 0.9317 0.9372 0.9428 0.9484 0.9539 0.9595 0.9650)
# outlet back pressure: '1d' = from the recovery above, or a value [Pa]. Mach 0.5 at the 1D value (49.4 kPa) asks the
# inlet for more than its throat passes, goes transonic there and diverges; 53 kPa (~0.35-0.39 kg/s) is below choking
P_BACKS=(1d 1d 1d 1d 53000 1d 1d 1d 1d)
INIT_FROM=none     # 'auto' would restart Mach 0.5 from its own failed run (same altitude, stopped by max_iterations)

i=$SLURM_ARRAY_TASK_ID

module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load paraview/5.11.2
module load python/3.11.5
module load openfoam/v2312

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy matplotlib

export SQUEUE_FORMAT='%i","%j","%t","%M","%L","%D","%C","%m","%b","%R'

python3 subsonic_run.py --mach ${MACHS[$i]} --alpha ${ALPHAS[$i]} --beta ${BETAS[$i]} \
                               --altitude ${ALTITUDES[$i]} --recovery ${RECOVERIES[$i]} \
                               --p-back ${P_BACKS[$i]} --init-from $INIT_FROM
