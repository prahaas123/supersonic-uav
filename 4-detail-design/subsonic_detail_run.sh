#!/bin/bash

# Job array: one 100-core job per Mach number, run in parallel (sbatch subsonic_detail_run.sh)
# The first job builds the subsonic mesh (detail_mesh_subsonic_half), the others wait for it and reuse it.
#SBATCH --job-name=subsonic_detail_run
#SBATCH --account=def-jphickey
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --output=subsonic_detail_run_%a.log
#SBATCH --open-mode=append
#SBATCH --mail-user=kprahaask@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

# Cases: one entry per array index (keep --array=0-N in step with the list length)
MACHS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
ALPHAS=(0 0 0 0 0 0 0 0 0)
BETAS=(0 0 0 0 0 0 0 0 0)
ALTITUDES=(15000 15000 15000 15000 15000 15000 15000 15000 15000)    # [m] ISA
INIT_FROM=auto

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

python3 subsonic_detail_run.py --mach ${MACHS[$i]} --alpha ${ALPHAS[$i]} --beta ${BETAS[$i]} \
                               --altitude ${ALTITUDES[$i]} --init-from $INIT_FROM
