#!/bin/bash

# Job array: one 100-core job per Mach number, run in parallel (sbatch supersonic_detail_run.sh)
#SBATCH --job-name=supersonic_detail_run
#SBATCH --account=def-jphickey
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --output=supersonic_detail_run_%a.log
#SBATCH --open-mode=append
#SBATCH --mail-user=kprahaask@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

# Cases: 
MACHS=(1.1 1.2 1.3 1.4)
ALPHAS=(0 0 0 0)
BETAS=(0 0 0 0)
P_BACKS=(capture capture capture capture)
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
pip install --find-links=$SCRATCH/pyfoam_wheel --no-index PyFoam matplotlib

export SQUEUE_FORMAT='%i","%j","%t","%M","%L","%D","%C","%m","%b","%R'

python3 supersonic_detail_run.py --mach ${MACHS[$i]} --alpha ${ALPHAS[$i]} --beta ${BETAS[$i]} \
                                 --p-back ${P_BACKS[$i]} --init-from $INIT_FROM
