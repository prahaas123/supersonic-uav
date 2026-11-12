#!/bin/bash

# 60 inlet designs, 10 running at a time. From 3-inlet-design/, after python3 dse.py generate:
#   mkdir -p dse/logs && sbatch dse_run.sh
# Resubmit unfinished designs only:  sbatch --array=$(python3 dse.py pending)%10 dse_run.sh

#SBATCH --job-name=inlet_dse
#SBATCH --account=def-jphickey
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=50
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --array=0-59%10
#SBATCH --output=dse/logs/design_%a.log
#SBATCH --open-mode=append

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

python3 dse.py run $SLURM_ARRAY_TASK_ID
