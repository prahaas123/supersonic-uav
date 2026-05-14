#!/bin/bash

#SBATCH --job-name=Wing
#SBATCH --account=def-jphickey
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=50
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --output=Wing.log
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
pip install --find-links=$SCRATCH/pyfoam_wheel --no-index PyFoam matplotlib

export SQUEUE_FORMAT='%i","%j","%t","%M","%L","%D","%C","%m","%b","%R'

python3 supersonic_run.py