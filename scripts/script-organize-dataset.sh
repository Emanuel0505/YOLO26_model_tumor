#!/bin/bash
#SBATCH --partition=gpu-4-a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --time=0-00:10
#SBATCH --output=slurm-%j.out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python  setup/organize-dataset.py