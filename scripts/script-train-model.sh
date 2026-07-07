#!/bin/bash
#SBATCH --partition=gpu-4-a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0-03:00
#SBATCH --output=slurm-%j.out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python ../setup/model-train.py