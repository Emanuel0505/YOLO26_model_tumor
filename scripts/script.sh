#!/bin/bash
#SBATCH --partition=gpu-4-a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0-03:00
#SBATCH --output=slurm-%j.out

set -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

case "$1" in
    train)
        script="setup/model-train.py"
        ;;  
    organize)
        script="setup/organize-dataset.py"
        ;;
    validation)
        script="setup/validation-train.py"
        ;;
    *)
    echo "Uso: sbatch scripts/script-train-model.sh {train|organize|validation}"
    exit 1
    ;;
esac

python3 "$script"