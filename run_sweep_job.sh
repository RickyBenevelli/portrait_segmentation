#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1           # richiede 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00        # tempo massimo
#SBATCH --partition=gpu        # usa la tua partizione corretta

# Nome del file .json passato come argomento
CONFIG_FILE=$1

echo "Running experiment with configuration: $CONFIG_FILE"
singularity exec --nv -B ~/portrait_segmentation:/app portrait_segmentation.sif \
  python main.py -c sweep_search/$CONFIG_FILE
