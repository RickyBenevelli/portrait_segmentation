#!/bin/bash
#SBATCH --account=iscrc_edl4fv
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Verifica argomenti
if [ $# -ne 1 ]; then
    echo "Usage: sbatch $0 <config_filename.json>"
    exit 1
fi

CONFIG_FILE=$1

# Salva la working directory al momento della sottomissione
WORKDIR="$SLURM_SUBMIT_DIR"

CONFIG_PATH="$WORKDIR/sweep_search/$CONFIG_FILE"
CONTAINER_PATH="$WORKDIR/container.sif"

echo "Running experiment with configuration: $CONFIG_FILE"
echo "Working directory: $WORKDIR"

# Verifica file di configurazione
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file $CONFIG_PATH not found!"
    exit 1
fi

# Verifica container
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container file $CONTAINER_PATH not found!"
    exit 1
fi

# Esecuzione del job con Singularity
singularity exec \
    --nv \
    -B "$WORKDIR":/app \
    -B /leonardo_scratch:/leonardo_scratch \
    -B /leonardo_work:/leonardo_work \
    "$CONTAINER_PATH" \
    python /app/main.py -c /app/sweep_search/$CONFIG_FILE --suffix="$SLURM_JOB_ID"
