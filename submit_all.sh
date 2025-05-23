#!/bin/bash

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$BASE_DIR/logs"

for config in "$BASE_DIR"/sweep_search/*.json; do
    cfg_file=$(basename "$config")
    echo "Submitting job for $cfg_file"
    sbatch "$BASE_DIR/run_sweep_job.sh" "$cfg_file"
done
