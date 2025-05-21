#!/bin/bash

mkdir -p logs

for config in sweep_search/*.json; do
    cfg_file=$(basename "$config")
    echo "Submitting job for $cfg_file"
    sbatch run_sweep_job.sh "$cfg_file"
done
