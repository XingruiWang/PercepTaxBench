#!/bin/bash
#SBATCH --job-name=regen_fields
#SBATCH --output=/path/to/SpatialReasonerDataGen/object_description/logs/regen_fields_%j.log
#SBATCH --error=/path/to/SpatialReasonerDataGen/object_description/logs/regen_fields_%j.err
#SBATCH --time=0:20:00
#SBATCH --partition=main
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --exclude=ccvl35

echo "Starting field regeneration at $(date)"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen/object_description/results/full_object_description

# Run the regeneration script
python3 -u regenerate_empty_fields.py

echo "Regeneration completed at $(date)"

