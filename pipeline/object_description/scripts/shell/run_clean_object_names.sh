#!/bin/bash

#SBATCH --job-name=clean_object_names
#SBATCH --output=logs/clean_object_names_%j.out
#SBATCH --error=logs/clean_object_names_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=main

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
WORKING_DIR="/path/to/SpatialReasonerDataGen"
cd "$WORKING_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting object name cleaning..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Run the cleaning script
python object_description/clean_object_names.py

echo "Object name cleaning completed!"
echo "Date: $(date)"
