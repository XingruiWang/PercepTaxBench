#!/bin/bash
#SBATCH --job-name=combine_filtered_objects
#SBATCH --output=logs/combine_filtered_objects_%j.out
#SBATCH --error=logs/combine_filtered_objects_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=main

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
WORKING_DIR="/path/to/SpatialReasonerDataGen"
cd "$WORKING_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run object combination with API key
python object_description/combine_filtered_objects.py \
    --api_key "${GEMINI_API_KEY}" \
    --output_dir "object_description/results/filtered_full"

echo "Object combination completed!"
