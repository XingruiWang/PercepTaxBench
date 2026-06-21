#!/bin/bash
#SBATCH --job-name=gen_missing_desc
#SBATCH --output=/path/to/SpatialReasonerDataGen/object_description/logs/generate_missing_descriptions_%j.log
#SBATCH --error=/path/to/SpatialReasonerDataGen/object_description/logs/generate_missing_descriptions_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=main
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ccvl35

echo "Starting missing object description generation at $(date)"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen

# Run the missing object description generation
python object_description/scripts/python/generate_descriptions_for_missing.py

echo "Job completed at $(date)"
