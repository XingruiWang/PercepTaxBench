#!/bin/bash
#SBATCH --job-name=clean_parsing
#SBATCH --output=/path/to/SpatialReasonerDataGen/object_description/logs/clean_parsing_%j.log
#SBATCH --error=/path/to/SpatialReasonerDataGen/object_description/logs/clean_parsing_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=main
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --exclude=ccvl35

echo "Starting parsing cleanup at $(date)"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen/object_description/results/object_list_final

# Run the cleanup script
python3 -u clean_parsing_issues.py

echo "Cleanup completed at $(date)"

