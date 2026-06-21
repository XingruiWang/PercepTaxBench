#!/bin/bash
#SBATCH --job-name=openimages_desc
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/openimages_descriptions_%j.out
#SBATCH --error=logs/openimages_descriptions_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate srdatagen

# Change to project directory
cd /path/to/project

echo "Starting OpenImages object description generation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

# Run the description generation script
python taxonomy_datagen/SpatialReasonerDataGen/openimages_3d_annotations/scripts/generate_openimages_descriptions.py

echo "OpenImages description generation completed at: $(date)"
