#!/bin/bash
#SBATCH --job-name=consolidate_objects
#SBATCH --output=logs/consolidate_objects_%j.out
#SBATCH --error=logs/consolidate_objects_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=main

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
WORKING_DIR="/path/to/SpatialReasonerDataGen"
cd "$WORKING_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run object consolidation with K-means for 800 clusters (more granular)
python openimages_3d_annotations/scripts/consolidate_hierarchical_objects.py \
    --objects_file "openimages_3d_annotations/data/open_images_objects_no_occupations.txt" \
    --output_dir "openimages_3d_annotations/results/consolidated_v13" \
    --min_cluster_size 2 \
    --min_samples 2 \
    --cluster_selection_epsilon 700

echo "Object consolidation completed!"
