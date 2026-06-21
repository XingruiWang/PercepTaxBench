#!/bin/bash
#SBATCH --job-name=consolidated_descriptions
#SBATCH --output=logs/consolidated_descriptions_%j.out
#SBATCH --error=logs/consolidated_descriptions_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4

# Set working directory
WORKING_DIR="/path/to/SpatialReasonerDataGen"
cd "$WORKING_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
eval "$(conda shell.bash hook)" && conda activate srdatagen

# Run consolidated descriptions generation
python openimages_3d_annotations/scripts/generate_consolidated_descriptions.py \
    --consolidated_objects "openimages_3d_annotations/results/consolidated_v13/consolidated_objects.txt" \
    --consolidation_mapping "openimages_3d_annotations/results/consolidated_v13/consolidation_mapping.json" \
    --existing_descriptions "object_description/results/object_descriptions_full.json" \
    --output_file "openimages_3d_annotations/results/consolidated_v13/consolidated_descriptions.json" \
    --api_key "${GEMINI_API_KEY}"

echo "Consolidated descriptions generation completed!"
