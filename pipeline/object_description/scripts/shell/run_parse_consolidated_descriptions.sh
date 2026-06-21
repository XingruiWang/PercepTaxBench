#!/bin/bash
#SBATCH --job-name=parse_consolidated
#SBATCH --output=logs/parse_consolidated_%j.out
#SBATCH --error=logs/parse_consolidated_%j.err
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

# Run consolidated descriptions parsing
python openimages_3d_annotations/scripts/parse_consolidated_descriptions.py \
    --descriptions_file "openimages_3d_annotations/results/consolidated_v13/consolidated_descriptions.json" \
    --output_file "openimages_3d_annotations/results/consolidated_v13/consolidated_descriptions_parsed.json" \
    --api_key "${GEMINI_API_KEY}"

echo "Consolidated descriptions parsing completed!"
