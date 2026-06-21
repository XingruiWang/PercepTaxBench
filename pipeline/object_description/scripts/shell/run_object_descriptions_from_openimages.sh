#!/bin/bash
#SBATCH --job-name=obj_desc_openimages
#SBATCH --output=logs/object_descriptions_openimages_%j.out
#SBATCH --error=logs/object_descriptions_openimages_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=main
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_user@your_cluster

# Load conda environment
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen/object_description

# Create logs directory if it doesn't exist
mkdir -p logs

# Set input and output directories
OPENIMAGES_OUTPUT="/path/to/project/openimages_unified_output"
OUTPUT_FILE="/path/to/project/object_descriptions_output/object_descriptions_full.json"

echo "Starting Object Description Generation from OpenImages Output"
echo "OpenImages Output Directory: $OPENIMAGES_OUTPUT"
echo "Output File: $OUTPUT_FILE"
echo "Using Gemini API for structured descriptions"

# Run the object description generation
python scripts/python/generate_object_descriptions.py \
    --api_key "${GEMINI_API_KEY}" \
    --output_dir "$OPENIMAGES_OUTPUT" \
    --output_file "$OUTPUT_FILE" \
    --sleep_sec 2.0 \
    --model "gemini-2.5-flash"

echo "Object description generation completed!"
echo "Output saved to: $OUTPUT_FILE"
