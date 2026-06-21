#!/bin/bash
#SBATCH --job-name=obj_desc_file
#SBATCH --output=logs/object_descriptions_file_%j.out
#SBATCH --error=logs/object_descriptions_file_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
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

# Set input and output files
INPUT_FILE="SM_names.txt"
OUTPUT_FILE="/path/to/project/object_descriptions_output/SM_descriptions.json"

echo "Starting Object Description Generation from Text File"
echo "Input File: $INPUT_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Using Gemini API for structured descriptions"

# Run the object description generation
python scripts/python/generate_object_descriptions.py \
    --api_key "${GEMINI_API_KEY}" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --sleep_sec 2.0 \
    --model "gemini-2.5-flash"

echo "Object description generation completed!"
echo "Output saved to: $OUTPUT_FILE"
