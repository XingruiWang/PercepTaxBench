#!/bin/bash
#SBATCH --job-name=fix_obj_desc
#SBATCH --output=logs/fix_object_descriptions_%j.out
#SBATCH --error=logs/fix_object_descriptions_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=main
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_user@your_cluster

# Load conda environment
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen

# Create logs directory if it doesn't exist
mkdir -p logs

# Set input and output files
INPUT_FILE="/path/to/project/object_descriptions_output/SM_descriptions.json"
OUTPUT_FILE="/path/to/project/object_descriptions_output/SM_descriptions_fixed.json"

echo "Starting Fix for Incomplete Object Descriptions"
echo "Input File: $INPUT_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Using Gemini API to fix only problematic entries"

# Run the fix script
python scripts/python/fix_incomplete_descriptions.py \
    --api_key "${GEMINI_API_KEY}" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --sleep_sec 2.0 \
    --model "gemini-2.0-flash-exp"

echo "Description fixing completed!"
echo "Output saved to: $OUTPUT_FILE"
