#!/bin/bash
#SBATCH --job-name=parse_descriptions_concurrent
#SBATCH --output=logs/parse_descriptions_concurrent_%j.out
#SBATCH --error=logs/parse_descriptions_concurrent_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=main
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12

echo "Starting concurrent object description parsing at $(date)"
echo "Parsing object_descriptions_full.json with 3 concurrent API keys"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen

# Run the concurrent parsing script with 3 API keys
python object_description/parse_mixed_concept_descriptions_concurrent.py \
    --input_file object_description/results/object_descriptions_full.json \
    --output_dir object_description/results/parsed \
    --api_keys ""${GEMINI_API_KEY}","${GEMINI_API_KEY}","${GEMINI_API_KEY}"" \
    --output_filename parsed_object_descriptions_full.json \
    --max_workers 3

echo "Concurrent parsing completed at $(date)"
echo "Check output: object_description/results/parsed/parsed_object_descriptions_full.json"
