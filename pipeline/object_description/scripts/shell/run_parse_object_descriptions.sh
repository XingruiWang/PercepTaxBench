#!/bin/bash
#SBATCH --job-name=parse_descriptions
#SBATCH --output=logs/parse_descriptions_%j.out
#SBATCH --error=logs/parse_descriptions_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=main
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "Starting object description parsing at $(date)"
echo "Parsing object_descriptions_full.json to extract structured attributes"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen

# Run the parsing script
python object_description/parse_mixed_concept_descriptions.py \
    --input_file object_description/results/object_descriptions_full.json \
    --output_dir object_description/results/parsed \
    --api_key "${GEMINI_API_KEY}" \
    --output_filename parsed_object_descriptions_full.json

echo "Parsing completed at $(date)"
echo "Check output: object_description/results/parsed/parsed_object_descriptions_full.json"
