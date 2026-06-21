#!/bin/bash
#SBATCH --job-name=simimage_qa_placement_tsv
#SBATCH --output=logs/simimage_qa_placement_tsv_%j.out
#SBATCH --error=logs/simimage_qa_placement_tsv_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Activate conda environment
source ~/.bashrc
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen/qa_gen/scripts/core

# Create logs directory
mkdir -p logs

# Run TSV conversion for placement data
echo "=============================================="
echo "Converting taxonomyQABench_sim_placement to TSV"
echo "=============================================="

python aggregate_unified_qa.py \
    --input_dir ../../taxonomyQABench_sim_placement \
    --output_file ../../../qa_eval/VLMEvalKit/Data/taxonomyQABench_sim_placement.tsv

echo ""
echo "=============================================="
echo "TSV conversion completed!"
echo "=============================================="
echo ""
echo "Output file: ../../../qa_eval/VLMEvalKit/Data/taxonomyQABench_sim_placement.tsv"
ls -lh ../../../qa_eval/VLMEvalKit/Data/taxonomyQABench_sim_placement.tsv 2>/dev/null || echo "File not found"

