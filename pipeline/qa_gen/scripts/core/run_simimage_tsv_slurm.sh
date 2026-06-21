#!/bin/bash
#SBATCH --job-name=simimage_tsv
#SBATCH --output=logs/simimage_tsv_%j.out
#SBATCH --error=logs/simimage_tsv_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# Activate conda environment
source ~/.bashrc
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen/qa_gen/scripts/core

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "Converting Sim Image Benchmark to TSV..."
echo "=============================================="

# Run TSV conversion for sim images (using unified script)
python aggregate_unified_qa.py \
    --input_dir ../../taxonomyQABench_simimage \
    --output_file /path/to/VLMEvalKit/Data/taxonomy_sim.tsv

echo ""
echo "=============================================="
echo "Sim image TSV conversion completed!"
echo "=============================================="

# Print summary
echo ""
echo "Summary:"
ls -lh /path/to/VLMEvalKit/Data/taxonomy_sim.tsv 2>/dev/null || echo "No TSV"
