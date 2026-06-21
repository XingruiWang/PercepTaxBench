#!/bin/bash
#SBATCH --job-name=realimage_tsv
#SBATCH --output=logs/realimage_tsv_%j.out
#SBATCH --error=logs/realimage_tsv_%j.err
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
echo "Converting Real Image Benchmark to TSV..."
echo "=============================================="

# Run TSV conversion for real images
python aggregate_unified_qa.py \
    --input_dir ../../taxonomyQABench_realimage_manual \
    --output_file /path/to/VLMEvalKit/Data/taxonomy_manual.tsv

echo ""
echo "=============================================="
echo "Real image TSV conversion completed!"
echo "=============================================="

# Print summary
echo ""
echo "Summary:"
ls -lh /path/to/VLMEvalKit/Data/taxonomy.tsv 2>/dev/null || echo "No TSV"
