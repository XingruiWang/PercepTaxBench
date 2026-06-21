#!/bin/bash
#SBATCH --job-name=realimage_qa
#SBATCH --output=logs/realimage_qa_%j.out
#SBATCH --error=logs/realimage_qa_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Activate conda environment
source ~/.bashrc
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen/qa_gen/scripts/core

# Create logs directory
mkdir -p logs

# Remove old output
rm -rf ../../taxonomyQABench_realimage

# Run real image benchmark
echo "=============================================="
echo "Running Real Image Benchmark (ALL images)"
echo "=============================================="

python generate_taxonomyqabench_realimage.py \
    --images_dir ../../openimages_unified_output \
    --output_dir taxonomyQABench_realimage \
    --seed 42

echo ""
echo "=============================================="
echo "Real image benchmark generation completed!"
echo "=============================================="

# Print summary
echo ""
echo "Summary:"
ls -lh ../../taxonomyQABench_realimage/*.json 2>/dev/null || echo "No results"
echo ""
echo "Next step: Convert to TSV using run_realimage_tsv_slurm.sh"

