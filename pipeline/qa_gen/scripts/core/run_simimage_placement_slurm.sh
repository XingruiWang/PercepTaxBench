#!/bin/bash
#SBATCH --job-name=simimage_qa_placement
#SBATCH --output=logs/simimage_qa_placement_%j.out
#SBATCH --error=logs/simimage_qa_placement_%j.err
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

# Remove old output directory before running
echo "=============================================="
echo "Cleaning previous results..."
echo "=============================================="
rm -rf ../../taxonomyQABench_sim_placement
echo "Previous output directory removed"
echo ""

# Run sim image benchmark for placement data
echo "=============================================="
echo "Running Sim Image Benchmark (placement data)"
echo "=============================================="

python generate_taxonomyqabench_simimage.py \
    --images_dir /path/to/sim_images/placement \
    --output_dir taxonomyQABench_sim_placement \
    --seed 42

echo ""
echo "=============================================="
echo "Sim image benchmark generation completed!"
echo "=============================================="

# Print summary
echo ""
echo "Summary:"
ls -lh ../../taxonomyQABench_sim_placement/*.json 2>/dev/null || echo "No results"
echo ""
echo "Next step: Convert to TSV using run_simimage_placement_tsv_slurm.sh"

