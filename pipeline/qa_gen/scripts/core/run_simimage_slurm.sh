#!/bin/bash
#SBATCH --job-name=simimage_qa
#SBATCH --output=logs/simimage_qa_%j.out
#SBATCH --error=logs/simimage_qa_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Activate conda environment in non-interactive shell
if [[ -x "$(command -v conda)" ]]; then
    eval "$(conda shell.bash hook)"
fi
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen/qa_gen/

# Use a large TMPDIR for unzip/extraction to avoid no-space issues
export TMPDIR="/path/to/tmp/simimage_tmp"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

# Create logs directory
mkdir -p logs

# Remove old output directory before running
echo "=============================================="
echo "Cleaning previous results..."
echo "=============================================="
find taxonomyQABench_simimage -mindepth 1 -delete 2>/dev/null
echo "Previous output directory removed"
echo ""

# Run sim image benchmark
echo "=============================================="
echo "Running Sim Image Benchmark (ALL groups)"
echo "=============================================="

python scripts/core/generate_taxonomyqabench_simimage.py \
    --images_dir /path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations \
    --output_dir taxonomyQABench_simimage \
    --seed 42 \
    --skip_reasoning_refresh

echo ""
echo "=============================================="
echo "Sim image benchmark generation completed!"
echo "=============================================="


echo ""
echo "Summary:"
ls -lh taxonomyQABench_simimage/*.json 2>/dev/null || echo "No results"
echo ""
