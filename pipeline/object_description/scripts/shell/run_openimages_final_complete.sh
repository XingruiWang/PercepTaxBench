#!/bin/bash
#SBATCH --job-name=openimages_final_complete
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/openimages_final_complete_%j.out
#SBATCH --error=logs/openimages_final_complete_%j.err

echo "Starting final OpenImages processing to complete all remaining images..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/project/taxonomy_datagen:$PYTHONPATH

# Run the processing script with range-based processing for missing images
cd /path/to/SpatialReasonerDataGen/scripts

# Process remaining images using range-based approach
python generate_3d_groundtruth_production.py \
    --image_path /path/to/project/openimages_train_10000 \
    --output_path /path/to/project/openimages_unified_output \
    --range_low 9076 \
    --range_high 10000 \
    --batch_size 1 \
    --device cuda

echo "Final OpenImages processing completed at $(date)"
echo "Check output directory: /path/to/project/openimages_unified_output"
echo "Check logs: /path/to/SpatialReasonerDataGen/logs"
