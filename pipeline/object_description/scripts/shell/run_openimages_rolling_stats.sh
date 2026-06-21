#!/bin/bash
#SBATCH --job-name=openimages_rolling_stats
#SBATCH --output=logs/openimages_rolling_stats_%j.out
#SBATCH --error=logs/openimages_rolling_stats_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=main

# Configuration for OpenImages processing with rolling statistics
WORKING_DIR="/path/to/SpatialReasonerDataGen"
OUTPUT_DIR="/path/to/project/openimages_unified_output"
LOG_DIR="$WORKING_DIR/logs"
STATS_DIR="$WORKING_DIR/rolling_stats"

# Create directories
mkdir -p "$LOG_DIR" "$STATS_DIR"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

cd "$WORKING_DIR"

echo "Starting OpenImages processing with ROLLING STATISTICS at $(date)"
echo "Total images to process: 10000"
echo "Working directory: $WORKING_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Stats directory: $STATS_DIR"
echo "Rolling stats will be saved every 100 images processed"
echo "Processing will resume from last checkpoint if interrupted"

# Process with rolling statistics and checkpointing
python "$WORKING_DIR/scripts/generate_3d_groundtruth_production_rolling.py" \
    --image_path /path/to/project/openimages_train_10000 \
    --output_path "$OUTPUT_DIR" \
    --stats_path "$STATS_DIR" \
    --range_low 0 \
    --range_high 10000 \
    --batch_size 32 \
    --max_workers 16 \
    --device cuda \
    --enable_pose3d \
    --enable_pose_filtering \
    --generate_annotations \
    --log_level INFO \
    --checkpoint_interval 100 \
    --resume_from_checkpoint

echo "OpenImages processing with rolling statistics finished at $(date)"
echo "Final statistics saved to: $STATS_DIR"
echo "Check output directory: $OUTPUT_DIR"
echo "Check logs: $LOG_DIR"
