#!/bin/bash
#SBATCH --job-name=openimages_resume
#SBATCH --output=logs/openimages_resume_%j.out
#SBATCH --error=logs/openimages_resume_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=main

# Configuration for resuming OpenImages processing from where we left off
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

echo "=== RESUMING OpenImages Processing ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Working directory: $WORKING_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Stats directory: $STATS_DIR"

# Check current status
TOTAL_IMAGES=$(ls /path/to/project/openimages_train_10000/ | grep -E "\.(jpg|jpeg|png)$" | wc -l)
PROCESSED_IMAGES=$(ls "$OUTPUT_DIR" | wc -l)
MISSING_IMAGES=$((TOTAL_IMAGES - PROCESSED_IMAGES))

echo "Status Check:"
echo "  Total images in source: $TOTAL_IMAGES"
echo "  Already processed: $PROCESSED_IMAGES"
echo "  Missing images: $MISSING_IMAGES"

if [ $MISSING_IMAGES -eq 0 ]; then
    echo "All images have been processed! No work needed."
    exit 0
fi

echo "Resuming processing for $MISSING_IMAGES missing images..."
echo "Using rolling statistics with checkpointing every 50 images"

# Process remaining images with rolling statistics and checkpointing
python "$WORKING_DIR/scripts/generate_3d_groundtruth_production_rolling.py" \
    --image_path /path/to/project/openimages_train_10000 \
    --output_path "$OUTPUT_DIR" \
    --stats_path "$STATS_DIR" \
    --range_low 0 \
    --range_high 10000 \
    --batch_size 16 \
    --max_workers 8 \
    --device cuda \
    --enable_pose3d \
    --enable_pose_filtering \
    --generate_annotations \
    --log_level INFO \
    --checkpoint_interval 50 \
    --resume_from_checkpoint

echo ""
echo "=== PROCESSING COMPLETED ==="
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID finished"

# Final status check
FINAL_PROCESSED=$(ls "$OUTPUT_DIR" | wc -l)
FINAL_MISSING=$((TOTAL_IMAGES - FINAL_PROCESSED))

echo "Final Status:"
echo "  Total images: $TOTAL_IMAGES"
echo "  Processed: $FINAL_PROCESSED"
echo "  Still missing: $FINAL_MISSING"
echo "  Completion rate: $(( (FINAL_PROCESSED * 100) / TOTAL_IMAGES ))%"

if [ $FINAL_MISSING -eq 0 ]; then
    echo "🎉 ALL IMAGES PROCESSED SUCCESSFULLY!"
else
    echo "⚠️  $FINAL_MISSING images still need processing"
fi
