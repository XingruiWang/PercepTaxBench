#!/bin/bash
#SBATCH --job-name=openimages_missing_only
#SBATCH --output=logs/openimages_missing_only_%j.out
#SBATCH --error=logs/openimages_missing_only_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=main

# Process only missing OpenImages
WORKING_DIR="/path/to/SpatialReasonerDataGen"
OUTPUT_DIR="/path/to/project/openimages_unified_output"
SOURCE_DIR="/path/to/project/openimages_train_10000"
LOG_DIR="$WORKING_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Activate conda environment properly
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set PYTHONPATH to include SAM2
export PYTHONPATH="/path/to/SpatialReasonerDataGen/sam-hq/sam-hq2:$PYTHONPATH"

cd "$WORKING_DIR"

echo "Starting Missing Images Only Processing at $(date)"
echo "Source directory: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Working directory: $WORKING_DIR"

# Count total source images
TOTAL_SOURCE=$(ls "$SOURCE_DIR"/*.jpg | wc -l)
echo "Total source images: $TOTAL_SOURCE"

# Count already processed images
PROCESSED_COUNT=$(ls "$OUTPUT_DIR" | wc -l)
echo "Already processed images: $PROCESSED_COUNT"

# Calculate remaining
REMAINING=$((TOTAL_SOURCE - PROCESSED_COUNT))
echo "Remaining images to process: $REMAINING"

if [ $REMAINING -eq 0 ]; then
    echo "All images already processed! Exiting."
    exit 0
fi

echo "Processing remaining $REMAINING images..."
echo "Using range-based processing to avoid reprocessing existing images"

# Process images in batches to avoid reprocessing
# We'll process from 8066+ to avoid conflicts
python "$WORKING_DIR/scripts/generate_3d_groundtruth_production.py" \
    --image_path "$SOURCE_DIR" \
    --output_path "$OUTPUT_DIR" \
    --range_low 8066 \
    --range_high 10000 \
    --batch_size 1 \
    --max_workers 1 \
    --device cuda \
    --enable_pose3d \
    --enable_pose_filtering \
    --generate_annotations

echo "Missing images processing completed at $(date)"
echo "Check output directory: $OUTPUT_DIR"
echo "Check logs: $LOG_DIR"
