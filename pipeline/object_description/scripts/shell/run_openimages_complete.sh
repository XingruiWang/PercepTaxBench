#!/bin/bash
#SBATCH --job-name=openimages_complete_processing
#SBATCH --output=logs/openimages_complete_processing_%j.out
#SBATCH --error=logs/openimages_complete_processing_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=main

# Complete OpenImages processing with proper skip logic
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

echo "Starting Complete OpenImages Processing at $(date)"
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

echo "Processing all images with built-in skip logic..."
echo "The script will automatically skip already processed images"

# Use the original script without range limits - it has built-in skip logic
python "$WORKING_DIR/scripts/generate_3d_groundtruth_production.py" \
    --image_path "$SOURCE_DIR" \
    --output_path "$OUTPUT_DIR" \
    --batch_size 1 \
    --max_workers 1 \
    --device cuda \
    --enable_pose3d \
    --enable_pose_filtering \
    --generate_annotations

echo "Complete processing finished at $(date)"
echo "Check output directory: $OUTPUT_DIR"
echo "Check logs: $LOG_DIR"
