#!/bin/bash
#SBATCH --job-name=3d_annot_only
#SBATCH --output=logs/3d_annotation_only_%j.out
#SBATCH --error=logs/3d_annotation_only_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=main

# Configuration for 3D annotation generation WITHOUT visualizations
WORKING_DIR="/path/to/SpatialReasonerDataGen/3d_annotation"
OUTPUT_DIR="/path/to/project/openimages_unified_output"
LOG_DIR="$WORKING_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Verify environment
echo "Using conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python path: $(which python)"

cd "$WORKING_DIR"

echo "Starting 3D annotation generation (NO visualizations) at $(date)"
echo "Working directory: $WORKING_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Generating: 3D point cloud, 2D bbox, pose annotations ONLY"
echo "Skipping: 3D visualizations and object crops"

# Process images with 3D annotations only (no visualizations or crops)
python "$WORKING_DIR/generate_3d_groundtruth_production.py" \
    --image_path /path/to/project/openimages_train_10000 \
    --output_path "$OUTPUT_DIR" \
    --range_low 0 \
    --range_high 10000 \
    --batch_size 32 \
    --max_workers 8 \
    --device cuda \
    --enable_pose3d \
    --enable_pose_filtering \
    --generate_annotations \
    --skip_visualizations \
    --skip_object_crops \
    --log_level INFO

echo "3D annotation generation finished at $(date)"
echo "Check output directory: $OUTPUT_DIR"
echo "Check logs: $LOG_DIR"

# Calculate processing statistics
echo ""
echo "=== Processing Statistics ==="
TOTAL_PROCESSED=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*" | grep -v "^\.$" | wc -l)
echo "Total images processed: $TOTAL_PROCESSED"
echo "Processing rate: $(echo "scale=2; $TOTAL_PROCESSED / 48" | bc) images per hour"
echo "Estimated completion time: $(echo "scale=1; (10000 - $TOTAL_PROCESSED) / ($TOTAL_PROCESSED / 48)" | bc) hours"
