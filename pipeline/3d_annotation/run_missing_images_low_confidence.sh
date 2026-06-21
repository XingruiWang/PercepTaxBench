#!/bin/bash
#SBATCH --job-name=missing_low_conf
#SBATCH --output=logs/missing_low_confidence_%j.out
#SBATCH --error=logs/missing_low_confidence_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=main

# Configuration for processing missing images with LOWER confidence thresholds
WORKING_DIR="/path/to/SpatialReasonerDataGen/3d_annotation"
MISSING_IMAGES_DIR="/path/to/project/openimages_missing"
OUTPUT_DIR="/path/to/project/openimages_missing_output_low_conf"
LOG_DIR="$WORKING_DIR/logs"

# Create separate output directory for review
mkdir -p "$OUTPUT_DIR"

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

echo "========================================"
echo "Processing Missing Images with LOW Confidence Thresholds"
echo "========================================"
echo "Start time: $(date)"
echo "Working directory: $WORKING_DIR"
echo "Missing images directory: $MISSING_IMAGES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "LOWERED THRESHOLDS:"
echo "  - Box threshold: 0.5 (was 0.65)"
echo "  - Text threshold: 0.5 (was 0.65)"
echo "  - Mask confidence: 0.35 (was 0.65)"
echo "  - Min confidence: 0.2 (was 0.3)"
echo "  - Min pose confidence: 0.3 (was 0.5)"
echo ""
echo "Total missing images: $(ls $MISSING_IMAGES_DIR/*.jpg 2>/dev/null | wc -l)"
echo "========================================"

# Temporarily swap config files to use low confidence settings
echo "Swapping to low confidence config..."
mv "$WORKING_DIR/config.py" "$WORKING_DIR/config_original_backup.py"
cp "$WORKING_DIR/config_low_confidence.py" "$WORKING_DIR/config.py"

# Process missing images with low confidence settings
python "$WORKING_DIR/generate_3d_groundtruth_production.py" \
    --image_path "$MISSING_IMAGES_DIR" \
    --output_path "$OUTPUT_DIR" \
    --range_low 0 \
    --range_high 923 \
    --batch_size 16 \
    --max_workers 4 \
    --device cuda \
    --enable_pose3d \
    --enable_pose_filtering \
    --generate_annotations \
    --skip_visualizations \
    --skip_object_crops \
    --log_level INFO

# Restore original config
echo ""
echo "Restoring original config..."
mv "$WORKING_DIR/config_original_backup.py" "$WORKING_DIR/config.py"

echo ""
echo "========================================"
echo "Missing images processing finished at $(date)"
echo "========================================"
echo "Check output directory: $OUTPUT_DIR"
echo "Check logs: $LOG_DIR"

# Calculate processing statistics
echo ""
echo "=== Processing Statistics ==="
TOTAL_IMAGES=923
PROCESSED_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d | grep -E "[0-9a-f]{16}" | wc -l)
echo "Total missing images: $TOTAL_IMAGES"
echo "Successfully processed: $PROCESSED_COUNT"
echo "Success rate: $(echo "scale=2; $PROCESSED_COUNT * 100 / $TOTAL_IMAGES" | bc)%"

# Check which images still failed
echo ""
echo "=== Checking for remaining failures ==="
STILL_MISSING=0
for img in "$MISSING_IMAGES_DIR"/*.jpg; do
    img_name=$(basename "$img" .jpg)
    if [ ! -d "$OUTPUT_DIR/$img_name" ]; then
        STILL_MISSING=$((STILL_MISSING + 1))
    fi
done
echo "Images still without detections: $STILL_MISSING"

# Restore original config
echo ""
echo "Restoring original config..."
mv "$WORKING_DIR/config_original_backup.py" "$WORKING_DIR/config.py"

# Run analysis script to identify new objects
echo ""
echo "========================================"
echo "Analyzing Detected Objects"
echo "========================================"
python "$WORKING_DIR/analyze_missing_detections.py" \
    --missing_output_dir "$OUTPUT_DIR" \
    --existing_descriptions "/path/to/SpatialReasonerDataGen/object_description/results/filtered_full/parsed_concepts_filtered_full.json"

echo ""
echo "========================================"
echo "REVIEW REQUIRED"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "NEXT STEPS:"
echo "1. Review detection_analysis.json in $OUTPUT_DIR"
echo "2. Check new_objects_list.txt for objects needing descriptions"
echo "3. Generate descriptions for new objects if needed"
echo "4. Add new objects to taxonomy if needed"
echo "5. Once approved, merge results to openimages_unified_output"
echo "======================================"
