#!/bin/bash
#SBATCH --job-name=openimages_actual_missing
#SBATCH --output=logs/openimages_actual_missing_%j.out
#SBATCH --error=logs/openimages_actual_missing_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=main

# Process only the actual missing OpenImages
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

echo "Starting Actual Missing Images Processing at $(date)"
echo "Source directory: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Working directory: $WORKING_DIR"

# Get the actual missing images from verification script
echo "Identifying actual missing images..."
python /path/to/project/verify_openimages_outputs.py > /tmp/missing_images_analysis.txt 2>&1

# Extract missing images from the analysis
grep -A 2000 "First 10 missing images:" /tmp/missing_images_analysis.txt | grep -E "^  [0-9a-f]" | sed 's/^  //' > /tmp/missing_images_list.txt

MISSING_COUNT=$(wc -l < /tmp/missing_images_list.txt)
echo "Found $MISSING_COUNT actual missing images"

if [ $MISSING_COUNT -eq 0 ]; then
    echo "No missing images found! Exiting."
    exit 0
fi

echo "First 10 missing images:"
head -10 /tmp/missing_images_list.txt

echo "Processing $MISSING_COUNT missing images..."
echo "Using individual image processing to avoid conflicts"

# Process each missing image individually
PROCESSED=0
FAILED=0

while IFS= read -r image_id; do
    if [ -z "$image_id" ]; then
        continue
    fi
    
    echo "Processing image $image_id ($((PROCESSED + FAILED + 1))/$MISSING_COUNT)..."
    
    # Check if image exists in source
    if [ ! -f "$SOURCE_DIR/${image_id}.jpg" ]; then
        echo "Source image ${image_id}.jpg not found, skipping"
        ((FAILED++))
        continue
    fi
    
    # Process single image using the original script
    python "$WORKING_DIR/scripts/generate_3d_groundtruth_production.py" \
        --image_path "$SOURCE_DIR" \
        --output_path "$OUTPUT_DIR" \
        --range_low 0 \
        --range_high 1 \
        --batch_size 1 \
        --max_workers 1 \
        --device cuda \
        --enable_pose3d \
        --enable_pose_filtering \
        --generate_annotations \
        --md5 "$image_id" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        ((PROCESSED++))
        echo "✅ Successfully processed $image_id"
    else
        ((FAILED++))
        echo "❌ Failed to process $image_id"
    fi
    
    # Progress update every 10 images
    if [ $((PROCESSED + FAILED)) -gt 0 ] && [ $((PROCESSED + FAILED)) -eq $((PROCESSED + FAILED)) ]; then
        echo "Progress: $((PROCESSED + FAILED))/$MISSING_COUNT processed, $PROCESSED successful, $FAILED failed"
    fi
    
done < /tmp/missing_images_list.txt

echo "Actual missing images processing completed at $(date)"
echo "Final results: $PROCESSED successful, $FAILED failed out of $MISSING_COUNT total"
echo "Check output directory: $OUTPUT_DIR"
echo "Check logs: $LOG_DIR"

# Clean up temporary files
rm -f /tmp/missing_images_analysis.txt /tmp/missing_images_list.txt
