#!/bin/bash
#SBATCH --job-name=img_obj_desc_test
#SBATCH --output=logs/image_object_descriptions_test_%j.out
#SBATCH --error=logs/image_object_descriptions_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=main
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_user@your_cluster

# Load conda environment
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
cd /path/to/SpatialReasonerDataGen/object_description

# Create logs directory if it doesn't exist
mkdir -p logs

# Set input and output directories
OPENIMAGES_OUTPUT="/path/to/project/openimages_unified_output"
OUTPUT_DIR="/path/to/project/image_object_descriptions_output_test"

echo "=========================================="
echo "Image-Specific Object Description Generation TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Input Directory: $OPENIMAGES_OUTPUT"
echo "Output Directory: $OUTPUT_DIR"
echo "Processing first 5 images only for testing"
echo "Using Gemini API for image-specific descriptions"
echo "=========================================="

# Check if input directory exists and has processed images
if [ ! -d "$OPENIMAGES_OUTPUT" ]; then
    echo "Error: Input directory $OPENIMAGES_OUTPUT does not exist"
    echo "Please run the 3D ground truth pipeline first"
    exit 1
fi

# Count processed images
PROCESSED_IMAGES=$(find "$OPENIMAGES_OUTPUT" -maxdepth 1 -type d | wc -l)
PROCESSED_IMAGES=$((PROCESSED_IMAGES - 1))  # Subtract 1 for the directory itself

echo "Found $PROCESSED_IMAGES processed images in $OPENIMAGES_OUTPUT"

if [ "$PROCESSED_IMAGES" -eq 0 ]; then
    echo "Error: No processed images found in $OPENIMAGES_OUTPUT"
    echo "Please run the 3D ground truth pipeline first"
    exit 1
fi

# Run the image-specific object description generation on first 5 images
echo "Starting image-specific object description generation (TEST - first 5 images)..."
echo "This will generate detailed descriptions for each detected object in the first 5 images"
echo "Using the same 11-key structured format as general object descriptions"

python scripts/python/generate_image_object_descriptions.py \
    --input_dir "$OPENIMAGES_OUTPUT" \
    --output_dir "$OUTPUT_DIR" \
    --api_key "${GEMINI_API_KEY}" \
    --model_name "gemini-2.0-flash-exp" \
    --sleep_sec 2.0 \
    --start_idx 0 \
    --end_idx 5

# Check results
echo "=========================================="
echo "Image Object Description Generation Test Complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Generated description files: $(ls -1 $OUTPUT_DIR/*.json | wc -l)"
    echo "Sample output structure:"
    ls -la "$OUTPUT_DIR" | head -10
    
    # Check summary if available
    if [ -f "$OUTPUT_DIR/processing_summary.json" ]; then
        echo "Processing summary:"
        cat "$OUTPUT_DIR/processing_summary.json"
    fi
    
    # Show sample of first description file
    FIRST_FILE=$(ls -1 "$OUTPUT_DIR"/*.json | head -1)
    if [ -n "$FIRST_FILE" ] && [ "$FIRST_FILE" != "$OUTPUT_DIR/processing_summary.json" ]; then
        echo "Sample description file content ($(basename "$FIRST_FILE")):"
        head -50 "$FIRST_FILE"
    fi
else
    echo "No output directory created - check for errors"
fi

echo "=========================================="
echo "Test job completed at: $(date)"
echo "=========================================="
