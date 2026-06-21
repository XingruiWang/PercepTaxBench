#!/bin/bash
#SBATCH --job-name=img_obj_desc
#SBATCH --output=logs/image_object_descriptions_%j.out
#SBATCH --error=logs/image_object_descriptions_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
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
OUTPUT_DIR="/path/to/project/image_object_descriptions_output"

echo "=========================================="
echo "Image-Specific Object Description Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Input Directory: $OPENIMAGES_OUTPUT"
echo "Output Directory: $OUTPUT_DIR"
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

# Run the image-specific object description generation
echo "Starting image-specific object description generation..."
echo "This will generate detailed descriptions for each detected object in each image"
echo "Using the same 11-key structured format as general object descriptions"

python scripts/python/generate_image_object_descriptions.py \
    --input_dir "$OPENIMAGES_OUTPUT" \
    --output_dir "$OUTPUT_DIR" \
    --api_key "${GEMINI_API_KEY}" \
    --model_name "gemini-2.5-flash" \
    --sleep_sec 2.0

# Check results
echo "=========================================="
echo "Image Object Description Generation Complete"
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
else
    echo "No output directory created - check for errors"
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
