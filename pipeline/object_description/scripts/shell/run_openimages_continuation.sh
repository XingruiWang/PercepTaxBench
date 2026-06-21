#!/bin/bash
#SBATCH --job-name=openimages_continuation
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a5000:1
#SBATCH --partition=main
#SBATCH --exclude=ccvl35

# Configuration for continuing OpenImages dataset processing
WORKING_DIR="/path/to/SpatialReasonerDataGen"
OUTPUT_DIR="/path/to/project/openimages_unified_output"
LOG_DIR="$WORKING_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Activate conda environment properly
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Verify we're in the right environment
echo "Using conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python path: $(which python)"

cd "$WORKING_DIR"

# Add models directory to Python path to fix SAM2 imports
export PYTHONPATH="$WORKING_DIR/models:$WORKING_DIR/../sam-hq/sam-hq2:$PYTHONPATH"
echo "Python path configured for local models"

# GPU optimization settings - conservative for stability
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:False
export CUDA_CACHE_DISABLE=1
echo "Conservative GPU settings configured for stability"

echo "Starting OpenImages continuation processing at $(date)"
echo "Continuing from image index: 9076"
echo "Processing to image index: 10000"
echo "Total remaining images: 924"
echo "Working directory: $WORKING_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: 2 (conservative for stability)"
echo "Max workers: 1 (single-threaded for stability)"
echo "GPU count: 1 (RTX A5000)"
echo "Device: CUDA (GPU acceleration enabled)"
echo "Generating: 3D visualizations and object crops enabled"
echo "Updating: Object descriptions summary for later processing"

# Check if script exists
if [ ! -f "$WORKING_DIR/scripts/generate_3d_groundtruth_production.py" ]; then
    echo "ERROR: Script not found at $WORKING_DIR/scripts/generate_3d_groundtruth_production.py"
    exit 1
fi

# Check if input directory exists
if [ ! -d "/path/to/project/openimages_train_10000" ]; then
    echo "ERROR: Input directory not found"
    exit 1
fi

# Continue processing from image 7065 to 10000
echo "Running Python script..."
export HYDRA_FULL_ERROR=1
python "$WORKING_DIR/scripts/generate_3d_groundtruth_production.py" \
    --image_path /path/to/project/openimages_train_10000 \
    --output_path "$OUTPUT_DIR" \
    --range_low 9076 \
    --range_high 10000 \
    --batch_size 2 \
    --max_workers 1 \
    --device cuda \
    --enable_pose3d \
    --enable_pose_filtering \
    --generate_annotations \
    --log_level INFO

# Check exit code
if [ $? -ne 0 ]; then
    echo "ERROR: Python script failed with exit code $?"
    exit 1
fi

echo "OpenImages continuation processing finished at $(date)"
echo "Check output directory: $OUTPUT_DIR"
echo "Check logs: $LOG_DIR"

# Update object descriptions summary using separate script
echo ""
echo "=== Updating Object Descriptions Summary ==="
echo "Running separate object descriptions script to update object_descriptions_full.json..."

# Run the separate object descriptions script
if [ -f "$WORKING_DIR/object_description/run_object_descriptions_from_openimages.sh" ]; then
    echo "Submitting object descriptions update job..."
    sbatch "$WORKING_DIR/object_description/run_object_descriptions_from_openimages.sh"
    echo "Object descriptions update job submitted successfully"
else
    echo "Warning: Object descriptions script not found at $WORKING_DIR/object_description/run_object_descriptions_from_openimages.sh"
    echo "Please run the object descriptions update manually after this job completes"
fi

# Calculate final processing statistics
echo ""
echo "=== Final Processing Statistics ==="
TOTAL_PROCESSED=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*" | grep -v "^\.$" | wc -l)
echo "Total images processed: $TOTAL_PROCESSED"
echo "Target images: 10000"
echo "Completion percentage: $(echo "scale=2; $TOTAL_PROCESSED * 100 / 10000" | bc)%"
