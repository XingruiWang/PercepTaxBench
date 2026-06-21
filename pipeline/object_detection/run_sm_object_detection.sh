#!/bin/bash
#SBATCH --job-name=sm_object_detection
#SBATCH --output=logs/sm_object_detection_%j.out
#SBATCH --error=logs/sm_object_detection_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=main
#SBATCH --gres=gpu:1

echo "Starting SM Object Detection..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

# Set working directory
WORKING_DIR="/path/to/SpatialReasonerDataGen"
cd "$WORKING_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Install required packages if not available
pip install ultralytics opencv-python

# Process all images in sim_images folder
SIM_IMAGES_DIR="object_detection/sim_images"
OUTPUT_DIR="object_detection/results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each image
for image_file in "$SIM_IMAGES_DIR"/*.png; do
    if [ -f "$image_file" ]; then
        # Extract filename without extension
        filename=$(basename "$image_file" .png)
        
        echo "Processing: $image_file"
        
        # Run the SM object detection for this image
        python object_detection/object_detection_sm.py \
            --image "$image_file" \
            --output "$OUTPUT_DIR/${filename}_detection_results.json" \
            --visualization "$OUTPUT_DIR/${filename}_detection_visualization.jpg" \
            --confidence 0.5 \
            --sm_names "SM_names.txt" \
            --taxonomy "clustering/results/final_taxonomy.json"
        
        echo "Completed: $image_file"
    fi
done

echo "All images processed successfully!"

echo "SM Object Detection job completed!"
echo "End time: $(date)"
