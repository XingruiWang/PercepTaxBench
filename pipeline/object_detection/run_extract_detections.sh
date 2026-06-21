#!/bin/bash

#SBATCH --job-name=extract_2d_detections
#SBATCH --output=../logs/extract_2d_detections_%j.out
#SBATCH --error=../logs/extract_2d_detections_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Starting 2D detection extraction..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

cd "$(dirname "$0")"

conda activate srdatagen

echo "Step 1: Extracting 2D detection data from unified outputs..."
python3 extract_2d_detections.py \
    --source_dir ../../openimages_unified_output \
    --target_dir ./results/object_detection_extracted

echo "Step 2: Generating 2D bounding box visualizations..."
python3 generate_2d_bbox_images.py \
    --detection_dir ./results/object_detection_extracted \
    --image_source_dir ../../openimages_train_10000 \
    --output_dir ./results/object_detection_extracted

echo "Extraction and visualization complete!"
echo "Results saved to: ./results/object_detection_extracted"
echo "Date: $(date)"
