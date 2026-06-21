#!/bin/bash
#SBATCH --job-name=hybrid_detection
#SBATCH --output=logs/hybrid_detection_%j.out
#SBATCH --error=logs/hybrid_detection_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "Starting Hybrid Object Detection (YOLOv8 + Gemini)..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

WORKING_DIR="/path/to/SpatialReasonerDataGen"
cd "$WORKING_DIR"

mkdir -p logs

SIM_IMAGES_DIR="object_detection/sim_images"
OUTPUT_DIR="clustering/results/hybrid_detection_results_final"

mkdir -p "$OUTPUT_DIR"

echo "Processing all images in ${SIM_IMAGES_DIR}"
echo "Using YOLOv8 confidence threshold: 0.10 (lower = more sensitive detection)"

python object_detection/hybrid_detector.py \
    --image_dir "$SIM_IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --sm_names "SM_names.txt" \
    --taxonomy "clustering/results/final_taxonomy.json" \
    --yolo_model "yolov8x.pt" \
    --confidence 0.10 \
    --iou 0.45

echo "All images processed successfully!"

echo "Creating summary file..."
python -c "
import json
import os
import glob

output_dir = 'clustering/results/hybrid_detection_results_final'
summary_file = os.path.join(output_dir, 'all_detections_summary.json')

all_results = {}
total_objects = 0
total_yolo = 0

for subdir in glob.glob(os.path.join(output_dir, '*/')):
    image_name = os.path.basename(os.path.normpath(subdir))
    result_file = os.path.join(subdir, f'{image_name}_detection.json')
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            all_results[image_name] = {
                'image_path': data.get('metadata', {}).get('image_path', ''),
                'yolo_detections': data.get('metadata', {}).get('yolo_detections', 0),
                'sm_matched': data.get('metadata', {}).get('total_objects_detected', 0),
                'detected_objects': [obj['object_name'] for obj in data.get('detected_objects', [])]
            }
            total_objects += data.get('metadata', {}).get('total_objects_detected', 0)
            total_yolo += data.get('metadata', {}).get('yolo_detections', 0)

summary = {
    'total_images_processed': len(all_results),
    'total_yolo_detections': total_yolo,
    'total_sm_matched_objects': total_objects,
    'results_by_image': all_results
}

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to: {summary_file}')
print(f'Total images processed: {len(all_results)}')
print(f'Total YOLO detections: {total_yolo}')
print(f'Total SM matched objects: {total_objects}')
"

echo "Hybrid Object Detection job completed!"
echo "End time: $(date)"

