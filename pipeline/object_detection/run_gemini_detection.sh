#!/bin/bash
#SBATCH --job-name=gemini_object_detection
#SBATCH --output=logs/gemini_object_detection_%j.out
#SBATCH --error=logs/gemini_object_detection_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=main

echo "Starting Gemini Object Detection..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

WORKING_DIR="/path/to/SpatialReasonerDataGen"
cd "$WORKING_DIR"

mkdir -p logs

SIM_IMAGES_DIR="object_detection/sim_images"
OUTPUT_DIR="clustering/results/gemini_detection_results"

mkdir -p "$OUTPUT_DIR"

echo "Processing all images in ${SIM_IMAGES_DIR}"

python object_detection/gemini_object_detection.py \
    --image_dir "$SIM_IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --sm_names "SM_names.txt" \
    --taxonomy "clustering/results/final_taxonomy.json"

echo "All images processed successfully!"

echo "Creating summary file..."
python -c "
import json
import os
import glob

output_dir = 'clustering/results/gemini_detection_results'
summary_file = os.path.join(output_dir, 'all_detections_summary.json')

all_results = {}
total_objects = 0

for subdir in glob.glob(os.path.join(output_dir, '*/')):
    image_name = os.path.basename(os.path.normpath(subdir))
    result_file = os.path.join(subdir, f'{image_name}_detection.json')
    
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                all_results[image_name] = {
                    'image_path': data.get('metadata', {}).get('image_path', ''),
                    'total_objects': data.get('metadata', {}).get('total_objects_detected', 0),
                    'detected_objects': [obj['object_name'] for obj in data.get('detected_objects', [])]
                }
                total_objects += data.get('metadata', {}).get('total_objects_detected', 0)
        except Exception as e:
            print(f'Error processing {result_file}: {e}')

summary = {
    'total_images_processed': len(all_results),
    'total_objects_detected': total_objects,
    'results_by_image': all_results
}

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to: {summary_file}')
print(f'Total images processed: {len(all_results)}')
print(f'Total objects detected: {total_objects}')
"

echo "Gemini Object Detection job completed!"
echo "End time: $(date)"
