#!/bin/bash

cd /path/to/SpatialReasonerDataGen/object_detection/vlm_refinement

API_KEY_1=""
API_KEY_2=""

sed "s/YOUR_API_KEY_1/$API_KEY_1/g; s/YOUR_API_KEY_2/$API_KEY_2/g" run_vlm_refinement.sh > run_vlm_refinement_temp.sh

sbatch run_vlm_refinement_temp.sh

rm run_vlm_refinement_temp.sh

echo "Job submitted! Check status with: squeue -u \$USER"
echo "Monitor progress with: ./monitor_batched_progress.sh"
echo "View logs with: tail -f logs/vlm_refinement_*.out"

