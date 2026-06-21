#!/bin/bash
#SBATCH --job-name=vlm_refine_all
#SBATCH --output=logs/vlm_refinement_full_%j.log
#SBATCH --error=logs/vlm_refinement_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=main

echo "=================================================="
echo "Comprehensive VLM Refinement - All Images"
echo "=================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

cd /path/to/SpatialReasonerDataGen

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate srdatagen

echo ""
echo "Running comprehensive VLM refinement on all images..."
echo ""

python -u object_detection/vlm_refinement/comprehensive_vlm_refinement_batched.py \
  --unified-output-dir ../../openimages_unified_output \
  --api-keys "" \
  --object-list object_description/results/full_object_description/merged_objects_list.txt \
  --checkpoint-file object_detection/vlm_refinement/refinement_checkpoint.json

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE

