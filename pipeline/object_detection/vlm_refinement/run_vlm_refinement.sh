#!/bin/bash
#SBATCH --job-name=vlm_refine
#SBATCH --output=logs/vlm_refinement_%j.out
#SBATCH --error=logs/vlm_refinement_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "========================================="
echo "VLM Object Name Refinement Job (BATCHED)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /path/to/SpatialReasonerDataGen/object_detection/vlm_refinement

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate srdatagen

API_KEY_1="YOUR_API_KEY_1"
API_KEY_2="YOUR_API_KEY_2"

echo "Processing all OpenImages unified outputs with RESUME support..."
echo "This will refine generic object names (army→soldier, person→man/woman/child, etc.)"
echo "Using 2 parallel workers with batched API calls"
echo ""

python refine_object_names_with_vlm_batched.py \
    --unified_dir /path/to/project/openimages_unified_output \
    --api_keys "$API_KEY_1,$API_KEY_2" \
    --output vlm_refinement_batched_results.json \
    --candidates_file detection_tag_analysis_refined_mappings.json \
    --parallel \
    --resume

echo ""
echo "Job completed at: $(date)"
echo "Results saved to: vlm_refinement_batched_results.json"
echo "Refined annotations saved in: <image_id>/annotations/<image_id>_refined.json"
echo "Progress files: vlm_refinement_batched_results_worker*_progress.json"

