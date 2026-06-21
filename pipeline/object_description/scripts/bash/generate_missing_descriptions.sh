#!/bin/bash
#SBATCH --job-name=gen_missing_desc
#SBATCH --output=logs/generate_missing_descriptions_%j.log
#SBATCH --error=logs/generate_missing_descriptions_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "Starting missing object descriptions generation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate srdatagen

cd /path/to/SpatialReasonerDataGen

python object_description/scripts/python/generate_descriptions_for_missing.py

echo "Job completed at $(date)"
