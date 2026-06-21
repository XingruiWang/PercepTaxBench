#!/bin/bash
#SBATCH --job-name=fix_phys_props
#SBATCH --output=/path/to/SpatialReasonerDataGen/object_description/logs/fix_phys_props_%j.log
#SBATCH --error=/path/to/SpatialReasonerDataGen/object_description/logs/fix_phys_props_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=main
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --exclude=ccvl35

echo "Starting physical properties fix at $(date)"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate srdatagen

cd /path/to/SpatialReasonerDataGen/object_description/results/full_object_description

python3 -u fix_physical_properties.py

echo "Completed at $(date)"
