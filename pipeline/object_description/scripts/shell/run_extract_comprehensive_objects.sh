#!/bin/bash
#SBATCH --job-name=extract_objects_comprehensive
#SBATCH --output=logs/extract_objects_comprehensive_%j.out
#SBATCH --error=logs/extract_objects_comprehensive_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=main
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo "Starting comprehensive OpenImages object extraction at $(date)"

# Set working directory
cd /path/to/SpatialReasonerDataGen

# Run the comprehensive object extraction
python openimages_3d_annotations/scripts/extract_comprehensive_objects.py

echo "Comprehensive object extraction completed at $(date)"
echo "Check output files in: openimages_3d_annotations/data/"
echo "  - openimages_detected_objects.txt (simple list)"
echo "  - openimages_detected_objects_database.json (comprehensive database)"
echo "  - openimages_object_stats.json (statistics)"
