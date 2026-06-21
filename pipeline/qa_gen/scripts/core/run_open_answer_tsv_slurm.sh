#!/bin/bash
#SBATCH --job-name=open_answer_tsv
#SBATCH --output=logs/open_answer_tsv_%j.out
#SBATCH --error=logs/open_answer_tsv_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00


source ~/.bashrc
conda activate srdatagen

cd /path/to/SpatialReasonerDataGen/qa_gen/scripts/core


mkdir -p logs
OUT_BASE=/path/to/VLMEvalKit/Data

echo "=============================================="
echo "Converting Open-Answer Benchmarks to TSV..."
echo "=============================================="


echo "Real image open_answer..."
python aggregate_unified_qa.py \
    --input_dir ../../taxonomyQABench_realimage_final_polished_open_answer \
    --output_file ${OUT_BASE}/taxonomy_real_open_answer.tsv

echo ""

echo "Sim image open_answer..."
python aggregate_unified_qa.py \
    --input_dir ../../taxonomyQABench_simimage_final_open_answer \
    --output_file ${OUT_BASE}/taxonomy_sim_open_answer.tsv

echo ""
echo "=============================================="
echo "Open-answer TSV conversion completed!"
echo "=============================================="


echo ""
echo "Summary:"
ls -lh ${OUT_BASE}/taxonomy_real_open_answer.tsv ${OUT_BASE}/taxonomy_sim_open_answer.tsv 2>/dev/null || echo "Check output above"
