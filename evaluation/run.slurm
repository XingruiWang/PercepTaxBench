#!/bin/bash
#SBATCH --job-name=vlm_eval
#SBATCH --output=log/job_%A_%a.out
#SBATCH --error=log/job_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --partition=main
#SBATCH --exclude=ccvl14
#SBATCH --nodelist=ccvl32
#SBATCH --mem=32G
#SBATCH --array=0

set -x
export GPU=$(nvidia-smi --list-gpus | wc -l)

module load conda

echo "Running on host: $(hostname)"

export LMUData='/path/to/Taxonomy/Data/questions/Data'

# Each array task gets a unique port
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID * 10))

MODELS=(
    "InternVL3_5-30B-A3B"
    # "InternVL3_5-8B"
    # "llava_onevision_qwen2_7b_ov"
    # "llava_next_vicuna_7b"
    # "llava_next_llama3"
)

BENCHMARKS=(
    # "TaxonomyBench"
    "TaxonomyBenchSim"
)

NUM_BENCHMARKS=${#BENCHMARKS[@]}
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_BENCHMARKS))
BENCH_IDX=$((SLURM_ARRAY_TASK_ID % NUM_BENCHMARKS))

MODEL=${MODELS[$MODEL_IDX]}
BENCHMARK=${BENCHMARKS[$BENCH_IDX]}

# Initialize conda shell functions (required for non-interactive shells)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment based on model
if [[ "$MODEL" == InternVL* ]]; then
    conda activate /path/to/.conda/envs/qwenomni3
else
    conda activate /path/to/.conda/envs/llava_next_flash_attn
fi


echo "Array task $SLURM_ARRAY_TASK_ID: model=$MODEL  benchmark=$BENCHMARK"
echo "Using port: $MASTER_PORT"

TORCHRUN=$(dirname $(which python))/torchrun
$TORCHRUN --nproc-per-node=$GPU --master-port=$MASTER_PORT run.py --data $BENCHMARK --model $MODEL --reuse