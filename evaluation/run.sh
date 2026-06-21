export LMUData=/path/to/Taxonomy/Data/questions/Data
# python run.py --data TaxonomyBench --model GeminiPro2-5 --reuse
set -x
export GPU=$(nvidia-smi --list-gpus | wc -l)


# MODELS=(
#     "GeminiPro2-5"
#     "InternVL3_5-30B-A3B",
    # "InternVL3_5-8B"

    # "llava_onevision_qwen2_7b_ov",
#     "llava_next_vicuna_7b",

#     "llava_next_llama3",
#     "llava_next_qwen_32b",
#     "llava_next_interleave_7b_dpo",

#     "Qwen3-VL-8B-Thinking",
    
# )

# cd ~/scratch/2025/Taxonomy/evaluation/VLMEvalKit
export MASTER_PORT=$((29500 + RANDOM % 1000))

MODELS=(
    # "GeminiPro2-5"
    # "InternVL3_5-30B-A3B"
    # "InternVL3_5-8B"
    "llava_next_vicuna_7b"
    # "llava_next_llama3"
    
    # "llava_next_interleave_7b_dpo"
    # "llava_onevision_qwen2_7b_ov"
    # "llava_next_qwen_32b"

)
# for model in ${MODELS[@]}; do
#     torchrun --nproc-per-node=$GPU --master-port=$MASTER_PORT  run.py --data TaxonomyBenchSim --model $model --reuse 
# done

# wait
python run.py --data TaxonomyBenchSim --model llava_next_vicuna_7b --reuse
# python run.py --data TaxonomyBench --model InternVL3_5-30B-A3B --reuse