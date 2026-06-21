#!/bin/bash
# Quick start script for image quality evaluation with parallel processing

# Default parameters
BASE_DIR="/path/to/Taxonomy/Data/simulationImage"
OUTPUT="/path/to/Taxonomy/Data/SimulationMetadata/scenes/image_quality_ratings.json"
MODEL="gemini-2.5-flash-lite"
NUM_WORKERS=4
SAVE_INTERVAL=30

echo "=========================================="
echo "Image Quality Evaluation"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo "Output file: $OUTPUT"
echo "Model: $MODEL"
echo "Workers: $NUM_WORKERS"
echo "=========================================="
echo ""

# Run evaluation with default API keys
python evaluate_image_quality.py \
    --base_dir "$BASE_DIR" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --num_workers "$NUM_WORKERS" \
    --save_interval "$SAVE_INTERVAL" \
    --resume \
    "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Done! Results saved to: $OUTPUT"
else
    echo "✗ Error occurred during evaluation (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE

