#!/bin/bash
# Test script - only evaluates first 10 images

BASE_DIR="/path/to/Taxonomy/Data/simulationImage"
OUTPUT="/path/to/Taxonomy/Data/SimulationMetadata/scenes/test_quality_ratings.json"

echo "=========================================="
echo "TEST MODE: Evaluating first 10 images"
echo "=========================================="

python evaluate_image_quality.py \
    --base_dir "$BASE_DIR" \
    --output "$OUTPUT" \
    --max_images 10 \
    --num_workers 2 \
    --no-resume \
    "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully!"
    echo ""
    echo "Results:"
    cat "$OUTPUT" | python -m json.tool | head -50
else
    echo "✗ Test failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE

