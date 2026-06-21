#!/bin/bash
# Batch process all scenes in simulationImage to generate annotations

BASE_DIR="/path/to/Taxonomy/Data/simulationImage"
OUTPUT_DIR="/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations"

echo "=========================================="
echo "Batch Scene Annotation Generation"
echo "=========================================="
echo "Input directory:  $BASE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run batch processing
python gen_scene_structure.py \
    "$BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch \
    --visualize \
    --num_workers 32 \
    "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Batch processing completed!"
    echo "Annotations saved to: $OUTPUT_DIR"
else
    echo "✗ Batch processing failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE

