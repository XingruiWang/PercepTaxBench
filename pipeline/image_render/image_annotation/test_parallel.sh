#!/bin/bash
# Test parallel batch processing with a subset of scenes

BASE_DIR="/path/to/Taxonomy/Data/simulationImage"
OUTPUT_DIR="/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations"

echo "=========================================="
echo "Testing Parallel Batch Processing"
echo "=========================================="
echo ""

# Create a temporary test directory with 10 scenes
TEST_DIR="/tmp/test_parallel_$$"
mkdir -p "$TEST_DIR"

echo "Creating test directory with 10 scenes..."
echo ""

# Find 10 scenes and copy their structure
count=0
for scene_path in $(find "$BASE_DIR" -name "seg.png" -type f | head -20); do
    scene_dir=$(dirname "$scene_path")
    
    # Check if has required files
    if [ ! -f "$scene_dir/seenable_obj_dict.json" ]; then
        continue
    fi
    
    # Extract path components
    # Path: .../simulationImage/user/scene/view_id/seg.png
    view_id=$(basename "$scene_dir")
    scene=$(basename $(dirname "$scene_dir"))
    user=$(basename $(dirname $(dirname "$scene_dir")))
    
    # Create directory structure in test dir
    target_dir="$TEST_DIR/$user/$scene/$view_id"
    mkdir -p "$target_dir"
    
    # Copy files (or create symlinks)
    ln -sf "$scene_dir"/* "$target_dir/" 2>/dev/null
    
    count=$((count + 1))
    echo "  [$count] $scene/$view_id"
    
    if [ $count -ge 10 ]; then
        break
    fi
done

echo ""
echo "Test directory ready: $TEST_DIR"
echo "Found $count scenes"
echo ""

if [ $count -eq 0 ]; then
    echo "Error: No valid scenes found!"
    rm -rf "$TEST_DIR"
    exit 1
fi

echo "=========================================="
echo "Running: python gen_scene_structure.py"
echo "Mode: Batch (Parallel)"
echo "Workers: 4"
echo "Visualize: Yes"
echo "=========================================="
echo ""

# Run the batch processing
python gen_scene_structure.py \
    "$TEST_DIR" \
    --batch \
    --visualize \
    --num_workers 4 \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "Check results in: $OUTPUT_DIR"
echo ""

# Cleanup
echo "Cleaning up test directory..."
rm -rf "$TEST_DIR"

exit $EXIT_CODE

