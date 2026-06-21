#!/bin/bash
# Test processing 10 samples with 3D bbox generation

BASE_DIR="/path/to/Taxonomy/Data/simulationImage"
OUTPUT_DIR="/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations"

echo "=========================================="
echo "Test: Process 10 Samples (with 3D bbox)"
echo "=========================================="
echo "Input directory:  $BASE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find 10 scenes directly
cd "$BASE_DIR"
echo "Finding 10 sample scenes..."

# Use Python to process directly with limit
python - <<EOF
import sys
sys.path.insert(0, '/path/to/Taxonomy/scripts/image_render/image_annotation')
from pathlib import Path
import subprocess

base_dir = Path('$BASE_DIR')
output_dir = Path('$OUTPUT_DIR')

# Find scenes with lit.png
scenes = []
for lit_file in base_dir.rglob('lit.png'):
    scene_dir = lit_file.parent
    # Check if has required files
    if (scene_dir / 'seg.png').exists() and (scene_dir / 'seenable_obj_dict.json').exists():
        scenes.append(scene_dir)
        if len(scenes) >= 10:
            break

print(f"\nFound {len(scenes)} scenes to process\n")

# Process each scene
for i, scene_dir in enumerate(scenes, 1):
    print(f"[{i}/10] Processing: {scene_dir.relative_to(base_dir)}")
    
    cmd = [
        'python',
        '/path/to/Taxonomy/scripts/image_render/image_annotation/gen_scene_structure.py',
        str(scene_dir),
        '--output_dir', str(output_dir),
        '--visualize'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Success")
    else:
        print(f"  ✗ Failed: {result.stderr[:100]}")
    print()

print("="*60)
print("✓ Test completed!")
print(f"Check output at: {output_dir}")
print("="*60)
EOF
