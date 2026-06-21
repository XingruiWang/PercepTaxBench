# Image Generation and Processing Pipeline

Complete guide for processing simulation images and extracting scene structures.

## Overview

This pipeline processes rendered simulation images to extract structured scene information including:
- Object detection and classification
- 2D/3D bounding boxes
- Scene quality assessment
- Metadata extraction

## Directory Structure

```
/path/to/Taxonomy/Data/
├── simulationImage/                    # Input: Raw rendered images
│   └── {user}/                        # e.g., jiawei, luoxin, zehan
│       └── {scene}/                   # e.g., 1940Office, Bakery
│           └── {view_id}/             # e.g., l000_r001, l001_r002
│               ├── lit.png           # Rendered image
│               ├── seg.png           # Segmentation mask
│               ├── normal.png        # Normal map (optional)
│               ├── depth.npy         # Depth map (optional)
│               ├── seenable_obj_dict.json  # Object color mapping
│               └── object_annots.json     # Object annotations
│
└── SimulationMetadata/               # Output: Processed results
    ├── objects/                      # Object metadata
    │   ├── background_objects.json   # Background object definitions
    │   └── objects_list/             # Object classification mappings
    │       ├── classes_to_instances.json
    │       └── all_object_list.json
    │
    └── scenes/                       # Scene processing results  
        ├── annotations/              # Scene structure annotations
        │   └── {scene}/             # e.g., 1940Office
        │       └── {view_id}/       # e.g., l000_r001  
        │           ├── scene_annotations_split.json
        │           ├── bbox_visualization_all.png
        │           └── bbox_3d_visualization.png
        │
        ├── image_quality_ratings.json    # Quality assessment results
        └── image_quality_ratings_errors.json  # Error log
```

---

## 1. Seenable Object Dictionary Generation

### Purpose
Generate color mapping for segmentation masks to identify visible objects in rendered images.

### Location
```
/path/to/Taxonomy/Data/simulationImage/code/
```

### How to Run
```bash
cd /path/to/Taxonomy/Data/simulationImage/code/
python generate_seenable_object_dict.py [options]
```

### Input
- Segmentation masks (`seg.png`)
- Object annotation files (`object_annots.json`)

### Output
- `seenable_obj_dict.json` - Maps RGB colors to object IDs

### Output Format
```json
{
  "SM_Chair_01": [255, 0, 0],
  "SM_Table_02": [0, 255, 0],
  "SM_Floor_4x4m": [0, 0, 255]
}
```

---

## 2. Image Quality Filter

### Purpose
Evaluate image quality using AI models to filter high-quality scenes for dataset curation.

### Location
```
/path/to/Taxonomy/scripts/image_render/image_process_quality_filter/
```

### Prerequisites
```bash
cd /path/to/Taxonomy/scripts/image_render/image_process_quality_filter/
pip install -r requirements.txt
```

### How to Run

#### Quick Start
```bash
bash run.sh
```

#### Custom Parameters  
```bash
python evaluate_image_quality.py \
    --base_dir /path/to/simulationImage \
    --output /path/to/output.json \
    --num_workers 4 \
    --model gemini-2.5-flash-lite
```

#### Test Mode (10 images)
```bash
python evaluate_image_quality.py --max_images 10 --no-resume
```

### Configuration
- **Model**: `gemini-2.5-flash-lite`
- **API Keys**: 4 concurrent keys for parallel processing
- **Workers**: 4 parallel threads
- **Save Interval**: Every 30 images
- **Image Resize**: 1/4 original size for efficiency

### Evaluation Criteria (0-10 scale)
1. **Scene Richness**: Number of distinct objects/elements
2. **Camera Composition**: Framing and visual balance  
3. **Lighting & Exposure**: Natural lighting and proper exposure
4. **Rendering Realism**: Texture quality and photorealism

**Final Score**: Average of four criteria

### Output Structure

#### Main Results (`image_quality_ratings.json`)
```json
{
  "/path/to/simulationImage/user/scene/view_id/lit.png": {
    "SceneRichness": 8.5,
    "Composition": 7.8,
    "LightingExposure": 9.0,
    "RealismClarity": 8.2,
    "FinalScore": 8.4,
    "user_dir": "jiawei",
    "scene_name": "1940Office", 
    "view_id": "l000_r001",
    "full_path": "/path/to/lit.png",
    "timestamp": "2024-11-06T15:30:00",
    "attempt": 1
  }
}
```

#### Error Log (`image_quality_ratings_errors.json`)
```json
{
  "/path/to/failed_image.png": {
    "error": "ResourceExhausted: 429 Too Many Requests",
    "timestamp": "2024-11-06T15:30:00"
  }
}
```

### Performance
- **Processing Speed**: ~1000-2000 images/hour (4 workers)
- **Error Handling**: Automatic retry with different API keys
- **Resume Support**: Restart from last checkpoint
- **Success Rate**: ~99%+ with robust error handling

---

## 3. Scene Structure Refinement

### Purpose
Extract detailed scene structure including object detection, bounding boxes, and spatial relationships.

### Location
```
/path/to/Taxonomy/scripts/image_render/image_annotation/
```

### How to Run

#### Batch Processing (Recommended)
```bash
# Process all scenes with parallel workers
bash run_batch_annotation.sh

# Or with custom parameters
python gen_scene_structure.py \
    --batch \
    --visualize \
    --num_workers 8
```

#### Test Mode (Limited samples)
```bash
# Process 10 samples for testing
python gen_scene_structure.py \
    --batch \
    --visualize \
    --n_samples 10 \
    --num_workers 4
```

#### Single Scene Processing
```bash
python gen_scene_structure.py \
    /path/to/simulationImage/user/scene/view_id \
    --visualize
```

#### Quality-Based Processing
```bash
# Process only high-quality scenes (requires quality ratings)
python gen_scene_structure.py \
    --use_json \
    --from_json /path/to/image_quality_ratings.json \
    --visualize \
    --num_workers 4
```

### Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch` | - | Enable batch processing mode |
| `--visualize` | - | Generate visualization images |
| `--num_workers` | 4 | Number of parallel workers |
| `--n_samples` | None | Limit number of samples (testing) |
| `--skip_3d` | False | Skip 3D bbox computation |
| `--use_json` | False | Process from quality ratings JSON |
| `--output_dir` | `SimulationMetadata/scenes/annotations` | Output directory |

### Processing Features
- ✅ **Parallel Processing**: Multi-threaded batch processing
- ✅ **Foreground Detection**: Automatic filtering of scenes with objects
- ✅ **3D Bounding Boxes**: AABB and OBB computation with projection
- ✅ **Object Classification**: Background vs. foreground categorization
- ✅ **Visualization**: 2D and 3D bbox overlay images
- ✅ **Error Handling**: Comprehensive error logging and recovery
- ✅ **Resume Support**: Skip already processed scenes

### Output Structure

#### Scene Annotations (`scene_annotations_split.json`)
```json
{
  "scene_name": "1940Office",
  "view_id": "l000_r001", 
  "image_path": "/path/to/lit.png",
  "camera": {
    "location": [811, -633, 259],
    "rotation": [0, 60, 0],
    "fov": 90,
    "c2w": [[...]], 
    "fxfycxcy": [960, 960, 960, 540]
  },
  "background": {
    "floor": [
      {
        "object_id": "SM_Floor_4x4m46",
        "bbox_2d": [x1, y1, x2, y2],
        "bbox_3d": {
          "aabb": {"center": [x,y,z], "extent": [w,h,d]},
          "obb": {"center": [x,y,z], "extent": [w,h,d], "rotation": [r,p,y]}
        },
        "color": [r, g, b]
      }
    ],
    "wall": [...],
    "ceiling": [...]
  },
  "foreground": {
    "chair": [...],
    "table": [...],
    "lamp": [...]
  }
}
```

#### Visualization Files
- `bbox_visualization_all.png` - 2D bounding boxes overlay (960×540)
- `bbox_3d_visualization.png` - 3D bounding boxes projection (960×540)

#### Processing Report (`processing_errors.txt`)
```
Processing Report
============================================================

Skipped (No Foreground Objects): 1250
------------------------------------------------------------
ConferenceRoom/l000_r001
ConferenceRoom/l000_r002
...

Errors: 5
------------------------------------------------------------
BadScene/l001_r001: FileNotFoundError: seg.png not found
...
```

### Performance Metrics
- **Processing Speed**: ~2-5 seconds/scene (single), ~100-200 scenes/minute (parallel)
- **Success Rate**: ~95-98% (depends on data quality)
- **Memory Usage**: ~500MB-2GB (depending on scene complexity)
- **Disk Usage**: ~1-3MB per scene (annotations + visualizations)

---

## 4. Complete Pipeline Workflow

### Step 1: Prepare Environment
```bash
# Install dependencies
pip install -r /path/to/image_process_quality_filter/requirements.txt

# Set up API keys (for quality assessment)
export GOOGLE_API_KEY="your-api-key"
```

### Step 2: Generate Object Dictionaries (if needed)
```bash
cd /path/to/Taxonomy/Data/simulationImage/code/
python generate_seenable_object_dict.py
```

### Step 3: Quality Assessment (Optional but Recommended)
```bash
cd /path/to/Taxonomy/scripts/image_render/image_process_quality_filter/
bash run.sh
```

### Step 4: Scene Structure Extraction
```bash
cd /path/to/Taxonomy/scripts/image_render/image_annotation/

# Option A: Process all scenes
bash run_batch_annotation.sh

# Option B: Process high-quality scenes only
python gen_scene_structure.py \
    --use_json \
    --visualize \
    --num_workers 8
```

### Step 5: Verify Results
```bash
# Check processing statistics
ls -la /path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations/

# View error logs
cat /path/to/processing_errors.txt

# Check quality distribution  
python -c "
import json
with open('image_quality_ratings.json') as f:
    data = json.load(f)
scores = [v['FinalScore'] for v in data.values()]
print(f'Mean score: {sum(scores)/len(scores):.2f}')
print(f'High quality (>8.0): {sum(1 for s in scores if s > 8.0)}')
"
```

---

## 5. Troubleshooting

### Common Issues

#### Issue: No scenes processed / All skipped
**Cause**: Scenes lack foreground objects
**Solution**: 
```bash
# Check if scenes have objects
python -c "
from pathlib import Path
import sys
sys.path.append('/path/to/annotation/')
from gen_scene_structure import check_has_foreground
print(check_has_foreground('/path/to/scene/dir'))
"
```

#### Issue: API quota exceeded (Quality filter)
**Cause**: Rate limiting on Gemini API
**Solution**:
- Reduce `--num_workers`
- Add more API keys
- Increase delay between requests

#### Issue: Out of memory during processing
**Cause**: Large scenes with many objects  
**Solution**:
- Reduce `--num_workers` 
- Process scenes individually
- Use `--skip_3d` to reduce memory usage

#### Issue: Missing object mappings
**Cause**: Outdated object classification files
**Solution**:
```bash
# Update object mappings
ls -la /path/to/Taxonomy/Data/SimulationMetadata/objects/
```

### Performance Optimization

#### For Large Datasets (>10K scenes)
```bash
# Use maximum parallelization
python gen_scene_structure.py \
    --batch \
    --num_workers 16 \
    --output_dir /fast/ssd/path/

# Run on high-memory machine
# Recommended: 32GB+ RAM, 16+ CPU cores
```

#### For Network Storage
```bash
# Copy to local SSD first
rsync -av /network/simulationImage/ /local/ssd/
python gen_scene_structure.py /local/ssd/ --batch
```

---

## 6. Output Usage

### Loading Scene Annotations
```python
import json
from pathlib import Path

# Load single scene
with open('scene_annotations_split.json') as f:
    scene_data = json.load(f)

print(f"Scene: {scene_data['scene_name']}")
print(f"Objects: {len(scene_data['foreground'])} foreground categories")

# Load all quality ratings  
with open('image_quality_ratings.json') as f:
    quality_data = json.load(f)

high_quality = {k: v for k, v in quality_data.items() 
                if v['FinalScore'] > 8.0}
print(f"High quality scenes: {len(high_quality)}")
```

### Integration with Other Tools
- **Dataset Creation**: Use quality scores to filter scenes
- **3D Reconstruction**: Use camera parameters and 3D bboxes
- **Object Detection Training**: Use 2D annotations
- **Scene Understanding**: Use structured object relationships

---

## 7. Maintenance

### Regular Updates
- Update object classification mappings as new objects are added
- Retrain quality assessment models for domain-specific criteria
- Monitor processing logs for systematic errors

### Backup Strategy
- Essential: `image_quality_ratings.json`, `scene_annotations_split.json`
- Optional: Visualization images (can be regenerated)
- Archives: Processing error logs for debugging

### Monitoring
```bash
# Check processing status
find annotations/ -name "scene_annotations_split.json" | wc -l

# Monitor disk usage  
du -sh annotations/

# Check recent errors
tail -100 processing_errors.txt
```

---

*For technical support or questions, refer to the individual tool READMEs in their respective directories.*
