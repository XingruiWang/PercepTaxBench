# Object Detection Module

This module extracts 2D object detection results from the unified OpenImages pipeline and provides tools for visualizing and working with 2D bounding boxes.

## Directory Structure

```
object_detection/
├── README.md                          # This file
├── extract_2d_detections.py          # Extract 2D detection data from unified outputs
├── generate_2d_bbox_images.py        # Generate 2D bounding box visualizations
├── run_object_detection.py           # Run object detection on new images
├── run_extract_detections.sh         # Batch extraction script (SLURM)
└── results/
    └── object_detection_extracted/   # Extracted 2D detection results
        ├── <image_id>/
        │   ├── annotations/
        │   │   └── <image_id>_2d_detections.json
        │   ├── object_crops/
        │   │   └── obj_XXX_<class>_confX.XX.png
        │   └── visualizations/
        │       └── <image_id>_2d_bbox.png
        ├── detection_records.json     # All detection records
        └── detection_summary.json     # Summary statistics
```

## Files Included in Object Detection Results

Each processed image directory contains:

1. **Annotations** (`annotations/<image_id>_2d_detections.json`):
   - Image metadata (path, dimensions)
   - Tags/labels
   - 2D detection data for each object:
     - Object name and class
     - 2D bounding box (xyxy format)
     - Confidence score
     - Box area

2. **Object Crops** (`object_crops/obj_XXX_<class>_confX.XX.png`):
   - Cropped images of each detected object
   - Named with object index, class name, and confidence

3. **Visualizations** (`visualizations/<image_id>_2d_bbox.png`):
   - Original image with 2D bounding boxes drawn
   - Each box labeled with class name and confidence

## Files Excluded (3D Annotation Data)

The following 3D annotation data is **NOT** included in this module:
- Point cloud data (`pcd`, `pcd_cano`)
- 3D bounding boxes (`pcd_axis_bbox`, `pcd_orient_bbox`, etc.)
- 3D visualization images
- Mask data

## Usage

### 1. Extract 2D Detection Data from Unified Outputs

Extract 2D detection data and object crops from the existing unified pipeline outputs:

```bash
python3 extract_2d_detections.py \
    --source_dir ../../openimages_unified_output \
    --target_dir ./results/object_detection_extracted \
    --limit 100  # Optional: process first 100 images
```

### 2. Generate 2D Bounding Box Visualizations

Create visualization images with 2D bounding boxes drawn:

```bash
python3 generate_2d_bbox_images.py \
    --detection_dir ./results/object_detection_extracted \
    --image_source_dir ../../openimages_train_10000 \
    --output_dir ./results/object_detection_extracted
```

### 3. Run Complete Extraction Pipeline (SLURM)

Extract and visualize all detection data:

```bash
chmod +x run_extract_detections.sh
sbatch run_extract_detections.sh
```

### 4. Run Object Detection on New Images

Process a single new image to generate the same output format:

```bash
python3 run_object_detection.py \
    --image_path /path/to/image.jpg \
    --output_dir ./results/object_detection_output \
    --image_id my_image_001
```

## Output Format

### Detection JSON Format

```json
{
  "image_info": {
    "file_path": "/path/to/image.jpg",
    "width": 1024,
    "height": 768
  },
  "tags": ["tag1", "tag2", "tag3"],
  "detections": [
    {
      "object_name": "obj_00_car",
      "class_name": "car",
      "bbox_xyxy": "[100.0 200.0 300.0 400.0]",
      "confidence": 0.95,
      "class_id": 3,
      "box_area": 40000.0,
      "area": 15000
    }
  ]
}
```

### Detection Records Format

The `detection_records.json` file contains a list of all processed images:

```json
[
  {
    "image_id": "000069a0b17c906e",
    "image_path": "/path/to/image.jpg",
    "image_size": {"width": 1024, "height": 673},
    "tags": ["bell tower", "church", "road"],
    "num_detections": 5,
    "detections": [...],
    "bbox_visualization": "./results/.../visualizations/...png"
  }
]
```

### Summary Format

The `detection_summary.json` file contains overall statistics:

```json
{
  "total_images": 9082,
  "successfully_processed": 9000,
  "failed": 82,
  "total_detections": 45000,
  "avg_detections_per_image": 5.0
}
```

## Dependencies

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- Standard library modules (json, os, pathlib, argparse)

## Notes

- The extraction process preserves only 2D object detection information
- All 3D annotation data (point clouds, 3D bounding boxes) is excluded
- Cropped objects and visualizations use the same naming convention as the unified pipeline
- The module is designed to work alongside the `clustering` and `object_description` modules
