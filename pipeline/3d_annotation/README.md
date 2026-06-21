# 3D Annotation Pipeline

This folder contains all scripts and modules related to 3D annotation generation for the SpatialReasonerDataGen pipeline.

## Files

### Main Scripts
- `generate_3d_groundtruth_production.py` - Main 3D ground truth generation pipeline
- `generate_3d_groundtruth_production_rolling.py` - Version with rolling statistics and checkpointing
- `generate_3d_groundtruth_production_ultra_fast.py` - Ultra-fast optimized version

### Visualization
- `visualize_3d_data.py` - 3D data visualization module

### Core Modules
- `config.py` - Configuration settings for all models
- `config_ultra_fast.py` - Ultra-fast configuration
- `reconstruct3d.py` - 3D reconstruction module
- `pose3d.py` - 3D pose estimation module
- `pose3d_orientanything.py` - OrientAnything-based pose estimation
- `pose_utils.py` - Pose utility functions
- `pose_validator.py` - Pose validation utilities
- `orientanything_utils.py` - OrientAnything utility functions

## Usage

### Basic 3D Annotation Generation
```bash
python generate_3d_groundtruth_production.py \
    --image_path /path/to/images \
    --output_path /path/to/output \
    --enable_pose3d \
    --generate_annotations
```

### Ultra-Fast Processing
```bash
python generate_3d_groundtruth_production_ultra_fast.py \
    --image_path /path/to/images \
    --output_path /path/to/output \
    --batch_size 128 \
    --max_workers 16 \
    --skip_visualizations \
    --skip_object_crops
```

### Rolling Statistics Version
```bash
python generate_3d_groundtruth_production_rolling.py \
    --image_path /path/to/images \
    --output_path /path/to/output \
    --stats_path /path/to/stats \
    --checkpoint_interval 100 \
    --resume_from_checkpoint
```

## Model Paths

All model paths in `config.py` are now relative to the parent directory:
- RAM model: `../models/ram/ram_plus_swin_large_14m.pth`
- GroundingDINO: `../models/GroundingDINO/groundingdino_swint_ogc.pth`
- SAM: `../models/sam/sam2.1_hq_hiera_large.pt`
- 3D Reconstruction: `../models/reconstruct3d/paramnet_360cities_edina_rpf.pth`
- Pose Estimation: `../models/pose/model_100.pth`
- OrientAnything: `../models/orientanything/dino_weight.pt`

## Dependencies

The scripts import from:
- `srdatagen.modules.TagAndSegment` - Object detection and segmentation
- `srdatagen.utils` - Utility functions
- `srdatagen.dnnlib` - Configuration utilities

## Output Structure

Each processed image creates a directory with:
- `annotations/` - JSON annotations and summaries
- `visualizations/` - 3D visualization images (if enabled)
- `object_crops/` - Cropped object images (if enabled)
- `pcd/` - Point cloud data (if enabled)

## Notes

- Object cropping functionality is currently commented out and needs implementation
- All paths have been updated to work from the new folder structure
- The pipeline supports parallel processing and checkpointing for large datasets
