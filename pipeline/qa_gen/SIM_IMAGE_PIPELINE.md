# Sim Image Benchmark Generation Pipeline

This document describes the complete pipeline for generating the Taxonomy QA Benchmark from simulated images, including image processing, object extraction, filtering, grouping, and visualization.

## Overview

The sim image benchmark uses 3D synthetic scenes from the 3DWorld simulation environment. Each scene contains multiple objects with 3D annotations, depth information, and visibility data.

---

## 1. Data Sources

### 1.1 Image Location
- **Path**: `/path/to/sim_images/jiawei`
- **Format**: Directory structure containing scene folders
- **Naming**: `{scene_id}/` (e.g., `l000_r000/`)

### 1.2 Scene Structure
Each scene folder contains:
```
{scene_id}/
├── lit.png                      # Original RGB image (lighted/scene image)
├── seg.png                      # Segmentation masks
├── depth.npy                     # Depth maps (if available)
├── camera_annots.json            # Camera parameters (FOV, pose)
├── object_annots.json             # Object annotations and poses
└── seenable_obj_dict.json       # Visibility mapping (SM names to colors)
```

### 1.3 Key Annotation Files

#### `object_annots.json`
Contains per-object metadata:
```json
{
  "obj_000001": {
    "class_name": "Apple",
    "location": [x, y, z],
    "rotation": [rx, ry, rz],
    "scale": [sx, sy, sz],
    "bbox_2d": [x_min, y_min, x_max, y_max]
  }
}
```

#### `seenable_obj_dict.json`
Maps SM (simulation) object names to RGB colors:
```json
{
  "Apple_000001": [255, 0, 0],
  "Cup_000002": [0, 255, 0]
}
```
This file is critical - only objects in this dict are visible in the segmentation.

---

## 2. Object Extraction and Filtering

### 2.1 Initial Object Extraction

**Process** (`extract_objects_from_sim_scene`):
1. Load `object_annots.json` to get all objects with poses
2. Load SM-to-human mapping to translate names
3. Load `seenable_obj_dict.json` to get color mapping

**Output**:
- `objects`: List of unique taxonomy object names
- `object_poses`: Dict mapping object names to pose data
- `taxonomy_to_sm_names`: Dict mapping taxonomy names to SM names

### 2.2 Filtering Process

Objects pass through **multiple filtering stages** before being used for questions:

#### Stage 1: Visibility Filter
- **Condition**: Object must be present in `seenable_obj_dict.json`
- **Purpose**: Only include objects actually visible in the scene
- **Effect**: Removes background elements, distant objects, occluded objects

#### Stage 2: Scene Object Category Filter
- **Purpose**: Filter objects to only include those relevant to the specific scene
- **Implementation**: Uses `scene_object_categories` mapping loaded from JSON files
- **Files**: Located in `modules/sim_scene_object/data/scene_with_object/*_object_category.json`
- **Process**: 
  - Scene-specific objects are loaded (e.g., "CompleteKitchen" has kitchen objects)
  - Only objects listed in the scene's category file are kept
  - This ensures objects belong to the scene they appear in
- **Result**: Objects that don't belong to a scene category are filtered out

#### Stage 3: Name-Based Filter
Exclude objects with these names:
- Background elements: `'walls'`, `'floor'`, `'ceiling'`, `'window'`, `'door'`
- Generic surfaces: `'table'`, `'shelf'`
- Unwanted objects: `'empty'`, `'unknown'`, `'boundary'`

#### Stage 4: Segmentation Size Filter
- **Condition**: Object must appear in segmentation with visible pixels
- **Purpose**: Remove tiny or fully occluded objects
- **Implementation**: Check if object color is present in `seg.png` and has reasonable bbox size

#### Stage 5: Depth-Based Filter
- **Condition**: Objects must be within reasonable distance from camera
- **Purpose**: Remove objects too far away to be clearly visible
- **Implementation**: Check depth values in depth map

### 2.3 Final Object List

After all filtering stages, we obtain:
- `objects_with_bbox`: Objects that are visible, close enough, and have valid segmentation

This list is used for question generation and visualization.

---

## 3. Object Grouping Strategy

### 3.1 Why Grouping?

To manage scenes with many objects:
- Bbox visualization has limited space (max ~6 objects)
- Questions need focused contexts
- Prevents overwhelming single images

### 3.2 Grouping Logic

**Goal**: Create groups of 3-6 objects per visualization with **depth-aware clustering**

#### Depth-Aware Grouping

**Why depth matters**: Objects at very different depths should NOT be in the same bbox image:
- **Visual clarity**: Far objects appear tiny, near objects appear large
- **Spatial coherence**: Questions about objects at different depths are confusing
- **Quality**: Mixed-depth bboxes are visually poor

**Grouping Process**:
1. **Sort by depth**: Objects ordered by Y-axis (Y-axis = front/back in 3D space)
2. **Calculate dynamic threshold**: Depth threshold adapts to scene's depth range
   - Large scenes (depth range >20 units): threshold = 10.0 units
   - Medium scenes (10-20 units): threshold = 5.0 units
   - Small scenes (<10 units): threshold = 3.0 units
3. **Cluster by depth**: Objects within threshold distance grouped together
4. **Split large clusters**: If a depth cluster has >6 objects, split evenly
5. **Validation**: Discard groups with <3 objects

#### Case 1: ≤ 6 Objects (All Similar Depth)
```python
if len(objects_with_bbox) <= 6:
    # Keep all objects together in one group
    object_groups = [objects_with_bbox]
```

**Rationale**: Small scenes can display all objects in one image

#### Case 2: > 6 Objects (Mixed Depths)
```python
# Sort by depth first, then split
# Example: [close_objs: 6, far_objs: 5] → [6 close], [5 far]
# Example: [all similar depth: 9] → [5, 4] (even split)
```

**Distribution Strategy**:
- **First**: Group objects by similar depth (within 2 unit threshold)
- **Then**: Within each depth group, split evenly if >6 objects
- **Result**: Objects at similar depths stay together
- **Discard**: Groups <3 objects (not enough for meaningful questions)

**Examples**:
- 12 objects: 6 close (Y<0), 6 far (Y>2) → Groups: `[6 close]`, `[6 far]`
- 9 objects all near camera → Groups: `[5 near objects]`, `[4 near objects]`
- 13 objects: 4 near, 9 mid → Groups: `[4 near]`, `[6 mid]`, `[3 mid]`

#### Case 3: Validation
Groups with fewer than 3 objects are **discarded**:
```python
if len(group) < 3:
    # Delete bbox image directory
    # Skip question generation
```

**Rationale**: At least 3 objects needed for multiple choice questions

### 3.3 Group Naming

Each group generates:
- **Image**: `{scene_id}_group{idx}/bbox.jpg`
- **Questions**: Tagged with `{scene_id}_group{idx}`

Example:
```
l000_r000_group0/
└── bbox.jpg         # Shows objects with colored bboxes + numbers
l000_r000_group1/
└── bbox.jpg         # Another subset of objects
```

---

## 4. Bounding Box Visualization

### 4.1 Color Matching

**Challenge**: Match taxonomy object names to segmentation colors

**Solution**: Use `taxonomy_to_sm_names` mapping
```python
# Get SM names for taxonomy object
sm_names = taxonomy_to_sm_names.get('Apple', [])

# Look up color in seenable_obj_dict
for sm_name in sm_names:
    color = seenable_obj_dict.get(sm_name)
    if color in segmentation:
        # Found bbox for this object
        break
```

This replaces unreliable fuzzy matching with explicit mapping.

### 4.2 Visualization Process

**Steps**:
1. Load RGB image from `lit.png`
2. Load segmentation mask from `seg.png`
3. Extract bboxes for objects in the group:
   - Look up SM names → colors → segmentation
   - Compute bounding boxes from color regions
4. Draw colored bboxes with thick lines (width=4)
5. Add numerical labels (font size=40) for reference
6. Resize to target dimensions (max 400x225, maintains aspect ratio)
7. Save as `{scene_id}_group{idx}/bbox.jpg`

**Color Coding**:
- Each object gets its native segmentation color
- Numbers (1-6) help identify objects in questions

### 4.3 Visualization Quality

- **Bbox Line Width**: 4 pixels
- **Label Font Size**: 80 points 
- **Image Resize**: Final size 400x225 (maintains aspect ratio from original ~1920x1080)
- **Resize Timing**: After drawing bboxes 

---

## 5. Question Generation

### 5.1 Object Validation

Before generating questions:
1. Verify objects in group actually have bboxes
2. Return list of `objects_with_bboxes_in_group`
3. Update question choices to match actual bboxes

**Critical**: Questions must only reference objects visible in the bbox image.

### 5.2 Question Post-Processing

After question generation:
```python
# Update each question to match actual bbox objects
for question in group_questions:
    question['objects'] = list(objects_with_bboxes_in_group)
    question['choices'] = list(objects_with_bboxes_in_group)
    question['image_path'] = f"{scene_id}_group{idx}/bbox.jpg"
```

This ensures questions always reference objects shown in the image.

---

## 6. Pipeline Summary

### Inputs
- Scene directories with RGB, segmentation, depth, annotations
- `seenable_obj_dict.json` (visibility mapping)
- `object_annots.json` (object poses and metadata)
- SM-to-taxonomy name mappings

### Processing Steps
1. **Extract** objects from scene annotations
2. **Filter** objects by visibility, category, name, size, depth
3. **Group** objects into sets of 3-6 (discard groups < 3)
4. **Visualize** bboxes for each group
5. **Generate** questions for objects in each group
6. **Validate** questions match bbox objects

### Outputs
- `taxonomyQABench_simimage/`
  - `all_questions.json` - All questions
  - `scene_statistics.json` - Scene stats
  - `generation_metadata.json` - Metadata
  - `images/{scene_id}_group{idx}/bbox.jpg` - Visualizations
- `taxonomy_sim.tsv` - TSV for evaluation

---

## 7. Key Improvements

### 7.1 Early Filtering
- Objects filtered by `seenable_obj_dict.json` **before** grouping
- Reduces wasted processing on invisible objects

### 7.2 Explicit Name Mapping
- Use `taxonomy_to_sm_names` instead of fuzzy matching
- Ensures correct color lookup in segmentation

### 7.3 Dynamic Object Lists
- Track which objects actually got bboxes
- Update questions to match bbox count

### 7.4 Smart Grouping
- Prioritize keeping small sets together (≤6 objects in one group)
- Even distribution for larger sets
- Discard groups too small for questions

### 7.5 Group Validation
- Check group size **before** generating questions
- Delete orphaned bbox images if group invalid

---

## 8. Quality Assurance

### Validation Checks
1. Bbox image exists for every group
2. Objects in questions match objects in image
3. Choices match actual bbox count
4. No references to invisible objects
5. Images are properly resized
6. Bboxes and labels are visible

### Error Handling
- Groups with < 3 objects: Discard image, skip questions
- Objects without bboxes: Exclude from group
- Missing segmentation: Skip object
- Invalid colors: Log warning, skip

---

## 9. Example Flow

**Scene**: `l000_r000`
- **Total Objects**: 12
- **After Filtering**: 8 visible objects
- **After Grouping**: 
  - `l000_r000_group0`: 4 objects
  - `l000_r000_group1`: 4 objects

**Generated**:
- `l000_r000_group0/bbox.jpg` - Shows 4 objects with bboxes
- `l000_r000_group1/bbox.jpg` - Shows 4 objects with bboxes
- Questions tagged with `l000_r000_group0` reference only the 4 objects
- Questions tagged with `l000_r000_group1` reference only the 4 objects

**Result**: Each bbox image matches the questions, no mismatches, clear visualization.

---

## 10. Comparison: Real vs Sim Images

### Real Images
- Use pre-annotated object detection
- Simpler filtering (category, background objects)
- No grouping needed (each image is one scene)
- Direct bbox extraction from annotations

### Sim Images
- Extract from 3D scene data
- Complex filtering (visibility, depth, segmentation)
- Grouping required for large scenes
- Color-based bbox extraction from segmentation

Both produce the same output format: questions with bbox visualizations.

