# Hybrid Spatial Question Plan: At Least One High-Reliability Object

## Overview

**New Strategy**: Use geometric orientation from high-reliability object as reference for **left/right** and **front/behind** questions. Check alignment/overlap with other object, then determine spatial relationship based on geometric orientation.

**Key Principle**:
- At least **ONE** object must be high-reliability (not both)
- Use the high-reliability object's orientation (left vector for left/right, front vector for front/behind) as reference
- Check if objects are aligned or have significant bbox overlap (for left/right)
- Check if objects are at similar depth (for front/behind)
- If not aligned/overlapping, ask spatial questions based on geometric orientation

**Applies to**:
- `spatial_left_right`: Use reference object's left vector
- `spatial_front_behind`: Use reference object's front vector

## Expected Coverage

- **Questions that would pass**: 124 (62.0% of current 200 questions)
- **Questions that would be skipped**: 76 (38.0% - only when both are low-reliability)
- **Breakdown**:
  - Both high-reliability: 21 (10.5%) ✓
  - One high, one low: 103 (51.5%) ✓ ← **NEW: These now pass!**
  - Both low-reliability: 76 (38.0%) ✗ (still skipped)

## Logic Flow

### Step 1: Check if at least one object is high-reliability
```
if (is_high_reliability(object1) OR is_high_reliability(object2)):
    proceed
else:
    skip question (both low-reliability)
```

### Step 2: Identify reference object (high-reliability one)
```
if is_high_reliability(object1) and has_pose_data(object1):
    reference_object = object1
    target_object = object2
elif is_high_reliability(object2) and has_pose_data(object2):
    reference_object = object2
    target_object = object1
else:
    skip question (high-reliability object missing pose data)
```

### Step 3: Check horizontal alignment/overlap
```
# Use 2D bbox to check if objects are horizontally aligned
# If objects overlap significantly in X and Y, they're aligned → skip question
if significant_bbox_overlap(object1_bbox, object2_bbox):
    return "unknown"  # Don't ask left/right for aligned objects
```

### Step 4a: Calculate left/right using geometric orientation
```
# Get reference object's left vector (normalized 3D direction)
left_vec = reference_object.get('left')

# Get 3D centers of both objects
reference_center = reference_object.get('pcd_center')
target_center = target_object.get('pcd_center')

# Calculate relative position
relative_pos = target_center - reference_center

# Project onto reference object's left vector
dot_product = relative_pos · left_vec

# Determine left/right
if dot_product > threshold:
    return "left"  # target is to the left of reference (geometrically)
elif dot_product < -threshold:
    return "right"  # target is to the right of reference (geometrically)
else:
    return "unknown"  # Too close to determine
```

### Step 4b: Calculate front/behind using geometric orientation
```
# Get reference object's front vector (normalized 3D direction)
front_vec = reference_object.get('front')

# Get 3D centers of both objects
reference_center = reference_object.get('pcd_center')
target_center = target_object.get('pcd_center')

# Calculate relative position
relative_pos = target_center - reference_center

# Project onto reference object's front vector
dot_product = relative_pos · front_vec

# Determine front/behind
if dot_product > threshold:
    return "front"  # target is in front of reference (geometrically)
elif dot_product < -threshold:
    return "behind"  # target is behind reference (geometrically)
else:
    return "unknown"  # Too close to determine (similar depth)
```

## Files That Need Changes

### 1. `object_utils.py`

#### Changes in `_calculate_from_annotations()`:
- **Current**: Uses 2D bbox centers for left/right (viewer perspective)
- **New**: Hybrid approach:
  1. Check if at least one object is high-reliability
  2. If yes, use geometric orientation from high-reliability object
  3. If no, return "unknown" (skip question)
  4. Still check for bbox overlap/alignment before using geometric calculation

**New method needed**: `_calculate_left_right_hybrid()`
```python
def _calculate_left_right_hybrid(self, object1: str, object2: str, 
                                 obj1_data: Dict, obj2_data: Dict) -> str:
    """
    Calculate left/right using hybrid approach:
    - At least one object must be high-reliability
    - Use high-reliability object's orientation as reference
    - Check for horizontal alignment/overlap
    - Return "left", "right", or "unknown"
    """
```

**Changes in `_calculate_from_annotations()`**:
- Replace current left/right calculation (lines ~772-810) with hybrid approach
- Keep bbox overlap check (already exists)
- Add high-reliability check before geometric calculation

#### New helper methods needed:
1. `_is_high_reliability_orientation(class_name: str) -> bool`
   - Check if class_name is in HIGH_RELIABILITY_CLASSES set
   
2. `_get_reference_object(obj1_data, obj2_data) -> Tuple[Dict, Dict, str]`
   - Returns: (reference_data, target_data, which_is_reference)
   - Reference = high-reliability object with pose data
   - Target = the other object

3. `_calculate_left_right_from_orientation(reference_data, target_data) -> str`
   - Uses reference object's left vector
   - Projects target's position onto left vector
   - Returns "left", "right", or "unknown"

### 2. `unified_qa_generation_utils.py`

#### Changes in `generate_additional_questions_not_in_qa_space()`:
- **Current**: No filtering for left/right questions (generates for all pairs)
- **New**: Add filter to check if at least one object is high-reliability

**Location**: Around line 801-814 (spatial question generation)

**New filter logic**:
```python
if question_type in ['spatial_left_right', 'spatial_front_behind']:
    # Get class names for both objects
    class_name_map = self._get_class_name_mapping(image_id, [target_object, reference_object])
    obj1_class = class_name_map.get(target_object, '')
    obj2_class = class_name_map.get(reference_object, '')
    
    # Check if at least one is high-reliability
    if not (self._is_high_reliability_orientation(obj1_class) or 
            self._is_high_reliability_orientation(obj2_class)):
        continue  # Skip: both are low-reliability
```

**New helper method needed**:
- `_get_class_name_mapping(image_id: str, object_names: List[str]) -> Dict[str, str]`
  - Load annotations once per image
  - Return mapping: `{object_name: class_name}`
  - Cache per image_id to avoid repeated loads

### 3. Constants/Configuration

#### New constant in `object_utils.py`:
```python
HIGH_RELIABILITY_CLASSES = {
    # People - Core
    'person', 'man', 'woman', 'child', 'boy', 'girl', 'baby', 'adult',
    # ... (all 60 classes from plan)
}
```

## Detailed Implementation Plan

### Phase 1: Add High-Reliability Class Set ✅
**File**: `object_utils.py`
**Location**: Module level (top of file, after imports, before class definition)
**Action**: Add `HIGH_RELIABILITY_CLASSES` constant (60 classes)
**Status**: ✅ COMPLETED - Constant added at top of `object_utils.py`

### Phase 2: Add Helper Methods in ObjectUtils
**File**: `object_utils.py`
**Methods to add**:
1. `_is_high_reliability_orientation(class_name: str) -> bool`
2. `_get_reference_object(obj1_data, obj2_data) -> Tuple[Dict, Dict, str]`
3. `_calculate_left_right_from_orientation(reference_data, target_data) -> str`
4. `_calculate_left_right_hybrid(obj1_data, obj2_data) -> str`

### Phase 3: Update `_calculate_from_annotations()` Method
**File**: `object_utils.py`
**Location**: Lines ~772-810 (left/right calculation section)
**Changes**:
- Replace current 2D bbox center-based left/right logic
- Call `_calculate_left_right_hybrid()` instead
- Keep bbox overlap check (used for alignment detection)

### Phase 4: Add Filter in Question Generation
**File**: `unified_qa_generation_utils.py`
**Location**: Lines ~801-814 (spatial question generation)
**Changes**:
- Add class name mapping helper
- Add filter check before generating `spatial_left_right` questions
- Cache class name mappings per image_id

### Phase 5: Update `get_spatial_answer()` Method
**File**: `object_utils.py`
**Location**: Lines ~990-1050
**Changes**:
- Ensure `spatial_left_right` question type returns correct answer
- Should already work, but verify

## Algorithm Details

### Horizontal Alignment Check
```python
def _check_horizontal_alignment(bbox1_2d, bbox2_2d, overlap_threshold=0.3):
    """
    Check if objects are horizontally aligned (significant overlap)
    Returns True if aligned (should skip question), False otherwise
    """
    # Extract coordinates
    x1_min, x1_max = bbox1_2d[0], bbox1_2d[2]
    x2_min, x2_max = bbox2_2d[0], bbox2_2d[2]
    y1_min, y1_max = bbox1_2d[1], bbox1_2d[3]
    y2_min, y2_max = bbox2_2d[1], bbox2_2d[3]
    
    # Check if bboxes overlap
    x_overlap = not (x1_max < x2_min or x1_min > x2_max)
    y_overlap = not (y1_max < y2_min or y1_min > y2_max)
    
    if x_overlap and y_overlap:
        # Calculate overlap ratios
        x_overlap_size = min(x1_max, x2_max) - max(x1_min, x2_min)
        obj1_width = x1_max - x1_min
        obj2_width = x2_max - x2_min
        x_overlap_ratio = max(x_overlap_size / obj1_width if obj1_width > 0 else 0,
                             x_overlap_size / obj2_width if obj2_width > 0 else 0)
        
        y_overlap_size = min(y1_max, y2_max) - max(y1_min, y2_min)
        obj1_height = y1_max - y1_min
        obj2_height = y2_max - y2_min
        y_overlap_ratio = max(y_overlap_size / obj1_height if obj1_height > 0 else 0,
                             y_overlap_size / obj2_height if obj2_height > 0 else 0)
        
        # If both X and Y overlap significantly, objects are aligned
        if x_overlap_ratio > overlap_threshold and y_overlap_ratio > overlap_threshold:
            return True  # Aligned, skip question
    
    return False  # Not aligned, can ask question
```

### Geometric Left/Right Calculation
```python
def _calculate_left_right_from_orientation(self, reference_data: Dict, target_data: Dict) -> str:
    """
    Calculate left/right using reference object's orientation
    Returns: "left", "right", or "unknown"
    """
    # Get reference object's left vector
    left_vec = self._parse_vector(reference_data.get('left', []))
    if not left_vec or len(left_vec) != 3:
        return "unknown"
    
    # Get 3D centers
    reference_center = self._parse_vector(reference_data.get('pcd_center', []))
    target_center = self._parse_vector(target_data.get('pcd_center', []))
    
    if not reference_center or not target_center:
        return "unknown"
    
    if len(reference_center) != 3 or len(target_center) != 3:
        return "unknown"
    
    # Validate vectors
    front_vec = self._parse_vector(reference_data.get('front', []))
    if front_vec:
        is_valid, _ = self._validate_orientation_vectors(front_vec, left_vec)
        if not is_valid:
            return "unknown"
    
    # Calculate relative position
    relative_pos = np.array(target_center) - np.array(reference_center)
    
    # Project onto left vector
    dot_product = np.dot(relative_pos, left_vec)
    
    # Threshold for determination (0.1m = 10cm)
    threshold = 0.1
    
    if dot_product > threshold:
        return "left"  # target is to the left of reference (geometrically)
    elif dot_product < -threshold:
        return "right"  # target is to the right of reference (geometrically)
    else:
        return "unknown"  # Too close to determine
```

## Edge Cases

### 1. Both objects are high-reliability
- **Solution**: Use object2 as reference (standard approach), or prefer the one with better pose data

### 2. High-reliability object missing pose data
- **Solution**: Skip question (fallback to "unknown")

### 3. Objects too close together
- **Solution**: Return "unknown" if dot_product < threshold

### 4. Objects horizontally aligned
- **Solution**: Return "unknown" if bbox overlap > 30% in both X and Y

### 5. Class name not found in annotations
- **Solution**: Treat as low-reliability (skip question)

## Testing Considerations

1. **Test cases**:
   - Person + Table (should pass, use person's orientation)
   - Person + Person (should pass, both high-reliability)
   - Table + Floor (should fail, both low-reliability)
   - Person + Car (should pass, both high-reliability)
   - Person + Bottle (should pass, use person's orientation)

2. **Validation**:
   - Check that bbox overlap detection works correctly
   - Verify geometric calculation uses correct reference object
   - Ensure fallbacks work when pose data is missing

3. **Performance**:
   - Cache class name mappings per image_id
   - Avoid repeated annotation file loads

## Summary of Changes

1. ✅ Add `HIGH_RELIABILITY_CLASSES` constant (60 classes)
2. ✅ Add helper methods in `ObjectUtils`:
   - `_is_high_reliability_orientation()`
   - `_get_reference_object()`
   - `_calculate_left_right_from_orientation()`
   - `_calculate_left_right_hybrid()`
   - `_check_horizontal_alignment()`
   - `_validate_orientation_vectors()`
   - `_parse_vector()`
3. ✅ Update `_calculate_from_annotations()` to use hybrid approach
4. ✅ Add filter in `unified_qa_generation_utils.py` to check at least one high-reliability
5. ✅ Add `_get_class_name_mapping()` helper in `UnifiedQAGenerationUtils`

**Expected Result**: 62% of left/right questions pass (vs 10.5% with strict filtering)

