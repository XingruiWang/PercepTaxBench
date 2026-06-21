# Conflict-Free QA Grouping

This module generates conflict-free object groups for QA generation, ensuring objects don't compete for the same questions.

## Files

### Core Scripts
- **`scripts/generate_conflict_free_qa_groups.py`** - Main script that creates conflict-free groups
- **`scripts/analyze_scene_qa_potential.py`** - Analyzes QA potential for each scene (dependency)

### Output Files
- **`conflict_free_qa_groups.json`** - Final output with 664 conflict-free groups and QA pairs
- **`scene_qa_potential_analysis.json`** - QA potential analysis results (dependency)

### Data Directory
- **`data/`** - Contains all input data files:
  - `scene_with_object/` - Scene files with objects
  - `mapping_placable/` - Placable object mappings
  - `all_objects_with_scenes.json` - Object name mappings

## Usage

```bash
# Generate conflict-free QA groups
python scripts/generate_conflict_free_qa_groups.py
```

## Results

- **Total scenes**: 63
- **Total groups**: 664 conflict-free groups
- **Total QA pairs**: 664 (one example QA per group)
- **Average groups per scene**: 10.5

Each group contains 6 objects that don't compete for the same questions (<30% overlap in question types), with one example QA pair generated for the first object in each group.
