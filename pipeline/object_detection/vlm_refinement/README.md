# VLM Object Name Refinement

This directory contains scripts and results for refining generic object names to more specific ones using Vision-Language Models (VLM).

## Overview

The VLM refinement process uses Gemini 2.5 Flash to analyze object crops and refine generic detection names (e.g., "army" → "soldier", "person" → "man/woman/child") to more specific, accurate names based on visual content.

## Files

### Scripts

- **`refine_object_names_with_vlm_batched.py`**: Main VLM refinement script with batching and parallel processing
- **`refine_object_names_with_vlm.py`**: Original single-object refinement script (deprecated)
- **`analyze_detection_tag_mismatch.py`**: Analyzes mismatches between detected objects and RAM tags
- **`run_vlm_refinement.sh`**: SLURM script to run VLM refinement
- **`submit_vlm_refinement.sh`**: Helper script to submit SLURM job with API keys
- **`test_vlm_refinement.sh`**: Test script for VLM refinement
- **`monitor_batched_progress.sh`**: Monitor progress of batched VLM refinement
- **`monitor_slurm_progress.sh`**: Monitor SLURM job progress

### Results

- **`vlm_refinement_batched_results.json`**: Combined results from all workers
- **`vlm_refinement_batched_results_worker0.json`**: Worker 0 results
- **`vlm_refinement_batched_results_worker1.json`**: Worker 1 results
- **`vlm_refinement_batched_results_worker*_progress.json`**: Progress tracking files
- **`vlm_refinement_results.json`**: Original single-object refinement results (deprecated)
- **`detection_tag_analysis_refined_mappings.json`**: Data-driven refinement mappings

## Refinement Statistics

- **Total refined files**: 1,254 / 9,079 (13.8%)
- **Total object refinements**: 3,278
- **Refinement types**: 6 unique mappings

### Refinement Mappings

| Generic Name | Specific Names |
|--------------|----------------|
| `animal` | `dog` |
| `army` | `soldier` |
| `food` | `fruit`, `sandwich` |
| `furniture` | `chair` |
| `person` | `child`, `man`, `pedestrian`, `woman` |
| `plant` | `flower` |

## Usage

### Run VLM Refinement

```bash
cd object_detection/vlm_refinement

# Submit SLURM job with API keys
bash submit_vlm_refinement.sh

# Or run directly
python refine_object_names_with_vlm_batched.py \
    --unified_output_dir /path/to/openimages_unified_output \
    --api_keys KEY1 KEY2 \
    --parallel \
    --resume
```

### Monitor Progress

```bash
# Monitor batched progress
bash monitor_batched_progress.sh

# Monitor SLURM job
bash monitor_slurm_progress.sh
```

### Analyze Detection Mismatches

```bash
python analyze_detection_tag_mismatch.py \
    --unified_output_dir /path/to/openimages_unified_output \
    --output_file detection_tag_mismatch_analysis.json
```

## Output Format

Refined annotations are saved as `*_refined.json` files alongside the original annotations:

```
openimages_unified_output/
├── <image_id>/
│   ├── annotations/
│   │   ├── <image_id>.json           ← Original
│   │   └── <image_id>_refined.json   ← VLM-refined
│   ├── object_crops/
│   └── <image_id>.jpg
```

### Refined Annotation Structure

```json
{
  "detections": [
    {
      "object_name": "obj_01_army",
      "class_name": "soldier",              // Refined name
      "original_class_name": "army",        // Original name
      "refined_by_vlm": true,               // VLM refinement flag
      ...
    }
  ]
}
```

## Integration with QA Generation

The QA generation scripts automatically use refined annotations when available:

- `qa_gen/generate_comprehensive_qa.py`
- `qa_gen/generate_full_qa_pipeline.py`

See `qa_gen/REFINED_QA_GENERATION.md` for details.

## Performance

- **Batching**: Processes 4 objects per API call (reduced from unlimited to avoid safety blocks)
- **Parallel Processing**: Uses multiple API keys with multiprocessing
- **Resume Functionality**: Skips already-refined files
- **Progress Saving**: Periodic checkpoints every 10 files
- **Processing Time**: ~3-4 hours for 9,079 images with 2 API keys

## Notes

- VLM refinement only occurs when generic names are detected
- Refinements are validated against the original OpenImages objects list
- All refined names are guaranteed to be in the original vocabulary
- The process is idempotent - can be safely re-run

