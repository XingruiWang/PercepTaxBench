# Cleaned Scripts Directory Summary

## Overview
This document summarizes the cleaned and organized scripts directory after removing unused scripts and keeping only the actively used ones.

## **ACTIVELY USED SCRIPTS (KEPT)**

### **Core Object Description Generation:**
- `generate_object_descriptions.py` - **Main consolidated script** for generating object descriptions
  - Handles both text file input (`SM_names.txt`) and OpenImages output directory
  - Uses Gemini API for structured JSON descriptions
  - Includes incremental saving and error handling
- `run_object_descriptions_from_file.sh` - **Bash script** for running with text file input
- `run_object_descriptions_from_openimages.sh` - **Bash script** for running with OpenImages output
- `fix_incomplete_descriptions.py` - **Script for fixing** problematic entries in existing JSON files
- `run_fix_incomplete_descriptions.sh` - **Bash script** for running the fix process

### **QA Generation:**
- `qa_gen/generate_qa_separate.py` - **Main QA generation script**
  - Enhanced with object descriptions and 3D pose information
  - Generates QA pairs with cropped images and visualizations
  - Uses Gemini API for logical answer generation
- `qa_gen/run_qa_generation_test.sh` - **Bash script** for testing QA generation

### **3D Ground Truth Pipeline:**
- `run_full_openimages_pipeline_optimized.sh` - **Main pipeline script**
  - Generates 3D ground truth annotations
  - Includes object detection, segmentation, depth estimation, and pose estimation
  - Optimized for memory and performance

### **Visualization and Cropping (RESTORED):**
- `visualize_3d_data.py` - **3D visualization script** (RESTORED)
  - Generates 3D visualizations from annotation data
  - Used by the main pipeline for creating visual outputs
  - Creates 3D plots showing object positions and orientations
- `generate_cropped_objects.py` - **Object cropping script** (RESTORED)
  - Generates cropped images of detected objects
  - Used by the main pipeline for creating object crops
  - Supports both bounding box and segmentation mask cropping

### **Essential Files:**
- `SM_names.txt` - **Category list** for object descriptions
- `pipeline.log` - **Log file** for pipeline execution
- `requirements_qa.txt` - **Dependencies** for QA generation

## **REMOVED SCRIPTS (No Longer Used)**

### **Redundant Object Description Scripts:**
- `generate_object_descriptions_from_file.py` - **Replaced** by consolidated `generate_object_descriptions.py`
- `run_object_description_full.sh` - **Replaced** by specific bash scripts
- `run_openimages_object_descriptions.sh` - **Replaced** by specific bash scripts

### **Test and Debug Scripts:**
- `test_single_class.py` - **Test script** no longer needed
- `test_parsing.py` - **Test script** no longer needed
- `test_object_description_generation.py` - **Test script** no longer needed
- `test_qa_generation_specific.sh` - **Test script** no longer needed
- `run_qa_generation_slurm.sh` - **Redundant** QA script
- `qa_generation_config.py` - **Configuration** integrated into main script

### **Pipeline Variants (Consolidated):**
- `run_full_openimages_memory_safe.sh` - **Consolidated** into main pipeline
- `run_full_openimages_conservative.sh` - **Consolidated** into main pipeline
- `run_full_openimages_ultra_resources.sh` - **Consolidated** into main pipeline
- `launch_ultra_fast.sh` - **Consolidated** into main pipeline
- `run_full_openimages_pipeline_ultra_fast.sh` - **Consolidated** into main pipeline

### **Utility Scripts (No Longer Used):**
- `batch_generate_qa.sh` - **Replaced** by enhanced QA generation
- `run_unified_openimages_local.sh` - **No longer used**
- `test_unprocessed_images_slurm.sh` - **No longer used**
- `monitor_pipeline.sh` - **No longer used**

### **Test Files:**
- `object_descriptions_test.json` - **Test file** no longer needed
- `test_object_descriptions.json` - **Test file** no longer needed
- `detected_classes.txt` - **Temporary file** no longer needed

## **Current Directory Structure**

```
taxonomy_datagen/SpatialReasonerDataGen/
├── generate_object_descriptions.py          # Main object description script
├── run_object_descriptions_from_file.sh     # Bash script for text file input
├── run_object_descriptions_from_openimages.sh # Bash script for OpenImages input
├── fix_incomplete_descriptions.py           # Fix script for problematic entries
├── run_fix_incomplete_descriptions.sh       # Bash script for running fixes
├── visualize_3d_data.py                     # 3D visualization script (RESTORED)
├── generate_cropped_objects.py              # Object cropping script (RESTORED)
├── run_full_openimages_pipeline_optimized.sh # Main pipeline script
├── SM_names.txt                             # Category list
├── pipeline.log                             # Pipeline log
├── requirements_qa.txt                      # QA dependencies
├── qa_gen/
│   ├── generate_qa_separate.py              # Main QA generation script
│   ├── run_qa_generation_test.sh            # QA test script
│   ├── README.md                            # QA documentation
│   └── logs/                               # QA logs
├── scripts/
│   ├── generate_3d_groundtruth_production.py # Core pipeline script
│   ├── generate_3d_groundtruth_production_ultra_fast.py # Fast pipeline variant
│   └── generate_qa_separate.py              # QA script copy
└── logs/                                    # Pipeline logs
```

## **Key Improvements**

1. **Consolidated Object Description Generation**: Single script handles both input types
2. **Enhanced QA Generation**: Integrated object descriptions and 3D pose information
3. **Restored Critical Scripts**: Visualization and cropping scripts are back
4. **Removed Redundancy**: Eliminated duplicate and unused scripts
5. **Clear Organization**: Logical grouping of related functionality
6. **Maintained Functionality**: All core features preserved

## **Usage Workflow**

1. **3D Ground Truth Generation**: `run_full_openimages_pipeline_optimized.sh`
2. **Object Description Generation**: 
   - Text file: `run_object_descriptions_from_file.sh`
   - OpenImages: `run_object_descriptions_from_openimages.sh`
3. **Fix Incomplete Descriptions**: `run_fix_incomplete_descriptions.sh`
4. **QA Generation**: `qa_gen/run_qa_generation_test.sh`

## **Notes**

- **RESTORED**: `visualize_3d_data.py` and `generate_cropped_objects.py` were accidentally deleted but have been recreated
- **CRITICAL**: These scripts are actively used by the main pipeline and are essential for generating visualizations and object crops
- **MAINTAINED**: All core functionality is preserved while removing unnecessary complexity
