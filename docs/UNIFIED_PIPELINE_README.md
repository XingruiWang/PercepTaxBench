# Unified OpenImages Processing Pipeline

This directory contains unified scripts that generate 3D data, annotations, and visualizations in a single run, eliminating the need for separate production and annotation steps.

## 🎯 **What These Scripts Do**

The unified pipeline processes each image through the complete workflow:

1. **3D Generation**: Creates 3D reconstructions, pose estimates, and point clouds
2. **Annotation Generation**: Produces annotated images with bounding boxes
3. **Summary Creation**: Generates text summaries of annotations
4. **File Organization**: Saves all outputs in organized directories

## 📁 **Output Structure**

For each processed image, you get **4 files**:

```
openimages_unified_output/           ← 3D data and JSON files
├── image1.json                     ← Complete 3D annotation data
├── image2.json
└── ...

openimages_unified_annotations/      ← Visualizations and summaries
├── image1_annotated.png            ← Annotated image with bounding boxes
├── image1_summary.txt              ← Text summary of annotations
├── image1.json                     ← Copy of JSON for convenience
├── image2_annotated.png
├── image2_summary.txt
├── image2.json
└── ...
```

## 🚀 **Usage Options**

### **Option 1: SLURM Batch (Recommended for Production)**

```bash
# Submit to SLURM queue
sbatch run_unified_openimages_slurm.sh

# Check job status
squeue -u $USER

# Monitor progress
tail -f openimages_unified_*.out
```

### **Option 2: Local Execution (Good for Testing)**

```bash
# Run directly on the current machine
./run_unified_openimages_local.sh

# Or run with bash
bash run_unified_openimages_local.sh
```

## ⚙️ **Configuration**

Edit the configuration section in either script to customize:

```bash
# Configuration - EASY TO MODIFY
MAX_IMAGES=50          # Change this to process more images later
BATCH_SIZE=4           # Processing batch size
DEVICE="cuda"          # Use "cpu" if GPU memory issues
INPUT_DIR="..."        # Input image directory
OUTPUT_DIR="..."       # 3D data output directory
ANNOTATIONS_DIR="..."  # Annotations output directory
```

## 🔄 **Resume Functionality**

Both scripts automatically support **resume processing**:

- **Progress Tracking**: Saves completed images in `processing_progress.txt`
- **Automatic Resume**: Re-run the script to continue from where it left off
- **Crash Recovery**: If the script crashes, just restart it
- **No Duplication**: Already processed images are automatically skipped

## 📊 **Progress Monitoring**

The scripts provide real-time progress updates:

```
Progress: 10/50 completed (Success: 9, Failed: 1)
Progress: 20/50 completed (Success: 18, Failed: 2)
...
```

## 🎛️ **Processing Control**

### **Start with Small Batch**
```bash
MAX_IMAGES=10  # Test with just 10 images first
```

### **Scale Up Gradually**
```bash
MAX_IMAGES=50   # Process 50 images
MAX_IMAGES=100  # Process 100 images
MAX_IMAGES=1000 # Process 1000 images
```

### **Full Dataset Processing**
```bash
MAX_IMAGES=10000  # Process all images (will take days)
```

## 🚨 **Troubleshooting**

### **GPU Memory Issues**
```bash
DEVICE="cpu"           # Use CPU instead of GPU
BATCH_SIZE=1           # Reduce batch size
```

### **Resume from Specific Point**
```bash
# Edit the progress file to remove specific images
vim openimages_unified_output/processing_progress.txt

# Or delete the progress file to start fresh
rm openimages_unified_output/processing_progress.txt
```

### **Check Logs**
```bash
# SLURM output logs
tail -f openimages_unified_*.out
tail -f openimages_unified_*.err

# Pipeline logs
tail -f pipeline.log
```

## 💡 **Best Practices**

1. **Start Small**: Begin with 10-50 images to test the pipeline
2. **Monitor Resources**: Watch GPU memory and CPU usage
3. **Use SLURM**: For production runs, always use the SLURM version
4. **Check Outputs**: Verify a few outputs before scaling up
5. **Backup Progress**: Keep the progress file safe for resume functionality

## 🔧 **Customization**

### **Add New Output Types**
Edit the `process_single_image()` function to add new processing steps.

### **Modify Processing Pipeline**
Change the Python script calls to use different models or parameters.

### **Output Format Changes**
Modify the file naming and organization in the script.

## 📈 **Performance Expectations**

- **GPU Processing**: ~1-2 minutes per image
- **CPU Processing**: ~5-10 minutes per image
- **50 Images**: ~1-2 hours on GPU, ~4-8 hours on CPU
- **1000 Images**: ~1-2 days on GPU, ~1 week on CPU

## 🎉 **Success Indicators**

When the pipeline completes successfully, you should see:

```
=== PROCESSING COMPLETE ===
Total processed: 50
Successful: 48
Failed: 2
Time finished: [timestamp]

Output files per image:
  - JSON: [directory]/*.json (3D data)
  - PNG: [directory]/*_annotated.png (visualizations)
  - TXT: [directory]/*_summary.txt (summaries)
  - JSON: [directory]/*.json (copied for convenience)
```

## 🚀 **Next Steps**

1. **Test with 10 images**: Verify the pipeline works correctly
2. **Scale to 50 images**: Process a meaningful batch
3. **Scale to 100+ images**: For production datasets
4. **Monitor and optimize**: Adjust parameters based on results
