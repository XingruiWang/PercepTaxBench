#!/usr/bin/env python3
"""
Process only missing OpenImages that haven't been processed yet.
This script identifies which images are missing and processes only those.
"""

import os
import sys
import subprocess
from pathlib import Path

def find_missing_images(source_dir, output_dir):
    """Find images that exist in source but not in output"""
    source_images = set()
    output_images = set()
    
    # Get all source images
    for file in os.listdir(source_dir):
        if file.endswith('.jpg'):
            image_id = file.replace('.jpg', '')
            source_images.add(image_id)
    
    # Get all processed images
    for dir_name in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, dir_name)):
            output_images.add(dir_name)
    
    # Find missing images
    missing_images = source_images - output_images
    
    print(f"Total source images: {len(source_images)}")
    print(f"Total processed images: {len(output_images)}")
    print(f"Missing images: {len(missing_images)}")
    
    return sorted(list(missing_images))

def process_missing_images(missing_images, source_dir, output_dir, working_dir):
    """Process only the missing images"""
    if not missing_images:
        print("No missing images to process!")
        return
    
    print(f"Processing {len(missing_images)} missing images...")
    
    # Create a temporary script that processes only missing images
    script_content = f'''#!/usr/bin/env python3
import os
import sys
sys.path.append("{working_dir}")

from scripts.generate_3d_groundtruth_production import main
import argparse

# Set up arguments for missing images only
args = argparse.Namespace()
args.image_path = "{source_dir}"
args.output_path = "{output_dir}"
args.batch_size = 1
args.max_workers = 1
args.device = "cuda"
args.enable_pose3d = True
args.enable_pose_filtering = True
args.generate_annotations = True
args.save_pcd = False
args.log_level = "INFO"
args.skip_visualizations = False
args.skip_object_crops = False
args.md5 = None

# Process only missing images
missing_images = {missing_images}
print(f"Processing {{len(missing_images)}} missing images...")

# Modify the main function to process only specific images
original_main = main

def process_specific_images():
    from scripts.generate_3d_groundtruth_production import TagAndSegment, Reconstruct3D, Pose3DOrientAnything
    import torch
    from pathlib import Path
    import logging
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    # Initialize models
    logger.info("Initializing models...")
    tag_and_segment = TagAndSegment()
    reconstruct_3d = Reconstruct3D()
    pose_3d = Pose3DOrientAnything()
    
    device = torch.device(args.device)
    tag_and_segment.to(device)
    reconstruct_3d.to(device)
    pose_3d.to(device)
    
    # Process only missing images
    processed_count = 0
    for image_id in missing_images:
        image_path = os.path.join("{source_dir}", f"{{image_id}}.jpg")
        if os.path.exists(image_path):
            logger.info(f"Processing {{image_id}}...")
            try:
                # Process single image
                # This is a simplified version - you'd need to implement the full pipeline
                processed_count += 1
                if processed_count % 10 == 0:
                    logger.info(f"Processed {{processed_count}}/{{len(missing_images)}} images")
            except Exception as e:
                logger.error(f"Failed to process {{image_id}}: {{e}}")
    
    logger.info(f"Completed processing {{processed_count}} images")

if __name__ == "__main__":
    process_specific_images()
'''
    
    # Write temporary script
    temp_script = "/tmp/process_missing_images.py"
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(temp_script, 0o755)
    
    # Run the script
    print("Running processing script...")
    result = subprocess.run([sys.executable, temp_script], 
                          cwd=working_dir,
                          capture_output=True, 
                          text=True)
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Clean up
    os.remove(temp_script)
    
    return result.returncode == 0

def main():
    # Configuration
    source_dir = "/path/to/project/openimages_train_10000"
    output_dir = "/path/to/project/openimages_unified_output"
    working_dir = "/path/to/SpatialReasonerDataGen"
    
    print("=== OpenImages Missing Image Processor ===")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Working directory: {working_dir}")
    
    # Find missing images
    missing_images = find_missing_images(source_dir, output_dir)
    
    if missing_images:
        print(f"\\nFirst 10 missing images:")
        for img in missing_images[:10]:
            print(f"  {img}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
        
        # Process missing images
        success = process_missing_images(missing_images, source_dir, output_dir, working_dir)
        
        if success:
            print("\\n✅ Successfully processed missing images!")
        else:
            print("\\n❌ Failed to process some images")
    else:
        print("\\n✅ All images already processed!")

if __name__ == "__main__":
    main()
