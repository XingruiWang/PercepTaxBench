#!/usr/bin/env python3
"""
Quick script to upload processing_errors.txt to Hugging Face
"""

import os

def quick_upload():
    """Quick upload function with all the necessary steps"""
    
    file_path = "/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations/processing_errors.txt"
    
    print("🚀 Quick Upload to Hugging Face")
    print("=" * 40)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Get file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"📁 File: processing_errors.txt ({file_size:.1f} MB)")
    
    # Show file preview
    print("\n📄 File preview (first 10 lines):")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"   {line.rstrip()}")
    
    print("\n" + "="*40)
    print("📋 INSTRUCTIONS FOR UPLOAD:")
    print("="*40)
    
    print("\n1️⃣ Install Hugging Face Hub (if not already installed):")
    print("   pip install huggingface_hub")
    
    print("\n2️⃣ Login to Hugging Face (if not already logged in):")
    print("   huggingface-cli login")
    print("   # Enter your token when prompted")
    
    print("\n3️⃣ Run the uploader script:")
    print("   python /path/to/Taxonomy/scripts/huggingface_uploader.py --processing-errors")
    
    print("\n4️⃣ Alternative: Manual upload using Python:")
    print('''
from huggingface_hub import upload_file, create_repo

# Replace with your username
username = "your_username"
repo_name = "taxonomy-processing-errors"
repo_id = f"{username}/{repo_name}"

# Create repository (optional if it doesn't exist)
create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# Upload file
url = upload_file(
    path_or_fileobj="/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations/processing_errors.txt",
    path_in_repo="processing_errors.txt",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload scene processing errors report"
)

print(f"✅ Uploaded successfully: {url}")
    ''')
    
    print("\n5️⃣ One-liner upload (if you have huggingface_hub installed):")
    print('''
python -c "
from huggingface_hub import upload_file, create_repo
username = 'YOUR_USERNAME'  # Replace with your username
repo_name = 'taxonomy-processing-errors'
repo_id = f'{username}/{repo_name}'
create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)
url = upload_file(
    path_or_fileobj='/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations/processing_errors.txt',
    path_in_repo='processing_errors.txt',
    repo_id=repo_id,
    repo_type='dataset',
    commit_message='Upload scene processing errors report'
)
print(f'✅ Uploaded: https://huggingface.co/datasets/{repo_id}')
"
    ''')
    
    print("\n" + "="*40)
    print("💡 TIPS:")
    print("="*40)
    print("• Create a meaningful repository name like 'taxonomy-scene-errors'")
    print("• Add a description in the repository settings")
    print("• Consider adding tags: computer-vision, dataset-processing, error-analysis")
    print("• The file contains info about 5607 skipped scenes with no foreground objects")


if __name__ == "__main__":
    quick_upload()


