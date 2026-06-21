#!/bin/bash

# Quick Hugging Face Upload Script
# Usage: ./upload_to_hf.sh [username] [repo_name]

FILE_PATH="/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations/processing_errors.txt"

echo "🤗 Hugging Face Quick Uploader"
echo "================================"

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "❌ Error: File not found at $FILE_PATH"
    exit 1
fi

# Get file size
FILE_SIZE=$(du -h "$FILE_PATH" | cut -f1)
echo "📁 File: processing_errors.txt ($FILE_SIZE)"

# Get username
if [ -z "$1" ]; then
    read -p "Enter your Hugging Face username: " USERNAME
else
    USERNAME="$1"
fi

# Get repository name
if [ -z "$2" ]; then
    read -p "Enter repository name [taxonomy-processing-errors]: " REPO_NAME
    REPO_NAME=${REPO_NAME:-taxonomy-processing-errors}
else
    REPO_NAME="$2"
fi

REPO_ID="${USERNAME}/${REPO_NAME}"

echo ""
echo "📤 Uploading to: https://huggingface.co/datasets/$REPO_ID"
echo ""

# Check if huggingface_hub is installed
python3 -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ huggingface_hub not installed. Installing..."
    pip install huggingface_hub
fi

# Upload using Python
python3 -c "
import sys
try:
    from huggingface_hub import upload_file, create_repo
    
    username = '$USERNAME'
    repo_name = '$REPO_NAME'
    repo_id = f'{username}/{repo_name}'
    
    print(f'🏗️ Creating/checking repository: {repo_id}')
    create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)
    
    print(f'📤 Uploading file...')
    url = upload_file(
        path_or_fileobj='$FILE_PATH',
        path_in_repo='processing_errors.txt',
        repo_id=repo_id,
        repo_type='dataset',
        commit_message='Upload scene processing errors report - $(date)'
    )
    
    print(f'✅ Upload successful!')
    print(f'🔗 Dataset URL: https://huggingface.co/datasets/{repo_id}')
    print(f'📄 File URL: https://huggingface.co/datasets/{repo_id}/blob/main/processing_errors.txt')
    
except Exception as e:
    print(f'❌ Upload failed: {e}')
    print('')
    print('💡 Make sure you are logged in:')
    print('   huggingface-cli login')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Upload completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "• Visit your dataset: https://huggingface.co/datasets/$REPO_ID"
    echo "• Add a description and README"
    echo "• Add relevant tags (computer-vision, dataset-processing, etc.)"
    echo "• Set license and other metadata"
else
    echo ""
    echo "❌ Upload failed. Please check the error above."
    echo ""
    echo "🔧 Troubleshooting:"
    echo "• Make sure you're logged in: huggingface-cli login"
    echo "• Check your internet connection"
    echo "• Verify your username and repository name"
fi


