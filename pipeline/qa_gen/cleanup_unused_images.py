#!/usr/bin/env python3
"""
Cleanup script to remove image directories that don't have QA in the benchmark.

This script identifies and removes directories from the images folder that
don't have corresponding questions in all_questions.json.

Usage:
    python cleanup_unused_images.py [--dry-run] [--backup-dir BACKUP_DIR] [--remove]
    
Options:
    --dry-run: Show what would be removed without actually removing (default)
    --backup-dir: Move directories here instead of deleting (optional)
    --remove: Actually remove directories (use with caution)
"""

import json
import shutil
from pathlib import Path
import argparse

IMAGES_DIR = Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_v2/images")
QUESTIONS_PATH = Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_v2/all_questions.json")

def main():
    parser = argparse.ArgumentParser(description='Clean up unused image directories')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be removed without actually removing (default)')
    parser.add_argument('--remove', action='store_true',
                       help='Actually remove directories (requires explicit flag)')
    parser.add_argument('--backup-dir', type=str, default=None,
                       help='Move directories to backup location instead of deleting')
    
    args = parser.parse_args()
    
    # Determine action mode
    if args.remove:
        action_mode = "remove"
        dry_run = False
    elif args.backup_dir:
        action_mode = "backup"
        dry_run = False
        backup_dir = Path(args.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    else:
        action_mode = "dry-run"
        dry_run = True
    
    print("=" * 60)
    print("Image Directory Cleanup Script")
    print("=" * 60)
    print(f"Mode: {action_mode.upper()}")
    if action_mode == "backup":
        print(f"Backup directory: {backup_dir}")
    print()
    
    # Load benchmark image IDs
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_PATH}")
        return
    
    print(f"Loading benchmark questions from: {QUESTIONS_PATH}")
    with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    benchmark_image_ids = set(q.get('image_id') for q in questions if q.get('image_id'))
    print(f"✅ Found {len(benchmark_image_ids)} unique image IDs in benchmark")
    print()
    
    # Check images directory
    if not IMAGES_DIR.exists():
        print(f"ERROR: Images directory not found: {IMAGES_DIR}")
        return
    
    print(f"Scanning images directory: {IMAGES_DIR}")
    all_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    print(f"✅ Found {len(all_dirs)} subdirectories")
    print()
    
    # Identify directories to remove
    dirs_to_remove = []
    dirs_to_keep = []
    
    for d in all_dirs:
        if d.name in benchmark_image_ids:
            dirs_to_keep.append(d)
        else:
            dirs_to_remove.append(d)
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Directories to KEEP (in benchmark): {len(dirs_to_keep)}")
    print(f"Directories to REMOVE (not in benchmark): {len(dirs_to_remove)}")
    print()
    
    if not dirs_to_remove:
        print("✅ No directories to remove! All directories are in the benchmark.")
        return
    
    # Show some examples
    print(f"Examples of directories to remove:")
    for d in sorted(dirs_to_remove)[:10]:
        print(f"  - {d.name}")
    if len(dirs_to_remove) > 10:
        print(f"  ... and {len(dirs_to_remove) - 10} more")
    print()
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN - No changes made")
        print("=" * 60)
        print("To actually remove directories, use: --remove")
        print("To backup directories instead, use: --backup-dir /path/to/backup")
        return
    
    # Perform action
    print("=" * 60)
    print(f"PROCESSING ({action_mode.upper()})...")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    for d in dirs_to_remove:
        try:
            if action_mode == "remove":
                shutil.rmtree(d)
                print(f"✅ Removed: {d.name}")
            elif action_mode == "backup":
                dest = backup_dir / d.name
                shutil.move(str(d), str(dest))
                print(f"✅ Moved to backup: {d.name} -> {backup_dir.name}/{d.name}")
            success_count += 1
        except Exception as e:
            print(f"❌ Error processing {d.name}: {e}")
            error_count += 1
    
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"✅ Successfully processed: {success_count}")
    if error_count > 0:
        print(f"❌ Errors: {error_count}")
    
    # Verify final state
    remaining_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    print(f"📁 Remaining directories: {len(remaining_dirs)}")
    print(f"📊 Expected: {len(benchmark_image_ids)}")
    
    if len(remaining_dirs) == len(benchmark_image_ids):
        print("✅ Verification passed: Directory count matches benchmark!")
    else:
        print(f"⚠️  Warning: Directory count doesn't match expected count")

if __name__ == "__main__":
    main()

