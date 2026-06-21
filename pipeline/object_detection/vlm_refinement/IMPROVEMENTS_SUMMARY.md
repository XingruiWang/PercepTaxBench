# VLM Refinement Script Improvements

## Problem Summary
- **Yesterday**: Wasted ~15,000-20,000 paid API calls using wrong logic (loading `_refined.json` instead of original `.json`)
- **Today**: Wasted ~5,000-10,000 paid API calls due to excessive rate limit retries
- **Total Waste**: ~20,000-30,000 paid API calls

## Improvements Made

### 1. Smart Rate Limit Handling ✅
**Problem**: Script kept retrying immediately when hitting rate limits, wasting API quota.

**Solution**:
- **Exponential backoff**: Wait time increases with each retry (4s, 8s, 16s, up to 30s)
- **Consecutive limit tracking**: After 3 consecutive rate limits, pause for 60 seconds
- **Smart retry logic**: Stop retrying after max attempts to avoid wasting calls
- **API call counting**: Track every API call made for cost monitoring

```python
if self.consecutive_rate_limits >= 3:
    wait_time = 60
    logger.warning(f"⚠️  Too many consecutive rate limits. Pausing for {wait_time}s...")
    time.sleep(wait_time)
    self.consecutive_rate_limits = 0
```

### 2. Progress Checkpointing & Resume Capability ✅
**Problem**: If job fails or is interrupted, need to start from scratch, wasting API calls on already-processed images.

**Solution**:
- **Checkpoint file**: Saves list of processed images to JSON file
- **Auto-resume**: On restart, skips already-processed images
- **Periodic saves**: Checkpoint saved after each image processed
- **Progress tracking**: Shows how many images already done vs remaining

```python
def save_checkpoint(self, image_id: str):
    """Save progress checkpoint"""
    self.processed_images.add(image_id)
    checkpoint_data = {
        'processed_images': list(self.processed_images),
        'last_updated': datetime.now().isoformat(),
        'api_calls_made': self.api_call_count
    }
    with open(self.checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
```

### 3. API Cost Tracking ✅
**Problem**: No visibility into how many paid API calls are being made.

**Solution**:
- **Real-time tracking**: Logs API call count after each image
- **Per-image stats**: Shows running total in progress logs
- **Final summary**: Reports total API calls made in summary

```
  💰 API calls so far: 1,234
```

### 4. Better Progress Monitoring ✅
**Problem**: Hard to know how long the job will take or if it's making progress.

**Solution**:
- **ETA calculation**: Shows estimated time remaining
- **Progress bar info**: `[123/9080] | ETA: 45.2 min | API calls: 1,234`
- **Speed tracking**: Shows avg time per image
- **Resume status**: Shows how many already processed on restart

### 5. Reduced Logging Noise ✅
**Solution**:
- Sample refinements shown (first 20 instead of all)
- Cleaner stats output
- Skip status for already-processed images

## Usage

### Test on 10 Images First (RECOMMENDED)
```bash
python object_detection/vlm_refinement/comprehensive_vlm_refinement_batched.py \
  --unified-output-dir ../../openimages_unified_output \
  --api-keys "KEY1,KEY2" \
  --object-list object_description/results/full_object_description/merged_objects_list.txt \
  --checkpoint-file object_detection/vlm_refinement/test_checkpoint.json \
  --image-ids "img1,img2,img3,img4,img5,img6,img7,img8,img9,img10"
```

### Run on All 9,080 Images (After Testing)
```bash
sbatch object_detection/vlm_refinement/run_comprehensive_vlm_refinement.sh
```

### Resume Interrupted Job
Just run the same command again - it will automatically skip already-processed images using the checkpoint file.

## Expected Behavior

### Rate Limiting
- **First rate limit**: Wait 4s, rotate key, retry
- **Second rate limit**: Wait 8s, rotate key, retry
- **Third rate limit**: Wait 16s, rotate key, retry
- **After 3 consecutive**: Pause 60s, reset counter, continue
- **Max retries reached**: Skip batch, log error, continue to next image

### Progress
```
Already processed: 1,234
Remaining: 7,846
[1235/9080] | ETA: 123.4 min | API calls: 5,678
```

### Cost Tracking
```
💰 Total API calls made: 15,234
⏱️  Total time: 245.6 minutes
⚡ Avg time per image: 1.6 seconds
```

## Next Steps
1. ✅ **Test on 10 images first** to verify the improvements work
2. ✅ **Check the "bronze statue" case** in test output to confirm correct logic
3. ✅ **Monitor API call count** during test run
4. ⏳ **Scale to full dataset** only after test succeeds

## Files Changed
- `comprehensive_vlm_refinement_batched.py` - Main script with all improvements
- `run_comprehensive_vlm_refinement.sh` - SLURM script with checkpoint enabled
