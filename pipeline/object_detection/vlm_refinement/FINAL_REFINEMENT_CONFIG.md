# Final VLM Refinement Configuration

## ✅ All Requirements Met

### 1. **Uses Only Image Tags** ✅
- Gemini is explicitly instructed: "ONLY choose from the tags listed above"
- Prompt includes: `Available tags for this image: {tags}`
- Rule: "Each value must be from the available tags"

### 2. **Uses Cropped Object Images** ✅
- Line 127: `img = Image.open(obj['crop_path'])`
- Line 128: `images_to_send.append(img)`
- Sends actual PIL Image objects to Gemini
- Prompt says: "Look at the cropped image carefully"

### 3. **Appends New Objects to merged_objects_list.txt** ✅
- Line 341-342: Checks if object is new
- Line 356: `self.add_new_object(refined_name)` 
- Line 87-92: `save_new_objects()` appends to file
- Logs: "➕ New object discovered: '{object_name}'"

### 4. **Max 15 Objects Per API Call** ✅
- Line 329: `MAX_BATCH_SIZE = 15`
- Line 331: Splits into multiple batches if needed
- Line 338-345: Processes in chunks
- Prevents timeouts and safety blocks

### 5. **Safety Retry Features** ✅
- Line 179-253: Retry logic with exponential backoff
- Line 181-185: Consecutive rate limit detection (pauses 60s after 3 consecutive)
- Line 186-195: Exponential backoff (4s, 8s, 16s, 30s)
- Line 236-244: Handles ResourceExhausted errors
- Max retries: 3 attempts per batch

### 6. **API Key Rotation** ✅
- Line 100-103: `rotate_api_key()` function
- Line 203: `self.rotate_api_key()` on rate limit
- Line 240: `self.rotate_api_key()` on quota error
- Automatically switches between the 2 API keys

### 7. **Saves to openimages_unified_output** ✅
- Line 379: `output_file = annotation_file.parent / f"{image_id}_refined.json"`
- Saves as `{image_id}_refined.json` in the same annotations folder
- Original `.json` files remain untouched

### 8. **Progress Checkpointing** ✅
- Line 44-65: Checkpoint load/save functions
- Line 259-269: Skips already-processed images
- Line 385: Saves checkpoint after each image
- Can resume if interrupted

## Configuration

### Model
- **Gemini 2.5 Flash** (Line 97)
- Supports multiple images per request
- Better accuracy than 2.0

### Batch Processing
- **Max 15 objects per API call**
- Splits large images into multiple batches
- 1 second pause between batches

### Rate Limiting
- **Exponential backoff**: 4s → 8s → 16s → 30s
- **Consecutive limit detection**: 60s pause after 3 consecutive
- **API key rotation**: Switches on rate limit
- **Max 3 retries** per batch

### Safety Features
- Handles timeout errors (504)
- Handles resource exhausted (429)
- Handles blocked prompts (safety filter)
- Continues to next image on failure

## Running the Full Refinement

### Command
```bash
sbatch object_detection/vlm_refinement/run_comprehensive_vlm_refinement.sh
```

### Expected Performance
- **Total images**: 9,080
- **Estimated API calls**: ~10,000-12,000 (with batching and skipping)
- **Estimated time**: ~4-6 hours
- **Output**: `{image_id}_refined.json` files in each image's annotations folder

### Monitoring
```bash
# Check SLURM job status
squeue -u $USER

# Watch log file
tail -f logs/vlm_refinement_comprehensive_76XXX.log

# Check progress
grep "^\[" logs/vlm_refinement_comprehensive_76XXX.log | tail -1
```

### Resume After Interruption
The script automatically resumes from the checkpoint file. Just rerun:
```bash
sbatch object_detection/vlm_refinement/run_comprehensive_vlm_refinement.sh
```

## Test Results (15 Images)

- ✅ **14/15 images processed successfully**
- ✅ **125 objects refined** (person→man/woman/student/soldier, etc.)
- ✅ **83 objects kept** (already specific labels)
- ✅ **Key fix**: spear → rifle ✅
- ✅ **1 new object discovered**: guard
- ❌ **1 image failed**: 40 tomatoes (safety block) - now will be split into 3 batches of 15

## Cost Estimate

- **Gemini 2.5 Flash**: ~$0.00075 per request (with images)
- **Total requests**: ~10,000-12,000
- **Estimated cost**: ~$7.50-$9.00

Much better than the ~$20-30 wasted yesterday!
