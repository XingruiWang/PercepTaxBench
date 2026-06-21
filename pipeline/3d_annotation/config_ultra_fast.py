"""
ULTRA-FAST configuration for maximum processing speed.
Optimized model settings, reduced precision, and speed-focused parameters.
"""

# ULTRA-FAST Configuration
ULTRA_FAST_CONFIG = {
    # Model optimization settings
    "model_optimizations": {
        "use_mixed_precision": True,  # Use FP16 for faster inference
        "use_torch_compile": True,    # Use PyTorch 2.0 compilation
        "use_channels_last": True,    # Optimize memory layout
        "use_autocast": True,        # Automatic mixed precision
    },
    
    # GroundingDINO optimizations
    "groundingdino": {
        "box_threshold": 0.35,       # Lower threshold for faster processing
        "text_threshold": 0.25,      # Lower threshold for faster processing
        "nms_threshold": 0.7,        # Optimized NMS threshold
        "max_detections": 20,        # Limit max detections for speed
        "use_fast_mode": True,       # Enable fast inference mode
    },
    
    # SAM optimizations
    "sam": {
        "points_per_side": 16,       # Reduce from 32 for speed
        "pred_iou_thresh": 0.88,    # Optimized threshold
        "stability_score_thresh": 0.95,  # Optimized threshold
        "box_nms_thresh": 0.7,      # Optimized NMS
        "crop_n_layers": 0,         # Disable crop layers for speed
        "crop_nms_thresh": 0.7,     # Optimized crop NMS
        "crop_overlap_ratio": 512,  # Optimized overlap
        "crop_n_points_downscale_factor": 1,  # Disable downscaling
        "point_grids": None,        # Disable point grids for speed
        "min_mask_region_area": 0,  # No minimum area filtering
    },
    
    # RAM optimizations
    "ram": {
        "use_fast_mode": True,       # Enable fast inference
        "max_tags": 50,             # Limit max tags for speed
        "confidence_threshold": 0.5, # Lower confidence threshold
    },
    
    # 3D Reconstruction optimizations
    "reconstruct3d": {
        "use_fast_mode": True,       # Enable fast reconstruction
        "point_cloud_density": 0.5,  # Reduce point cloud density
        "max_points": 10000,        # Limit max points for speed
        "skip_refinement": True,     # Skip refinement steps
    },
    
    # Pose estimation optimizations
    "pose3d": {
        "use_fast_mode": True,       # Enable fast pose estimation
        "skip_validation": True,     # Skip pose validation
        "max_iterations": 5,        # Reduce iterations for speed
    },
    
    # Pipeline optimizations
    "pipeline": {
        "skip_visualizations": True, # Skip 3D visualizations
        "skip_object_crops": True,   # Skip object crop generation
        "skip_annotations": False,   # Keep essential annotations
        "use_parallel_processing": True,  # Enable parallel processing
        "max_workers": 16,          # Maximum parallel workers
        "batch_size": 128,          # Large batch size for GPU efficiency
    },
    
    # Memory optimizations
    "memory": {
        "clear_cache_frequency": 10, # Clear GPU cache every 10 batches
        "use_gradient_checkpointing": False,  # Disable for speed
        "use_amp": True,            # Use automatic mixed precision
        "pin_memory": True,         # Pin memory for faster transfer
    },
    
    # Logging optimizations
    "logging": {
        "level": "WARNING",          # Minimal logging for speed
        "log_frequency": 100,       # Log every 100 images
        "skip_debug_logs": True,    # Skip debug logging
    }
}

# Apply optimizations to existing config
def apply_ultra_fast_config(base_config):
    """Apply ultra-fast optimizations to base configuration"""
    optimized_config = base_config.copy()
    
    # Apply model optimizations
    for key, value in ULTRA_FAST_CONFIG.items():
        if key in optimized_config:
            if isinstance(value, dict):
                optimized_config[key].update(value)
            else:
                optimized_config[key] = value
        else:
            optimized_config[key] = value
    
    return optimized_config

# Environment variables for maximum performance
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking CUDA operations
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Enable CUDA device-side assertions
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Optimize memory allocation
os.environ['CUDA_CACHE_DISABLE'] = '0'    # Enable CUDA cache
os.environ['CUDA_CACHE_PATH'] = '/tmp/cuda_cache'  # Set cache path
os.environ['OMP_NUM_THREADS'] = '64'      # Maximum OpenMP threads
os.environ['MKL_NUM_THREADS'] = '64'      # Maximum MKL threads
os.environ['NUMEXPR_NUM_THREADS'] = '64'  # Maximum NumExpr threads

# PyTorch optimizations
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True      # Enable cuDNN benchmarking
    torch.backends.cudnn.deterministic = False # Disable deterministic mode for speed
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for speed
    torch.backends.cudnn.allow_tf32 = True     # Allow TF32 for cuDNN
    torch.backends.cuda.enable_flash_sdp(True) # Enable flash attention
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # Enable memory efficient attention
    torch.backends.cuda.enable_math_sdp(True)  # Enable math attention

print("ULTRA-FAST configuration loaded with maximum performance optimizations!")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
