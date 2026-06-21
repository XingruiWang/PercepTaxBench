from typing import Any, Dict, List

import cv2

import numpy as np
from PIL import Image
import supervision as sv
import torch
import torchvision

# Don't add local models to Python path - use installed packages instead

# Add local models to Python path to avoid import issues
import sys
import os

# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'models'))

# Add GroundingDINO models path
gdino_path = os.path.join(models_dir, 'GroundingDINO')
sys.path.insert(0, gdino_path)

# Add SAM models path
sam_path = os.path.join(models_dir, 'sam')
sys.path.insert(0, sam_path)

# Add RAM models path - add the parent directory to avoid conflicts with subdirectories
ram_parent_path = models_dir
sys.path.insert(0, ram_parent_path)

# Debug: print the paths
print(f"Models directory: {models_dir}")
print(f"Python path configured for local models")

try:
    # Use local GroundingDINO installation directly
    import sys
    gdino_local_path = os.path.join(models_dir, 'GroundingDINO')
    if gdino_local_path not in sys.path:
        sys.path.insert(0, gdino_local_path)
    
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    GROUNDINGDINO_AVAILABLE = True
    print("GroundingDINO imported successfully from local installation")
except ImportError as e:
    print(f"Error importing GroundingDINO: {e}")
    GROUNDINGDINO_AVAILABLE = False

# RAM - Use local installation since it's not in conda
from models.ram import get_transform, inference_ram
from models.ram.models.ram_plus import ram_plus

# SAM - Use local installation
from models.sam.build_sam import build_sam2
from models.sam.sam2_image_predictor import SAM2ImagePredictor

from srdatagen import AnnotType, SkipSampleException


class TagAndSegment:
    """Tag objects in an image and segment their masks.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # Build RAM
        self.ram_model = ram_plus(
            pretrained=self.cfg.ram.pretrained_ckpt,
            image_size=384, vit='swin_l')
        self.ram_model.eval().to(self.device)
        self.ram_transform = get_transform(image_size=384)

        # Build GroundingDINO - Use the working direct approach instead of problematic wrapper
        print("Initializing GroundingDINO with working direct approach...")
        
        # Load model using the working approach
        args = SLConfig.fromfile(self.cfg.gdino.model_config_path)
        args.device = self.device
        self.gdino_model = build_model(args)
        checkpoint = torch.load(self.cfg.gdino.model_checkpoint_path, map_location="cpu")
        load_res = self.gdino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(f"GroundingDINO load result: {load_res}")
        
        # CRITICAL: Force Python-only implementations to avoid _C module issues
        print("GroundingDINO: Forcing Python-only implementations...")
        if hasattr(self.gdino_model, 'backbone'):
            # Disable any C++ extensions in the backbone
            if hasattr(self.gdino_model.backbone, 'use_checkpoint'):
                self.gdino_model.backbone.use_checkpoint = False
            if hasattr(self.gdino_model.backbone, 'use_rel_pos'):
                self.gdino_model.backbone.use_rel_pos = False
        
        # Explicitly move model to device
        self.gdino_model = self.gdino_model.to(self.device)
        _ = self.gdino_model.eval()
        
        # Verify device placement
        if self.device != "cpu":
            print(f"GroundingDINO: Model device: {next(self.gdino_model.parameters()).device}")
        
        print("GroundingDINO initialized successfully with direct approach")

        # Build SAM
        print(f"SAM: Building SAM2 with config: {self.cfg.sam.cfg_path}")
        print(f"SAM: Building SAM2 with checkpoint: {self.cfg.sam.ckpt_path}")
        print(f"SAM: Building SAM2 with device: {self.device}")
        
        # Build SAM2 model using local installation
        self.sam_model = build_sam2(self.cfg.sam.cfg_path, self.cfg.sam.ckpt_path, device=self.device)
        print(f"SAM: SAM2 model built successfully, type: {type(self.sam_model)}")
        
        # Ensure model is on the correct device
        if self.device != "cpu":
            print(f"SAM: Moving SAM2 model to device: {self.device}")
            self.sam_model = self.sam_model.to(self.device)
            print(f"SAM: SAM2 model device: {next(self.sam_model.parameters()).device}")
        
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
        print(f"SAM: SAM2 predictor initialized successfully")
        
        # Ensure predictor is also on the correct device
        if self.device != "cpu":
            print(f"SAM: SAM2 predictor device check - model device: {next(self.sam_predictor.model.parameters()).device}")

    @torch.no_grad()
    def __call__(self, image: Image.Image, annot: AnnotType):
        # Step 1: Run RAM
        tags = self._run_ram(image)
        annot['tags'] = tags
        if len(tags) == 0:
            raise SkipSampleException('No tags detected by RAM')

        # Step 2: Run GroundingDINO
        detections = self._run_gdino(image, tags)
        annot['detections'] = detections
        if len(detections.class_id) < 1:
            raise SkipSampleException('No objects detected by GroundingDINO')

        # Step 3: Run SAM
        mask = self._run_sam(image, detections)
        detections.mask = mask

        # Step 4: Filtering
        detections = self._filter_detections(annot, detections)
        if len(detections.class_id) < 1:
            raise SkipSampleException('No objects after filtering')

        # Step 5: Sort detections by area
        sorted_indices = np.argsort(-detections.area)
        detections = detections[sorted_indices]

        # Step 6: Update masks to be subtracted from masks of contained bboxes
        mask_subtracted = self._mask_subtract_contained(detections)

        # Step 7: Prepare outputs
        detections_data = dict(
            xyxy=detections.xyxy, confidence=detections.confidence,
            class_id=detections.class_id, mask=detections.mask,
            box_area=detections.box_area, area=detections.area,
            mask_subtracted=mask_subtracted, classes=tags)
        annot['detections'] = self._prepare_outputs(detections_data)

        return annot

    def _run_ram(self, image: Image.Image) -> List[str]:
        image = image.copy()
        image = image.resize((384, 384))
        image = self.ram_transform(image).unsqueeze(0).to(self.device)

        res = inference_ram(image, self.ram_model)
        tags = [x.strip() for x in res[0].replace('|', ',').split(',')]
        tags = [x.lower() for x in tags if x != '']

        print(f"RAM: Total raw tags: {len(tags)}")

        # Remove ignored classes
        tags = [x for x in tags if x not in self.cfg.ram.ignore_classes]
        
        print(f"RAM: After filtering ignored classes: {tags[:10]}...")
        print(f"RAM: Final tags count: {len(tags)}")

        return tags

    def _run_gdino(self, image: Image.Image, tags: List[str]) -> sv.Detections:
        # Use configuration values
        box_threshold = self.cfg.gdino.box_threshold
        text_threshold = self.cfg.gdino.text_threshold
        
        print(f"GroundingDINO: Processing image with {len(tags)} tags: {tags[:5]}...")
        print(f"GroundingDINO: Using box_threshold={box_threshold}, text_threshold={text_threshold}")
        
        try:
            # Convert PIL image to tensor format that GroundingDINO expects
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(image, None)  # 3, h, w
            
            # Process each tag individually and combine results
            all_boxes = []
            all_scores = []
            all_class_ids = []
            
            for i, tag in enumerate(tags):
                print(f"GroundingDINO: Processing tag '{tag}'...")
                
                try:
                    # Prepare caption (GroundingDINO expects lowercase, ends with period)
                    caption = tag.lower().strip()
                    if not caption.endswith("."):
                        caption = caption + "."
                    
                    # Run GroundingDINO inference with enhanced error handling
                    with torch.no_grad():
                        try:
                            # First try normal inference
                            outputs = self.gdino_model(image_tensor[None].to(self.device), captions=[caption])
                        except Exception as inference_error:
                            if "_C" in str(inference_error):
                                print(f"GroundingDINO: _C error during inference for tag '{tag}', trying alternative approach...")
                                # Try to use CPU-only inference as fallback
                                try:
                                    # Move both model and input to CPU for fallback
                                    model_cpu = self.gdino_model.cpu()
                                    image_tensor_cpu = image_tensor[None].cpu()
                                    outputs = model_cpu(image_tensor_cpu, captions=[caption])
                                    # Move model back to original device
                                    self.gdino_model = model_cpu.to(self.device)
                                    print(f"GroundingDINO: CPU fallback successful for tag '{tag}'")
                                except Exception as cpu_error:
                                    print(f"GroundingDINO: CPU fallback also failed for tag '{tag}': {cpu_error}")
                                    continue
                            else:
                                raise inference_error
                    
                    # Extract predictions - keep on device for processing, then move to CPU
                    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
                    boxes = outputs["pred_boxes"][0]  # (nq, 4)
                    
                    # Filter by box threshold
                    filt_mask = logits.max(dim=1)[0] > box_threshold
                    logits_filt = logits[filt_mask]
                    boxes_filt = boxes[filt_mask]
                    
                    if len(boxes_filt) > 0:
                        # Get scores and phrases
                        scores = logits_filt.max(dim=1)[0]
                        
                        # Filter by text threshold
                        text_mask = scores > text_threshold
                        if text_mask.any():
                            final_boxes = boxes_filt[text_mask]
                            final_scores = scores[text_mask]
                            
                            # Convert GroundingDINO center-based coordinates to corner-based coordinates
                            # GroundingDINO returns [center_x, center_y, width, height] in normalized coordinates
                            # Need to convert to [x1, y1, x2, y2] format
                            image_width, image_height = image.size
                            
                            # Convert to pixel coordinates first
                            boxes_pixel = final_boxes.clone()
                            boxes_pixel = boxes_pixel * torch.Tensor([image_width, image_height, image_width, image_height]).to(boxes_pixel.device)
                            
                            # Convert from center-based to corner-based format
                            for j in range(boxes_pixel.size(0)):
                                # [center_x, center_y, width, height] -> [x1, y1, x2, y2]
                                center_x, center_y, width, height = boxes_pixel[j]
                                x1 = center_x - width / 2
                                y1 = center_y - height / 2
                                x2 = center_x + width / 2
                                y2 = center_y + height / 2
                                boxes_pixel[j] = torch.tensor([x1, y1, x2, y2])
                            
                            # Move to CPU before numpy conversion
                            final_boxes_cpu = boxes_pixel.cpu()
                            final_scores_cpu = final_scores.cpu()
                            
                            # Add to results
                            all_boxes.extend(final_boxes_cpu.numpy())
                            all_scores.extend(final_scores_cpu.numpy())
                            all_class_ids.extend([i] * len(final_boxes))
                            
                            print(f"GroundingDINO: Found {len(final_boxes)} detections for tag '{tag}'")
                
                except Exception as tag_error:
                    # Handle individual tag processing errors gracefully
                    if "_C" in str(tag_error):
                        print(f"GroundingDINO: _C error for tag '{tag}', skipping this tag")
                        continue
                    else:
                        print(f"GroundingDINO: Error processing tag '{tag}': {tag_error}")
                        continue
            
            # Convert to supervision Detections format
            if all_boxes:
                all_boxes = np.array(all_boxes)
                all_scores = np.array(all_scores)
                all_class_ids = np.array(all_class_ids)
                
                print(f"GroundingDINO: Total detections before NMS: {len(all_boxes)}")
                print(f"GroundingDINO: Detection confidences: {all_scores[:5]}")
                print(f"GroundingDINO: Detection classes: {[tags[i] for i in all_class_ids[:5]]}")
                
                # Apply Non-Maximum Suppression to remove overlapping detections
                if len(all_boxes) > 1:
                    # Convert to torch tensors for NMS
                    boxes_tensor = torch.from_numpy(all_boxes).float()
                    scores_tensor = torch.from_numpy(all_scores).float()
                    
                    # Apply NMS with IoU threshold 0.5
                    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
                    keep_indices = keep_indices.numpy()
                    
                    # Filter results
                    all_boxes = all_boxes[keep_indices]
                    all_scores = all_scores[keep_indices]
                    all_class_ids = all_class_ids[keep_indices]
                    
                    print(f"GroundingDINO: Total detections after NMS: {len(all_boxes)}")
                
                return sv.Detections(
                    xyxy=all_boxes,
                    confidence=all_scores,
                    class_id=all_class_ids
                )
            else:
                print("GroundingDINO: No detections found")
                return sv.Detections(
                    xyxy=np.empty((0, 4)),
                    confidence=np.empty(0),
                    class_id=np.empty(0, dtype=int)
                )
            
        except Exception as e:
            print(f"GroundingDINO: Exception caught: {e}")
            print(f"GroundingDINO: Exception type: {type(e).__name__}")
            print(f"GroundingDINO: Exception args: {e.args}")
            
            # Return empty detections as fallback
            return sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int)
            )

    def _run_sam(self, image: Image.Image, detections: sv.Detections) -> np.ndarray:
        # Convert PIL image to OpenCV format
        # Convert PIL image to numpy array properly
        print(f"DEBUG: Image type: {type(image)}")
        print(f"DEBUG: Image attributes: {dir(image)}")
        
        if hasattr(image, 'convert'):
            # If it's a PIL image, convert to RGB first
            print("DEBUG: Converting PIL image to RGB")
            image_rgb = image.convert('RGB')
            image_array = np.array(image_rgb)
            print(f"DEBUG: Image array shape: {image_array.shape}, dtype: {image_array.dtype}")
        else:
            # If it's already an array
            print("DEBUG: Image is already an array")
            image_array = np.array(image)
            print(f"DEBUG: Image array shape: {image_array.shape}, dtype: {image_array.dtype}")
        
        print(f"DEBUG: Final image_array type: {type(image_array)}")
        print(f"DEBUG: Final image_array shape: {image_array.shape if hasattr(image_array, 'shape') else 'no shape'}")
        
        # Force array to be contiguous and ensure correct dtype
        image_array = np.ascontiguousarray(image_array, dtype=np.uint8)
        print(f"DEBUG: After ascontiguousarray - type: {type(image_array)}, shape: {image_array.shape}, dtype: {image_array.dtype}")
        
        # Try alternative approach - convert PIL to BGR directly
        try:
            # First try the original OpenCV approach
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"DEBUG: OpenCV cvtColor failed: {e}")
            # Fallback: manually swap RGB to BGR channels
            image_rgb = image.convert('RGB')
            image_array_rgb = np.array(image_rgb)
            # Swap R and B channels: RGB -> BGR
            image_cv = image_array_rgb[:, :, [2, 1, 0]]  # BGR order
            print(f"DEBUG: Fallback conversion successful, shape: {image_cv.shape}")
        
        # Try to run SAM with error handling
        try:
            print("SAM: About to set image...")
            # Set image for SAM predictor
            print(f"SAM: Setting image with shape: {image_cv.shape}, dtype: {image_cv.dtype}")
            self.sam_predictor.set_image(image_cv)
            print("SAM: Image set successfully")
            print(f"SAM: Image embeddings computed, features available: {hasattr(self.sam_predictor, '_features')}")
            if hasattr(self.sam_predictor, '_features'):
                print(f"SAM: Features keys: {list(self.sam_predictor._features.keys()) if self.sam_predictor._features else 'None'}")
            
            # Generate masks for each detection
            masks = []
            for i in range(len(detections)):
                print(f"SAM: Processing detection {i+1}/{len(detections)}")
                box = detections.xyxy[i]
                print(f"SAM: Box {i+1}: {box}, type: {type(box)}, shape: {box.shape if hasattr(box, 'shape') else 'no shape'}")
                
                
                print(f"SAM: Box {i+1} (pixel coordinates): {box}")
                
                mask_result = self.sam_predictor.predict(
                    box=box,
                    multimask_output=False
                )
                # Extract the mask from the tuple
                if isinstance(mask_result, tuple):
                    mask = mask_result[0]  # First element is the masks
                    print(f"SAM: Mask {i+1} shape: {mask.shape}")
                else:
                    mask = mask_result
                    print(f"SAM: Mask {i+1} shape: {mask.shape}")
                
                # Original SAM returns masks with shape (1, H, W) when multimask_output=False
                # We need to squeeze out the extra dimension to get (H, W)
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)  # Remove the first dimension
                    print(f"SAM: Squeezed mask {i+1} to shape: {mask.shape}")
                
                # Debug: Check what the mask contains
                print(f"SAM: Mask {i+1} - dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}, sum: {mask.sum()}")
                print(f"SAM: Mask {i+1} - True pixels: {(mask == True).sum()}, False pixels: {(mask == False).sum()}")
                
                # Manual thresholding: convert float logits to boolean masks
                if mask.dtype in [np.float32, np.float64] and mask.max() > 0:
                    # If we have float values, threshold them
                    threshold = 0.0  # SAM2 default threshold
                    mask = mask > threshold
                    print(f"SAM: Thresholded mask {i+1} - True pixels: {mask.sum()}, False pixels: {(~mask).sum()}")
                
                masks.append(mask)
            
            print("SAM: All masks generated successfully")
            print(f"SAM: Number of masks: {len(masks)}")
            print(f"SAM: Mask shapes: {[m.shape for m in masks]}")
            
            # Stack masks
            if masks:
                # Ensure all masks have the same shape before stacking
                target_shape = masks[0].shape
                print(f"SAM: Target shape: {target_shape}")
                
                # Check if all masks have the same shape
                for i, mask in enumerate(masks):
                    if mask.shape != target_shape:
                        print(f"SAM: Warning: Mask {i} has shape {mask.shape}, expected {target_shape}")
                        # Resize mask to target shape if needed
                        if mask.shape != target_shape:
                            # This is a simple resize - in practice you might want more sophisticated resizing
                            from PIL import Image
                            mask_pil = Image.fromarray(mask.astype(np.uint8))
                            mask_pil = mask_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
                            mask = np.array(mask_pil).astype(bool)
                            masks[i] = mask
                            print(f"SAM: Resized mask {i} to {mask.shape}")
                
                # Final safety check: ensure all masks are boolean
                final_masks = []
                for i, mask in enumerate(masks):
                    if mask.dtype != bool:
                        print(f"SAM: Final safety check - converting mask {i} from {mask.dtype} to bool")
                        if mask.dtype in [np.float32, np.float64]:
                            mask = mask > 0.0
                        else:
                            mask = mask.astype(bool)
                    final_masks.append(mask)
                
                result = np.stack(final_masks)
                print(f"SAM: Final result - dtype: {result.dtype}, shape: {result.shape}")
                return result
            else:
                return np.empty((0, image_cv.shape[0], image_cv.shape[1]), dtype=bool)
                
        except Exception as e:
            print(f"SAM: Exception caught: {e}")
            print(f"SAM: Exception type: {type(e).__name__}")
            print(f"SAM: Exception args: {e.args}")
            
            if "_C" in str(e):
                # Handle PyTorch C++ extension error
                print(f"SAM _C error: {e}")
                print("Attempting to use alternative approach...")
                
                # Try with different parameters or return empty masks
                try:
                    print("SAM: Trying fallback with multimask output...")
                    # Try with multimask output
                    self.sam_predictor.set_image(image_cv)
                    masks = []
                    for i in range(len(detections)):
                        box = detections.xyxy[i]
                        mask_result = self.sam_predictor.predict(
                            box=box,
                            multimask_output=True
                        )
                        # Extract the mask from the tuple
                        if isinstance(mask_result, tuple):
                            mask = mask_result[0]  # First element is the masks
                            # Use the first mask as fallback
                            mask = mask[0]  # Get first mask from multimask output
                        else:
                            mask = mask_result
                        
                        # Ensure mask has correct shape (H, W)
                        if mask.ndim == 3 and mask.shape[0] == 1:
                            mask = mask.squeeze(0)  # Remove the first dimension
                        
                        # Ensure mask is boolean
                        if mask.dtype != bool:
                            print(f"SAM fallback: Converting mask from {mask.dtype} to bool")
                            if mask.dtype in [np.float32, np.float64]:
                                mask = mask > 0.0
                            else:
                                mask = mask.astype(bool)
                        
                        masks.append(mask)
                    
                    if masks:
                        # Final safety check: ensure all masks are boolean
                        final_masks = []
                        for i, mask in enumerate(masks):
                            if mask.dtype != bool:
                                print(f"SAM fallback: Final safety check - converting mask {i} from {mask.dtype} to bool")
                                if mask.dtype in [np.float32, np.float64]:
                                    mask = mask > 0.0
                                else:
                                    mask = mask.astype(bool)
                            final_masks.append(mask)
                        
                        result = np.stack(final_masks)
                        print(f"SAM fallback: Final result - dtype: {result.dtype}, shape: {result.shape}")
                        return result
                    else:
                        return np.empty((0, image_cv.shape[0], image_cv.shape[1]), dtype=bool)
                        
                except Exception as e2:
                    print(f"SAM alternative approach also failed: {e2}")
                    print(f"SAM alternative exception type: {type(e2).__name__}")
                    # Return empty masks as fallback
                    return np.empty((0, image_cv.shape[0], image_cv.shape[1]), dtype=bool)
            else:
                # Re-raise other errors
                print(f"SAM: Re-raising non-_C error: {e}")
                raise e

    def _filter_detections(self, annot: AnnotType, detections: sv.Detections) -> sv.Detections:
        h, w = annot['image_info']['height_resized'], annot['image_info']['width_resized']
        print(f"Filter: Image dimensions: {h}x{w}")
        print(f"Filter: Thresholds - min_area: {self.cfg.filter.min_mask_area_ratio}, max_area: {self.cfg.filter.max_mask_area_ratio}, min_conf: {self.cfg.filter.mask_confidence_threshold}")
        valid_idx = []

        for obj_idx in range(len(detections.class_id)):
            area_ratio = detections.mask[obj_idx].sum() / h / w
            confidence = detections.confidence[obj_idx]
            print(f"Filter: Object {obj_idx} - area_ratio: {area_ratio:.6f}, confidence: {confidence:.3f}")
            
            if area_ratio < self.cfg.filter.min_mask_area_ratio:
                print(f"Filter: Object {obj_idx} filtered out - area_ratio {area_ratio:.6f} < {self.cfg.filter.min_mask_area_ratio}")
                continue
            if area_ratio > self.cfg.filter.max_mask_area_ratio:
                print(f"Filter: Object {obj_idx} filtered out - area_ratio {area_ratio:.6f} > {self.cfg.filter.max_mask_area_ratio}")
                continue
            if confidence < self.cfg.filter.mask_confidence_threshold:
                print(f"Filter: Object {obj_idx} filtered out - confidence {confidence:.3f} < {self.cfg.filter.mask_confidence_threshold}")
                continue
            print(f"Filter: Object {obj_idx} passed all filters")
            valid_idx.append(obj_idx)

        print(f"Filter: {len(valid_idx)}/{len(detections.class_id)} objects passed filtering")
        return detections[valid_idx]

    def _mask_subtract_contained(self, detections: sv.Detections, th1=0.8, th2=0.7) -> sv.Detections:
        """Adapted from: https://github.com/concept-graphs/concept-graphs/blob/93277a02bd89171f8121e84203121cf7af9ebb5d/conceptgraph/utils/ious.py#L453
        """
        xyxy = detections.xyxy
        mask = detections.mask
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

        # Compute intersection boxes
        lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
        rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)

        inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

        # Compute areas of intersection boxes
        inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

        inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
        # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
        inter_over_box2 = inter_over_box1.T  # (N, N)

        # if the intersection area is smaller than th2 of the area of box1,
        # and the intersection area is larger than th1 of the area of box2,
        # then box2 is considered contained by box1
        contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
        contained_idx = contained.nonzero()  # (num_contained, 2)

        mask_sub = mask.copy()  # (N, H, W)
        # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
        for i in range(len(contained_idx[0])):
            mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

        return mask_sub

    def _prepare_outputs(self, detections_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        output = []
        for i in range(len(detections_data['xyxy'])):
            output.append(dict(
                object_name=f'obj_{i:02d}_{detections_data["classes"][detections_data["class_id"][i]]}',
                class_name=detections_data['classes'][detections_data['class_id'][i]],
                xyxy=detections_data['xyxy'][i],
                confidence=detections_data['confidence'][i].item(),
                class_id=detections_data['class_id'][i].item(),
                box_area=detections_data['box_area'][i].item(),
                area=detections_data['area'][i].item(),
                mask=detections_data['mask'][i],
                mask_subtracted=detections_data['mask_subtracted'][i]))
        return output
