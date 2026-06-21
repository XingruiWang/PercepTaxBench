#!/usr/bin/env python3
"""
Image Quality Evaluation Script using Google Gemini API
Evaluates images in the simulation dataset based on scene richness, 
composition, lighting, and rendering realism.

Features:
- Multi-API parallel processing
- Robust error handling
- Automatic retry with different APIs
- Resume capability
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from PIL import Image
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback
from datetime import datetime

EVAL_PROMPT = """
You are an image quality evaluator for a visual dataset.
Given an image, evaluate the following **four criteria**, each scored on a **0–10 scale**.
Then compute the final quality score as the **average** of the four sub-scores.

Evaluation criteria:

1. **Scene richness (0–10):** How many distinct objects or elements are visible in the scene? 
   Reward images with diverse, detailed, and well-populated environments.
2. **Camera composition (0–10):** Is the camera framing natural, balanced, and visually pleasing?
   Reward harmonious perspectives, good spatial layout, and avoidance of awkward cropping.
3. **Lighting and exposure (0–10):** Is the lighting natural and exposure balanced?
   Penalize overexposure, darkness, or unrealistic lighting.
4. **Rendering realism / clarity (0–10):** Are textures, reflections, and geometry realistic and sharp?
   Reward consistent materials, focus, and photorealistic shading.

Output strictly in **JSON** with the following format:

{
  "SceneRichness": <float 0–10>,
  "Composition": <float 0–10>,
  "LightingExposure": <float 0–10>,
  "RealismClarity": <float 0–10>,
  "FinalScore": <average of the four values, rounded to one decimal>
}
"""

# Default API keys
DEFAULT_API_KEYS = [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY")
]


class ImageQualityEvaluator:
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash-lite"):
        """Initialize the evaluator with multiple Gemini API keys."""
        self.api_keys = api_keys
        self.model_name = model_name
        self.clients = []
        
        # Initialize clients for each API key
        for api_key in api_keys:
            try:
                client = genai.Client(api_key=api_key)
                self.clients.append(client)
            except Exception as e:
                print(f"Warning: Failed to initialize client with API key: {e}")
        
        if not self.clients:
            raise ValueError("No valid API clients could be initialized")
        
        print(f"Initialized {len(self.clients)} API clients")
        
        self.results = {}
        self.errors = {}
        self.lock = Lock()
        self.api_index = 0
        
    def find_all_images(self, base_dir: str, image_name: str = "lit.png") -> List[str]:
        """Find all images to evaluate in the directory tree."""
        base_path = Path(base_dir)
        image_paths = []
        
        # Walk through directory to find all lit.png files
        for img_path in base_path.rglob(image_name):
            image_paths.append(str(img_path))
        
        print(f"Found {len(image_paths)} images to evaluate")
        return sorted(image_paths)
    
    def extract_scene_info(self, image_path: str) -> Dict[str, str]:
        """Extract scene name and room ID from image path."""
        # Example path: .../simulationImage/jiawei/1940Office/l000_r001/lit.png
        parts = Path(image_path).parts
        
        scene_name = "unknown"
        room_id = "unknown"
        user_dir = "unknown"
        
        # Find scene name and room ID
        for i, part in enumerate(parts):
            if part == "simulationImage" and i + 3 < len(parts):
                user_dir = parts[i + 1]
                scene_name = parts[i + 2]
                room_id = parts[i + 3]
                break
        
        return {
            "user_dir": user_dir,
            "scene_name": scene_name,
            "room_id": room_id,
            "full_path": image_path
        }
    
    def get_next_client(self) -> genai.Client:
        """Get the next API client in round-robin fashion."""
        with self.lock:
            client = self.clients[self.api_index % len(self.clients)]
            self.api_index += 1
            return client
    
    def evaluate_image(self, image_path: str, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Evaluate a single image using Gemini API with retry logic.
        Returns: (result_dict, error_message)
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Get next client
                client = self.get_next_client()
                
                # Load image
                img = Image.open(image_path)
                # resize the image to 1/4
                img = img.resize((img.width // 4, img.height // 4))
                
                # Call Gemini API
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        EVAL_PROMPT,
                        img,
                    ]
                )
                
                # Parse JSON response
                response_text = response.text.strip()
                
                # Try to extract JSON from markdown code blocks if present
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                elif '```' in response_text:
                    # Try to extract from any code block
                    json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(1)
                
                # Parse JSON
                result = json.loads(response_text)
                
                # Validate required fields
                required_fields = ["SceneRichness", "Composition", "LightingExposure", 
                                 "RealismClarity", "FinalScore"]
                if not all(field in result for field in required_fields):
                    last_error = f"Missing required fields in response"
                    continue
                
                # Add metadata
                scene_info = self.extract_scene_info(image_path)
                result.update(scene_info)
                result['timestamp'] = datetime.now().isoformat()
                result['attempt'] = attempt + 1
                
                return result, None
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        # All retries failed
        return None, last_error
    
    def process_single_image(self, img_path: str) -> Tuple[str, Optional[Dict], Optional[str]]:
        """Process a single image. Returns (path, result, error)."""
        result, error = self.evaluate_image(img_path)
        return img_path, result, error
    
    def evaluate_all(self, base_dir: str, output_file: str, 
                    resume: bool = True, num_workers: int = 3,
                    max_images: Optional[int] = None,
                    save_interval: int = 10):
        """Evaluate all images in parallel and save results."""
        
        # Load existing results if resuming
        error_file = output_file.replace('.json', '_errors.json')
        if resume and os.path.exists(output_file):
            print(f"Resuming from existing file: {output_file}")
            with open(output_file, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded {len(self.results)} existing evaluations")
            
            # Load error log if exists
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    self.errors = json.load(f)
                print(f"Loaded {len(self.errors)} error records")
        
        # Find all images
        image_paths = self.find_all_images(base_dir)
        
        if max_images:
            image_paths = image_paths[:max_images]
            print(f"Limiting to first {max_images} images for testing")
        
        # Filter out already processed images
        pending_images = [p for p in image_paths if p not in self.results]
        
        total = len(image_paths)
        already_done = len(image_paths) - len(pending_images)
        
        print(f"\n{'='*60}")
        print(f"Total images: {total}")
        print(f"Already processed: {already_done}")
        print(f"To process: {len(pending_images)}")
        print(f"Workers: {num_workers}")
        print(f"{'='*60}\n")
        
        if not pending_images:
            print("All images already processed!")
            return
        
        # Process images in parallel
        evaluated = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, img_path): img_path 
                for img_path in pending_images
            }
            
            # Process completed tasks
            for future in as_completed(future_to_path):
                img_path, result, error = future.result()
                
                with self.lock:
                    if result:
                        self.results[img_path] = result
                        evaluated += 1
                        print(f"✓ [{evaluated + failed}/{len(pending_images)}] Score: {result['FinalScore']:.1f} - {Path(img_path).parent.name}/{Path(img_path).name}")
                    else:
                        self.errors[img_path] = {
                            'error': error,
                            'timestamp': datetime.now().isoformat()
                        }
                        failed += 1
                        print(f"✗ [{evaluated + failed}/{len(pending_images)}] Failed: {Path(img_path).parent.name}/{Path(img_path).name} - {error}")
                    
                    # Save periodically
                    if (evaluated + failed) % save_interval == 0:
                        self.save_results(output_file)
                        self.save_errors(error_file)
                        print(f"→ Checkpoint saved ({len(self.results)} results, {len(self.errors)} errors)")
                    
                    # Progress update
                    if (evaluated + failed) % 50 == 0:
                        print(f"\n{'='*60}")
                        print(f"Progress: {evaluated + failed}/{len(pending_images)}")
                        print(f"Success: {evaluated}, Failed: {failed}")
                        print(f"Success rate: {evaluated/(evaluated+failed)*100:.1f}%")
                        print(f"{'='*60}\n")
        
        # Final save
        self.save_results(output_file)
        self.save_errors(error_file)
        
        print(f"\n{'='*60}")
        print(f"Evaluation complete!")
        print(f"Total processed: {evaluated + failed}")
        print(f"Successful: {evaluated}")
        print(f"Failed: {failed}")
        print(f"Success rate: {evaluated/(evaluated+failed)*100:.1f}%")
        print(f"Total in database: {len(self.results)}")
        print(f"Results saved to: {output_file}")
        print(f"Errors saved to: {error_file}")
        print(f"{'='*60}")
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def save_errors(self, error_file: str):
        """Save error log to JSON file."""
        if not self.errors:
            return
        os.makedirs(os.path.dirname(error_file), exist_ok=True)
        with open(error_file, 'w') as f:
            json.dump(self.errors, f, indent=2)
    
    def generate_summary(self, output_file: str):
        """Generate summary statistics."""
        if not self.results:
            print("No results to summarize")
            return
        
        scores = [r['FinalScore'] for r in self.results.values() if 'FinalScore' in r]
        
        if not scores:
            print("No valid scores found")
            return
        
        summary = {
            "total_images": len(scores),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": {
                "excellent (8-10)": len([s for s in scores if s >= 8]),
                "good (6-8)": len([s for s in scores if 6 <= s < 8]),
                "fair (4-6)": len([s for s in scores if 4 <= s < 6]),
                "poor (0-4)": len([s for s in scores if s < 4])
            }
        }
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total images: {summary['total_images']}")
        print(f"Mean score: {summary['mean_score']:.2f}")
        print(f"Min score: {summary['min_score']:.1f}")
        print(f"Max score: {summary['max_score']:.1f}")
        print("\nScore distribution:")
        for category, count in summary['score_distribution'].items():
            pct = count / summary['total_images'] * 100
            print(f"  {category}: {count} ({pct:.1f}%)")
        print("="*60)
        
        # Save summary
        summary_file = output_file.replace('.json', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate image quality using Google Gemini API"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/path/to/Taxonomy/Data/simulationImage',
        help='Base directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/path/to/Taxonomy/Data/SimulationMetadata/scenes/image_quality_ratings.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--api_keys',
        type=str,
        nargs='+',
        default=None,
        help='Google API keys (space separated). If not provided, uses default keys.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash-lite',
        help='Gemini model to use'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from existing output file'
    )
    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Start fresh (ignore existing output)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, one per API key)'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=30,
        help='Save checkpoint every N images (default: 30)'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum number of images to evaluate (for testing)'
    )
    parser.add_argument(
        '--image_name',
        type=str,
        default='lit.png',
        help='Name of image files to evaluate (default: lit.png)'
    )
    
    args = parser.parse_args()
    
    # Get API keys
    api_keys = args.api_keys or DEFAULT_API_KEYS
    if not api_keys:
        print("Error: API keys required. Provide --api_keys or use default keys")
        return 1
    
    print(f"Using {len(api_keys)} API keys")
    
    # Create evaluator
    print(f"Initializing evaluator with model: {args.model}")
    try:
        evaluator = ImageQualityEvaluator(api_keys, args.model)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return 1
    
    # Run evaluation
    evaluator.evaluate_all(
        args.base_dir,
        args.output,
        resume=args.resume,
        num_workers=args.num_workers,
        max_images=args.max_images,
        save_interval=args.save_interval
    )
    
    # Generate summary
    evaluator.generate_summary(args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())

