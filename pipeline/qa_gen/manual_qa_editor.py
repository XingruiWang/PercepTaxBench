#!/usr/bin/env python3
"""
Manual QA Editor for Taxonomy QA Benchmark (Multi-User)

This script allows manual addition of QA pairs for images from the real image benchmark.
It provides a clear UI showing bbox objects and supports multiple concurrent users.

Usage:
    python manual_qa_editor.py
"""

import os
from pathlib import Path
gradio_temp_dir = Path(__file__).parent / "gradio_temp"
gradio_temp_dir.mkdir(exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)

import gradio as gr
import json
from typing import Dict, List, Tuple, Any, Optional
import datetime
import uuid
from collections import Counter
from PIL import Image

# Import question type grouping utility
import sys
sys.path.append(str(Path(__file__).parent / "scripts" / "modules" / "qa_modules"))
from question_type_grouping import get_simplified_question_type

# Configuration
IMAGES_DIR = Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_v2/images")
QUESTIONS_PATH = Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_v2/all_questions.json")
RESULTS_DIR = Path("/path/to/SpatialReasonerDataGen/qa_gen/additional_qa_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Color order must match visualization_utils.py
COLOR_NAMES = ["Red", "Green", "Blue", "Yellow", "Orange", "Pink", "Purple", "Magenta", "Cyan", "Rose", "Violet", "Turquoise"]

# Question category handling
NEW_CATEGORY_CHOICES = [
    "taxonomy_description",
    "taxonomy_reasoning",
    "spatial_relation",
    "other",
]

LEGACY_CATEGORY_CHOICES = [
    "affordance",
    "capability",
    "compositional",
    "counterfactual",
    "description",
    "function",
    "latent",
    "material",
    "repurposing",
    "spatial",
    "other",
]

# Normalise legacy category labels to current taxonomy
CATEGORY_NORMALIZATION_MAP = {
    "spatial": "spatial_relation",
    "": None,
}

ALL_CATEGORY_CHOICES = list(dict.fromkeys(NEW_CATEGORY_CHOICES + LEGACY_CATEGORY_CHOICES))


def normalize_question_category_value(category: Optional[str]) -> Optional[str]:
    """Normalize question category to align with current benchmark labels."""
    if category is None:
        return None
    normalized = category.strip()
    if not normalized:
        return None
    normalized = CATEGORY_NORMALIZATION_MAP.get(normalized, normalized)
    if normalized is None:
        return None
    if normalized not in ALL_CATEGORY_CHOICES and normalized != "other":
        return "other"
    return normalized

# Load existing questions ONLY to get object mappings for images
def load_existing_questions():
    """Load existing questions to extract object mappings"""
    if QUESTIONS_PATH.exists():
        with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions
    return []

def get_image_directories():
    """Get image directory paths that are actually in the benchmark"""
    if not IMAGES_DIR.exists():
        return []
    
    # Get set of image IDs that are actually in the benchmark
    benchmark_image_ids = set()
    if QUESTIONS_PATH.exists():
        with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            benchmark_image_ids = set(q.get('image_id') for q in questions if q.get('image_id'))
    
    # Only include directories that:
    # 1. Have bbox.jpg (actual image directories)
    # 2. Are in the benchmark (have questions generated)
    image_dirs = []
    for d in IMAGES_DIR.iterdir():
        if (d.is_dir() and 
            (d / "bbox.jpg").exists() and 
            d.name in benchmark_image_ids):
            image_dirs.append(d)
    
    return sorted(image_dirs)

def get_objects_for_image(image_id: str, existing_questions: List[Dict]) -> Tuple[List[str], Dict[str, str]]:
    """Get objects and box_to_object mapping for an image from existing questions (only for object info, not QA)"""
    image_questions = [q for q in existing_questions if q.get('image_id') == image_id]
    if image_questions:
        first_q = image_questions[0]
        objects = first_q.get('objects', [])
        box_to_object = first_q.get('box_to_object', {})
        # If box_to_object is empty, create from objects order
        if not box_to_object and objects:
            box_to_object = {f"{COLOR_NAMES[i % len(COLOR_NAMES)]} box": obj for i, obj in enumerate(objects)}
        return objects, box_to_object
    return [], {}

# Load existing questions ONLY to extract object mappings (not for displaying QA)
print("🚀 Loading existing questions for object mappings...")
existing_questions = load_existing_questions()
print(f"✅ Loaded {len(existing_questions)} existing questions (for object reference only)")

# Get all image directories
image_directories = get_image_directories()
print(f"✅ Found {len(image_directories)} image directories")

def init_user_session():
    """Initialize session state for a new user"""
    session_id = str(uuid.uuid4())[:8]
    current_image_idx = 0
    manual_questions = []  # Questions created by this user
    return {
        'session_id': session_id,
        'user_id': '',
        'current_image_idx': current_image_idx,
        'manual_questions': manual_questions,
        'image_directories': image_directories  # Shared reference
    }

def load_user_manual_questions(user_id):
    """Load manual questions for a specific user from all their files (including old timestamped ones)"""
    if not user_id or not RESULTS_DIR.exists():
        return []
    
    # Main consolidated file (new format)
    main_file = RESULTS_DIR / f"manual_qa_{user_id}.json"
    
    # Also check for old timestamped files (backward compatibility)
    old_files = list(RESULTS_DIR.glob(f"manual_qa_{user_id}_*.json"))
    
    all_questions = []
    seen_question_ids = set()
    
    # Load from main file first (if exists)
    if main_file.exists():
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for q in data.get('questions', []):
                    # Use image_id + question text as unique identifier to avoid duplicates
                    q_id = (q.get('image_id', ''), q.get('original_question', q.get('question', '')))
                    if q_id not in seen_question_ids:
                        normalized_category = normalize_question_category_value(q.get('question_category'))
                        if normalized_category != q.get('question_category'):
                            if normalized_category is None:
                                q.pop('question_category', None)
                            else:
                                q['question_category'] = normalized_category
                        all_questions.append(q)
                        seen_question_ids.add(q_id)
        except Exception as e:
            print(f"Warning: Error loading {main_file.name}: {e}")
    
    # Load from old timestamped files (for backward compatibility)
    for result_file in old_files:
        if result_file == main_file:
            continue  # Already loaded
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for q in data.get('questions', []):
                    # Avoid duplicates
                    q_id = (q.get('image_id', ''), q.get('original_question', q.get('question', '')))
                    if q_id not in seen_question_ids:
                        normalized_category = normalize_question_category_value(q.get('question_category'))
                        if normalized_category != q.get('question_category'):
                            if normalized_category is None:
                                q.pop('question_category', None)
                            else:
                                q['question_category'] = normalized_category
                        all_questions.append(q)
                        seen_question_ids.add(q_id)
        except Exception as e:
            print(f"Warning: Error loading {result_file.name}: {e}")
            continue
    
    return all_questions

def save_user_manual_question(state, question_data):
    """Save a manual question for the user to a consolidated file (replaces previous save)"""
    if not state.get('user_id'):
        return False, "Please enter your name/ID first"
    
    try:
        # Use consolidated filename (same as load function expects)
        filename = f"manual_qa_{state['user_id']}.json"
        filepath = RESULTS_DIR / filename
        
        # Load existing questions from file to merge with current state
        existing_questions = load_user_manual_questions(state['user_id'])
        
        # Add new question to the list (avoid duplicates)
        question_id = (question_data.get('image_id', ''), question_data.get('original_question', question_data.get('question', '')))
        existing_ids = {(q.get('image_id', ''), q.get('original_question', q.get('question', ''))) for q in existing_questions}
        
        if question_id in existing_ids:
            return False, f"⚠️ Duplicate question skipped (same image_id and question text already exists). You have {len(existing_questions)} questions saved."
        
        existing_questions.append(question_data)
        
        # Save all questions to consolidated file
        existing_data = {
            'user_id': state['user_id'],
            'session_id': state['session_id'],
            'timestamp': datetime.datetime.now().isoformat(),
            'total_questions': len(existing_questions),
            'questions': existing_questions
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        # Update state with all questions (including existing ones from file)
        state['manual_questions'] = existing_questions
        
        return True, f"✅ Question saved! ({len(existing_questions)} total)"
    except Exception as e:
        return False, f"❌ Error saving: {e}"

# Create Gradio interface
css = """
.qa-editor-container {
    max-width: 1600px;
    margin: 0 auto;
}
.object-display {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    border: 2px solid #ddd;
}
.object-item {
    padding: 8px;
    margin: 5px 0;
    background-color: white;
    border-left: 4px solid;
    border-radius: 3px;
}
.statistics-display {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
"""

with gr.Blocks(title="Manual QA Editor", theme=gr.themes.Soft(), css=css) as app:
    gr.Markdown("# ✏️ Manual QA Editor - Taxonomy QA Benchmark")
    gr.Markdown("Create new QA pairs for images. View the bbox image and objects, then create questions. All objects in the image are available as answer choices.")
    
    # User identification (same as multiuser_app_vN.py)
    with gr.Row():
        with gr.Column(scale=3):
            user_id_input = gr.Textbox(
                label="Step 1: Your Name/ID (for saving your questions)",
                placeholder="Enter your name or ID to save and resume your work",
                value=""
            )
        with gr.Column(scale=1):
            load_user_btn = gr.Button("🔄 Load My Questions", variant="secondary")
            user_status = gr.Textbox(label="Status", interactive=False, visible=True)
    
    # Session state
    user_state = gr.State(value=init_user_session())
    
    # Image navigation
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Image Navigation")
            with gr.Row():
                prev_image_btn = gr.Button("◀ Previous Image", size="sm")
                next_image_btn = gr.Button("Next Image ▶", size="sm")
                jump_to_image_input = gr.Textbox(
                    label="Jump to Image ID",
                    placeholder="Enter image ID (e.g., 00019e09fe61747c)",
                    scale=3
                )
                jump_image_btn = gr.Button("🎯 Go", size="sm", scale=1)
            
            image_progress = gr.Textbox(label="Progress", value="Image 1 of 0", interactive=False)
            
            # Image display
            image_display = gr.Image(
                label="Image with Bounding Boxes",
                type="pil",
                interactive=False,
                height=600
            )
            
        with gr.Column(scale=1):
            # Objects display
            gr.Markdown("### 📦 Objects in Image")
            objects_display = gr.Markdown("**No objects loaded**")
            
            # Current image info
            current_image_info = gr.Textbox(
                label="Current Image Info",
                interactive=False,
                lines=3
            )
    
    # QA Creation Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ➕ Create New QA Pair")
            
            with gr.Row():
                question_category = gr.Dropdown(
                    choices=ALL_CATEGORY_CHOICES,
                    label="Step 2: Choose Question Category",
                    value=None,
                    info="choose one that best fit the QA you are creating if no fit then select 'other'"
                )
                question_type_input = gr.Textbox(
                    label="Question Type (optional)",
                    placeholder="e.g., material_property, affordance_furniture"
                )
            
            question_text = gr.Textbox(
                label="Step 3: Question Text",
                placeholder="e.g., Which object is made of 'wood'?",
                lines=2
            )
            
            with gr.Row():
                answer_object_dropdown = gr.Dropdown(
                    label="Step 4: Choose Correct Answer Object (from list)",
                    choices=[],
                    value=None,
                    info="Select from existing objects, or enter manually below",
                    scale=2
                )
                answer_object_manual = gr.Textbox(
                    label="Or Enter Object Name Manually",
                    placeholder="e.g., chair",
                    info="Use if object not in dropdown",
                    scale=2
                )
            
            reasoning_text = gr.Textbox(
                label="Step 5: Reasoning (optional)",
                placeholder="Chain-of-thought reasoning explaining the answer",
                lines=4
            )
            
            with gr.Row():
                create_qa_btn = gr.Button("✅ Create QA Pair", variant="primary", size="lg")
                clear_form_btn = gr.Button("🗑️ Clear Form", variant="secondary")
            
            create_status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            # User's manual questions
            gr.Markdown("### 📋 Your Manual Questions")
            user_questions_display = gr.Markdown("**No questions created yet**")
            refresh_user_questions_btn = gr.Button("🔄 Refresh My Questions", size="sm")
            save_all_btn = gr.Button("💾 Save All Questions", variant="stop")
            save_status = gr.Textbox(label="Save Status", interactive=False, visible=False)
    
    # Statistics
    with gr.Row():
        stats_display = gr.Markdown("**Statistics will appear here**")
    
    # Helper functions
    def get_current_image_data(state):
        """Get current image data"""
        if not state or not state['image_directories'] or state['current_image_idx'] >= len(state['image_directories']):
            return None, "No image", [], {}, "", f"Image 0 of {len(state['image_directories']) if state else 0}"
        
        image_dir = state['image_directories'][state['current_image_idx']]
        image_id = image_dir.name
        bbox_path = image_dir / "bbox.jpg"
        
        # Load image with PIL (like multiuser_app_vN.py does)
        image = None
        if bbox_path.exists():
            try:
                image = Image.open(bbox_path)
            except Exception as e:
                print(f"Error loading image: {e}")
        
        if not image:
            return None, f"Image {state['current_image_idx'] + 1} of {len(state['image_directories'])}", [], {}, "", ""
        
        # Get objects from existing questions
        objects, box_to_object = get_objects_for_image(image_id, existing_questions)
        
        # Format objects display - show objects present in the scene with their bbox colors
        if objects and box_to_object:
            objects_text = "**Objects in Scene (Bounding Box Colors):**\n\n"
            # Sort by color order
            sorted_boxes = sorted(box_to_object.items(), key=lambda x: COLOR_NAMES.index(x[0].split()[0]) if x[0].split()[0] in COLOR_NAMES else 999)
            for box_name, obj_name in sorted_boxes:
                objects_text += f"🔲 **{box_name}** → `{obj_name}`\n"
        elif objects:
            objects_text = "**Objects in Scene (color order):**\n\n"
            for i, obj in enumerate(objects):
                box_name = f"{COLOR_NAMES[i % len(COLOR_NAMES)]} box"
                objects_text += f"🔲 **{box_name}** → `{obj}`\n"
        else:
            objects_text = "**No objects found for this image.**\n\nYou can still create QA pairs by manually entering object names. The answer object will be used to determine the colored box."
        
        info_text = f"**Image ID:** {image_id}\n**Image Path:** {image_id}/bbox.jpg\n**Objects Count:** {len(objects)}"
        
        progress_text = f"Image {state['current_image_idx'] + 1} of {len(state['image_directories'])}"
        
        return image, objects_text, objects, box_to_object, info_text, progress_text
    
    def navigate_image(state, direction, target_image_id=None):
        """Navigate between images"""
        if state is None:
            state = init_user_session()
        
        if direction == "next" and state['current_image_idx'] < len(state['image_directories']) - 1:
            state['current_image_idx'] += 1
        elif direction == "previous" and state['current_image_idx'] > 0:
            state['current_image_idx'] -= 1
        elif direction == "jump" and target_image_id:
            # Find image by ID
            for idx, img_dir in enumerate(state['image_directories']):
                if img_dir.name == target_image_id.strip():
                    state['current_image_idx'] = idx
                    break
        
        image, objects_text, objects, box_to_object, info_text, progress_text = get_current_image_data(state)
        
        # Update answer object dropdown with box colors
        answer_choices = []
        if objects and box_to_object:
            # Create mapping from object to box color
            object_to_box = {obj: box for box, obj in box_to_object.items()}
            for obj in objects:
                box_color = object_to_box.get(obj, "")
                if box_color:
                    # Format: "Red box - market stall"
                    answer_choices.append(f"{box_color} - {obj}")
                else:
                    answer_choices.append(obj)
        elif objects:
            answer_choices = objects
        
        return state, image, objects_text, info_text, progress_text, gr.update(choices=answer_choices, value=None)
    
    def load_user_data(state, user_id):
        """Load user's previous manual questions"""
        if not user_id or not user_id.strip():
            return state, "⚠️ Please enter a name/ID first", gr.update(value="**No questions loaded**")
        
        user_id = user_id.strip()
        state['user_id'] = user_id
        state['manual_questions'] = load_user_manual_questions(user_id)
        
        status_msg = f"✅ Loaded {len(state['manual_questions'])} previous questions for: {user_id}"
        questions_text = format_user_questions(state['manual_questions'])
        
        return state, status_msg, gr.update(value=questions_text)
    
    def format_user_questions(questions):
        """Format user's questions for display"""
        if not questions:
            return "**No questions created yet**"
        
        text = f"**Your Questions ({len(questions)} total):**\n\n"
        for i, q in enumerate(questions[-10:], 1):  # Show last 10
            image_id = q.get('image_id', 'unknown')
            question_text = q.get('question', '')[:60] + "..." if len(q.get('question', '')) > 60 else q.get('question', '')
            text += f"{len(questions) - 10 + i}. [{image_id}] {question_text}\n"
        
        if len(questions) > 10:
            text += f"\n... and {len(questions) - 10} more"
        
        return text
    
    def create_qa_pair(state, user_id_input_val, question_text, answer_object_dropdown, answer_object_manual, question_category, question_type, reasoning):
        """Create a new QA pair"""
        if state is None:
            state = init_user_session()
        
        # Auto-update user_id from input field if provided
        if user_id_input_val and user_id_input_val.strip():
            state['user_id'] = user_id_input_val.strip()
            # Load previous questions if they exist (but don't require it)
            if not state.get('manual_questions'):
                state['manual_questions'] = load_user_manual_questions(state['user_id'])
        
        if not state.get('user_id'):
            return state, "⚠️ Please enter your name/ID first", gr.update(value="**No questions created yet**")
        
        if not question_text.strip():
            return state, "❌ Question text is required", gr.update(value="**No questions created yet**")
        
        # Get answer object from dropdown or manual input
        # Extract object name if dropdown value includes box color (format: "Red box - market stall")
        if answer_object_dropdown:
            if " - " in answer_object_dropdown:
                answer_object = answer_object_dropdown.split(" - ", 1)[1]
            else:
                answer_object = answer_object_dropdown
        else:
            answer_object = answer_object_manual.strip()
        
        if not answer_object:
            return state, "❌ Please select or enter an answer object", gr.update(value="**No questions created yet**")
        
        # Auto-derive question_category from question_type if provided (matches benchmark behavior)
        if question_type.strip():
            # Use get_simplified_question_type to derive category from question_type
            derived_category = get_simplified_question_type(question_type.strip())
            # Use derived category, but allow manual override if category is also provided
            if question_category:
                # Warn if they don't match, but use derived category to match benchmark
                if derived_category != question_category:
                    print(f"Warning: question_type '{question_type.strip()}' maps to '{derived_category}', but '{question_category}' was selected. Using '{derived_category}' to match benchmark.")
            question_category = derived_category
        elif not question_category:
            return state, "❌ Question category is required (or provide question_type for auto-derivation)", gr.update(value="**No questions created yet**")

        question_category = normalize_question_category_value(question_category)
        if not question_category:
            return state, "❌ Question category is required (or provide question_type for auto-derivation)", gr.update(value="**No questions created yet**")
        
        # Get current image data - objects present in the scene
        image_dir = state['image_directories'][state['current_image_idx']]
        image_id = image_dir.name
        objects, box_to_object = get_objects_for_image(image_id, existing_questions)
        
        # Ensure answer object is in the objects list
        # If objects list is empty, create it with the answer object
        if not objects:
            objects = [answer_object]
            box_to_object = {f"{COLOR_NAMES[0]} box": answer_object}
        elif answer_object not in objects:
            # Add answer object to objects list if not present
            objects.append(answer_object)
            # Assign next available color based on current count
            next_color_idx = len(objects) - 1
            box_to_object[f"{COLOR_NAMES[next_color_idx % len(COLOR_NAMES)]} box"] = answer_object
        
        # Ensure complete box_to_object mapping for all objects
        # Create mapping in color order if not already complete
        if len(box_to_object) < len(objects):
            # Rebuild mapping to ensure all objects have boxes in color order
            box_to_object = {}
            for i, obj in enumerate(objects):
                box_name = f"{COLOR_NAMES[i % len(COLOR_NAMES)]} box"
                box_to_object[box_name] = obj
        
        # Find answer object's colored box
        answer_box = None
        if box_to_object:
            for box, obj in box_to_object.items():
                if obj == answer_object:
                    answer_box = box
                    break
        
        # If not found, create from order
        if not answer_box:
            try:
                obj_idx = objects.index(answer_object)
                answer_box = f"{COLOR_NAMES[obj_idx % len(COLOR_NAMES)]} box"
            except ValueError:
                # Fallback: use first color
                answer_box = f"{COLOR_NAMES[0]} box"
                box_to_object[answer_box] = answer_object
        
        # Format question with colored boxes if needed
        formatted_question = question_text
        if "Option objects:" not in question_text and "Red box" not in question_text:
            options = [f"{COLOR_NAMES[i % len(COLOR_NAMES)]} box" for i in range(len(objects))]
            formatted_question = f"{question_text} Option objects: {', '.join(options)}"
        
        # Create question data with proper formatting
        # Ensure box_to_object mapping is complete
        complete_box_to_object = box_to_object if box_to_object else {f"{COLOR_NAMES[i % len(COLOR_NAMES)]} box": obj for i, obj in enumerate(objects)}
        
        # Format answer with colored box
        answer_with_box = answer_box
        
        # Create question data following the benchmark format
        # Ensure all fields are properly formatted for integration
        question_data = {
            "question": formatted_question,  # Question with colored box options
            "answer": answer_with_box,  # Answer as colored box (e.g., "Red box")
            "original_question": question_text,  # Original question text without colored boxes
            "original_answer": answer_object,  # Answer as object name
            "answer_object": answer_object,  # Object that is the correct answer
            "target_object": answer_object,  # Target object (same as answer for manual QA)
            "objects": objects.copy(),  # All objects present in the scene (default choices)
            "choices": objects.copy(),  # Answer choices (default: all objects in the image)
            "reasoning": reasoning.strip() if reasoning else "",  # Chain-of-thought reasoning
            "question_category": question_category,  # Simplified category
            "question_type": question_type.strip() if question_type else "",  # Detailed question type
            "image_id": image_id,  # Image identifier
            "image_path": f"{image_id}/bbox.jpg",  # Path to bbox image
            "box_to_object": complete_box_to_object.copy(),  # Complete mapping: colored box -> object name
            "created_timestamp": datetime.datetime.now().isoformat(),
            "created_by": state['user_id'],
            "is_manual": True  # Flag to indicate this is manually created
        }
        
        # Save question
        success, status_msg = save_user_manual_question(state, question_data)
        
        # Update display
        questions_text = format_user_questions(state['manual_questions'])
        
        return state, status_msg, gr.update(value=questions_text)
    
    def save_all_questions(state, user_id_input_val):
        """Save all user questions to a single file"""
        if state is None:
            state = init_user_session()
        
        # Auto-update user_id from input field if provided
        if user_id_input_val and user_id_input_val.strip():
            state['user_id'] = user_id_input_val.strip()
            # Load previous questions if they exist (but don't require it)
            if not state.get('manual_questions'):
                state['manual_questions'] = load_user_manual_questions(state['user_id'])
        
        if not state or not state.get('user_id'):
            return state, gr.update(visible=True, value="⚠️ Please enter your name/ID first")
        
        if not state['manual_questions']:
            return state, gr.update(visible=True, value="⚠️ No questions to save")
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manual_qa_{state['user_id']}_complete_{timestamp}.json"
            filepath = RESULTS_DIR / filename
            
            data = {
                'user_id': state['user_id'],
                'session_id': state['session_id'],
                'timestamp': datetime.datetime.now().isoformat(),
                'total_questions': len(state['manual_questions']),
                'questions': state['manual_questions']
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            summary = f"✅ **All questions saved!**\n\n"
            summary += f"📁 File: {filename}\n"
            summary += f"📊 Total Questions: {len(state['manual_questions'])}\n"
            
            return state, gr.update(visible=True, value=summary)
        except Exception as e:
            return state, gr.update(visible=True, value=f"❌ Error saving: {e}")
    
    def initialize_display(state):
        """Initialize display on load"""
        if state is None:
            state = init_user_session()
        
        image, objects_text, objects, box_to_object, info_text, progress_text = get_current_image_data(state)
        
        # Format answer choices with box colors
        answer_choices = []
        if objects and box_to_object:
            # Create mapping from object to box color
            object_to_box = {obj: box for box, obj in box_to_object.items()}
            for obj in objects:
                box_color = object_to_box.get(obj, "")
                if box_color:
                    # Format: "Red box - market stall"
                    answer_choices.append(f"{box_color} - {obj}")
                else:
                    answer_choices.append(obj)
        elif objects:
            answer_choices = objects
        
        return state, image, objects_text, info_text, progress_text, gr.update(choices=answer_choices, value=None), "", gr.update(value="**No questions created yet**"), ""
    
    # Event handlers
    initialize_display_handle = lambda state: initialize_display(state)
    
    app.load(
        initialize_display_handle,
        inputs=[user_state],
        outputs=[
            user_state, image_display, objects_display, current_image_info, image_progress,
            answer_object_dropdown, user_status, user_questions_display, answer_object_manual
        ]
    )
    
    prev_image_btn.click(
        lambda state: navigate_image(state, "previous"),
        inputs=[user_state],
        outputs=[user_state, image_display, objects_display, current_image_info, image_progress, answer_object_dropdown]
    )
    
    next_image_btn.click(
        lambda state: navigate_image(state, "next"),
        inputs=[user_state],
        outputs=[user_state, image_display, objects_display, current_image_info, image_progress, answer_object_dropdown]
    )
    
    jump_image_btn.click(
        lambda state, img_id: navigate_image(state, "jump", img_id),
        inputs=[user_state, jump_to_image_input],
        outputs=[user_state, image_display, objects_display, current_image_info, image_progress, answer_object_dropdown]
    )
    
    load_user_btn.click(
        load_user_data,
        inputs=[user_state, user_id_input],
        outputs=[user_state, user_status, user_questions_display]
    )
    
    create_qa_btn.click(
        create_qa_pair,
        inputs=[user_state, user_id_input, question_text, answer_object_dropdown, answer_object_manual, question_category, question_type_input, reasoning_text],
        outputs=[user_state, create_status, user_questions_display]
    )
    
    refresh_user_questions_btn.click(
        lambda state: (state, gr.update(value=format_user_questions(state.get('manual_questions', [])))),
        inputs=[user_state],
        outputs=[user_state, user_questions_display]
    )
    
    save_all_btn.click(
        save_all_questions,
        inputs=[user_state, user_id_input],
        outputs=[user_state, save_status]
    )
    
    clear_form_btn.click(
        lambda: ("", None, "", None, "", ""),
        outputs=[question_text, answer_object_dropdown, answer_object_manual, question_category, question_type_input, reasoning_text]
    )
    
    # Instructions
    with gr.Accordion("📖 Instructions", open=False):
        gr.Markdown("""
        ### How to Use
        
        1. **Enter Your Name/ID**: Enter your identifier and click "Load My Questions" to load your previous work
        
        2. **Navigate Images**: 
           - Use Previous/Next buttons to browse images
           - Or enter an image ID and click "Go" to jump to a specific image
        
        3. **View Objects**: The bbox image shows bounding boxes with colors. The objects list shows which colored box corresponds to which object name.
        
        4. **Create QA Pair**:
           - Select a **Question Category** from dropdown
           - Optionally enter a **Question Type** (e.g., `material_property`)
           - Enter your **Question Text**
           - Select the **Correct Answer Object** from the dropdown
           - Optionally add **Reasoning**
           - Click "Create QA Pair"
        
        5. **Your Questions**: View all questions you've created. Questions are auto-saved as you create them.
        
        6. **Save All**: Click "Save All Questions" to create a complete backup file with all your questions.
        
        ### Notes
        
        - Questions are automatically saved per user (identified by your name/ID)
        - Each user's questions are stored separately in `manual_qa_editor_results/`
        - The colored boxes (Red box, Green box, etc.) match the bounding box colors in the image
        - Object selection uses the existing object names from the benchmark
        """)

print("🎉 Gradio interface ready!")

if __name__ == "__main__":
    print("🌐 Launching Manual QA Editor...")
    launch_kwargs = {
        "share": True,
        "debug": True,
        "server_name": "0.0.0.0",
    }
    # Only add allowed_paths if images dir lies outside current working directory
    try:
        cwd_path = Path.cwd().resolve()
        images_path = IMAGES_DIR.resolve()
        if images_path.is_dir() and not str(images_path).startswith(str(cwd_path)):
            launch_kwargs["allowed_paths"] = [str(images_path)]
    except Exception:
        pass
    app.launch(**launch_kwargs)
