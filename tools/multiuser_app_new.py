import os
from pathlib import Path
# Set custom temp directory for Gradio BEFORE importing gradio
# Gradio reads GRADIO_TEMP_DIR during import, so this must be first
gradio_temp_dir = Path(__file__).parent / "gradio_temp"
gradio_temp_dir.mkdir(exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)

import gradio as gr
import json
from typing import Dict, List, Tuple, Any, Optional
import datetime
import uuid
from PIL import Image

# Load questions data once (shared across all users, read-only)
print("🚀 Initializing Taxonomy QA Survey System...")
DATASET_PATH = Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_v2/all_questions.json")
# RESULTS_DIR = Path("survey_results")
RESULTS_DIR = Path("./survey_results_new")
RESULTS_DIR.mkdir(exist_ok=True)

IMAGES_DIR = Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_v2/images")

# Load questions data (shared, read-only)
questions_data = []
try:
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    print(f"✅ Loaded {len(questions_data)} questions from dataset")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    questions_data = []

def load_user_assessments(user_id):
    """Load assessments for a specific user from their saved files"""
    if not user_id or not RESULTS_DIR.exists() or not (RESULTS_DIR / user_id).exists():
        return {}, None, 0

    # Main save file (used by both auto-save and manual save)
    main_save_file = RESULTS_DIR / user_id / f"taxonomy_qa_survey_summary.json"
    result_files = []
    
    # Check main save file (auto-save and manual save both use this)
    if main_save_file.exists():
        result_files.append(main_save_file)
    
    # Also check old timestamped files and old auto-save files for backward compatibility
    old_files = list((RESULTS_DIR / user_id).glob(f"*.json"))
    old_autosaves = list((RESULTS_DIR / user_id).glob(f"auto_saved/*.json"))
    result_files.extend(old_files)
    result_files.extend(old_autosaves)
    
    if not result_files:
        return {}, None, 0
    
    merged_assessments = {}
    latest_file = None
    latest_timestamp = None
    latest_index = 0
    
    # First pass: collect all file data with timestamps
    file_data_list = []
    result_files = list(set(result_files))
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this file belongs to the user
            file_user_id = data.get('user_id')
            if file_user_id != user_id:
                continue
            
            file_data_list.append({
                'file': result_file,
                'data': data,
                'timestamp': data.get('timestamp', '')
            })
        except Exception as e:
            print(f"Warning: Error loading {result_file.name}: {e}")
            continue
    
    # Sort files by timestamp (newest first) to prioritize recent saves
    file_data_list.sort(key=lambda x: x['timestamp'] or '', reverse=True)
    
    # Track the latest file by timestamp for current_index
    for file_info in file_data_list:
        file_timestamp = file_info['timestamp']
        result_file = file_info['file']
        data = file_info['data']
        
        if latest_timestamp is None or file_timestamp > latest_timestamp:
            latest_timestamp = file_timestamp
            latest_file = result_file
            latest_index = data.get('current_index', 0)
        elif file_timestamp == latest_timestamp and result_file == main_save_file:
            # Prefer main save file over auto-save if timestamps are same
            latest_file = result_file
            latest_index = data.get('current_index', 0)
    
    # Merge assessments: keep the most recent assessment for each question
    # Process files in timestamp order (newest first) so newer assessments override older ones
    for file_info in file_data_list:
        data = file_info['data']
        result_file = file_info['file']
        
        for assessment in data.get('assessments', []):
            question_idx = assessment.get('question_index')
            if question_idx is not None and question_idx < len(questions_data):
                current_question = questions_data[question_idx]
                if current_question.get('question') == assessment.get('question'):
                    assessment_timestamp = assessment.get('assessment_timestamp', '')
                    
                    # Only add/update if this assessment is newer than existing one
                    if question_idx not in merged_assessments:
                        merged_assessments[question_idx] = {
                            'quality': assessment.get('quality_assessment'),
                            'notes': assessment.get('notes', ''),
                            'timestamp': assessment_timestamp,
                            'question_index': question_idx,
                            'question_id': current_question.get('image_id', 'unknown'),
                            'loaded_from_file': result_file.name
                        }
                    else:
                        # Compare timestamps - keep the most recent assessment
                        existing_timestamp = merged_assessments[question_idx].get('timestamp', '')
                        if assessment_timestamp > existing_timestamp:
                            merged_assessments[question_idx] = {
                                'quality': assessment.get('quality_assessment'),
                                'notes': assessment.get('notes', ''),
                                'timestamp': assessment_timestamp,
                                'question_index': question_idx,
                                'question_id': current_question.get('image_id', 'unknown'),
                                'loaded_from_file': result_file.name
                            }
    
    return merged_assessments, latest_file, latest_index

def init_user_session(user_id=None):
    """Initialize session state for a user (with optional user_id for persistence)"""
    session_id = str(uuid.uuid4())[:8]
    current_index = 0
    quality_assessments = {}
    
    # If user_id provided, try to load their previous work
    if user_id:
        quality_assessments, latest_file, saved_index = load_user_assessments(user_id)
        if latest_file:
            current_index = saved_index
            # If all questions from saved position are assessed, find first unassessed
            if current_index in quality_assessments:
                for i in range(current_index, len(questions_data)):
                    if i not in quality_assessments:
                        current_index = i
                        break
                else:
                    # All questions assessed, stay at saved position
                    pass
        elif quality_assessments:
            # Has assessments but no saved position, find first unassessed
            for i in range(len(questions_data)):
                if i not in quality_assessments:
                    current_index = i
                    break
    
    return {
        'session_id': session_id,
        'user_id': user_id or "",  # User identifier for persistence
        'current_index': current_index,
        'quality_assessments': quality_assessments,
        'questions_data': questions_data  # Reference to shared data
    }

# Create Gradio interface
css = """
.question-text {
    font-size: 18px !important;
    line-height: 1.5 !important;
    margin: 20px 0 !important;
}

.answer-text {
    font-size: 16px !important;
    font-weight: bold !important;
    color: #2e7d32 !important;
    margin: 15px 0 !important;
}

.metadata-text {
    font-size: 14px !important;
    color: #666 !important;
    margin: 10px 0 !important;
}

.quality-button {
    min-width: 120px !important;
    margin: 5px !important;
}

.nav-button {
    min-width: 100px !important;
}

/* Image display */
.gradio-image {
    max-height: 400px !important;
    margin: 20px auto !important;
}
"""

with gr.Blocks(title="Taxonomy QA Quality Survey", theme=gr.themes.Soft(), css=css) as app:
    gr.Markdown("# 🔍 Taxonomy QA Dataset - Quality Assessment Survey")
    gr.Markdown("Review questions and their provided answers, then assess the quality of each Q&A pair.")
    
    # User identification section
    with gr.Row():
        with gr.Column(scale=3):
            user_id_input = gr.Textbox(
                label="Step 1:Your Name/ID (for saving your progress)",
                placeholder="Enter your name or ID to save and resume your work",
                value=""
            )
        with gr.Column(scale=1):
            load_user_btn = gr.Button("🔄 Load My Progress", variant="secondary")
            user_status = gr.Textbox(label="Status", interactive=False, visible=True)
    
    # Progress and statistics row
    with gr.Row():
        progress_text = gr.Textbox(label="Progress", value="Question 1 of 0", interactive=False)
        stats_display = gr.Markdown("**Survey Statistics**\n\nNo data loaded")
    
    # Main content area
    with gr.Row():
        with gr.Column(scale=2):
            # Question display
            question_display = gr.Markdown("", elem_classes="question-text")
            
            # Image display
            image_display = gr.Image(label="Question Image", type="pil", interactive=False)
            
            # Answer and objects
            answer_display = gr.Markdown("", elem_classes="answer-text")
            objects_display = gr.Markdown("", elem_classes="metadata-text")
            
            # Metadata
            metadata_display = gr.Markdown("", elem_classes="metadata-text")
            
            # Reasoning (collapsible)
            with gr.Accordion("Model Reasoning", open=False):
                reasoning_display = gr.Markdown("")
        
        with gr.Column(scale=1):
            # Quality assessment section
            gr.Markdown("### 🎯 Step 2:Quality Assessment")
            gr.Markdown("Is the answer correct and high quality?")
            
            quality_radio = gr.Radio(
                choices=[
                    "High Quality - Correct",
                    "Acceptable - Mostly Correct",
                    "Low Quality - Incorrect",
                    "Ambiguous/Unclear"
                ],
                label="Answer Quality",
                value=None
            )
            
            notes_input = gr.Textbox(
                label="Notes (optional)",
                placeholder="Add any observations about this Q&A pair...",
                lines=3
            )
            
            submit_assessment_btn = gr.Button("📝 Submit Assessment", variant="primary")
            assessment_status = gr.Textbox(label="Status", interactive=False)
            
            # Navigation controls
            gr.Markdown("### 🧭 Navigation")
            
            with gr.Row():
                first_btn = gr.Button("⏮ First", elem_classes="nav-button")
                prev_btn = gr.Button("◀ Previous", elem_classes="nav-button")
            
            with gr.Row():
                next_btn = gr.Button("Next ▶", elem_classes="nav-button")
                last_btn = gr.Button("Last ⏭", elem_classes="nav-button")
            
            # Jump to question
            with gr.Row():
                jump_input = gr.Number(
                    label="Jump to question #",
                    minimum=1,
                    maximum=len(questions_data) if questions_data else 1,
                    value=1,
                    precision=0
                )
                jump_btn = gr.Button("🎯 Go")
            
            # Jump to next unassessed
            next_unassessed_btn = gr.Button("⏭️ Next Unassessed", variant="secondary")
    
    # Bottom controls
    with gr.Row():
        save_btn = gr.Button("💾 Save Results", variant="stop", size="lg")
        refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
        show_unassessed_btn = gr.Button("📋 Show Unassessed")
        
        # Filter options
        filter_dropdown = gr.Dropdown(
            choices=[
                "High Quality - Correct",
                "Acceptable - Mostly Correct", 
                "Low Quality - Incorrect",
                "Ambiguous/Unclear"
            ],
            label="Filter by quality"
        )
        filter_btn = gr.Button("🔍 Show Filtered")
    
    save_status = gr.Textbox(label="Save Status", visible=False)
    filter_results = gr.Textbox(label="Filtered Results", visible=False, lines=10)
    unassessed_summary = gr.Textbox(label="Unassessed Questions", visible=False, lines=10)
    
    # Session state for each user (isolated per browser session)
    # Initialize with actual dict, not function reference
    user_state = gr.State(value=init_user_session())
    
    # Helper functions that work with state
    def format_question_display(state, question_index):
        """Format question data for display"""
        if not state or not state['questions_data'] or question_index < 0 or question_index >= len(state['questions_data']):
            return ("No more questions", "", "", "", None, "", "", "", "")
        
        question_data = state['questions_data'][question_index]
        question_text = question_data.get('question', '')
        correct_answer = question_data.get('answer', '')
        original_answer = question_data.get('original_answer', question_data.get('answer_object', ''))
        question_type = question_data.get('question_type', '')
        reasoning = question_data.get('reasoning', '')
        objects = question_data.get('objects', [])
        image_path = question_data.get('image_path', '')
        
        question_display = f"**Question {question_index + 1}/{len(state['questions_data'])}**\n\n{question_text}"
        
        box_to_object = question_data.get('box_to_object', {})
        answer_display = f"**Correct Answer:** {correct_answer}"
        if box_to_object and correct_answer in box_to_object:
            actual_object = box_to_object[correct_answer]
            answer_display += f" ({actual_object})"
        elif original_answer:
            answer_display += f" ({original_answer})"
        
        if box_to_object:
            color_order = ["Red", "Green", "Blue", "Yellow", "Orange", "Pink", "Purple", "Magenta", "Cyan", "Rose", "Violet", "Turquoise"]
            def sort_key(item):
                box_name = item[0]
                for idx, color in enumerate(color_order):
                    if box_name.startswith(color):
                        return (idx, box_name)
                return (999, box_name)
            box_object_pairs = sorted(box_to_object.items(), key=sort_key)
            objects_display = "**Objects in image (from box_to_object mapping):**\n"
            objects_display += "\n".join([f"- {box}: **{obj}**" for box, obj in box_object_pairs])
        else:
            color_names = ["red", "green", "blue", "yellow", "orange", "pink", "purple", "magenta", "cyan", "rose", "violet", "turquoise"]
            objects_display = "**Objects in image:** " + ", ".join([f"{color_names[i % len(color_names)].capitalize()} box ({obj})" for i, obj in enumerate(objects)])
            objects_display += "\n\n⚠️ **Warning:** Missing box_to_object mapping - using fallback display (may not match bbox colors)"
        
        metadata_display = f"**Question Type:** {question_type}\n**Image ID:** {question_data.get('image_id', 'N/A')}"
        
        image = None
        if image_path:
            full_image_path = Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_v2/images") / image_path
            if full_image_path.exists():
                try:
                    image = Image.open(full_image_path)
                except Exception as e:
                    print(f"Error loading image: {e}")
        
        reasoning_display = f"**Model Reasoning:**\n{reasoning}" if reasoning else ""
        progress = f"Question {question_index + 1} of {len(state['questions_data'])}"
        
        existing_assessment = state['quality_assessments'].get(question_index, {})
        quality_value = existing_assessment.get('quality', None)
        notes_value = existing_assessment.get('notes', '')
        
        if quality_value:
            progress += f" - ✅ Previously assessed: {quality_value}"
        
        return (question_display, answer_display, objects_display, metadata_display, 
                image, reasoning_display, progress, quality_value, notes_value)
    
    def get_statistics_from_state(state):
        """Get statistics from state"""
        if not state or not state['questions_data']:
            return "No dataset loaded"
        
        stats = f"**Survey Progress**\n\n"
        stats += f"📊 Total Questions: {len(state['questions_data'])}\n"
        stats += f"✅ Assessed: {len(state['quality_assessments'])}\n"
        stats += f"⏳ Remaining: {len(state['questions_data']) - len(state['quality_assessments'])}\n"
        stats += f"📍 Current: {state['current_index'] + 1}/{len(state['questions_data'])}\n\n"
        
        if state['quality_assessments']:
            stats += "**Quality Distribution:**\n"
            quality_counts = {}
            for assessment in state['quality_assessments'].values():
                quality = str(assessment['quality'])
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            for quality, count in sorted(quality_counts.items()):
                percentage = (count / len(state['quality_assessments'])) * 100
                stats += f"- {quality}: {count} ({percentage:.1f}%)\n"
        
        return stats
    
    # Event handlers (all accept and return state)
    def initialize_display(state):
        """Initialize the display with first question for new user"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        if not isinstance(state, dict) or 'current_index' not in state:
            state = init_user_session()
        data = format_question_display(state, state['current_index'])
        stats = get_statistics_from_state(state)
        user_status_msg = ""
        if state.get('user_id'):
            user_status_msg = f"✅ Loaded progress for: {state['user_id']}"
            if state['quality_assessments']:
                user_status_msg += f" ({len(state['quality_assessments'])} assessments restored)"
        if len(data) == 8:
            return state, *data, "", stats, user_status_msg
        return state, *data, stats, user_status_msg
    
    def handle_load_user(state, user_id):
        """Load a user's previous work"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        
        if not user_id or not user_id.strip():
            status_msg = "⚠️ Please enter a name/ID first"
            data = format_question_display(state, state['current_index'])
            stats = get_statistics_from_state(state)
            if len(data) == 8:
                return state, *data, "", stats, status_msg
            return state, *data, stats, status_msg
        
        user_id = user_id.strip()
        # Initialize new session with this user_id
        new_state = init_user_session(user_id)
        new_state['session_id'] = state['session_id'] if state else str(uuid.uuid4())[:8]
        
        data = format_question_display(new_state, new_state['current_index'])
        stats = get_statistics_from_state(new_state)
        
        if new_state['quality_assessments']:
            status_msg = f"✅ Loaded {len(new_state['quality_assessments'])} previous assessments. Resuming from question {new_state['current_index'] + 1}."
        else:
            status_msg = f"✅ User ID set to: {user_id}. Starting fresh (no previous assessments found)."
        
        if len(data) == 8:
            return new_state, *data, "", stats, status_msg
        return new_state, *data, stats, status_msg
    
    def auto_save_assessment(state):
        """Auto-save current assessment to the same file as manual save (replaces previous save for this user)"""
        if state is None or not isinstance(state, dict) or not state.get('quality_assessments'):
            return
        
        user_id = state.get('user_id', '')
        if not user_id:
            return  # Only auto-save if user has identified themselves
        
        try:
            # Use same filename as manual save - replaces previous save
            save_file = RESULTS_DIR / f"{user_id}/auto_saved/auto_saved_taxonomy_qa_survey_session_{state['session_id']}.json"
            results = {
                'session_id': state['session_id'],
                'user_id': user_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'current_index': state['current_index'],
                'total_questions': len(state['questions_data']),
                'assessed_questions': len(state['quality_assessments']),
                'assessments': []
            }
            
            for idx, assessment in state['quality_assessments'].items():
                question_data = state['questions_data'][idx]
                assessment_record = {
                    'question_index': idx,
                    'question': question_data.get('question', ''),
                    'answer': question_data.get('answer', ''),
                    'answer_object': question_data.get('original_answer', question_data.get('answer_object', '')),
                    'question_type': question_data.get('question_type', ''),
                    'image_id': question_data.get('image_id', ''),
                    'quality_assessment': assessment['quality'],
                    'notes': assessment.get('notes', ''),
                    'assessment_timestamp': assessment['timestamp']
                }
                results['assessments'].append(assessment_record)
            
            quality_counts = {}
            for assessment in state['quality_assessments'].values():
                quality = assessment['quality']
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            results['statistics'] = quality_counts
            
            # Replace previous file if it exists
            os.makedirs(save_file.parent, exist_ok=True)
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Auto-save failed: {e}")
    
    def handle_assessment_submit(state, quality, notes, user_id_input_val):
        """Handle quality assessment submission"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        
        # Update user_id if provided
        if user_id_input_val and user_id_input_val.strip():
            state['user_id'] = user_id_input_val.strip()
        
        if not state.get('questions_data'):
            return state, "No questions loaded", get_statistics_from_state(state), *format_question_display(state, state['current_index']), ""
        
        # Record assessment
        state['quality_assessments'][state['current_index']] = {
            'quality': quality,
            'notes': notes,
            'timestamp': datetime.datetime.now().isoformat(),
            'question_index': state['current_index'],
            'question_id': state['questions_data'][state['current_index']].get('image_id', 'unknown')
        }
        
        # Auto-save after each assessment (if user_id is set)
        auto_save_assessment(state)
        
        assessed_count = len(state['quality_assessments'])
        total_count = len(state['questions_data'])
        auto_save_note = " [Auto-saved]" if state.get('user_id') else " [Enter name/ID above to enable auto-save]"
        status = f"✅ Assessment recorded ({assessed_count}/{total_count} completed){auto_save_note}"
        stats = get_statistics_from_state(state)
        
        # Auto-advance to next question
        if state['current_index'] < len(state['questions_data']) - 1:
            state['current_index'] += 1
        
        display_data = format_question_display(state, state['current_index'])
        if len(display_data) == 8:
            display_data = display_data + ("",)
        return state, status, stats, *display_data
    
    def handle_navigation(state, direction):
        """Handle navigation buttons"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        
        if direction == "next" and state['current_index'] < len(state['questions_data']) - 1:
            state['current_index'] += 1
        elif direction == "previous" and state['current_index'] > 0:
            state['current_index'] -= 1
        elif direction == "first":
            state['current_index'] = 0
        elif direction == "last":
            state['current_index'] = len(state['questions_data']) - 1
        
        data = format_question_display(state, state['current_index'])
        if len(data) == 8:
            return state, *data, ""
        return state, *data
    
    def handle_jump(state, question_num):
        """Handle jump to specific question"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        
        question_idx = int(question_num) - 1
        if 0 <= question_idx < len(state['questions_data']):
            state['current_index'] = question_idx
        
        data = format_question_display(state, state['current_index'])
        if len(data) == 8:
            return state, *data, ""
        return state, *data
    
    def handle_save(state):
        """Handle save results (replaces previous save file for this user)"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        
        if not state.get('quality_assessments'):
            return state, gr.update(visible=True, value="No assessments to save")
        
        user_id = state.get('user_id', 'anonymous')
        session_id = state.get('session_id', '')
        
        # Validate user_id and session_id
        if not user_id or user_id.strip() == '':
            return state, gr.update(visible=True, value="Error: User ID is empty")
        
        if not session_id or session_id.strip() == '':
            return state, gr.update(visible=True, value="Error: Session ID is empty")
        
        # Create user directory if it doesn't exist
        user_dir = RESULTS_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Use consistent filename - replaces previous save for this user
        filename = f"taxonomy_qa_survey_{session_id}.json"
        filepath = user_dir / filename
        
        # Check if previous file exists and remove it (will be replaced)
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception as e:
                print(f"Warning: Could not remove previous file: {e}")
        
        results = {
            'session_id': state['session_id'],
            'user_id': user_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'current_index': state['current_index'],
            'total_questions': len(state['questions_data']),
            'assessed_questions': len(state['quality_assessments']),
            'assessments': []
        }
        
        for idx, assessment in state['quality_assessments'].items():
            question_data = state['questions_data'][idx]
            assessment_record = {
                'question_index': idx,
                'question': question_data.get('question', ''),
                'answer': question_data.get('answer', ''),
                'answer_object': question_data.get('original_answer', question_data.get('answer_object', '')),
                'question_type': question_data.get('question_type', ''),
                'image_id': question_data.get('image_id', ''),
                'quality_assessment': assessment['quality'],
                'notes': assessment.get('notes', ''),
                'assessment_timestamp': assessment['timestamp']
            }
            results['assessments'].append(assessment_record)
        
        quality_counts = {}
        for assessment in state['quality_assessments'].values():
            quality = assessment['quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        results['statistics'] = quality_counts
        
        # Save (replaces previous file if it existed)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        summary = f"✅ **Results saved successfully!**\n\n"
        summary += f"📁 File: {filename} \n"
        summary += f"📊 Assessed: {len(state['quality_assessments'])}/{len(state['questions_data'])} questions\n\n"
        summary += "**Quality Distribution:**\n"
        for quality, count in quality_counts.items():
            percentage = (count / len(state['quality_assessments'])) * 100
            summary += f"- {quality}: {count} ({percentage:.1f}%)\n"
        
        # combine all the results saved for this user
        all_results = list((RESULTS_DIR / user_id).glob(f"taxonomy_qa_survey_*.json"))
        all_results = list(set(all_results))
        all_results_display = ""
        for result in all_results:
            all_results_display += f"📁 File: {result.name} \n"
        summary += f"📁 All results saved for this user: \n" + all_results_display
        
        all_records = {}        

        for result in sorted(all_results):
            with open(result, 'r', encoding='utf-8') as f:
                result_json = json.load(f)
            for assessment in result_json['assessments']:
                all_records[assessment['question_index']] = assessment
        all_records= list(all_records.values())
        summary_json = {
            'user_id': user_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'total_questions': len(state['questions_data']),
            'assessed_questions': len(all_records),
            'assessments': all_records
        }
        statistics = {
            'High Quality - Correct': 0,
            'Low Quality - Incorrect': 0,
            'High Quality - Incorrect': 0,
            'Low Quality - Correct': 0,
            'Unassessed': 0
        }
        for assessment in all_records:
            if assessment['quality_assessment'] == 'High Quality - Correct':
                statistics['High Quality - Correct'] += 1
            elif assessment['quality_assessment'] == 'Low Quality - Incorrect':
                statistics['Low Quality - Incorrect'] += 1
            elif assessment['quality_assessment'] == 'High Quality - Incorrect':
                statistics['High Quality - Incorrect'] += 1
            elif assessment['quality_assessment'] == 'Low Quality - Correct':
                statistics['Low Quality - Correct'] += 1
            else:
                statistics['Unassessed'] += 1
        summary_json['statistics'] = statistics
        
        with open(RESULTS_DIR / user_id / f"taxonomy_qa_survey_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)

        return state, gr.update(visible=True, value=summary)
    
    def handle_filter(state, quality_filter):
        """Handle filter by quality"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        
        if not state.get('quality_assessments'):
            return state, gr.update(visible=True, value="No assessments yet")
        
        filtered = []
        for idx, assessment in state['quality_assessments'].items():
            if assessment['quality'] == quality_filter:
                question = state['questions_data'][idx]
                filtered.append(f"Q{idx+1}: {question.get('question', '')[:50]}...")
        
        if filtered:
            result = f"**Questions marked as '{quality_filter}':**\n\n" + "\n".join(filtered)
        else:
            result = f"No questions marked as '{quality_filter}'"
        
        return state, gr.update(visible=True, value=result)
    
    def handle_next_unassessed(state):
        """Jump to next unassessed question"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        
        next_idx = -1
        for i in range(len(state['questions_data'])):
            if i not in state['quality_assessments']:
                next_idx = i
                break
        
        if next_idx >= 0:
            state['current_index'] = next_idx
        # else: all assessed, stay on current
        
        data = format_question_display(state, state['current_index'])
        if len(data) == 8:
            return state, *data, ""
        return state, *data
    
    def handle_show_unassessed(state):
        """Show summary of unassessed questions"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        
        unassessed = []
        for i in range(len(state['questions_data'])):
            if i not in state['quality_assessments']:
                unassessed.append(i)
        
        if not unassessed:
            result = "🎉 All questions have been assessed!"
        else:
            summary = f"**Unassessed Questions ({len(unassessed)} remaining):**\n\n"
            for idx in unassessed[:10]:
                question = state['questions_data'][idx]
                summary += f"- Q{idx+1}: {question.get('question_type', 'unknown')}\n"
            if len(unassessed) > 10:
                summary += f"\n... and {len(unassessed) - 10} more"
            result = summary
        
        return state, gr.update(visible=True, value=result)
    
    def handle_refresh_stats(state):
        """Handle refresh statistics"""
        if state is None or not isinstance(state, dict):
            state = init_user_session()
        return state, get_statistics_from_state(state)
    
    # Connect event handlers (all include state in inputs and outputs)
    submit_assessment_btn.click(
        handle_assessment_submit,
        inputs=[user_state, quality_radio, notes_input, user_id_input],
        outputs=[
            user_state, assessment_status, stats_display,
            question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input
        ]
    )
    
    load_user_btn.click(
        handle_load_user,
        inputs=[user_state, user_id_input],
        outputs=[
            user_state, question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input,
            stats_display, user_status
        ]
    )
    
    # Navigation handlers
    first_btn.click(
        lambda state: handle_navigation(state, "first"),
        inputs=[user_state],
        outputs=[
            user_state, question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input
        ]
    )
    
    prev_btn.click(
        lambda state: handle_navigation(state, "previous"),
        inputs=[user_state],
        outputs=[
            user_state, question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input
        ]
    )
    
    next_btn.click(
        lambda state: handle_navigation(state, "next"),
        inputs=[user_state],
        outputs=[
            user_state, question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input
        ]
    )
    
    last_btn.click(
        lambda state: handle_navigation(state, "last"),
        inputs=[user_state],
        outputs=[
            user_state, question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input
        ]
    )
    
    jump_btn.click(
        handle_jump,
        inputs=[user_state, jump_input],
        outputs=[
            user_state, question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input
        ]
    )
    
    next_unassessed_btn.click(
        handle_next_unassessed,
        inputs=[user_state],
        outputs=[
            user_state, question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input
        ]
    )
    
    save_btn.click(
        handle_save,
        inputs=[user_state],
        outputs=[user_state, save_status]
    )
    
    refresh_stats_btn.click(
        handle_refresh_stats,
        inputs=[user_state],
        outputs=[user_state, stats_display]
    )
    
    show_unassessed_btn.click(
        handle_show_unassessed,
        inputs=[user_state],
        outputs=[user_state, unassessed_summary]
    )
    
    filter_btn.click(
        handle_filter,
        inputs=[user_state, filter_dropdown],
        outputs=[user_state, filter_results]
    )
    
    # Initialize display on load
    app.load(
        initialize_display,
        inputs=[user_state],
        outputs=[
            user_state, question_display, answer_display, objects_display, metadata_display,
            image_display, reasoning_display, progress_text, quality_radio, notes_input, stats_display, user_status
        ]
    )

print("🎉 Gradio interface ready!")

if __name__ == "__main__":
    print("🌐 Launching Taxonomy QA Survey...")
    launch_kwargs = {
        'share': True,
        'debug': True,
        'server_name': '0.0.0.0',
    }
    try:
        cwd_path = Path.cwd().resolve()
        images_path = Path(IMAGES_DIR).resolve()
        if images_path.is_dir() and not str(images_path).startswith(str(cwd_path)):
            launch_kwargs['allowed_paths'] = [str(images_path)]
    except Exception:
        pass
    app.launch(**launch_kwargs)
