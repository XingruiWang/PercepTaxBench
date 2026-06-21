#!/usr/bin/env python3

import json
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

from modules.qa_modules.taxonomy_utils import TaxonomyUtils
from modules.qa_modules.object_utils import ObjectUtils
from modules.qa_modules.filter_utils import is_void_cluster
from modules.qa_modules.data_loading_utils import DataLoadingUtils


class AnswerChoicePropertyAdder:
    def __init__(self, taxonomy_dir: Path):
        self.taxonomy_utils = TaxonomyUtils(taxonomy_dir)
        self.object_utils = ObjectUtils(taxonomy_utils=self.taxonomy_utils)
        self.data_loader = DataLoadingUtils()
        self.qa_space_data = self.data_loader.load_qa_space_data()
    
    def get_required_physical_properties_for_question_type(self, question_type: str) -> List[str]:
        """Get the specific physical property clusters required for a question type from QA space"""
        if not self.qa_space_data:
            return []
        
        question_answer_mappings = self.qa_space_data.get("question_answer_mappings", {})
        question_data = question_answer_mappings.get(question_type, {})
        
        required_props = question_data.get("required_properties", [])
        filter_criteria = question_data.get("filter_criteria", [])
        
        physical_props = []
        
        for prop in required_props:
            if prop.lower() in ["rigid", "movable", "hollow", "flexible", "stable", "fragile", "unstable", "light", "heavy", "durable", "smooth"]:
                physical_props.append(prop.lower())
        
        for criterion in filter_criteria:
            criterion_lower = criterion.lower()
            # Check for physical property mentions in filter criteria
            physical_property_keywords = [
                "rigid", "movable", "hollow", "flexible", "stable", "fragile", 
                "unstable", "light", "heavy", "durable", "smooth", "rough",
                "liquid", "solid", "fixed"
            ]
            for keyword in physical_property_keywords:
                if keyword in criterion_lower and keyword not in physical_props:
                    physical_props.append(keyword)
        
        return physical_props
    
    def format_properties_as_natural_language(self, object_name: str, question_type: str, choice_label: str) -> str:
        if not object_name:
            return ""
        
        question_type_lower = question_type.lower()
        property_parts = []
        
        if question_type_lower.startswith("material") or "material" in question_type_lower:
            material = self.object_utils.get_object_material(object_name)
            material_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_material")
            non_void_clusters = [c for c in material_clusters if not is_void_cluster(c, "material")]
            
            if material:
                property_parts.append(f"made of {material}")
            if non_void_clusters:
                cluster_text = ", ".join(non_void_clusters)
                property_parts.append(f"material cluster: {cluster_text}")
        
        if question_type_lower.startswith("affordance") or "affordance" in question_type_lower:
            affordance_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_affordances")
            non_void_clusters = [c for c in affordance_clusters if not is_void_cluster(c, "affordance")]
            
            if non_void_clusters:
                cluster_text = ", ".join(non_void_clusters)
                property_parts.append(f"affordance: {cluster_text}")
        
        if question_type_lower.startswith("function") or "function" in question_type_lower:
            function = self.object_utils.get_object_function(object_name)
            function_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_function")
            non_void_clusters = [c for c in function_clusters if not is_void_cluster(c, "function")]
            
            if function:
                property_parts.append(f"function: {function}")
            if non_void_clusters:
                cluster_text = ", ".join(non_void_clusters)
                property_parts.append(f"function cluster: {cluster_text}")
        
        if question_type_lower.startswith("physical") or "physical" in question_type_lower:
            physical_properties = self.object_utils.get_object_physical_properties(object_name)
            physical_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_physical_properties")
            non_void_clusters = [c for c in physical_clusters if not is_void_cluster(c, "physical")]
            
            if physical_properties:
                props_text = ", ".join(physical_properties)
                property_parts.append(f"physical properties: {props_text}")
            if non_void_clusters:
                cluster_text = ", ".join(non_void_clusters)
                property_parts.append(f"physical property cluster: {cluster_text}")
        
        if question_type_lower.startswith("repurposing") or "repurposing" in question_type_lower:
            affordance_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_affordances")
            non_void_affordances = [c for c in affordance_clusters if not is_void_cluster(c, "affordance")]
            material = self.object_utils.get_object_material(object_name)
            physical_properties = self.object_utils.get_object_physical_properties(object_name)
            
            if non_void_affordances:
                cluster_text = ", ".join(non_void_affordances)
                property_parts.append(f"affordance: {cluster_text}")
            if material:
                property_parts.append(f"made of {material}")
            if physical_properties:
                props_text = ", ".join(physical_properties)
                property_parts.append(f"physical properties: {props_text}")
        
        if question_type_lower.startswith("compositional") or "compositional" in question_type_lower:
            material = self.object_utils.get_object_material(object_name)
            affordance_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_affordances")
            non_void_affordances = [c for c in affordance_clusters if not is_void_cluster(c, "affordance")]
            physical_properties = self.object_utils.get_object_physical_properties(object_name)
            function = self.object_utils.get_object_function(object_name)
            
            if material:
                property_parts.append(f"made of {material}")
            if non_void_affordances:
                cluster_text = ", ".join(non_void_affordances)
                property_parts.append(f"affordance: {cluster_text}")
            if physical_properties:
                props_text = ", ".join(physical_properties)
                property_parts.append(f"physical properties: {props_text}")
            if function:
                property_parts.append(f"function: {function}")
        
        if question_type_lower.startswith("counterfactual") or "counterfactual" in question_type_lower:
            material = self.object_utils.get_object_material(object_name)
            physical_properties = self.object_utils.get_object_physical_properties(object_name)
            
            if material:
                property_parts.append(f"made of {material}")
            if physical_properties:
                props_text = ", ".join(physical_properties)
                property_parts.append(f"physical properties: {props_text}")
        
        if question_type_lower.startswith("functional") or "functional" in question_type_lower:
            function = self.object_utils.get_object_function(object_name)
            affordance_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_affordances")
            non_void_affordances = [c for c in affordance_clusters if not is_void_cluster(c, "affordance")]
            
            if function:
                property_parts.append(f"function: {function}")
            if non_void_affordances:
                cluster_text = ", ".join(non_void_affordances)
                property_parts.append(f"affordance: {cluster_text}")
        
        if question_type_lower == "description_matching" or "description" in question_type_lower:
            description = self.object_utils.get_object_description(object_name)
            material = self.object_utils.get_object_material(object_name)
            function = self.object_utils.get_object_function(object_name)
            
            if description:
                property_parts.append(f"description: {description}")
            if material:
                property_parts.append(f"made of {material}")
            if function:
                property_parts.append(f"function: {function}")
        
        if not property_parts:
            return ""
        
        return f" ({'; '.join(property_parts)})"
    
    def add_properties_to_question(self, question: Dict, include_all_properties: bool = True) -> Dict:
        question_type = question.get("question_type", "")
        if not question_type:
            return question
        
        question_type_lower = question_type.lower()
        
        # Skip description matching questions
        if question_type_lower == "description_matching" or "description" in question_type_lower:
            return question
        
        box_to_object = question.get("box_to_object", {})
        choices = question.get("choices", [])
        
        if not box_to_object or not choices:
            return question
        
        question_text = question.get("question", "")
        if not question_text:
            return question
        
        # Get required physical properties for this question type (if filtering)
        required_physical_props = None
        if not include_all_properties:
            required_physical_props = self.get_required_physical_properties_for_question_type(question_type)
        
        # Create reverse mapping from object to box color
        object_to_box = {v: k for k, v in box_to_object.items()}
        
        material_data = {}  # {box_color: material_value}
        affordance_data = {}  # {box_color: [affordance_clusters]}
        function_data = {}  # {box_color: function_value}
        physical_property_data = {}  # {box_color: [physical_properties]}
        
        for choice in choices:
            if isinstance(choice, dict):
                choice_label = choice.get("choice", "")
            else:
                choice_label = choice
            
            # Get object name - check if choice_label is a box color or object name
            object_name = box_to_object.get(choice_label, "")
            if not object_name:
                # Choice might be object name, need to find box color
                box_color = object_to_box.get(choice_label, choice_label)
                object_name = choice_label
            else:
                box_color = choice_label
            
            # Determine which properties to include based on question type and mode
            should_include_material = include_all_properties or "material" in question_type_lower
            should_include_affordance = include_all_properties or "affordance" in question_type_lower or "repurposing" in question_type_lower or "compositional" in question_type_lower or "functional" in question_type_lower
            should_include_function = include_all_properties or "function" in question_type_lower or "compositional" in question_type_lower or "functional" in question_type_lower
            should_include_physical = include_all_properties or "physical" in question_type_lower or "repurposing" in question_type_lower or "compositional" in question_type_lower or "counterfactual" in question_type_lower
            
            # Collect material properties
            if should_include_material:
                material = self.object_utils.get_object_material(object_name)
                material_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_material")
                non_void_material = [c for c in material_clusters if not is_void_cluster(c, "material")]
                if material:
                    material_data[box_color] = material
                elif material_clusters and not non_void_material:
                    material_data[box_color] = "no clear material"
            
            # Collect affordance properties
            if should_include_affordance:
                affordance_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_affordances")
                non_void_clusters = [c for c in affordance_clusters if not is_void_cluster(c, "affordance")]
                if non_void_clusters:
                    affordance_data[box_color] = non_void_clusters
                elif affordance_clusters:
                    affordance_data[box_color] = ["no clear affordance"]
            
            # Collect function properties
            if should_include_function:
                function = self.object_utils.get_object_function(object_name)
                function_clusters = self.taxonomy_utils.get_object_clusters(object_name, "final_taxonomy_function")
                non_void_function = [c for c in function_clusters if not is_void_cluster(c, "function")]
                if function:
                    function_data[box_color] = function
                elif function_clusters and not non_void_function:
                    function_data[box_color] = "no clear function"
            
            # Collect physical properties
            if should_include_physical:
                physical_properties = self.object_utils.get_object_physical_properties(object_name)
                if physical_properties:
                    if include_all_properties:
                        # Include all physical properties
                        physical_property_data[box_color] = physical_properties
                    else:
                        # Filter by required properties for this question type
                        if required_physical_props:
                            filtered_props = [p for p in physical_properties if p.lower() in required_physical_props]
                            if filtered_props:
                                physical_property_data[box_color] = filtered_props
                        else:
                            # If no specific requirements, include all
                            physical_property_data[box_color] = physical_properties
        
        
        property_lines = []
        
        if material_data:
            material_items = []
            for box_color, material_value in material_data.items():
                # Format material names to natural language
                formatted_material = material_value.replace(" & ", " and ").replace("&", " and ")
                formatted_material = formatted_material.replace("  ", " ").strip()
                material_items.append(f"{box_color}: {formatted_material}")
            if len(material_items) > 1:
                choices_text = material_items[0] + ". " + ". ".join(material_items[1:])
            else:
                choices_text = material_items[0]
            property_lines.append(f"material property: {choices_text}")
        
        if affordance_data:
            affordance_items = []
            for box_color, affordance_clusters in affordance_data.items():
                # Format cluster names to natural language
                formatted_clusters = []
                for cluster in affordance_clusters:
                    # Replace "/" with " or ", "&" with " and ", and clean up
                    formatted = cluster.replace(" / ", " or ").replace("/", " or ")
                    formatted = formatted.replace(" & ", " and ").replace("&", " and ")
                    formatted = formatted.replace("  ", " ").strip()
                    formatted_clusters.append(formatted)
                # Join multiple clusters with "; " (semicolon)
                if len(formatted_clusters) > 1:
                    clusters_text = "; ".join(formatted_clusters)
                else:
                    clusters_text = formatted_clusters[0] if formatted_clusters else ""
                affordance_items.append(f"{box_color}: {clusters_text}")
            if len(affordance_items) > 1:
                choices_text = affordance_items[0] + ". " + ". ".join(affordance_items[1:])
            else:
                choices_text = affordance_items[0]
            property_lines.append(f"affordance property: {choices_text}")
        
        if function_data:
            function_items = []
            for box_color, function_value in function_data.items():
                # Format function names to natural language
                formatted_function = function_value.replace(" & ", " and ").replace("&", " and ")
                formatted_function = formatted_function.replace(" / ", " or ").replace("/", " or ")
                formatted_function = formatted_function.replace("  ", " ").strip()
                function_items.append(f"{box_color}: {formatted_function}")
            if len(function_items) > 1:
                choices_text = function_items[0] + ". " + ". ".join(function_items[1:])
            else:
                choices_text = function_items[0]
            property_lines.append(f"function property: {choices_text}")
        
        if physical_property_data:
            physical_items = []
            for box_color, physical_props in physical_property_data.items():
                # Join multiple physical properties with ", " (comma)
                if len(physical_props) > 1:
                    props_text = ", ".join(physical_props)
                else:
                    props_text = physical_props[0] if physical_props else ""
                physical_items.append(f"{box_color}: {props_text}")
            if len(physical_items) > 1:
                choices_text = physical_items[0] + ". " + ". ".join(physical_items[1:])
            else:
                choices_text = physical_items[0]
            property_lines.append(f"physical property: {choices_text}")
        
        if property_lines:
            context_text = "\n" + "\n".join(property_lines) + "."
            question_text = question_text.rstrip(".") + context_text
            question["question"] = question_text
        
        return question
    
    def should_skip_question(self, question: Dict, is_real_benchmark: bool = False) -> bool:
        question_type = question.get("question_type", "").lower()
        question_category = question.get("question_category", "").lower()
        
        if question_type.startswith("spatial_") or question_category == "spatial_relation":
            return True
        
        if is_real_benchmark:
            if question.get("manual_entry") or question.get("is_manual"):
                return True
            if question_type.startswith("manual_"):
                return True
        
        return False
    
    def process_benchmark(self, input_dir: Path, output_dir: Path, is_real_benchmark: bool = False, include_all_properties: bool = True):
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=False)
        
        all_questions_path = input_dir / "all_questions.json"
        if not all_questions_path.exists():
            raise FileNotFoundError(f"all_questions.json not found in {input_dir}")
        
        print(f"Loading questions from {all_questions_path}")
        with open(all_questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        print(f"Processing {len(questions)} questions...")
        
        processed_questions = []
        skipped_count = 0
        
        for i, question in enumerate(questions):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(questions)} questions...")
            
            if self.should_skip_question(question, is_real_benchmark):
                skipped_count += 1
                continue
            
            processed_question = self.add_properties_to_question(question, include_all_properties=include_all_properties)
            processed_questions.append(processed_question)
        
        print(f"Omitted {skipped_count} questions (spatial/manual)")
        print(f"Added properties to {len(processed_questions)} questions")
        
        output_questions_path = output_dir / "all_questions.json"
        print(f"Writing processed questions to {output_questions_path}")
        with open(output_questions_path, 'w', encoding='utf-8') as f:
            json.dump(processed_questions, f, indent=2, ensure_ascii=False)
        
        input_images_dir = input_dir / "images"
        output_images_dir = output_dir / "images"
        
        if input_images_dir.exists():
            print(f"Creating symlink from {input_images_dir} to {output_images_dir}")
            if output_images_dir.exists() or output_images_dir.is_symlink():
                if output_images_dir.is_symlink():
                    output_images_dir.unlink()
                else:
                    shutil.rmtree(output_images_dir)
            output_images_dir.symlink_to(input_images_dir.resolve(), target_is_directory=True)
        else:
            print(f"Warning: Images directory not found at {input_images_dir}")
        
        other_files = ["combining_summary.json", "question_type_statistics.json", 
                      "scene_statistics.json", "generation_metadata.json"]
        
        for filename in other_files:
            source_file = input_dir / filename
            if source_file.exists():
                dest_file = output_dir / filename
                shutil.copy2(source_file, dest_file)
                print(f"Copied {filename}")
        
        print(f"Processing complete! Output saved to {output_dir}")
    
    def extract_example_questions(self, benchmark_dir: Path, output_file: Path, benchmark_name: str):
        """Extract one example question from each question type for inspection"""
        all_questions_path = benchmark_dir / "all_questions.json"
        
        if not all_questions_path.exists():
            print(f"Warning: {all_questions_path} not found, skipping example extraction")
            return
        
        print(f"Loading questions from {all_questions_path} for example extraction...")
        with open(all_questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        question_type_examples = {}
        
        for question in questions:
            question_type = question.get("question_type", "unknown")
            
            if question_type not in question_type_examples:
                question_type_examples[question_type] = question
        
        examples = list(question_type_examples.values())
        examples.sort(key=lambda x: x.get("question_type", ""))
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.exists():
            output_file.unlink()
            print(f"Removed existing examples file: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted {len(examples)} example questions (one per question type) to {output_file}")
        print(f"Question types: {', '.join(sorted(question_type_examples.keys()))}")


def main():
    script_dir = Path(__file__).resolve().parent
    taxonomy_dir = script_dir / "taxonomy"
    
    if not taxonomy_dir.exists():
        raise FileNotFoundError(f"Taxonomy directory not found: {taxonomy_dir}")
    
    qa_gen_dir = script_dir
    
    real_input_dir = qa_gen_dir / "taxonomyQABench_realimage_final_polished"
    sim_input_dir = qa_gen_dir / "taxonomyQABench_simimage_final"
    
    # Output directories for designated properties (filtered)
    real_output_dir_designated = qa_gen_dir / "taxonomyQABench_realimage_final_polished_with_properties"
    sim_output_dir_designated = qa_gen_dir / "taxonomyQABench_simimage_final_with_properties"
    
    # Output directories for all properties
    real_output_dir_all = qa_gen_dir / "taxonomyQABench_realimage_final_polished_with_all_properties"
    sim_output_dir_all = qa_gen_dir / "taxonomyQABench_simimage_final_with_all_properties"
    
    adder = AnswerChoicePropertyAdder(taxonomy_dir)
    
    print("=" * 80)
    print("Processing REAL IMAGE benchmark - DESIGNATED PROPERTIES")
    print("=" * 80)
    adder.process_benchmark(real_input_dir, real_output_dir_designated, is_real_benchmark=True, include_all_properties=False)
    
    print("\n" + "=" * 80)
    print("Processing REAL IMAGE benchmark - ALL PROPERTIES")
    print("=" * 80)
    adder.process_benchmark(real_input_dir, real_output_dir_all, is_real_benchmark=True, include_all_properties=True)
    
    print("\n" + "=" * 80)
    print("Processing SIM IMAGE benchmark - DESIGNATED PROPERTIES")
    print("=" * 80)
    adder.process_benchmark(sim_input_dir, sim_output_dir_designated, is_real_benchmark=False, include_all_properties=False)
    
    print("\n" + "=" * 80)
    print("Processing SIM IMAGE benchmark - ALL PROPERTIES")
    print("=" * 80)
    adder.process_benchmark(sim_input_dir, sim_output_dir_all, is_real_benchmark=False, include_all_properties=True)
    
    print("\n" + "=" * 80)
    print("Extracting example questions for inspection...")
    print("=" * 80)
    
    examples_dir = qa_gen_dir / "question_with_property_example"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    real_examples_file = examples_dir / "real_question_examples.json"
    sim_examples_file = examples_dir / "sim_question_examples.json"
    
    print("\nExtracting real image examples (from all properties version)...")
    adder.extract_example_questions(real_output_dir_all, real_examples_file, "real")
    
    print("\nExtracting sim image examples (from all properties version)...")
    adder.extract_example_questions(sim_output_dir_all, sim_examples_file, "sim")
    
    print("\n" + "=" * 80)
    print("All processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

