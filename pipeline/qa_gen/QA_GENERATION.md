# QA Generation Documentation

This document describes how questions and answers are generated for the Taxonomy QA Benchmark.

## Overview

The QA generation system creates question-answer pairs that test visual understanding, taxonomic reasoning, spatial relationships, and compositional thinking about objects in scenes.

## Question Generation Process

### 1. Object Extraction
- **Real Images**: Extract objects from OpenImages annotations
- **Sim Images**: Extract objects from 3D scenes with pose data
- **Filters**: Apply visibility, segmentation, and relevance filters

### 2. Question Type Selection
Each object can generate multiple question types based on available properties:

- **Material Questions**: Based on object materials
- **Function Questions**: Based on object affordances
- **Spatial Questions**: Based on object relationships
- **Affordance Questions**: Based on object capabilities
- **Repurposing Questions**: Alternative uses
- **Counterfactual Questions**: What-if scenarios

### 3. Answer Generation
- Extract correct answer from object annotations
- Generate multiple choice options from other objects
- Validate answer-choices alignment

### 4. CoT Reasoning
Generate Chain-of-Thought reasoning explaining the answer

---

## Question Types

### Material Property Questions

Tests understanding of what materials objects are made from.

**Types**:
- `material_property` - General material questions
- `material_metals_and_alloys` - Metallic objects
- `material_textiles_fibers_and_leather` - Fabric materials
- `material_wood_and_plant_based_solids` - Wooden objects
- `material_plastics_rubber_and_polymers` - Synthetic materials
- `material_stone_concrete_and_mineral` - Stone/mineral materials

**Example**:
- **Q**: "Which object is made of 'cotton'?"
- **A**: "Shirt"
- **CoT**: "To identify which object is made of cotton, I examine the material properties of each object. Among [Objects], the shirt displays the characteristic properties of cotton while others are made of different materials."

**Filters**:
- Objects must have valid material data
- Excludes void clusters ("no clear affordance")
- Removes malformed text

**Count in Real Images**: 7,702 questions

---

### Function Knowledge Questions

Tests understanding of what objects are used for.

**Type**: `function_knowledge`

**Example**:
- **Q**: "Which object is primarily used for 'sitting'?"
- **A**: "Chair"
- **CoT**: "To identify which object is used for sitting, I examine the functional characteristics of each object. Among [Objects], the chair has the function of sitting while others serve different purposes."

**Filters**:
- Objects must have valid function data
- Excludes void clusters
- Validates affordance taxonomy

**Count**: 7,645 questions

---

### Description Matching Questions

Tests understanding of object descriptions and visual features.

**Type**: `description_matching`

**Example**:
- **Q**: "Which object matches this description: 'A red, soft, rectangular seating object'?"
- **A**: "Red Couch"
- **CoT**: "To match the given description 'red, soft, rectangular seating object', I examine the visual characteristics, size, shape, color, and other distinguishing features of each object. Among [Objects], the red couch best matches the description while others do not fit as well."

**Count**: 7,849 questions (most common)

---

### Spatial Relationship Questions

Tests understanding of object positions and spatial relationships.

**Types**:
- `spatial_left` - "Which object is to the left of {object}?"
- `spatial_right` - "Which object is to the right of {object}?"
- `spatial_above` - "Which object is above {object}?"
- `spatial_below` - "Which object is below {object}?"
- `spatial_near` - "Which object is near {object}?"
- `spatial_far` - "Which object is far from {object}?"

**Note**: Only available for sim images (requires 3D pose data)

**Example**:
- **Q**: "Which object is to the left of the red chair?"
- **A**: "Blue Table"
- **CoT**: "To determine which object is to the left of the red chair, I examine the spatial relationship between each object and the reference object. Among [Objects], the blue table has spatial relationship 'to the left of' relative to the red chair."

---

### Affordance-Based Questions

Tests understanding of object capabilities and affordances.

**Example**:
- **Q**: "Which object can be used to sit on?"
- **A**: "Chair"
- **Multiple choices**: Chair, Book, Lamp

---

### Repurposing Questions

Tests creative thinking about alternative uses.

**Filters**:
- Must be unexpected or creative repurposing
- Validates against taxonomy constraints
- Excludes impossible repurposing

**Example**:
- **Q**: "How could someone repurpose a book as a doorstop?"
- **A**: "Use the book's weight to hold the door open"

---

### Counterfactual Questions

Tests reasoning about hypothetical scenarios.

**Example**:
- **Q**: "What would happen if the table were made of glass instead of wood?"
- **A**: "It would be more fragile and see-through, requiring careful handling"

---

## Quality Controls

### 1. Void Cluster Filtering
Excludes objects without clear affordances or materials from material/function questions.

### 2. Malformed Text Cleaning
Removes incomplete text, placeholder strings, and overly long descriptions from material/function fields.

### 3. Answer Validation
Ensures answer exists in choices and is actually a valid option.

### 4. Choice Diversity
Ensures multiple choice options are diverse and realistic.

### 5. Spatial Validation
For sim images, validates spatial relationships using 3D pose data.

---

## CoT Reasoning Generation

Chain-of-Thought reasoning is generated for each question to explain the answer.

### Generic Templates
- Material: Uses material-specific templates with visual/physical properties
- Function: Uses function-specific reasoning templates
- Spatial: Explains spatial relationships using coordinates
- Repurposing: Explains alternative uses through affordances

### Property-Based Reasoning
For questions about specific properties (sound, thermal, etc.):
- Explains the physical basis for the property
- References material characteristics
- Uses object-specific details

---

## Output Format

### JSON Structure
```json
{
  "question": "Which object is made of 'cotton'?",
  "question_type": "material_property",
  "answer": "Shirt",
  "choices": ["Shirt", "Book", "Table", "Lamp"],
  "objects": ["Shirt", "Book", "Table", "Lamp"],
  "reasoning": "To identify which object is made of cotton...",
  "image_path": "scene_123_group0/bbox.jpg",
  "scene_id": "scene_123_group0"
}
```

### TSV Format (for VLMEvalKit)
```
index	image	question	answer	other	question_type
0	scene_123_group0/bbox.jpg	Which object is made of 'cotton'?	Shirt	Book, Table, Lamp	material_property
```

---

## Statistics

### Real Image Benchmark
- **Total Questions**: ~7,700
- **Types**: Material (7,702), Function (7,645), Description (7,849)
- **With CoT**: All questions include reasoning

### Sim Image Benchmark
- **Total Questions**: ~1,000+
- **Types**: All types including spatial relationships
- **Unique Features**: 3D pose data, depth-aware grouping

---

## See Also

- **[SIM_IMAGE_PIPELINE.md](SIM_IMAGE_PIPELINE.md)** - Complete sim image pipeline
- **[QUESTION_TYPES_DOCUMENTATION.md](scripts/modules/qa_modules/QUESTION_TYPES_DOCUMENTATION.md)** - Detailed question type specifications
- **[README.md](README.md)** - Quick start and usage guide

