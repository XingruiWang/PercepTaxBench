# Taxonomy QA Benchmark - Question Types Documentation

## Overview

This document describes **only the question types that are actively generated** for the **real image benchmark pipeline**. Questions are designed to test visual understanding, taxonomic reasoning, and compositional thinking about objects in scenes.

The pipeline generates questions using a QA space analysis that pre-validates which objects can answer which question types, ensuring high-quality, unambiguous questions.

---

## Question Type Categories

Questions are categorized as:
- **Easy Questions**: Direct property/material/affordance/function matching
- **Hard Questions**: Require reasoning about repurposing, counterfactuals, composition, or latent properties
- **Spatial Questions**: Test spatial relationship understanding

**Limits per image:**
- Easy questions: Up to 4 per image
- Spatial questions: Random 0, 1, or 2 per image
- Hard questions: All valid questions (prioritized)
- **Total cap: 10 questions per image** (prioritizing hard questions)

---

## 1. Easy Question Types

### 1.1 Description Matching

**`description_matching`**
- **Question Template**: "Which object matches this description: '{description}'?"
- **Filter Conditions**:
  - Objects must have valid description text (not empty, not placeholder)
  - Excludes objects in void clusters (No-Physical-Properties, No clear affordance, etc.)
  - Checks for taxonomy cluster conflicts (if description source is material/affordance/function/texture)
- **Description Sources**: General description, material, texture, affordance, or function (NOT physical property)
- **CoT Reasoning**: "To match the given description '{description}', I examine the visual characteristics, size, shape, color, and other distinguishing features of each object. Among {objects}, the {answer} best matches the description based on its specific features while others do not fit as well."
- **Usage**: Most common question type in real images (~28.6%)

### 1.2 Function Knowledge

**`function_knowledge`**
- **Question Template**: "Which object is used as '{function}'?"
- **Filter Conditions**:
  - Objects must have valid function annotations (not void cluster)
  - Function field must not be empty
  - Checks for taxonomy cluster conflicts (if other objects in scene share same function cluster, question is skipped)
- **CoT Reasoning**: "To identify which object performs {function_name}, I examine the functional characteristics of each object. Among {objects}, the {answer} has the function of {function_name} while others serve different purposes or lack this capability."
- **Usage**: ~27.9% of questions in real images

### 1.3 Material Property

**`material_property`**
- **Question Template**: "Which object is made of '{material}'?"
- **Filter Conditions**:
  - Objects must have valid material annotations (not void cluster)
  - Material field must not be empty or contain placeholder text
  - Material must be unique among objects in the scene
  - Checks for taxonomy cluster conflicts (if other objects in scene share same material cluster, question is skipped)
- **CoT Reasoning**: "To identify which object is made of {material_name}, I examine the material properties of each object. Among {objects}, the {answer} displays the characteristic properties of {material_name} while others are made of different materials."
- **Usage**: ~27.8% of questions in real images

**Note**: The real image pipeline uses **only** `material_property` (which uses actual material strings from annotations), NOT the specific material cluster variants like `material_metals_and_alloys`, `material_textiles_*`, etc. Those variants are only used in the simulated image pipeline.

### 1.4 Affordance Questions

Affordance questions test understanding of what actions or purposes objects enable.

**Format**: `affordance_{cluster_name}` where cluster_name is derived from affordance taxonomy clusters.

**Common Affordance Question Types:**
- `affordance_furniture`: "Which object is furniture?"
- `affordance_contain__carry__package`: "Which object can contain or carry items?"
- `affordance_grip__carry__operate`: "Which object can be gripped and carried?"
- `affordance_operate__use_device`: "Which object is a device or tool that can be operated?" (devices/tools, NOT machinery)
- `affordance_mechanical_control`: "Which object requires mechanical control to operate?" (vehicles/machinery requiring control)
- `affordance_mediated_action_and_meaning`: "Which object involves reading or communication?"
- `affordance_food_—_ingredients_and_produce`: "Which object is food or produce?"
- `affordance_food_—_prepared_dishes`: "Which object is prepared food?"
- `affordance_cleaning_and_sanitation`: "Which object is used for cleaning?"
- `affordance_control__express__light`: "Which object controls or produces light?"
- `affordance_grow__plant_(vegetation)`: "Which object is used for growing plants?"
- `affordance_enclosures_and_venues_(enter_use)`: "Which object is an enclosed space or venue?"
- `affordance_place__support__work_on`: "Which object can have items placed on it?"
- `affordance_build__span__occupy`: "Which object is used for building, spanning, or occupying space?"
- And other affordance clusters from the taxonomy

**Filter Conditions**:
- Objects must be in the specified affordance cluster
- Excludes inappropriate clusters (Human Roles, Natural Scenes, Phenomena, Unclassified)
- Checks for taxonomy cluster conflicts (if other objects in scene share same affordance cluster, question is skipped)
- Special cross-cluster conflict detection:
  - `affordance_enclosures_and_venues_(enter_use)` excludes objects in "Build / Span / Occupy"
  - `affordance_build__span__occupy` excludes objects in "Enclosures & Venues (Enter/Use)"

**CoT Reasoning**: "To identify which object has the affordance of {affordance_type}, I examine the affordance characteristics of each object. Among {objects}, the {answer} has the affordance of {affordance_type} while others lack this capability."

### 1.5 Functional Questions

**`functional_seating`**
- **Question Template**: "Which object can be used for sitting or resting?"
- **Filter Conditions**:
  - Objects must be in the "Sit / Ride / Attend" affordance cluster
  - Checks for taxonomy cluster conflicts
- **CoT Reasoning**: "To identify which object can be used for sitting or resting, I examine the functional and affordance characteristics of each object. Among {objects}, the {answer} is designed for sitting or resting while others serve different purposes."

**`functional_foldable`**
- **Question Template**: "Which object can be folded or collapsed to save space?"
- **Filter Conditions**:
  - **MUST be in Flexible physical property cluster** (strict requirement)
  - Must be in foldable material clusters (Textiles, Fibers & Leather OR Plastics, Rubber & Polymers)
  - OR has foldable keywords (fold, collapsible, retractable, blanket, towel, cloth, curtain, shirt, clothing, pillow, rug, sheet)
  - Checks for taxonomy cluster conflicts
- **CoT Reasoning**: "To identify which object can be folded or collapsed, I examine the material and physical properties of each object. Among {objects}, the {answer} is flexible and made of foldable materials while others are rigid or not foldable."

---

## 2. Hard Question Types

### 2.1 Repurposing Questions

Repurposing questions test creative thinking about alternative uses for objects based on their properties.

**Format**: `repurposing_{concept}_concept`

**Common Repurposing Question Types:**

**`repurposing_shield_concept`**
- **Question**: "Which object could be repurposed as a shield?"
- **Filter**: Rigid, stable, movable, NOT fragile, flat surface (width/length > height)
- **Excludes**: Food, electronics, furniture, large architectural items, various affordance/function clusters
- **Purpose**: Tests understanding of protection and blocking properties

**`repurposing_container_concept`**
- **Question**: "Which object could be repurposed as a container?"
- **Filter**: Has hollow property OR "Contain / Carry / Package" affordance
- **Excludes**: Various equipment/infrastructure affordance and function clusters
- **Purpose**: Tests understanding of storage and holding capabilities

**`repurposing_reflector_concept`**
- **Question**: "Which object could be repurposed as a reflector?"
- **Filter**: Liquid physical property OR smooth texture OR (Metals/Glass/Ceramic materials)
- **Includes**: Liquid objects in Outdoor Environments (lakes, rivers, streams, etc.)
- **Excludes**: Various equipment/infrastructure affordance and function clusters
- **Purpose**: Tests understanding of light reflection and signaling properties

**`repurposing_cushion_concept`**
- **Question**: "Which object could be repurposed as a cushion?"
- **Filter**: Flexible property OR Textiles materials
- **Purpose**: Tests understanding of comfort and protection properties

**`repurposing_stepstool_concept`**
- **Question**: "Which object could be repurposed as a stepstool?"
- **Filter**: Rigid, stable, movable, flat surfaces
- **Excludes**: Various equipment/infrastructure affordance and function clusters, furniture
- **Purpose**: Tests understanding of elevation and reaching properties

**`repurposing_bookend_concept`**
- **Question**: "Which object could be repurposed as a bookend?"
- **Filter**: Rigid, stable, movable, upright orientation
- **Purpose**: Tests understanding of organization and support properties

**CoT Reasoning**: "To identify which object could be repurposed as {concept_name}, I examine the physical properties, material composition, and shape characteristics of each object. Among {objects}, the {answer} has the necessary properties ({new_purpose}) to serve as {concept_name} while others lack these key characteristics."

### 2.2 Counterfactual Questions

Counterfactual questions test understanding of how objects would be affected in hypothetical scenarios.

**`counterfactual_water`**
- **Question**: "If water spills, which object gets damaged first?"
- **Filter**: Objects made of water-sensitive materials:
  - Paper, Cardboard & Pulp
  - Biological (Plants/Flowers)
  - Organic Food & Edible Matter
- **CoT Reasoning**: "To identify which object would be damaged first by water, I examine the material composition of each object. Among {objects}, the {answer} is made of water-sensitive materials that would be damaged by moisture while others are more water-resistant."

**`counterfactual_heat`**
- **Question**: "Which object would be most affected by high heat?"
- **Filter**: Objects made of heat-sensitive materials:
  - Textiles, Fibers & Leather
  - Plastics, Rubber & Polymers
  - Paper, Cardboard & Pulp
- **CoT Reasoning**: "To identify which object would be most affected by high heat, I examine the material composition and thermal properties of each object. Among {objects}, the {answer} is made of heat-sensitive materials that would be damaged by high temperatures while others are more heat-resistant."

### 2.3 Compositional Questions

Compositional questions test logical reasoning about property combinations with exclusions.

**`compositional_set_subtraction_container`**
- **Question**: "Which object is rigid, movable, but CANNOT be a container? (exclude living organism as choice)"
- **Required Properties**: rigid, movable
- **Forbidden Properties**: contain (checked via affordance cluster "Contain / Carry / Package" OR container-related keywords in object name OR related affordance clusters with container keywords), hollow
- **Filter**: Objects must be rigid and movable but NOT designed as containers
- **CoT Reasoning**: "To identify which object meets these compositional requirements, I examine the physical properties and affordances of each object. Among {objects}, the {answer} is rigid and movable but cannot function as a container, while others either lack these properties or can serve as containers."

**`compositional_set_subtraction_hollow`**
- **Question**: "Which object is hollow?"
- **Required Properties**: hollow, rigid
- **Forbidden Properties**: contain (same filtering as above for container exclusion)
- **Filter**: Objects that are hollow and rigid but NOT designed as containers
- **CoT Reasoning**: "To identify which object is hollow but not designed as a container, I examine the physical properties and design purpose of each object. Among {objects}, the {answer} has a hollow structure but is not designed for containment while others either lack hollow structure or are designed as containers."

### 2.4 Latent State Questions

Latent state questions test understanding of hidden or non-obvious properties.

**`latent_containment`**
- **Question**: "Which object can hide small items while keeping the area tidy?"
- **Filter**: Objects with BOTH container affordance AND hollow property
- **Excludes**: Equipment, infrastructure, vehicles, sports equipment (via affordance/function clusters), fragile objects
- **CoT Reasoning**: "To identify which object can hide items while keeping things tidy, I examine the structural properties and design purpose of each object. Among {objects}, the {answer} has hidden storage capabilities while others lack this feature."

**`latent_compressible`**
- **Question**: "Which object can be compressed to fit in tight spaces?"
- **Filter**: Flexible property AND compressible materials (Textiles, Plastics, Paper)
- **CoT Reasoning**: "To identify which object can be compressed, I examine the material and physical properties of each object. Among {objects}, the {answer} is made of compressible materials while others are rigid and cannot be compressed."

---

## 3. Spatial Question Types

Spatial questions test understanding of relative positions and distances between objects in 3D space.

**Limit**: Random 0, 1, or 2 spatial questions per image (optional, not required)

### 3.1 Spatial Relationship Questions

**`spatial_left_right`**
- **Question Template**: "Is {object1} to the left or right of {object2}?"
- **Calculation**: Uses 3D coordinates from annotations and 2D bounding box overlap detection
- **Answer**: "left" or "right" based on relative positions

**`spatial_above_below`**
- **Question Template**: "Is {object1} above or below {object2}?"
- **Calculation**: Uses 3D coordinates and 2D bounding box overlap
- **Answer**: "above" or "below"

**`spatial_front_behind`**
- **Question Template**: "Is {object1} in front of or behind {object2}?"
- **Calculation**: Uses 3D coordinates and depth information
- **Answer**: "front" or "behind"

**`spatial_closer_to_camera`**
- **Question Template**: "Is {object1} or {object2} closer to the camera?"
- **Calculation**: Uses 3D distance from camera position
- **Answer**: Object name that is closer

**Filter Conditions**:
- Requires at least 2 objects with valid spatial annotations
- Checks for 2D bounding box overlap to avoid ambiguous questions
- Uses `SpatialUtils.calculate_3d_distance()` for distance confirmation
- Ambiguous relationships (overlapping bboxes) are excluded

**CoT Reasoning**: "To determine the spatial relationship, I examine the 3D coordinates and 2D bounding boxes of the objects. Based on their relative positions, {object1} is {relationship} {object2}."

---

## 4. Question Generation Process

### 4.1 QA Space Analysis

Before generation, a comprehensive QA space analysis is performed:
1. All objects in the taxonomy are tested against each question type's filter criteria
2. Valid objects for each question type are identified
3. Results are stored in `scripts/analysis/results/question_answer_space_analysis.json`

### 4.2 Generation for Real Images

For each image:
1. **Identify available objects**: Objects detected in the image that are in the QA space
2. **Generate from QA space**: For each object, check if it's valid for a question type AND no other objects in the scene are also valid (conflict detection)
3. **Apply additional filters**:
   - Taxonomy cluster conflicts: Skip if multiple objects share the same relevant taxonomy cluster
   - Material uniqueness: For `material_property`, ensure material string is unique
   - Description conflicts: For `description_matching`, ensure descriptions are distinct
4. **Apply limits**:
   - Easy questions: Up to 4 per image
   - Spatial questions: Random 0-2 per image
   - Hard questions: All valid questions
   - **Total cap: 10 questions per image** (prioritizing hard questions)
5. **Format questions**: Replace object names with colored boxes for visual presentation

### 4.3 Conflict Detection

To prevent ambiguous questions, the pipeline implements several conflict detection mechanisms:

1. **Taxonomy Cluster Conflicts**: For taxonomy-based questions (`function_knowledge`, `material_property`, `affordance_*`, `functional_*`), if multiple objects in the scene belong to the same relevant taxonomy cluster, the question is skipped.

2. **Cross-Cluster Conflicts**: Special handling for related clusters:
   - `affordance_enclosures_and_venues_(enter_use)` vs `affordance_build__span__occupy`: Skip if objects from the other cluster are present

3. **Material Uniqueness**: For `material_property`, the actual material string must be unique among scene objects.

4. **Description Source Tracking**: For `description_matching`, if the description comes from a taxonomy field (material/affordance/function/texture), cluster conflict checks are applied.

---

## 5. Output Format

Each question in the benchmark includes:

```json
{
  "question": "Which object is made of 'cotton'?",
  "answer": "Red box",
  "original_question": "Which object is made of 'cotton'?",
  "original_answer": "t-shirt",
  "question_type": "material_property",
  "question_category": "material",
  "question_index": 1234,
  "image_id": "abc123",
  "image_path": "abc123/image.jpg",
  "objects": ["t-shirt", "jeans", "hat"],
  "choices": ["t-shirt", "jeans", "hat"],
  "box_to_object": {
    "Red box": "t-shirt",
    "Green box": "jeans",
    "Blue box": "hat"
  },
  "reasoning": "To identify which object is made of cotton, I examine...",
  "answer_object": "t-shirt"
}
```

**Key Fields**:
- `question`: Question text with colored boxes (e.g., "Red box")
- `answer`: Answer as colored box name
- `original_question`: Question with object names
- `original_answer`: Answer as object name
- `box_to_object`: Mapping from colored box names to object names (source of truth for bbox colors)
- `question_type`: Detailed question type identifier
- `question_category`: Simplified category (material, function, spatial, etc.)

---

## 6. Question Type Statistics (Real Images)

Based on current benchmark generation:

| Question Type Category | Approximate Percentage |
|------------------------|------------------------|
| description_matching | ~28.6% |
| function_knowledge | ~27.9% |
| material_property | ~27.8% |
| spatial_* (combined) | ~21% |
| affordance_* | ~5-10% |
| functional_* | ~1-2% |
| repurposing_* | ~2-3% |
| counterfactual_* | ~1% |
| compositional_* | ~1% |
| latent_* | <1% |

**Total Questions**: ~27,000+ across all images in benchmark

---

## 7. Notes

### 7.1 Material Questions
- **Only `material_property` is used** in the real image pipeline (uses actual material strings from annotations)
- Specific material cluster variants (`material_metals_and_alloys`, `material_textiles_*`, etc.) are **NOT used** in real images (only in simulated images)
- Material inference questions (`material_sound_absorption`, `material_thermal_touch`, `material_scratch_resistance`) are **NOT used** in real images

### 7.2 Question Limits
- The pipeline prioritizes quality over quantity
- Hard questions are always kept (up to the 10-question cap)
- Easy questions are limited to prevent overwhelming easy matching questions
- Spatial questions are optional (0-2 randomly selected)

### 7.3 Bounding Box Alignment
- All questions include a `box_to_object` mapping that reflects the actual colors used when drawing bounding boxes
- This ensures questions match the visual presentation exactly
- Object names are replaced with colored box references in the final question text

---

## Conclusion

The Taxonomy QA Benchmark for real images focuses on high-quality, unambiguous questions that test comprehensive visual understanding through:
- Material properties (using actual material annotations)
- Functional knowledge
- Affordances
- Spatial relationships
- Creative repurposing scenarios
- Counterfactual reasoning
- Compositional logic
- Latent properties

All questions include detailed Chain-of-Thought reasoning explaining the answer selection process using taxonomic knowledge, physical properties, and visual characteristics of objects.
