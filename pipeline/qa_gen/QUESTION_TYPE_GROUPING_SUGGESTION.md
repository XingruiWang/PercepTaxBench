# Question Type Grouping Suggestion

## Overview

Group detailed question types (e.g., `affordance_furniture`, `material_property`) into simplified labels (e.g., `affordance`, `material`) for JSON output, while keeping detailed types in code.

## Current vs Proposed JSON Format

### Current JSON Format
```json
{
  "question": "Which object is furniture?",
  "question_type": "affordance_furniture",
  "answer": "chair",
  ...
}
```

### Proposed JSON Format (with grouping)
```json
{
  "question": "Which object is furniture?",
  "question_category": "affordance",            // Simplified category (replaces question_type in JSON)
  "answer": "chair",
  ...
}
```

**Note**: Detailed `question_type` (e.g., `affordance_furniture`) is still used in code/logging, but JSON output uses simplified `question_category` (e.g., `affordance`).

## Grouping Mappings

### Simple Grouping (Recommended)

All question types are grouped by their prefix into simple categories:

| Detailed Type | Simplified Category |
|--------------|-------------------|
| `affordance_furniture` | `affordance` |
| `affordance_contain__carry__package` | `affordance` |
| `affordance_*` (all 22 types) | `affordance` |
| `material_property` | `material` |
| `material_metals_and_alloys` | `material` |
| `material_*` (all 19 types) | `material` |
| `spatial_above_below` | `spatial` |
| `spatial_left_right` | `spatial` |
| `spatial_*` (all 6 types) | `spatial` |
| `repurposing_cushion_concept` | `repurposing` |
| `repurposing_*` (all 8 types) | `repurposing` |
| `counterfactual_water` | `counterfactual` |
| `counterfactual_heat` | `counterfactual` |
| `compositional_set_subtraction_*` | `compositional` |
| `latent_containment` | `latent` |
| `functional_seating` | `functional` |
| `description_matching` | `description` |
| `function_knowledge` | `function` |

### Category Summary

- **affordance**: 22 question types → 1 category
- **material**: 19 question types → 1 category
- **spatial**: 6 question types → 1 category
- **repurposing**: 8 question types → 1 category
- **compositional**: 6 question types → 1 category
- **counterfactual**: 2 question types → 1 category
- **latent**: 2 question types → 1 category
- **functional**: 3 question types → 1 category
- **description**: 1 question type → 1 category
- **function**: 1 question type → 1 category

**Total**: ~70 detailed types → 10 simplified categories

## Usage Example

### In Code (Use Detailed Types)

The code continues using detailed question types as before:
```python
question_type = "affordance_furniture"  # Detailed type (used in code)
```

### Use Simplified Category for JSON Output

In the question generation/saving code, replace `question_type` with `question_category`:
```python
from modules.qa_modules.question_type_grouping import get_simplified_question_type

# When saving questions to JSON - use simplified category instead of detailed type
question['question_category'] = get_simplified_question_type(detailed_type)  # e.g., "affordance"
# Note: Don't save detailed question_type to JSON, only use simplified category
```

### Example Transformation

**Before:**
```json
{
  "question": "Which object is furniture?",
  "question_type": "affordance_furniture",
  "answer": "chair"
}
```

**After:**
```json
{
  "question": "Which object is furniture?",
  "question_category": "affordance",           // Simplified category (replaces question_type in JSON)
  "answer": "chair"
}
```

**Code still uses detailed type:**
- Code/logging: `question_type = "affordance_furniture"` (detailed)
- JSON output: `question_category = "affordance"` (simplified)

## Benefits

1. **Easy Analysis**: Group questions by category for statistics
   - "How many affordance questions?" → Count all `question_category == "affordance"`
   - "Distribution of question types" → Group by `question_category`

2. **Simplified Filtering**: Filter by broad categories
   - "Show all material questions" → Filter `question_category == "material"`

3. **Cleaner JSON**: Simplified labels in output files
   - Easier to read and analyze
   - Less verbose than detailed types

4. **Flexible**: Can add more granular grouping later
   - Simple grouping: all affordance → "affordance"
   - Granular grouping: furniture-related → "affordance_furniture"

5. **Code Logic Unchanged**: Detailed types still used in code
   - Code continues using `question_type = "affordance_furniture"` for logic
   - Only JSON output uses simplified `question_category`

## Implementation Notes

### Where to Add

1. **In `generate_taxonomyqabench_realimage.py`** - When saving questions:
   ```python
   from modules.qa_modules.question_type_grouping import get_simplified_question_type
   
   for question in all_questions:
       # Replace question_type with question_category for JSON output
       detailed_type = question.pop('question_type', None)  # Remove detailed type
       question['question_category'] = get_simplified_question_type(detailed_type)  # Add simplified category
   ```

2. **In `generate_taxonomyqabench_simimage.py`** - Same pattern:
   ```python
   for question in questions:
       detailed_type = question.pop('question_type', None)
       question['question_category'] = get_simplified_question_type(detailed_type)
   ```

3. **In `unified_qa_generation_utils.py`** - When creating question objects, you can keep `question_type` in the object for code logic, then convert when saving:
   ```python
   question_obj = {
       'question': question_text,
       'question_type': question_type,  # Keep for code logic (can remove later)
       ...
   }
   # Later when saving: convert to question_category
   ```

### Testing

Test the grouping utility:
```bash
python taxonomy_datagen/SpatialReasonerDataGen/qa_gen/scripts/modules/qa_modules/question_type_grouping.py
```

## Example Analysis Queries

With `question_category` field, you can easily:

```python
# Count questions by category
category_counts = {}
for q in questions:
    cat = q.get('question_category', 'unknown')  # Note: field is now 'question_category', not 'question_type'
    category_counts[cat] = category_counts.get(cat, 0) + 1

# Filter affordance questions
affordance_qs = [q for q in questions if q.get('question_category') == 'affordance']

# Get material question distribution
material_qs = [q for q in questions if q.get('question_category') == 'material']
```

## Recommendation

**Use simple grouping** (all affordance_* → "affordance") because:
- Clear and intuitive
- Easy to understand in analysis
- Sufficient for most use cases
- Can add granular grouping later if needed

The utility module is ready to use - just import and call `get_simplified_question_type()` when saving questions to JSON, and replace `question_type` with `question_category` in the output.

