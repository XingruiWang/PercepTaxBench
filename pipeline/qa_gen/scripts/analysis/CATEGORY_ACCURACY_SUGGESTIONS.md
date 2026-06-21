# Category Accuracy Analysis - Suggestions for Enhancements

## Current Features ✅

The script `analyze_category_accuracy.py` currently provides:
- ✅ Category-wise accuracy breakdown
- ✅ Detailed question type accuracy (if available)
- ✅ Overall statistics
- ✅ JSON output with structured results
- ✅ Command-line interface

## Results Summary

From the test run, we can see:
- **Best performing**: Affordance (76.19%), Material (66.67%), Capability (66.67%)
- **Worst performing**: Counterfactual (27.27%), Spatial (42.86%), Compositional (52.63%)

## Suggested Enhancements

### 1. **Answer Normalization** (High Priority)
Some models might predict "Red Box" vs "Red box" (case sensitivity). Add normalization:
```python
def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    return answer.strip().lower()
```

**Impact**: Could improve accuracy by catching case-insensitive matches

### 2. **Error Analysis Report** (High Priority)
Generate a detailed error report showing:
- Most common incorrect predictions per category
- Questions with highest confidence errors
- Pattern analysis (e.g., "always predicts Red box for spatial questions")

**Example output:**
```json
{
  "error_analysis": {
    "affordance": {
      "most_common_error": {
        "ground_truth": "Red box",
        "prediction": "Green box",
        "count": 5
      },
      "error_patterns": [
        {"pattern": "color_swap", "count": 8},
        {"pattern": "wrong_object", "count": 7}
      ]
    }
  }
}
```

### 3. **Confusion Matrix Analysis** (Medium Priority)
Add per-category confusion matrices to understand error patterns:
- Which answer choices are confused with each other?
- Are there systematic errors (e.g., always predicting "Red box")?

**Implementation:**
```python
# Track confusion: ground_truth -> predicted_answer
confusion_matrix = defaultdict(lambda: defaultdict(int))
```

### 4. **CSV Export** (Medium Priority)
Export results to CSV for easy spreadsheet analysis:
```bash
--export-csv results.csv
```

**Use case**: Easy to filter, sort, and visualize in Excel/Google Sheets

### 5. **Visualization Support** (Medium Priority)
Add optional matplotlib/plotly visualizations:
- Bar chart of accuracy by category
- Heatmap of confusion matrices
- Error distribution charts

**Command-line flag:**
```bash
--visualize  # Generate plots
--output-dir results/  # Save plots here
```

### 6. **Filtering Options** (Low Priority)
Allow filtering analysis:
```bash
--min-samples 10  # Only show categories with >= 10 samples
--categories affordance spatial material  # Only analyze specific categories
--exclude-categories counterfactual  # Exclude certain categories
```

### 7. **Statistical Significance Testing** (Low Priority)
For comparing categories:
- Which categories perform significantly better/worse?
- Confidence intervals for accuracy estimates
- Chi-square tests for independence

### 8. **Model Comparison** (Low Priority)
Compare multiple model evaluations:
```bash
python analyze_category_accuracy.py \
    --eval-results model1.json model2.json model3.json \
    --questions questions.json \
    --compare
```

### 9. **Question Difficulty Analysis** (Low Priority)
Correlate accuracy with:
- Question length
- Number of objects in scene
- Question type complexity
- Image complexity (if metadata available)

### 10. **Integration with Metadata** (Low Priority)
Use `generation_metadata.json` to:
- Correlate with scene statistics
- Analyze by scene category
- Cross-reference with QA space analysis

## Quick Wins (Easy to Implement)

1. **Answer normalization** (case-insensitive matching) - ~10 lines
2. **CSV export** - ~20 lines
3. **Most common errors** per category - ~30 lines
4. **Filtering by minimum samples** - ~15 lines

## Example Enhanced Usage

```bash
# Basic analysis (current)
python analyze_category_accuracy.py \
    --eval-results eval.json \
    --questions questions.json

# With answer normalization
python analyze_category_accuracy.py \
    --eval-results eval.json \
    --questions questions.json \
    --normalize-answers

# With error analysis
python analyze_category_accuracy.py \
    --eval-results eval.json \
    --questions questions.json \
    --error-analysis \
    --output-dir results/

# With visualizations
python analyze_category_accuracy.py \
    --eval-results eval.json \
    --questions questions.json \
    --visualize \
    --output-dir results/

# Filtered analysis
python analyze_category_accuracy.py \
    --eval-results eval.json \
    --questions questions.json \
    --categories affordance spatial material \
    --min-samples 20 \
    --export-csv results.csv
```

## Current Output Structure

The script outputs:
```json
{
  "summary": {
    "total_evaluated": 200,
    "total_correct": 125,
    "overall_accuracy": 0.625,
    "matched_questions": 200,
    "unmatched_questions": 0
  },
  "category_accuracy": {
    "affordance": {
      "total": 63,
      "correct": 48,
      "incorrect": 15,
      "accuracy": 0.7619,
      "percentage": "76.19%"
    },
    ...
  },
  "detailed_type_accuracy": {
    "affordance_furniture": {
      "total": 10,
      "correct": 8,
      "accuracy": 0.8,
      ...
    },
    ...
  }
}
```

## Notes

- The script currently matches questions by `question_index` from eval results to `question_index` in questions JSON
- If `question_type` field is missing in questions JSON, detailed type accuracy will show as "unknown"
- All questions are matched successfully (200/200 in test run)

