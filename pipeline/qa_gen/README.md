# QA Generation Pipeline

Complete pipeline for generating Taxonomy QA Benchmarks from real and simulated images.

##  Documentation Index

- **[Quick Start](#quick-start)** - Get started immediately
- **[Prerequisites](#prerequisites)** - Setup requirements
- **[Benchmark Generation](#3-taxonomy-benchmark-pipeline)** - Generate benchmarks
- **[Script Organization](#directory-structure)** - Code structure
- **[Technical Details](#technical-documentation)** - Deep dive docs
- **[QA Generation](#qa-generation)** - How questions are generated
- **[Troubleshooting](#troubleshooting)** - Common issues

## Quick Start

### Generate Real Image Benchmark

**Using SLURM** (recommended):
```bash
cd /path/to/SpatialReasonerDataGen/qa_gen/scripts/core
sbatch run_realimage_slurm.sh           # Generate benchmark
sbatch run_realimage_tsv_slurm.sh       # Convert to TSV
```

**Direct execution**:
```bash
cd /path/to/project
cd taxonomy_datagen/SpatialReasonerDataGen/qa_gen/scripts/core
conda activate srdatagen

# Generate benchmark
python generate_taxonomyqabench_realimage.py \
    --images_dir ../../openimages_unified_output \
    --output_dir taxonomyQABench_realimage \
    --seed 42

# Convert to TSV
python aggregate_unified_qa.py \
    --input_dir ../../taxonomyQABench_realimage \
    --output_file ../../../qa_eval/VLMEvalKit/Data/taxonomy.tsv
```

### Generate Sim Image Benchmark

**Using SLURM** (recommended):
```bash
cd /path/to/SpatialReasonerDataGen/qa_gen/scripts/core
sbatch run_simimage_slurm.sh            # Generate benchmark
sbatch run_simimage_tsv_slurm.sh        # Convert to TSV
```

**Direct execution**:
```bash
cd /path/to/project
cd taxonomy_datagen/SpatialReasonerDataGen/qa_gen/scripts/core
conda activate srdatagen

# Generate benchmark
python generate_taxonomyqabench_simimage.py \
    --images_dir /path/to/sim_images/jiawei \
    --output_dir taxonomyQABench_simimage \
    --seed 42

# Convert to TSV (unified script works for both real and sim images)
python aggregate_unified_qa.py \
    --input_dir ../../taxonomyQABench_simimage \
    --output_file ../../../qa_eval/VLMEvalKit/Data/taxonomy_sim.tsv
```

### Output Locations

**Real Image Benchmark**:
- JSON files: `taxonomyQABench_realimage/`
- TSV file: `qa_eval/VLMEvalKit/Data/taxonomy.tsv`

**Sim Image Benchmark**:
- JSON files: `taxonomyQABench_simimage/`
- TSV file: `qa_eval/VLMEvalKit/Data/taxonomy_sim.tsv`

> **Note**: 
> - TSV files have different names (`taxonomy.tsv` vs `taxonomy_sim.tsv`) to allow separate evaluations.
> - The `aggregate_unified_qa.py` script automatically handles both real and sim images by detecting the input format.

---

## Prerequisites

### 1. Conda Environment
```bash
conda activate srdatagen
```

### 2. Working Directory
Always run from the project root:
```bash
cd /path/to/project
```

### 3. Required Data
- `openimages_unified_output/` - Real image annotations
- `scripts/modules/sim_scene_object/data/` - Sim scene data
- Sim images from `/path/to/sim_images/jiawei`

---

## Directory Structure

```
qa_gen/
├── README.md                                    # This file
├── QA_GENERATION.md                             # QA generation details
├── SIM_IMAGE_PIPELINE.md                       # Sim image technical details
├── scene_qa_potential_analysis.json             # Scene QA potential
├── conflict_free_qa_groups.json                # Conflict-free QA groups
├── object_name_mappings.json                   # Object name mappings
│
├── taxonomyQABench_realimage/                  # Real image benchmark output
│   ├── all_questions.json
│   ├── scene_statistics.json
│   ├── generation_metadata.json
│   └── images/                                  # Images with bboxes
│
├── taxonomyQABench_simimage/                   # Sim image benchmark output
│   ├── all_questions.json
│   ├── scene_statistics.json
│   ├── generation_metadata.json
│   └── images/                                  # Images with bboxes
│
├── openimages_unified_output/                   # Real image annotations
│
└── scripts/
    ├── core/                                    # Core generation scripts
    │   ├── generate_taxonomyqabench_realimage.py
    │   ├── generate_taxonomyqabench_simimage.py
    │   ├── aggregate_unified_qa.py
    │   ├── aggregate_unified_qa_sim.py
    │   ├── run_realimage_slurm.sh
    │   ├── run_simimage_slurm.sh
    │   ├── run_realimage_tsv_slurm.sh
    │   ├── run_simimage_tsv_slurm.sh
    │   └── README.md                            # Deprecated (use main README)
    │
    ├── analysis/                                # Analysis scripts
    │   ├── analyze_question_answer_space.py     # QA space analysis
    │   ├── analyze_category_accuracy.py         # Category accuracy analysis
    │   ├── results/                              # Analysis results
    │   │   ├── question_answer_space_analysis.json  # QA space analysis results
    │   │   └── category_accuracy_results.json       # Example accuracy results
    │   └── CATEGORY_ACCURACY_SUGGESTIONS.md     # Enhancement suggestions
    │
    ├── modules/                                 # QA modules
    │   ├── qa_modules/                          # Question utilities
    │   └── sim_scene_object/                    # Sim scene analysis
    │       └── README.md                        # Conflict-free QA groups
    │
    └── README.md                                # Deprecated (use main README)
```

---

## Three Main Pipelines

### 1. QA Space Analysis Pipeline

**Purpose**: Analyzes which objects can answer which question types.

**Location**: `scripts/analysis/`

**Run**:
```bash
cd /path/to/SpatialReasonerDataGen/qa_gen
conda activate srdatagen
python scripts/analysis/analyze_question_answer_space.py
```

**Output**: `scripts/analysis/results/question_answer_space_analysis.json`

### 1b. Category Accuracy Analysis

**Purpose**: Analyzes VLM evaluation results by question category.

**Location**: `scripts/analysis/`

**Run**:
```bash
cd /path/to/SpatialReasonerDataGen/qa_gen
conda activate srdatagen

# Real image benchmark
python scripts/analysis/analyze_category_accuracy.py \\
    --eval-results /path/to/VLMEvalKit/outputs/Model_TaxonomyBench_score.json \\
    --questions taxonomyQABench_realimage_v2/all_questions.json \\
    --output scripts/analysis/category_accuracy_realimage.json

# Sim image benchmark
python scripts/analysis/analyze_category_accuracy.py \\
    --eval-results /path/to/VLMEvalKit/outputs/Model_TaxonomyBench_score.json \\
    --questions taxonomyQABench_simimage/all_questions.json \\
    --output scripts/analysis/category_accuracy_simimage.json
```

**Output**: `category_accuracy_*.json` with per-category accuracy breakdown

---

### 2. Sim Scene Object Analysis Pipeline

**Purpose**: Analyzes synthetic scenes for QA potential and conflict-free groups.

**Location**: `scripts/modules/sim_scene_object/scripts/`

**Run**:
```bash
cd scripts/modules/sim_scene_object/scripts/
conda activate srdatagen

# Analyze scene potential
python analyze_scene_qa_potential.py

# Generate conflict-free groups
python generate_conflict_free_qa_groups.py
```

**Outputs**:
- `scene_qa_potential_analysis.json`
- `conflict_free_qa_groups.json`

---

### 3. Taxonomy Benchmark Pipeline

**Purpose**: Generates complete benchmarks with questions, answers, and images.

**Location**: `scripts/core/`

**Features**:
- Multiple choice questions with correct answers
- Chain-of-Thought reasoning
- Spatial reasoning questions
- Bounding box visualizations
- TSV format for VLM evaluation

**Scripts**:
- Real images: `run_realimage_slurm.sh`
- Sim images: `run_simimage_slurm.sh`
- Convert to TSV: `run_*_tsv_slurm.sh`

See [Quick Start](#quick-start) for usage.

---

## Technical Documentation

For deep technical details, see:

- **[QA_GENERATION.md](QA_GENERATION.md)** - QA generation guide
  - How questions are generated
  - Question types and examples
  - Quality controls
  - CoT reasoning generation
  - Output formats

- **[SIM_IMAGE_PIPELINE.md](SIM_IMAGE_PIPELINE.md)** - Sim image pipeline
  - Data sources and structure
  - Object extraction and filtering
  - Depth-aware grouping
  - Bounding box visualization
  - Question generation process

- **[QUESTION_TYPES_DOCUMENTATION.md](scripts/modules/qa_modules/QUESTION_TYPES_DOCUMENTATION.md)** - Question specifications
  - All question types
  - CoT reasoning templates
  - Filters and constraints

---

## Monitoring SLURM Jobs

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/realimage_qa_*.out
tail -f logs/simimage_qa_*.out

# Check errors
tail -f logs/realimage_qa_*.err
tail -f logs/simimage_qa_*.err
```

---

## Troubleshooting

### Common Issues

1. **Path Issues**: Always run from `/path/to/project`
2. **Missing Data**: Check that input files exist in expected locations
3. **Conda Environment**: Ensure `srdatagen` environment is activated
4. **Permissions**: Make scripts executable: `chmod +x *.sh`

### Check Results

```bash
# Real image results
ls -lh taxonomyQABench_realimage/*.json
ls -lh ../qa_eval/VLMEvalKit/Data/taxonomy.tsv

# Sim image results  
ls -lh taxonomyQABench_simimage/*.json
ls -lh ../qa_eval/VLMEvalKit/Data/taxonomy_sim.tsv

# View question counts
python -c "import json; d=json.load(open('taxonomyQABench_realimage/all_questions.json')); print(f'Total questions: {len(d)}')"
```

---

## Key Features

### Question Types
- Material questions (color, texture, etc.)
- Function questions (purpose, affordance)
- Spatial questions (position, relationships)
- Repurposing questions (unintended uses)
- Counterfactual questions (what if scenarios)

### Quality Controls
- Void cluster filtering (removes meaningless questions)
- Segmentation validation (only visible objects)
- Depth-aware grouping (for sim images)
- Choice validation (matches bbox count)
- Malformed text cleaning

### Output Formats
- **JSON**: Complete questions with metadata
- **TSV**: Compatible with VLMEvalKit
- **Images**: Bounding box visualizations

---

## Pipeline Dependencies

1. **QA Space Analysis** → `scripts/analysis/results/question_answer_space_analysis.json`
2. **Sim Scene Analysis** → `scene_qa_potential_analysis.json`
3. **Taxonomy Benchmark** → Final benchmark output

Run pipelines in order for best results.

---

## Notes

- Pipeline uses random sampling, so results may vary between runs
- All paths are relative to the script directories
- TSV output is VLMEvalKit compatible
- **Image Resize Dimensions**:
  - Sim images: 400x225 (max, maintains aspect ratio from ~1920x1080)
  - Real images: 400x300-400 (varies, max width 400px)
  - Labels drawn after resize with font size 80 for better visibility
