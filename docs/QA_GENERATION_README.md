# Question-Answer Pair Generation System

This system automatically generates diverse question-answer pairs from 3D ground truth annotations using Google Gemini API. It creates comprehensive QA pairs covering object attributes, spatial relationships, functions, and scene understanding.

## Features

### 🎯 **QA Categories Covered**
1. **Object Attributes**
   - Physical properties (material, color, shape, texture, size)
   - State/condition (fullness, cleanliness, damage, age)
   - Quantitative features (counts, dimensions, weight, volume)

2. **Spatial Relationships**
   - Relative positions (above, below, left, right, in front, behind)
   - Proximity (near, far, adjacent, surrounding)
   - Orientation (facing, pointing, aligned, perpendicular)

3. **Object Functions**
   - Human-object interactions (grasping, sitting, writing, cooking)
   - Object-object interactions (supporting, containing, connecting)
   - Purpose and affordances (storage, transportation, communication)

4. **Scene Understanding**
   - Location classification (indoor/outdoor, room types)
   - Activity inference (cooking, working, relaxing, socializing)
   - Context understanding (time, weather, environment)

5. **Quantitative Features**
   - Object counts and distributions
   - Spatial measurements and distances
   - Size comparisons and proportions

### 🚀 **Key Capabilities**
- **Intelligent Analysis**: Automatically analyzes 3D poses, semantic attributes, and spatial relationships
- **Gemini API Integration**: Uses Google's latest AI model for natural, diverse question generation
- **Fallback Generation**: Template-based QA generation when API is unavailable
- **Batch Processing**: Process multiple annotation files in parallel
- **Comprehensive Reports**: Detailed analysis and statistics for each annotation
- **Quality Control**: Difficulty levels and category classification for each QA pair

## Installation

### 1. Install Dependencies
```bash
cd taxonomy_datagen/SpatialReasonerDataGen
pip install -r requirements_qa.txt
```

### 2. Get Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Use one of these methods to provide the key:
   - **Environment Variable** (Recommended): `export GOOGLE_GEMINI_API_KEY='your_api_key'`
   - **Command Line**: `--api_key your_api_key`
   - **Edit Script**: Modify the default value in the script

## Usage

### 🎯 **Single Annotation Processing**

```bash
python generate_qa_pairs.py \
    --annotation_path /path/to/annotation.json \
    --output_dir /path/to/output \
    --api_key YOUR_GEMINI_API_KEY \
    --generate_report
```

**Arguments:**
- `--annotation_path`: Path to the annotation JSON file
- `--output_dir`: Directory to save QA pairs and reports
- `--api_key`: Your Google Gemini API key (default: uses GOOGLE_GEMINI_API_KEY env var)
- `--model_name`: Gemini model to use (default: gemini-1.5-flash)
- `--generate_report`: Generate detailed summary report

**API Key Configuration Options:**
1. **Environment Variable** (Recommended):
   ```bash
   export GOOGLE_GEMINI_API_KEY='your_actual_api_key'
   python generate_qa_pairs.py --annotation_path file.json --output_dir output/
   ```

2. **Command Line Argument**:
   ```bash
   python generate_qa_pairs.py --annotation_path file.json --output_dir output/ --api_key your_actual_api_key
   ```

3. **Edit Script Default**:
   Edit the `default='your_gemini_api_key_here'` line in the script

### 🔄 **Batch Processing**

```bash
# Edit the API key in the script first
nano batch_generate_qa.sh

# Make executable and run
chmod +x batch_generate_qa.sh
./batch_generate_qa.sh
```

**Configuration in batch script:**
- `API_KEY`: Your Gemini API key
- `OUTPUT_DIR`: Output directory for all QA pairs
- `ANNOTATIONS_DIR`: Directory containing annotation files
- `MAX_PARALLEL_JOBS`: Maximum concurrent processing jobs

### 🧪 **Testing Without API**

```bash
python test_qa_generation.py
```

This tests the system with sample data without requiring API calls.

## Output Structure

### 📁 **Generated Files**
```
output_directory/
├── annotation_name_qa_pairs.json     # Main QA pairs file
├── annotation_name_qa_report.txt     # Detailed analysis report
└── batch_summary.txt                 # Overall batch processing summary
```

### 📊 **QA Pairs Format**
```json
{
  "metadata": {
    "generated_at": "2025-08-27T12:00:00",
    "total_qa_pairs": 18,
    "categories": ["Object Attributes", "Spatial Relationships", "Scene Understanding"],
    "difficulties": ["Easy", "Medium", "Hard"]
  },
  "qa_pairs": [
    {
      "question": "What color is the chair?",
      "answer": "The chair is brown.",
      "category": "Object Attributes",
      "difficulty": "Easy"
    },
    {
      "question": "Where is the person relative to the table?",
      "answer": "The person is in front of the table.",
      "category": "Spatial Relationships",
      "difficulty": "Medium"
    }
  ]
}
```

### 📋 **Report Format**
```
QA Generation Summary Report
============================

Generated at: 2025-08-27 12:00:00

Scene Analysis:
- Total objects detected: 3
- Objects with pose data: 3
- Spatial relationships analyzed: 3
- Environment type: indoor_living

QA Pairs Generated:
- Total QA pairs: 18
- Categories: Object Attributes: 8, Spatial Relationships: 5, Scene Understanding: 5
- Difficulties: Easy: 10, Medium: 6, Hard: 2

Object Details:
- person #1: person (blue, rectangular, large) - facing_forward
- chair #1: chair (brown, rectangular, medium) - facing_forward
- table #1: table (brown, rectangular, large) - facing_forward
```

## Example QA Pairs

### 🎨 **Object Attributes**
- **Q**: What material is the chair made of?
- **A**: The chair is made of wood.
- **Category**: Object Attributes
- **Difficulty**: Easy

- **Q**: What is the shape of the table?
- **A**: The table is rectangular.
- **Category**: Object Attributes
- **Difficulty**: Easy

### 🗺️ **Spatial Relationships**
- **Q**: Where is the person relative to the chair?
- **A**: The person is in front of the chair.
- **Category**: Spatial Relationships
- **Difficulty**: Medium

- **Q**: Is the chair above or below the table?
- **A**: The chair is above the table.
- **Category**: Spatial Relationships
- **Difficulty**: Medium

### 🏠 **Scene Understanding**
- **Q**: What type of environment is this scene?
- **A**: This is an indoor living environment.
- **Category**: Scene Understanding
- **Difficulty**: Easy

- **Q**: What activity is likely happening in this scene?
- **A**: Someone is sitting and working at a table.
- **Category**: Scene Understanding
- **Difficulty**: Hard

## Advanced Features

### 🔍 **Pose Analysis**
- Automatically interprets 3D Euler angles into human-readable orientations
- Analyzes relative positions between objects with 3D pose data
- Validates pose consistency and physical plausibility

### 🎯 **Semantic Attribute Extraction**
- Leverages semantic annotations for color, shape, size, and material
- Generates attribute-specific questions based on available data
- Falls back gracefully when semantic data is incomplete

### 📐 **Spatial Relationship Detection**
- Calculates relative positions in 3D space
- Identifies proximity and orientation relationships
- Generates spatial reasoning questions

### 🧠 **Intelligent Question Generation**
- Uses Gemini API for natural, diverse question formulation
- Ensures question variety across difficulty levels
- Maintains consistency with annotation data

## Error Handling

### 🚨 **Common Issues**
1. **API Key Invalid**: Check your Gemini API key and billing status
2. **Rate Limiting**: The system automatically handles API rate limits
3. **Network Issues**: Fallback generation ensures QA pairs are always created
4. **Invalid Annotations**: Graceful handling of malformed annotation data

### 🛠️ **Fallback Mechanisms**
- Template-based QA generation when API fails
- Robust parsing of API responses
- Comprehensive error logging and reporting

## Performance Optimization

### ⚡ **Batch Processing Tips**
- Use `MAX_PARALLEL_JOBS=3` for optimal API usage
- Process annotations in smaller batches for large datasets
- Monitor API usage and costs

### 💾 **Memory Management**
- Processes one annotation at a time to minimize memory usage
- Efficient JSON parsing and data structures
- Automatic cleanup of temporary data

## Integration with Pipeline

### 🔗 **Pipeline Integration**
The QA generation system integrates seamlessly with the 3D ground truth pipeline:

1. **Run 3D Pipeline**: Generate annotations with `--enable_semantic --enable_pose3d`
2. **Generate QA Pairs**: Process annotations to create QA pairs
3. **Review Results**: Check QA quality and adjust prompts if needed
4. **Batch Process**: Scale to process entire datasets

### 📊 **Data Flow**
```
3D Pipeline → Annotations → QA Generator → QA Pairs + Reports
     ↓              ↓            ↓              ↓
  Object      Semantic      Gemini API    Structured
Detection    Attributes    Generation    QA Dataset
```

## Customization

### 🎨 **Custom QA Categories**
Edit the `qa_categories` in `QAGenerator` class to add new categories:

```python
self.qa_categories = {
    "custom_category": {
        "subcategory": ["feature1", "feature2", "feature3"]
    }
}
```

### 🔧 **Custom Prompts**
Modify the `generate_qa_prompt` method to customize the prompt sent to Gemini API.

### 📝 **Template Questions**
Add new template questions in `_generate_fallback_qa_pairs` for fallback generation.

## Troubleshooting

### ❓ **FAQ**
1. **Q**: Why are some QA pairs missing?
   **A**: Check if the annotation has the required fields (detections, semantic_annotations, pose data)

2. **Q**: How to improve QA quality?
   **A**: Refine the prompt in `generate_qa_prompt` method and adjust difficulty thresholds

3. **Q**: API calls failing?
   **A**: Verify API key, check billing status, and ensure network connectivity

### 🐛 **Debug Mode**
Enable detailed logging by modifying the logging level in the script:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Contributing

### 🔧 **Development**
1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Add tests
5. Submit a pull request

### 📋 **Testing**
- Run `test_qa_generation.py` to verify functionality
- Test with various annotation formats
- Validate QA pair quality and diversity

## License

This QA generation system is part of the 3D World project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs and reports
3. Test with sample data
4. Contact the development team

---

**Note**: This system requires a valid Google Gemini API key and internet connectivity for optimal functionality. Fallback generation ensures QA pairs are always created even when the API is unavailable.
