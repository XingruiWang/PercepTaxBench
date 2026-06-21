#!/bin/bash

# Script to convert all four benchmark versions to TSV format
# - Designated properties (real and sim)
# - All properties (real and sim)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QA_GEN_DIR="$SCRIPT_DIR"
SCRIPTS_DIR="$QA_GEN_DIR/scripts/core"
OUTPUT_DIR="/path/to/VLMEvalKit/Data"

echo "=================================================================================="
echo "Converting all benchmark versions to TSV format"
echo "=================================================================================="
echo ""

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# 1. Convert real image benchmark with designated properties
echo "Converting REAL IMAGE benchmark (DESIGNATED PROPERTIES)..."
echo "Input: $QA_GEN_DIR/taxonomyQABench_realimage_final_polished_with_properties"
echo "Output: $OUTPUT_DIR/taxonomy_properties.tsv"
echo ""
python3 "$SCRIPTS_DIR/aggregate_unified_qa.py" \
    --input_dir "$QA_GEN_DIR/taxonomyQABench_realimage_final_polished_with_properties" \
    --output_file "$OUTPUT_DIR/taxonomy_properties.tsv" \
    --target_image_size 512
echo ""

# 2. Convert sim image benchmark with designated properties
echo "Converting SIM IMAGE benchmark (DESIGNATED PROPERTIES)..."
echo "Input: $QA_GEN_DIR/taxonomyQABench_simimage_final_with_properties"
echo "Output: $OUTPUT_DIR/taxonomy_properties_sim.tsv"
echo ""
python3 "$SCRIPTS_DIR/aggregate_unified_qa.py" \
    --input_dir "$QA_GEN_DIR/taxonomyQABench_simimage_final_with_properties" \
    --output_file "$OUTPUT_DIR/taxonomy_properties_sim.tsv" \
    --target_image_size 512
echo ""

# 3. Convert real image benchmark with all properties
echo "Converting REAL IMAGE benchmark (ALL PROPERTIES)..."
echo "Input: $QA_GEN_DIR/taxonomyQABench_realimage_final_polished_with_all_properties"
echo "Output: $OUTPUT_DIR/taxonomy_all_properties.tsv"
echo ""
python3 "$SCRIPTS_DIR/aggregate_unified_qa.py" \
    --input_dir "$QA_GEN_DIR/taxonomyQABench_realimage_final_polished_with_all_properties" \
    --output_file "$OUTPUT_DIR/taxonomy_all_properties.tsv" \
    --target_image_size 512
echo ""

# 4. Convert sim image benchmark with all properties
echo "Converting SIM IMAGE benchmark (ALL PROPERTIES)..."
echo "Input: $QA_GEN_DIR/taxonomyQABench_simimage_final_with_all_properties"
echo "Output: $OUTPUT_DIR/taxonomy_all_properties_sim.tsv"
echo ""
python3 "$SCRIPTS_DIR/aggregate_unified_qa.py" \
    --input_dir "$QA_GEN_DIR/taxonomyQABench_simimage_final_with_all_properties" \
    --output_file "$OUTPUT_DIR/taxonomy_all_properties_sim.tsv" \
    --target_image_size 512
echo ""

echo "=================================================================================="
echo "All conversions complete!"
echo "=================================================================================="
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/taxonomy_properties.tsv (real, designated properties)"
echo "  - $OUTPUT_DIR/taxonomy_properties_sim.tsv (sim, designated properties)"
echo "  - $OUTPUT_DIR/taxonomy_all_properties.tsv (real, all properties)"
echo "  - $OUTPUT_DIR/taxonomy_all_properties_sim.tsv (sim, all properties)"
echo ""

# Show file sizes
echo "File sizes:"
ls -lh "$OUTPUT_DIR"/taxonomy*.tsv 2>/dev/null | awk '{print "  " $9 ": " $5}'
echo ""

