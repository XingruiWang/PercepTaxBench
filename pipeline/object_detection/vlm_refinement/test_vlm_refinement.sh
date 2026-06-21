#!/bin/bash

echo "========================================="
echo "Testing VLM Object Name Refinement"
echo "========================================="
echo ""

cd /path/to/SpatialReasonerDataGen/qa_gen

# Test on specific images we know have refinement opportunities
# 80ba9d77fb3de066 - has "army" → should be "soldier"
# 34026f49202b936a - has multiple "person" → should be "man/woman/child"

echo "Test images:"
echo "  1. 80ba9d77fb3de066 - army → soldier"
echo "  2. 34026f49202b936a - person → man/woman/child"
echo ""

# You need to set your API keys here
API_KEYS="YOUR_API_KEY_1,YOUR_API_KEY_2"

python refine_object_names_with_vlm.py \
    --unified_dir /path/to/project/openimages_unified_output \
    --api_keys "$API_KEYS" \
    --max_files 5 \
    --output vlm_refinement_test_results.json \
    --candidates_file detection_tag_analysis_refined_mappings.json

echo ""
echo "Test complete! Check results in vlm_refinement_test_results.json"
echo ""
echo "To view refined annotations:"
echo "  cat /path/to/project/openimages_unified_output/80ba9d77fb3de066/annotations/80ba9d77fb3de066_refined.json"

