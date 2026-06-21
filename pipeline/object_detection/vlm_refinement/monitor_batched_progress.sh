#!/bin/bash

cd /path/to/SpatialReasonerDataGen/object_detection/vlm_refinement

echo "=== VLM REFINEMENT PROGRESS (BATCHED & PARALLEL) ==="
echo ""

# Check if process is running
if ps -p $(cat vlm_refinement_batched.pid 2>/dev/null) > /dev/null 2>&1; then
    echo "✅ Status: RUNNING"
else
    echo "❌ Status: NOT RUNNING"
fi

echo ""
echo "📊 Worker Progress:"
echo ""

# Worker 0 progress
W0_PROGRESS=$(grep "Worker 0: \[" logs/vlm_refinement_batched_v2.log 2>/dev/null | tail -1)
if [ ! -z "$W0_PROGRESS" ]; then
    echo "  Worker 0: $W0_PROGRESS"
fi

# Worker 1 progress
W1_PROGRESS=$(grep "Worker 1: \[" logs/vlm_refinement_batched_v2.log 2>/dev/null | tail -1)
if [ ! -z "$W1_PROGRESS" ]; then
    echo "  Worker 1: $W1_PROGRESS"
fi

echo ""
echo "🔄 Recent Refinements (last 10):"
grep "Refined:" logs/vlm_refinement_batched_v2.log 2>/dev/null | tail -10 | sed 's/^.*INFO - /  /'

echo ""
echo "📈 Total Refined Files Created:"
echo "  $(find /path/to/project/openimages_unified_output -name '*_refined.json' | wc -l) files"

echo ""
echo "⏰ Runtime:"
if [ -f vlm_refinement_batched.pid ]; then
    ps -p $(cat vlm_refinement_batched.pid) -o etime= 2>/dev/null | xargs echo "  " || echo "  N/A"
fi

echo ""
echo "Monitor live: tail -f logs/vlm_refinement_batched_v2.log"

