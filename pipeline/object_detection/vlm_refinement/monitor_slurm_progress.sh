#!/bin/bash

BASE_DIR="/path/to/project/openimages_unified_output"
PROGRESS_DIR="/path/to/SpatialReasonerDataGen/object_detection/vlm_refinement"
TOTAL_FILES=9079

while true; do
    clear
    echo "============================================================"
    echo "VLM Object Name Refinement Progress Monitor (SLURM)"
    echo "============================================================"
    echo ""
    
    # Count refined files
    REFINED_COUNT=$(find "$BASE_DIR" -name "*_refined.json" 2>/dev/null | wc -l)
    
    # Calculate progress
    PROGRESS_PCT=$(echo "scale=1; $REFINED_COUNT * 100 / $TOTAL_FILES" | bc)
    
    # Create progress bar
    BAR_LENGTH=50
    FILLED=$(echo "scale=0; $REFINED_COUNT * $BAR_LENGTH / $TOTAL_FILES" | bc)
    EMPTY=$((BAR_LENGTH - FILLED))
    
    echo "Total files to process: $TOTAL_FILES"
    echo "Files completed: $REFINED_COUNT ($PROGRESS_PCT%)"
    echo ""
    
    # Progress bar
    printf "Progress: ["
    for ((i=0; i<$FILLED; i++)); do printf "#"; done
    for ((i=0; i<$EMPTY; i++)); do printf "."; done
    printf "]\n"
    echo ""
    
    # Check worker progress files
    echo "Worker Status:"
    TOTAL_REFINEMENTS=0
    for worker_file in "$PROGRESS_DIR"/vlm_refinement_batched_results_worker*_progress.json; do
        if [ -f "$worker_file" ]; then
            WORKER_ID=$(basename "$worker_file" | grep -oP 'worker\K[0-9]+')
            PROCESSED=$(jq -r '.processed_files // 0' "$worker_file" 2>/dev/null)
            TOTAL=$(jq -r '.total_files // 0' "$worker_file" 2>/dev/null)
            REFINEMENTS=$(jq -r '.total_refinements // 0' "$worker_file" 2>/dev/null)
            TIMESTAMP=$(jq -r '.timestamp // "N/A"' "$worker_file" 2>/dev/null)
            echo "  Worker $WORKER_ID: $PROCESSED/$TOTAL files, $REFINEMENTS refinements (Last: $TIMESTAMP)"
            TOTAL_REFINEMENTS=$((TOTAL_REFINEMENTS + REFINEMENTS))
        fi
    done
    
    if [ $TOTAL_REFINEMENTS -gt 0 ]; then
        echo ""
        echo "Total refinements made: $TOTAL_REFINEMENTS"
    fi
    echo ""
    
    # Estimate time remaining (assuming 0.8 seconds per file with 2 workers)
    REMAINING=$((TOTAL_FILES - REFINED_COUNT))
    TIME_REMAINING=$(echo "scale=1; $REMAINING * 0.8 / 2 / 60" | bc)
    echo "Estimated time remaining: $TIME_REMAINING minutes"
    
    # Check SLURM job status
    echo ""
    echo "SLURM Jobs:"
    squeue -u $USER -o "%.10i %.9P %.20j %.8T %.10M %.6D %R" | grep -E "JOBID|vlm_refine" || echo "  No active VLM refinement jobs"
    
    # Show recent log entries
    echo ""
    echo "Recent Activity (last 5 log entries):"
    LATEST_LOG=$(ls -t logs/vlm_refinement_*.out 2>/dev/null | head -1)
    if [ -f "$LATEST_LOG" ]; then
        tail -5 "$LATEST_LOG" | sed 's/^/  /'
    else
        echo "  No log file found"
    fi
    
    echo ""
    echo "============================================================"
    echo "Press Ctrl+C to exit monitoring"
    echo "View full logs: tail -f logs/vlm_refinement_*.out"
    echo ""
    
    sleep 15
done

