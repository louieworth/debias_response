#!/bin/bash

###############################################################################
# Batch Run Variant Experiments
# Supports both Average (JSON) and Individual (Parquet) prediction
# Results saved to: results/{directory}_{prefix}_result.csv
#
# Configurable parameters (modify at the top of the script)
###############################################################################

# Feature variant options (space-separated, leave empty to run all)
# Available: x_only, x_avg_llm, x_all_llm, x_one_llm
VARIANTS="x_only x_avg_llm x_all_llm x_one_llm"

# Model type options (space-separated, leave empty to run all)
# Available: ridge, rf, mlp, xgboost
MODEL_TYPES="ridge rf mlp xgboost"

# Data file paths (parquet format)
# Examples:
#   - Average prediction: dataset/EEDI/Aggreated/eedi_train.parquet
#   - Individual prediction: dataset/OpinionQA/individual_train.parquet
TRAIN_FILE="dataset/OpinionQA/individual_train.parquet"
TEST_FILE="dataset/OpinionQA/individual_test.parquet"

###############################################################################
# Script content below (usually no need to modify)
###############################################################################

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

# Ensure results directory exists
mkdir -p results

# Infer result file name from training file path
# Handles both JSON and parquet formats
# Examples:
#   - JSON: dataset/EEDI/Aggreated/eedi_train.json -> EEDI_Aggreated_eedi_result.csv
#   - Parquet: dataset/OpinionQA/individual_train.parquet -> OpinionQA_individual_result.csv
TRAIN_DIR=$(basename "$(dirname "$TRAIN_FILE")")
TRAIN_PREFIX=$(basename "$TRAIN_FILE" | sed 's/_train\.json$//' | sed 's/_train\.parquet$//' | sed 's/\.json$//' | sed 's/\.parquet$//')
RESULT_FILE="results/${TRAIN_DIR}_${TRAIN_PREFIX}_result.csv"

# Clear old results (optional)
read -p "Clear old results $RESULT_FILE? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$RESULT_FILE"
    echo "Old results cleared"
fi

# Print configuration
echo "========================================================================"
echo "Batch Run Variant Experiments"
echo "========================================================================"
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo "Result file: $RESULT_FILE"
echo "Variants: $VARIANTS"
echo "Model types: $MODEL_TYPES"
echo "========================================================================"
echo ""

# Counter
TOTAL_EXPERIMENTS=0
for model in $MODEL_TYPES; do
    for variant in $VARIANTS; do
        ((TOTAL_EXPERIMENTS++))
    done
done

CURRENT_EXPERIMENT=0

# Loop through all model types
for model_type in $MODEL_TYPES; do
    echo "========================================================================"
    echo "Model type: $model_type"
    echo "========================================================================"

    # Loop through all variants
    for variant in $VARIANTS; do
        ((CURRENT_EXPERIMENT++))
        echo ""
        echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Running: $model_type + $variant"
        echo "------------------------------------------------------------------------"

        # Run experiment
        python -m debias.debias_variants \
            --train_file "$TRAIN_FILE" \
            --test_file "$TEST_FILE" \
            --variant "$variant" \
            --model_type "$model_type"

        # Check if successful
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $model_type + $variant"
        else
            echo "✗ Failed: $model_type + $variant"
        fi
    done
done

echo ""
echo "========================================================================"
echo "All experiments completed!"
echo "========================================================================"
echo "Results saved to: $RESULT_FILE"

# Show result statistics
if [ -f "$RESULT_FILE" ]; then
    TOTAL_RESULTS=$(tail -n +2 "$RESULT_FILE" | wc -l | xargs)
    echo "Total results: $TOTAL_RESULTS"
fi
echo "========================================================================"

# Show result preview
echo ""
echo "Result preview (last 5 lines):"
echo "------------------------------------------------------------------------"
if [ -f "$RESULT_FILE" ]; then
    tail -6 "$RESULT_FILE"
else
    echo "No results yet"
fi
echo "------------------------------------------------------------------------"

echo ""
echo "Done! Use the following command to view full results:"
echo "  cat $RESULT_FILE"
echo "  column -t -s, $RESULT_FILE  # Formatted display"
echo ""
