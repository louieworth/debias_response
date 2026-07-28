#!/bin/bash
# Run group scale ablations on the filtered-derived aggreated_ablation datasets.
#
# Default workload:
#   3 datasets x 3 variants (one, mean, vector) x 17 scale settings = 153 MLP runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

DATASETS="${DATASETS:-EEDI OpinionQA Twin}"
SCALE_FAMILIES="${SCALE_FAMILIES:-persona llm all}"
VARIANTS="${VARIANTS:-x_one_llm x_avg_llm x_all_llm}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
PY="${PY:-python}"
JOBS="${JOBS:-16}"
DRY_RUN="${DRY_RUN:-n}"
CLEAR_OLD_RESULTS="${CLEAR_OLD_RESULTS:-n}"
CUDA_DEVICE_COUNT="${CUDA_DEVICE_COUNT:-8}"

MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-512,256,128}"
MLP_ALPHA="${MLP_ALPHA:-0.01}"
MLP_LR_INIT="${MLP_LR_INIT:-0.0005}"
MLP_MAX_ITER="${MLP_MAX_ITER:-1500}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-64}"
MLP_DROPOUT="${MLP_DROPOUT:-0.0}"
MLP_VALIDATION_FRACTION="${MLP_VALIDATION_FRACTION:-0.1}"
MLP_N_ITER_NO_CHANGE="${MLP_N_ITER_NO_CHANGE:-20}"
MLP_MIN_DELTA="${MLP_MIN_DELTA:-1e-6}"
MLP_STANDARDIZE="${MLP_STANDARDIZE:-y}"

resolve_device() {
    if [[ "$DEVICE" != "auto" ]]; then
        echo "$DEVICE"
        return
    fi
    "$PY" - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
PY
}

dataset_paths() {
    local dataset="$1"
    case "$dataset" in
        EEDI)
            echo "dataset/EEDI/aggreated_ablation/eedi_train.parquet dataset/EEDI/aggreated_ablation/eedi_test.parquet group/EEDI/scale_ablation/aggreated_eedi_scale_ablation_seed_${SEED}.csv"
            ;;
        OpinionQA)
            echo "dataset/OpinionQA/aggreated_ablation/opinionqa_train.parquet dataset/OpinionQA/aggreated_ablation/opinionqa_test.parquet group/OpinionQA/scale_ablation/aggreated_opinionqa_scale_ablation_seed_${SEED}.csv"
            ;;
        Twin)
            echo "dataset/Twin-2K-500/aggreated_ablation/twin_train.parquet dataset/Twin-2K-500/aggreated_ablation/twin_test.parquet group/twin/scale_ablation/aggreated_twin_scale_ablation_seed_${SEED}.csv"
            ;;
        *)
            echo "[ERROR] Unknown dataset=$dataset" >&2
            return 1
            ;;
    esac
}

if [[ "$CLEAR_OLD_RESULTS" =~ ^[Yy]$ ]]; then
    for dataset in $DATASETS; do
        read -r _ _ out_rel <<< "$(dataset_paths "$dataset")"
        rm -f "results/$out_rel" "results/$out_rel.lock"
    done
fi

RESOLVED_DEVICE="$(resolve_device)"
echo "Device: $RESOLVED_DEVICE"
echo "Datasets: $DATASETS"
echo "Variants: $VARIANTS"
echo "Scale families: $SCALE_FAMILIES"
echo "Seed: $SEED"
echo "Jobs: $JOBS"
if [[ "$RESOLVED_DEVICE" == "cuda" ]]; then
    echo "CUDA device round-robin count: $CUDA_DEVICE_COUNT"
fi

COMMAND_FILE="$(mktemp)"
trap 'rm -f "$COMMAND_FILE"' EXIT

COMMAND_INDEX=0
for dataset in $DATASETS; do
    read -r train_file test_file out_rel <<< "$(dataset_paths "$dataset")"
    if [[ ! -f "$train_file" || ! -f "$test_file" ]]; then
        echo "[ERROR] Missing ablation parquet for $dataset. Run build_group_scale_ablation.py first." >&2
        exit 1
    fi
    mkdir -p "$(dirname "results/$out_rel")"

    for family in $SCALE_FAMILIES; do
        if [[ "$family" == "all" ]]; then
            k_values=(64)
        else
            k_values=(1 2 3 4 5 6 7 8)
        fi

        for k in "${k_values[@]}"; do
            if [[ "$family" == "all" ]]; then
                field="all_64_norm"
                llm_dim="64"
                input_name="all_64"
            else
                field="${family}_${k}_norm"
                llm_dim="$k"
                input_name="${family}_${k}"
            fi

            for variant in $VARIANTS; do
                run_device="$RESOLVED_DEVICE"
                if [[ "$RESOLVED_DEVICE" == "cuda" ]]; then
                    gpu_id="$((COMMAND_INDEX % CUDA_DEVICE_COUNT))"
                    run_device="cuda:${gpu_id}"
                fi
                cmd=(
                    "$PY" -m debias.debias_variants
                    --train_file "$train_file"
                    --test_file "$test_file"
                    --variant "$variant"
                    --model_type mlp
                    --llm_field "$field"
                    --llm_input_mode "scale_${family}"
                    --llm_input_name "$input_name"
                    --random_state "$SEED"
                    --result_file "$out_rel"
                    --no_split_results
                    --device "$run_device"
                    --hidden_layers "$MLP_HIDDEN_LAYERS"
                    --mlp_alpha "$MLP_ALPHA"
                    --learning_rate_init "$MLP_LR_INIT"
                    --max_iter "$MLP_MAX_ITER"
                    --batch_size "$MLP_BATCH_SIZE"
                    --mlp_dropout "$MLP_DROPOUT"
                    --validation_fraction "$MLP_VALIDATION_FRACTION"
                    --n_iter_no_change "$MLP_N_ITER_NO_CHANGE"
                    --min_delta "$MLP_MIN_DELTA"
                )
                if [[ "$variant" == "x_all_llm" || "$variant" == "x_avg_llm" ]]; then
                    cmd+=(--llm_dim "$llm_dim")
                fi
                if [[ ! "$MLP_STANDARDIZE" =~ ^[Yy]$ ]]; then
                    cmd+=(--no_mlp_standardize)
                fi
                printf '%q ' "${cmd[@]}" >> "$COMMAND_FILE"
                printf '\n' >> "$COMMAND_FILE"
                COMMAND_INDEX="$((COMMAND_INDEX + 1))"
            done
        done
    done
done

echo "Planned commands: $(wc -l < "$COMMAND_FILE" | xargs)"

if [[ "$DRY_RUN" =~ ^[Yy]$ ]]; then
    sed -n '1,20p' "$COMMAND_FILE"
    exit 0
fi

xargs -P "$JOBS" -I {} bash -lc '{}' < "$COMMAND_FILE"

echo "Completed scale ablation runs."
for dataset in $DATASETS; do
    read -r _ _ out_rel <<< "$(dataset_paths "$dataset")"
    out_file="results/$out_rel"
    if [[ -f "$out_file" ]]; then
        rows="$(tail -n +2 "$out_file" | wc -l | xargs)"
        echo "$out_file rows=$rows"
    else
        echo "$out_file missing"
    fi
done
