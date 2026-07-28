#!/usr/bin/env bash
# Replace the x_one_llm rows in results/individual/individual_<DS>_result.csv with
# a fresh 5-seed x_one_llm sweep whose llm_field = gpt-4o_norm (EEDI/OpQA have
# real gpt-4o; Twin individual's gpt-4o_norm is actually GPT4.1 per
# extract_closed_one.py — documented in the llm_field column of the new rows).
set -xeuo pipefail

PROJ_ROOT=/home/jiangli/debias_response
cd "${PROJ_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

DATASETS=(EEDI OpinionQA Twin-2K-500)
SEEDS=(1 2 3 4 5)
VARIANT=x_one_llm
FIELD=gpt-4o_norm

# Individual MLP profile (matches run_individual.sh).
HIDDEN=6144,3072,1536,768,384
ALPHA=1e-4
LR=0.0002
MAX_ITER=3500
BSZ=512
DROPOUT=0.05

LOG_DIR="${PROJ_ROOT}/logs/qwen_debias"
TMP_DIR="${PROJ_ROOT}/results/individual/_tmp_gpt4o"
mkdir -p "${LOG_DIR}" "${TMP_DIR}"

############ Phase 1: drop x_one_llm rows (backup first) ############
python3 - <<'PY'
import pandas as pd
from pathlib import Path
for ds in ("EEDI", "OpinionQA", "Twin-2K-500"):
    p = Path(f"/home/jiangli/debias_response/results/individual/individual_{ds}_result.csv")
    bak = p.with_suffix(".pre_gpt4o.csv")
    if not bak.exists():
        p.rename(bak)
        pd.read_csv(bak).to_csv(p, index=False)  # restore a working copy
    df = pd.read_csv(p)
    kept = df[df["variant"] != "x_one_llm"].reset_index(drop=True)
    kept.to_csv(p, index=False)
    dropped = len(df) - len(kept)
    print(f"[drop] {ds}: removed {dropped} x_one_llm rows, kept {len(kept)}")
PY

############ Phase 2: run 5 seeds × 3 datasets ############
for DS in "${DATASETS[@]}"; do
  TRAIN="dataset/${DS}/individual/individual_train.parquet"
  TEST="dataset/${DS}/individual/individual_test.parquet"
  for SEED in "${SEEDS[@]}"; do
    REL="individual/_tmp_gpt4o/${DS}_${VARIANT}_gpt4o_seed${SEED}.csv"
    ABS="${PROJ_ROOT}/results/${REL}"
    LOG="${LOG_DIR}/${DS}_individual_gpt4o_seed${SEED}.log"
    if [[ -f "${ABS}" ]]; then
      echo "[skip] ${ABS}"
      continue
    fi
    echo "[debias] ${DS} seed=${SEED} gpt-4o -> ${ABS}"
    python -m debias.debias_variants \
      --train_file "${TRAIN}" --test_file "${TEST}" \
      --variant "${VARIANT}" --model_type mlp \
      --device cuda:0 --random_state "${SEED}" \
      --llm_field "${FIELD}" --llm_vector_transform raw \
      --result_file "${REL}" \
      --hidden_layers "${HIDDEN}" --mlp_alpha "${ALPHA}" \
      --learning_rate_init "${LR}" --max_iter "${MAX_ITER}" \
      --batch_size "${BSZ}" --mlp_dropout "${DROPOUT}" \
      --validation_fraction 0.1 --n_iter_no_change 60 --min_delta 1e-6 \
      > "${LOG}" 2>&1
    if [[ -f "${PROJ_ROOT}/results/results/${REL}" ]]; then
      mv "${PROJ_ROOT}/results/results/${REL}" "${ABS}"
    fi
  done
done
rm -rf "${PROJ_ROOT}/results/results" 2>/dev/null || true

############ Phase 3: append tmp rows (with seed) into the main CSV ############
python3 - <<'PY'
import pandas as pd
from pathlib import Path
TMP = Path("/home/jiangli/debias_response/results/individual/_tmp_gpt4o")
for ds in ("EEDI", "OpinionQA", "Twin-2K-500"):
    main_p = Path(f"/home/jiangli/debias_response/results/individual/individual_{ds}_result.csv")
    main = pd.read_csv(main_p)
    new_rows = []
    for seed in (1, 2, 3, 4, 5):
        tp = TMP / f"{ds}_x_one_llm_gpt4o_seed{seed}.csv"
        if not tp.exists():
            print(f"[warn] missing {tp}")
            continue
        r = pd.read_csv(tp)
        r.insert(0, "seed", seed)
        new_rows.append(r)
    if not new_rows:
        continue
    new = pd.concat(new_rows, ignore_index=True)
    # Align columns with the main CSV.
    for c in main.columns:
        if c not in new.columns:
            new[c] = pd.NA
    new = new[main.columns]
    combined = pd.concat([main, new], ignore_index=True)
    combined.to_csv(main_p, index=False)
    print(f"[append] {ds}: +{len(new)} rows -> {main_p} (total {len(combined)})")
PY

echo "=== update_individual_gpt4o DONE ==="
