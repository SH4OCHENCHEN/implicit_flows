#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-cdpv3_coef_sweep_results}"
SEEDS_STR="${SEEDS:-0}"
TASK_IDS_STR="${TASK_IDS:-task1}"

# Grid search ranges (can be overridden by env vars).
DRIFT_TEMPS_STR="${DRIFT_TEMPS:-0.1 10 50}"
POS_PROB_TEMPS_STR="${POS_PROB_TEMPS:-0.0001 0.01 1}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a TASK_IDS <<< "$TASK_IDS_STR"
IFS=' ' read -r -a DRIFT_TEMPS <<< "$DRIFT_TEMPS_STR"
IFS=' ' read -r -a POS_PROB_TEMPS <<< "$POS_PROB_TEMPS_STR"

mkdir -p "$SAVE_ROOT"

COMMON_FLAGS=(
  "--save_dir=${SAVE_ROOT}"
  "--enable_wandb=0"
  "--wandb_mode=disabled"
  "--wandb_no_local_files=1"
  "--video_episodes=0"
)

FAILED_JOBS=()

run_exp() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  if ! "$PYTHON_BIN" main.py "$@" "${COMMON_FLAGS[@]}"; then
    echo "[$(date '+%F %T')] FAIL  ${name}"
    FAILED_JOBS+=("$name")
    return 1
  fi
  echo "[$(date '+%F %T')] DONE  ${name}"
  return 0
}

for seed in "${SEED_LIST[@]}"; do
  for task_id in "${TASK_IDS[@]}"; do
    for drift_temp in "${DRIFT_TEMPS[@]}"; do
      for pos_prob_temp in "${POS_PROB_TEMPS[@]}"; do
        run_exp "cdp_v3-cube-double-${task_id}-seed${seed}-dt${drift_temp}-ppt${pos_prob_temp}" \
          --seed="${seed}" \
          --env_name="cube-double-play-singletask-${task_id}-v0" \
          --agent=agents/cdp_v3.py \
          --agent.discount=0.995 \
          --agent.drift_temp="${drift_temp}" \
          --agent.pos_prob_temp="${pos_prob_temp}" || true
      done
    done
  done
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  exit 1
fi

echo "All cdp_v3 coefficient sweep experiments finished."
