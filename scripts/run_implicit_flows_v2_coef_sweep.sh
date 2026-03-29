#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-implicit_flows_v2_coef_sweep_result}"
SEEDS_STR="${SEEDS:-0}"
TASK_IDS_STR="${TASK_IDS:-task1}"

# Sweep values (can be overridden by env var).
CONF_WEIGHT_TEMPS_STR="${CONF_WEIGHT_TEMPS:-0.1 1 5 10 20 50}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a TASK_IDS <<< "$TASK_IDS_STR"
IFS=' ' read -r -a CONF_WEIGHT_TEMPS <<< "$CONF_WEIGHT_TEMPS_STR"

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
    for cwt in "${CONF_WEIGHT_TEMPS[@]}"; do
      run_exp "implicit_flows_v2-cube-double-${task_id}-seed${seed}-cwt${cwt}" \
        --seed="${seed}" \
        --env_name="cube-double-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows_v2.py \
        --agent.discount=0.995 \
        --agent.confidence_weight_temp="${cwt}" || true
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

echo "All implicit_flows_v2 coefficient sweep experiments finished."
