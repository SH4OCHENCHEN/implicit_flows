#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-tdflow_result}"
SEEDS_STR="${SEEDS:-0}"

# Test targets (override when needed).
CUBE_TASK_ID="${CUBE_TASK_ID:-task1}"
PUZZLE_TASK_ID="${PUZZLE_TASK_ID:-task1}"

# Lightweight defaults for quick verification.
OFFLINE_STEPS="${OFFLINE_STEPS:-20000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"

mkdir -p "$SAVE_ROOT"

COMMON_FLAGS=(
  "--save_dir=${SAVE_ROOT}"
  "--enable_wandb=0"
  "--wandb_mode=disabled"
  "--wandb_no_local_files=1"
  "--video_episodes=0"
  "--offline_steps=${OFFLINE_STEPS}"
  "--online_steps=0"
  "--log_interval=${LOG_INTERVAL}"
  "--eval_interval=${EVAL_INTERVAL}"
  "--save_interval=${SAVE_INTERVAL}"
  "--eval_episodes=${EVAL_EPISODES}"
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
  run_exp "tdflow-cube-triple-${CUBE_TASK_ID}-seed${seed}" \
    --seed="${seed}" \
    --env_name="cube-triple-play-singletask-${CUBE_TASK_ID}-v0" \
    --agent=agents/tdflow.py \
    --agent.discount=0.995 || true

  run_exp "tdflow-puzzle4-${PUZZLE_TASK_ID}-seed${seed}" \
    --seed="${seed}" \
    --env_name="puzzle-4x4-play-singletask-${PUZZLE_TASK_ID}-v0" \
    --agent=agents/tdflow.py \
    --agent.q_agg=min || true
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  exit 1
fi

echo "All TDFlow test experiments finished."
