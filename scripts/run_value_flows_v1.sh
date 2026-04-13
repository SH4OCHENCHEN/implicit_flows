#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-value_flows_v1_result}"
SEEDS_STR="${SEEDS:-0}"

# Target tasks (override when needed).
CUBE_TASK_ID="${CUBE_TASK_ID:-task1}"
PUZZLE_TASK_ID="${PUZZLE_TASK_ID:-task1}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"

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
  run_exp "value_flows_v1-cube-triple-${CUBE_TASK_ID}-seed${seed}" \
    --seed="${seed}" \
    --env_name="cube-triple-play-singletask-${CUBE_TASK_ID}-v0" \
    --agent=agents/value_flowsv1.py \
    --agent.discount=0.995 || true

  run_exp "value_flows_v1-puzzle4-${PUZZLE_TASK_ID}-seed${seed}" \
    --seed="${seed}" \
    --env_name="puzzle-4x4-play-singletask-${PUZZLE_TASK_ID}-v0" \
    --agent=agents/value_flowsv1.py \
    --agent.q_agg=min || true
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  exit 1
fi

echo "All value_flows_v1 test experiments finished."