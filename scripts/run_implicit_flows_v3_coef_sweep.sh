#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-implicit_flows_v3_coef_sweep_result}"
SEEDS_STR="${SEEDS:-0}"

# Target task (override when needed).
CUBE_TASK_ID="${CUBE_TASK_ID:-task1}"

# Sweep values (can be overridden by env var).
ALPHAS_STR="${ALPHAS:-0.1 0.5 1 3}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a ALPHAS <<< "$ALPHAS_STR"

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
  for alpha in "${ALPHAS[@]}"; do
    run_exp "implicit_flows_v3-cube-triple-${CUBE_TASK_ID}-seed${seed}-alpha${alpha}" \
      --seed="${seed}" \
      --env_name="cube-triple-play-singletask-${CUBE_TASK_ID}-v0" \
      --agent=agents/implicit_flows_v3.py \
      --agent.discount=0.995 \
      --agent.alpha="${alpha}" || true
  done
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  exit 1
fi

echo "All implicit_flows_v3 coefficient sweep experiments finished."
