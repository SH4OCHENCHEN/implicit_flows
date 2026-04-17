#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-cdpv5_mppi_sweep_results}"
SEEDS_STR="${SEEDS:-0}"
TASK_IDS_STR="${TASK_IDS:-task1}"
MPPI_ITERS_STR="${MPPI_ITERS:-2 3 4}"
MPPI_NUM_POS_STR="${MPPI_NUM_POS:-8 16}"
MPPI_TEMP_STR="${MPPI_TEMP:-0.1 1 10}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a TASK_IDS <<< "$TASK_IDS_STR"
IFS=' ' read -r -a MPPI_ITERS_LIST <<< "$MPPI_ITERS_STR"
IFS=' ' read -r -a MPPI_NUM_POS_LIST <<< "$MPPI_NUM_POS_STR"
IFS=' ' read -r -a MPPI_TEMP_LIST <<< "$MPPI_TEMP_STR"

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
    for mppi_iters in "${MPPI_ITERS_LIST[@]}"; do
      for mppi_num_pos in "${MPPI_NUM_POS_LIST[@]}"; do
        for mppi_temp in "${MPPI_TEMP_LIST[@]}"; do
          run_exp "cdp_v5-cube-double-${task_id}-seed${seed}-iters${mppi_iters}-npos${mppi_num_pos}-temp${mppi_temp}" \
            --seed="${seed}" \
            --env_name="cube-double-play-singletask-${task_id}-v0" \
            --agent=agents/cdp_v5.py \
            --agent.discount=0.995 \
            --agent.mppi_iters="${mppi_iters}" \
            --agent.mppi_num_pos="${mppi_num_pos}" \
            --agent.mppi_temp="${mppi_temp}" || true
        done
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

echo "All cdp_v5 cube-double MPPI sweep experiments finished."
