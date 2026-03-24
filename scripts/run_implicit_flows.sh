#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-implicit_flows_result}"
SEEDS_STR="${SEEDS:-0}"
TASK_IDS_STR="${TASK_IDS:-task1}"
RUN_OGBENCH="${RUN_OGBENCH:-1}"
RUN_D4RL="${RUN_D4RL:-0}"
RUN_ONLINE="${RUN_ONLINE:-0}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a TASK_IDS <<< "$TASK_IDS_STR"

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
  if [[ "$RUN_OGBENCH" == "1" ]]; then
    for task_id in "${TASK_IDS[@]}"; do
      run_exp "implicit_flows-cube-triple-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows.py \
        --agent.discount=0.995 || true

      run_exp "implicit_flows-cube-double-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-double-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows.py \
        --agent.discount=0.995 || true

      run_exp "implicit_flows-scene-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="scene-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows.py \
        --agent.ret_agg=min || true
    done
  fi

  if [[ "$RUN_D4RL" == "1" ]]; then
    run_exp "implicit_flows-pen-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=pen-cloned-v1 \
      --agent=agents/implicit_flows.py \
      --agent.q_agg=min || true

    run_exp "implicit_flows-door-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=door-cloned-v1 \
      --agent=agents/implicit_flows.py \
      --agent.ret_agg=min \
      --agent.q_agg=min || true

    run_exp "implicit_flows-hammer-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=hammer-cloned-v1 \
      --agent=agents/implicit_flows.py \
      --agent.q_agg=min || true

    run_exp "implicit_flows-relocate-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=relocate-cloned-v1 \
      --agent=agents/implicit_flows.py \
      --agent.q_agg=min || true
  fi

  if [[ "$RUN_ONLINE" == "1" ]]; then
    run_exp "implicit_flows-antmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=antmaze-large-navigate-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/implicit_flows.py \
      --agent.q_agg=min || true
  fi
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  exit 1
fi

echo "All implicit_flows test experiments finished."
