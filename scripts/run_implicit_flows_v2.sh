#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-implicit_flows_v2_result}"
SEEDS_STR="${SEEDS:-1}"
TASK_IDS_STR="${TASK_IDS:-task1 task2 task3 task4 task5}"
RUN_OGBENCH="${RUN_OGBENCH:-1}"
RUN_D4RL="${RUN_D4RL:-1}"
RUN_ONLINE="${RUN_ONLINE:-1}"

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
      run_exp "implicit_flows_v2-cube-double-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-double-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows_v2.py \
        --agent.discount=0.995 || true

      run_exp "implicit_flows_v2-cube-triple-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows_v2.py \
        --agent.discount=0.995 || true

      run_exp "implicit_flows_v2-puzzle3-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-3x3-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows_v2.py \
        --agent.ret_agg=min || true

      run_exp "implicit_flows_v2-puzzle4-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-4x4-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows_v2.py \
        --agent.confidence_weight_temp=100 \
        --agent.q_agg=min || true

      run_exp "implicit_flows_v2-scene-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="scene-play-singletask-${task_id}-v0" \
        --agent=agents/implicit_flows_v2.py \
        --agent.ret_agg=min || true
    done
  fi

  if [[ "$RUN_D4RL" == "1" ]]; then
    run_exp "implicit_flows_v2-pen-human-seed${seed}" \
      --seed="${seed}" \
      --env_name=pen-human-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.ret_agg=min \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-pen-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=pen-cloned-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-pen-expert-seed${seed}" \
      --seed="${seed}" \
      --env_name=pen-expert-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-door-human-seed${seed}" \
      --seed="${seed}" \
      --env_name=door-human-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-door-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=door-cloned-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.ret_agg=min \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-door-expert-seed${seed}" \
      --seed="${seed}" \
      --env_name=door-expert-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.ret_agg=min \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-hammer-human-seed${seed}" \
      --seed="${seed}" \
      --env_name=hammer-human-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-hammer-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=hammer-cloned-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-hammer-expert-seed${seed}" \
      --seed="${seed}" \
      --env_name=hammer-expert-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.ret_agg=min \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-relocate-human-seed${seed}" \
      --seed="${seed}" \
      --env_name=relocate-human-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.ret_agg=min \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-relocate-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=relocate-cloned-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-relocate-expert-seed${seed}" \
      --seed="${seed}" \
      --env_name=relocate-expert-v1 \
      --agent=agents/implicit_flows_v2.py \
      --agent.ret_agg=min \
      --agent.q_agg=min || true
  fi

  if [[ "$RUN_ONLINE" == "1" ]]; then
    run_exp "implicit_flows_v2-antmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=antmaze-large-navigate-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/implicit_flows_v2.py \
      --agent.alpha=30 \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-humanoidmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/implicit_flows_v2.py \
      --agent.discount=0.995 \
      --agent.alpha=100 \
      --agent.q_agg=min \
      --agent.ret_agg=min || true

    run_exp "implicit_flows_v2-cube-double-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=cube-double-play-singletask-task2-v0 \
      --online_steps=1000000 \
      --agent=agents/implicit_flows_v2.py \
      --agent.discount=0.995 \
      --agent.alpha=300 || true

    run_exp "implicit_flows_v2-cube-triple-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=cube-triple-play-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/implicit_flows_v2.py \
      --agent.discount=0.995 \
      --agent.alpha=300 || true

    run_exp "implicit_flows_v2-puzzle4-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=puzzle-4x4-play-singletask-task4-v0 \
      --online_steps=1000000 \
      --agent=agents/implicit_flows_v2.py \
      --agent.alpha=300 \
      --agent.q_agg=min || true

    run_exp "implicit_flows_v2-scene-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=scene-play-singletask-task2-v0 \
      --online_steps=1000000 \
      --agent=agents/implicit_flows_v2.py \
      --agent.alpha=300 \
      --agent.ret_agg=min || true
  fi
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  exit 1
fi

echo "All implicit_flows_v2 experiments finished."

