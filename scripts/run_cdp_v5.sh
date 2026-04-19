#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-cdpv5_results}"
SEEDS_STR="${SEEDS:-0}"
TASK_IDS_STR="${TASK_IDS:-task1 task2 task3 task4 task5}"
RUN_OGBENCH="${RUN_OGBENCH:-1}"
RUN_D4RL="${RUN_D4RL:-1}"
RUN_ONLINE="${RUN_ONLINE:-1}"

# Fixed online tasks used by the baseline script.
ONLINE_ANTMAZE_TASK_ID="${ONLINE_ANTMAZE_TASK_ID:-task1}"
ONLINE_HUMANOIDMAZE_TASK_ID="${ONLINE_HUMANOIDMAZE_TASK_ID:-task1}"
ONLINE_CUBE_DOUBLE_TASK_ID="${ONLINE_CUBE_DOUBLE_TASK_ID:-task2}"
ONLINE_CUBE_TRIPLE_TASK_ID="${ONLINE_CUBE_TRIPLE_TASK_ID:-task1}"
ONLINE_PUZZLE4_TASK_ID="${ONLINE_PUZZLE4_TASK_ID:-task4}"
ONLINE_SCENE_TASK_ID="${ONLINE_SCENE_TASK_ID:-task2}"

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
      run_exp "cdp_v5-cube-double-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-double-play-singletask-${task_id}-v0" \
        --agent=agents/cdp_v5.py \
        --agent.discount=0.995 || true

      run_exp "cdp_v5-cube-triple-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/cdp_v5.py \
        --agent.discount=0.995 || true

      run_exp "cdp_v5-puzzle3-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-3x3-play-singletask-${task_id}-v0" \
        --agent=agents/cdp_v5.py \
        --agent.discount=0.995 || true

      run_exp "cdp_v5-puzzle4-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-4x4-play-singletask-${task_id}-v0" \
        --agent=agents/cdp_v5.py \
        --agent.discount=0.995 || true

      run_exp "cdp_v5-scene-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="scene-play-singletask-${task_id}-v0" \
        --agent=agents/cdp_v5.py \
        --agent.discount=0.995 || true
    done
  fi

  if [[ "$RUN_D4RL" == "1" ]]; then
    run_exp "cdp_v5-pen-human-seed${seed}" \
      --seed="${seed}" \
      --env_name=pen-human-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-pen-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=pen-cloned-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-pen-expert-seed${seed}" \
      --seed="${seed}" \
      --env_name=pen-expert-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-door-human-seed${seed}" \
      --seed="${seed}" \
      --env_name=door-human-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-door-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=door-cloned-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-door-expert-seed${seed}" \
      --seed="${seed}" \
      --env_name=door-expert-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-hammer-human-seed${seed}" \
      --seed="${seed}" \
      --env_name=hammer-human-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-hammer-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=hammer-cloned-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-hammer-expert-seed${seed}" \
      --seed="${seed}" \
      --env_name=hammer-expert-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-relocate-human-seed${seed}" \
      --seed="${seed}" \
      --env_name=relocate-human-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-relocate-cloned-seed${seed}" \
      --seed="${seed}" \
      --env_name=relocate-cloned-v1 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-relocate-expert-seed${seed}" \
      --seed="${seed}" \
      --env_name=relocate-expert-v1 \
      --agent=agents/cdp_v5.py || true
  fi

  if [[ "$RUN_ONLINE" == "1" ]]; then
    run_exp "cdp_v5-antmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name="antmaze-large-navigate-singletask-${ONLINE_ANTMAZE_TASK_ID}-v0" \
      --online_steps=1000000 \
      --agent=agents/cdp_v5.py || true

    run_exp "cdp_v5-humanoidmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name="humanoidmaze-medium-navigate-singletask-${ONLINE_HUMANOIDMAZE_TASK_ID}-v0" \
      --online_steps=1000000 \
      --agent=agents/cdp_v5.py \
      --agent.discount=0.995 || true

    run_exp "cdp_v5-cube-double-online-seed${seed}" \
      --seed="${seed}" \
      --env_name="cube-double-play-singletask-${ONLINE_CUBE_DOUBLE_TASK_ID}-v0" \
      --online_steps=1000000 \
      --agent=agents/cdp_v5.py \
      --agent.discount=0.995 || true

    run_exp "cdp_v5-cube-triple-online-seed${seed}" \
      --seed="${seed}" \
      --env_name="cube-triple-play-singletask-${ONLINE_CUBE_TRIPLE_TASK_ID}-v0" \
      --online_steps=1000000 \
      --agent=agents/cdp_v5.py \
      --agent.discount=0.995 || true

    run_exp "cdp_v5-puzzle4-online-seed${seed}" \
      --seed="${seed}" \
      --env_name="puzzle-4x4-play-singletask-${ONLINE_PUZZLE4_TASK_ID}-v0" \
      --online_steps=1000000 \
      --agent=agents/cdp_v5.py \
      --agent.discount=0.995 || true

    run_exp "cdp_v5-scene-online-seed${seed}" \
      --seed="${seed}" \
      --env_name="scene-play-singletask-${ONLINE_SCENE_TASK_ID}-v0" \
      --online_steps=1000000 \
      --agent=agents/cdp_v5.py \
      --agent.discount=0.995 || true
  fi
done

if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
  echo "================ FAILED JOBS ================"
  for job in "${FAILED_JOBS[@]}"; do
    echo "$job"
  done
  exit 1
fi

echo "All cdp_v5 baseline-environment experiments finished."
