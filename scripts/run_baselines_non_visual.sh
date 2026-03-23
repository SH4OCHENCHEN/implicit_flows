#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-baseline_result}"
RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_ONLINE="${RUN_ONLINE:-1}"
RUN_D4RL="${RUN_D4RL:-1}"
SEEDS_STR="${SEEDS:-0}"
TASK_IDS_STR="${TASK_IDS:-task1 task2 task3 task4 task5}"

IFS=' ' read -r -a SEED_LIST <<< "$SEEDS_STR"
IFS=' ' read -r -a TASK_IDS <<< "$TASK_IDS_STR"

mkdir -p "$SAVE_ROOT"
mkdir -p "$SAVE_ROOT/logs"

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
  if [[ "$RUN_OFFLINE" == "1" ]]; then
    for task_id in "${TASK_IDS[@]}"; do
      run_exp "IQL-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/iql.py \
        --agent.discount=0.995 \
        --agent.alpha=10 || true

      run_exp "ReBRAC-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/rebrac.py \
        --agent.discount=0.995 \
        --agent.alpha_actor=0.03 \
        --agent.alpha_critic=0.0 || true

      run_exp "FBRAC-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/fbrac.py \
        --agent.discount=0.995 \
        --agent.alpha=100 || true

      run_exp "IFQL-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/ifql.py \
        --agent.discount=0.995 || true

      run_exp "FQL-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/fql.py \
        --agent.discount=0.995 \
        --agent.alpha=300 || true

      run_exp "C51-cube-double-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-double-play-singletask-${task_id}-v0" \
        --agent=agents/c51.py \
        --agent.discount=0.995 \
        --agent.num_atoms=101 || true

      run_exp "C51-cube-triple-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/c51.py \
        --agent.discount=0.995 || true

      run_exp "C51-puzzle3-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-3x3-play-singletask-${task_id}-v0" \
        --agent=agents/c51.py \
        --agent.discount=0.995 \
        --agent.num_atoms=101 || true

      run_exp "C51-puzzle4-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-4x4-play-singletask-${task_id}-v0" \
        --agent=agents/c51.py \
        --agent.num_atoms=101 \
        --agent.q_agg=min || true

      run_exp "C51-scene-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="scene-play-singletask-${task_id}-v0" \
        --agent=agents/c51.py || true

      run_exp "IQN-cube-double-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-double-play-singletask-${task_id}-v0" \
        --agent=agents/iqn.py \
        --agent.discount=0.995 || true

      run_exp "IQN-cube-triple-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/iqn.py \
        --agent.discount=0.995 \
        --agent.kappa=0.8 || true

      run_exp "IQN-puzzle3-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-3x3-play-singletask-${task_id}-v0" \
        --agent=agents/iqn.py \
        --agent.discount=0.995 \
        --agent.kappa=0.8 || true

      run_exp "IQN-puzzle4-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-4x4-play-singletask-${task_id}-v0" \
        --agent=agents/iqn.py \
        --agent.kappa=0.95 || true

      run_exp "IQN-scene-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="scene-play-singletask-${task_id}-v0" \
        --agent=agents/iqn.py \
        --agent.kappa=0.95 || true

      run_exp "CODAC-cube-double-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-double-play-singletask-${task_id}-v0" \
        --agent=agents/codac.py \
        --agent.discount=0.995 \
        --agent.kappa=0.95 \
        --agent.alpha=300 || true

      run_exp "CODAC-puzzle3-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-3x3-play-singletask-${task_id}-v0" \
        --agent=agents/codac.py \
        --agent.discount=0.995 \
        --agent.kappa=0.95 \
        --agent.alpha=1000 || true

      run_exp "CODAC-scene-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="scene-play-singletask-${task_id}-v0" \
        --agent=agents/codac.py \
        --agent.kappa=0.95 \
        --agent.alpha=100 || true

      run_exp "CODAC-puzzle4-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="puzzle-4x4-play-singletask-${task_id}-v0" \
        --agent=agents/codac.py \
        --agent.kappa=0.95 \
        --agent.alpha=1000 || true

      run_exp "CODAC-cube-triple-${task_id}-seed${seed}" \
        --seed="${seed}" \
        --env_name="cube-triple-play-singletask-${task_id}-v0" \
        --agent=agents/codac.py \
        --agent.discount=0.995 \
        --agent.kappa=0.95 \
        --agent.alpha=100 || true
    done
  fi

  if [[ "$RUN_D4RL" == "1" ]]; then
    run_exp "C51-pen-human-seed${seed}" --seed="${seed}" --env_name=pen-human-v1 --agent=agents/c51.py --agent.q_agg=min || true
    run_exp "C51-pen-cloned-seed${seed}" --seed="${seed}" --env_name=pen-cloned-v1 --agent=agents/c51.py || true
    run_exp "C51-pen-expert-seed${seed}" --seed="${seed}" --env_name=pen-expert-v1 --agent=agents/c51.py --agent.q_agg=min || true
    run_exp "C51-door-human-seed${seed}" --seed="${seed}" --env_name=door-human-v1 --agent=agents/c51.py --agent.num_atoms=101 || true
    run_exp "C51-door-cloned-seed${seed}" --seed="${seed}" --env_name=door-cloned-v1 --agent=agents/c51.py --agent.q_agg=min || true
    run_exp "C51-door-expert-seed${seed}" --seed="${seed}" --env_name=door-expert-v1 --agent=agents/c51.py || true
    run_exp "C51-hammer-human-seed${seed}" --seed="${seed}" --env_name=hammer-human-v1 --agent=agents/c51.py || true
    run_exp "C51-hammer-cloned-seed${seed}" --seed="${seed}" --env_name=hammer-cloned-v1 --agent=agents/c51.py || true
    run_exp "C51-hammer-expert-seed${seed}" --seed="${seed}" --env_name=hammer-expert-v1 --agent=agents/c51.py --agent.q_agg=min || true
    run_exp "C51-relocate-human-seed${seed}" --seed="${seed}" --env_name=relocate-human-v1 --agent=agents/c51.py --agent.num_atoms=101 --agent.q_agg=min || true
    run_exp "C51-relocate-cloned-seed${seed}" --seed="${seed}" --env_name=relocate-cloned-v1 --agent=agents/c51.py --agent.q_agg=min || true
    run_exp "C51-relocate-expert-seed${seed}" --seed="${seed}" --env_name=relocate-expert-v1 --agent=agents/c51.py --agent.num_atoms=101 --agent.q_agg=min || true

    run_exp "IQN-pen-human-seed${seed}" --seed="${seed}" --env_name=pen-human-v1 --agent=agents/iqn.py --agent.kappa=0.8 --agent.quantile_agg=min || true
    run_exp "IQN-pen-cloned-seed${seed}" --seed="${seed}" --env_name=pen-cloned-v1 --agent=agents/iqn.py --agent.kappa=0.8 --agent.quantile_agg=min || true
    run_exp "IQN-pen-expert-seed${seed}" --seed="${seed}" --env_name=pen-expert-v1 --agent=agents/iqn.py --agent.kappa=0.8 --agent.quantile_agg=min || true
    run_exp "IQN-door-human-seed${seed}" --seed="${seed}" --env_name=door-human-v1 --agent=agents/iqn.py --agent.quantile_agg=min || true
    run_exp "IQN-door-cloned-seed${seed}" --seed="${seed}" --env_name=door-cloned-v1 --agent=agents/iqn.py --agent.quantile_agg=min || true
    run_exp "IQN-door-expert-seed${seed}" --seed="${seed}" --env_name=door-expert-v1 --agent=agents/iqn.py || true
    run_exp "IQN-hammer-human-seed${seed}" --seed="${seed}" --env_name=hammer-human-v1 --agent=agents/iqn.py --agent.kappa=0.7 || true
    run_exp "IQN-hammer-cloned-seed${seed}" --seed="${seed}" --env_name=hammer-cloned-v1 --agent=agents/iqn.py --agent.kappa=0.7 --agent.quantile_agg=min || true
    run_exp "IQN-hammer-expert-seed${seed}" --seed="${seed}" --env_name=hammer-expert-v1 --agent=agents/iqn.py --agent.kappa=0.7 || true
    run_exp "IQN-relocate-human-seed${seed}" --seed="${seed}" --env_name=relocate-human-v1 --agent=agents/iqn.py || true
    run_exp "IQN-relocate-cloned-seed${seed}" --seed="${seed}" --env_name=relocate-cloned-v1 --agent=agents/iqn.py || true
    run_exp "IQN-relocate-expert-seed${seed}" --seed="${seed}" --env_name=relocate-expert-v1 --agent=agents/iqn.py --agent.quantile_agg=min || true

    run_exp "CODAC-pen-human-seed${seed}" --seed="${seed}" --env_name=pen-human-v1 --agent=agents/codac.py --agent.kappa=0.8 --agent.alpha=10000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-pen-cloned-seed${seed}" --seed="${seed}" --env_name=pen-cloned-v1 --agent=agents/codac.py --agent.kappa=0.8 --agent.alpha=10000 || true
    run_exp "CODAC-pen-expert-seed${seed}" --seed="${seed}" --env_name=pen-expert-v1 --agent=agents/codac.py --agent.kappa=0.8 --agent.alpha=10000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-door-human-seed${seed}" --seed="${seed}" --env_name=door-human-v1 --agent=agents/codac.py --agent.alpha=10000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-door-cloned-seed${seed}" --seed="${seed}" --env_name=door-cloned-v1 --agent=agents/codac.py --agent.alpha=30000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-door-expert-seed${seed}" --seed="${seed}" --env_name=door-expert-v1 --agent=agents/codac.py --agent.alpha=10000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-hammer-human-seed${seed}" --seed="${seed}" --env_name=hammer-human-v1 --agent=agents/codac.py --agent.kappa=0.8 --agent.alpha=30000 || true
    run_exp "CODAC-hammer-cloned-seed${seed}" --seed="${seed}" --env_name=hammer-cloned-v1 --agent=agents/codac.py --agent.kappa=0.8 --agent.alpha=10000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-hammer-expert-seed${seed}" --seed="${seed}" --env_name=hammer-expert-v1 --agent=agents/codac.py --agent.kappa=0.8 --agent.alpha=10000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-relocate-human-seed${seed}" --seed="${seed}" --env_name=relocate-human-v1 --agent=agents/codac.py --agent.alpha=30000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-relocate-cloned-seed${seed}" --seed="${seed}" --env_name=relocate-cloned-v1 --agent=agents/codac.py --agent.alpha=30000 --agent.alpha_penalty=0.01 || true
    run_exp "CODAC-relocate-expert-seed${seed}" --seed="${seed}" --env_name=relocate-expert-v1 --agent=agents/codac.py --agent.alpha=10000 || true
  fi

  if [[ "$RUN_ONLINE" == "1" ]]; then
    run_exp "IQL-antmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=antmaze-large-navigate-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/iql.py \
      --agent.alpha=10 || true

    run_exp "IQL-cube-triple-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=cube-triple-play-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/iql.py \
      --agent.discount=0.995 \
      --agent.alpha=10 || true

    run_exp "IFQL-antmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=antmaze-large-navigate-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/ifql.py || true

    run_exp "IFQL-cube-triple-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=cube-triple-play-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/ifql.py \
      --agent.discount=0.995 || true

    run_exp "FQL-antmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=antmaze-large-navigate-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/fql.py \
      --agent.alpha=10 || true

    run_exp "FQL-cube-triple-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=cube-triple-play-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/fql.py \
      --agent.discount=0.995 \
      --agent.alpha=300 || true

    run_exp "IQN-antmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=antmaze-large-navigate-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/iqn.py \
      --agent.kappa=0.7 || true

    run_exp "IQN-humanoidmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/iqn.py \
      --agent.discount=0.995 \
      --agent.kappa=0.7 || true

    run_exp "IQN-cube-double-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=cube-double-play-singletask-task2-v0 \
      --online_steps=1000000 \
      --agent=agents/iqn.py \
      --agent.discount=0.995 || true

    run_exp "IQN-cube-triple-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=cube-triple-play-singletask-task1-v0 \
      --online_steps=1000000 \
      --agent=agents/iqn.py \
      --agent.discount=0.995 \
      --agent.kappa=0.8 || true

    run_exp "IQN-puzzle4-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=puzzle-4x4-play-singletask-task4-v0 \
      --online_steps=1000000 \
      --agent=agents/iqn.py \
      --agent.kappa=0.8 || true

    run_exp "IQN-scene-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=scene-play-singletask-task2-v0 \
      --online_steps=1000000 \
      --agent=agents/iqn.py \
      --agent.kappa=0.95 || true

    run_exp "RLPD-antmaze-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=antmaze-large-navigate-singletask-task1-v0 \
      --offline_steps=0 \
      --online_steps=1000000 \
      --balanced_sampling=1 \
      --agent=agents/sac.py || true

    run_exp "RLPD-cube-triple-online-seed${seed}" \
      --seed="${seed}" \
      --env_name=cube-triple-play-singletask-task1-v0 \
      --offline_steps=0 \
      --online_steps=1000000 \
      --balanced_sampling=1 \
      --agent=agents/sac.py \
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

echo "All requested baseline experiments finished."
