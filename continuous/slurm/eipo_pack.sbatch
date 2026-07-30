#!/bin/bash
# Runs all 5 seeds of one environment concurrently on a single GPU.
# The workload is CPU-bound (MuJoCo stepping); each run uses <1 GB VRAM,
# so packing 5 runs on one H200 is fine. Budget ~8 CPUs per run.
#
# Usage:
#   sbatch continuous/slurm/eipo_pack.sbatch HalfCheetah-v4 3000000
#   sbatch continuous/slurm/eipo_pack.sbatch AntMaze_UMaze-v5 10000000 --no-normalize-ext-reward
#
# Submit from the repo root. Adjust the #SBATCH lines and the environment
# activation below to your cluster.

#SBATCH --job-name=eipo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# --- site-specific setup: EDIT ME ---------------------------------------- #
# module load cuda/12.8
source "$HOME/venvs/eipo/bin/activate"   # or: conda activate eipo
export MUJOCO_GL=egl                      # headless rendering backend
# -------------------------------------------------------------------------- #

ENV_ID=${1:?usage: sbatch eipo_pack.sbatch <env_id> <total_timesteps> [extra args...]}
STEPS=${2:?missing total_timesteps}
shift 2

for seed in 1 2 3 4 5; do
  python continuous/eipo_ppo_continuous.py \
    --env-id "$ENV_ID" \
    --total-timesteps "$STEPS" \
    --seed "$seed" \
    "$@" &
done
wait
