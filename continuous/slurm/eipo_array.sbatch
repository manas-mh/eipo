#!/bin/bash
# Job-array variant: one SLURM task per (env, seed) pair. Use this instead of
# eipo_pack.sbatch if your cluster allows GPU sharing across jobs (MIG/MPS) or
# has several GPUs. Submit from the repo root:
#
#   sbatch continuous/slurm/eipo_array.sbatch

#SBATCH --job-name=eipo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm_outputs/%x_%A_%a.out
#SBATCH --error=slurm_outputs/%x_%A_%a.err

# --- site-specific setup: EDIT ME ---------------------------------------- #
# module load cuda/12.8
source "$HOME/venvs/eipo/bin/activate"   # or: conda activate eipo
export MUJOCO_GL=egl
# -------------------------------------------------------------------------- #

ENVS=(HalfCheetah-v4 AntMaze_UMaze-v5)
STEPS=(3000000 10000000)
EXTRA=("" "--no-normalize-ext-reward")

env_idx=$(( SLURM_ARRAY_TASK_ID / 5 ))
seed=$(( SLURM_ARRAY_TASK_ID % 5 + 1 ))

python continuous/eipo_ppo_continuous.py \
  --env-id "${ENVS[$env_idx]}" \
  --total-timesteps "${STEPS[$env_idx]}" \
  --seed "$seed" \
  ${EXTRA[$env_idx]}
