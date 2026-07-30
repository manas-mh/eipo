#!/bin/bash
# Tanuh AI cluster: job-array variant -- one (env, seed) pair per job, one GPU
# each. With 4 free H200s Slurm will run 4 at a time and queue the rest.
# Usually eipo_pack.sbatch is the better fit (5 seeds share one GPU); use this
# when you want each run on its own card.
#
# Usage (from the repo root on the master node):
#   sbatch continuous/slurm/eipo_array.sbatch

#SBATCH --job-name=eipo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --nodelist=n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%4
#SBATCH --output=slurm_outputs/%x_%A_%a.out
#SBATCH --error=slurm_outputs/%x_%A_%a.err

module purge
# uv-managed venv, created on the master node with `uv sync` (lives on NFS,
# so it is visible on n2). Jobs must be submitted from the repo root.
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

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
