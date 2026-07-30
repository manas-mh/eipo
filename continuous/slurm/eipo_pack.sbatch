#!/bin/bash
# Tanuh AI cluster: runs all 5 seeds of one environment concurrently on a
# single H200 on n2. The workload is CPU-bound (MuJoCo stepping); each run
# uses <1 GB VRAM, so packing 5 runs on one GPU is fine.
#
# Usage (from the repo root on the master node):
#   sbatch continuous/slurm/eipo_pack.sbatch HalfCheetah-v4 3000000
#   sbatch continuous/slurm/eipo_pack.sbatch AntMaze_UMaze-v5 10000000 --no-normalize-ext-reward
#
# Prereqs: conda env "eipo" built on the master node (see continuous/README.md)
# and `mkdir -p slurm_outputs` in the repo root.

#SBATCH --job-name=eipo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --nodelist=n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

module purge
module load miniconda
source activate eipo

# Compute nodes have NO internet access: everything (pip packages, wandb, etc.)
# must already be installed on the master node. TensorBoard logging is local
# and works fine; do not pass --track here.

# SyncVectorEnv steps serially, so each run needs only a few cores; cap torch
# threads to avoid 5 runs oversubscribing the node.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Slurm sets CUDA_VISIBLE_DEVICES from --gres. If the scheduler's view is stale
# (e.g. someone is using GPUs outside Slurm) pin a known-free card explicitly:
#   sbatch --export=ALL,PIN_GPU=3 continuous/slurm/eipo_pack.sbatch ...
if [[ -n "${PIN_GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$PIN_GPU"
fi

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
