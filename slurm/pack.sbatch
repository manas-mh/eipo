#!/bin/bash
# Tanuh AI cluster: generic seed-pack launcher. Runs N seeds of ANY training
# script in this repo concurrently on a single GPU on n2.
#
# Usage (from the repo root on the master node):
#   sbatch slurm/pack.sbatch <n_seeds> <script.py> [script args...]
#
# Examples:
#   sbatch slurm/pack.sbatch 5 kds_vs_eipo.py --env halfcheetah --method kds --bonus rnd --steps 300000
#   sbatch slurm/pack.sbatch 5 continuous/kds_ppo_continuous.py --env-id HalfCheetah-v4 --total-timesteps 300000
#   sbatch slurm/pack.sbatch 3 atari/ppo_atari_intrinsic.py --method eipo --total-timesteps 10000000
#
# Seeds 1..N are appended as --seed <i>. Pin a free GPU with
# --export=ALL,PIN_GPU=<n> as with the other scripts.

#SBATCH --job-name=pack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --nodelist=n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

module purge
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
if [[ -n "${PIN_GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$PIN_GPU"
fi

N_SEEDS=${1:?usage: sbatch slurm/pack.sbatch <n_seeds> <script.py> [args...]}
SCRIPT=${2:?missing script path}
shift 2

for seed in $(seq 1 "$N_SEEDS"); do
  python "$SCRIPT" "$@" --seed "$seed" &
done
wait
