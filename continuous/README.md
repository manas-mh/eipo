# EIPO for continuous control (HalfCheetah, AntMaze)

Standalone, single-file port of EIPO ([Redeeming Intrinsic Rewards via Constrained
Optimization](https://williamd4112.github.io/pubs/neurips22_eipo.pdf), NeurIPS'22)
with RND intrinsic rewards to continuous-control environments, PPO base.

The min-max algorithm is ported from this repo's rlpyt implementation
(`rlpyt/algos/pg/ppo.py`, `rlpyt/algos/pg/base.py`); the Atari-specific parts
(conv RND, LSTM policy, image preprocessing) are replaced by MLPs over state
vectors. Nothing in this directory depends on the rlpyt code or the root
`requirements.txt`.

## Setup on the Tanuh AI cluster

All installs happen on the **master node** (10.16.63.40) — compute nodes have
no internet access. Everything lands on NFS (`/home`), so it is automatically
visible on n1/n2.

Dependencies are managed with **uv** via the root `pyproject.toml`. One-time
setup:

```bash
ssh <user>@10.16.63.40
git clone https://github.com/manas-mh/eipo.git && cd eipo
mkdir -p slurm_outputs            # sbatch fails silently without it

uv sync                           # creates ./.venv and installs everything
git add uv.lock && git commit -m "Add lockfile"   # commit the lock for reproducibility
```

The sbatch scripts activate `./.venv` via `$SLURM_SUBMIT_DIR`, so always submit
jobs from the repo root. No conda and no `module load miniconda` needed at job
time.

Notes:
- **The root `requirements.txt` is NOT for this baseline** — it is the legacy
  2020 rlpyt/Atari stack (torch 1.5, gym 0.17, absl-py 0.9) and does not build
  on modern Python. `uv sync` uses `pyproject.toml`, which contains only the
  continuous-control deps (`continuous/requirements.txt` mirrors them for
  non-uv users). Do not run `uv add -r requirements.txt`.
- The H200s (n2) work with the default torch wheel (CUDA 12.x, sm_90). For
  Blackwell cards elsewhere (RTX 5060) use the cu128 torch index.
- No MuJoCo license/key needed — `gymnasium[mujoco]` ships the simulator.

## Smoke test (run this before burning cluster time)

Per cluster rules, don't compute on the master node — use an interactive
session on n2:

```bash
srun --partition=normal --nodelist=n2 --gres=gpu:1 --cpus-per-task=8 --pty bash
module purge
cd ~/eipo && source .venv/bin/activate
python continuous/eipo_ppo_continuous.py --env-id HalfCheetah-v4 \
    --total-timesteps 40000 --num-envs 4 --num-steps 128
python continuous/eipo_ppo_continuous.py --env-id AntMaze_UMaze-v5 \
    --total-timesteps 40000 --num-envs 4 --num-steps 128 --no-normalize-ext-reward
exit   # release the GPU
```

Each should finish in a couple of minutes, print iteration lines with returns /
alpha / stage, and write TensorBoard logs to `results_continuous/`. Note the SPS
value — total runtime ≈ total_timesteps / SPS.

## Launch the real runs

From the repo root on the master node (check `sinfo` first):

```bash
sbatch continuous/slurm/eipo_pack.sbatch HalfCheetah-v4 3000000
sbatch continuous/slurm/eipo_pack.sbatch AntMaze_UMaze-v5 10000000 --no-normalize-ext-reward
```

Each job requests one H200 on n2 and packs 5 seeds concurrently on it (the
workload is CPU-bound; each run needs <1 GB VRAM). The two jobs together use 2
of the free GPUs. Slurm assigns the card via `--gres`; if its accounting is
stale because someone is computing outside Slurm, pin a known-free card
(e.g. 3, 4, 5, or 7):

```bash
sbatch --export=ALL,PIN_GPU=3 continuous/slurm/eipo_pack.sbatch HalfCheetah-v4 3000000
sbatch --export=ALL,PIN_GPU=4 continuous/slurm/eipo_pack.sbatch AntMaze_UMaze-v5 10000000 --no-normalize-ext-reward
```

`eipo_array.sbatch` is a one-run-per-GPU alternative (`--array=0-9%4` keeps it
within the 4 free cards).

Monitoring: `squeue -u $USER` and `tail -f slurm_outputs/eipo_<jobid>.out`.
For TensorBoard, run it on the master node (it's lightweight) and tunnel:
`ssh -L 6006:localhost:6006 <user>@10.16.63.40`, then
`tensorboard --logdir ~/eipo/results_continuous`. Key scalars:
`charts/episodic_return`, `charts/success_rate` (AntMaze), `eipo/alpha`,
`eipo/is_max_step`, `charts/intrinsic_reward_mean`.

Wandb (`--track`) will NOT work from compute nodes (no internet) — stick to
TensorBoard, or sync offline runs from the master node afterwards.

## Curiosity modules

`--curiosity rnd` (default) or `--curiosity disagreement` (ensemble of 5 MLP
forward models; bonus = ensemble prediction variance, ported from
rlpyt/models/curiosity/disagreement.py). Run names are tagged
`eipo_rnd` / `eipo_disagreement` and `aggregate.py` reports them as separate
rows (runs from before this option existed are tagged `eipo_ppo` = RND).

## KDS vs EIPO comparison suite

Three entry points, all supporting `--seed` and launchable with the generic
packer `sbatch slurm/pack.sbatch <n_seeds> <script.py> [args...]`:

- `kds_vs_eipo.py` (repo root) — SAC base, `--method {kds,eipo}`,
  `--bonus {disagreement,rnd}`, `--steps 300000`. Writes JSON eval logs.
- `continuous/kds_ppo_continuous.py` — PPO base KDS for HalfCheetah/AntMaze,
  `--curiosity {rnd,disagreement}`; pairs with `eipo_ppo_continuous.py` and
  shares its logging/eval/aggregate tooling (runs tagged `kds_<curiosity>`).
- `atari/ppo_atari_intrinsic.py` — hard-exploration Atari
  (ALE/MontezumaRevenge-v5), `--method {eipo,kds}`, conv-RND bonus, 3M or 10M
  steps. Logs to `results_atari/`.

## Defaults and knobs

PPO: 16 envs x 256 steps (batch 4096), lr 3e-4, clip 0.2, 4 epochs x 4
minibatches, gamma 0.99 (ext and int), GAE lambda 0.95, grad norm 1.0.
EIPO (paper config): alpha0 0.5, adaptive alpha with lr 0.005 and gradient clip
±0.05, `diff` stage switching. RND: feature dim 256, predictor update
proportion 0.25, intrinsic rewards normalized by running std of their
discounted accumulation.

- `--no-normalize-ext-reward` — use for sparse-reward envs (AntMaze): the
  running-std normalizer explodes when almost all rewards are zero.
- AntMaze observations are `[observation, desired_goal]` flattened; success
  rate is logged as the fraction of episodes that ever receive reward > 0.5.
- Port deviation (documented in the file header): the `diff` switching rule
  uses pre-normalization advantage means. The rlpyt code applied it after
  advantage normalization, where the mean is 0 by construction, making the
  rule degenerate to near-unconditional flipping; use `--minmax-switch none`
  for that behavior.
- Truncation (e.g. HalfCheetah's 1000-step time limit) is treated as
  termination, as in the original rlpyt code and classic CleanRL PPO.
