# EIPO for continuous control (HalfCheetah, AntMaze)

Standalone, single-file port of EIPO ([Redeeming Intrinsic Rewards via Constrained
Optimization](https://williamd4112.github.io/pubs/neurips22_eipo.pdf), NeurIPS'22)
with RND intrinsic rewards to continuous-control environments, PPO base.

The min-max algorithm is ported from this repo's rlpyt implementation
(`rlpyt/algos/pg/ppo.py`, `rlpyt/algos/pg/base.py`); the Atari-specific parts
(conv RND, LSTM policy, image preprocessing) are replaced by MLPs over state
vectors. Nothing in this directory depends on the rlpyt code or the root
`requirements.txt`.

## Setup (cluster)

```bash
git clone <this repo> && cd eipo
python -m venv ~/venvs/eipo && source ~/venvs/eipo/bin/activate
pip install -r continuous/requirements.txt
mkdir -p slurm_outputs
```

Notes:
- On Blackwell GPUs (RTX 5060) install torch from the cu128 index (see
  `continuous/requirements.txt`). The H200 works with the default wheel.
- No MuJoCo license/key needed — `gymnasium[mujoco]` ships the simulator.

## Smoke test (run this before burning cluster time)

```bash
python continuous/eipo_ppo_continuous.py --env-id HalfCheetah-v4 \
    --total-timesteps 40000 --num-envs 4 --num-steps 128
python continuous/eipo_ppo_continuous.py --env-id AntMaze_UMaze-v5 \
    --total-timesteps 40000 --num-envs 4 --num-steps 128 --no-normalize-ext-reward
```

Each should finish in a couple of minutes, print iteration lines with returns /
alpha / stage, and write TensorBoard logs to `results_continuous/`. Note the SPS
value — total runtime ≈ total_timesteps / SPS.

## Launch the real runs

Edit the `EDIT ME` block in `continuous/slurm/eipo_pack.sbatch` (partition/module
loads/venv path), then from the repo root:

```bash
sbatch continuous/slurm/eipo_pack.sbatch HalfCheetah-v4 3000000
sbatch continuous/slurm/eipo_pack.sbatch AntMaze_UMaze-v5 10000000 --no-normalize-ext-reward
```

Each job packs 5 seeds concurrently on one GPU (the workload is CPU-bound;
each run needs <1 GB VRAM and ~8 CPU cores). `eipo_array.sbatch` is a
one-run-per-job alternative if your cluster has multiple GPUs.

Monitoring: `tensorboard --logdir results_continuous`. Key scalars:
`charts/episodic_return`, `charts/success_rate` (AntMaze), `eipo/alpha`,
`eipo/is_max_step`, `charts/intrinsic_reward_mean`.

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
