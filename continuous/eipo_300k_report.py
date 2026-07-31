"""Clean 300k-step report for PPO-based EIPO: RND vs Disagreement bonus.

Reads TensorBoard logs from results_continuous/ and prints, for each bonus:
  - HalfCheetah: training return at <= 300k steps (mean of last 10 points)
  - AntMaze:     training success rate at <= 300k steps (mean of last 10 points)
aggregated as mean +- std (population) across seeds.

Run tags: 'eipo_ppo' = the original PPO+EIPO runs (RND bonus, trained past
300k -- metrics are truncated at 300k); 'eipo_rnd' = later RND runs if any;
'eipo_disagreement' = the Disagreement runs.

Usage:
  python continuous/eipo_300k_report.py [--seeds 1 2 3] [--max-step 300000]
"""

import argparse
import glob
import os
import re

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BONUS_TAGS = {"EIPO (RND)": ("eipo_ppo", "eipo_rnd"),
              "EIPO (Disagreement)": ("eipo_disagreement",),
              "KDS (RND)": ("kds_rnd",),
              "KDS (Disagreement)": ("kds_disagreement",)}
ENV_METRIC = [("HalfCheetah-v4", "charts/episodic_return", "return", 1),
              ("AntMaze_UMaze-v5", "charts/success_rate", "success rate", 2)]


def final_value(run_dir, tag, max_step, last_k=10):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    vals = [s.value for s in ea.Scalars(tag) if s.step <= max_step]
    return float(np.mean(vals[-last_k:])) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="results_continuous")
    p.add_argument("--max-step", type=int, default=300_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()

    print(f"PPO + EIPO @ {args.max_step:,} steps, seeds {args.seeds} "
          f"(training metric, mean of last 10 points <= max step)\n")
    for env_id, tag, metric, nd in ENV_METRIC:
        print(f"{env_id} (mean ± std of {metric}):")
        for bonus, exp_names in BONUS_TAGS.items():
            vals, used = [], []
            for exp in exp_names:
                for run in sorted(glob.glob(os.path.join(args.logdir, f"{env_id}__{exp}__seed*__*"))):
                    m = re.search(r"__seed(\d+)__", run)
                    if m and int(m.group(1)) in args.seeds:
                        v = final_value(run, tag, args.max_step)
                        if v is not None:
                            vals.append(v)
                            used.append(os.path.basename(run))
            if vals:
                print(f"  {bonus:20s} ---> {np.mean(vals):.{nd}f} ± {np.std(vals):.{nd}f}   (n={len(vals)})")
                for u, v in zip(used, vals):
                    print(f"      {u}: {v:.{nd}f}")
            else:
                print(f"  {bonus:20s} ---> no runs found")
        print()


if __name__ == "__main__":
    main()
