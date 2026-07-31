"""Clean 300k-step report for the SAC comparison (kds_vs_eipo.py JSON logs).

For each method (EIPO, KDS) x bonus (RND, Disagreement) prints:
  - HalfCheetah: deterministic eval return at <= 300k (mean of last 3 eval points)
  - AntMaze:     deterministic eval success rate at <= 300k (same protocol)
aggregated as mean +- std (population) across seeds. For EIPO runs the final
alpha is shown per seed; alpha pinned at a clip bound (<=0.001 or >=10.0) marks
a run from the pre-fix code whose baseline is degenerate -- rerun those.

Usage:
  python continuous/sac_300k_report.py [--sac-dir .] [--seeds 1 2 3] [--max-step 300000]
"""

import argparse
import glob
import json
import os
import re

import numpy as np

CELLS = [("halfcheetah", "eval_return", "return", 1),
         ("antmaze", "eval_success", "success rate", 2)]
METHODS = [("EIPO", "eipo"), ("KDS", "kds")]
BONUSES = [("RND", "rnd"), ("Disagreement", "disagreement")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sac-dir", default=".")
    p.add_argument("--max-step", type=int, default=300_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--last-k", type=int, default=3)
    args = p.parse_args()

    print(f"SAC @ {args.max_step:,} steps, seeds {args.seeds} "
          f"(deterministic eval, mean of last {args.last_k} eval points <= max step)\n")
    for env, key, metric, nd in CELLS:
        print(f"{env} (mean ± std of {metric}):")
        for mlabel, m in METHODS:
            for blabel, b in BONUSES:
                vals, notes = [], []
                for path in sorted(glob.glob(os.path.join(args.sac_dir, f"{m}_{b}_{env}_seed*.json"))):
                    sm = re.search(r"_seed(\d+)\.json$", path)
                    if not sm or int(sm.group(1)) not in args.seeds:
                        continue
                    with open(path) as f:
                        log = json.load(f)
                    series = [v for s, v in zip(log["steps"], log.get(key, []))
                              if s <= args.max_step and v is not None]
                    if not series:
                        continue
                    v = float(np.mean(series[-args.last_k:]))
                    vals.append(v)
                    note = f"seed{sm.group(1)}={v:.{nd}f}"
                    if m == "eipo":
                        a = (log.get("alpha") or [None])[-1]
                        if a is not None:
                            flag = "  <-- ALPHA AT BOUND, pre-fix run, RERUN" if a <= 0.0011 or a >= 9.999 else ""
                            note += f" (alpha={a:.3f}){flag}"
                    notes.append(note)
                label = f"SAC + {mlabel} ({blabel})"
                if vals:
                    print(f"  {label:28s} ---> {np.mean(vals):.{nd}f} ± {np.std(vals):.{nd}f}   (n={len(vals)})")
                    for n_ in notes:
                        print(f"      {n_}")
                else:
                    print(f"  {label:28s} ---> no runs found")
        print()


if __name__ == "__main__":
    main()
