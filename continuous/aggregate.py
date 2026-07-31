"""Aggregate runs into a shareable table (markdown + CSV).

Collects, per (environment, algorithm) across seeds:
  - PPO runs (TensorBoard dirs in --logdir): final training return, plus
    deterministic eval from eval.json if continuous/eval.py was run;
  - SAC runs (kds_vs_eipo.py JSON logs, --sac-dir): the in-training
    deterministic eval curve, reported as the mean of the last 3 eval points
    (algorithm tagged sac_<method>_<bonus>).

Usage:
  python continuous/aggregate.py [--logdir results_continuous] [--sac-dir .]
  python continuous/aggregate.py --max-step 300000   # budget-matched metrics
Writes <logdir>/summary.md and <logdir>/summary.csv and prints the table.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tb_final(run_dir, tag, last_k=10, max_step=None):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    scalars = ea.Scalars(tag)
    if max_step is not None:
        scalars = [s for s in scalars if s.step <= max_step]
    vals = [s.value for s in scalars]
    return float(np.mean(vals[-last_k:])) if vals else None


SAC_ENV_NAMES = {"halfcheetah": "HalfCheetah-v4", "antmaze": "AntMaze_UMaze (SAC ids vary)"}


def sac_json_rows(sac_dir, max_step=None, last_k=3):
    """Parse kds_vs_eipo.py logs named <method>_<bonus>_<env>_seed<N>.json."""
    import re
    rows = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(sac_dir, "*_seed*.json"))):
        m = re.match(r"(kds|eipo)_(disagreement|rnd)_(halfcheetah|antmaze)_seed(\d+)\.json$",
                     os.path.basename(path))
        if not m:
            continue
        method, bonus, env_key, seed = m.groups()
        with open(path) as f:
            log = json.load(f)
        pts = [(s, r, u) for s, r, u in zip(log["steps"], log["eval_return"],
                                            log.get("eval_success", [None] * len(log["steps"])))
               if max_step is None or s <= max_step]
        if not pts:
            continue
        tail = pts[-last_k:]
        row = {"run": os.path.basename(path),
               "train_return": None,
               "eval_return": float(np.mean([r for _, r, _ in tail])),
               "eval_success": (float(np.mean([u for _, _, u in tail]))
                                if tail[0][2] is not None else None),
               "alpha_final": (log["alpha"][-1] if log.get("alpha") else None)}
        rows[(SAC_ENV_NAMES[env_key], f"sac_{method}_{bonus}")].append(row)
    return rows


def fmt(mean, std=None, nd=1):
    if mean is None:
        return "-"
    return f"{mean:.{nd}f} ± {std:.{nd}f}" if std is not None else f"{mean:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", nargs="+", default=["results_continuous", "results_atari"],
                   help="TensorBoard results dirs to scan (summary is written to the first)")
    p.add_argument("--sac-dir", default=".",
                   help="directory containing kds_vs_eipo.py JSON logs")
    p.add_argument("--max-step", type=int, default=None,
                   help="only use training scalars logged at global_step <= this "
                        "(e.g. 300000 to compare all algorithms at 3e5 steps)")
    args = p.parse_args()

    by_env = defaultdict(list)
    run_dirs = [d for logdir in args.logdir
                for d in sorted(glob.glob(os.path.join(logdir, "*__*__seed*__*")))]
    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        env_id, exp_name = name.split("__")[:2]
        row = {"run": name,
               "train_return": tb_final(run_dir, "charts/episodic_return", max_step=args.max_step),
               "train_success": tb_final(run_dir, "charts/success_rate", max_step=args.max_step)}
        eval_path = os.path.join(run_dir, "eval.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                ev = json.load(f)
            row["eval_return"] = ev["pi"]["return_mean"]
            row["eval_success"] = ev["pi"]["success_rate"]
            row["alpha_final"] = ev.get("alpha_final")
        by_env[(env_id, exp_name)].append(row)

    for key, rows in sac_json_rows(args.sac_dir, max_step=args.max_step).items():
        by_env[key].extend(rows)

    step_note = f" (train metrics at step <= {args.max_step:,})" if args.max_step else ""
    md = [f"| Environment | Algorithm | Seeds | Train return{step_note} | Eval return (deterministic π) | Eval success rate |",
          "|---|---|---|---|---|---|"]
    csv_rows = [["env", "algo", "n_seeds", "train_return_mean", "train_return_std",
                 "eval_return_mean", "eval_return_std", "eval_success_mean", "eval_success_std"]]

    for (env_id, exp_name), rows in sorted(by_env.items()):
        tr = [r["train_return"] for r in rows if r["train_return"] is not None]
        er = [r["eval_return"] for r in rows if r.get("eval_return") is not None]
        es = [r["eval_success"] for r in rows if r.get("eval_success") is not None]
        is_maze = "maze" in env_id.lower()
        md.append("| {} | {} | {} | {} | {} | {} |".format(
            env_id, exp_name, len(rows),
            fmt(np.mean(tr) if tr else None, np.std(tr) if tr else None),
            fmt(np.mean(er) if er else None, np.std(er) if er else None),
            fmt(np.mean(es) if es and is_maze else None,
                np.std(es) if es and is_maze else None, nd=2)))
        csv_rows.append([env_id, exp_name, len(rows),
                         np.mean(tr) if tr else "", np.std(tr) if tr else "",
                         np.mean(er) if er else "", np.std(er) if er else "",
                         np.mean(es) if es else "", np.std(es) if es else ""])

    md_body = "\n".join(md)
    per_seed = ["", "Per-seed detail:", ""]
    for (env_id, exp_name), rows in sorted(by_env.items()):
        for r in rows:
            per_seed.append(f"- {r['run']}: train {fmt(r['train_return'])}, "
                            f"eval {fmt(r.get('eval_return'))}, "
                            f"success {fmt(r.get('eval_success'), nd=2)}, "
                            f"alpha {fmt(r.get('alpha_final'), nd=3)}")
    md_body += "\n" + "\n".join(per_seed)

    out_dir = args.logdir[0]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md_body + "\n")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(md_body)
    print(f"\nwrote {out_dir}/summary.md and {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
