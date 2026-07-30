"""Aggregate EIPO runs into a shareable table (markdown + CSV).

Collects, per environment across seeds:
  - final training return (mean of the last 10 logged points of
    charts/episodic_return in the TensorBoard events), and
  - deterministic eval results from eval.json if continuous/eval.py was run.

Usage:
  python continuous/aggregate.py [--logdir results_continuous]
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


def tb_final(run_dir, tag, last_k=10):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    vals = [s.value for s in ea.Scalars(tag)]
    return float(np.mean(vals[-last_k:])) if vals else None


def fmt(mean, std=None, nd=1):
    if mean is None:
        return "-"
    return f"{mean:.{nd}f} ± {std:.{nd}f}" if std is not None else f"{mean:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="results_continuous")
    args = p.parse_args()

    by_env = defaultdict(list)
    for run_dir in sorted(glob.glob(os.path.join(args.logdir, "*__eipo_ppo__*"))):
        name = os.path.basename(run_dir)
        env_id = name.split("__")[0]
        row = {"run": name,
               "train_return": tb_final(run_dir, "charts/episodic_return"),
               "train_success": tb_final(run_dir, "charts/success_rate")}
        eval_path = os.path.join(run_dir, "eval.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                ev = json.load(f)
            row["eval_return"] = ev["pi"]["return_mean"]
            row["eval_success"] = ev["pi"]["success_rate"]
            row["alpha_final"] = ev.get("alpha_final")
        by_env[env_id].append(row)

    md = ["| Environment | Seeds | Train return (final) | Eval return (deterministic π) | Eval success rate |",
          "|---|---|---|---|---|"]
    csv_rows = [["env", "n_seeds", "train_return_mean", "train_return_std",
                 "eval_return_mean", "eval_return_std", "eval_success_mean", "eval_success_std"]]

    for env_id, rows in sorted(by_env.items()):
        tr = [r["train_return"] for r in rows if r["train_return"] is not None]
        er = [r["eval_return"] for r in rows if r.get("eval_return") is not None]
        es = [r["eval_success"] for r in rows if r.get("eval_success") is not None]
        is_maze = "maze" in env_id.lower()
        md.append("| {} | {} | {} | {} | {} |".format(
            env_id, len(rows),
            fmt(np.mean(tr) if tr else None, np.std(tr) if tr else None),
            fmt(np.mean(er) if er else None, np.std(er) if er else None),
            fmt(np.mean(es) if es and is_maze else None,
                np.std(es) if es and is_maze else None, nd=2)))
        csv_rows.append([env_id, len(rows),
                         np.mean(tr) if tr else "", np.std(tr) if tr else "",
                         np.mean(er) if er else "", np.std(er) if er else "",
                         np.mean(es) if es else "", np.std(es) if es else ""])

    md_body = "\n".join(md)
    per_seed = ["", "Per-seed detail:", ""]
    for env_id, rows in sorted(by_env.items()):
        for r in rows:
            per_seed.append(f"- {r['run']}: train {fmt(r['train_return'])}, "
                            f"eval {fmt(r.get('eval_return'))}, "
                            f"success {fmt(r.get('eval_success'), nd=2)}, "
                            f"alpha {fmt(r.get('alpha_final'), nd=3)}")
    md_body += "\n" + "\n".join(per_seed)

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "summary.md"), "w") as f:
        f.write(md_body + "\n")
    with open(os.path.join(args.logdir, "summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(md_body)
    print(f"\nwrote {args.logdir}/summary.md and {args.logdir}/summary.csv")


if __name__ == "__main__":
    main()
