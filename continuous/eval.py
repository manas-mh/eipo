"""Evaluate a trained EIPO checkpoint (model.pt) deterministically.

Runs the max-player policy pi (the EIPO policy, trained on extrinsic+intrinsic)
with mean actions for --episodes fresh episodes, and writes eval.json next to
the checkpoint. Also evaluates pi' (extrinsic-only min-player) for reference.

Usage:
  python continuous/eval.py results_continuous/<run>/model.pt --episodes 20
  python continuous/eval.py 'results_continuous/HalfCheetah*/model.pt'   # glob ok
"""

import argparse
import glob
import json
import os

import gymnasium as gym
import numpy as np
import torch

from eipo_ppo_continuous import Actor, FlattenGoalObservation


def evaluate_actor(actor, env, obs_mean, obs_std, episodes, seed):
    returns, lengths, successes = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret, ep_len, ep_suc = False, 0.0, 0, 0.0
        while not done:
            n_obs = np.clip((obs.astype(np.float64) - obs_mean) / obs_std, -10.0, 10.0)
            with torch.no_grad():
                action = actor.mean(torch.as_tensor(n_obs, dtype=torch.float32).unsqueeze(0))
            action = action.squeeze(0).numpy().clip(env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            ep_len += 1
            ep_suc = max(ep_suc, float(reward > 0.5))
        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(ep_suc)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "returns": [float(r) for r in returns],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="model.pt path(s) or glob(s)")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000, help="eval episode seed base")
    args = p.parse_args()

    paths = sorted(set(sum((glob.glob(c) for c in args.checkpoints), [])))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.checkpoints}")

    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        train_args = ckpt["args"]
        env_id = train_args["env_id"]
        if env_id.lower().startswith("antmaze"):
            import gymnasium_robotics
            gym.register_envs(gymnasium_robotics)
        env = gym.make(env_id)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenGoalObservation(env)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        obs_mean = np.asarray(ckpt["obs_rms"]["mean"])
        obs_std = np.sqrt(np.asarray(ckpt["obs_rms"]["var"])) + 1e-8

        result = {"env_id": env_id, "seed": train_args["seed"],
                  "episodes": args.episodes, "alpha_final": ckpt.get("alpha")}
        for name, key in [("pi", "actor_pi"), ("pi_prime", "actor_pi_prime")]:
            actor = Actor(obs_dim, act_dim)
            actor.load_state_dict(ckpt[key])
            actor.eval()
            result[name] = evaluate_actor(actor, env, obs_mean, obs_std,
                                          args.episodes, args.seed)
        env.close()

        out = os.path.join(os.path.dirname(path), "eval.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"{env_id} seed {train_args['seed']}: "
              f"pi {result['pi']['return_mean']:.1f}±{result['pi']['return_std']:.1f} "
              f"(success {result['pi']['success_rate']:.2f}) | "
              f"pi' {result['pi_prime']['return_mean']:.1f}±{result['pi_prime']['return_std']:.1f} "
              f"-> {out}")


if __name__ == "__main__":
    main()
