"""KDS on a PPO base for continuous control (HalfCheetah, AntMaze).

On-policy adaptation of the KDS reward shaping from kds_vs_eipo.py (Eqs. 14-17:
Sinkhorn-Knopp reweighting of the intrinsic bonus over a batch, with the target
distribution built from advantage and bonus logits). Where the SAC version
applies KDS per replay minibatch, here it is applied once per PPO rollout batch,
before GAE:

  1. collect a rollout with the (single) Gaussian policy;
  2. compute the intrinsic bonus (RND or Disagreement, same modules and
     normalization as eipo_ppo_continuous.py);
  3. compute extrinsic GAE advantages A_E with the current value function --
     these play the role of the one-step TD advantage in the SAC version;
  4. w = kds_weights(phi(s), A_E, bonus) over the whole T*B batch
     (phi = fixed random projection of the raw state, Eq. 15);
  5. shaped reward r = r_E + beta * bonus * w; standard PPO (clip, epochs,
     minibatches) on GAE of the shaped reward.

PPO hyperparameters match eipo_ppo_continuous.py; KDS hyperparameters default to
the paper's Table 3 (tau_A=tau_b=0.5, eps=0.05, [delta,cap]=[0.2,3.0], beta=0.1).

Example:
  python continuous/kds_ppo_continuous.py --env-id HalfCheetah-v4 \
      --total-timesteps 300000 --curiosity rnd --seed 1
  python continuous/kds_ppo_continuous.py --env-id AntMaze_UMaze-v5 \
      --total-timesteps 300000 --curiosity disagreement \
      --no-normalize-ext-reward --seed 1
"""

import argparse
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from eipo_ppo_continuous import (
    Actor, Disagreement, RND, RewardForwardFilter, RunningMeanStd,
    clipped_pg_loss, gae, layer_init, make_env,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=None,
                   help="run-name tag; defaults to kds_<curiosity>")
    p.add_argument("--env-id", type=str, default="HalfCheetah-v4")
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-dir", type=str, default="results_continuous")
    p.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)

    # PPO (matches eipo_ppo_continuous.py defaults)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gamma-int", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--normalize-ext-reward", action=argparse.BooleanOptionalAction, default=True)

    # Curiosity module (same options as eipo_ppo_continuous.py)
    p.add_argument("--curiosity", type=str, default="disagreement", choices=["rnd", "disagreement"])
    p.add_argument("--prediction-beta", type=float, default=1.0)
    p.add_argument("--rnd-update-proportion", type=float, default=0.25)
    p.add_argument("--rnd-feature-size", type=int, default=256)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--forward-loss-wt", type=float, default=0.2)

    # KDS (paper Table 3)
    p.add_argument("--kds-beta", type=float, default=0.1, help="intrinsic shaping scale")
    p.add_argument("--kds-tau-a", type=float, default=0.5)
    p.add_argument("--kds-tau-b", type=float, default=0.5)
    p.add_argument("--kds-eps", type=float, default=0.05)
    p.add_argument("--kds-delta", type=float, default=0.2)
    p.add_argument("--kds-cap", type=float, default=3.0)
    p.add_argument("--kds-iters", type=int, default=50)
    p.add_argument("--kds-proj-dim", type=int, default=32)

    args = p.parse_args()
    if args.exp_name is None:
        args.exp_name = f"kds_{args.curiosity}"
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


# ---- KDS pieces (verbatim from kds_vs_eipo.py, Eqs. 14-17) ----------------- #
class RandomProjection(nn.Module):
    def __init__(self, obs_dim, proj_dim=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(obs_dim, proj_dim, generator=g) / np.sqrt(obs_dim)
        self.register_buffer("W", W)

    def forward(self, obs):
        return obs @ self.W


def kds_weights(phi_obs, adv, bonus, tau_A=0.5, tau_b=0.5, eps=0.05, iters=50,
                delta=0.2, cap=3.0):
    N = phi_obs.shape[0]
    logits = adv / tau_A + bonus / tau_b
    logits = logits - logits.max()
    nu = torch.exp(logits)
    nu = nu / nu.sum()
    mu = torch.full((N,), 1.0 / N, device=phi_obs.device)

    C = torch.cdist(phi_obs, phi_obs, p=2) ** 2
    C = C / (C.mean() + 1e-8)
    K = torch.exp(-C / eps)

    u = torch.full((N,), 1.0 / N, device=phi_obs.device)
    v = torch.full((N,), 1.0 / N, device=phi_obs.device)
    for _ in range(iters):
        u = mu / (K @ v + 1e-12)
        v = nu / (K.t() @ u + 1e-12)

    vbar = v.mean()
    w = torch.clamp(v / vbar, delta, cap)
    w = w / w.mean()
    return w
# --------------------------------------------------------------------------- #


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__seed{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.env_id.lower().startswith("antmaze"):
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)

    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id) for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    actor = Actor(obs_dim, act_dim).to(device)
    critic = ValueNet(obs_dim).to(device)
    if args.curiosity == "rnd":
        curiosity = RND(obs_dim, args.rnd_feature_size, args.prediction_beta,
                        args.gamma_int, args.rnd_update_proportion, device).to(device)
        curiosity_params = list(curiosity.predictor.parameters())
    else:
        curiosity = Disagreement(obs_dim, act_dim, args.ensemble_size,
                                 args.prediction_beta, args.gamma_int,
                                 args.forward_loss_wt, device).to(device)
        curiosity_params = list(curiosity.ensemble.parameters())
    phi = RandomProjection(obs_dim, args.kds_proj_dim, seed=args.seed).to(device)
    params = list(actor.parameters()) + list(critic.parameters()) + curiosity_params
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ext_rff = RewardForwardFilter(args.gamma)
    ext_rms = RunningMeanStd()

    def norm_obs(o):
        o = np.clip((o - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8), -10.0, 10.0)
        return torch.as_tensor(o, dtype=torch.float32, device=device)

    T, B = args.num_steps, args.num_envs
    obs_buf = torch.zeros((T, B, obs_dim), device=device)
    raw_obs_buf = torch.zeros((T, B, obs_dim), device=device)
    next_obs_buf = torch.zeros((T, B, obs_dim), device=device)
    act_buf = torch.zeros((T, B, act_dim), device=device)
    logp_buf = torch.zeros((T, B), device=device)
    rew_buf = torch.zeros((T, B), device=device)
    done_buf = torch.zeros((T, B), device=device)
    val_buf = torch.zeros((T, B), device=device)

    ep_ret = np.zeros(B)
    ep_len = np.zeros(B, dtype=np.int64)
    ep_success = np.zeros(B)
    recent_returns = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)
    recent_success = deque(maxlen=50)

    raw_next_obs, _ = envs.reset(seed=args.seed)
    raw_next_obs = raw_next_obs.astype(np.float32)
    obs_rms.update(raw_next_obs)

    global_step = 0
    start_time = time.time()
    num_iterations = args.total_timesteps // args.batch_size

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for t in range(T):
            global_step += B
            raw_obs_buf[t] = torch.as_tensor(raw_next_obs, dtype=torch.float32, device=device)
            n_obs = norm_obs(raw_next_obs)
            obs_buf[t] = n_obs
            with torch.no_grad():
                dist = actor.dist(n_obs)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action).sum(-1)
                val_buf[t] = critic(n_obs)
            act_buf[t] = action

            raw_next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            raw_next_obs = raw_next_obs.astype(np.float32)
            done = np.logical_or(terminated, truncated).astype(np.float32)
            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            done_buf[t] = torch.as_tensor(done, device=device)
            next_obs_buf[t] = torch.as_tensor(raw_next_obs, dtype=torch.float32, device=device)
            obs_rms.update(raw_next_obs)

            ep_ret += reward
            ep_len += 1
            ep_success = np.maximum(ep_success, (np.asarray(reward) > 0.5).astype(np.float64))
            for i in np.nonzero(done)[0]:
                recent_returns.append(ep_ret[i])
                recent_lengths.append(ep_len[i])
                recent_success.append(ep_success[i])
                ep_ret[i], ep_len[i], ep_success[i] = 0.0, 0, 0.0

        # ---- intrinsic bonus (normalized + done-masked, as in EIPO file) ---#
        if args.curiosity == "rnd":
            int_rew = curiosity.compute_bonus(next_obs_buf, done_buf)
        else:
            int_rew = curiosity.compute_bonus(raw_obs_buf, next_obs_buf, act_buf, done_buf)

        if args.normalize_ext_reward:
            rew_np = rew_buf.cpu().numpy()
            nd_np = (1.0 - done_buf).cpu().numpy()
            rff = np.array([ext_rff.update(rew_np[t], not_done=nd_np[t]) for t in range(T)])
            ext_rms.update_from_moments(rff.mean(), rff.var(), len(rff))
            ext_rew = rew_buf / float(np.sqrt(ext_rms.var) + 1e-10)
        else:
            ext_rew = rew_buf

        # ---- KDS reweighting over the rollout batch ---------------------- #
        with torch.no_grad():
            next_value = critic(norm_obs(raw_next_obs))
            adv_ext, _ = gae(ext_rew, val_buf, done_buf, next_value, args.gamma, args.gae_lambda)
            w = kds_weights(phi(raw_obs_buf.reshape(T * B, obs_dim)),
                            adv_ext.reshape(T * B), int_rew.reshape(T * B),
                            args.kds_tau_a, args.kds_tau_b, args.kds_eps,
                            args.kds_iters, args.kds_delta, args.kds_cap)
            shaped_rew = ext_rew + args.kds_beta * int_rew * w.reshape(T, B)
            advantages, returns = gae(shaped_rew, val_buf, done_buf, next_value,
                                      args.gamma, args.gae_lambda)
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-6)

        # ---- PPO update --------------------------------------------------- #
        b_obs = obs_buf.reshape(T * B, obs_dim)
        b_raw_obs = raw_obs_buf.reshape(T * B, obs_dim)
        b_next_obs = next_obs_buf.reshape(T * B, obs_dim)
        b_act = act_buf.reshape(T * B, act_dim)
        b_logp = logp_buf.reshape(T * B)
        b_adv = advantages.reshape(T * B)
        b_ret = returns.reshape(T * B)

        idxs = np.arange(T * B)
        for _ in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = idxs[start:start + args.minibatch_size]
                dist = actor.dist(b_obs[mb])
                logp = dist.log_prob(b_act[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                value = critic(b_obs[mb])

                pi_loss = clipped_pg_loss(logp, b_logp[mb], b_adv[mb], args.clip_coef)
                v_loss = args.vf_coef * 0.5 * ((value - b_ret[mb]) ** 2).mean()
                if args.curiosity == "rnd":
                    curiosity_loss = curiosity.loss(b_next_obs[mb])
                else:
                    curiosity_loss = curiosity.loss(b_raw_obs[mb], b_next_obs[mb], b_act[mb])
                loss = pi_loss + v_loss - args.ent_coef * entropy + curiosity_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        if recent_returns:
            writer.add_scalar("charts/episodic_return", np.mean(recent_returns), global_step)
            writer.add_scalar("charts/episodic_length", np.mean(recent_lengths), global_step)
            writer.add_scalar("charts/success_rate", np.mean(recent_success), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/intrinsic_reward_mean", int_rew.mean().item(), global_step)
        writer.add_scalar("kds/weight_mean", w.mean().item(), global_step)
        writer.add_scalar("kds/weight_max", w.max().item(), global_step)
        writer.add_scalar("losses/policy_loss", pi_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/curiosity_loss", curiosity_loss.item(), global_step)

        if iteration % 10 == 0 or iteration == num_iterations:
            ret = f"{np.mean(recent_returns):9.2f}" if recent_returns else "      n/a"
            print(f"iter {iteration:5d}/{num_iterations} step {global_step:>10,d} "
                  f"return {ret} SPS {sps}", flush=True)

    if args.save_final_model:
        path = os.path.join(args.log_dir, run_name, "model.pt")
        torch.save({
            "actor_pi": actor.state_dict(),
            "critic": critic.state_dict(),
            "curiosity_type": args.curiosity,
            "curiosity": curiosity.state_dict(),
            "obs_rms": {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
            "args": vars(args),
        }, path)
        print(f"saved model to {path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
