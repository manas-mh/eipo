"""EIPO and KDS on a hard-exploration Atari game (PPO base, RND bonus).

Single-file PPO for ALE Atari with a conv-RND intrinsic bonus and two methods:

  --method eipo   min-max EIPO (same algorithm as continuous/eipo_ppo_continuous.py,
                  ported to a discrete Categorical policy with a shared Nature-CNN
                  trunk and five heads: pi, pi', V_E^pi, V_I^pi, V_E^pi' -- this
                  mirrors the original rlpyt Atari model, minus the LSTM)
  --method kds    single-policy PPO with KDS Sinkhorn reweighting of the RND
                  bonus per rollout batch (Eqs. 14-17 of the KDS paper); the
                  cost-matrix embedding phi(s) is the RND target network's
                  features (a fixed random deep projection of the frame)

Intrinsic bonus is RND only: that is what the EIPO paper used on Atari, and
pixel-based Disagreement would require an action-conditioned latent world model
that neither paper's Atari setup includes.

Environment defaults follow the EIPO/RND Atari conventions: ALE v5 sticky
actions (p=0.25), frame skip 4, 84x84 grayscale, 4-frame stack, no
terminal-on-life-loss (important for Montezuma), sign-clipped extrinsic rewards
for training while logging raw episodic score.

Example (hard-exploration benchmark):
  python atari/ppo_atari_intrinsic.py --method eipo --env-id ALE/MontezumaRevenge-v5 \
      --total-timesteps 10000000 --seed 1
  python atari/ppo_atari_intrinsic.py --method kds --env-id ALE/MontezumaRevenge-v5 \
      --total-timesteps 10000000 --seed 1

Note: steps are agent steps (1 agent step = 4 frames).
"""

import argparse
import os
import random
import time
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=None,
                   help="run-name tag; defaults to <method>_rnd")
    p.add_argument("--method", type=str, default="eipo", choices=["eipo", "kds"])
    p.add_argument("--env-id", type=str, default="ALE/MontezumaRevenge-v5")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-dir", type=str, default="results_atari")
    p.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)

    # PPO (EIPO Atari config: lr 1e-4, clip 0.1, ent 0.001, 4 epochs x 4 minibatches)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gamma-int", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)

    # RND
    p.add_argument("--prediction-beta", type=float, default=1.0)
    p.add_argument("--rnd-update-proportion", type=float, default=0.25)

    # EIPO
    p.add_argument("--minmax-alpha", type=float, default=0.5)
    p.add_argument("--use-adapt-alpha", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alpha-lr", type=float, default=0.005)
    p.add_argument("--alpha-g-clip", type=float, default=0.05)
    p.add_argument("--minmax-switch", type=str, default="diff", choices=["diff", "none"])

    # KDS (paper Table 3)
    p.add_argument("--kds-beta", type=float, default=0.1)
    p.add_argument("--kds-tau-a", type=float, default=0.5)
    p.add_argument("--kds-tau-b", type=float, default=0.5)
    p.add_argument("--kds-eps", type=float, default=0.05)
    p.add_argument("--kds-delta", type=float, default=0.2)
    p.add_argument("--kds-cap", type=float, default=3.0)
    p.add_argument("--kds-iters", type=int, default=50)

    args = p.parse_args()
    if args.exp_name is None:
        args.exp_name = f"{args.method}_rnd"
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.update_from_moments(x.mean(axis=0), x.var(axis=0), x.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews, not_done):
        if self.rewems is None:
            self.rewems = rews.copy()
        else:
            self.rewems = self.rewems * self.gamma * not_done + rews
        return self.rewems.copy()


def make_env(env_id):
    def thunk():
        env = gym.make(env_id, frameskip=1)  # ALE v5: sticky actions p=0.25 by default
        env = gym.wrappers.AtariPreprocessing(
            env, frame_skip=4, screen_size=84, grayscale_obs=True,
            terminal_on_life_loss=False, scale_obs=False)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariAgent(nn.Module):
    """Nature-CNN trunk shared by all heads (mirrors the rlpyt EIPO Atari model,
    feed-forward variant): two policies and three value heads."""

    def __init__(self, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.pi = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.pi_prime = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.v = layer_init(nn.Linear(512, 1), std=1.0)
        self.v_int = layer_init(nn.Linear(512, 1), std=1.0)
        self.v_prime = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        h = self.trunk(x.float() / 255.0)
        return (self.pi(h), self.pi_prime(h),
                self.v(h).squeeze(-1), self.v_int(h).squeeze(-1),
                self.v_prime(h).squeeze(-1))


class ConvRND(nn.Module):
    """Conv RND over the last (single) frame, ported from
    rlpyt/models/curiosity/rnd.py: obs normalized by running mean/std (clip
    +-5), bonus normalized by a running std of its discounted accumulation and
    done-masked; predictor trained on a random subset of samples."""

    def __init__(self, beta, gamma_int, update_proportion):
        super().__init__()
        self.beta = beta
        self.update_proportion = update_proportion
        self.feature_size = 512
        self.obs_rms = RunningMeanStd(shape=(1, 84, 84))
        self.rew_rms = RunningMeanStd()
        self.rew_rff = RewardForwardFilter(gamma_int)

        def convs():
            return [layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
                    layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
                    layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
                    nn.Flatten()]
        self.predictor = nn.Sequential(
            *convs(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )
        self.target = nn.Sequential(*convs(), layer_init(nn.Linear(64 * 7 * 7, 512)))
        for prm in self.target.parameters():
            prm.requires_grad = False

    def _normalize(self, frames):
        """frames: float tensor [..., 1, 84, 84]."""
        mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=frames.device)
        std = torch.as_tensor(np.sqrt(self.obs_rms.var), dtype=torch.float32, device=frames.device)
        return torch.clamp((frames - mean) / (std + 1e-10), -5.0, 5.0)

    @torch.no_grad()
    def features(self, frames_u8):
        """Fixed random projection of frames (used as phi(s) for KDS)."""
        norm = self._normalize(frames_u8.float().unsqueeze(-3))
        return self.target(norm)

    @torch.no_grad()
    def compute_bonus(self, next_frames_u8, dones):
        """next_frames_u8: uint8 [T, B, 84, 84]; dones: [T, B]."""
        T, B = next_frames_u8.shape[:2]
        not_done = (1.0 - dones).cpu().numpy()
        norm = self._normalize(next_frames_u8.float().unsqueeze(2))  # [T,B,1,84,84]
        flat = norm.reshape(T * B, 1, 84, 84)
        err = ((self.predictor(flat) - self.target(flat)) ** 2).sum(-1) / self.feature_size
        bonus = err.reshape(T, B)

        frames_np = next_frames_u8.cpu().numpy().reshape(T * B, 1, 84, 84)
        valid = not_done.reshape(T * B) > 0
        if valid.any():
            self.obs_rms.update(frames_np[valid])

        bonus_np = bonus.cpu().numpy()
        rff = np.array([self.rew_rff.update(bonus_np[t], not_done=not_done[t]) for t in range(T)])
        self.rew_rms.update_from_moments(rff.mean(), rff.var(), not_done.sum())
        bonus = bonus / float(np.sqrt(self.rew_rms.var) + 1e-10)
        bonus = bonus * torch.as_tensor(not_done, dtype=torch.float32, device=bonus.device)
        return self.beta * bonus

    def loss(self, next_frames_u8):
        norm = self._normalize(next_frames_u8.float().unsqueeze(1))  # [N,1,84,84]
        with torch.no_grad():
            phi = self.target(norm)
        err = ((self.predictor(norm) - phi) ** 2).sum(-1) / self.feature_size
        mask = (torch.rand(err.shape, device=err.device) < self.update_proportion).float()
        return (err * mask).sum() / mask.sum().clamp(min=1.0)


def clipped_pg_loss(new_logp, old_logp, advantage, clip_coef):
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage
    return -torch.min(surr1, surr2).mean()


def gae(rewards, values, dones, next_value, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        nextvalue = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextvalue * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv, adv + values


def kds_weights(phi_obs, adv, bonus, tau_A, tau_b, eps, iters, delta, cap):
    """Verbatim from kds_vs_eipo.py (Eqs. 14, 16, 17)."""
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

    w = torch.clamp(v / v.mean(), delta, cap)
    return w / w.mean()


def main():
    args = parse_args()
    run_name = (f"{args.env_id.replace('/', '-')}"
                f"__{args.exp_name}__seed{args.seed}__{int(time.time())}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    gym.register_envs(ale_py)
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id) for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    n_actions = int(envs.single_action_space.n)

    agent = AtariAgent(n_actions).to(device)
    rnd = ConvRND(args.prediction_beta, args.gamma_int, args.rnd_update_proportion).to(device)
    params = list(agent.parameters()) + list(rnd.predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-5)

    # EIPO state
    alpha = args.minmax_alpha
    is_max_step = True
    last_max_adv, last_min_adv = 0.0, 0.0

    T, B = args.num_steps, args.num_envs
    obs_buf = torch.zeros((T, B, 4, 84, 84), dtype=torch.uint8, device=device)
    next_frame_buf = torch.zeros((T, B, 84, 84), dtype=torch.uint8, device=device)
    act_buf = torch.zeros((T, B), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, B), device=device)
    rew_buf = torch.zeros((T, B), device=device)      # sign-clipped (training)
    done_buf = torch.zeros((T, B), device=device)
    v_buf = torch.zeros((T, B), device=device)
    v_int_buf = torch.zeros((T, B), device=device)
    v_prime_buf = torch.zeros((T, B), device=device)

    ep_ret = np.zeros(B)   # raw, unclipped score
    ep_len = np.zeros(B, dtype=np.int64)
    recent_returns = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)

    global_step = 0
    start_time = time.time()
    num_iterations = args.total_timesteps // args.batch_size

    for iteration in range(1, num_iterations + 1):
        use_pi_prime = args.method == "eipo" and is_max_step

        for t in range(T):
            global_step += B
            obs_buf[t] = next_obs
            with torch.no_grad():
                logits_pi, logits_pr, v, v_int, v_prime = agent(next_obs)
                dist = Categorical(logits=logits_pr if use_pi_prime else logits_pi)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action)
            v_buf[t], v_int_buf[t], v_prime_buf[t] = v, v_int, v_prime
            act_buf[t] = action

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)
            done = np.logical_or(terminated, truncated).astype(np.float32)
            rew_buf[t] = torch.as_tensor(np.sign(reward), dtype=torch.float32, device=device)
            done_buf[t] = torch.as_tensor(done, device=device)
            next_frame_buf[t] = next_obs[:, -1]

            ep_ret += reward
            ep_len += 1
            for i in np.nonzero(done)[0]:
                recent_returns.append(ep_ret[i])
                recent_lengths.append(ep_len[i])
                ep_ret[i], ep_len[i] = 0.0, 0

        int_rew = rnd.compute_bonus(next_frame_buf, done_buf)

        with torch.no_grad():
            _, _, nv, nv_int, nv_prime = agent(next_obs)
        adv_E_pi, ret_E_pi = gae(rew_buf, v_buf, done_buf, nv, args.gamma, args.gae_lambda)
        adv_I, ret_I = gae(int_rew, v_int_buf, done_buf, nv_int, args.gamma_int, args.gae_lambda)
        adv_E_pr, ret_E_pr = gae(rew_buf, v_prime_buf, done_buf, nv_prime, args.gamma, args.gae_lambda)

        if args.method == "eipo":
            if is_max_step:
                U = rew_buf + int_rew + alpha * adv_E_pr
                aux_adv = adv_E_pr
            else:
                U = (alpha - 1.0) * rew_buf - int_rew + (adv_E_pi + adv_I)
                aux_adv = adv_E_pi + adv_I
            switch_metric = U.mean().item()
            if args.normalize_advantage:
                U = (U - U.mean()) / U.std().clamp(min=1e-6)
        else:  # KDS: shaped-reward advantages for a single policy
            with torch.no_grad():
                phi_s = rnd.features(next_frame_buf.reshape(T * B, 84, 84))
                w = kds_weights(phi_s, adv_E_pi.reshape(T * B), int_rew.reshape(T * B),
                                args.kds_tau_a, args.kds_tau_b, args.kds_eps,
                                args.kds_iters, args.kds_delta, args.kds_cap)
                shaped = rew_buf + args.kds_beta * int_rew * w.reshape(T, B)
                U, ret_E_pi = gae(shaped, v_buf, done_buf, nv, args.gamma, args.gae_lambda)
            if args.normalize_advantage:
                U = (U - U.mean()) / U.std().clamp(min=1e-6)

        b_obs = obs_buf.reshape(T * B, 4, 84, 84)
        b_next_frame = next_frame_buf.reshape(T * B, 84, 84)
        b_act = act_buf.reshape(T * B)
        b_logp = logp_buf.reshape(T * B)
        b_U = U.reshape(T * B)
        b_ret_E_pi = ret_E_pi.reshape(T * B)
        b_ret_I = ret_I.reshape(T * B)
        b_ret_E_pr = ret_E_pr.reshape(T * B)
        if args.method == "eipo":
            b_aux = aux_adv.reshape(T * B)

        idxs = np.arange(T * B)
        for epoch in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = idxs[start:start + args.minibatch_size]
                logits_pi, logits_pr, v, v_int, v_prime = agent(b_obs[mb])
                dist_pi = Categorical(logits=logits_pi)
                dist_pr = Categorical(logits=logits_pr)
                logp_pi = dist_pi.log_prob(b_act[mb])
                logp_pr = dist_pr.log_prob(b_act[mb])

                if args.method == "eipo":
                    entropy = dist_pr.entropy().mean()
                    if is_max_step:
                        pi_loss = clipped_pg_loss(logp_pi, b_logp[mb], b_U[mb], args.clip_coef)
                        pr_loss = clipped_pg_loss(logp_pr, b_logp[mb], b_aux[mb], args.clip_coef)
                        v_loss = args.vf_coef * 0.5 * ((v_prime - b_ret_E_pr[mb]) ** 2).mean()
                        if args.use_adapt_alpha and epoch == args.update_epochs - 1:
                            with torch.no_grad():
                                a_loss = clipped_pg_loss(logp_pi, b_logp[mb], b_aux[mb],
                                                         args.clip_coef).item()
                            alpha -= args.alpha_lr * float(
                                np.clip(a_loss, -args.alpha_g_clip, args.alpha_g_clip))
                    else:
                        pr_loss = clipped_pg_loss(logp_pr, b_logp[mb], b_U[mb], args.clip_coef)
                        pi_loss = clipped_pg_loss(logp_pi, b_logp[mb], b_aux[mb], args.clip_coef)
                        v_loss = args.vf_coef * 0.5 * (((v - b_ret_E_pi[mb]) ** 2).mean()
                                                       + ((v_int - b_ret_I[mb]) ** 2).mean())
                    loss = pi_loss + pr_loss + v_loss - args.ent_coef * entropy
                else:
                    entropy = dist_pi.entropy().mean()
                    pi_loss = clipped_pg_loss(logp_pi, b_logp[mb], b_U[mb], args.clip_coef)
                    v_loss = args.vf_coef * 0.5 * (((v - b_ret_E_pi[mb]) ** 2).mean()
                                                   + ((v_int - b_ret_I[mb]) ** 2).mean())
                    loss = pi_loss + v_loss - args.ent_coef * entropy

                loss = loss + rnd.loss(b_next_frame[mb])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()

        if args.method == "eipo":
            if args.minmax_switch == "diff":
                if is_max_step:
                    if switch_metric - last_max_adv <= 0:
                        is_max_step = False
                    last_max_adv = switch_metric
                else:
                    if switch_metric - last_min_adv <= 0:
                        is_max_step = True
                    last_min_adv = switch_metric
            else:
                is_max_step = not is_max_step

        sps = int(global_step / (time.time() - start_time))
        if recent_returns:
            writer.add_scalar("charts/episodic_return", np.mean(recent_returns), global_step)
            writer.add_scalar("charts/episodic_length", np.mean(recent_lengths), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/intrinsic_reward_mean", int_rew.mean().item(), global_step)
        writer.add_scalar("losses/policy_loss", pi_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        if args.method == "eipo":
            writer.add_scalar("eipo/alpha", alpha, global_step)
            writer.add_scalar("eipo/is_max_step", int(is_max_step), global_step)
        else:
            writer.add_scalar("kds/weight_mean", w.mean().item(), global_step)
            writer.add_scalar("kds/weight_max", w.max().item(), global_step)

        if iteration % 10 == 0 or iteration == num_iterations:
            ret = f"{np.mean(recent_returns):9.1f}" if recent_returns else "      n/a"
            extra = (f" alpha {alpha:+.4f} stage {'max' if is_max_step else 'min'}"
                     if args.method == "eipo" else "")
            print(f"iter {iteration:5d}/{num_iterations} step {global_step:>11,d} "
                  f"score {ret}{extra} SPS {sps}", flush=True)

    if args.save_final_model:
        path = os.path.join(args.log_dir, run_name, "model.pt")
        torch.save({"agent": agent.state_dict(), "rnd": rnd.state_dict(),
                    "alpha": alpha, "args": vars(args)}, path)
        print(f"saved model to {path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
