"""
kds_vs_eipo.py — Compare KDS against an EIPO baseline on HalfCheetah-v4 and AntMaze.

Usage:
    python kds_vs_eipo.py --env halfcheetah --method kds  --seed 0 --steps 300000
    python kds_vs_eipo.py --env halfcheetah --method eipo --seed 0 --steps 300000
    python kds_vs_eipo.py --env antmaze     --method kds  --seed 0 --steps 300000
    python kds_vs_eipo.py --env antmaze     --method eipo --seed 0 --steps 300000

    # intrinsic bonus is selectable: --bonus disagreement (default) or --bonus rnd
    python kds_vs_eipo.py --env halfcheetah --method kds --bonus rnd --seed 0 --steps 300000

Requirements:
    pip install torch gymnasium[mujoco] gymnasium-robotics

Design notes
------------
Base algorithm: SAC (twin Q-networks, automatic entropy tuning, 256-unit 2-layer
MLPs), matching the paper's Appendix D.1 setup, shared by both methods for a fair
comparison.

Intrinsic bonus: Disagreement (variance across a 5-model forward-dynamics
ensemble), the paper's primary bonus choice (Appendix D.2), or RND (Burda et al.
2019: prediction error of a trained network against a fixed random target
network, computed on the next state). Both are normalized identically in the
training loop (batch-mean normalization), so the choice of bonus is a clean
ablation axis.

KDS: implements Eq. 14 (target distribution), Eq. 15 (cost matrix on a fixed
random projection of the state), Eq. 16 (Sinkhorn-Knopp iterations), and Eq. 17
(clipped reweighting), applied per SAC minibatch. Default hyperparameters match
Table 3 of the paper (tau_A=tau_b=0.5, eps=0.05, [delta,cap]=[0.2,3.0], P=5,
beta=0.1).

EIPO: adapted from Chen et al. (NeurIPS 2022), "Redeeming Intrinsic Rewards via
Constrained Optimization." The original algorithm is on-policy (PPO), using
alternating min/max stages with PPO-clipped importance-weighted surrogates and
GAE. Here it is adapted to the off-policy SAC setting used throughout this
paper's benchmarks: two full SAC agents are maintained -- pi_E (extrinsic-only)
and pi_mix (trained on the merged reward (1+alpha)*r_extrinsic + r_intrinsic,
matching the paper's J_{E+I}^alpha objective directly below their Eq. 4) -- with
a single scalar alpha updated by dual ascent (Eq. 32/34) using an off-policy
one-step Q-value gap between the two policies' extrinsic value estimates, in
place of the on-policy Monte Carlo return gap used in the original paper. This
is a faithful adaptation of the *mechanism*, not a re-implementation of the
original PPO-based code -- report it as such if used in a rebuttal or paper.
(A PPO-based re-implementation of the original EIPO is in
continuous/eipo_ppo_continuous.py; a PPO-based KDS is in
continuous/kds_ppo_continuous.py.)
"""

import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

try:
    import gymnasium_robotics  # noqa: F401  registers AntMaze-* environments
except ImportError:
    gymnasium_robotics = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------

ANTMAZE_ID_CANDIDATES = [
    "AntMaze_UMaze-v4",
    "AntMaze_UMaze-v5",
    "AntMaze_UMaze-v3",
    "antmaze-umaze-v2",
]


def make_env(env_key, seed):
    if env_key == "halfcheetah":
        env = gym.make("HalfCheetah-v4")
    elif env_key == "antmaze":
        if gymnasium_robotics is None:
            raise ImportError(
                "AntMaze requires `pip install gymnasium-robotics`."
            )
        last_err = None
        env = None
        for env_id in ANTMAZE_ID_CANDIDATES:
            try:
                env = gym.make(env_id, reward_type="sparse")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if env is None:
            raise RuntimeError(
                f"Could not create any AntMaze env from {ANTMAZE_ID_CANDIDATES}. "
                f"Check your gymnasium-robotics version and adjust the id list. "
                f"Last error: {last_err}"
            )
    else:
        raise ValueError(env_key)
    env.reset(seed=seed)
    return env


def flatten_obs(obs):
    """Handles both plain Box observations (HalfCheetah) and Dict observations
    with 'observation'/'achieved_goal'/'desired_goal' keys (AntMaze)."""
    if isinstance(obs, dict):
        parts = [obs["observation"]]
        if "desired_goal" in obs:
            parts.append(obs["desired_goal"])
        return np.concatenate(parts).astype(np.float32)
    return obs.astype(np.float32)


def get_success(env_key, info, terminated):
    if env_key == "antmaze":
        return float(info.get("success", terminated))
    return None


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MIN, LOG_STD_MAX = -20, 2


class SquashedGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        pi = mu if deterministic else dist.rsample()
        if with_logprob:
            logp = dist.log_prob(pi).sum(-1)
            logp -= (2 * (np.log(2) - pi - F.softplus(-2 * pi))).sum(-1)
        else:
            logp = None
        pi = torch.tanh(pi) * self.act_limit
        return pi, logp


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class DynamicsEnsemble(nn.Module):
    """Forward-dynamics ensemble; bonus = prediction disagreement (variance)."""

    def __init__(self, obs_dim, act_dim, n_models=5, hidden=256):
        super().__init__()
        self.models = nn.ModuleList(
            [mlp([obs_dim + act_dim, hidden, hidden, obs_dim]) for _ in range(n_models)]
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return torch.stack([m(x) for m in self.models], dim=0)  # (P, N, obs_dim)

    def bonus(self, obs, act):
        with torch.no_grad():
            preds = self.forward(obs, act)
            return preds.var(dim=0).mean(dim=-1)

    def train_step(self, obs, act, next_obs, opt):
        preds = self.forward(obs, act)
        target = next_obs.unsqueeze(0).expand_as(preds)
        loss = ((preds - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()


class RNDNetwork(nn.Module):
    """Random Network Distillation (Burda et al. 2019): bonus = squared error of
    a trained predictor against a fixed, randomly initialized target network,
    computed on the NEXT state (novelty of where the transition lands). The
    predictor is one layer deeper than the target, as in the original paper.
    Interface mirrors DynamicsEnsemble so the two are drop-in interchangeable."""

    def __init__(self, obs_dim, feature_dim=256, hidden=256):
        super().__init__()
        self.predictor = mlp([obs_dim, hidden, hidden, hidden, feature_dim])
        self.target = mlp([obs_dim, hidden, hidden, feature_dim])
        for p in self.target.parameters():
            p.requires_grad = False

    def _error(self, next_obs):
        phi = self.target(next_obs)
        pred = self.predictor(next_obs)
        return ((pred - phi) ** 2).mean(dim=-1)

    def bonus(self, next_obs):
        with torch.no_grad():
            return self._error(next_obs)

    def train_step(self, next_obs, opt):
        loss = self._error(next_obs).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()


class IntrinsicBonus:
    """Uniform wrapper over DynamicsEnsemble / RNDNetwork: bonus(o, a, no) and
    train(o, a, no). Keeps the two training loops bonus-agnostic."""

    def __init__(self, bonus_type, obs_dim, act_dim, lr=1e-3):
        self.bonus_type = bonus_type
        if bonus_type == "disagreement":
            self.module = DynamicsEnsemble(obs_dim, act_dim).to(DEVICE)
        elif bonus_type == "rnd":
            self.module = RNDNetwork(obs_dim).to(DEVICE)
        else:
            raise ValueError(bonus_type)
        self.opt = torch.optim.Adam(self.module.parameters(), lr=lr)

    def bonus(self, obs, act, next_obs):
        if self.bonus_type == "disagreement":
            return self.module.bonus(obs, act)
        return self.module.bonus(next_obs)

    def train(self, obs, act, next_obs):
        if self.bonus_type == "disagreement":
            return self.module.train_step(obs, act, next_obs, self.opt)
        return self.module.train_step(next_obs, self.opt)


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6)):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=DEVICE),
            torch.as_tensor(self.act[idx], device=DEVICE),
            torch.as_tensor(self.rew[idx], device=DEVICE),
            torch.as_tensor(self.next_obs[idx], device=DEVICE),
            torch.as_tensor(self.done[idx], device=DEVICE),
        )


# -----------------------------------------------------------------------------
# KDS reward shaping (Eqs. 14-17)
# -----------------------------------------------------------------------------

class RandomProjection(nn.Module):
    """Fixed random projection phi(s) used for the KDS cost matrix (Eq. 15)."""

    def __init__(self, obs_dim, proj_dim=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(obs_dim, proj_dim, generator=g) / np.sqrt(obs_dim)
        self.register_buffer("W", W)

    def forward(self, obs):
        return obs @ self.W


def kds_weights(phi_obs, adv, bonus, tau_A=0.5, tau_b=0.5, eps=0.05, iters=50,
                 delta=0.2, cap=3.0):
    """Batch Sinkhorn-Knopp reward shaping, matching Eqs. 14, 16, 17."""
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


# -----------------------------------------------------------------------------
# SAC agent (shared by KDS and EIPO)
# -----------------------------------------------------------------------------

class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, hidden=256, gamma=0.99, tau=5e-3,
                 pi_lr=3e-4, q_lr=3e-4, alpha_lr=3e-4, target_entropy=None):
        self.pi = SquashedGaussianPolicy(obs_dim, act_dim, act_limit, hidden).to(DEVICE)
        self.q1 = QNetwork(obs_dim, act_dim, hidden).to(DEVICE)
        self.q2 = QNetwork(obs_dim, act_dim, hidden).to(DEVICE)
        self.q1_targ = copy.deepcopy(self.q1)
        self.q2_targ = copy.deepcopy(self.q2)
        for p in self.q1_targ.parameters():
            p.requires_grad = False
        for p in self.q2_targ.parameters():
            p.requires_grad = False

        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=pi_lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=q_lr
        )

        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim

        self.gamma, self.tau = gamma, tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a, _ = self.pi(obs_t, deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, obs, act, rew, next_obs, done):
        with torch.no_grad():
            next_a, next_logp = self.pi(next_obs)
            q1_t = self.q1_targ(next_obs, next_a)
            q2_t = self.q2_targ(next_obs, next_a)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            backup = rew + self.gamma * (1 - done) * q_t

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        pi_a, logp = self.pi(obs)
        q1_pi = self.q1(obs, pi_a)
        q2_pi = self.q2(obs, pi_a)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

        return q1.mean().item()


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate(env_key, agent, seed, n_episodes=5):
    env = make_env(env_key, seed)
    rets, succs = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = flatten_obs(obs)
        done, ep_ret, ep_succ = False, 0.0, 0.0
        while not done:
            a = agent.act(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            obs = flatten_obs(obs)
            ep_ret += r
            s = get_success(env_key, info, term)
            if s is not None:
                ep_succ = max(ep_succ, s)
            done = term or trunc
        rets.append(ep_ret)
        succs.append(ep_succ)
    env.close()
    mean_succ = float(np.mean(succs)) if env_key == "antmaze" else None
    return float(np.mean(rets)), mean_succ


# -----------------------------------------------------------------------------
# Method: KDS
# -----------------------------------------------------------------------------

def train_kds(env_key, seed, total_steps, batch_size=256, start_steps=10000,
              beta=0.1, tau_A=0.5, tau_b=0.5, eps=0.05, delta=0.2, cap=3.0,
              eval_every=5000, log_path=None, bonus_type="disagreement"):
    env = make_env(env_key, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = flatten_obs(env.observation_space.sample() if not isinstance(
        env.observation_space, gym.spaces.Dict) else {
        k: v.sample() for k, v in env.observation_space.spaces.items()
    }).shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_limit)
    intrinsic = IntrinsicBonus(bonus_type, obs_dim, act_dim)
    phi = RandomProjection(obs_dim, seed=seed).to(DEVICE)
    buf = ReplayBuffer(obs_dim, act_dim)

    obs, _ = env.reset(seed=seed)
    obs = flatten_obs(obs)
    log = {"steps": [], "eval_return": [], "eval_success": [], "bonus": bonus_type}

    for t in range(total_steps):
        a = env.action_space.sample() if t < start_steps else agent.act(obs)
        next_obs, r, term, trunc, info = env.step(a)
        next_obs = flatten_obs(next_obs)
        buf.store(obs, a, r, next_obs, float(term))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
            obs = flatten_obs(obs)

        if t >= start_steps:
            o, a_, r_, no, d = buf.sample(batch_size)
            intrinsic.train(o, a_, no)
            with torch.no_grad():
                bonus = intrinsic.bonus(o, a_, no)
                bonus = bonus / (bonus.mean() + 1e-8)
                next_a, _ = agent.pi(no)
                v_s = torch.min(agent.q1(o, a_), agent.q2(o, a_))
                v_ns = torch.min(agent.q1_targ(no, next_a), agent.q2_targ(no, next_a))
                adv = r_ + agent.gamma * (1 - d) * v_ns - v_s
                phi_o = phi(o)
            w = kds_weights(phi_o, adv, bonus, tau_A, tau_b, eps, 50, delta, cap)
            shaped_r = r_ + beta * bonus * w
            agent.update(o, a_, shaped_r, no, d)

        if (t + 1) % eval_every == 0:
            ret, succ = evaluate(env_key, agent, seed + 1000)
            log["steps"].append(t + 1)
            log["eval_return"].append(ret)
            log["eval_success"].append(succ)
            print(f"[KDS][{env_key}][{bonus_type}] step={t + 1} "
                  f"eval_return={ret:.2f} eval_success={succ}", flush=True)

    if log_path:
        with open(log_path, "w") as f:
            json.dump(log, f)
    return log


# -----------------------------------------------------------------------------
# Method: EIPO (off-policy SAC adaptation)
# -----------------------------------------------------------------------------

def train_eipo(env_key, seed, total_steps, batch_size=256, start_steps=10000,
               alpha_init=0.5, alpha_lr=1e-3, eval_every=5000, log_path=None,
               bonus_type="disagreement"):
    env = make_env(env_key, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = flatten_obs(env.observation_space.sample() if not isinstance(
        env.observation_space, gym.spaces.Dict) else {
        k: v.sample() for k, v in env.observation_space.spaces.items()
    }).shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    pi_E = SACAgent(obs_dim, act_dim, act_limit)
    pi_mix = SACAgent(obs_dim, act_dim, act_limit)
    intrinsic = IntrinsicBonus(bonus_type, obs_dim, act_dim)
    buf = ReplayBuffer(obs_dim, act_dim)

    alpha = alpha_init
    obs, _ = env.reset(seed=seed)
    obs = flatten_obs(obs)
    log = {"steps": [], "eval_return": [], "eval_success": [], "alpha": [],
           "bonus": bonus_type}

    for t in range(total_steps):
        # environment collected by pi_mix (the exploring policy), as in KDS's loop
        a = env.action_space.sample() if t < start_steps else pi_mix.act(obs)
        next_obs, r, term, trunc, info = env.step(a)
        next_obs = flatten_obs(next_obs)
        buf.store(obs, a, r, next_obs, float(term))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
            obs = flatten_obs(obs)

        if t >= start_steps:
            o, a_, r_, no, d = buf.sample(batch_size)
            intrinsic.train(o, a_, no)
            with torch.no_grad():
                bonus = intrinsic.bonus(o, a_, no)
                bonus = bonus / (bonus.mean() + 1e-8)

            # pi_E: pure extrinsic-only SAC update
            pi_E.update(o, a_, r_, no, d)

            # pi_mix: merged objective (1+alpha)*r_E + r_I, matching the paper's
            # J_{E+I}^alpha objective (definition immediately after their Eq. 4)
            mixed_r = (1 + alpha) * r_ + bonus
            pi_mix.update(o, a_, mixed_r, no, d)

            # dual ascent on alpha (Eq. 32/34): alpha <- alpha - beta*(J_E(pi_mix)-J_E(pi_E)),
            # approximated off-policy via a one-step Q-gap under pi_E's own extrinsic
            # critic, evaluated at both policies' actions on the current batch of states
            with torch.no_grad():
                a_E_batch, _ = pi_E.pi(o)
                a_mix_batch, _ = pi_mix.pi(o)
                J_E = torch.min(pi_E.q1(o, a_E_batch), pi_E.q2(o, a_E_batch)).mean().item()
                J_mix_extrinsic = torch.min(
                    pi_E.q1(o, a_mix_batch), pi_E.q2(o, a_mix_batch)
                ).mean().item()
            alpha = float(np.clip(alpha - alpha_lr * (J_mix_extrinsic - J_E), 1e-3, 10.0))

        if (t + 1) % eval_every == 0:
            ret, succ = evaluate(env_key, pi_mix, seed + 1000)
            log["steps"].append(t + 1)
            log["eval_return"].append(ret)
            log["eval_success"].append(succ)
            log["alpha"].append(alpha)
            print(
                f"[EIPO][{env_key}][{bonus_type}] step={t + 1} eval_return={ret:.2f} "
                f"eval_success={succ} alpha={alpha:.3f}", flush=True
            )

    if log_path:
        with open(log_path, "w") as f:
            json.dump(log, f)
    return log


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["halfcheetah", "antmaze"], required=True)
    p.add_argument("--method", choices=["kds", "eipo"], required=True)
    p.add_argument("--bonus", choices=["disagreement", "rnd"], default="disagreement")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--start_steps", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or f"{args.method}_{args.bonus}_{args.env}_seed{args.seed}.json"

    if args.method == "kds":
        train_kds(args.env, args.seed, args.steps, start_steps=args.start_steps,
                  eval_every=args.eval_every, log_path=out, bonus_type=args.bonus)
    else:
        train_eipo(args.env, args.seed, args.steps, start_steps=args.start_steps,
                   eval_every=args.eval_every, log_path=out, bonus_type=args.bonus)
