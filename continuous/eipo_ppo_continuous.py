"""EIPO (Extrinsic-Intrinsic Policy Optimization, NeurIPS'22) with RND intrinsic
rewards for continuous-control environments, PPO base. Single-file, CleanRL style.

This is a faithful port of the min-max logic in this repo's rlpyt implementation
(rlpyt/algos/pg/ppo.py: _optimize_agent_minmax / loss_max_step / loss_min_step and
rlpyt/algos/pg/base.py: process_returns_{max,min}_step), with the image-based
pieces (conv RND, Atari LSTM policy) replaced by MLPs over state vectors:

  - Two Gaussian policies: pi (max-player, trained on extrinsic+intrinsic) and
    pi' (min-player, extrinsic only). One critic trunk with three value heads:
    V_E^pi, V_I^pi, V_E^pi'.
  - Max step (rollouts from pi'):  train pi on  U+ = r_E + r_I + alpha * A_E^pi'
                                   train pi' on A_E^pi'
  - Min step (rollouts from pi):   train pi' on U- = (alpha-1) r_E - r_I + (A_E^pi + A_I^pi)
                                   train pi  on A_E^pi + A_I^pi
  - alpha is adapted from the clipped surrogate of pi's extrinsic advantage under
    pi' rollouts (last epoch of each max step), gradient clipped to +-alpha_g_clip.
  - Stage switching: 'diff' rule -- flip when the current stage's mean U advantage
    stops improving. NOTE: the rlpyt code evaluated this rule on advantages already
    normalized to zero mean (so it effectively flipped ~every iteration); here the
    rule uses pre-normalization means, which is the semantics described in the
    paper. Use --minmax-switch none to flip unconditionally every iteration.

Environments: any gymnasium Box-observation env (e.g. HalfCheetah-v4). Dict
goal-observation envs (AntMaze from gymnasium-robotics) are flattened to
[observation, desired_goal]. Requires gymnasium>=1.1 (SAME_STEP autoreset).

Example:
  python eipo_ppo_continuous.py --env-id HalfCheetah-v4 --total-timesteps 3000000 --seed 1
  python eipo_ppo_continuous.py --env-id AntMaze_UMaze-v5 --total-timesteps 10000000 \
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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default="eipo_ppo")
    p.add_argument("--env-id", type=str, default="HalfCheetah-v4")
    p.add_argument("--total-timesteps", type=int, default=3_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-dir", type=str, default="results_continuous")
    p.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--track", action=argparse.BooleanOptionalAction, default=False,
                   help="log to Weights & Biases in addition to TensorBoard")
    p.add_argument("--wandb-project", type=str, default="eipo-continuous")

    # PPO
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gamma-int", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=1.0, help="value loss = vf_coef * 0.5 * MSE")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--normalize-ext-reward", action=argparse.BooleanOptionalAction, default=True,
                   help="divide extrinsic rewards by a running std of discounted returns. "
                        "Disable for sparse-reward envs such as AntMaze.")

    # RND
    p.add_argument("--rnd-beta", type=float, default=1.0, help="intrinsic reward scale (prediction_beta)")
    p.add_argument("--rnd-update-proportion", type=float, default=0.25,
                   help="fraction of samples used for the predictor loss")
    p.add_argument("--rnd-feature-size", type=int, default=256)

    # EIPO min-max
    p.add_argument("--minmax-alpha", type=float, default=0.5)
    p.add_argument("--use-adapt-alpha", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alpha-lr", type=float, default=0.005)
    p.add_argument("--alpha-g-clip", type=float, default=0.05, help="clip on the alpha gradient")
    p.add_argument("--minmax-switch", type=str, default="diff", choices=["diff", "none"])

    args = p.parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


# --------------------------------------------------------------------------- #
# Running statistics (same semantics as rlpyt/utils/averages.py)
# --------------------------------------------------------------------------- #
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
    """Discounted accumulation of rewards, reset where episodes end."""

    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews, not_done):
        if self.rewems is None:
            self.rewems = rews.copy()
        else:
            self.rewems = self.rewems * self.gamma * not_done + rews
        return self.rewems.copy()


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
class FlattenGoalObservation(gym.ObservationWrapper):
    """Dict goal obs -> concat([observation, desired_goal]) as float32."""

    def __init__(self, env):
        super().__init__(env)
        spaces = env.observation_space.spaces
        dim = int(np.prod(spaces["observation"].shape)) + int(np.prod(spaces["desired_goal"].shape))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs["observation"].ravel(), obs["desired_goal"].ravel()]).astype(np.float32)


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenGoalObservation(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.logstd = nn.Parameter(torch.zeros(1, act_dim))

    def dist(self, x):
        mean = self.mean(x)
        return Normal(mean, torch.exp(self.logstd.expand_as(mean)))


class Critic(nn.Module):
    """Shared trunk, three heads: V_E^pi, V_I^pi, V_E^pi' (mirrors the rlpyt model)."""

    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.v_ext = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.v_int = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.v_ext_prime = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x):
        h = self.trunk(x)
        return self.v_ext(h).squeeze(-1), self.v_int(h).squeeze(-1), self.v_ext_prime(h).squeeze(-1)


class RND(nn.Module):
    """MLP RND over raw state vectors; normalizes obs internally (clip +-5) and
    intrinsic rewards by a running std of their discounted accumulation."""

    def __init__(self, obs_dim, feature_size, beta, gamma_int, update_proportion, device):
        super().__init__()
        self.beta = beta
        self.update_proportion = update_proportion
        self.device = device
        self.feature_size = feature_size
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.rew_rms = RunningMeanStd()
        self.rew_rff = RewardForwardFilter(gamma_int)

        self.predictor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.LeakyReLU(),
            layer_init(nn.Linear(256, 256)), nn.LeakyReLU(),
            layer_init(nn.Linear(256, feature_size)), nn.ReLU(),
            layer_init(nn.Linear(feature_size, feature_size)), nn.ReLU(),
            layer_init(nn.Linear(feature_size, feature_size)),
        )
        self.target = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.LeakyReLU(),
            layer_init(nn.Linear(256, 256)), nn.LeakyReLU(),
            layer_init(nn.Linear(256, feature_size)),
        )
        for prm in self.target.parameters():
            prm.requires_grad = False

    def _normalize(self, obs):
        mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=obs.device)
        std = torch.as_tensor(np.sqrt(self.obs_rms.var), dtype=torch.float32, device=obs.device)
        return torch.clamp((obs - mean) / (std + 1e-10), -5.0, 5.0)

    @torch.no_grad()
    def compute_bonus(self, next_obs, dones):
        """next_obs: [T, B, D] raw observations; dones: [T, B]. Returns [T, B]."""
        T, B = next_obs.shape[:2]
        not_done = (1.0 - dones).cpu().numpy()
        norm = self._normalize(next_obs)
        phi = self.target(norm)
        pred = self.predictor(norm)
        bonus = nn.functional.mse_loss(pred, phi, reduction="none").sum(-1) / self.feature_size

        # update obs stats on non-terminal steps
        obs_np = next_obs.cpu().numpy().reshape(T * B, -1)
        valid = not_done.reshape(T * B) > 0
        if valid.any():
            self.obs_rms.update(obs_np[valid])

        # normalize by running std of discounted accumulated bonus
        bonus_np = bonus.cpu().numpy()
        rff = np.array([self.rew_rff.update(bonus_np[t], not_done=not_done[t]) for t in range(T)])
        self.rew_rms.update_from_moments(rff.mean(), rff.var(), not_done.sum())
        bonus = bonus / float(np.sqrt(self.rew_rms.var) + 1e-10)
        bonus = bonus * torch.as_tensor(not_done, dtype=torch.float32, device=bonus.device)
        return self.beta * bonus

    def loss(self, next_obs):
        """Predictor loss on a minibatch [N, D], on a random subset of samples."""
        norm = self._normalize(next_obs)
        with torch.no_grad():
            phi = self.target(norm)
        pred = self.predictor(norm)
        err = nn.functional.mse_loss(pred, phi, reduction="none").sum(-1) / self.feature_size
        mask = (torch.rand(err.shape, device=err.device) < self.update_proportion).float()
        return (err * mask).sum() / mask.sum().clamp(min=1.0)


# --------------------------------------------------------------------------- #
# PPO pieces
# --------------------------------------------------------------------------- #
def clipped_pg_loss(new_logp, old_logp, advantage, clip_coef):
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage
    return -torch.min(surr1, surr2).mean()


def gae(rewards, values, dones, next_value, gamma, lam):
    """rewards/values/dones: [T, B]; dones[t] = episode ended at step t."""
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


# --------------------------------------------------------------------------- #
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
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args), sync_tensorboard=True)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id) for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    actor_pi = Actor(obs_dim, act_dim).to(device)        # max-player, extrinsic + intrinsic
    actor_pi_prime = Actor(obs_dim, act_dim).to(device)  # min-player, extrinsic only
    critic = Critic(obs_dim).to(device)
    rnd = RND(obs_dim, args.rnd_feature_size, args.rnd_beta, args.gamma_int,
              args.rnd_update_proportion, device).to(device)
    params = (list(actor_pi.parameters()) + list(actor_pi_prime.parameters())
              + list(critic.parameters()) + list(rnd.predictor.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ext_rff = RewardForwardFilter(args.gamma)
    ext_rms = RunningMeanStd()

    def norm_obs(o):
        o = np.clip((o - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8), -10.0, 10.0)
        return torch.as_tensor(o, dtype=torch.float32, device=device)

    # EIPO state
    alpha = args.minmax_alpha
    is_max_step = True  # rlpyt AtariLstmAgent initializes is_max_step=True
    last_max_adv, last_min_adv = 0.0, 0.0

    # rollout storage
    T, B = args.num_steps, args.num_envs
    obs_buf = torch.zeros((T, B, obs_dim), device=device)          # normalized (policy input)
    next_obs_buf = torch.zeros((T, B, obs_dim), device=device)     # raw (RND input)
    act_buf = torch.zeros((T, B, act_dim), device=device)
    logp_buf = torch.zeros((T, B), device=device)                  # behavior-policy logprob
    rew_buf = torch.zeros((T, B), device=device)                   # raw extrinsic
    done_buf = torch.zeros((T, B), device=device)
    v_buf = torch.zeros((T, B), device=device)
    v_int_buf = torch.zeros((T, B), device=device)
    v_prime_buf = torch.zeros((T, B), device=device)

    # episode bookkeeping (manual: robust across gymnasium versions)
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

        behavior = actor_pi_prime if is_max_step else actor_pi

        for t in range(T):
            global_step += B
            n_obs = norm_obs(raw_next_obs)
            obs_buf[t] = n_obs
            with torch.no_grad():
                dist = behavior.dist(n_obs)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action).sum(-1)
                v, v_int, v_prime = critic(n_obs)
            v_buf[t], v_int_buf[t], v_prime_buf[t] = v, v_int, v_prime
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

        # ---- rewards ---------------------------------------------------- #
        int_rew = rnd.compute_bonus(next_obs_buf, done_buf)  # [T, B], done-masked, normalized

        if args.normalize_ext_reward:
            rew_np = rew_buf.cpu().numpy()
            nd_np = (1.0 - done_buf).cpu().numpy()
            rff = np.array([ext_rff.update(rew_np[t], not_done=nd_np[t]) for t in range(T)])
            ext_rms.update_from_moments(rff.mean(), rff.var(), len(rff))
            ext_rew = rew_buf / float(np.sqrt(ext_rms.var) + 1e-10)
        else:
            ext_rew = rew_buf

        # ---- advantages (rlpyt process_returns_{max,min}_step) ----------- #
        with torch.no_grad():
            nv, nv_int, nv_prime = critic(norm_obs(raw_next_obs))
        adv_E_pi, ret_E_pi = gae(ext_rew, v_buf, done_buf, nv, args.gamma, args.gae_lambda)
        adv_I, ret_I = gae(int_rew, v_int_buf, done_buf, nv_int, args.gamma_int, args.gae_lambda)
        adv_E_pr, ret_E_pr = gae(ext_rew, v_prime_buf, done_buf, nv_prime, args.gamma, args.gae_lambda)

        if is_max_step:
            U = ext_rew + int_rew + alpha * adv_E_pr          # U+
            aux_adv = adv_E_pr                                 # advantage_pi_prime
        else:
            U = (alpha - 1.0) * ext_rew - int_rew + (adv_E_pi + adv_I)  # U-
            aux_adv = adv_E_pi + adv_I                         # advantage_pi

        switch_metric = U.mean().item()  # pre-normalization (see module docstring)
        if args.normalize_advantage:
            U = (U - U.mean()) / U.std().clamp(min=1e-6)

        # ---- optimize ----------------------------------------------------#
        b_obs = obs_buf.reshape(T * B, obs_dim)
        b_next_obs = next_obs_buf.reshape(T * B, obs_dim)
        b_act = act_buf.reshape(T * B, act_dim)
        b_logp = logp_buf.reshape(T * B)
        b_U = U.reshape(T * B)
        b_aux = aux_adv.reshape(T * B)
        b_ret_E_pi = ret_E_pi.reshape(T * B)
        b_ret_I = ret_I.reshape(T * B)
        b_ret_E_pr = ret_E_pr.reshape(T * B)

        idxs = np.arange(T * B)
        alpha_grads = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = idxs[start:start + args.minibatch_size]
                dist_pi = actor_pi.dist(b_obs[mb])
                dist_pr = actor_pi_prime.dist(b_obs[mb])
                logp_pi = dist_pi.log_prob(b_act[mb]).sum(-1)
                logp_pr = dist_pr.log_prob(b_act[mb]).sum(-1)
                v, v_int, v_prime = critic(b_obs[mb])
                # rlpyt uses the entropy of pi' in both stages
                entropy = dist_pr.entropy().sum(-1).mean()

                if is_max_step:
                    pi_loss = clipped_pg_loss(logp_pi, b_logp[mb], b_U[mb], args.clip_coef)
                    pr_loss = clipped_pg_loss(logp_pr, b_logp[mb], b_aux[mb], args.clip_coef)
                    v_loss = args.vf_coef * 0.5 * ((v_prime - b_ret_E_pr[mb]) ** 2).mean()
                    if args.use_adapt_alpha and epoch == args.update_epochs - 1:
                        with torch.no_grad():
                            a_loss = clipped_pg_loss(logp_pi, b_logp[mb], b_aux[mb], args.clip_coef).item()
                        alpha_grads.append(a_loss)
                        alpha -= args.alpha_lr * float(np.clip(a_loss, -args.alpha_g_clip, args.alpha_g_clip))
                else:
                    pr_loss = clipped_pg_loss(logp_pr, b_logp[mb], b_U[mb], args.clip_coef)
                    pi_loss = clipped_pg_loss(logp_pi, b_logp[mb], b_aux[mb], args.clip_coef)
                    v_loss = args.vf_coef * 0.5 * (((v - b_ret_E_pi[mb]) ** 2).mean()
                                                   + ((v_int - b_ret_I[mb]) ** 2).mean())

                rnd_loss = rnd.loss(b_next_obs[mb])
                loss = pi_loss + pr_loss + v_loss - args.ent_coef * entropy + rnd_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()

        # ---- stage switching ('diff' rule) ------------------------------- #
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

        # ---- logging ------------------------------------------------------#
        sps = int(global_step / (time.time() - start_time))
        if recent_returns:
            writer.add_scalar("charts/episodic_return", np.mean(recent_returns), global_step)
            writer.add_scalar("charts/episodic_length", np.mean(recent_lengths), global_step)
            writer.add_scalar("charts/success_rate", np.mean(recent_success), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("eipo/alpha", alpha, global_step)
        writer.add_scalar("eipo/is_max_step", int(is_max_step), global_step)
        writer.add_scalar("eipo/switch_metric", switch_metric, global_step)
        writer.add_scalar("losses/policy_loss", pi_loss.item(), global_step)
        writer.add_scalar("losses/pi_prime_loss", pr_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/rnd_loss", rnd_loss.item(), global_step)
        writer.add_scalar("charts/intrinsic_reward_mean", int_rew.mean().item(), global_step)
        if alpha_grads:
            writer.add_scalar("eipo/alpha_grad", float(np.mean(alpha_grads)), global_step)

        if iteration % 10 == 0 or iteration == num_iterations:
            ret = f"{np.mean(recent_returns):9.2f}" if recent_returns else "      n/a"
            print(f"iter {iteration:5d}/{num_iterations} step {global_step:>10,d} "
                  f"return {ret} alpha {alpha:+.4f} "
                  f"stage {'max' if is_max_step else 'min'} SPS {sps}", flush=True)

    if args.save_final_model:
        path = os.path.join(args.log_dir, run_name, "model.pt")
        torch.save({
            "actor_pi": actor_pi.state_dict(),
            "actor_pi_prime": actor_pi_prime.state_dict(),
            "critic": critic.state_dict(),
            "rnd_predictor": rnd.predictor.state_dict(),
            "rnd_target": rnd.target.state_dict(),
            "obs_rms": {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
            "alpha": alpha,
            "args": vars(args),
        }, path)
        print(f"saved model to {path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
