#!/usr/bin/env python3
"""
Advanced PPO bootstrap training for SnakePixels.

Implements:
- Reward shaping (food/death/step/survival)
- Curriculum schedule (max_steps ramp)
- VecNormalize (obs/reward normalization)
- PPO stability defaults + configurable hyperparams
- Simple intrinsic exploration bonus (count-hash novelty)
- Fixed-seed deterministic evaluation + best-checkpoint selection
- Metrics CSV/JSON logging
- Discord tqdm progress via tqdm.contrib.discord

Example:
  PYTHONPATH=. python scripts/train_rl_bootstrap.py \
    --env SnakePixels-v0 \
    --timesteps 2000000 \
    --n-envs 8 \
    --discord-channel-id 1471943945540866139 \
    --out-dir results/rl_bootstrap_snake_2m_adv \
    --export-genome results/rl_bootstrap_snake_2m_adv/bootstrap_genome.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from src.snake_env import ensure_snake_registered
from src.genome import Policy, GaussianMutator, Genome


def _load_discord_token_from_openclaw() -> str | None:
    try:
        cfg_path = os.path.expanduser('~/.openclaw/openclaw.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get('channels', {}).get('discord', {}).get('token')
    except Exception:
        return None


class SnakeRewardWrapper(gym.Wrapper):
    """Reward shaping + optional novelty bonus + curriculum hook."""

    def __init__(
        self,
        env,
        reward_food: float = 10.0,
        reward_death: float = -10.0,
        reward_step: float = -0.001,
        reward_survival: float = 0.0,
        intrinsic_bonus: float = 0.0,
        obs_hash_bins: int = 4096,
    ):
        super().__init__(env)
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_survival = reward_survival
        self.intrinsic_bonus = intrinsic_bonus
        self.obs_hash_bins = obs_hash_bins
        self._visit_counts = defaultdict(int)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def _obs_hash(self, obs: np.ndarray) -> int:
        # Downsample by stride to keep hashing cheap
        x = np.asarray(obs, dtype=np.uint8).ravel()[::8]
        return int(np.sum(x.astype(np.int64) * 1315423911) % self.obs_hash_bins)

    def set_max_steps(self, v: int):
        # Reach underlying SnakePixelsEnv regardless of wrapper depth
        base = self.env
        while hasattr(base, 'env'):
            if hasattr(base, 'max_steps'):
                break
            base = base.env
        if hasattr(base, 'max_steps'):
            base.max_steps = int(v)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Base reward decomposition (SnakePixels currently: +1 food, -1 death, -0.01 step)
        shaped = 0.0
        if reward >= 0.9:
            shaped += self.reward_food
        elif terminated and reward <= -0.9:
            shaped += self.reward_death
        else:
            shaped += self.reward_step
            shaped += self.reward_survival

        # Intrinsic novelty bonus (simple count-based hash)
        if self.intrinsic_bonus > 0.0:
            h = self._obs_hash(obs)
            self._visit_counts[h] += 1
            shaped += self.intrinsic_bonus / np.sqrt(self._visit_counts[h])

        return obs, float(shaped), terminated, truncated, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train PPO baseline and export bootstrap genome')
    p.add_argument('--env', default='SnakePixels-v0')
    p.add_argument('--timesteps', type=int, default=1_000_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--eval-freq', type=int, default=50_000)
    p.add_argument('--save-freq', type=int, default=100_000)
    p.add_argument('--out-dir', type=str, default='results/rl_bootstrap')
    p.add_argument('--export-genome', type=str, default='')
    p.add_argument('--device', type=str, default='auto', help='cpu|cuda|auto')

    # Reward shaping
    p.add_argument('--reward-food', type=float, default=10.0)
    p.add_argument('--reward-death', type=float, default=-10.0)
    p.add_argument('--reward-step', type=float, default=-0.001)
    p.add_argument('--reward-survival', type=float, default=0.0)

    # Exploration bonus (lightweight intrinsic reward)
    p.add_argument('--intrinsic-bonus', type=float, default=0.03)

    # Curriculum: max_steps schedule (start -> mid -> final)
    p.add_argument('--curriculum-start-max-steps', type=int, default=180)
    p.add_argument('--curriculum-mid-max-steps', type=int, default=240)
    p.add_argument('--curriculum-final-max-steps', type=int, default=300)
    p.add_argument('--curriculum-mid-frac', type=float, default=0.35)
    p.add_argument('--curriculum-final-frac', type=float, default=0.70)

    # PPO stability params
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-range', type=float, default=0.15)
    p.add_argument('--ent-coef', type=float, default=0.005)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--n-steps', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--n-epochs', type=int, default=10)
    p.add_argument('--max-grad-norm', type=float, default=0.5)

    # Normalization
    p.add_argument('--no-norm-obs', action='store_true')
    p.add_argument('--no-norm-reward', action='store_true')

    # Discord tqdm
    p.add_argument('--discord-channel-id', type=int, default=1471943945540866139)
    p.add_argument('--no-discord-tqdm', action='store_true')

    return p.parse_args()


def evaluate_policy_fixed(model, env_id: str, episodes: int, seed: int, max_steps: int = 300) -> float:
    env = gym.make(env_id)
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def export_ppo_to_genome(model, env_id: str, out_path: Path) -> None:
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError('Exporter currently supports only Discrete action spaces')
    act_dim = int(env.action_space.n)
    env.close()

    policy = Policy(obs_dim, act_dim, hidden=64)

    sb3_pi_layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    if len(sb3_pi_layers) != 2:
        raise RuntimeError(f'Expected 2 actor linear layers, got {len(sb3_pi_layers)}')
    sb3_out = model.policy.action_net
    if not isinstance(sb3_out, nn.Linear):
        raise RuntimeError('Expected linear action_net')

    tgt_l1, tgt_l2, tgt_l3 = policy.net[0], policy.net[2], policy.net[4]
    with torch.no_grad():
        tgt_l1.weight.copy_(sb3_pi_layers[0].weight)
        tgt_l1.bias.copy_(sb3_pi_layers[0].bias)
        tgt_l2.weight.copy_(sb3_pi_layers[1].weight)
        tgt_l2.bias.copy_(sb3_pi_layers[1].bias)
        tgt_l3.weight.copy_(sb3_out.weight)
        tgt_l3.bias.copy_(sb3_out.bias)

    mutator = GaussianMutator()
    genome = Genome(policy=policy, mutator=mutator, mutator_type='gaussian', compat_net=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genome.save(str(out_path))


class EvalBestCallback:
    def __init__(self, args, model, vec_env, out_dir: Path):
        self.args = args
        self.model = model
        self.vec_env = vec_env
        self.out_dir = out_dir
        self.best_eval = -1e18
        self.next_eval = args.eval_freq
        self.next_save = args.save_freq
        self.metrics_path = out_dir / 'train_metrics.csv'
        self.metrics_file = open(self.metrics_path, 'w', newline='')
        self.metrics_writer = csv.DictWriter(
            self.metrics_file,
            fieldnames=['timesteps', 'eval_mean_reward', 'best_eval_mean_reward'],
        )
        self.metrics_writer.writeheader()

        self.discord_bar = None
        if not args.no_discord_tqdm:
            token = os.getenv('TQDM_DISCORD_TOKEN') or _load_discord_token_from_openclaw()
            if token and args.discord_channel_id:
                from tqdm.contrib.discord import tqdm as tqdm_discord
                self.discord_bar = tqdm_discord(
                    total=args.timesteps,
                    token=token,
                    channel_id=int(args.discord_channel_id),
                    desc='🐍 RL Bootstrap',
                    unit='steps',
                    mininterval=0,
                    miniters=1,
                )

    def _apply_curriculum(self, t: int):
        frac = t / max(1, self.args.timesteps)
        if frac < self.args.curriculum_mid_frac:
            ms = self.args.curriculum_start_max_steps
        elif frac < self.args.curriculum_final_frac:
            ms = self.args.curriculum_mid_max_steps
        else:
            ms = self.args.curriculum_final_max_steps
        try:
            self.vec_env.env_method('set_max_steps', ms)
        except Exception:
            pass

    def on_step(self):
        t = int(self.model.num_timesteps)
        self._apply_curriculum(t)

        if self.discord_bar is not None:
            delta = t - self.discord_bar.n
            if delta > 0:
                self.discord_bar.update(delta)

        if t >= self.next_save:
            ckpt = self.out_dir / f'ckpt_{t}.zip'
            self.model.save(str(ckpt))
            self.next_save += self.args.save_freq

        if t >= self.next_eval:
            eval_mean = evaluate_policy_fixed(self.model, self.args.env, self.args.eval_episodes, self.args.seed)
            if eval_mean > self.best_eval:
                self.best_eval = eval_mean
                self.model.save(str(self.out_dir / 'best_eval_model'))
            self.metrics_writer.writerow({
                'timesteps': t,
                'eval_mean_reward': eval_mean,
                'best_eval_mean_reward': self.best_eval,
            })
            self.metrics_file.flush()
            self.next_eval += self.args.eval_freq

    def close(self):
        if self.discord_bar is not None:
            self.discord_bar.close()
        self.metrics_file.close()


def main() -> None:
    args = parse_args()
    ensure_snake_registered()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def wrapper(env):
        return SnakeRewardWrapper(
            env,
            reward_food=args.reward_food,
            reward_death=args.reward_death,
            reward_step=args.reward_step,
            reward_survival=args.reward_survival,
            intrinsic_bonus=args.intrinsic_bonus,
        )

    vec_env = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed, wrapper_class=wrapper)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=(not args.no_norm_obs),
        norm_reward=(not args.no_norm_reward),
        clip_obs=10.0,
    )

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        'MlpPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=1,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        max_grad_norm=args.max_grad_norm,
    )

    cb = EvalBestCallback(args, model, vec_env, out_dir)

    target = int(args.timesteps)
    while model.num_timesteps < target:
        step_chunk = min(args.eval_freq, target - model.num_timesteps)
        model.learn(total_timesteps=step_chunk, reset_num_timesteps=False, progress_bar=False)
        cb.on_step()

    cb.close()

    model_path = out_dir / 'ppo_snakepixels'
    model.save(str(model_path))
    vec_env.save(str(out_dir / 'vecnormalize.pkl'))

    # Final deterministic fixed-seed eval on raw env
    mean_reward = evaluate_policy_fixed(model, args.env, args.eval_episodes, args.seed)

    if args.export_genome:
        export_path = Path(args.export_genome)
    else:
        export_path = out_dir / 'bootstrap_genome.pt'

    # Prefer best-eval checkpoint for export if available
    best_eval_model_path = out_dir / 'best_eval_model.zip'
    if best_eval_model_path.exists():
        best_model = PPO.load(str(best_eval_model_path))
        export_ppo_to_genome(best_model, args.env, export_path)
    else:
        export_ppo_to_genome(model, args.env, export_path)

    metrics = {
        'env': args.env,
        'timesteps': args.timesteps,
        'seed': args.seed,
        'n_envs': args.n_envs,
        'eval_episodes': args.eval_episodes,
        'mean_reward': mean_reward,
        'model_path': str(model_path),
        'bootstrap_genome_path': str(export_path),
        'best_eval_model_path': str(best_eval_model_path) if best_eval_model_path.exists() else None,
        'config': {
            'reward_food': args.reward_food,
            'reward_death': args.reward_death,
            'reward_step': args.reward_step,
            'reward_survival': args.reward_survival,
            'intrinsic_bonus': args.intrinsic_bonus,
            'curriculum': {
                'start_max_steps': args.curriculum_start_max_steps,
                'mid_max_steps': args.curriculum_mid_max_steps,
                'final_max_steps': args.curriculum_final_max_steps,
                'mid_frac': args.curriculum_mid_frac,
                'final_frac': args.curriculum_final_frac,
            },
            'ppo': {
                'lr': args.lr,
                'gamma': args.gamma,
                'gae_lambda': args.gae_lambda,
                'clip_range': args.clip_range,
                'ent_coef': args.ent_coef,
                'vf_coef': args.vf_coef,
                'n_steps': args.n_steps,
                'batch_size': args.batch_size,
                'n_epochs': args.n_epochs,
                'max_grad_norm': args.max_grad_norm,
            },
            'normalization': {
                'norm_obs': not args.no_norm_obs,
                'norm_reward': not args.no_norm_reward,
            },
        },
    }

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\nDone.')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
