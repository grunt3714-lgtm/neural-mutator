#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch

from src.snake_env import ensure_snake_registered
from src.genome import Policy, DualMixtureCorrectorMutator, Genome


def _load_discord_token_from_openclaw() -> str | None:
    try:
        cfg_path = os.path.expanduser('~/.openclaw/openclaw.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get('channels', {}).get('discord', {}).get('token')
    except Exception:
        return None


class SnakeRewardWrapper(gym.Wrapper):
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

    def _obs_hash(self, obs: np.ndarray) -> int:
        x = np.asarray(obs, dtype=np.uint8).ravel()[::8]
        return int(np.sum(x.astype(np.int64) * 1315423911) % self.obs_hash_bins)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0
        if reward >= 0.9:
            shaped += self.reward_food
        elif terminated and reward <= -0.9:
            shaped += self.reward_death
        else:
            shaped += self.reward_step + self.reward_survival

        if self.intrinsic_bonus > 0.0:
            h = self._obs_hash(obs)
            self._visit_counts[h] += 1
            shaped += self.intrinsic_bonus / np.sqrt(self._visit_counts[h])

        return obs, float(shaped), terminated, truncated, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train DQN baseline and export bootstrap genome')
    p.add_argument('--env', default='SnakePixels-v0')
    p.add_argument('--timesteps', type=int, default=1_000_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--eval-freq', type=int, default=50_000)
    p.add_argument('--save-freq', type=int, default=100_000)
    p.add_argument('--out-dir', type=str, default='results/dqn_bootstrap')
    p.add_argument('--export-genome', type=str, default='')
    p.add_argument('--device', type=str, default='auto', help='cpu|cuda|auto')

    p.add_argument('--reward-food', type=float, default=10.0)
    p.add_argument('--reward-death', type=float, default=-10.0)
    p.add_argument('--reward-step', type=float, default=-0.001)
    p.add_argument('--reward-survival', type=float, default=0.0)
    p.add_argument('--intrinsic-bonus', type=float, default=0.01)

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--buffer-size', type=int, default=300_000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--train-freq', type=int, default=4)
    p.add_argument('--gradient-steps', type=int, default=1)
    p.add_argument('--target-update-interval', type=int, default=10_000)
    p.add_argument('--learning-starts', type=int, default=20_000)
    p.add_argument('--exploration-initial-eps', type=float, default=1.0)
    p.add_argument('--exploration-final-eps', type=float, default=0.05)
    p.add_argument('--exploration-fraction', type=float, default=0.3)

    p.add_argument('--discord-channel-id', type=int, default=1471943945540866139)
    p.add_argument('--no-discord-tqdm', action='store_true')
    return p.parse_args()


def evaluate_policy_fixed(model, env_id: str, episodes: int, seed: int, max_steps: int = 300) -> float:
    env = gym.make(env_id)
    env = SnakeRewardWrapper(env, intrinsic_bonus=0.0)
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


def export_dqn_to_genome(model, env_id: str, out_path: Path) -> None:
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    env.close()

    policy = Policy(obs_dim, act_dim, hidden=64)

    q_layers = [m for m in model.q_net if isinstance(m, torch.nn.Linear)]
    if len(q_layers) < 3:
        raise RuntimeError(f'Expected at least 3 linear layers in q_net, got {len(q_layers)}')

    l1, l2, l3 = policy.net[0], policy.net[2], policy.net[4]
    with torch.no_grad():
        l1.weight.copy_(q_layers[0].weight)
        l1.bias.copy_(q_layers[0].bias)
        l2.weight.copy_(q_layers[1].weight)
        l2.bias.copy_(q_layers[1].bias)
        l3.weight.copy_(q_layers[2].weight)
        l3.bias.copy_(q_layers[2].bias)

    genome = Genome(policy=policy, mutator=DualMixtureCorrectorMutator(), mutator_type='dualmixture', compat_net=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genome.save(str(out_path))


def main() -> None:
    args = parse_args()
    ensure_snake_registered()

    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = gym.make(args.env)
    train_env = SnakeRewardWrapper(
        train_env,
        reward_food=args.reward_food,
        reward_death=args.reward_death,
        reward_step=args.reward_step,
        reward_survival=args.reward_survival,
        intrinsic_bonus=args.intrinsic_bonus,
    )

    model = DQN(
        policy='MlpPolicy',
        env=train_env,
        seed=args.seed,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        device=args.device,
    )

    metrics_csv = out_dir / 'train_metrics.csv'
    with open(metrics_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['timesteps', 'eval_mean_reward'])
        w.writeheader()

    class EvalBestCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_eval = -1e18

        def _on_step(self) -> bool:
            ts = int(self.num_timesteps)
            if ts > 0 and ts % int(args.eval_freq) == 0:
                mean_r = evaluate_policy_fixed(model, args.env, args.eval_episodes, args.seed)
                with open(metrics_csv, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=['timesteps', 'eval_mean_reward']).writerow({
                        'timesteps': ts,
                        'eval_mean_reward': mean_r,
                    })
                if mean_r > self.best_eval:
                    self.best_eval = mean_r
                    model.save(str(out_dir / 'best_eval_model'))
            return True

    ckpt_cb = CheckpointCallback(save_freq=max(1, args.save_freq), save_path=str(out_dir), name_prefix='ckpt')

    callbacks = [EvalBestCallback(), ckpt_cb]

    pbar = None
    if not args.no_discord_tqdm:
        try:
            from tqdm.contrib.discord import tqdm as discord_tqdm
            token = _load_discord_token_from_openclaw()
            if token:
                pbar = discord_tqdm(total=args.timesteps, token=token, channel_id=args.discord_channel_id, desc='🐍 DQN Bootstrap')
        except Exception:
            pbar = None

    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.prev = 0
        def _on_step(self) -> bool:
            if pbar is not None:
                cur = int(self.num_timesteps)
                if cur > self.prev:
                    pbar.update(cur - self.prev)
                    self.prev = cur
            return True

    callbacks.append(ProgressCallback())

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(str(out_dir / 'dqn_snakepixels'))
    if pbar is not None:
        pbar.close()

    export_path = Path(args.export_genome) if args.export_genome else (out_dir / 'bootstrap_genome.pt')
    export_dqn_to_genome(model, args.env, export_path)

    mean_reward = evaluate_policy_fixed(model, args.env, args.eval_episodes, args.seed)
    summary = {
        'env': args.env,
        'timesteps': args.timesteps,
        'seed': args.seed,
        'eval_episodes': args.eval_episodes,
        'mean_reward': mean_reward,
        'model_path': str(out_dir / 'dqn_snakepixels'),
        'bootstrap_genome_path': str(export_path),
        'best_eval_model_path': str(out_dir / 'best_eval_model.zip'),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print('\nDone.')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
