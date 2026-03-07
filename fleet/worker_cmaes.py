#!/usr/bin/env python3
"""Fleet worker for CMA-ES CarRacing baseline.

Receives flat parameter vectors from a FleetManager queue, evaluates them,
and returns (idx, fitness, steps, worker_name).
"""

import argparse
import os
import socket
import time
import multiprocessing as _mp
from multiprocessing.managers import BaseManager


class FleetManager(BaseManager):
    pass


FleetManager.register('get_work_queue')
FleetManager.register('get_result_queue')
FleetManager.register('get_register_queue')


class PolicyCNNLarge:
    """Exact architecture used by cmaes baseline."""

    def __init__(self, obs_shape=(96, 96, 3), act_dim=3):
        import torch.nn as nn
        self.obs_shape = obs_shape
        h, w, c = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        conv_out = 8 * (h // 16) * (w // 16)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 32),
            nn.Tanh(),
            nn.Linear(32, act_dim),
            nn.Tanh(),
        )

    def parameters(self):
        for p in self.conv.parameters():
            yield p
        for p in self.fc.parameters():
            yield p

    def forward(self, x):
        b = x.shape[0]
        h, w, c = self.obs_shape
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).float() / 255.0
        y = self.conv(x)
        y = y.reshape(b, -1)
        return self.fc(y)

    def act(self, obs):
        import torch
        with torch.no_grad():
            x = torch.FloatTensor(obs.flatten()).unsqueeze(0)
            action = self.forward(x).squeeze(0)
        return action.numpy()


def set_params(policy, flat_params):
    import torch
    idx = 0
    for p in policy.parameters():
        n = p.numel()
        p.data.copy_(torch.from_numpy(flat_params[idx:idx+n]).reshape(p.shape).float())
        idx += n


def evaluate_one(job):
    """Job: (idx, flat_params, n_episodes, max_steps|None, seed_base)"""
    idx, flat_params, n_episodes, max_steps, seed_base = job

    import gymnasium as gym
    import numpy as np

    policy = PolicyCNNLarge()
    set_params(policy, flat_params)

    env = gym.make('CarRacing-v3')
    rewards = []
    total_steps = 0
    for ep in range(n_episodes):
        seed = None if seed_base is None else int(seed_base + ep)
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        step_i = 0
        while True:
            action = policy.act(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            step_i += 1
            if terminated or truncated:
                break
            if max_steps is not None and step_i >= max_steps:
                break
        rewards.append(total_reward)
    env.close()

    return idx, float(np.mean(rewards)), int(total_steps)


def _is_connection_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(k in text for k in [
        'brokenpipeerror', 'eoferror', 'connectionreseterror',
        'connection refused', 'broken pipe', 'eof', 'reset by peer'
    ])


def run_worker(host, port, authkey, local_workers, name=None, prefetch=1):
    if name is None:
        name = f"cma-worker-{socket.gethostname()}"

    while True:
        print(f"[{name}] Connecting to {host}:{port}...")
        manager = FleetManager(address=(host, port), authkey=authkey)

        connected = False
        for attempt in range(30):
            try:
                manager.connect()
                connected = True
                break
            except Exception as e:
                if attempt < 29:
                    time.sleep(2)
                else:
                    print(f"[{name}] Could not connect after 30 attempts: {e}")
        if not connected:
            time.sleep(5)
            continue

        work_q = manager.get_work_queue()
        result_q = manager.get_result_queue()
        register_q = manager.get_register_queue()

        try:
            register_q.put(name)
        except Exception as e:
            print(f"[{name}] register failed: {e}")
            time.sleep(2)
            continue

        print(f"[{name}] Connected, {local_workers} local workers")

        pool = None
        if local_workers > 1:
            pool = _mp.get_context('spawn').Pool(local_workers)

        jobs_pulled = 0
        jobs_sent = 0

        try:
            while True:
                batch = []
                try:
                    job = work_q.get(timeout=30)
                    if job is None:
                        print(f"[{name}] Received shutdown signal")
                        return
                    batch.append(job)
                except Exception as e:
                    if _is_connection_error(e):
                        print(f"[{name}] connection lost: {e}")
                        break
                    continue

                for _ in range(max(0, prefetch - 1)):
                    try:
                        job = work_q.get_nowait()
                        if job is None:
                            return
                        batch.append(job)
                    except Exception:
                        break

                jobs_pulled += len(batch)

                if pool is None:
                    results = [evaluate_one(j) for j in batch]
                else:
                    results = list(pool.imap_unordered(evaluate_one, batch))

                for idx, fit, steps in results:
                    try:
                        result_q.put((idx, fit, int(steps), name))
                        jobs_sent += 1
                    except Exception as e:
                        if _is_connection_error(e):
                            break
                        raise

        except KeyboardInterrupt:
            return
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
            print(f"[{name}] Reconnecting... (pulled={jobs_pulled}, sent={jobs_sent})")
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description='CMA-ES Fleet Worker')
    ap.add_argument('--host', required=True)
    ap.add_argument('--port', type=int, default=5557)
    ap.add_argument('--authkey', default='cmafleet')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--prefetch', type=int, default=1)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'
    run_worker(
        host=args.host,
        port=args.port,
        authkey=args.authkey.encode(),
        local_workers=args.workers,
        name=args.name,
        prefetch=args.prefetch,
    )


if __name__ == '__main__':
    main()
