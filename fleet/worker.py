#!/usr/bin/env python3
"""
Fleet worker — runs on each node, connects to the manager and evaluates genomes.

Usage:
    python -m fleet.worker --host 192.168.1.94 --port 5555 --authkey secret --workers 7
"""

import argparse
import os
import socket
import sys
import time
from multiprocessing import Pool, set_start_method
from multiprocessing.managers import BaseManager


class FleetManager(BaseManager):
    pass

# Register same names as master (client side — no callables needed)
FleetManager.register('get_work_queue')
FleetManager.register('get_result_queue')
FleetManager.register('get_register_queue')


def _eval_one(args):
    """Evaluate a single genome in a worker subprocess."""
    genome_bytes, env_id, n_episodes, max_steps = args

    # Lazy imports inside worker to avoid fork issues
    import numpy as np
    import gymnasium as gym
    import torch

    # Ensure snake env registered
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.snake_env import ensure_snake_registered
    from src.genome import Genome
    ensure_snake_registered()

    genome = Genome.load_bytes(genome_bytes)
    env = gym.make(env_id)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = genome.policy.act(obs)
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(np.argmax(action))
            else:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards))


def run_worker(host: str, port: int, authkey: bytes, local_workers: int,
               name: str = None):
    """Connect to fleet manager and process evaluation jobs."""
    if name is None:
        name = f"worker-{socket.gethostname()}"

    print(f"[{name}] Connecting to {host}:{port}...")
    manager = FleetManager(address=(host, port), authkey=authkey)

    # Retry connection
    for attempt in range(30):
        try:
            manager.connect()
            break
        except ConnectionRefusedError:
            if attempt < 29:
                time.sleep(2)
            else:
                print(f"[{name}] Could not connect after 30 attempts")
                return

    work_q = manager.get_work_queue()
    result_q = manager.get_result_queue()
    register_q = manager.get_register_queue()

    # Announce ourselves
    register_q.put(name)
    print(f"[{name}] Connected, {local_workers} local workers")

    # Create local process pool
    pool = Pool(local_workers)

    try:
        while True:
            # Collect a batch of jobs
            batch = []
            try:
                job = work_q.get(timeout=30)
                if job is None:  # poison pill
                    print(f"[{name}] Received shutdown signal")
                    break
                batch.append(job)
            except Exception:
                continue

            # Grab more if available (up to local_workers worth)
            for _ in range(local_workers * 2 - 1):
                try:
                    job = work_q.get_nowait()
                    if job is None:
                        break
                    batch.append(job)
                except Exception:
                    break

            if not batch:
                continue

            # Evaluate batch locally in parallel
            eval_args = [(gb, env_id, n_ep, ms) for (idx, gb, env_id, n_ep, ms) in batch]
            indices = [idx for (idx, gb, env_id, n_ep, ms) in batch]

            try:
                fitnesses = pool.map(_eval_one, eval_args)
                for idx, fit in zip(indices, fitnesses):
                    result_q.put((idx, fit))
            except Exception as e:
                print(f"[{name}] Eval error: {e}")
                # Return worst fitness for failed jobs
                for idx in indices:
                    result_q.put((idx, -999.0))

    except KeyboardInterrupt:
        pass
    finally:
        pool.terminate()
        pool.join()
        print(f"[{name}] Shut down")


def main():
    parser = argparse.ArgumentParser(description='Fleet worker')
    parser.add_argument('--host', required=True, help='Manager host IP')
    parser.add_argument('--port', type=int, default=5555, help='Manager port')
    parser.add_argument('--authkey', default='neuralfleet', help='Auth key')
    parser.add_argument('--workers', type=int, default=7, help='Local worker processes')
    parser.add_argument('--name', default=None, help='Worker name')
    args = parser.parse_args()

    # OMP_NUM_THREADS=1 to avoid PyTorch deadlock in fork pools
    os.environ['OMP_NUM_THREADS'] = '1'

    run_worker(
        host=args.host,
        port=args.port,
        authkey=args.authkey.encode(),
        local_workers=args.workers,
        name=args.name,
    )


if __name__ == '__main__':
    main()
