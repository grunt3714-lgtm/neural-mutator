#!/usr/bin/env python3
"""Fleet worker for standard NEAT genome evaluation."""

import argparse
import os
import pickle
import socket
import time
import multiprocessing as _mp
from multiprocessing.managers import BaseManager


class FleetManager(BaseManager):
    pass


FleetManager.register('get_work_queue')
FleetManager.register('get_result_queue')
FleetManager.register('get_register_queue')


def _eval_one_indexed(args):
    idx, genome_bytes, env_id, n_episodes, max_steps, cfg_path = args

    import neat
    import numpy as np
    import gymnasium as gym

    genome = pickle.loads(genome_bytes)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg_path,
    )

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make(env_id)
    rewards = []
    steps = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(max_steps):
            out = net.activate(obs)
            action = int(np.argmax(out))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()

    return idx, float(np.mean(rewards)), int(steps)


def run_worker(host: str, port: int, authkey: bytes, workers: int, name: str):
    while True:
        mgr = FleetManager(address=(host, port), authkey=authkey)
        try:
            mgr.connect()
        except Exception:
            time.sleep(2)
            continue

        work_q = mgr.get_work_queue()
        result_q = mgr.get_result_queue()
        register_q = mgr.get_register_queue()
        register_q.put(name)

        pool = _mp.get_context('spawn').Pool(workers) if workers > 1 else None
        try:
            while True:
                job = work_q.get(timeout=30)
                if job is None:
                    return
                # job shape from trainer: (idx, genome_bytes, env, episodes, max_steps, cfg_path)
                if pool is None:
                    idx, fit, steps = _eval_one_indexed(job)
                    result_q.put((idx, fit, steps, name))
                else:
                    idx, fit, steps = pool.apply(_eval_one_indexed, (job,))
                    result_q.put((idx, fit, steps, name))
        except Exception:
            time.sleep(1)
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True)
    ap.add_argument('--port', type=int, default=5556)
    ap.add_argument('--authkey', default='neatfleet')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    name = args.name or f"neat-{socket.gethostname()}"
    os.environ['OMP_NUM_THREADS'] = '1'
    run_worker(args.host, args.port, args.authkey.encode(), args.workers, name)


if __name__ == '__main__':
    main()
