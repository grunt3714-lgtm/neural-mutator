#!/usr/bin/env python3
"""ZeroMQ worker.

Connects to a master ROUTER socket, sends READY, receives eval jobs, returns results.

Protocol (multipart frames):
Worker -> Master: [b"READY", payload]
Master -> Worker: [b"JOB", payload]
Worker -> Master: [b"RESULT", payload]

Payload is pickled dict.
"""

from __future__ import annotations

import argparse
import socket
import time
from typing import Any, Dict, List

# Allow running as a script from repo root without installing as a package
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zmq

from fleet_zmq.common import dumps, loads


def _pool_eval_one(arg):
    """Multiprocessing Pool target (must be top-level/picklable).

    arg: (genome_bytes, env_id, n_episodes, max_steps, seeds)
    """
    genome_bytes, env_id, n_episodes, max_steps, seeds = arg

    import torch
    from src.evolution import evaluate_genome
    from src.genome import Genome
    from src.snake_env import ensure_snake_registered

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    ensure_snake_registered()
    g = Genome.load_bytes(genome_bytes)
    return float(evaluate_genome(g, env_id, int(n_episodes), int(max_steps), seeds=seeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp://192.168.1.95:5555")
    ap.add_argument("--name", default="")
    ap.add_argument("--heartbeat", type=float, default=10.0, help="Seconds between READY pings if idle")

    # Local parallelism (inside one worker)
    ap.add_argument("--local-workers", type=int, default=4,
                    help="Local process workers to evaluate a batch in parallel (default: 4).")
    ap.add_argument("--mp-start", type=str, default="spawn", choices=["spawn", "fork", "forkserver"],
                    help="multiprocessing start method (default: spawn)")

    args = ap.parse_args()

    name = args.name or socket.gethostname()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    # Give the worker a stable identity so the master can track it
    sock.setsockopt(zmq.IDENTITY, name.encode("utf-8"))
    sock.connect(args.connect)

    # Lazy imports after zmq setup
    import multiprocessing as mp
    import torch

    from src.evolution import evaluate_genome
    from src.genome import Genome
    from src.snake_env import ensure_snake_registered

    mp_ctx = mp.get_context(args.mp_start)
    pool = None
    if args.local_workers and args.local_workers > 1:
        pool = mp_ctx.Pool(processes=int(args.local_workers))

    def send_ready():
        sock.send_multipart([b"READY", dumps({"name": name, "ts": time.time(), "local_workers": args.local_workers})])

    ensure_snake_registered()
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    send_ready()
    last_ready = time.time()

    try:
        while True:
            try:
                if sock.poll(timeout=250):
                    msg = sock.recv_multipart()
                    if not msg:
                        continue
                    kind = msg[0]
                    payload = loads(msg[1]) if len(msg) > 1 else {}

                    if kind == b"JOB":
                        job_id = payload["job_id"]
                        env_id = payload["env_id"]
                        n_episodes = int(payload.get("n_episodes", 5))
                        max_steps = int(payload.get("max_steps", 1000))
                        genomes_bytes: List[bytes] = payload["genomes"]
                        seeds_list = payload.get("seeds")
                        if seeds_list is None:
                            seeds_list = [None] * len(genomes_bytes)

                        if pool is None:
                            fitnesses = []
                            for gb, seeds in zip(genomes_bytes, seeds_list):
                                g = Genome.load_bytes(gb)
                                fit = evaluate_genome(g, env_id, n_episodes, max_steps, seeds=seeds)
                                fitnesses.append(float(fit))
                        else:
                            args_list = [(gb, env_id, n_episodes, max_steps, seeds) for gb, seeds in zip(genomes_bytes, seeds_list)]
                            fitnesses = pool.map(_pool_eval_one, args_list)

                        sock.send_multipart([b"RESULT", dumps({
                            "job_id": job_id,
                            "name": name,
                            "fitnesses": fitnesses,
                            "ts": time.time(),
                        })])
                        # Immediately request more work
                        send_ready()
                        last_ready = time.time()

                    elif kind == b"STOP":
                        break

                # idle heartbeat
                if time.time() - last_ready > args.heartbeat:
                    send_ready()
                    last_ready = time.time()

            except KeyboardInterrupt:
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


if __name__ == "__main__":
    main()
