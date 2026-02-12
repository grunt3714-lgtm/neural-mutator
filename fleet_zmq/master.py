#!/usr/bin/env python3
"""ZeroMQ master (job dispatcher).

This is a *standalone* dispatcher that demonstrates distributing genome evaluation.
Integration into `src.train` (so evolution uses the fleet for eval) comes next.

Master binds a ROUTER socket and hands out jobs to workers (DEALER identities).

Protocol frames are routed by ZMQ; ROUTER receives [identity, empty?, kind, payload].
We keep payload as a single frame to simplify.
"""

from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Allow running as a script from repo root without installing as a package
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zmq

from fleet_zmq.common import dumps, loads


@dataclass
class Job:
    job_id: str
    genomes: List[bytes]
    env_id: str
    n_episodes: int
    max_steps: int
    offset: int  # index offset in full population


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield i, lst[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://*:5555")
    ap.add_argument("--env", default="SnakePixels-v0")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--min-workers", type=int, default=2)
    args = ap.parse_args()

    # Demo population: random genomes
    from src.snake_env import ensure_snake_registered
    import gymnasium as gym
    import numpy as np
    import torch
    from src.evolution import create_population

    ensure_snake_registered()
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    # Create a population (gaussian, speciation+flex+cc defaults live in src.train, not here)
    pop = create_population(
        pop_size=args.pop,
        obs_dim=obs_dim,
        act_dim=act_dim,
        mutator_type="gaussian",
        hidden=64,
        chunk_size=64,
        speciation=False,
        flex=False,
    )

    genomes_bytes = [g.to_bytes() for g in pop]

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(args.bind)

    print(f"master bound: {args.bind}")
    print("waiting for workers...")

    workers: Dict[bytes, float] = {}

    # Wait for workers
    while len(workers) < args.min_workers:
        ident, kind, payload = _recv_router(sock)
        if kind == b"READY":
            info = loads(payload)
            workers[ident] = time.time()
            print(f"worker online: {ident.decode('utf-8', 'ignore')} info={info}")

    # Create jobs
    jobs: List[Job] = []
    job_map: Dict[str, Job] = {}
    for off, batch in chunk(genomes_bytes, args.batch):
        jid = str(uuid.uuid4())
        j = Job(jid, batch, args.env, args.episodes, args.max_steps, off)
        jobs.append(j)
        job_map[jid] = j

    pending: Dict[str, Job] = {j.job_id: j for j in jobs}
    results: List[Optional[float]] = [None] * len(genomes_bytes)

    # Dispatch loop
    idle_workers = set(workers.keys())
    print(f"dispatching {len(jobs)} jobs to {len(workers)} workers...")

    while pending:
        # Send jobs to idle workers
        while idle_workers and pending:
            wid = idle_workers.pop()
            jid, job = next(iter(pending.items()))
            del pending[jid]
            _send_router(sock, wid, b"JOB", dumps({
                "job_id": job.job_id,
                "env_id": job.env_id,
                "n_episodes": job.n_episodes,
                "max_steps": job.max_steps,
                "genomes": job.genomes,
            }))

        # Wait for messages
        ident, kind, payload = _recv_router(sock)
        workers[ident] = time.time()

        if kind == b"READY":
            idle_workers.add(ident)

        elif kind == b"RESULT":
            msg = loads(payload)
            jid = msg["job_id"]
            job = job_map[jid]
            fits = msg["fitnesses"]
            for i, f in enumerate(fits):
                results[job.offset + i] = float(f)
            idle_workers.add(ident)
            done = sum(r is not None for r in results)
            print(f"result {jid[:8]} from {msg.get('name')} | done {done}/{len(results)}")

    print("all results collected")
    print("fitness stats:")
    arr = [r for r in results if r is not None]
    import numpy as np

    print(f"  mean={np.mean(arr):.3f} best={np.max(arr):.3f} worst={np.min(arr):.3f}")


def _recv_router(sock) -> Tuple[bytes, bytes, bytes]:
    msg = sock.recv_multipart()
    # ROUTER might include an empty delimiter depending on peer; handle both.
    if len(msg) == 4 and msg[1] == b"":
        ident, _, kind, payload = msg
    elif len(msg) == 3:
        ident, kind, payload = msg
    else:
        raise RuntimeError(f"unexpected frames: {len(msg)}")
    return ident, kind, payload


def _send_router(sock, ident: bytes, kind: bytes, payload: bytes):
    # Use 3-frame form (no empty delimiter)
    sock.send_multipart([ident, kind, payload])


if __name__ == "__main__":
    main()
