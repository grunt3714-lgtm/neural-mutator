#!/usr/bin/env python3
"""
Fleet master — runs on gateway, serves work/result queues via multiprocessing.managers.

Usage (standalone):
    python -m fleet.manager --port 5555 --authkey secret

Usage (from evolution.py):
    from fleet.manager import FleetEvaluator
    fleet = FleetEvaluator(port=5555, authkey=b'secret', min_workers=4)
    fitnesses = fleet.evaluate_population(genome_bytes_list, env_id, n_episodes, max_steps)
"""

import time
import struct
from multiprocessing.managers import BaseManager
from multiprocessing import Queue
from typing import List


class FleetManager(BaseManager):
    pass


class FleetEvaluator:
    """Drop-in replacement for local evaluation — dispatches to remote workers."""

    def __init__(self, port: int = 5555, authkey: bytes = b'neuralfleet',
                 min_workers: int = 1, bind: str = '0.0.0.0'):
        self.port = port
        self.authkey = authkey
        self.min_workers = min_workers
        self.bind = bind

        # Queues
        self._work_queue = Queue()
        self._result_queue = Queue()
        self._register_queue = Queue()

        # Register queue accessors
        FleetManager.register('get_work_queue', callable=lambda: self._work_queue)
        FleetManager.register('get_result_queue', callable=lambda: self._result_queue)
        FleetManager.register('get_register_queue', callable=lambda: self._register_queue)

        # Start manager server
        self._manager = FleetManager(address=(bind, port), authkey=authkey)
        self._manager.start()
        print(f"[fleet] Manager listening on {bind}:{port}")

        # Wait for workers
        self._workers = set()
        if min_workers > 0:
            print(f"[fleet] Waiting for {min_workers} worker(s)...")
            while len(self._workers) < min_workers:
                try:
                    name = self._register_queue.get(timeout=2)
                    self._workers.add(name)
                    print(f"[fleet] Worker connected: {name} ({len(self._workers)}/{min_workers})")
                except Exception:
                    pass
            print(f"[fleet] All {len(self._workers)} workers ready")

    # ── Async dispatch/collect API (for pipelined evolution) ──────────

    def _check_new_workers(self):
        """Non-blocking drain of register queue."""
        while not self._register_queue.empty():
            try:
                name = self._register_queue.get_nowait()
                if name not in self._workers:
                    self._workers.add(name)
                    print(f"[fleet] New worker: {name} (total: {len(self._workers)})")
            except Exception:
                break

    def _drain_stale_results(self):
        """Drain stale results from previous generation."""
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Exception:
                break

    def dispatch(self, genomes_bytes: List[bytes], env_id: str,
                 n_episodes: int = 5, max_steps: int = 1000) -> dict:
        """Send all genomes to workers (non-blocking). Returns dispatch metadata.

        Call collect() later to gather results.
        """
        n = len(genomes_bytes)
        self._drain_stale_results()
        self._check_new_workers()

        t0 = time.time()
        for i, gb in enumerate(genomes_bytes):
            self._work_queue.put((i, gb, env_id, n_episodes, max_steps))
        dispatch_sec = time.time() - t0

        # Store pending state
        self._pending_n = n
        self._pending_t0 = time.time()
        self._pending_dispatch_sec = dispatch_sec
        return {'dispatch_sec': dispatch_sec, 'n': n}

    def collect(self) -> tuple:
        """Block until all dispatched results arrive.

        Returns:
            (fitnesses, eval_profile)
        """
        n = self._pending_n
        results = [None] * n
        step_counts = [0] * n
        collected = 0

        while collected < n:
            try:
                payload = self._result_queue.get(timeout=120)
                if len(payload) == 2:
                    idx, fitness = payload
                    steps = 0
                else:
                    idx, fitness, steps = payload
                if results[idx] is None:
                    results[idx] = fitness
                    step_counts[idx] = int(steps)
                    collected += 1
            except Exception:
                elapsed = time.time() - self._pending_t0
                print(f"[fleet] Warning: timeout waiting for results "
                      f"({collected}/{n} after {elapsed:.0f}s)")
                for i in range(n):
                    if results[i] is None:
                        results[i] = -999.0
                        step_counts[i] = 0
                        collected += 1

        remote_eval_sec = time.time() - self._pending_t0
        total_sec = self._pending_dispatch_sec + remote_eval_sec
        profile = {
            'dispatch_sec': float(self._pending_dispatch_sec),
            'remote_eval_sec': float(remote_eval_sec),
            'result_gather_sec': 0.0,
            'eval_total_sec': float(total_sec),
            'env_steps': int(sum(step_counts)),
            'genomes_evaluated': int(n),
            'active_workers': int(len(self._workers)),
        }
        return results, profile

    # ── Synchronous convenience (backwards-compatible) ──────────────

    def evaluate_population(self, genomes_bytes: List[bytes], env_id: str,
                            n_episodes: int = 5, max_steps: int = 1000):
        """Send genomes to workers, collect fitnesses. Blocks until all done.

        Returns:
            (fitnesses, eval_profile)
        """
        self.dispatch(genomes_bytes, env_id, n_episodes, max_steps)
        return self.collect()

    def shutdown(self):
        # Send poison pills
        for _ in range(len(self._workers) + 4):
            self._work_queue.put(None)
        self._manager.shutdown()
        print("[fleet] Manager shut down")
