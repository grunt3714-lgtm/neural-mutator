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

    def evaluate_population(self, genomes_bytes: List[bytes], env_id: str,
                            n_episodes: int = 5, max_steps: int = 1000) -> List[float]:
        """Send genomes to workers, collect fitnesses. Blocks until all done."""
        n = len(genomes_bytes)

        # Drain any stale results
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Exception:
                break

        # Check for new worker registrations (non-blocking)
        while not self._register_queue.empty():
            try:
                name = self._register_queue.get_nowait()
                if name not in self._workers:
                    self._workers.add(name)
                    print(f"[fleet] New worker: {name} (total: {len(self._workers)})")
            except Exception:
                break

        # Dispatch all jobs
        for i, gb in enumerate(genomes_bytes):
            self._work_queue.put((i, gb, env_id, n_episodes, max_steps))

        # Collect results
        results = [None] * n
        collected = 0
        t0 = time.time()

        while collected < n:
            try:
                idx, fitness = self._result_queue.get(timeout=120)
                if results[idx] is None:
                    results[idx] = fitness
                    collected += 1
            except Exception:
                elapsed = time.time() - t0
                print(f"[fleet] Warning: timeout waiting for results "
                      f"({collected}/{n} after {elapsed:.0f}s)")
                # Fill missing with worst possible
                for i in range(n):
                    if results[i] is None:
                        results[i] = -999.0
                        collected += 1

        return results

    def shutdown(self):
        # Send poison pills
        for _ in range(len(self._workers) + 4):
            self._work_queue.put(None)
        self._manager.shutdown()
        print("[fleet] Manager shut down")
