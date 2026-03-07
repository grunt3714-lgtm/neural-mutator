#!/usr/bin/env python3
"""
Fleet worker for HyperNEAT — evaluates CPPN genomes on remote nodes.

Usage:
    python -m fleet.worker_hyperneat --host 192.168.1.94 --port 5555 --authkey neuralfleet --workers 7
"""

import argparse
import os
import socket
import sys
import time
import multiprocessing as _mp
from multiprocessing.managers import BaseManager

Pool = _mp.Pool


class FleetManager(BaseManager):
    pass

FleetManager.register('get_work_queue')
FleetManager.register('get_result_queue')
FleetManager.register('get_register_queue')


def _eval_one_indexed(args):
    """Evaluate a single CPPN genome by index."""
    idx, genome_bytes, env_id, n_episodes, max_steps = args

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.cppn import CPPN
    from src.hyperneat_evolution import evaluate_genome

    cppn = CPPN.from_bytes(genome_bytes)
    fit, steps = evaluate_genome(cppn, env_id, n_episodes=n_episodes, max_steps=max_steps)
    return idx, float(fit), int(steps)


def _is_connection_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(k in text for k in [
        'brokenpipeerror', 'eoferror', 'connectionreseterror',
        'connection refused', 'broken pipe', 'eof', 'reset by peer'
    ])


def run_worker(host, port, authkey, local_workers, name=None, prefetch=1):
    if name is None:
        name = f"hn-worker-{socket.gethostname()}"

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
            ctx = _mp.get_context('spawn')
            pool = ctx.Pool(local_workers)

        jobs_pulled = 0
        jobs_sent = 0
        last_report = time.time()

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
                    except Exception as e:
                        if _is_connection_error(e):
                            break
                        break

                if not batch:
                    continue

                jobs_pulled += len(batch)
                indexed_args = [(idx, gb, env_id, n_ep, ms) for (idx, gb, env_id, n_ep, ms) in batch]

                try:
                    if pool is not None:
                        for idx, fit, steps in pool.imap_unordered(_eval_one_indexed, indexed_args):
                            try:
                                result_q.put((idx, fit, int(steps), name))
                                jobs_sent += 1
                            except Exception as e:
                                if _is_connection_error(e):
                                    break
                                raise
                    else:
                        for a in indexed_args:
                            idx, fit, steps = _eval_one_indexed(a)
                            try:
                                result_q.put((idx, fit, int(steps), name))
                                jobs_sent += 1
                            except Exception as e:
                                if _is_connection_error(e):
                                    break
                                raise
                except Exception as e:
                    print(f"[{name}] Eval error: {e}")

                now = time.time()
                if jobs_sent > 0 and (jobs_sent % 20 == 0 or (now - last_report) > 120):
                    print(f"[{name}] stats pulled={jobs_pulled} sent={jobs_sent}")
                    last_report = now

        except KeyboardInterrupt:
            return
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
            print(f"[{name}] Reconnecting... (pulled={jobs_pulled}, sent={jobs_sent})")
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description='HyperNEAT Fleet Worker')
    parser.add_argument('--host', required=True, help='Manager host IP')
    parser.add_argument('--port', type=int, default=5555, help='Manager port')
    parser.add_argument('--authkey', default='neuralfleet', help='Auth key')
    parser.add_argument('--workers', type=int, default=7, help='Local worker processes')
    parser.add_argument('--prefetch', type=int, default=1, help='Jobs to prefetch')
    parser.add_argument('--name', default=None, help='Worker name')
    args = parser.parse_args()

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
