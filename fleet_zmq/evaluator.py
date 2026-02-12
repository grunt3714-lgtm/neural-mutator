from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import zmq

from fleet_zmq.common import dumps, loads


@dataclass
class Job:
    job_id: str
    genomes: List[bytes]
    offset: int


class FleetZmqEvaluator:
    """ROUTER-side job dispatcher used by the *training* process.

    Workers run `fleet_zmq/worker.py` and connect via DEALER.

    This keeps evolution centralized but distributes fitness evaluation.
    """

    def __init__(
        self,
        bind: str = "tcp://*:5555",
        min_workers: int = 1,
        batch_size: int = 8,
        recv_timeout_ms: int = 30_000,
        startup_timeout_s: float = 60.0,
        verbose: bool = True,
    ):
        self.bind = bind
        self.min_workers = min_workers
        self.batch_size = batch_size
        self.recv_timeout_ms = recv_timeout_ms
        self.startup_timeout_s = startup_timeout_s
        self.verbose = verbose

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(self.bind)

        self._workers_last_seen: Dict[bytes, float] = {}
        self._ready_once: bool = False

        if self.verbose:
            print(f"FleetZmqEvaluator bound: {self.bind}")

    def close(self):
        try:
            self._sock.close(0)
        except Exception:
            pass

    def _recv(self, timeout_ms: Optional[int] = None) -> Optional[Tuple[bytes, bytes, bytes]]:
        timeout = self.recv_timeout_ms if timeout_ms is None else timeout_ms
        if not self._sock.poll(timeout=timeout):
            return None
        msg = self._sock.recv_multipart()
        # Expect [ident, kind, payload] or [ident, b"", kind, payload]
        if len(msg) == 4 and msg[1] == b"":
            ident, _, kind, payload = msg
        elif len(msg) == 3:
            ident, kind, payload = msg
        else:
            raise RuntimeError(f"unexpected frames: {len(msg)}")
        return ident, kind, payload

    def _send(self, ident: bytes, kind: bytes, payload: bytes):
        self._sock.send_multipart([ident, kind, payload])

    def ensure_workers(self):
        """Block until at least min_workers have checked in.

        Only blocks on the *first* call. After that we assume the fleet is up;
        if workers disconnect later, eval will naturally stall/timeout.
        """
        if self._ready_once:
            return

        deadline = time.time() + self.startup_timeout_s
        if self.verbose:
            print(f"Waiting for {self.min_workers} fleet workers...")

        while len(self._workers_last_seen) < self.min_workers:
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for workers: have {len(self._workers_last_seen)}/{self.min_workers}"
                )
            msg = self._recv(timeout_ms=int(min(1000, remaining * 1000)))
            if msg is None:
                continue
            ident, kind, payload = msg
            self._workers_last_seen[ident] = time.time()
            if kind == b"READY" and self.verbose:
                info = loads(payload) if payload else {}
                wid = ident.decode("utf-8", "ignore")
                print(f"  worker online: {wid} info={info}")

        self._ready_once = True

    def evaluate_population(
        self,
        genomes_bytes: List[bytes],
        env_id: str,
        n_episodes: int,
        max_steps: int,
        seeds_per_genome: List[List[int]] | None = None,
    ) -> List[float]:
        """Distribute evaluation across workers.

        Returns fitness list aligned with genomes_bytes.
        """
        self.ensure_workers()

        results: List[Optional[float]] = [None] * len(genomes_bytes)

        if seeds_per_genome is None:
            seeds_per_genome = [[] for _ in range(len(genomes_bytes))]

        # Create jobs
        jobs: Dict[str, Job] = {}
        pending: List[Job] = []
        for off in range(0, len(genomes_bytes), self.batch_size):
            jid = str(uuid.uuid4())
            batch = genomes_bytes[off : off + self.batch_size]
            batch_seeds = seeds_per_genome[off : off + self.batch_size]
            job = Job(job_id=jid, genomes=batch, offset=off)
            # stash seeds on the Job object dynamically (keep dataclass simple)
            job.seeds = batch_seeds  # type: ignore
            jobs[jid] = job
            pending.append(job)

        # Workers become idle when they send READY.
        idle = set(self._workers_last_seen.keys())
        in_flight: Dict[str, bytes] = {}  # job_id -> worker_id

        def dispatch(worker_id: bytes, job: Job):
            in_flight[job.job_id] = worker_id
            self._send(
                worker_id,
                b"JOB",
                dumps(
                    {
                        "job_id": job.job_id,
                        "env_id": env_id,
                        "n_episodes": int(n_episodes),
                        "max_steps": int(max_steps),
                        "genomes": job.genomes,
                        "seeds": getattr(job, 'seeds', None),
                    }
                ),
            )

        # Main loop
        while pending or in_flight:
            while idle and pending:
                wid = idle.pop()
                job = pending.pop(0)
                dispatch(wid, job)

            msg = self._recv()
            if msg is None:
                # Timeout: if we have idle workers but no messages, continue
                # If a worker died, user can restart workers; we don't aggressively reassign here.
                continue

            ident, kind, payload = msg
            self._workers_last_seen[ident] = time.time()

            if kind == b"READY":
                idle.add(ident)
                continue

            if kind == b"RESULT":
                data = loads(payload)
                jid = data["job_id"]
                job = jobs.get(jid)
                if job is None:
                    continue

                fits = data["fitnesses"]
                for i, f in enumerate(fits):
                    results[job.offset + i] = float(f)

                in_flight.pop(jid, None)
                idle.add(ident)
                continue

        # Fill check
        missing = [i for i, v in enumerate(results) if v is None]
        if missing:
            raise RuntimeError(f"Fleet eval missing results for {len(missing)} genomes")

        return [float(v) for v in results]  # type: ignore
