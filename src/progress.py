"""Progress reporting sinks for training runs.

Decouples train/evolution loops from concrete output targets
(console, discord tqdm, progress-file, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Optional, Iterable


@dataclass
class ProgressEvent:
    gen: int
    total: int
    best: float
    mean: float
    best_ever: float


class ProgressSink:
    def on_progress(self, event: ProgressEvent) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class ProgressFileSink(ProgressSink):
    """Write latest progress JSON for external aggregation."""

    def __init__(self, path: str = '/tmp/train_progress.json', node_name: Optional[str] = None,
                 seed: Optional[int] = None):
        self.path = path
        self.node_name = node_name or 'unknown'
        self.seed = seed

    def on_progress(self, event: ProgressEvent) -> None:
        data = {
            'gen': event.gen,
            'total': event.total,
            'best': event.best,
            'mean': event.mean,
            'best_ever': event.best_ever,
            'node': self.node_name,
            'seed': self.seed,
            'done': (event.gen == event.total - 1),
            'ts': time.time(),
        }
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, self.path)


class DiscordTqdmSink(ProgressSink):
    """tqdm.contrib.discord-based progress sink with optional genome-level bar."""

    def __init__(self, total: int, token: str, channel_id: int, pop_size: int = 0):
        from tqdm.contrib.discord import tqdm as tqdm_discord
        self._bar = tqdm_discord(
            total=total,
            token=token,
            channel_id=int(channel_id),
            desc='🐍 Fleet Training',
            unit='gen',
            mininterval=0,
            miniters=1,
        )
        self._last = 0
        self._pop_size = pop_size
        self._genome_bar = None
        self._token = token
        self._channel_id = channel_id
        if pop_size > 0:
            self._genome_bar = tqdm_discord(
                total=pop_size,
                token=token,
                channel_id=int(channel_id),
                desc='🧬 Genomes',
                unit='genome',
                mininterval=5,
                miniters=1,
            )

    def on_genome_progress(self, collected: int, total: int) -> None:
        """Called per genome result during fleet collect."""
        if self._genome_bar is not None:
            if self._genome_bar.total != total:
                self._genome_bar.total = total
            delta = collected - self._genome_bar.n
            if delta > 0:
                self._genome_bar.update(delta)
                self._genome_bar.refresh()

    def on_progress(self, event: ProgressEvent) -> None:
        # Reset genome bar for next gen
        if self._genome_bar is not None:
            self._genome_bar.reset()

        done = event.gen + 1
        delta = done - self._last
        if delta > 0:
            self._bar.set_postfix({
                'best': f'{event.best:+.2f}',
                'mean': f'{event.mean:+.2f}',
                'best_ever': f'{event.best_ever:+.2f}',
            }, refresh=False)
            self._bar.update(delta)
            self._last = done
        if event.gen == event.total - 1:
            self._bar.close()
            if self._genome_bar is not None:
                self._genome_bar.close()


class ProgressReporter:
    """Adapter callable used by run_evolution(progress_callback=...)."""

    def __init__(self, sinks: Iterable[ProgressSink]):
        self.sinks = [s for s in sinks if s is not None]

    def __call__(self, gen: int, total: int, best: float, mean: float, best_ever: float):
        event = ProgressEvent(gen=gen, total=total, best=best, mean=mean, best_ever=best_ever)
        for sink in self.sinks:
            try:
                sink.on_progress(event)
            except Exception as e:
                print(f'[progress] sink error ({type(sink).__name__}): {e}')

    def get_genome_callback(self):
        """Return a callable(collected, total) for genome-level progress, or None."""
        for sink in self.sinks:
            if hasattr(sink, 'on_genome_progress'):
                return sink.on_genome_progress
        return None
