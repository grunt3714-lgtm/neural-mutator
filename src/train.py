#!/usr/bin/env python3
"""
Train neural mutator genomes via evolution with true self-replication.

Usage:
    python -m src.train --env CartPole-v1 --mutator chunk --generations 100
    python -m src.train --env LunarLander-v3 --mutator transformer --generations 200
    python -m src.train --env CartPole-v1 --mutator chunk --speciation --generations 200
"""

import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from .evolution import run_evolution


class ProgressFileWriter:
    """Writes progress to a JSON file each generation for fleet aggregation."""

    def __init__(self, path: str = '/tmp/train_progress.json', node_name: str = None,
                 seed: int = None):
        self.path = path
        self.node_name = node_name or 'unknown'
        self.seed = seed

    def __call__(self, gen: int, total: int, best: float, mean: float, best_ever: float):
        import json
        data = {
            'gen': gen, 'total': total,
            'best': best, 'mean': mean, 'best_ever': best_ever,
            'node': self.node_name, 'seed': self.seed,
            'done': (gen == total - 1),
            'ts': time.time(),
        }
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, self.path)


class DiscordTqdmProgress:
    """Built-in tqdm.contrib.discord progress for fleet runs."""

    def __init__(self, total: int, token: str, channel_id: int):
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

    def __call__(self, gen: int, total: int, best: float, mean: float, best_ever: float):
        done = gen + 1
        delta = done - self._last
        if delta > 0:
            self._bar.set_postfix({
                'best': f'{best:+.2f}',
                'mean': f'{mean:+.2f}',
                'best_ever': f'{best_ever:+.2f}',
            }, refresh=False)
            self._bar.update(delta)
            self._last = done
        if gen == total - 1:
            self._bar.close()


class CombinedProgress:
    def __init__(self, callbacks):
        self.callbacks = [cb for cb in callbacks if cb is not None]

    def __call__(self, gen: int, total: int, best: float, mean: float, best_ever: float):
        for cb in self.callbacks:
            try:
                cb(gen, total, best, mean, best_ever)
            except Exception as e:
                print(f'[progress] callback error: {e}')


def _load_discord_token_from_openclaw() -> str | None:
    try:
        cfg_path = os.path.expanduser('~/.openclaw/openclaw.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get('channels', {}).get('discord', {}).get('token')
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Neural Mutator Evolution')
    parser.add_argument('--env', default='CartPole-v1', help='Gymnasium environment')
    parser.add_argument('--mutator', default='chunk',
                        choices=['chunk', 'transformer', 'gaussian', 'corrector'],
                        help='Mutator architecture')
    parser.add_argument('--pop-size', type=int, default=30, help='Population size')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--episodes', type=int, default=5, help='Eval episodes per genome')
    parser.add_argument('--crossover-rate', type=float, default=0.3, help='Crossover probability')
    parser.add_argument('--hidden', type=int, default=64, help='Policy hidden size')
    parser.add_argument('--chunk-size', type=int, default=64, help='Mutator chunk size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--speciation', action='store_true',
                        help='Enable learned speciation (compat net in genome)')
    parser.add_argument('--compat-threshold', type=float, default=0.5,
                        help='Compatibility threshold for crossover (0-1)')
    parser.add_argument('--flex', action='store_true',
                        help='Enable flexible architecture with structural mutations')
    parser.add_argument('--complexity-cost', type=float, default=0.0,
                        help='Per-parameter fitness penalty (encourages smaller networks)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers for genome evaluation (default: 1)')
    parser.add_argument('--optimized', action='store_true',
                        help='Use optimized evolution (adaptive mutation, stagnation detection)')
    parser.add_argument('--fleet', action='store_true',
                        help='Enable distributed fleet evaluation via multiprocessing.managers')
    parser.add_argument('--fleet-port', type=int, default=5555,
                        help='Fleet manager port (default: 5555)')
    parser.add_argument('--fleet-workers', type=int, default=1,
                        help='Minimum fleet workers to wait for before starting')
    parser.add_argument('--fleet-authkey', default='neuralfleet',
                        help='Fleet auth key')
    parser.add_argument('--discord-channel-id', type=int, default=1471943945540866139,
                        help='Discord channel for built-in tqdm progress (default: #training)')
    parser.add_argument('--no-discord-tqdm', action='store_true',
                        help='Disable built-in tqdm.contrib.discord progress updates')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"{'='*60}")
    title = "Neural Mutator Evolution (True Self-Replication)"
    if args.speciation:
        title += " + Learned Speciation"
    print(title)
    print(f"{'='*60}")

    fleet = None
    if args.fleet:
        from fleet.manager import FleetEvaluator
        fleet = FleetEvaluator(
            port=args.fleet_port,
            authkey=args.fleet_authkey.encode(),
            min_workers=args.fleet_workers,
        )
        print(f"Fleet evaluation: port {args.fleet_port} (min workers: {args.fleet_workers})")

    # Built-in progress callbacks (always file; fleet also Discord tqdm)
    progress_callbacks = [ProgressFileWriter(node_name=os.uname().nodename, seed=args.seed)]
    if args.fleet and not args.no_discord_tqdm:
        token = os.getenv('TQDM_DISCORD_TOKEN') or _load_discord_token_from_openclaw()
        if token and args.discord_channel_id:
            progress_callbacks.append(
                DiscordTqdmProgress(args.generations, token, args.discord_channel_id)
            )
            print(f"Discord tqdm: enabled (channel {args.discord_channel_id})")
        else:
            print("Discord tqdm: disabled (missing token/channel)")

    progress_callback = CombinedProgress(progress_callbacks)

    start = time.time()
    if args.optimized:
        from .optimized_evolution import run_optimized_evolution
        history = run_optimized_evolution(
            env_id=args.env,
            pop_size=args.pop_size,
            generations=args.generations,
            mutator_type=args.mutator,
            n_eval_episodes=args.episodes,
            crossover_rate=args.crossover_rate,
            hidden=args.hidden,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )
    else:
        history = run_evolution(
            env_id=args.env,
            pop_size=args.pop_size,
            generations=args.generations,
            mutator_type=args.mutator,
            n_eval_episodes=args.episodes,
            crossover_rate=args.crossover_rate,
            hidden=args.hidden,
            chunk_size=args.chunk_size,
            seed=args.seed,
            speciation=args.speciation,
            compat_threshold=args.compat_threshold,
            flex=args.flex,
            complexity_cost=args.complexity_cost,
            output_dir=args.output,
            n_workers=args.workers,
            fleet=fleet,
            progress_callback=progress_callback,
        )
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Final best:  {history['best'][-1]:.2f}")
    print(f"Final mean:  {history['mean'][-1]:.2f}")
    print(f"Final mean fidelity: {history['mean_fidelity'][-1]:.4f}")
    print(f"Final mean drift: {history['mean_mutator_drift'][-1]:.4f}")
    if args.speciation:
        print(f"Final species count: {history['num_species'][-1]}")
        print(f"Final compat rate: {history['crossover_compat_rate'][-1]:.2f}")

    # Save results
    spec_tag = "_spec" if args.speciation else ""
    tag = f"{args.env}_{args.mutator}{spec_tag}_s{args.seed}"
    result = {
        'env': args.env,
        'mutator': args.mutator,
        'speciation': args.speciation,
        'compat_threshold': args.compat_threshold,
        'pop_size': args.pop_size,
        'generations': args.generations,
        'seed': args.seed,
        'elapsed_sec': elapsed,
        'history': history,
    }
    # Convert numpy types for JSON serialization
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    json_path = os.path.join(args.output, f"{tag}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=json_safe)
    print(f"Results saved: {json_path}")

    # Plot
    has_complexity = args.complexity_cost > 0
    n_plots = 3 + int(args.speciation) + int(has_complexity)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4.5 * n_plots))
    gens = range(len(history['best']))

    # Fitness
    ax = axes[0]
    ax.plot(gens, history['best'], label='Best', color='green', linewidth=2)
    ax.plot(gens, history['mean'], label='Mean', color='blue', linewidth=1.5)
    ax.fill_between(gens, history['worst'], history['best'], alpha=0.12, color='blue')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    subtitle = f'{args.env} | pop {args.pop_size} | {args.mutator}'
    extras = []
    if args.speciation:
        extras.append('speciation')
    if args.flex:
        extras.append('flex')
    if args.complexity_cost > 0:
        extras.append(f'cc={args.complexity_cost}')
    if extras:
        subtitle += f' ({", ".join(extras)})'
    ax.set_title(subtitle)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fidelity
    ax = axes[1]
    ax.plot(gens, history['mean_fidelity'], label='Mean Fidelity', color='orange')
    ax.plot(gens, history['best_fidelity'], label='Best Genome Fidelity', color='red')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Self-Replication Fidelity (L2, lower=better)')
    ax.set_title('Self-Replication Fidelity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Drift
    ax = axes[2]
    ax.plot(gens, history['mean_mutator_drift'], label='Mean Drift', color='purple')
    ax.plot(gens, history['max_mutator_drift'], label='Max Drift', color='magenta', alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mutator Weight Drift (L2 from gen 0)')
    ax.set_title('Mutator Weight Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Parameter count
    plot_idx = 3
    if has_complexity:
        ax = axes[plot_idx]
        ax.plot(gens, history['mean_policy_params'], label='Mean Params', color='teal')
        ax.plot(gens, history['min_policy_params'], label='Min Params', color='darkgreen', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Policy Parameters')
        ax.set_title(f'Network Size (complexity cost={args.complexity_cost})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Speciation
    if args.speciation:
        ax = axes[plot_idx]
        ax2 = ax.twinx()
        ax.plot(gens, history['num_species'], label='Num Species', color='teal', linewidth=2)
        ax2.plot(gens, history['crossover_compat_rate'], label='Compat Rate',
                 color='coral', alpha=0.8, linestyle='--')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Number of Species', color='teal')
        ax2.set_ylabel('Crossover Compatibility Rate', color='coral')
        ax.set_title('Learned Speciation Dynamics')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output, f"{tag}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")


if __name__ == '__main__':
    main()
