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
from .genome import available_mutator_types
from .progress import ProgressFileSink, DiscordTqdmSink, ProgressReporter
from .config import build_configs


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
    parser.add_argument('--mutator', default='dualmixture',
                        choices=available_mutator_types(),
                        help='Mutator architecture')
    parser.add_argument('--pop-size', type=int, default=30, help='Population size')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--episodes', type=int, default=5, help='Eval episodes per genome')
    parser.add_argument('--crossover-rate', type=float, default=0.3, help='Crossover probability')
    parser.add_argument('--hidden', type=int, default=64, help='Policy hidden size')
    parser.add_argument('--policy-arch', default='mlp', choices=['mlp', 'cnn', 'cnn-small', 'cnn-large'],
                        help='Policy architecture (mlp or cnn)')
    parser.add_argument('--chunk-size', type=int, default=64, help='Mutator chunk size')
    parser.add_argument('--dualmix-p-gauss-policy', type=float, default=0.20,
                        help='Dualmixture: probability of Gaussian escape on policy slice')
    parser.add_argument('--dualmix-gauss-scale-policy', type=float, default=0.03,
                        help='Dualmixture: Gaussian escape noise scale on policy slice')
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
    parser.add_argument('--seed-genome', type=str, default=None,
                        help='Path to genome .pt file to seed population from')
    parser.add_argument('--discord-channel-id', type=int, default=1471943945540866139,
                        help='Discord channel for built-in tqdm progress (default: #training)')
    parser.add_argument('--no-discord-tqdm', action='store_true',
                        help='Disable built-in tqdm.contrib.discord progress updates')
    args = parser.parse_args()
    train_cfg, mut_cfg = build_configs(args)

    os.makedirs(train_cfg.output, exist_ok=True)

    print(f"{'='*60}")
    title = "Neural Mutator Evolution (True Self-Replication)"
    if train_cfg.speciation:
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

    # Progress reporting sinks (always file; optionally Discord tqdm)
    progress_sinks = [ProgressFileSink(node_name=os.uname().nodename, seed=train_cfg.seed)]
    if not args.no_discord_tqdm:
        token = os.getenv('TQDM_DISCORD_TOKEN') or _load_discord_token_from_openclaw()
        if token and args.discord_channel_id:
            progress_sinks.append(DiscordTqdmSink(train_cfg.generations, token, args.discord_channel_id,
                                                     pop_size=train_cfg.pop_size))
            print(f"Discord tqdm: enabled (channel {args.discord_channel_id})")
        else:
            print("Discord tqdm: disabled (missing token/channel)")

    progress_callback = ProgressReporter(progress_sinks)

    start = time.time()
    if args.optimized:
        from .optimized_evolution import run_optimized_evolution
        history = run_optimized_evolution(
            env_id=train_cfg.env,
            pop_size=train_cfg.pop_size,
            generations=train_cfg.generations,
            mutator_type=mut_cfg.mutator_type,
            n_eval_episodes=train_cfg.episodes,
            crossover_rate=train_cfg.crossover_rate,
            hidden=train_cfg.hidden,
            chunk_size=mut_cfg.chunk_size,
            seed=train_cfg.seed,
        )
    else:
        history = run_evolution(
            env_id=train_cfg.env,
            pop_size=train_cfg.pop_size,
            generations=train_cfg.generations,
            mutator_type=mut_cfg.mutator_type,
            n_eval_episodes=train_cfg.episodes,
            crossover_rate=train_cfg.crossover_rate,
            hidden=train_cfg.hidden,
            chunk_size=mut_cfg.chunk_size,
            seed=train_cfg.seed,
            speciation=train_cfg.speciation,
            compat_threshold=train_cfg.compat_threshold,
            flex=train_cfg.flex,
            policy_arch=train_cfg.policy_arch,
            complexity_cost=train_cfg.complexity_cost,
            output_dir=train_cfg.output,
            n_workers=train_cfg.workers,
            fleet=fleet,
            progress_callback=progress_callback,
            mutator_kwargs=mut_cfg.to_kwargs(),
            seed_genome_path=args.seed_genome,
        )
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Final best:  {history['best'][-1]:.2f}")
    print(f"Final mean:  {history['mean'][-1]:.2f}")
    print(f"Final mean fidelity: {history['mean_fidelity'][-1]:.4f}")
    print(f"Final mean drift: {history['mean_mutator_drift'][-1]:.4f}")
    if train_cfg.speciation:
        print(f"Final species count: {history['num_species'][-1]}")
        print(f"Final compat rate: {history['crossover_compat_rate'][-1]:.2f}")

    if history.get('env_steps_per_sec'):
        print(f"Avg env steps/sec: {float(np.mean(history['env_steps_per_sec'])):.1f}")
        print(f"Avg genomes/sec:   {float(np.mean(history['genomes_per_sec'])):.2f}")
        print("Avg timing split per gen:")
        print(f"  dispatch:        {float(np.mean(history['dispatch_sec'])):.3f}s")
        print(f"  remote eval:     {float(np.mean(history['remote_eval_sec'])):.3f}s")
        print(f"  evolve step:     {float(np.mean(history['evolution_step_sec'])):.3f}s")
        print(f"  log/checkpoint:  {float(np.mean(history['logging_checkpoint_sec'])):.3f}s")

    # Save results
    spec_tag = "_spec" if train_cfg.speciation else ""
    tag = f"{train_cfg.env}_{mut_cfg.mutator_type}{spec_tag}_s{train_cfg.seed}"
    result = {
        'env': train_cfg.env,
        'mutator': mut_cfg.mutator_type,
        'mutator_config': mut_cfg.to_kwargs(),
        'speciation': train_cfg.speciation,
        'compat_threshold': train_cfg.compat_threshold,
        'pop_size': train_cfg.pop_size,
        'generations': train_cfg.generations,
        'seed': train_cfg.seed,
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

    json_path = os.path.join(train_cfg.output, f"{tag}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=json_safe)
    print(f"Results saved: {json_path}")

    # Plot
    has_complexity = train_cfg.complexity_cost > 0
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
    subtitle = f'{train_cfg.env} | pop {train_cfg.pop_size} | {mut_cfg.mutator_type}'
    extras = []
    if train_cfg.speciation:
        extras.append('speciation')
    if train_cfg.flex:
        extras.append('flex')
    if train_cfg.complexity_cost > 0:
        extras.append(f'cc={train_cfg.complexity_cost}')
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
        ax.set_title(f'Network Size (complexity cost={train_cfg.complexity_cost})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Speciation
    if train_cfg.speciation:
        ax = axes[plot_idx]
        ax2 = ax.twinx()
        ax.plot(gens, history['num_species'], label='Num Species', color='teal', linewidth=2)
        if 'species_entropy' in history and history['species_entropy']:
            ax.plot(gens, history['species_entropy'], label='Species Entropy', color='navy', linewidth=1.5, alpha=0.8)
        ax2.plot(gens, history['crossover_compat_rate'], label='Compat Rate',
                 color='coral', alpha=0.8, linestyle='--')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Species / Entropy', color='teal')
        ax2.set_ylabel('Crossover Compatibility Rate', color='coral')
        ax.set_title('Learned Speciation Dynamics')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(train_cfg.output, f"{tag}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")


if __name__ == '__main__':
    main()
