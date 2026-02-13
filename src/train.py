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
    parser.add_argument('--fleet', type=str, default=None,
                        help='ZeroMQ fleet master bind address (e.g. tcp://*:5555). '
                             'Workers must connect to this address.')
    parser.add_argument('--fleet-workers', type=int, default=1,
                        help='Minimum fleet workers to wait for before starting')
    parser.add_argument('--fleet-batch', type=int, default=8,
                        help='Genomes per fleet job batch')
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
        from fleet_zmq.evaluator import FleetZmqEvaluator
        fleet = FleetZmqEvaluator(
            bind=args.fleet,
            min_workers=args.fleet_workers,
            batch_size=args.fleet_batch,
        )
        print(f"Fleet evaluation: {args.fleet} (min workers: {args.fleet_workers}, batch: {args.fleet_batch})")

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
    ax.plot(gens, history['best'], label='Best', color='green')
    ax.plot(gens, history['mean'], label='Mean', color='blue')
    ax.fill_between(gens, history['worst'], history['best'], alpha=0.15, color='blue')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    subtitle = f'{args.mutator} on {args.env} (self-replication'
    if args.speciation:
        subtitle += ' + learned speciation)'
    else:
        subtitle += ')'
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

    # Parameter count plot (when complexity cost is active)
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

    # Speciation plot
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
