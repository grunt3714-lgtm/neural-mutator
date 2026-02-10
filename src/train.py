#!/usr/bin/env python3
"""
Train neural mutator genomes via evolution with true self-replication.

Usage:
    python -m src.train --env CartPole-v1 --mutator chunk --generations 100
    python -m src.train --env LunarLander-v3 --mutator transformer --generations 200
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
                        choices=['chunk', 'transformer', 'gaussian'],
                        help='Mutator architecture')
    parser.add_argument('--pop-size', type=int, default=30, help='Population size')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--episodes', type=int, default=5, help='Eval episodes per genome')
    parser.add_argument('--crossover-rate', type=float, default=0.3, help='Crossover probability')
    parser.add_argument('--hidden', type=int, default=64, help='Policy hidden size')
    parser.add_argument('--chunk-size', type=int, default=64, help='Mutator chunk size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default='results', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"{'='*60}")
    print(f"Neural Mutator Evolution (True Self-Replication)")
    print(f"{'='*60}")

    start = time.time()
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
    )
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Final best:  {history['best'][-1]:.2f}")
    print(f"Final mean:  {history['mean'][-1]:.2f}")
    print(f"Final mean fidelity: {history['mean_fidelity'][-1]:.4f}")
    print(f"Final mean drift: {history['mean_mutator_drift'][-1]:.4f}")

    # Save results
    tag = f"{args.env}_{args.mutator}_s{args.seed}"
    result = {
        'env': args.env,
        'mutator': args.mutator,
        'pop_size': args.pop_size,
        'generations': args.generations,
        'seed': args.seed,
        'elapsed_sec': elapsed,
        'history': history,
    }
    json_path = os.path.join(args.output, f"{tag}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved: {json_path}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    gens = range(len(history['best']))

    # Fitness
    ax = axes[0]
    ax.plot(gens, history['best'], label='Best', color='green')
    ax.plot(gens, history['mean'], label='Mean', color='blue')
    ax.fill_between(gens, history['worst'], history['best'], alpha=0.15, color='blue')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title(f'{args.mutator} on {args.env} (true self-replication)')
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

    plt.tight_layout()
    plot_path = os.path.join(args.output, f"{tag}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")


if __name__ == '__main__':
    main()
