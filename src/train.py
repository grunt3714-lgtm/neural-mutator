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
from .lineage import generate_lineage_plots
from .progress import ProgressFileSink, DiscordTqdmSink, DiscordMessageSink, ProgressReporter
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
    parser.add_argument('--policy-arch', default='mlp', choices=['mlp', 'cnn', 'cnn-large'],
                        help='Policy architecture (mlp or cnn)')
    parser.add_argument('--chunk-size', type=int, default=64, help='Mutator chunk size')
    parser.add_argument('--dualmix-p-gauss-policy', type=float, default=0.20,
                        help='Dualmixture: probability of Gaussian escape on policy slice')
    parser.add_argument('--dualmix-gauss-scale-policy', type=float, default=0.03,
                        help='Dualmixture: Gaussian escape noise scale on policy slice')
    parser.add_argument('--dualmix-v2-ref-dim', type=int, default=16,
                        help='Dualmixture v2: chunk encoder/reference dimension')
    parser.add_argument('--dualmix-v2-hidden', type=int, default=64,
                        help='Dualmixture v2: hidden width for mutation heads')
    parser.add_argument('--dualmix-v2-lowrank-rank', type=int, default=4,
                        help='Dualmixture v2: low-rank factor rank for policy deltas')
    parser.add_argument('--dualmix-v2-max-policy-groups', type=int, default=8,
                        help='Dualmixture v2: max number of policy chunk groups')
    parser.add_argument('--dualmix-v2-policy-corr-scale', type=float, default=0.025,
                        help='Dualmixture v2: policy corrector scale')
    parser.add_argument('--dualmix-v2-policy-noise-scale', type=float, default=0.008,
                        help='Dualmixture v2: policy exploration noise scale')
    parser.add_argument('--dualmix-v2-meta-corr-scale', type=float, default=0.01,
                        help='Dualmixture v2: meta (mutator+compat) corrector scale')
    parser.add_argument('--dualmix-v2-meta-noise-scale', type=float, default=0.002,
                        help='Dualmixture v2: meta (mutator+compat) exploration noise scale')
    parser.add_argument('--global-lowrank-block-size', type=int, default=64,
                        help='Global low-rank: block size')
    parser.add_argument('--global-lowrank-ref-dim', type=int, default=16,
                        help='Global low-rank: reference/context dimension')
    parser.add_argument('--global-lowrank-hidden', type=int, default=64,
                        help='Global low-rank: hidden width for heads')
    parser.add_argument('--global-lowrank-rank', type=int, default=4,
                        help='Global low-rank: low-rank factor rank')
    parser.add_argument('--global-lowrank-policy-scale', type=float, default=0.02,
                        help='Global low-rank: policy delta scale')
    parser.add_argument('--global-lowrank-meta-scale', type=float, default=0.006,
                        help='Global low-rank: meta delta scale')
    parser.add_argument('--metastable-lowrank-block-size', type=int, default=64,
                        help='Metastable low-rank: block size')
    parser.add_argument('--metastable-lowrank-ref-dim', type=int, default=16,
                        help='Metastable low-rank: reference/context dimension')
    parser.add_argument('--metastable-lowrank-hidden', type=int, default=64,
                        help='Metastable low-rank: hidden width for heads')
    parser.add_argument('--metastable-lowrank-rank', type=int, default=4,
                        help='Metastable low-rank: low-rank factor rank')
    parser.add_argument('--metastable-lowrank-policy-scale', type=float, default=0.02,
                        help='Metastable low-rank: policy delta scale')
    parser.add_argument('--metastable-lowrank-meta-scale', type=float, default=0.006,
                        help='Metastable low-rank: conservative meta delta scale')
    parser.add_argument('--metastable-lowrank-risk-base', type=float, default=0.08,
                        help='Metastable low-rank: baseline endogenous risk level')
    parser.add_argument('--metastable-lowrank-risk-gain', type=float, default=0.28,
                        help='Metastable low-rank: learned per-block risk gain')
    parser.add_argument('--metastable-lowrank-tail-scale', type=float, default=0.10,
                        help='Metastable low-rank: heavy-tail exploration scale on policy slice')
    parser.add_argument('--perceiver-lite-block-size', type=int, default=64,
                        help='Perceiver-lite: block size')
    parser.add_argument('--perceiver-lite-ref-dim', type=int, default=16,
                        help='Perceiver-lite: token/latent dimension')
    parser.add_argument('--perceiver-lite-hidden', type=int, default=64,
                        help='Perceiver-lite: hidden width for decoder')
    parser.add_argument('--perceiver-lite-latent-count', type=int, default=8,
                        help='Perceiver-lite: number of latent slots')
    parser.add_argument('--perceiver-lite-policy-scale', type=float, default=0.018,
                        help='Perceiver-lite: policy delta scale')
    parser.add_argument('--perceiver-lite-meta-scale', type=float, default=0.004,
                        help='Perceiver-lite: conservative meta delta scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--speciation', action='store_true',
                        help='Enable learned speciation (compat net in genome)')
    parser.add_argument('--compat-threshold', type=float, default=0.5,
                        help='Initial/fixed compatibility threshold for crossover (0-1)')
    parser.add_argument('--learn-compat-threshold', action='store_true',
                        help='Evolve per-genome compatibility thresholds instead of using a fixed global value')
    parser.add_argument('--compat-rate-guardrail', action='store_true',
                        help='Enable a gentle run-level compatibility-rate guardrail for crossover gating')
    parser.add_argument('--compat-rate-target-low', type=float, default=0.35,
                        help='Guardrail lower target for crossover compatibility rate')
    parser.add_argument('--compat-rate-target-high', type=float, default=0.75,
                        help='Guardrail upper target for crossover compatibility rate')
    parser.add_argument('--compat-rate-adjust', type=float, default=0.03,
                        help='Guardrail per-generation gate bias adjustment step')
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
    parser.add_argument('--survivor-fraction', type=float, default=0.25,
                        help='Fraction of top genomes to carry over each generation (prevents near-100% cull)')
    parser.add_argument('--init-genome', default=None,
                        help='Optional path to a seed genome (.pt) used to initialize population (policy + mutator)')
    parser.add_argument('--init-mutator-from-genome', default=None,
                        help='Optional path to a seed genome (.pt) used to initialize mutator only; policy stays random')
    parser.add_argument('--mutation-decay', action='store_true',
                        help='Enable legacy generation-based mutation decay (default: off, constant mutation scale)')
    parser.add_argument('--compat-binary', action='store_true',
                        help='Use binary-classifier pretraining for compatibility net (instead of contrastive)')
    parser.add_argument('--unified-mating', action='store_true',
                        help='Use unified mating/speciation decision path')
    parser.add_argument('--unified-mate-head', action='store_true',
                        help='In unified mode, use learned pairwise head over mutator embeddings (model-owned)')
    parser.add_argument('--legacy-unified-affinity', action='store_true',
                        help='Deprecated: in unified mode, keep legacy mutator-affinity gating instead of pair-head')
    parser.add_argument('--no-compat-pretrain', action='store_true',
                        help='Disable compatibility-net pretraining (cold-start compat net)')
    parser.add_argument('--mate-choice', action='store_true',
                        help='Enable experimental mutual mate-choice selection')
    parser.add_argument('--mate-choice-topk', type=int, default=10,
                        help='Top-k by fitness used as mate-choice candidate pool')
    parser.add_argument('--mate-choice-threshold', type=float, default=0.0,
                        help='Mutual preference threshold for accepting a pair')
    parser.add_argument('--mate-choice-temperature', type=float, default=1.0,
                        help='Softmax temperature for mate-choice candidate sampling')
    parser.add_argument('--lineage-min-distance', type=int, default=0,
                        help='Minimum lineage distance (shared-ancestor hops) required for crossover; 0 disables.')
    parser.add_argument('--lineage-max-distance', type=int, default=-1,
                        help='Maximum lineage distance allowed for crossover; -1 disables upper bound.')
    parser.add_argument('--lineage-distance-depth', type=int, default=32,
                        help='Ancestor search depth used for lineage distance checks.')
    parser.add_argument('--lineage-assign-require-common-ancestor', action='store_true',
                        help='During species assignment, only compare against species members sharing a common ancestor within depth.')
    parser.add_argument('--lineage-assign-depth', type=int, default=32,
                        help='Ancestor search depth used for lineage-aware species assignment filtering.')
    args = parser.parse_args()
    if args.init_genome and args.init_mutator_from_genome:
        parser.error('Use only one of --init-genome or --init-mutator-from-genome')
    if args.unified_mate_head and not args.unified_mating:
        parser.error('--unified-mate-head requires --unified-mating')
    if args.unified_mate_head and args.legacy_unified_affinity:
        parser.error('Choose only one of --unified-mate-head or --legacy-unified-affinity')
    effective_unified_mate_head = bool(args.unified_mating and (args.unified_mate_head or not args.legacy_unified_affinity))
    if args.unified_mating and (not args.unified_mate_head) and (not args.legacy_unified_affinity):
        print("Unified mating default: enabling mutator-owned pair-head (use --legacy-unified-affinity to keep old affinity mode)")
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
            try:
                progress_sinks.append(DiscordTqdmSink(train_cfg.generations, token, args.discord_channel_id,
                                                         pop_size=train_cfg.pop_size))
                print(f"Discord tqdm: enabled (channel {args.discord_channel_id})")
            except Exception as e:
                print(f"Discord tqdm: init failed ({e}), trying REST fallback")
                try:
                    progress_sinks.append(DiscordMessageSink(token, args.discord_channel_id))
                    print(f"Discord REST fallback: enabled")
                except Exception as e2:
                    print(f"Discord progress: all methods failed ({e2})")
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
            learn_compat_threshold=args.learn_compat_threshold,
            compat_binary=args.compat_binary,
            compat_pretrain=(not args.no_compat_pretrain),
            unified_mating=args.unified_mating,
            unified_mate_head=effective_unified_mate_head,
            compat_rate_guardrail=train_cfg.compat_rate_guardrail,
            compat_rate_target_low=train_cfg.compat_rate_target_low,
            compat_rate_target_high=train_cfg.compat_rate_target_high,
            compat_rate_adjust=train_cfg.compat_rate_adjust,
            flex=train_cfg.flex,
            policy_arch=train_cfg.policy_arch,
            complexity_cost=train_cfg.complexity_cost,
            output_dir=train_cfg.output,
            n_workers=train_cfg.workers,
            fleet=fleet,
            progress_callback=progress_callback,
            mutator_kwargs=mut_cfg.to_kwargs(),
            init_genome_path=args.init_genome,
            init_mutator_from_genome_path=args.init_mutator_from_genome,
            mutation_decay=args.mutation_decay,
            mate_choice=train_cfg.mate_choice,
            mate_choice_topk=train_cfg.mate_choice_topk,
            mate_choice_threshold=train_cfg.mate_choice_threshold,
            mate_choice_temperature=train_cfg.mate_choice_temperature,
            lineage_min_distance=args.lineage_min_distance,
            lineage_max_distance=args.lineage_max_distance,
            lineage_distance_depth=args.lineage_distance_depth,
            lineage_assign_require_common_ancestor=args.lineage_assign_require_common_ancestor,
            lineage_assign_depth=args.lineage_assign_depth,
            survivor_fraction=args.survivor_fraction,
        )
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Final best:  {history['best'][-1]:.2f}")
    print(f"Final mean:  {history['mean'][-1]:.2f}")
    print(f"Final mean fidelity: {history['mean_fidelity'][-1]:.4f}")
    print(f"Final mean drift: {history['mean_mutator_drift'][-1]:.4f}")
    if history.get('mean_policy_l2'):
        print(f"Final policy ||w||₂ (mean/best): {history['mean_policy_l2'][-1]:.3f}/{history['best_policy_l2'][-1]:.3f}")
        print(f"Final policy |w| (mean/best): {history['mean_policy_abs'][-1]:.5f}/{history['best_policy_abs'][-1]:.5f}")
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
        'learn_compat_threshold': args.learn_compat_threshold,
        'compat_binary': args.compat_binary,
        'compat_pretrain': (not args.no_compat_pretrain),
        'unified_mating': args.unified_mating,
        'unified_mate_head': effective_unified_mate_head,
        'legacy_unified_affinity': args.legacy_unified_affinity,
        'compat_rate_guardrail': train_cfg.compat_rate_guardrail,
        'compat_rate_target_low': train_cfg.compat_rate_target_low,
        'compat_rate_target_high': train_cfg.compat_rate_target_high,
        'compat_rate_adjust': train_cfg.compat_rate_adjust,
        'pop_size': train_cfg.pop_size,
        'generations': train_cfg.generations,
        'seed': train_cfg.seed,
        'init_genome': args.init_genome,
        'init_mutator_from_genome': args.init_mutator_from_genome,
        'mutation_decay': args.mutation_decay,
        'mate_choice': train_cfg.mate_choice,
        'mate_choice_topk': train_cfg.mate_choice_topk,
        'mate_choice_threshold': train_cfg.mate_choice_threshold,
        'mate_choice_temperature': train_cfg.mate_choice_temperature,
        'survivor_fraction': args.survivor_fraction,
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
    has_weight_scale = bool(history.get('mean_policy_l2'))
    n_plots = 3 + int(has_weight_scale) + int(args.speciation) + int(has_complexity)
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

    # Policy weight scale
    plot_idx = 3
    if has_weight_scale:
        ax = axes[plot_idx]
        ax.plot(gens, history['mean_policy_l2'], label='Mean ||w||₂', color='slateblue')
        ax.plot(gens, history['best_policy_l2'], label='Best-genome ||w||₂', color='indigo', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(gens, history['mean_policy_abs'], label='Mean |w|', color='darkorange', linestyle='--', alpha=0.8)
        ax2.plot(gens, history['best_policy_abs'], label='Best-genome |w|', color='orangered', linestyle='-.', alpha=0.8)
        ax.set_xlabel('Generation')
        ax.set_ylabel('L2 norm')
        ax2.set_ylabel('Mean absolute weight')
        ax.set_title('Policy Weight Scale (track implicit shrinkage/expansion)')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Parameter count
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

    lineage_info = history.get('lineage_artifacts', {})
    lineage_path = lineage_info.get('lineage_jsonl')
    lineage_meta = lineage_info.get('lineage_meta_json')
    if lineage_path and os.path.exists(lineage_path):
        color_mode = 'species' if train_cfg.speciation else 'method'
        lineage_plot = generate_lineage_plots(
            lineage_path=lineage_path,
            meta_path=lineage_meta,
            out_prefix=os.path.join(train_cfg.output, 'lineage_tree'),
            color_by=color_mode,
            formats=('png', 'svg'),
        )
        print(f"Lineage DOT saved: {lineage_plot['dot_path']}")
        for rendered in lineage_plot.get('rendered_paths', []):
            print(f"Lineage plot saved: {rendered}")
        if lineage_plot.get('warning'):
            print(f"Lineage plot warning: {lineage_plot['warning']}")


if __name__ == '__main__':
    main()
