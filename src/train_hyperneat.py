#!/usr/bin/env python3

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from .hyperneat_evolution import run_hyperneat_evolution
from .progress import DiscordMessageSink, DiscordTqdmSink, ProgressFileSink, ProgressReporter


def _load_discord_token_from_openclaw() -> str | None:
    try:
        cfg_path = os.path.expanduser("~/.openclaw/openclaw.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("channels", {}).get("discord", {}).get("token")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="HyperNEAT Evolution")
    parser.add_argument("--env", default="LunarLander-v3", help="Gymnasium environment")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    parser.add_argument("--generations", type=int, default=300, help="Number of generations")
    parser.add_argument("--episodes", type=int, default=10, help="Eval episodes per genome")
    parser.add_argument("--crossover-rate", type=float, default=0.3, help="Crossover probability")
    parser.add_argument("--elitism", type=float, default=0.10, help="Elitism fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--species-threshold", type=float, default=3.0, help="CPPN species distance threshold")
    parser.add_argument("--use-expression-output", action="store_true", help="Enable CPPN link expression output")
    parser.add_argument("--expression-threshold", type=float, default=None, help="Link expression threshold")
    parser.add_argument("--survival-threshold", type=float, default=0.2, help="Fraction of species that reproduces")
    parser.add_argument("--max-stagnation", type=int, default=20, help="Gens before stagnant species killed")
    parser.add_argument("--species-elitism", type=int, default=2, help="Top N species protected from stagnation")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--discord-channel-id", type=int, default=1471943945540866139,
                        help="Discord channel for built-in tqdm progress")
    parser.add_argument("--no-discord-tqdm", action="store_true", help="Disable Discord tqdm progress updates")
    parser.add_argument("--fleet", action="store_true", help="Enable distributed fleet evaluation")
    parser.add_argument("--fleet-port", type=int, default=5555, help="Fleet manager port")
    parser.add_argument("--fleet-workers", type=int, default=1, help="Min fleet workers to wait for")
    parser.add_argument("--fleet-authkey", default="neuralfleet", help="Fleet auth key")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"{'=' * 60}")
    print("HyperNEAT Evolution")
    print(f"{'=' * 60}")
    print(f"Environment: {args.env}")
    print(f"Population: {args.pop_size}, Generations: {args.generations}, Episodes: {args.episodes}")

    progress_sinks = [ProgressFileSink(node_name=os.uname().nodename, seed=args.seed)]
    if not args.no_discord_tqdm:
        token = os.getenv("TQDM_DISCORD_TOKEN") or _load_discord_token_from_openclaw()
        if token and args.discord_channel_id:
            try:
                progress_sinks.append(
                    DiscordTqdmSink(args.generations, token, args.discord_channel_id, pop_size=args.pop_size)
                )
                print(f"Discord tqdm: enabled (channel {args.discord_channel_id})")
            except Exception as e:
                print(f"Discord tqdm: init failed ({e}), trying REST fallback")
                try:
                    progress_sinks.append(DiscordMessageSink(token, args.discord_channel_id))
                    print("Discord REST fallback: enabled")
                except Exception as e2:
                    print(f"Discord progress: all methods failed ({e2})")
        else:
            print("Discord tqdm: disabled (missing token/channel)")

    progress_callback = ProgressReporter(progress_sinks)

    fleet = None
    if args.fleet:
        from fleet.manager import FleetEvaluator
        fleet = FleetEvaluator(
            port=args.fleet_port,
            authkey=args.fleet_authkey.encode(),
            min_workers=args.fleet_workers,
        )
        print(f"Fleet evaluation: port {args.fleet_port} (min workers: {args.fleet_workers})")

    start = time.time()
    history = run_hyperneat_evolution(
        env_id=args.env,
        pop_size=args.pop_size,
        generations=args.generations,
        n_eval_episodes=args.episodes,
        crossover_rate=args.crossover_rate,
        elitism_frac=args.elitism,
        seed=args.seed,
        n_workers=args.workers,
        species_distance_threshold=args.species_threshold,
        use_expression_output=args.use_expression_output,
        expression_threshold=args.expression_threshold,
        progress_callback=progress_callback,
        fleet=fleet,
        survival_threshold=args.survival_threshold,
        max_stagnation=args.max_stagnation,
        species_elitism=args.species_elitism,
    )
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Final best: {history['best'][-1]:.2f}")
    print(f"Final mean: {history['mean'][-1]:.2f}")
    print(f"Final species count: {history['num_species'][-1]}")

    tag = f"{args.env}_hyperneat_s{args.seed}"
    result = {
        "env": args.env,
        "algorithm": "hyperneat",
        "pop_size": args.pop_size,
        "generations": args.generations,
        "episodes": args.episodes,
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "history": history,
    }

    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    json_path = os.path.join(args.output, f"{tag}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=json_safe)
    print(f"Results saved: {json_path}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    gens = range(len(history["best"]))

    ax = axes[0]
    ax.plot(gens, history["best"], label="Best", color="green", linewidth=2)
    ax.plot(gens, history["mean"], label="Mean", color="blue", linewidth=1.5)
    ax.fill_between(gens, history["worst"], history["best"], alpha=0.12, color="blue")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{args.env} | HyperNEAT | pop {args.pop_size}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gens, history["num_species"], label="Num Species", color="teal", linewidth=2)
    if history.get("species_entropy"):
        ax.plot(gens, history["species_entropy"], label="Species Entropy", color="navy", alpha=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Species")
    ax.set_title("Speciation Dynamics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output, f"{tag}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
