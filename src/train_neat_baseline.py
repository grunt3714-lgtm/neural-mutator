#!/usr/bin/env python3
import argparse
import json
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import neat
import numpy as np

from .progress import DiscordMessageSink, DiscordTqdmSink, ProgressFileSink, ProgressReporter


_ENV_ID = "LunarLander-v3"
_EPISODES = 10
_MAX_STEPS = 1000


def _load_discord_token_from_openclaw() -> str | None:
    try:
        cfg_path = os.path.expanduser("~/.openclaw/openclaw.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("channels", {}).get("discord", {}).get("token")
    except Exception:
        return None


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make(_ENV_ID)
    rewards = []
    for _ in range(_EPISODES):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(_MAX_STEPS):
            out = net.activate(obs)
            action = int(np.argmax(out))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


class NeatProgressReporter(neat.reporting.BaseReporter):
    def __init__(self, progress_cb, total_gens: int):
        self.progress_cb = progress_cb
        self.total_gens = total_gens
        self.gen = -1
        self.best_ever = float("-inf")

    def start_generation(self, generation):
        self.gen = int(generation)

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        if not fits:
            return
        best = float(np.max(fits))
        mean = float(np.mean(fits))
        self.best_ever = max(self.best_ever, best)
        if self.progress_cb is not None:
            self.progress_cb(self.gen, self.total_gens, best, mean, self.best_ever)


def main():
    p = argparse.ArgumentParser(description="Standard NEAT baseline run")
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--pop-size", type=int, default=100)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="src/neat_lunarlander.cfg")
    p.add_argument("--output", default="results/neat_lunar_p100_g300_s42")
    p.add_argument("--discord-channel-id", type=int, default=1469544182963110040)
    p.add_argument("--no-discord-tqdm", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )
    config.pop_size = args.pop_size

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # Progress sinks
    sinks = [ProgressFileSink(node_name=os.uname().nodename, seed=args.seed)]
    if not args.no_discord_tqdm:
        token = os.getenv("TQDM_DISCORD_TOKEN") or _load_discord_token_from_openclaw()
        if token and args.discord_channel_id:
            try:
                sinks.append(DiscordTqdmSink(args.generations, token, args.discord_channel_id, pop_size=0))
                print(f"Discord tqdm: enabled (channel {args.discord_channel_id})")
            except Exception as e:
                print(f"Discord tqdm: init failed ({e}), trying REST fallback")
                try:
                    sinks.append(DiscordMessageSink(token, args.discord_channel_id))
                    print("Discord REST fallback: enabled")
                except Exception as e2:
                    print(f"Discord progress: all methods failed ({e2})")
        else:
            print("Discord tqdm: disabled (missing token/channel)")

    progress_cb = ProgressReporter(sinks)
    pop.add_reporter(NeatProgressReporter(progress_cb, args.generations))

    global _ENV_ID, _EPISODES, _MAX_STEPS
    _ENV_ID = args.env
    _EPISODES = args.episodes
    _MAX_STEPS = args.max_steps

    pe = neat.ParallelEvaluator(args.workers, eval_genome)

    t0 = time.time()
    winner = pop.run(pe.evaluate, args.generations)
    elapsed = time.time() - t0

    best_hist = [float(g.fitness if g.fitness is not None else -9999) for g in stats.most_fit_genomes]
    mean_hist = [float(x) for x in stats.get_fitness_mean()]
    stdev_hist = [float(x) for x in stats.get_fitness_stdev()]
    worst_hist = [m - s for m, s in zip(mean_hist, stdev_hist)]

    result = {
        "algorithm": "neat",
        "env": args.env,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "episodes": args.episodes,
        "workers": args.workers,
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "winner_fitness": float(winner.fitness if winner.fitness is not None else -9999),
        "history": {
            "best": best_hist,
            "mean": mean_hist,
            "worst": worst_hist,
        },
    }

    json_path = os.path.join(args.output, f"{args.env}_neat_s{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    gens = range(len(best_hist))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best_hist, label="Best", color="green", linewidth=2)
    ax.plot(gens, mean_hist, label="Mean", color="blue", linewidth=1.5)
    ax.fill_between(gens, worst_hist, best_hist, alpha=0.12, color="blue")
    ax.axhline(200, color="red", linestyle="--", alpha=0.6, label="Solved=200")
    ax.set_title(f"NEAT baseline | {args.env} | pop={args.pop_size} | eps={args.episodes}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_path = os.path.join(args.output, f"{args.env}_neat_s{args.seed}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    print(f"Done in {elapsed:.1f}s")
    print(f"Winner fitness: {result['winner_fitness']:.2f}")
    print(f"Results: {json_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
