#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import time
from multiprocessing import Queue
from multiprocessing.managers import BaseManager

import matplotlib.pyplot as plt
import neat
import numpy as np

from .progress import DiscordMessageSink, DiscordTqdmSink, ProgressFileSink, ProgressReporter


class FleetManager(BaseManager):
    pass


def _load_discord_token_from_openclaw() -> str | None:
    try:
        cfg_path = os.path.expanduser('~/.openclaw/openclaw.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get('channels', {}).get('discord', {}).get('token')
    except Exception:
        return None


class NeatProgressReporter(neat.reporting.BaseReporter):
    def __init__(self, progress_cb, total_gens: int):
        self.progress_cb = progress_cb
        self.total_gens = total_gens
        self.gen = -1
        self.best_ever = float('-inf')

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
    p = argparse.ArgumentParser(description='Standard NEAT with fleet worker pool')
    p.add_argument('--env', default='LunarLander-v3')
    p.add_argument('--generations', type=int, default=300)
    p.add_argument('--pop-size', type=int, default=100)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--config', default='src/neat_lunarlander.cfg')
    p.add_argument('--output', default='results/neat_lunar_fleet_p100_g300_s42')
    p.add_argument('--fleet-port', type=int, default=5556)
    p.add_argument('--fleet-authkey', default='neatfleet')
    p.add_argument('--fleet-workers', type=int, default=4)
    p.add_argument('--discord-channel-id', type=int, default=1469544182963110040)
    p.add_argument('--no-discord-tqdm', action='store_true')
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

    # Fleet manager queues
    work_q = Queue()
    result_q = Queue()
    register_q = Queue()
    FleetManager.register('get_work_queue', callable=lambda: work_q)
    FleetManager.register('get_result_queue', callable=lambda: result_q)
    FleetManager.register('get_register_queue', callable=lambda: register_q)
    mgr = FleetManager(address=('0.0.0.0', args.fleet_port), authkey=args.fleet_authkey.encode())
    mgr.start()

    workers = set()
    print(f'[neat-fleet] Waiting for {args.fleet_workers} workers on :{args.fleet_port} ...')
    while len(workers) < args.fleet_workers:
        try:
            workers.add(register_q.get(timeout=2))
            print(f'[neat-fleet] workers: {len(workers)}/{args.fleet_workers}')
        except Exception:
            pass

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    sinks = [ProgressFileSink(node_name=os.uname().nodename, seed=args.seed)]
    if not args.no_discord_tqdm:
        token = os.getenv('TQDM_DISCORD_TOKEN') or _load_discord_token_from_openclaw()
        if token and args.discord_channel_id:
            try:
                sinks.append(DiscordTqdmSink(args.generations, token, args.discord_channel_id, pop_size=args.pop_size))
            except Exception:
                try:
                    sinks.append(DiscordMessageSink(token, args.discord_channel_id))
                except Exception:
                    pass
    progress_cb = ProgressReporter(sinks)
    pop.add_reporter(NeatProgressReporter(progress_cb, args.generations))

    cfg_path_abs = os.path.abspath(args.config)

    def fleet_eval(genomes, _config):
        # genomes: list[(genome_id, genome)]
        for idx, (_, genome) in enumerate(genomes):
            gb = pickle.dumps(genome, protocol=pickle.HIGHEST_PROTOCOL)
            work_q.put((idx, gb, args.env, args.episodes, args.max_steps, cfg_path_abs))

        results = [None] * len(genomes)
        received = 0
        while received < len(genomes):
            idx, fit, _steps, _worker = result_q.get(timeout=600)
            if results[idx] is None:
                results[idx] = fit
                received += 1

        for idx, (_, genome) in enumerate(genomes):
            genome.fitness = float(results[idx])

    t0 = time.time()
    winner = pop.run(fleet_eval, args.generations)
    elapsed = time.time() - t0

    best_hist = [float(g.fitness if g.fitness is not None else -9999) for g in stats.most_fit_genomes]
    mean_hist = [float(x) for x in stats.get_fitness_mean()]
    stdev_hist = [float(x) for x in stats.get_fitness_stdev()]
    worst_hist = [m - s for m, s in zip(mean_hist, stdev_hist)]

    result = {
        'algorithm': 'neat-fleet',
        'env': args.env,
        'pop_size': args.pop_size,
        'generations': args.generations,
        'episodes': args.episodes,
        'seed': args.seed,
        'elapsed_sec': elapsed,
        'winner_fitness': float(winner.fitness if winner.fitness is not None else -9999),
        'history': {'best': best_hist, 'mean': mean_hist, 'worst': worst_hist},
    }

    json_path = os.path.join(args.output, f"{args.env}_neat_fleet_s{args.seed}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    gens = range(len(best_hist))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best_hist, label='Best', color='green', linewidth=2)
    ax.plot(gens, mean_hist, label='Mean', color='blue', linewidth=1.5)
    ax.fill_between(gens, worst_hist, best_hist, alpha=0.12, color='blue')
    ax.axhline(200, color='red', linestyle='--', alpha=0.6, label='Solved=200')
    ax.set_title(f"NEAT fleet | {args.env} | pop={args.pop_size} | eps={args.episodes}")
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_path = os.path.join(args.output, f"{args.env}_neat_fleet_s{args.seed}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')

    print(f'Done in {elapsed:.1f}s')
    print(f'Winner fitness: {result["winner_fitness"]:.2f}')
    print(f'Results: {json_path}')
    print(f'Plot: {plot_path}')

    # Shutdown workers
    for _ in range(len(workers) + 4):
        work_q.put(None)
    mgr.shutdown()


if __name__ == '__main__':
    main()
