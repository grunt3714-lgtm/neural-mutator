from __future__ import annotations

import time
import multiprocessing as _mp
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from .cppn import CPPN
from .hyperneat import build_hyperneat_policy, Substrate
from .snake_env import ensure_snake_registered


_Pool = _mp.Pool
_worker_pool = None
_worker_pool_size = 0


@dataclass
class HyperNEATGenome:
    cppn: CPPN
    fitness: float = float("-inf")
    species_id: int = -1

    def copy(self) -> "HyperNEATGenome":
        return HyperNEATGenome(cppn=self.cppn.copy(), fitness=self.fitness, species_id=self.species_id)

    def to_bytes(self) -> bytes:
        return self.cppn.to_bytes()

    @classmethod
    def from_bytes(cls, blob: bytes) -> "HyperNEATGenome":
        return cls(cppn=CPPN.from_bytes(blob))


@dataclass
class SpeciesRecord:
    species_id: int
    representative: HyperNEATGenome
    members: List[HyperNEATGenome] = field(default_factory=list)
    best_fitness: float = float("-inf")
    stagnation_gens: int = 0


def _line_coords(n: int, y: float, z: float):
    if n <= 1:
        xs = [0.0]
    else:
        xs = np.linspace(-1.0, 1.0, n)
    return [(float(x), float(y), float(z)) for x in xs]


def _build_substrate(input_dim: int, output_dim: int) -> Substrate:
    # Slightly wider hidden layers for harder tasks like CarRacing.
    h = 32 if input_dim > 32 else 16
    return Substrate(
        input_coords=_line_coords(input_dim, y=-1.0, z=0.0),
        hidden1_coords=_line_coords(h, y=-0.33, z=0.0),
        hidden2_coords=_line_coords(h, y=0.33, z=0.0),
        output_coords=_line_coords(output_dim, y=1.0, z=0.0),
    )


def _obs_to_vector(obs) -> np.ndarray:
    # Gym wrappers may return dict observations (e.g., VizDoom).
    if isinstance(obs, dict):
        if 'screen' in obs:
            obs = obs['screen']
        elif 'observation' in obs:
            obs = obs['observation']
        else:
            # fallback: first value
            obs = next(iter(obs.values()))

    arr = np.asarray(obs)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # Image obs (e.g., CarRacing/VizDoom): grayscale + 8x8 average pooling => 64 dims.
        if arr.shape[-1] in (3, 4):
            gray = arr[..., :3].mean(axis=2).astype(np.float32)
        else:
            gray = arr.mean(axis=2).astype(np.float32)
        h, w = gray.shape
        bh = max(1, h // 8)
        bw = max(1, w // 8)
        gray = gray[:bh * 8, :bw * 8]
        pooled = gray.reshape(8, bh, 8, bw).mean(axis=(1, 3))
        return (pooled / 255.0).reshape(-1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


# ── Evaluation ────────────────────────────────────────────────────


def evaluate_genome(
    cppn: CPPN,
    env_id: str,
    n_episodes: int = 5,
    max_steps: int = 1000,
    seeds=None,
    expression_threshold: float | None = None,
) -> tuple[float, int]:
    ensure_snake_registered()
    if env_id.startswith("Vizdoom"):
        import vizdoom.gymnasium_wrapper  # registers VizDoom env ids
    env = gym.make(env_id)
    rewards = []
    total_steps = 0

    # Build substrate from first observation and action-space shape.
    first_obs, _ = env.reset()
    first_vec = _obs_to_vector(first_obs)
    if isinstance(env.action_space, gym.spaces.Discrete):
        out_dim = int(env.action_space.n)
    else:
        out_dim = int(np.prod(env.action_space.shape))
    substrate = _build_substrate(int(first_vec.shape[0]), out_dim)
    policy = build_hyperneat_policy(cppn, substrate=substrate, expression_threshold=expression_threshold)

    for ep in range(n_episodes):
        seed = seeds[ep] if seeds and ep < len(seeds) else None
        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            obs_vec = _obs_to_vector(obs)
            action = policy.act(obs_vec)
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(np.argmax(action))
            else:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if terminated or truncated:
                break
        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards)), int(total_steps)


def _eval_worker(args):
    cppn_bytes, env_id, n_episodes, max_steps, expression_threshold = args
    cppn = CPPN.from_bytes(cppn_bytes)
    return evaluate_genome(cppn, env_id, n_episodes, max_steps, expression_threshold=expression_threshold)


def _get_pool(n_workers: int):
    global _worker_pool, _worker_pool_size
    if _worker_pool is None or _worker_pool_size != n_workers:
        if _worker_pool is not None:
            _worker_pool.terminate()
        _worker_pool = _Pool(n_workers)
        _worker_pool_size = n_workers
    return _worker_pool


def evaluate_population(
    population: List[HyperNEATGenome],
    env_id: str,
    n_episodes: int = 5,
    max_steps: int = 1000,
    n_workers: int = 1,
    expression_threshold: float | None = None,
    fleet=None,
    genome_callback=None,
) -> tuple[List[float], Dict]:
    t0 = time.time()

    if fleet is not None:
        genomes_bytes = [g.to_bytes() for g in population]
        fleet.dispatch(genomes_bytes, env_id, n_episodes, max_steps)
        del genomes_bytes
        import gc; gc.collect()
        return fleet.collect(genome_callback=genome_callback)

    if n_workers <= 1:
        pairs = [
            evaluate_genome(g.cppn, env_id, n_episodes, max_steps, expression_threshold=expression_threshold)
            for g in population
        ]
        fitnesses = [p[0] for p in pairs]
        steps = int(sum(p[1] for p in pairs))
        elapsed = time.time() - t0
        profile = {
            "dispatch_sec": 0.0, "remote_eval_sec": elapsed,
            "result_gather_sec": 0.0, "eval_total_sec": elapsed,
            "env_steps": steps, "genomes_evaluated": len(population),
        }
        return fitnesses, profile

    t_dispatch0 = time.time()
    args = [
        (g.to_bytes(), env_id, n_episodes, max_steps, expression_threshold)
        for g in population
    ]
    pool = _get_pool(n_workers)
    dispatch_sec = time.time() - t_dispatch0

    t_eval0 = time.time()
    pairs = pool.map(_eval_worker, args)
    remote_eval_sec = time.time() - t_eval0

    fitnesses = [p[0] for p in pairs]
    steps = int(sum(p[1] for p in pairs))
    elapsed = time.time() - t0
    profile = {
        "dispatch_sec": dispatch_sec, "remote_eval_sec": remote_eval_sec,
        "result_gather_sec": 0.0, "eval_total_sec": elapsed,
        "env_steps": steps, "genomes_evaluated": len(population),
    }
    return fitnesses, profile


def create_population(
    pop_size: int,
    use_expression_output: bool = False,
    weight_scale: float = 2.0,
) -> List[HyperNEATGenome]:
    return [
        HyperNEATGenome(cppn=CPPN(n_inputs=8, n_outputs=1,
                                   use_expression_output=use_expression_output,
                                   weight_scale=weight_scale))
        for _ in range(pop_size)
    ]


# ── Speciation (persistent, with stagnation) ─────────────────────


def speciate(
    population: List[HyperNEATGenome],
    prev_species: List[SpeciesRecord],
    distance_threshold: float,
    max_stagnation: int = 20,
    species_elitism: int = 2,
) -> tuple[List[SpeciesRecord], Dict]:
    """Assign genomes to species using persistent representatives.

    Species that stagnate for max_stagnation gens are killed (except
    the top species_elitism species by best-ever fitness).
    """
    # Start with empty member lists, keep representatives from last gen
    new_species: List[SpeciesRecord] = []
    for sp in prev_species:
        new_sp = SpeciesRecord(
            species_id=sp.species_id,
            representative=sp.representative,
            members=[],
            best_fitness=sp.best_fitness,
            stagnation_gens=sp.stagnation_gens,
        )
        new_species.append(new_sp)

    unplaced: List[HyperNEATGenome] = []
    next_id = max((sp.species_id for sp in new_species), default=-1) + 1

    for genome in population:
        placed = False
        for sp in new_species:
            if genome.cppn.distance(sp.representative.cppn) <= distance_threshold:
                genome.species_id = sp.species_id
                sp.members.append(genome)
                placed = True
                break
        if not placed:
            # New species
            genome.species_id = next_id
            sp = SpeciesRecord(
                species_id=next_id,
                representative=genome,
                members=[genome],
                best_fitness=float("-inf"),
                stagnation_gens=0,
            )
            new_species.append(sp)
            next_id += 1

    # Remove empty species
    new_species = [sp for sp in new_species if sp.members]

    # Update stagnation + best fitness per species
    for sp in new_species:
        sp_best = max(g.fitness for g in sp.members)
        if sp_best > sp.best_fitness + 1e-6:
            sp.best_fitness = sp_best
            sp.stagnation_gens = 0
        else:
            sp.stagnation_gens += 1
        # Pick new representative (random member)
        sp.representative = np.random.choice(sp.members).copy()

    # Kill stagnant species (but protect top species_elitism by best_fitness)
    if len(new_species) > species_elitism:
        sorted_by_best = sorted(new_species, key=lambda s: s.best_fitness, reverse=True)
        protected = set(s.species_id for s in sorted_by_best[:species_elitism])
        new_species = [
            sp for sp in new_species
            if sp.species_id in protected or sp.stagnation_gens < max_stagnation
        ]

    # Stats
    sizes = {sp.species_id: len(sp.members) for sp in new_species}
    total = max(1, sum(sizes.values()))
    probs = np.asarray([s / total for s in sizes.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if len(probs) > 0 else 0.0
    stag_counts = {sp.species_id: sp.stagnation_gens for sp in new_species}

    return new_species, {
        "num_species": len(new_species),
        "species_sizes": sizes,
        "species_entropy": entropy,
        "stagnation": stag_counts,
    }


# ── Selection & reproduction ──────────────────────────────────────


def tournament_select(members: List[HyperNEATGenome], k: int = 3) -> HyperNEATGenome:
    contestants = np.random.choice(len(members), size=min(k, len(members)), replace=False)
    best = max(contestants, key=lambda i: members[i].fitness)
    return members[int(best)]


def evolve_generation(
    population: List[HyperNEATGenome],
    prev_species: List[SpeciesRecord],
    distance_threshold: float,
    crossover_rate: float = 0.3,
    elitism_frac: float = 0.10,
    survival_threshold: float = 0.2,
    max_stagnation: int = 20,
    species_elitism: int = 2,
) -> tuple[List[HyperNEATGenome], List[SpeciesRecord], Dict]:
    pop_size = len(population)

    # Speciate
    species_list, species_info = speciate(
        population, prev_species, distance_threshold,
        max_stagnation=max_stagnation,
        species_elitism=species_elitism,
    )

    if not species_list:
        # Extinction event — re-seed
        print("[hyperneat] Total extinction! Re-seeding population.")
        new_pop = create_population(pop_size)
        return new_pop, [], species_info

    # Adjusted fitness (explicit fitness sharing)
    for sp in species_list:
        sp_size = len(sp.members)
        for g in sp.members:
            g.fitness = g.fitness / max(1, sp_size)

    # Compute offspring allocation proportional to total adjusted fitness per species
    total_adj = sum(max(0.0, sum(g.fitness for g in sp.members)) for sp in species_list)
    if total_adj <= 0:
        total_adj = 1.0

    offspring_counts = {}
    remaining = pop_size
    for sp in species_list:
        sp_total = max(0.0, sum(g.fitness for g in sp.members))
        count = max(1, int(round((sp_total / total_adj) * pop_size)))
        offspring_counts[sp.species_id] = count
        remaining -= count

    # Distribute remainder
    if remaining > 0:
        best_sp = max(species_list, key=lambda s: s.best_fitness)
        offspring_counts[best_sp.species_id] += remaining
    elif remaining < 0:
        # Trim from largest species
        for sp in sorted(species_list, key=lambda s: offspring_counts[s.species_id], reverse=True):
            if remaining >= 0:
                break
            trim = min(-remaining, offspring_counts[sp.species_id] - 1)
            offspring_counts[sp.species_id] -= trim
            remaining += trim

    next_pop: List[HyperNEATGenome] = []

    for sp in species_list:
        n_offspring = offspring_counts.get(sp.species_id, 0)
        if n_offspring <= 0:
            continue

        # Sort members by raw fitness (undo sharing for selection)
        members_sorted = sorted(sp.members, key=lambda g: g.fitness, reverse=True)

        # Species elitism: copy best member
        n_elite = max(1, int(round(n_offspring * elitism_frac)))
        for i in range(min(n_elite, len(members_sorted))):
            next_pop.append(members_sorted[i].copy())

        # Survival threshold: only top fraction reproduces
        n_survivors = max(1, int(round(len(members_sorted) * survival_threshold)))
        breeders = members_sorted[:n_survivors]

        while len(next_pop) < sum(offspring_counts.get(s.species_id, 0)
                                   for s in species_list[:species_list.index(sp) + 1]):
            p1 = tournament_select(breeders)
            child_cppn = p1.cppn.copy()

            if np.random.random() < crossover_rate and len(breeders) > 1:
                p2 = tournament_select(breeders)
                child_cppn = p1.cppn.crossover(p2.cppn, self_fitness=p1.fitness, other_fitness=p2.fitness)

            child_cppn.mutate()
            next_pop.append(HyperNEATGenome(cppn=child_cppn))

    # Trim to exact pop_size
    next_pop = next_pop[:pop_size]
    # Pad if needed
    while len(next_pop) < pop_size:
        sp = np.random.choice(species_list)
        p = tournament_select(sp.members)
        child = p.cppn.copy()
        child.mutate()
        next_pop.append(HyperNEATGenome(cppn=child))

    return next_pop, species_list, species_info


# ── Main evolution loop ───────────────────────────────────────────


def run_hyperneat_evolution(
    env_id: str = "LunarLander-v3",
    pop_size: int = 100,
    generations: int = 300,
    n_eval_episodes: int = 10,
    crossover_rate: float = 0.3,
    elitism_frac: float = 0.10,
    seed: int = 42,
    n_workers: int = 1,
    species_distance_threshold: float = 3.0,
    use_expression_output: bool = False,
    expression_threshold: float | None = None,
    progress_callback=None,
    log_interval: int = 1,
    fleet=None,
    survival_threshold: float = 0.2,
    max_stagnation: int = 20,
    species_elitism: int = 2,
) -> Dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    ensure_snake_registered()

    if env_id.startswith("Vizdoom"):
        import vizdoom.gymnasium_wrapper  # registers VizDoom env ids
    env = gym.make(env_id)
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
        act_kind = "discrete"
    else:
        act_dim = int(np.prod(env.action_space.shape))
        act_kind = "continuous"
    env.close()

    print(f"[hyperneat] env={env_id} obs_shape={obs_shape} action_dim={act_dim} ({act_kind})")

    population = create_population(
        pop_size=pop_size,
        use_expression_output=use_expression_output,
        weight_scale=2.0,
    )

    prev_species: List[SpeciesRecord] = []

    history = {
        "best": [], "mean": [], "worst": [],
        "num_species": [], "species_entropy": [],
        "gen_wall_sec": [],
        "dispatch_sec": [], "remote_eval_sec": [], "result_gather_sec": [],
        "evolution_step_sec": [], "logging_checkpoint_sec": [],
        "env_steps": [], "env_steps_per_sec": [],
        "genomes_evaluated": [], "genomes_per_sec": [],
    }

    best_ever = float("-inf")

    for gen in range(generations):
        gen_t0 = time.time()

        raw_fitnesses, eval_profile = evaluate_population(
            population, env_id,
            n_episodes=n_eval_episodes,
            n_workers=n_workers,
            expression_threshold=expression_threshold,
            fleet=fleet,
        )

        for g, fit in zip(population, raw_fitnesses):
            g.fitness = float(fit)

        fitnesses = np.asarray(raw_fitnesses, dtype=np.float64)
        best_fit = float(np.max(fitnesses))
        mean_fit = float(np.mean(fitnesses))
        worst_fit = float(np.min(fitnesses))

        t_log0 = time.time()
        history["best"].append(best_fit)
        history["mean"].append(mean_fit)
        history["worst"].append(worst_fit)

        best_ever = max(best_ever, best_fit)
        if progress_callback is not None:
            progress_callback(gen, generations, best_fit, mean_fit, best_ever)

        logging_checkpoint_sec = time.time() - t_log0

        t_evolve0 = time.time()
        population, prev_species, species_info = evolve_generation(
            population,
            prev_species,
            distance_threshold=species_distance_threshold,
            crossover_rate=crossover_rate,
            elitism_frac=elitism_frac,
            survival_threshold=survival_threshold,
            max_stagnation=max_stagnation,
            species_elitism=species_elitism,
        )
        evolution_step_sec = time.time() - t_evolve0

        history["num_species"].append(int(species_info["num_species"]))
        history["species_entropy"].append(float(species_info.get("species_entropy", 0.0)))

        gen_wall_sec = time.time() - gen_t0
        env_steps = int(eval_profile.get("env_steps", 0))
        genomes_evald = int(eval_profile.get("genomes_evaluated", len(population)))
        steps_per_sec = (env_steps / gen_wall_sec) if gen_wall_sec > 0 else 0.0
        genomes_per_sec = (genomes_evald / gen_wall_sec) if gen_wall_sec > 0 else 0.0

        history["gen_wall_sec"].append(float(gen_wall_sec))
        history["dispatch_sec"].append(float(eval_profile.get("dispatch_sec", 0.0)))
        history["remote_eval_sec"].append(float(eval_profile.get("remote_eval_sec", 0.0)))
        history["result_gather_sec"].append(float(eval_profile.get("result_gather_sec", 0.0)))
        history["evolution_step_sec"].append(float(evolution_step_sec))
        history["logging_checkpoint_sec"].append(float(logging_checkpoint_sec))
        history["env_steps"].append(env_steps)
        history["env_steps_per_sec"].append(float(steps_per_sec))
        history["genomes_evaluated"].append(genomes_evald)
        history["genomes_per_sec"].append(float(genomes_per_sec))

        if gen % log_interval == 0:
            stag_str = ""
            if species_info.get("stagnation"):
                max_stag = max(species_info["stagnation"].values())
                stag_str = f" | MaxStag: {max_stag}"
            print(
                f"Gen {gen:4d} | Best: {best_fit:8.2f} | Mean: {mean_fit:8.2f} | "
                f"Worst: {worst_fit:8.2f} | Species: {species_info['num_species']}"
                f"{stag_str} | Perf: {steps_per_sec:8.1f} env_steps/s"
            )

    return history
