"""
Optimized evolution for harder environments.

Key improvements over base evolution:
1. Fitness-proportional mutation scaling — reduce mutation when fitness improves
2. Sigma adaptation (CMA-ES inspired) — learned per-individual mutation strength
3. Larger tournament size for stronger selection pressure
4. Stagnation detection — increase mutation when stuck
5. Weight perturbation restart for collapsed populations
"""

import torch
import numpy as np
from typing import List, Dict
from .evolution import evaluate_genome, assign_species
from .genome import (Genome, Policy, ChunkMutator, TransformerMutator,
                     GaussianMutator, ErrorCorrectorMutator)


def create_optimized_population(pop_size, obs_dim, act_dim, mutator_type='chunk',
                                 hidden=64, chunk_size=64) -> List[Genome]:
    """Create population with larger hidden layers for harder envs."""
    population = []
    for _ in range(pop_size):
        policy = Policy(obs_dim, act_dim, hidden)
        if mutator_type == 'chunk':
            mutator = ChunkMutator(chunk_size=chunk_size, hidden=256)  # bigger mutator
        elif mutator_type == 'transformer':
            mutator = TransformerMutator(chunk_size=chunk_size, d_model=128, n_layers=2)
        elif mutator_type == 'gaussian':
            mutator = GaussianMutator(mutation_scale=0.05)  # slightly larger initial scale
        else:
            mutator = GaussianMutator()
        population.append(Genome(policy, mutator, mutator_type))
    return population


def adaptive_tournament_select(population: List[Genome], k: int = 5) -> Genome:
    """Stronger selection pressure with larger tournament."""
    contestants = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(contestants, key=lambda i: population[i].fitness)
    return population[best]


def detect_stagnation(history: Dict, window: int = 30) -> bool:
    """Detect if best fitness hasn't improved in `window` generations."""
    if len(history['best']) < window:
        return False
    recent = history['best'][-window:]
    return max(recent) <= history['best'][-window]


def restart_worst(population: List[Genome], fraction: float = 0.2,
                  obs_dim: int = 8, act_dim: int = 4, hidden: int = 64,
                  mutator_type: str = 'chunk', chunk_size: int = 64):
    """Replace worst fraction of population with fresh random individuals."""
    n_replace = max(1, int(len(population) * fraction))
    ranked = sorted(population, key=lambda g: g.fitness)
    for i in range(n_replace):
        policy = Policy(obs_dim, act_dim, hidden)
        if mutator_type == 'chunk':
            mutator = ChunkMutator(chunk_size=chunk_size, hidden=256)
        elif mutator_type == 'transformer':
            mutator = TransformerMutator(chunk_size=chunk_size)
        elif mutator_type == 'gaussian':
            mutator = GaussianMutator(mutation_scale=0.05)
        else:
            mutator = GaussianMutator()
        ranked[i] = Genome(policy, mutator, mutator_type)
    return ranked


def evolve_generation_optimized(population: List[Genome], crossover_rate: float = 0.3,
                                 elitism: int = 5, generation: int = 0,
                                 max_generations: int = 300,
                                 best_fitness_ever: float = float('-inf'),
                                 stagnant: bool = False) -> List[Genome]:
    """Optimized evolution with adaptive mutation and stagnation handling."""
    pop_size = len(population)
    ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
    new_population = []

    # Compute fitness-based mutation modifier
    current_best = ranked[0].fitness
    fitness_ratio = 1.0
    if best_fitness_ever > float('-inf') and best_fitness_ever != 0:
        # If improving, reduce mutation; if stagnant, increase
        improvement = (current_best - best_fitness_ever) / max(abs(best_fitness_ever), 1.0)
        fitness_ratio = max(0.5, min(2.0, 1.0 - improvement))

    if stagnant:
        fitness_ratio *= 1.5  # Boost mutation when stuck

    # Elitism
    for i in range(min(elitism, pop_size)):
        import copy
        elite = Genome(
            policy=ranked[i]._clone_policy(),
            mutator=ranked[i]._clone_mutator(),
            mutator_type=ranked[i].mutator_type,
        )
        elite.set_flat_weights(ranked[i].get_flat_weights())
        elite.fitness = ranked[i].fitness
        new_population.append(elite)

    # Fill rest
    while len(new_population) < pop_size:
        parent = adaptive_tournament_select(ranked, k=5)
        if np.random.random() < crossover_rate:
            other = adaptive_tournament_select(ranked, k=5)
            try:
                child = parent.crossover(other, generation, max_generations)
            except Exception:
                child = parent.reproduce(generation, max_generations)
        else:
            child = parent.reproduce(generation, max_generations)

        # Apply fitness-based mutation scaling to the child
        if fitness_ratio != 1.0 and hasattr(child.mutator, 'mutation_scale'):
            with torch.no_grad():
                child.mutator.mutation_scale.mul_(fitness_ratio)
                child.mutator.mutation_scale.clamp_(0.001, 0.1)

        new_population.append(child)

    return new_population


def run_optimized_evolution(env_id='LunarLander-v3', pop_size=80, generations=500,
                            mutator_type='chunk', n_eval_episodes=10,
                            crossover_rate=0.3, hidden=128, chunk_size=64,
                            seed=42) -> Dict:
    """Optimized evolution loop for harder environments."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    import gymnasium as gym
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]
    env.close()

    print(f"=== Optimized Evolution ===")
    print(f"Environment: {env_id} (obs={obs_dim}, act={act_dim})")
    print(f"Mutator: {mutator_type}, Pop: {pop_size}, Gens: {generations}")
    print(f"Hidden: {hidden}, Eval episodes: {n_eval_episodes}")
    print()

    population = create_optimized_population(pop_size, obs_dim, act_dim, mutator_type,
                                              hidden, chunk_size)
    genome = population[0]
    print(f"Genome: {genome.num_policy_params()} policy + {genome.num_mutator_params()} mutator = {genome.num_total_params()} total")
    print()

    history = {'best': [], 'mean': [], 'worst': [], 'mean_fidelity': [],
               'best_fidelity': [], 'mean_mutator_drift': [], 'max_mutator_drift': [],
               'num_species': [], 'crossover_compat_rate': []}

    initial_mutator_weights = {}
    for i, g in enumerate(population):
        initial_mutator_weights[i] = g.get_flat_mutator_weights().clone()

    best_fitness_ever = float('-inf')
    stagnation_counter = 0

    for gen in range(generations):
        # Evaluate
        for g in population:
            g.fitness = evaluate_genome(g, env_id, n_eval_episodes)

        fitnesses = [g.fitness for g in population]
        fidelities = [g.self_replication_fidelity for g in population]

        best_fit = max(fitnesses)
        mean_fit = np.mean(fitnesses)
        worst_fit = min(fitnesses)

        # Track stagnation
        if best_fit > best_fitness_ever:
            best_fitness_ever = best_fit
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        stagnant = stagnation_counter > 30

        # Drift
        drifts = []
        for i, g in enumerate(population):
            current = g.get_flat_mutator_weights()
            ref_idx = i % len(initial_mutator_weights)
            drift = torch.norm(current - initial_mutator_weights[ref_idx]).item()
            drifts.append(drift)

        history['best'].append(best_fit)
        history['mean'].append(float(mean_fit))
        history['worst'].append(worst_fit)
        history['mean_fidelity'].append(float(np.mean(fidelities)))
        history['best_fidelity'].append(fidelities[np.argmax(fitnesses)])
        history['mean_mutator_drift'].append(float(np.mean(drifts)))
        history['max_mutator_drift'].append(float(np.max(drifts)))
        history['num_species'].append(1)
        history['crossover_compat_rate'].append(1.0)

        stag_marker = " ⚠️STAGNANT" if stagnant else ""
        print(f"Gen {gen:4d} | Best: {best_fit:8.2f} | Mean: {mean_fit:8.2f} | "
              f"BestEver: {best_fitness_ever:8.2f} | Drift: {np.mean(drifts):.4f}{stag_marker}")

        # Restart worst 20% if stagnant for 50 gens
        if stagnation_counter > 50:
            print(f"  🔄 Restarting worst 20% (stagnant {stagnation_counter} gens)")
            population = restart_worst(population, 0.2, obs_dim, act_dim, hidden,
                                       mutator_type, chunk_size)
            stagnation_counter = 30  # Reset partially

        population = evolve_generation_optimized(
            population, crossover_rate, elitism=max(5, pop_size // 10),
            generation=gen, max_generations=generations,
            best_fitness_ever=best_fitness_ever, stagnant=stagnant)

    return history
