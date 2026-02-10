"""
Evolutionary algorithm with true self-replication and quine-style fitness bonus.

Features:
- Tournament selection
- Self-replication: mutator processes full genome (policy + mutator weights)
- Quine lambda: fitness bonus for self-replication fidelity
- Tracks mutator drift and fidelity metrics per generation
"""

import torch
import numpy as np
import gymnasium as gym
from typing import List, Dict
from .genome import Genome, Policy, ChunkMutator, TransformerMutator, GaussianMutator


def evaluate_genome(genome: Genome, env_id: str, n_episodes: int = 5,
                    max_steps: int = 1000) -> float:
    """Evaluate a genome's policy on an environment. Returns mean reward."""
    env = gym.make(env_id)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = genome.policy.act(obs)
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(np.argmax(action))
            else:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards))


def create_population(pop_size: int, obs_dim: int, act_dim: int,
                      mutator_type: str = 'chunk', hidden: int = 64,
                      chunk_size: int = 64) -> List[Genome]:
    """Create initial population."""
    population = []
    for _ in range(pop_size):
        policy = Policy(obs_dim, act_dim, hidden)
        if mutator_type == 'chunk':
            mutator = ChunkMutator(chunk_size=chunk_size)
        elif mutator_type == 'transformer':
            mutator = TransformerMutator(chunk_size=chunk_size)
        elif mutator_type == 'gaussian':
            mutator = GaussianMutator()
        else:
            raise ValueError(f"Unknown mutator type: {mutator_type}")
        population.append(Genome(policy, mutator, mutator_type))
    return population


def tournament_select(population: List[Genome], k: int = 3) -> Genome:
    contestants = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(contestants, key=lambda i: population[i].fitness)
    return population[best]


def evolve_generation(population: List[Genome], crossover_rate: float = 0.3,
                      elitism: int = 5, generation: int = 0,
                      max_generations: int = 100) -> List[Genome]:
    pop_size = len(population)
    ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
    new_population = []

    # Elitism
    for i in range(min(elitism, pop_size)):
        elite = Genome(
            policy=ranked[i]._clone_policy(),
            mutator=ranked[i]._clone_mutator(),
            mutator_type=ranked[i].mutator_type,
        )
        elite.set_flat_weights(ranked[i].get_flat_weights())
        elite.fitness = ranked[i].fitness
        elite.self_replication_fidelity = ranked[i].self_replication_fidelity
        new_population.append(elite)

    while len(new_population) < pop_size:
        parent = tournament_select(ranked)
        if np.random.random() < crossover_rate:
            other = tournament_select(ranked)
            try:
                child = parent.crossover(other, generation, max_generations)
            except Exception:
                child = parent.reproduce(generation, max_generations)
        else:
            try:
                child = parent.reproduce(generation, max_generations)
            except Exception:
                child = Genome(
                    policy=parent._clone_policy(),
                    mutator=parent._clone_mutator(),
                    mutator_type=parent.mutator_type,
                )
                flat = parent.get_flat_weights()
                noise = torch.randn_like(flat) * 0.01
                child.set_flat_weights(flat + noise)
        new_population.append(child)

    return new_population


def run_evolution(env_id: str = 'CartPole-v1', pop_size: int = 30,
                  generations: int = 100, mutator_type: str = 'chunk',
                  n_eval_episodes: int = 5, crossover_rate: float = 0.3,
                  hidden: int = 64, chunk_size: int = 64,
                  log_interval: int = 1, seed: int = 42,
                  quine_lambda: float = 0.0) -> Dict:
    """
    Main evolution loop with quine-style self-replication tracking.

    quine_lambda: weight for self-replication fidelity bonus in fitness.
                  Higher = more selection pressure for stable self-replication.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]
    env.close()

    print(f"Environment: {env_id} (obs={obs_dim}, act={act_dim})")

    population = create_population(pop_size, obs_dim, act_dim, mutator_type,
                                   hidden, chunk_size)
    genome = population[0]
    print(f"Genome: {genome.num_policy_params()} policy params + "
          f"{genome.num_mutator_params()} mutator params = "
          f"{genome.num_total_params()} total")
    print(f"Mutator type: {mutator_type}, Quine lambda: {quine_lambda}")
    print(f"Population: {pop_size}, Generations: {generations}")
    print()

    # Track initial mutator weights for drift measurement
    initial_mutator_weights = {}
    for i, g in enumerate(population):
        initial_mutator_weights[i] = g.get_flat_mutator_weights().clone()

    history = {
        'best': [], 'mean': [], 'worst': [],
        'mean_fidelity': [],  # mean self-replication fidelity (lower = better)
        'best_fidelity': [],  # best genome's fidelity
        'mean_mutator_drift': [],  # mean L2 drift from initial mutator weights
        'max_mutator_drift': [],
    }

    for gen in range(generations):
        # Evaluate raw fitness
        for g in population:
            g.fitness = evaluate_genome(g, env_id, n_eval_episodes)

        # Compute quine bonus: reward low self_replication_fidelity
        # Fidelity is L2 distance (lower = better), so bonus = -fidelity
        if quine_lambda > 0:
            fidelities = [g.self_replication_fidelity for g in population]
            max_fid = max(fidelities) if max(fidelities) > 0 else 1.0
            for g in population:
                # Normalize fidelity bonus to [0, 1] range, invert so lower distance = higher bonus
                quine_bonus = (1.0 - g.self_replication_fidelity / max_fid)
                # Scale bonus relative to typical fitness range
                g.fitness += quine_lambda * quine_bonus * 100.0  # CartPole max ~500

        fitnesses = [g.fitness for g in population]
        fidelities = [g.self_replication_fidelity for g in population]

        best_idx = np.argmax(fitnesses)
        best_fit = fitnesses[best_idx]
        mean_fit = np.mean(fitnesses)
        worst_fit = min(fitnesses)

        # Compute mutator drift from generation 0
        drifts = []
        for i, g in enumerate(population):
            current_mutator = g.get_flat_mutator_weights()
            # Compare to closest initial (use index mod initial size)
            ref_idx = i % len(initial_mutator_weights)
            drift = torch.norm(current_mutator - initial_mutator_weights[ref_idx]).item()
            drifts.append(drift)

        history['best'].append(best_fit)
        history['mean'].append(float(mean_fit))
        history['worst'].append(worst_fit)
        history['mean_fidelity'].append(float(np.mean(fidelities)))
        history['best_fidelity'].append(fidelities[best_idx])
        history['mean_mutator_drift'].append(float(np.mean(drifts)))
        history['max_mutator_drift'].append(float(np.max(drifts)))

        if gen % log_interval == 0:
            print(f"Gen {gen:4d} | Best: {best_fit:8.2f} | Mean: {mean_fit:8.2f} | "
                  f"Fidelity: {np.mean(fidelities):.4f} | Drift: {np.mean(drifts):.4f}")

        population = evolve_generation(population, crossover_rate,
                                       elitism=5, generation=gen,
                                       max_generations=generations)

    return history
