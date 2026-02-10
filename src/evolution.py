"""
Evolutionary algorithm for neural mutator genomes.

Standard generational EA with:
- Tournament selection
- Self-replication via mutator networks
- Crossover by feeding one genome's policy into another's mutator
- Increased elitism and mutation scale decay for stability
"""

import torch
import numpy as np
import gymnasium as gym
from typing import List, Optional, Tuple
from .genome import Genome, Policy, ChunkMutator, TransformerMutator, CPPNMutator, GaussianMutator


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
    """Create initial population with random weights."""
    population = []
    for _ in range(pop_size):
        policy = Policy(obs_dim, act_dim, hidden)

        if mutator_type == 'chunk':
            mutator = ChunkMutator(chunk_size=chunk_size)
        elif mutator_type == 'transformer':
            mutator = TransformerMutator(chunk_size=chunk_size)
        elif mutator_type == 'cppn':
            mutator = CPPNMutator()
        elif mutator_type == 'gaussian':
            mutator = GaussianMutator()
        else:
            raise ValueError(f"Unknown mutator type: {mutator_type}")

        genome = Genome(policy, mutator, mutator_type)
        population.append(genome)

    return population


def tournament_select(population: List[Genome], k: int = 3) -> Genome:
    """Tournament selection: pick k random, return the fittest."""
    contestants = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(contestants, key=lambda i: population[i].fitness)
    return population[best]


def evolve_generation(population: List[Genome], crossover_rate: float = 0.3,
                      elitism: int = 5, generation: int = 0,
                      max_generations: int = 100) -> List[Genome]:
    """
    Produce next generation via self-replication and crossover.

    - Top `elitism` genomes are copied unchanged (increased from 2 to 5)
    - Remaining slots filled by reproduction or crossover
    - Generation number passed for mutation scale decay
    """
    pop_size = len(population)
    ranked = sorted(population, key=lambda g: g.fitness, reverse=True)

    new_population = []

    # Elitism: keep top performers unchanged
    for i in range(min(elitism, pop_size)):
        elite = Genome(
            policy=ranked[i]._clone_policy(),
            mutator=ranked[i]._clone_mutator(),
            mutator_type=ranked[i].mutator_type,
        )
        elite.set_flat_weights(ranked[i].get_flat_weights())
        elite.fitness = ranked[i].fitness
        new_population.append(elite)

    # Fill remaining slots
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
                  log_interval: int = 1, seed: int = 42) -> dict:
    """
    Main evolution loop.
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
    print(f"Mutator type: {mutator_type}")
    print(f"Population: {pop_size}, Generations: {generations}")
    print()

    history = {'best': [], 'mean': [], 'worst': []}

    for gen in range(generations):
        # Evaluate
        fitnesses = []
        for g in population:
            g.fitness = evaluate_genome(g, env_id, n_eval_episodes)
            fitnesses.append(g.fitness)

        best_fit = max(fitnesses)
        mean_fit = np.mean(fitnesses)
        worst_fit = min(fitnesses)
        history['best'].append(best_fit)
        history['mean'].append(float(mean_fit))
        history['worst'].append(worst_fit)

        if gen % log_interval == 0:
            print(f"Gen {gen:4d} | Best: {best_fit:8.2f} | Mean: {mean_fit:8.2f} | "
                  f"Worst: {worst_fit:8.2f}")

        # Evolve (pass generation for decay)
        population = evolve_generation(population, crossover_rate,
                                       elitism=5, generation=gen,
                                       max_generations=generations)

    return history
