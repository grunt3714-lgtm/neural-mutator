"""
Evolutionary algorithm with true self-replication.

Features:
- Tournament selection
- Self-replication: mutator processes full genome (policy + mutator weights)
- Asymmetric mutation rates: mutator self-modifies at lower rate
- Tracks mutator drift and fidelity metrics per generation
"""

import os
import torch
from multiprocessing import Pool as _Pool
import numpy as np
import gymnasium as gym
from typing import List, Dict
from .genome import (Genome, Policy, FlexiblePolicy, ChunkMutator, TransformerMutator, 
                      GaussianMutator, ErrorCorrectorMutator, CompatibilityNet)

# Module-level pool for reuse across generations
_worker_pool = None
_worker_pool_size = 0


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


def _eval_worker(args):
    """Worker for parallel genome evaluation. Receives serialized genome bytes."""
    genome_bytes, env_id, n_episodes, max_steps = args
    import io
    genome = Genome.load_bytes(genome_bytes)
    return evaluate_genome(genome, env_id, n_episodes, max_steps)


def _get_pool(n_workers: int) -> _Pool:
    """Get or create a persistent worker pool."""
    global _worker_pool, _worker_pool_size
    if _worker_pool is None or _worker_pool_size != n_workers:
        if _worker_pool is not None:
            _worker_pool.terminate()
        _worker_pool = _Pool(n_workers)
        _worker_pool_size = n_workers
    return _worker_pool


def evaluate_population(population: List[Genome], env_id: str,
                        n_episodes: int = 5, max_steps: int = 1000,
                        n_workers: int = 1) -> List[float]:
    """Evaluate all genomes, optionally in parallel."""
    if n_workers <= 1:
        return [evaluate_genome(g, env_id, n_episodes, max_steps) for g in population]

    args = [(g.to_bytes(), env_id, n_episodes, max_steps) for g in population]
    pool = _get_pool(n_workers)
    fitnesses = pool.map(_eval_worker, args)
    return fitnesses


def create_population(pop_size: int, obs_dim: int, act_dim: int,
                      mutator_type: str = 'chunk', hidden: int = 64,
                      chunk_size: int = 64,
                      speciation: bool = False,
                      flex: bool = False) -> List[Genome]:
    """Create initial population. If speciation=True, each genome gets a CompatibilityNet."""
    population = []
    for _ in range(pop_size):
        if flex:
            # Random small architecture: 1-2 layers, 32-64 neurons each
            n_layers = np.random.randint(1, 3)
            layer_sizes = [np.random.choice([32, 48, 64]) for _ in range(n_layers)]
            policy = FlexiblePolicy(obs_dim, act_dim, layer_sizes)
        else:
            policy = Policy(obs_dim, act_dim, hidden)
        if mutator_type == 'chunk':
            mutator = ChunkMutator(chunk_size=chunk_size)
        elif mutator_type == 'transformer':
            mutator = TransformerMutator(chunk_size=chunk_size)
        elif mutator_type == 'gaussian':
            mutator = GaussianMutator()
        elif mutator_type == 'corrector':
            mutator = ErrorCorrectorMutator(chunk_size=chunk_size)
        else:
            raise ValueError(f"Unknown mutator type: {mutator_type}")
        
        compat_net = None
        if speciation:
            policy_params = sum(p.numel() for p in policy.parameters())
            mutator_params = sum(p.numel() for p in mutator.parameters())
            tmp_compat = CompatibilityNet(policy_params + mutator_params)
            compat_params = sum(p.numel() for p in tmp_compat.parameters())
            compat_net = CompatibilityNet(policy_params + mutator_params + compat_params)
        
        population.append(Genome(policy, mutator, mutator_type, compat_net))
    
    # Pre-train compat nets on the initial population's genome distances
    if speciation and len(population) > 1:
        print("Pre-training compatibility networks...")
        genomes_flat = [g.get_flat_weights().detach() for g in population]
        for g in population:
            g.compat_net.pretrain(genomes_flat, steps=300)
        
        # Verify: check initial species structure
        test_info = assign_species(population)
        print(f"  Initial species after pretrain: {test_info['num_species']} "
              f"(sizes: {dict(sorted(test_info['species_sizes'].items()))})")
    
    return population


def tournament_select(population: List[Genome], k: int = 3) -> Genome:
    contestants = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(contestants, key=lambda i: population[i].fitness)
    return population[best]


def assign_species(population: List[Genome], compat_threshold: float = 0.5) -> Dict:
    """Assign species using learned species tags (embedding distance clustering)."""
    if population[0].compat_net is None:
        for g in population:
            g.species_id = 0
        return {'num_species': 1, 'species_sizes': {0: len(population)},
                'crossover_attempts': 0, 'crossover_compatible': 0}
    
    # Get all species tags
    tags = [g.get_species_tag() for g in population]
    
    # Greedy clustering by tag distance
    from src.genome import CompatibilityNet
    max_dist = compat_threshold * (CompatibilityNet.EMBED_DIM ** 0.5)
    
    representatives = []  # (species_id, tag_tensor)
    species_counter = 0
    
    for i, (g, tag) in enumerate(zip(population, tags)):
        assigned = False
        for sid, rep_tag in representatives:
            dist = torch.norm(tag - rep_tag).item()
            if dist < max_dist:
                g.species_id = sid
                assigned = True
                break
        if not assigned:
            g.species_id = species_counter
            representatives.append((species_counter, tag))
            species_counter += 1
    
    species_sizes = {}
    for g in population:
        species_sizes[g.species_id] = species_sizes.get(g.species_id, 0) + 1
    
    return {'num_species': len(species_sizes), 'species_sizes': species_sizes,
            'crossover_attempts': 0, 'crossover_compatible': 0}


def evolve_generation(population: List[Genome], crossover_rate: float = 0.3,
                      elitism: int = 5, generation: int = 0,
                      max_generations: int = 100,
                      compat_threshold: float = 0.5) -> tuple:
    """Returns (new_population, species_info)."""
    pop_size = len(population)
    ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
    new_population = []

    # Assign species
    species_info = assign_species(ranked, compat_threshold)
    crossover_attempts = 0
    crossover_compatible = 0

    # Elitism
    for i in range(min(elitism, pop_size)):
        elite = Genome(
            policy=ranked[i]._clone_policy(),
            mutator=ranked[i]._clone_mutator(),
            mutator_type=ranked[i].mutator_type,
            compat_net=ranked[i]._clone_compat() if ranked[i].compat_net else None,
        )
        elite.set_flat_weights(ranked[i].get_flat_weights())
        elite.fitness = ranked[i].fitness
        elite.self_replication_fidelity = ranked[i].self_replication_fidelity
        elite.species_id = ranked[i].species_id
        new_population.append(elite)

    while len(new_population) < pop_size:
        parent = tournament_select(ranked)
        if np.random.random() < crossover_rate:
            other = tournament_select(ranked)
            crossover_attempts += 1
            # Check learned compatibility
            if parent.is_compatible(other, compat_threshold):
                crossover_compatible += 1
                try:
                    child = parent.crossover(other, generation, max_generations)
                except Exception:
                    child = parent.reproduce(generation, max_generations)
            else:
                # Incompatible — reproduce asexually instead
                try:
                    child = parent.reproduce(generation, max_generations)
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
                    compat_net=parent._clone_compat() if parent.compat_net else None,
                )
                flat = parent.get_flat_weights()
                noise = torch.randn_like(flat) * 0.01
                child.set_flat_weights(flat + noise)
        new_population.append(child)

    species_info['crossover_attempts'] = crossover_attempts
    species_info['crossover_compatible'] = crossover_compatible
    return new_population, species_info


def run_evolution(env_id: str = 'CartPole-v1', pop_size: int = 30,

                  generations: int = 100, mutator_type: str = 'chunk',
                  n_eval_episodes: int = 5, crossover_rate: float = 0.3,
                  hidden: int = 64, chunk_size: int = 64,
                  log_interval: int = 1, seed: int = 42,
                  speciation: bool = False,
                  compat_threshold: float = 0.5,
                  flex: bool = False,
                  complexity_cost: float = 0.0,
                  output_dir: str = None,
                  n_workers: int = 1) -> Dict:
    """
    Main evolution loop with true self-replication.

    The mutator processes the full genome (policy + mutator weights) and
    outputs a new full genome. Natural selection is the only filter —
    mutators that destroy themselves die, those that improve themselves thrive.
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
                                   hidden, chunk_size, speciation=speciation,
                                   flex=flex)
    genome = population[0]
    print(f"Genome: {genome.num_policy_params()} policy params + "
          f"{genome.num_mutator_params()} mutator params + "
          f"{genome.num_compat_params()} compat params = "
          f"{genome.num_total_params()} total")
    print(f"Mutator type: {mutator_type}")
    print(f"Speciation: {'learned' if speciation else 'off'}")
    print(f"Population: {pop_size}, Generations: {generations}")
    print()

    # Track initial mutator weights for drift measurement
    initial_mutator_weights = {}
    for i, g in enumerate(population):
        initial_mutator_weights[i] = g.get_flat_mutator_weights().clone()

    history = {
        'best': [], 'mean': [], 'worst': [],
        'mean_fidelity': [],
        'best_fidelity': [],
        'mean_mutator_drift': [],
        'max_mutator_drift': [],
        'num_species': [],
        'crossover_compat_rate': [],
        'mean_layers': [],
        'max_layers': [],
        'mean_neurons': [],
        'max_neurons': [],
        'structural_mutations': [],
        'mean_policy_params': [],
        'min_policy_params': [],
    }

    if complexity_cost > 0:
        print(f"Complexity cost: {complexity_cost} per parameter")
    if n_workers > 1:
        print(f"Parallel evaluation: {n_workers} workers")

    for gen in range(generations):
        # Evaluate fitness — pure environment reward
        raw_fitnesses = evaluate_population(population, env_id, n_eval_episodes,
                                            n_workers=n_workers)
        for g, raw in zip(population, raw_fitnesses):
            g.fitness = raw
        
        # Complexity penalty: penalize larger networks (encourages parsimony)
        if complexity_cost > 0:
            for g in population:
                penalty = complexity_cost * g.num_policy_params()
                g.fitness -= penalty

        # Fitness sharing: divide by species size to reward diversity
        if speciation:
            spec_info_pre = assign_species(population, compat_threshold)
            for g in population:
                species_size = spec_info_pre['species_sizes'].get(g.species_id, 1)
                g.fitness = g.fitness / (species_size ** 0.5)

        # Use raw fitnesses for tracking (not shared)
        fitnesses = raw_fitnesses if speciation else [g.fitness for g in population]
        fidelities = [g.self_replication_fidelity for g in population]

        best_idx = np.argmax(fitnesses)
        best_fit = fitnesses[best_idx]
        mean_fit = np.mean(fitnesses)
        worst_fit = min(fitnesses)

        # Save best genome periodically
        if output_dir is not None:
            import os
            best_genome_path = os.path.join(output_dir, 'best_genome.pt')
            population[best_idx].save(best_genome_path)

        # Compute mutator drift from generation 0
        drifts = []
        for i, g in enumerate(population):
            current_mutator = g.get_flat_mutator_weights()
            ref_idx = i % len(initial_mutator_weights)
            drift = torch.norm(current_mutator - initial_mutator_weights[ref_idx]).item()
            drifts.append(drift)

        # Architecture stats (flex mode)
        if flex:
            layers_list = [len(g.policy.layer_sizes) for g in population]
            neurons_list = [sum(g.policy.layer_sizes) for g in population]
            struct_muts = sum(1 for g in population if g.last_structural_mutation is not None)
            history['mean_layers'].append(float(np.mean(layers_list)))
            history['max_layers'].append(max(layers_list))
            history['mean_neurons'].append(float(np.mean(neurons_list)))
            history['max_neurons'].append(max(neurons_list))
            history['structural_mutations'].append(struct_muts)

        # Track parameter counts
        param_counts = [g.num_policy_params() for g in population]
        history['mean_policy_params'].append(float(np.mean(param_counts)))
        history['min_policy_params'].append(min(param_counts))

        history['best'].append(best_fit)
        history['mean'].append(float(mean_fit))
        history['worst'].append(worst_fit)
        history['mean_fidelity'].append(float(np.mean(fidelities)))
        history['best_fidelity'].append(fidelities[best_idx])
        history['mean_mutator_drift'].append(float(np.mean(drifts)))
        history['max_mutator_drift'].append(float(np.max(drifts)))

        if gen % log_interval == 0:
            msg = (f"Gen {gen:4d} | Best: {best_fit:8.2f} | Mean: {mean_fit:8.2f} | "
                   f"Fidelity: {np.mean(fidelities):.4f} | Drift: {np.mean(drifts):.4f}")
            if flex:
                msg += (f" | Layers: {np.mean(layers_list):.1f} "
                        f"| Neurons: {np.mean(neurons_list):.0f}")
            print(msg)

        population, species_info = evolve_generation(
            population, crossover_rate, elitism=5, generation=gen,
            max_generations=generations, compat_threshold=compat_threshold)
        
        history['num_species'].append(species_info['num_species'])
        compat_rate = (species_info['crossover_compatible'] / max(species_info['crossover_attempts'], 1))
        history['crossover_compat_rate'].append(compat_rate)
        
        if speciation and gen % log_interval == 0:
            sizes = species_info['species_sizes']
            print(f"         Species: {species_info['num_species']} | "
                  f"Compat rate: {compat_rate:.2f} | "
                  f"Sizes: {dict(sorted(sizes.items()))}")

    return history
