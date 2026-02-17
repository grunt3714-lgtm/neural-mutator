"""
Evolutionary algorithm with true self-replication.

Features:
- Tournament selection
- Self-replication: mutator processes full genome (policy + mutator weights)
- Asymmetric mutation rates: mutator self-modifies at lower rate
- Tracks mutator drift and fidelity metrics per generation
"""

import os
import time
import torch
import multiprocessing as _mp
_Pool = _mp.Pool
import numpy as np
import gymnasium as gym
from typing import List, Dict
from .genome import (
    Genome,
    Policy,
    PolicyCNN,
    FlexiblePolicy,
    CompatibilityNet,
    create_mutator,
    available_mutator_types,
)

from .snake_env import ensure_snake_registered
from .speciation import assign_species
# Module-level pool for reuse across generations
_worker_pool = None
_worker_pool_size = 0


def evaluate_genome(genome: Genome, env_id: str, n_episodes: int = 5,
                    max_steps: int = 1000, seeds=None) -> tuple[float, int]:
    """Evaluate a genome's policy on an environment.

    Returns:
        (mean_reward, env_steps)
    """
    ensure_snake_registered()
    env = gym.make(env_id)
    rewards = []
    total_steps = 0
    for ep in range(n_episodes):
        seed = seeds[ep] if seeds and ep < len(seeds) else None
        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = genome.policy.act(obs)
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
    """Worker for parallel genome evaluation. Receives serialized genome bytes."""
    genome_bytes, env_id, n_episodes, max_steps = args
    genome = Genome.load_bytes(genome_bytes)
    return evaluate_genome(genome, env_id, n_episodes, max_steps)


def _get_pool(n_workers: int):
    """Get or create a persistent worker pool."""
    global _worker_pool, _worker_pool_size
    if _worker_pool is None or _worker_pool_size != n_workers:
        if _worker_pool is not None:
            _worker_pool.terminate()
        ctx = _mp.get_context('spawn')  # 'spawn' avoids fork issues with pygame/OpenGL
        _worker_pool = ctx.Pool(n_workers)
        _worker_pool_size = n_workers
    return _worker_pool


def evaluate_population(population: List[Genome], env_id: str,
                        n_episodes: int = 5, max_steps: int = 1000,
                        n_workers: int = 1, fleet=None) -> tuple[List[float], Dict]:
    """Evaluate all genomes, optionally in parallel or via fleet.

    Returns:
        (fitnesses, eval_profile)
    """
    if fleet is not None:
        genomes_bytes = [g.to_bytes() for g in population]
        return fleet.evaluate_population(genomes_bytes, env_id, n_episodes, max_steps)

    t0 = time.time()

    if n_workers <= 1:
        pairs = [evaluate_genome(g, env_id, n_episodes, max_steps) for g in population]
        fitnesses = [p[0] for p in pairs]
        steps = int(sum(p[1] for p in pairs))
        elapsed = time.time() - t0
        profile = {
            'dispatch_sec': 0.0,
            'remote_eval_sec': elapsed,
            'result_gather_sec': 0.0,
            'eval_total_sec': elapsed,
            'env_steps': steps,
            'genomes_evaluated': len(population),
        }
        return fitnesses, profile

    t_dispatch0 = time.time()
    args = [(g.to_bytes(), env_id, n_episodes, max_steps) for g in population]
    pool = _get_pool(n_workers)
    dispatch_sec = time.time() - t_dispatch0

    t_eval0 = time.time()
    pairs = pool.map(_eval_worker, args)
    remote_eval_sec = time.time() - t_eval0

    fitnesses = [p[0] for p in pairs]
    steps = int(sum(p[1] for p in pairs))
    elapsed = time.time() - t0
    profile = {
        'dispatch_sec': dispatch_sec,
        'remote_eval_sec': remote_eval_sec,
        'result_gather_sec': 0.0,
        'eval_total_sec': elapsed,
        'env_steps': steps,
        'genomes_evaluated': len(population),
    }
    return fitnesses, profile


def create_population(pop_size: int, obs_dim: int, act_dim: int,
                      mutator_type: str = 'chunk', hidden: int = 64,
                      chunk_size: int = 64,
                      speciation: bool = False,
                      flex: bool = False,
                      policy_arch: str = 'mlp',
                      output_dir: str = None,
                      mutator_kwargs: dict | None = None,
                      obs_shape: tuple | None = None) -> List[Genome]:
    """Create initial population. If speciation=True, each genome gets a CompatibilityNet."""
    population = []
    for _ in range(pop_size):
        if policy_arch == 'cnn-large' and obs_shape is not None:
            from .genome import PolicyCNNLarge
            policy = PolicyCNNLarge(obs_shape, act_dim)
        elif policy_arch == 'cnn':
            policy = PolicyCNN(obs_dim, act_dim)
        elif flex:
            # Random small architecture: 1-2 layers, 32-64 neurons each
            n_layers = np.random.randint(1, 3)
            layer_sizes = [np.random.choice([32, 48, 64]) for _ in range(n_layers)]
            policy = FlexiblePolicy(obs_dim, act_dim, layer_sizes)
        else:
            policy = Policy(obs_dim, act_dim, hidden)
        if mutator_type not in available_mutator_types():
            raise ValueError(f"Unknown mutator type: {mutator_type}")
        mk = dict(mutator_kwargs or {})
        mk.setdefault('chunk_size', chunk_size)
        mutator = create_mutator(mutator_type, **mk)
        
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
        genomes_flat = [g.get_flat_weights().detach() for g in population]
        genome_dim = population[0].compat_net.genome_dim
        
        # Check for cached pre-trained weights
        cache_dir = os.path.join(output_dir, '.cache') if output_dir else None
        cache_path = os.path.join(cache_dir, f'compat_pretrained_{genome_dim}.pt') if cache_dir else None
        
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached pre-trained compat net from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            for g in population:
                g.compat_net.encoder.load_state_dict(cached['encoder'])
                g.compat_net.scorer.load_state_dict(cached['scorer'])
        else:
            print("Pre-training compatibility network (one template, cloned to all)...")
            # Pre-train just ONE compat net, then clone weights to all others
            template = population[0].compat_net
            template.pretrain(genomes_flat, steps=300)
            encoder_state = template.encoder.state_dict()
            scorer_state = template.scorer.state_dict()
            for g in population[1:]:
                g.compat_net.encoder.load_state_dict(encoder_state)
                g.compat_net.scorer.load_state_dict(scorer_state)
            # Cache for future runs
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({'encoder': encoder_state, 'scorer': scorer_state}, cache_path)
                print(f"  Cached pre-trained weights to {cache_path}")
        
        # Verify: check initial species structure
        test_info = assign_species(population)
        print(f"  Initial species after pretrain: {test_info['num_species']} "
              f"(sizes: {dict(sorted(test_info['species_sizes'].items()))})")
    
    return population


def tournament_select(population: List[Genome], k: int = 3) -> Genome:
    contestants = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(contestants, key=lambda i: population[i].fitness)
    return population[best]


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
                  policy_arch: str = 'mlp',
                  complexity_cost: float = 0.0,
                  output_dir: str = None,
                  n_workers: int = 1,
                  fleet=None,
                  progress_callback=None,
                  mutator_kwargs: dict | None = None) -> Dict:
    """
    Main evolution loop with true self-replication.

    The mutator processes the full genome (policy + mutator weights) and
    outputs a new full genome. Natural selection is the only filter —
    mutators that destroy themselves die, those that improve themselves thrive.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    ensure_snake_registered()

    env = gym.make(env_id)
    obs_shape = env.observation_space.shape  # full shape for image envs
    if len(obs_shape) == 1:
        obs_dim = obs_shape[0]
    else:
        obs_dim = int(np.prod(obs_shape))  # flattened for param counting
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]
    env.close()

    # Auto-detect CNN for image environments
    if len(obs_shape) == 3 and policy_arch == 'mlp':
        policy_arch = 'cnn-large'
        print(f"Auto-detected image environment, using cnn-large policy")

    print(f"Environment: {env_id} (obs={obs_shape}, act={act_dim})")

    population = create_population(pop_size, obs_dim, act_dim, mutator_type,
                                   hidden, chunk_size, speciation=speciation,
                                   flex=flex, policy_arch=policy_arch, output_dir=output_dir,
                                   mutator_kwargs=mutator_kwargs,
                                   obs_shape=obs_shape if len(obs_shape) == 3 else None)
    genome = population[0]
    print(f"Genome: {genome.num_policy_params()} policy params + "
          f"{genome.num_mutator_params()} mutator params + "
          f"{genome.num_compat_params()} compat params = "
          f"{genome.num_total_params()} total")
    print(f"Mutator type: {mutator_type}")
    print(f"Policy arch: {policy_arch}")
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
        'species_entropy': [],
        'species_tag_dist_mean': [],
        'species_tag_dist_std': [],
        'crossover_compat_rate': [],
        'mean_layers': [],
        'max_layers': [],
        'mean_neurons': [],
        'max_neurons': [],
        'structural_mutations': [],
        'mean_policy_params': [],
        'min_policy_params': [],
        # Profiling / throughput
        'gen_wall_sec': [],
        'dispatch_sec': [],
        'remote_eval_sec': [],
        'result_gather_sec': [],
        'evolution_step_sec': [],
        'logging_checkpoint_sec': [],
        'env_steps': [],
        'env_steps_per_sec': [],
        'genomes_evaluated': [],
        'genomes_per_sec': [],
    }

    if complexity_cost > 0:
        print(f"Complexity cost: {complexity_cost} per parameter")
    if n_workers > 1:
        print(f"Parallel evaluation: {n_workers} workers")

    # ── Helper: process results for a generation ──────────────────
    def _process_gen(gen, population, raw_fitnesses, eval_profile, gen_t0):
        """Score population, log stats, evolve next gen. Returns (new_pop, species_info, evolution_step_sec, logging_checkpoint_sec)."""
        for g, raw in zip(population, raw_fitnesses):
            g.fitness = raw

        # Complexity penalty
        if complexity_cost > 0:
            for g in population:
                penalty = complexity_cost * g.num_policy_params()
                g.fitness -= penalty

        # Fitness sharing
        if speciation:
            spec_info_pre = assign_species(population, compat_threshold)
            for g in population:
                species_size = spec_info_pre['species_sizes'].get(g.species_id, 1)
                g.fitness = g.fitness / (species_size ** 0.5)

        fitnesses = raw_fitnesses if speciation else [g.fitness for g in population]
        fidelities = [g.self_replication_fidelity for g in population]

        best_idx = np.argmax(fitnesses)
        best_fit = fitnesses[best_idx]
        mean_fit = np.mean(fitnesses)
        worst_fit = min(fitnesses)

        t_log0 = time.time()

        if output_dir is not None:
            best_genome_path = os.path.join(output_dir, 'best_genome.pt')
            population[best_idx].save(best_genome_path)
            if not hasattr(run_evolution, '_best_ever_fit') or best_fit > run_evolution._best_ever_fit:
                run_evolution._best_ever_fit = best_fit
                best_ever_path = os.path.join(output_dir, 'best_ever_genome.pt')
                population[best_idx].save(best_ever_path)

        drifts = []
        for i, g in enumerate(population):
            current_mutator = g.get_flat_mutator_weights()
            ref_idx = i % len(initial_mutator_weights)
            drift = torch.norm(current_mutator - initial_mutator_weights[ref_idx]).item()
            drifts.append(drift)

        if flex:
            layers_list = [len(g.policy.layer_sizes) for g in population]
            neurons_list = [sum(g.policy.layer_sizes) for g in population]
            struct_muts = sum(1 for g in population if g.last_structural_mutation is not None)
            history['mean_layers'].append(float(np.mean(layers_list)))
            history['max_layers'].append(max(layers_list))
            history['mean_neurons'].append(float(np.mean(neurons_list)))
            history['max_neurons'].append(max(neurons_list))
            history['structural_mutations'].append(struct_muts)

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

        if progress_callback is not None:
            best_ever = max(history['best']) if history['best'] else best_fit
            progress_callback(gen, generations, best_fit, mean_fit, best_ever)

        logging_checkpoint_sec = time.time() - t_log0

        t_evolve0 = time.time()
        new_population, species_info = evolve_generation(
            population, crossover_rate, elitism=5, generation=gen,
            max_generations=generations, compat_threshold=compat_threshold)
        evolution_step_sec = time.time() - t_evolve0

        # Record species/perf stats
        history['num_species'].append(species_info['num_species'])
        history['species_entropy'].append(float(species_info.get('species_entropy', 0.0)))
        history['species_tag_dist_mean'].append(float(species_info.get('tag_dist_mean', 0.0)))
        history['species_tag_dist_std'].append(float(species_info.get('tag_dist_std', 0.0)))
        compat_rate = (species_info['crossover_compatible'] / max(species_info['crossover_attempts'], 1))
        history['crossover_compat_rate'].append(compat_rate)

        gen_wall_sec = time.time() - gen_t0
        env_steps = int(eval_profile.get('env_steps', 0))
        genomes_evald = int(eval_profile.get('genomes_evaluated', len(population)))
        steps_per_sec = (env_steps / gen_wall_sec) if gen_wall_sec > 0 else 0.0
        genomes_per_sec = (genomes_evald / gen_wall_sec) if gen_wall_sec > 0 else 0.0

        history['gen_wall_sec'].append(float(gen_wall_sec))
        history['dispatch_sec'].append(float(eval_profile.get('dispatch_sec', 0.0)))
        history['remote_eval_sec'].append(float(eval_profile.get('remote_eval_sec', 0.0)))
        history['result_gather_sec'].append(float(eval_profile.get('result_gather_sec', 0.0)))
        history['evolution_step_sec'].append(float(evolution_step_sec))
        history['logging_checkpoint_sec'].append(float(logging_checkpoint_sec))
        history['env_steps'].append(env_steps)
        history['env_steps_per_sec'].append(float(steps_per_sec))
        history['genomes_evaluated'].append(genomes_evald)
        history['genomes_per_sec'].append(float(genomes_per_sec))

        if speciation and gen % log_interval == 0:
            sizes = species_info['species_sizes']
            print(f"         Species: {species_info['num_species']} | "
                  f"Entropy: {species_info.get('species_entropy', 0.0):.3f} | "
                  f"TagDist μ/σ: {species_info.get('tag_dist_mean', 0.0):.3f}/{species_info.get('tag_dist_std', 0.0):.3f} | "
                  f"Compat rate: {compat_rate:.2f} | "
                  f"Sizes: {dict(sorted(sizes.items()))}")

        if gen % log_interval == 0:
            print(f"         Perf: {steps_per_sec:8.1f} env_steps/s | "
                  f"{genomes_per_sec:6.2f} genomes/s | "
                  f"dispatch {eval_profile.get('dispatch_sec', 0.0):.3f}s | "
                  f"eval {eval_profile.get('remote_eval_sec', 0.0):.3f}s | "
                  f"gather {eval_profile.get('result_gather_sec', 0.0):.3f}s | "
                  f"evolve {evolution_step_sec:.3f}s | "
                  f"log/ckpt {logging_checkpoint_sec:.3f}s")

        return new_population

    # ── Pipelined fleet mode vs standard sequential ─────────────
    use_pipeline = (fleet is not None and hasattr(fleet, 'dispatch'))

    if use_pipeline:
        # Pipeline: overlap evolve/log of gen N with eval of gen N+1
        #
        # Gen 0: dispatch → collect (blocking, no overlap possible)
        # Gen 1+: while workers eval gen N+1, we process gen N results
        #
        # Timeline:
        #   dispatch(pop_0)
        #   collect(pop_0) results  ← blocking
        #   process gen 0 + evolve → pop_1
        #   dispatch(pop_1)         ← workers start immediately
        #   [process gen 0 stats already done, workers running]
        #   collect(pop_1) results  ← blocking, but overlapped with dispatch
        #   process gen 1 + evolve → pop_2
        #   dispatch(pop_2)
        #   ...

        for gen in range(generations):
            gen_t0 = time.time()

            if gen == 0:
                # First gen: synchronous (nothing to overlap with)
                raw_fitnesses, eval_profile = evaluate_population(
                    population, env_id, n_eval_episodes, n_workers=n_workers, fleet=fleet
                )
            else:
                # Collect results dispatched at end of previous iteration
                raw_fitnesses, eval_profile = fleet.collect()

            # Process results + evolve next generation
            population = _process_gen(gen, population, raw_fitnesses, eval_profile, gen_t0)

            # Pre-dispatch next gen to workers (if not last gen)
            if gen < generations - 1:
                genomes_bytes = [g.to_bytes() for g in population]
                fleet.dispatch(genomes_bytes, env_id, n_eval_episodes)
    else:
        # Standard sequential mode (local eval or non-pipeline fleet)
        for gen in range(generations):
            gen_t0 = time.time()

            raw_fitnesses, eval_profile = evaluate_population(
                population, env_id, n_eval_episodes, n_workers=n_workers, fleet=fleet
            )
            population = _process_gen(gen, population, raw_fitnesses, eval_profile, gen_t0)

    return history
