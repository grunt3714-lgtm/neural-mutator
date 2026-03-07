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
from .lineage import LineageTracker
# Module-level pool for reuse across generations
_worker_pool = None
_worker_pool_size = 0


def _species_diagnostics_disabled(population: List[Genome]) -> Dict:
    """Species assignment removed: force single diagnostics species only.

    This keeps downstream logging/history keys stable while avoiding the
    expensive species assignment pass.
    """
    for g in population:
        g.species_id = 0
    n = len(population)
    return {
        'num_species': 1,
        'species_sizes': {0: int(n)},
        'species_entropy': 0.0,
        'tag_dist_mean': 0.0,
        'tag_dist_std': 0.0,
        'tag_dist_p50': 0.0,
        'lineage_assign_require_common_ancestor': False,
        'lineage_assign_depth': 0,
        'lineage_assign_checks': 0,
        'lineage_assign_filtered': 0,
        'lineage_assign_filter_rate': 0.0,
        'crossover_attempts': 0,
        'crossover_compatible': 0,
    }


def _extract_obs(obs):
    """Normalize wrapped/dict observations to ndarray for policy.act()."""
    if isinstance(obs, dict):
        if 'screen' in obs:
            return obs['screen']
        if 'observation' in obs:
            return obs['observation']
        return next(iter(obs.values()))
    return obs


def evaluate_genome(genome: Genome, env_id: str, n_episodes: int = 5,
                    max_steps: int = 1000, seeds=None) -> tuple[float, int]:
    """Evaluate a genome's policy on an environment.

    Returns:
        (mean_reward, env_steps)
    """
    ensure_snake_registered()
    if env_id.startswith('Vizdoom'):
        import vizdoom.gymnasium_wrapper  # registers vizdoom env ids
    env = gym.make(env_id)
    rewards = []
    total_steps = 0
    for ep in range(n_episodes):
        seed = seeds[ep] if seeds and ep < len(seeds) else None
        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            obs_in = _extract_obs(obs)
            action = genome.policy.act(obs_in)
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
    # Supports both legacy args and indexed args for progress reporting.
    if len(args) == 4:
        genome_bytes, env_id, n_episodes, max_steps = args
        idx = None
    else:
        idx, genome_bytes, env_id, n_episodes, max_steps = args

    genome = Genome.load_bytes(genome_bytes)
    reward, steps = evaluate_genome(genome, env_id, n_episodes, max_steps)
    if idx is None:
        return reward, steps
    return idx, reward, steps


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
                        n_workers: int = 1, fleet=None,
                        genome_callback=None) -> tuple[List[float], Dict]:
    """Evaluate all genomes, optionally in parallel or via fleet.

    Returns:
        (fitnesses, eval_profile)
    """
    if fleet is not None:
        genomes_bytes = [g.to_bytes() for g in population]
        fleet.dispatch(genomes_bytes, env_id, n_episodes, max_steps)
        del genomes_bytes  # free ~1.2GB before blocking on collect
        import gc; gc.collect()
        return fleet.collect(genome_callback=genome_callback)

    t0 = time.time()

    if n_workers <= 1:
        pairs = []
        n = len(population)
        for i, g in enumerate(population, start=1):
            p = evaluate_genome(g, env_id, n_episodes, max_steps)
            pairs.append(p)
            if genome_callback is not None:
                try:
                    genome_callback(i, n)
                except Exception as e:
                    print(f'[local] genome_callback error: {e}')
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
    args = [(i, g.to_bytes(), env_id, n_episodes, max_steps) for i, g in enumerate(population)]
    pool = _get_pool(n_workers)
    dispatch_sec = time.time() - t_dispatch0

    t_eval0 = time.time()
    n = len(population)
    fitnesses = [0.0] * n
    steps = 0
    collected = 0
    # Stream results as they complete so genome-level progress can update in Discord.
    for item in pool.imap_unordered(_eval_worker, args, chunksize=1):
        idx, reward, s = item
        fitnesses[idx] = reward
        steps += int(s)
        collected += 1
        if genome_callback is not None:
            try:
                genome_callback(collected, n)
            except Exception as e:
                print(f'[local] genome_callback error: {e}')
    remote_eval_sec = time.time() - t_eval0

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


def _apply_mutator_crossover_config(mutator: torch.nn.Module, cfg: dict) -> None:
    """Apply crossover-gate runtime overrides to mutator buffers when present."""
    if hasattr(mutator, 'crossover_gated_blend') and 'crossover_gated_blend' in cfg:
        mutator.crossover_gated_blend.fill_(1.0 if bool(cfg['crossover_gated_blend']) else 0.0)
    if hasattr(mutator, 'crossover_gate_mode_id') and 'crossover_gate_mode' in cfg:
        mode = str(cfg['crossover_gate_mode']).lower().strip()
        mutator.crossover_gate_mode_id.fill_(1.0 if mode == 'gumbel' else 0.0)
    if hasattr(mutator, 'crossover_gumbel_tau') and 'crossover_gumbel_tau' in cfg:
        mutator.crossover_gumbel_tau.fill_(float(cfg['crossover_gumbel_tau']))
    if hasattr(mutator, 'crossover_gumbel_hard') and 'crossover_gumbel_hard' in cfg:
        mutator.crossover_gumbel_hard.fill_(1.0 if bool(cfg['crossover_gumbel_hard']) else 0.0)
    if hasattr(mutator, 'crossover_gate_clamp') and 'crossover_gate_clamp' in cfg:
        mutator.crossover_gate_clamp.fill_(float(cfg['crossover_gate_clamp']))


def create_population(pop_size: int, obs_dim: int, act_dim: int,
                      mutator_type: str = 'dualmixture', hidden: int = 64,
                      chunk_size: int = 64,
                      speciation: bool = False,
                      compat_threshold: float = 0.5,
                      learn_compat_threshold: bool = False,
                      compat_binary: bool = False,
                      compat_pretrain: bool = True,
                      unified_mating: bool = False,
                      unified_mate_head: bool = False,
                      flex: bool = False,
                      policy_arch: str = 'mlp',
                      output_dir: str = None,
                      mutator_kwargs: dict | None = None,
                      obs_shape: tuple | None = None,
                      init_genome_path: str | None = None,
                      init_mutator_from_genome_path: str | None = None) -> List[Genome]:
    """Create initial population. If speciation=True, each genome gets a CompatibilityNet.

    If init_genome_path is provided, seed the population from that genome
    (elite + reproduced children) instead of pure random initialization.

    If init_mutator_from_genome_path is provided, initialize mutator weights
    from that genome while keeping policy initialization random.
    """
    population = []
    mk = dict(mutator_kwargs or {})
    compat_mode = (
        'mutator_pair_head' if (unified_mating and unified_mate_head)
        else ('mutator_affinity' if unified_mating else ('binary_strict' if compat_binary else 'distance'))
    )

    if init_genome_path and init_mutator_from_genome_path:
        raise ValueError("Choose only one of init_genome_path or init_mutator_from_genome_path")

    seed_mutator_state = None
    if init_mutator_from_genome_path:
        seed_mutator_state = Genome.load(init_mutator_from_genome_path).mutator.state_dict()

    if init_genome_path:
        seed = Genome.load(init_genome_path)
        _apply_mutator_crossover_config(seed.mutator, mk)
        seed.compat_threshold = float(compat_threshold)
        seed.evolve_compat_threshold = bool(learn_compat_threshold)
        seed.compat_mode = compat_mode
        population.append(seed)
        for _ in range(pop_size - 1):
            population.append(seed.reproduce(generation=0, max_generations=100))

    while len(population) < pop_size:
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
        mk.setdefault('chunk_size', chunk_size)
        mutator = create_mutator(mutator_type, **mk)
        _apply_mutator_crossover_config(mutator, mk)
        if seed_mutator_state is not None:
            mutator.load_state_dict(seed_mutator_state)
            _apply_mutator_crossover_config(mutator, mk)

        compat_net = None
        if speciation and (not unified_mating):
            policy_params = sum(p.numel() for p in policy.parameters())
            mutator_params = sum(p.numel() for p in mutator.parameters())
            if compat_mode == 'mutator_pair_head':
                compat_net = CompatibilityNet(mutator_params)
            else:
                tmp_compat = CompatibilityNet(policy_params + mutator_params)
                compat_params = sum(p.numel() for p in tmp_compat.parameters())
                compat_net = CompatibilityNet(policy_params + mutator_params + compat_params)
        
        population.append(Genome(
            policy,
            mutator,
            mutator_type,
            compat_net,
            compat_threshold=compat_threshold,
            evolve_compat_threshold=learn_compat_threshold,
            compat_mode=compat_mode,
        ))
    
    # Optional warm-start for unified learned pair-head over mutator embeddings
    if speciation and unified_mate_head and len(population) > 1 and compat_pretrain:
        mut_flats = [g.get_flat_mutator_core_weights().detach() for g in population]
        print("Warm-starting unified mutator pair-head (binary mode, one template, cloned to all)...")
        template = population[0].ensure_mutator_pair_head()
        template.pretrain(mut_flats, steps=200, mode='binary')
        encoder_state = template.encoder.state_dict()
        scorer_state = template.scorer.state_dict()
        proj_state = template.projection.detach().clone()
        for g in population[1:]:
            g.ensure_mutator_pair_head().encoder.load_state_dict(encoder_state)
            g.ensure_mutator_pair_head().scorer.load_state_dict(scorer_state)
            g.ensure_mutator_pair_head().projection.copy_(proj_state)
        test_info = _species_diagnostics_disabled(population)
        print(f"  Initial species after warm-start: {test_info['num_species']} "
              f"(sizes: {dict(sorted(test_info['species_sizes'].items()))})")

    # Optional pre-train compat nets on initial population distances
    if speciation and (not unified_mating) and len(population) > 1 and compat_pretrain:
        genomes_flat = [g.get_flat_weights().detach() for g in population]
        genome_dim = population[0].compat_net.genome_dim
        
        # Check for cached pre-trained weights
        cache_dir = os.path.join(output_dir, '.cache') if output_dir else None
        pretrain_mode = 'binary' if compat_binary else 'contrastive'
        cache_path = os.path.join(cache_dir, f'compat_pretrained_{genome_dim}_{pretrain_mode}.pt') if cache_dir else None
        
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached pre-trained compat net from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            for g in population:
                g.compat_net.encoder.load_state_dict(cached['encoder'])
                g.compat_net.scorer.load_state_dict(cached['scorer'])
        else:
            print(f"Pre-training compatibility network ({pretrain_mode} mode, one template, cloned to all)...")
            # Pre-train just ONE compat net, then clone weights to all others
            template = population[0].compat_net
            template.pretrain(genomes_flat, steps=300, mode=pretrain_mode)
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
        test_info = _species_diagnostics_disabled(population)
        print(f"  Initial species after pretrain: {test_info['num_species']} "
              f"(sizes: {dict(sorted(test_info['species_sizes'].items()))})")
    elif speciation and (not unified_mating) and len(population) > 1 and not compat_pretrain:
        print("Skipping compatibility pretraining (cold-start compat net)")

    return population


def tournament_select(population: List[Genome], k: int = 3) -> Genome:
    contestants = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(contestants, key=lambda i: population[i].fitness)
    return population[best]


def _sample_softmax_index(scores: list[float], temperature: float) -> int | None:
    if not scores:
        return None
    if len(scores) == 1:
        return 0
    temp = max(float(temperature), 1e-6)
    x = np.asarray(scores, dtype=np.float64) / temp
    x = x - np.max(x)
    p = np.exp(x)
    total = float(np.sum(p))
    if (not np.isfinite(total)) or total <= 0.0:
        p = np.full(len(scores), 1.0 / len(scores), dtype=np.float64)
    else:
        p = p / total
    return int(np.random.choice(len(scores), p=p))


def _policy_distance_norm(a: Genome, b: Genome, cache: dict[tuple[int, int], float]) -> float:
    key = (id(a), id(b)) if id(a) < id(b) else (id(b), id(a))
    if key in cache:
        return cache[key]
    with torch.no_grad():
        da = a.get_flat_policy_weights()
        db = b.get_flat_policy_weights()
        n = int(min(da.shape[0], db.shape[0]))
        if n <= 0:
            dist_norm = 1.0
        else:
            dist = torch.norm(da[:n] - db[:n]).item()
            dist_norm = float(dist / (dist + 1.0))
    cache[key] = float(np.clip(dist_norm, 0.0, 1.0))
    return cache[key]


def _mate_preference_score(chooser: Genome,
                           candidate: Genome,
                           rank_index: dict[int, int],
                           policy_dist_cache: dict[tuple[int, int], float]) -> float:
    # Prefer existing learned compatibility signal when present.
    if chooser.compat_net is not None and candidate.compat_net is not None:
        with torch.no_grad():
            dist = torch.norm(chooser.get_species_tag() - candidate.get_species_tag()).item()
        scale = max(CompatibilityNet.EMBED_DIM ** 0.5, 1e-6)
        return float(np.exp(-dist / scale))

    # Deterministic proxy: combine rank distance and policy distance.
    max_rank = max(len(rank_index) - 1, 1)
    rank_gap = abs(rank_index[id(chooser)] - rank_index[id(candidate)]) / max_rank
    policy_gap = _policy_distance_norm(chooser, candidate, policy_dist_cache)
    score = 1.0 - 0.5 * float(rank_gap) - 0.5 * float(policy_gap)
    return float(np.clip(score, 0.0, 1.0))


def _select_mate_choice_pair(ranked: List[Genome],
                             elitism: int,
                             topk: int,
                             threshold: float,
                             temperature: float,
                             max_attempts: int) -> tuple[Genome | None, Genome | None, bool]:
    if len(ranked) < 2:
        return None, None, False

    candidate_count = max(2, min(int(topk), len(ranked)))
    candidate_pool = ranked[:candidate_count]
    chooser_count = max(1, min(len(ranked), max(int(elitism), len(ranked) // 2)))
    chooser_pool = ranked[:chooser_count]
    rank_index = {id(g): i for i, g in enumerate(ranked)}
    policy_dist_cache: dict[tuple[int, int], float] = {}

    for _ in range(max(1, int(max_attempts))):
        chooser = chooser_pool[np.random.randint(len(chooser_pool))]
        candidates = [g for g in candidate_pool if g is not chooser]
        if not candidates:
            continue
        chooser_scores = [
            _mate_preference_score(chooser, c, rank_index, policy_dist_cache)
            for c in candidates
        ]
        idx = _sample_softmax_index(chooser_scores, temperature)
        if idx is None:
            continue
        candidate = candidates[idx]
        score_ab = _mate_preference_score(chooser, candidate, rank_index, policy_dist_cache)
        score_ba = _mate_preference_score(candidate, chooser, rank_index, policy_dist_cache)
        if score_ab >= float(threshold) and score_ba >= float(threshold):
            return chooser, candidate, True

    return None, None, False


def _build_relative_pool(anchor: Genome,
                         ranked: List[Genome],
                         elitism: int,
                         lineage_tracker: LineageTracker | None = None,
                         require_common_ancestor: bool = False,
                         ancestry_depth: int = 32,
                         topk: int | None = None) -> List[Genome]:
    """Return candidate mates, optionally filtered by common-ancestor relation."""
    del elitism  # kept in signature for backward compatibility
    # Minimal hard filter: only exclude self.
    pool = [g for g in ranked if g is not anchor]
    if not pool:
        return []

    if topk is not None and int(topk) > 0:
        limit = max(1, min(int(topk), len(pool)))
        pool = pool[:limit]

    if require_common_ancestor and lineage_tracker is not None:
        related = []
        max_depth = max(1, int(ancestry_depth))
        for cand in pool:
            try:
                dist = lineage_tracker.lineage_distance(anchor, cand, max_depth=max_depth)
            except Exception:
                dist = None
            if dist is not None:
                related.append(cand)
        if related:
            pool = related
    return pool


def _select_random_relative_mate(anchor: Genome,
                                 ranked: List[Genome],
                                 elitism: int,
                                 lineage_tracker: LineageTracker | None = None,
                                 require_common_ancestor: bool = False,
                                 ancestry_depth: int = 32,
                                 topk: int | None = None) -> Genome | None:
    """Pick a random mate from the (optionally ancestry-filtered) relative pool."""
    pool = _build_relative_pool(
        anchor=anchor,
        ranked=ranked,
        elitism=elitism,
        lineage_tracker=lineage_tracker,
        require_common_ancestor=require_common_ancestor,
        ancestry_depth=ancestry_depth,
        topk=topk,
    )
    if not pool:
        return None
    return pool[int(np.random.randint(len(pool)))]


def _pair_head_prob(a: Genome, b: Genome) -> float:
    """Pair-head compatibility probability (mutator embedding -> scorer)."""
    return float(a.pair_head_probability(b))


def _select_pair_head_mate(anchor: Genome,
                           ranked: List[Genome],
                           elitism: int,
                           temperature: float = 1.0,
                           decision_boundary: float = 0.5,
                           lineage_tracker: LineageTracker | None = None,
                           require_common_ancestor: bool = False,
                           ancestry_depth: int = 32,
                           topk: int | None = None) -> tuple[Genome | None, float, np.ndarray | None]:
    """Pick highest-confidence mutual-pass mate from the relative pool.

    A candidate is eligible only if both directions pass the fixed boundary:
      p(anchor->cand) >= decision_boundary and p(cand->anchor) >= decision_boundary.

    Among eligible mates, choose the one with highest mutual confidence
    (p_ab * p_ba). This prefers pairings the classifier is most confident
    should be compatible.
    """
    pool = _build_relative_pool(
        anchor=anchor,
        ranked=ranked,
        elitism=elitism,
        lineage_tracker=lineage_tracker,
        require_common_ancestor=require_common_ancestor,
        ancestry_depth=ancestry_depth,
        topk=topk,
    )
    if not pool:
        return None, 0.0, None

    gate = float(np.clip(float(decision_boundary), 0.01, 0.99))
    eligible = []
    mutual_scores = []
    mutual_vals = []
    for cand in pool:
        p_ab = _pair_head_prob(anchor, cand)
        p_ba = _pair_head_prob(cand, anchor)
        mutual = float(p_ab * p_ba)
        mutual_vals.append(mutual)
        if (p_ab >= gate) and (p_ba >= gate):
            eligible.append(cand)
            mutual_scores.append(mutual)

    mutual_probs = np.asarray(mutual_vals, dtype=np.float64)
    if not eligible:
        return None, 0.0, mutual_probs

    # Hard accept/reject gate decides eligibility; among passing candidates choose
    # the highest-confidence mate (max mutual score).
    del temperature  # kept in signature for CLI/backward compatibility.
    best_idx = int(np.argmax(np.asarray(mutual_scores, dtype=np.float64)))
    mate = eligible[best_idx]
    mutual = float(mutual_scores[best_idx])
    return mate, mutual, mutual_probs


def _clip_compat_gate(value: float) -> float:
    return float(np.clip(float(value), 0.05, 2.0))


def _update_compat_rate_bias(current_bias: float,
                             compat_rate: float,
                             target_low: float,
                             target_high: float,
                             adjust: float,
                             max_abs_bias: float = 0.35) -> float:
    """Adjust run-level crossover gate bias toward compatibility-rate target band."""
    new_bias = float(current_bias)
    if compat_rate < target_low:
        new_bias += float(adjust)
    elif compat_rate > target_high:
        new_bias -= float(adjust)
    else:
        # Gentle relaxation when inside target band to avoid drift lock-in.
        new_bias *= 0.9
    return float(np.clip(new_bias, -abs(max_abs_bias), abs(max_abs_bias)))


def evolve_generation(population: List[Genome], crossover_rate: float = 0.3,
                      elitism: int = 5, generation: int = 0,
                      max_generations: int = 100,
                      compat_threshold: float | None = 0.5,
                      learn_compat_threshold: bool = False,
                      compat_rate_guardrail: bool = False,
                      compat_rate_bias: float = 0.0,
                      mutation_decay: bool = False,
                      mate_choice: bool = False,
                      mate_choice_topk: int = 10,
                      mate_choice_threshold: float = 0.0,
                      mate_choice_temperature: float = 1.0,
                      lineage_tracker: LineageTracker | None = None,
                      lineage_min_distance: int = 0,
                      lineage_max_distance: int = -1,
                      lineage_distance_depth: int = 32,
                      lineage_assign_require_common_ancestor: bool = False,
                      lineage_assign_depth: int = 32,
                      survivor_fraction: float = 0.25) -> tuple:
    """Returns (new_population, species_info)."""
    pop_size = len(population)
    ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
    new_population = []
    child_generation = int(generation) + 1

    # Species assignment removed: keep diagnostics-only single species.
    species_info = _species_diagnostics_disabled(ranked)
    crossover_attempts = 0
    crossover_compatible = 0
    crossover_executed = 0
    crossover_rejected_pair_head = 0
    crossover_rejected_compat = 0
    crossover_rejected_lineage = 0
    bias = float(compat_rate_bias) if compat_rate_guardrail else 0.0
    mate_choice_attempts = 0
    mate_choice_accepts = 0
    mate_choice_fallbacks = 0
    mutator_affinity_cosines = []
    pair_head_pool_mutuals = []
    pair_head_selected_mutuals = []
    pair_head_selected_parent_ids = []
    pair_head_selected_pairs = []
    pair_head_selected_parent_species = []
    crossover_gate_alpha_means = []
    crossover_gate_alpha_stds = []
    lineage_gate_attempts = 0
    lineage_gate_blocked = 0
    crossover_edges = set()  # undirected parent-parent edges for behavioral-species graph

    # Survivors (avoid near-full cull): keep a configurable top fraction,
    # but never fewer than elitism and never all genomes.
    survivor_count = max(int(round(pop_size * float(survivor_fraction))), int(elitism))
    survivor_count = max(1, min(survivor_count, pop_size - 1))
    for i in range(survivor_count):
        elite = Genome(
            policy=ranked[i]._clone_policy(),
            mutator=ranked[i]._clone_mutator(),
            mutator_type=ranked[i].mutator_type,
            compat_net=ranked[i]._clone_compat() if ranked[i].compat_net else None,
            compat_threshold=ranked[i].compat_threshold,
            evolve_compat_threshold=ranked[i].evolve_compat_threshold,
            compat_mode=ranked[i].compat_mode,
        )
        elite.set_flat_weights(ranked[i].get_flat_weights())
        elite.fitness = ranked[i].fitness
        elite.self_replication_fidelity = ranked[i].self_replication_fidelity
        elite.species_id = ranked[i].species_id
        if lineage_tracker is not None:
            lineage_tracker.register_child(
                child=elite,
                parents=[ranked[i]],
                generation=child_generation,
                method='elite',
            )
        new_population.append(elite)

    while len(new_population) < pop_size:
        parent = tournament_select(ranked)
        if np.random.random() < crossover_rate:
            compat_mode = str(getattr(parent, 'compat_mode', 'distance'))
            if compat_mode == 'mutator_pair_head':
                # Relative-pool selection with hard mutual pass/fail, then pick
                # highest-confidence compatible mate.
                other, _mutual, _pool_mutual = _select_pair_head_mate(
                    anchor=parent,
                    ranked=ranked,
                    elitism=elitism,
                    temperature=max(float(mate_choice_temperature), 1e-6),
                    lineage_tracker=lineage_tracker,
                    require_common_ancestor=bool(lineage_assign_require_common_ancestor),
                    ancestry_depth=int(lineage_assign_depth),
                    topk=None,
                )
                if _pool_mutual is not None and _pool_mutual.size:
                    pair_head_pool_mutuals.extend(_pool_mutual.tolist())
                crossover_attempts += 1
                compatible = other is not None
                if not compatible:
                    crossover_rejected_pair_head += 1
                if compatible:
                    pair_head_selected_mutuals.append(float(_mutual))
                    pair_head_selected_parent_ids.extend([id(parent), id(other)])
                    pair_head_selected_pairs.append(tuple(sorted((id(parent), id(other)))))
                    pair_head_selected_parent_species.extend([
                        int(getattr(parent, 'species_id', -1)),
                        int(getattr(other, 'species_id', -1)),
                    ])
            else:
                if mate_choice:
                    mate_choice_attempts += 1
                    pair_attempts = max(4, min(16, int(mate_choice_topk)))
                    parent_mc, other_mc, accepted = _select_mate_choice_pair(
                        ranked=ranked,
                        elitism=elitism,
                        topk=mate_choice_topk,
                        threshold=mate_choice_threshold,
                        temperature=mate_choice_temperature,
                        max_attempts=pair_attempts,
                    )
                    if accepted and parent_mc is not None and other_mc is not None:
                        parent = parent_mc
                        other = other_mc
                        mate_choice_accepts += 1
                    else:
                        parent = tournament_select(ranked)
                        other = _select_random_relative_mate(
                            anchor=parent,
                            ranked=ranked,
                            elitism=elitism,
                            lineage_tracker=lineage_tracker,
                            require_common_ancestor=bool(lineage_assign_require_common_ancestor),
                            ancestry_depth=int(lineage_assign_depth),
                            topk=mate_choice_topk,
                        )
                        if other is None:
                            other = tournament_select(ranked)
                        mate_choice_fallbacks += 1
                else:
                    other = _select_random_relative_mate(
                        anchor=parent,
                        ranked=ranked,
                        elitism=elitism,
                        lineage_tracker=lineage_tracker,
                        require_common_ancestor=bool(lineage_assign_require_common_ancestor),
                        ancestry_depth=int(lineage_assign_depth),
                    )
                    if other is None:
                        other = tournament_select(ranked)
                crossover_attempts += 1
                if compat_mode == 'mutator_affinity':
                    try:
                        ma = parent.get_flat_mutator_weights()
                        mb = other.get_flat_mutator_weights()
                        denom = (torch.norm(ma) * torch.norm(mb)).item()
                        cos = 0.0 if denom <= 1e-12 else float(torch.dot(ma, mb).item() / denom)
                        mutator_affinity_cosines.append(cos)
                    except Exception:
                        pass
                # Check learned compatibility (bidirectional — both must agree)
                if learn_compat_threshold:
                    if compat_rate_guardrail:
                        gate_parent = _clip_compat_gate(parent.compat_threshold + bias)
                        gate_other = _clip_compat_gate(other.compat_threshold + bias)
                        compatible = parent.is_compatible(other, gate_parent) and other.is_compatible(parent, gate_other)
                    else:
                        compatible = parent.is_compatible(other, None) and other.is_compatible(parent, None)
                else:
                    gate = _clip_compat_gate(float(compat_threshold) + bias) if compat_rate_guardrail else compat_threshold
                    compatible = parent.is_compatible(other, gate) and other.is_compatible(parent, gate)
                if not compatible:
                    crossover_rejected_compat += 1
            if compatible and lineage_tracker is not None and int(lineage_min_distance) > 0:
                lineage_gate_attempts += 1
                try:
                    dist = lineage_tracker.lineage_distance(
                        parent,
                        other,
                        max_depth=max(int(lineage_distance_depth), 1),
                    )
                except Exception:
                    dist = None
                too_close = (dist is not None and dist < int(lineage_min_distance))
                too_far = (
                    int(lineage_max_distance) >= 0 and dist is not None and dist > int(lineage_max_distance)
                )
                if too_close or too_far:
                    compatible = False
                    lineage_gate_blocked += 1
                    crossover_rejected_lineage += 1

            if compatible:
                crossover_compatible += 1
                try:
                    child = parent.crossover(other, generation, max_generations,
                                             mutation_decay=mutation_decay)
                    child_method = 'crossover'
                    crossover_executed += 1
                    crossover_edges.add(tuple(sorted((id(parent), id(other)))))
                except Exception:
                    child = parent.reproduce(generation, max_generations,
                                             mutation_decay=mutation_decay)
                    child_method = 'mutation_fallback'
            else:
                # Incompatible — reproduce asexually instead
                try:
                    child = parent.reproduce(generation, max_generations,
                                             mutation_decay=mutation_decay)
                    child_method = 'mutation_incompatible'
                except Exception:
                    child = parent.reproduce(generation, max_generations,
                                             mutation_decay=mutation_decay)
                    child_method = 'mutation_error'
        else:
            try:
                child = parent.reproduce(generation, max_generations,
                                         mutation_decay=mutation_decay)
                child_method = 'mutation'
            except Exception:
                child = Genome(
                    policy=parent._clone_policy(),
                    mutator=parent._clone_mutator(),
                    mutator_type=parent.mutator_type,
                    compat_net=parent._clone_compat() if parent.compat_net else None,
                    compat_threshold=parent.compat_threshold,
                    evolve_compat_threshold=parent.evolve_compat_threshold,
                    compat_mode=parent.compat_mode,
                )
                flat = parent.get_flat_weights()
                noise = torch.randn_like(flat) * 0.01
                child.set_flat_weights(flat + noise)
                child_method = 'mutation_fallback'
        if child_method == 'crossover':
            crossover_gate_alpha_means.append(float(getattr(child, 'crossover_gate_alpha_mean', 0.5)))
            crossover_gate_alpha_stds.append(float(getattr(child, 'crossover_gate_alpha_std', 0.0)))
        if lineage_tracker is not None:
            parents = [parent]
            if child_method == 'crossover':
                parents = [parent, other]
            lineage_tracker.register_child(
                child=child,
                parents=parents,
                generation=child_generation,
                method=child_method,
            )
        new_population.append(child)

    species_info['crossover_attempts'] = crossover_attempts
    species_info['crossover_compatible'] = crossover_compatible
    species_info['crossover_executed'] = crossover_executed
    species_info['crossover_rejected_pair_head'] = crossover_rejected_pair_head
    species_info['crossover_rejected_compat'] = crossover_rejected_compat
    species_info['crossover_rejected_lineage'] = crossover_rejected_lineage
    species_info['compat_bias'] = float(bias)
    if learn_compat_threshold:
        effective = np.array([_clip_compat_gate(g.compat_threshold + bias) for g in ranked], dtype=np.float64)
    else:
        effective = np.array([_clip_compat_gate(float(compat_threshold) + bias)], dtype=np.float64)
    species_info['compat_gate_mean'] = float(effective.mean())
    species_info['compat_gate_std'] = float(effective.std())
    species_info['compat_gate_min'] = float(effective.min())
    species_info['compat_gate_max'] = float(effective.max())
    species_info['mate_choice_enabled'] = bool(mate_choice)
    species_info['mate_choice_attempts'] = int(mate_choice_attempts)
    species_info['mate_choice_accepts'] = int(mate_choice_accepts)
    species_info['mate_choice_fallbacks'] = int(mate_choice_fallbacks)
    species_info['mate_choice_accept_rate'] = (
        float(mate_choice_accepts) / float(max(mate_choice_attempts, 1))
    )
    species_info['mate_choice_fallback_rate'] = (
        float(mate_choice_fallbacks) / float(max(mate_choice_attempts, 1))
    )
    species_info['lineage_gate_min_distance'] = int(lineage_min_distance)
    species_info['lineage_gate_max_distance'] = int(lineage_max_distance)
    species_info['lineage_gate_depth'] = int(lineage_distance_depth)
    species_info['lineage_gate_attempts'] = int(lineage_gate_attempts)
    species_info['lineage_gate_blocked'] = int(lineage_gate_blocked)
    species_info['lineage_gate_block_rate'] = (
        float(lineage_gate_blocked) / float(max(lineage_gate_attempts, 1))
    )
    if mutator_affinity_cosines:
        arr = np.asarray(mutator_affinity_cosines, dtype=np.float64)
        species_info['mutator_affinity_cos_mean'] = float(arr.mean())
        species_info['mutator_affinity_cos_std'] = float(arr.std())
        species_info['mutator_affinity_cos_p10'] = float(np.quantile(arr, 0.10))
        species_info['mutator_affinity_cos_p50'] = float(np.quantile(arr, 0.50))
        species_info['mutator_affinity_cos_p90'] = float(np.quantile(arr, 0.90))
        species_info['mutator_affinity_pass_rate'] = float((arr >= 0.0).mean())
    else:
        species_info['mutator_affinity_cos_mean'] = 0.0
        species_info['mutator_affinity_cos_std'] = 0.0
        species_info['mutator_affinity_cos_p10'] = 0.0
        species_info['mutator_affinity_cos_p50'] = 0.0
        species_info['mutator_affinity_cos_p90'] = 0.0
        species_info['mutator_affinity_pass_rate'] = 0.0

    if pair_head_pool_mutuals:
        pool = np.asarray(pair_head_pool_mutuals, dtype=np.float64)
        species_info['pair_head_pool_mean'] = float(pool.mean())
        species_info['pair_head_pool_std'] = float(pool.std())
        species_info['pair_head_pool_p10'] = float(np.quantile(pool, 0.10))
        species_info['pair_head_pool_p50'] = float(np.quantile(pool, 0.50))
        species_info['pair_head_pool_p90'] = float(np.quantile(pool, 0.90))
    else:
        species_info['pair_head_pool_mean'] = 0.0
        species_info['pair_head_pool_std'] = 0.0
        species_info['pair_head_pool_p10'] = 0.0
        species_info['pair_head_pool_p50'] = 0.0
        species_info['pair_head_pool_p90'] = 0.0

    if pair_head_selected_mutuals:
        sel = np.asarray(pair_head_selected_mutuals, dtype=np.float64)
        species_info['pair_head_selected_mean'] = float(sel.mean())
        species_info['pair_head_selected_std'] = float(sel.std())
        species_info['pair_head_selected_p10'] = float(np.quantile(sel, 0.10))
        species_info['pair_head_selected_p50'] = float(np.quantile(sel, 0.50))
        species_info['pair_head_selected_p90'] = float(np.quantile(sel, 0.90))
    else:
        species_info['pair_head_selected_mean'] = 0.0
        species_info['pair_head_selected_std'] = 0.0
        species_info['pair_head_selected_p10'] = 0.0
        species_info['pair_head_selected_p50'] = 0.0
        species_info['pair_head_selected_p90'] = 0.0

    species_info['pair_head_selected_minus_pool_mean'] = (
        float(species_info['pair_head_selected_mean']) - float(species_info['pair_head_pool_mean'])
    )
    species_info['pair_head_unique_parent_rate'] = (
        float(len(set(pair_head_selected_parent_ids))) / float(max(len(pair_head_selected_parent_ids), 1))
    )
    species_info['pair_head_unique_pair_rate'] = (
        float(len(set(pair_head_selected_pairs))) / float(max(len(pair_head_selected_pairs), 1))
    )
    if crossover_gate_alpha_means:
        alpha_mean_arr = np.asarray(crossover_gate_alpha_means, dtype=np.float64)
        alpha_std_arr = np.asarray(crossover_gate_alpha_stds, dtype=np.float64)
        species_info['crossover_gate_alpha_mean'] = float(alpha_mean_arr.mean())
        species_info['crossover_gate_alpha_std'] = float(alpha_mean_arr.std())
        species_info['crossover_gate_alpha_chunkstd_mean'] = float(alpha_std_arr.mean())
        species_info['crossover_gate_alpha_chunkstd_std'] = float(alpha_std_arr.std())
    else:
        species_info['crossover_gate_alpha_mean'] = 0.0
        species_info['crossover_gate_alpha_std'] = 0.0
        species_info['crossover_gate_alpha_chunkstd_mean'] = 0.0
        species_info['crossover_gate_alpha_chunkstd_std'] = 0.0
    if pair_head_selected_parent_species:
        vals = np.asarray(pair_head_selected_parent_species, dtype=np.int64)
        counts = np.bincount(vals - vals.min())
        probs = counts[counts > 0].astype(np.float64)
        probs /= probs.sum()
        species_info['pair_head_parent_species_entropy'] = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        species_info['pair_head_parent_species_entropy'] = 0.0

    # Behavioral-species graph (what actually crossed this generation).
    # Nodes are ranked parents; undirected edges are successful crossover pairs.
    n_nodes = len(ranked)
    id_to_idx = {id(g): i for i, g in enumerate(ranked)}
    adj = [set() for _ in range(n_nodes)]
    edge_count = 0
    for a_id, b_id in crossover_edges:
        ia = id_to_idx.get(a_id)
        ib = id_to_idx.get(b_id)
        if ia is None or ib is None or ia == ib:
            continue
        if ib not in adj[ia]:
            adj[ia].add(ib)
            adj[ib].add(ia)
            edge_count += 1

    visited = [False] * n_nodes
    comp_sizes = []
    isolated = 0
    for i in range(n_nodes):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        sz = 0
        while stack:
            u = stack.pop()
            sz += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comp_sizes.append(sz)
        if sz == 1:
            isolated += 1

    denom = max((n_nodes * (n_nodes - 1)) // 2, 1)
    species_info['behavioral_components'] = int(len(comp_sizes))
    species_info['behavioral_largest_component'] = int(max(comp_sizes) if comp_sizes else 0)
    species_info['behavioral_largest_component_frac'] = float((max(comp_sizes) if comp_sizes else 0) / max(n_nodes, 1))
    species_info['behavioral_isolated_nodes'] = int(isolated)
    species_info['behavioral_isolated_frac'] = float(isolated / max(n_nodes, 1))
    species_info['behavioral_edge_density'] = float(edge_count / denom)

    return new_population, species_info


def run_evolution(env_id: str = 'CartPole-v1', pop_size: int = 30,

                  generations: int = 100, mutator_type: str = 'dualmixture',
                  n_eval_episodes: int = 5, crossover_rate: float = 0.3,
                  hidden: int = 64, chunk_size: int = 64,
                  log_interval: int = 1, seed: int = 42,
                  speciation: bool = False,
                  compat_threshold: float = 0.5,
                  learn_compat_threshold: bool = False,
                  compat_binary: bool = False,
                  compat_pretrain: bool = True,
                  unified_mating: bool = False,
                  unified_mate_head: bool = False,
                  compat_rate_guardrail: bool = False,
                  compat_rate_target_low: float = 0.35,
                  compat_rate_target_high: float = 0.75,
                  compat_rate_adjust: float = 0.03,
                  flex: bool = False,
                  policy_arch: str = 'mlp',
                  complexity_cost: float = 0.0,
                  output_dir: str = None,
                  n_workers: int = 1,
                  fleet=None,
                  progress_callback=None,
                  mutator_kwargs: dict | None = None,
                  init_genome_path: str | None = None,
                  init_mutator_from_genome_path: str | None = None,
                  mutation_decay: bool = False,
                  mate_choice: bool = False,
                  mate_choice_topk: int = 10,
                  mate_choice_threshold: float = 0.0,
                  mate_choice_temperature: float = 1.0,
                  lineage_min_distance: int = 0,
                  lineage_max_distance: int = -1,
                  lineage_distance_depth: int = 32,
                  lineage_assign_require_common_ancestor: bool = False,
                  lineage_assign_depth: int = 32,
                  survivor_fraction: float = 0.25) -> Dict:
    """
    Main evolution loop with true self-replication.

    The mutator processes the full genome (policy + mutator weights) and
    outputs a new full genome. Natural selection is the only filter —
    mutators that destroy themselves die, those that improve themselves thrive.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    ensure_snake_registered()

    survivor_fraction = float(np.clip(survivor_fraction, 0.01, 0.95))

    if env_id.startswith('Vizdoom'):
        import vizdoom.gymnasium_wrapper  # registers vizdoom env ids
    env = gym.make(env_id)

    # Support plain Box observations and Dict wrappers (VizDoom).
    if hasattr(env.observation_space, 'shape') and env.observation_space.shape is not None:
        obs_shape = env.observation_space.shape
    else:
        obs0, _ = env.reset()
        obs_shape = np.asarray(_extract_obs(obs0)).shape

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
                                   compat_threshold=compat_threshold,
                                   learn_compat_threshold=learn_compat_threshold,
                                   compat_binary=compat_binary,
                                   compat_pretrain=compat_pretrain,
                                   unified_mating=unified_mating,
                                   unified_mate_head=unified_mate_head,
                                   flex=flex, policy_arch=policy_arch, output_dir=output_dir,
                                   mutator_kwargs=mutator_kwargs,
                                   obs_shape=obs_shape if len(obs_shape) == 3 else None,
                                   init_genome_path=init_genome_path,
                                   init_mutator_from_genome_path=init_mutator_from_genome_path)
    lineage_tracker = LineageTracker(run_metadata={
        'env': env_id,
        'seed': int(seed),
        'pop_size': int(pop_size),
        'generations': int(generations),
        'mutator': mutator_type,
        'speciation': bool(speciation),
        'lineage_min_distance': int(lineage_min_distance),
        'lineage_max_distance': int(lineage_max_distance),
        'lineage_distance_depth': int(lineage_distance_depth),
        'lineage_assign_require_common_ancestor': bool(lineage_assign_require_common_ancestor),
        'lineage_assign_depth': int(lineage_assign_depth),
    })
    lineage_tracker.register_initial_population(population, generation=0)
    if init_genome_path:
        print(f"Initialized population from seed genome: {init_genome_path}")
    if init_mutator_from_genome_path:
        print(f"Initialized mutator weights from seed genome: {init_mutator_from_genome_path}")
    genome = population[0]
    print(f"Genome: {genome.num_policy_params()} policy params + "
          f"{genome.num_mutator_params()} mutator params + "
          f"{genome.num_compat_params()} compat params = "
          f"{genome.num_total_params()} total")
    print(f"Mutator type: {mutator_type}")
    print(f"Policy arch: {policy_arch}")
    print(f"Speciation: {'learned' if speciation else 'off'}")
    if speciation and unified_mating:
        if unified_mate_head:
            print('Mating policy: unified (learned mutator pair-head)')
        else:
            print('Mating policy: unified (mutator-affinity)')
    if speciation and (not unified_mating):
        if learn_compat_threshold:
            print(f"Compat threshold mode: evolvable per-genome (init={compat_threshold:.3f})")
        else:
            print(f"Compat threshold mode: fixed ({compat_threshold:.3f})")
        if compat_pretrain:
            print(f"Compat pretrain: {'binary-classifier' if compat_binary else 'contrastive'}")
        else:
            print("Compat pretrain: off (cold-start)")
        if compat_rate_guardrail:
            print("Compat-rate guardrail: on "
                  f"(target={compat_rate_target_low:.2f}-{compat_rate_target_high:.2f}, "
                  f"adjust={compat_rate_adjust:.3f}, max|bias|=0.35)")
        else:
            print("Compat-rate guardrail: off")
    print(f"Population: {pop_size}, Generations: {generations}")
    print(f"Survivor fraction: {survivor_fraction:.2f}")
    print(f"Mutation decay: {'on (legacy schedule)' if mutation_decay else 'off (constant mutation scale)'}")
    print(f"Mate choice: {'on' if mate_choice else 'off'}")
    if int(lineage_min_distance) > 0 or int(lineage_max_distance) >= 0:
        max_txt = 'off' if int(lineage_max_distance) < 0 else str(int(lineage_max_distance))
        print(
            f"Lineage gate: on (min_dist={int(lineage_min_distance)}, "
            f"max_dist={max_txt}, depth={int(lineage_distance_depth)})"
        )
    if bool(lineage_assign_require_common_ancestor):
        print(
            f"Lineage assignment filter: on "
            f"(require_common_ancestor=True, depth={int(lineage_assign_depth)})"
        )
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
        # Policy-weight scale tracking (to detect implicit shrinkage/expansion)
        'mean_policy_l2': [],
        'max_policy_l2': [],
        'best_policy_l2': [],
        'mean_policy_abs': [],
        'best_policy_abs': [],
        'mean_policy_p90_abs': [],
        'best_policy_p90_abs': [],
        'num_species': [],
        'species_entropy': [],
        'species_tag_dist_mean': [],
        'species_tag_dist_std': [],
        'crossover_attempts': [],
        'crossover_compatible': [],
        'crossover_executed': [],
        'crossover_compat_rate': [],
        'crossover_executed_rate': [],
        'crossover_rejected_pair_head': [],
        'crossover_rejected_compat': [],
        'crossover_rejected_lineage': [],
        'crossover_gate_alpha_mean': [],
        'crossover_gate_alpha_std': [],
        'crossover_gate_alpha_chunkstd_mean': [],
        'crossover_gate_alpha_chunkstd_std': [],
        'mean_compat_threshold': [],
        'std_compat_threshold': [],
        'compat_guardrail_bias': [],
        'compat_effective_gate_mean': [],
        'compat_effective_gate_std': [],
        'compat_effective_gate_min': [],
        'compat_effective_gate_max': [],
        'mutator_affinity_cos_mean': [],
        'mutator_affinity_cos_std': [],
        'mutator_affinity_cos_p10': [],
        'mutator_affinity_cos_p50': [],
        'mutator_affinity_cos_p90': [],
        'mutator_affinity_pass_rate': [],
        'pair_head_pool_mean': [],
        'pair_head_pool_std': [],
        'pair_head_pool_p10': [],
        'pair_head_pool_p50': [],
        'pair_head_pool_p90': [],
        'pair_head_selected_mean': [],
        'pair_head_selected_std': [],
        'pair_head_selected_p10': [],
        'pair_head_selected_p50': [],
        'pair_head_selected_p90': [],
        'pair_head_selected_minus_pool_mean': [],
        'pair_head_unique_parent_rate': [],
        'pair_head_unique_pair_rate': [],
        'pair_head_parent_species_entropy': [],
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
        'mate_choice_enabled': bool(mate_choice),
        'mate_choice_accept_rate': [],
        'mate_choice_fallback_rate': [],
        'lineage_gate_attempts': [],
        'lineage_gate_blocked': [],
        'lineage_gate_block_rate': [],
        'lineage_assign_checks': [],
        'lineage_assign_filtered': [],
        'lineage_assign_filter_rate': [],
        # Behavioral-species metrics from crossover graph connectivity
        'behavioral_components': [],
        'behavioral_largest_component': [],
        'behavioral_largest_component_frac': [],
        'behavioral_isolated_nodes': [],
        'behavioral_isolated_frac': [],
        'behavioral_edge_density': [],
    }

    if complexity_cost > 0:
        print(f"Complexity cost: {complexity_cost} per parameter")
    if n_workers > 1:
        print(f"Parallel evaluation: {n_workers} workers")

    # ── Helper: process results for a generation ──────────────────
    compat_rate_bias = 0.0

    def _process_gen(gen, population, raw_fitnesses, eval_profile, gen_t0):
        """Score population, log stats, evolve next gen. Returns (new_pop, species_info, evolution_step_sec, logging_checkpoint_sec)."""
        nonlocal compat_rate_bias
        current_compat = None if learn_compat_threshold else compat_threshold

        for g, raw in zip(population, raw_fitnesses):
            g.fitness = float(raw)

        # Complexity penalty
        if complexity_cost > 0:
            for g in population:
                penalty = complexity_cost * g.num_policy_params()
                g.fitness -= penalty

        # Species remains diagnostics-only; mating/reproduction decisions use
        # compatibility gates and parent-relative proximity, not species sizes.
        fitnesses = [g.fitness for g in population]
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
        policy_l2 = []
        policy_mean_abs = []
        policy_p90_abs = []
        for i, g in enumerate(population):
            current_mutator = g.get_flat_mutator_weights()
            ref_idx = i % len(initial_mutator_weights)
            drift = torch.norm(current_mutator - initial_mutator_weights[ref_idx]).item()
            drifts.append(drift)

            flat_policy = g.get_flat_policy_weights()
            abs_policy = torch.abs(flat_policy)
            policy_l2.append(torch.norm(flat_policy).item())
            policy_mean_abs.append(abs_policy.mean().item())
            policy_p90_abs.append(torch.quantile(abs_policy, 0.90).item())

        has_flex_layers = flex and hasattr(population[0].policy, 'layer_sizes')
        if has_flex_layers:
            layers_list = [len(g.policy.layer_sizes) for g in population if hasattr(g.policy, 'layer_sizes')]
            neurons_list = [sum(g.policy.layer_sizes) for g in population if hasattr(g.policy, 'layer_sizes')]
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
        history['mean_policy_l2'].append(float(np.mean(policy_l2)))
        history['max_policy_l2'].append(float(np.max(policy_l2)))
        history['best_policy_l2'].append(float(policy_l2[best_idx]))
        history['mean_policy_abs'].append(float(np.mean(policy_mean_abs)))
        history['best_policy_abs'].append(float(policy_mean_abs[best_idx]))
        history['mean_policy_p90_abs'].append(float(np.mean(policy_p90_abs)))
        history['best_policy_p90_abs'].append(float(policy_p90_abs[best_idx]))

        if gen % log_interval == 0:
            msg = (f"Gen {gen:4d} | Best: {best_fit:8.2f} | Mean: {mean_fit:8.2f} | "
                   f"Fidelity: {np.mean(fidelities):.4f} | Drift: {np.mean(drifts):.4f} | "
                   f"|w|μ(best/all): {policy_mean_abs[best_idx]:.4f}/{np.mean(policy_mean_abs):.4f} | "
                   f"||w||₂(best/all): {policy_l2[best_idx]:.2f}/{np.mean(policy_l2):.2f}")
            if has_flex_layers:
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
            max_generations=generations,
            compat_threshold=current_compat,
            learn_compat_threshold=learn_compat_threshold,
            compat_rate_guardrail=compat_rate_guardrail,
            compat_rate_bias=compat_rate_bias,
            mutation_decay=mutation_decay,
            mate_choice=mate_choice,
            mate_choice_topk=mate_choice_topk,
            mate_choice_threshold=mate_choice_threshold,
            mate_choice_temperature=mate_choice_temperature,
            lineage_tracker=lineage_tracker,
            lineage_min_distance=lineage_min_distance,
            lineage_max_distance=lineage_max_distance,
            lineage_distance_depth=lineage_distance_depth,
            lineage_assign_require_common_ancestor=lineage_assign_require_common_ancestor,
            lineage_assign_depth=lineage_assign_depth,
            survivor_fraction=survivor_fraction,
        )
        evolution_step_sec = time.time() - t_evolve0
        lineage_tracker.record_population_snapshot(population, generation=gen)

        # Record species/perf stats
        history['num_species'].append(species_info['num_species'])
        history['species_entropy'].append(float(species_info.get('species_entropy', 0.0)))
        history['species_tag_dist_mean'].append(float(species_info.get('tag_dist_mean', 0.0)))
        history['species_tag_dist_std'].append(float(species_info.get('tag_dist_std', 0.0)))
        crossover_attempts_gen = float(species_info.get('crossover_attempts', 0))
        crossover_compatible_gen = float(species_info.get('crossover_compatible', 0))
        crossover_executed_gen = float(species_info.get('crossover_executed', 0))
        history['crossover_attempts'].append(crossover_attempts_gen)
        history['crossover_compatible'].append(crossover_compatible_gen)
        history['crossover_executed'].append(crossover_executed_gen)
        compat_rate = (crossover_compatible_gen / max(crossover_attempts_gen, 1.0))
        history['crossover_compat_rate'].append(compat_rate)
        history['crossover_executed_rate'].append(
            crossover_executed_gen / float(max(crossover_attempts_gen, 1.0))
        )
        history['crossover_rejected_pair_head'].append(float(species_info.get('crossover_rejected_pair_head', 0)))
        history['crossover_rejected_compat'].append(float(species_info.get('crossover_rejected_compat', 0)))
        history['crossover_rejected_lineage'].append(float(species_info.get('crossover_rejected_lineage', 0)))
        history['crossover_gate_alpha_mean'].append(float(species_info.get('crossover_gate_alpha_mean', 0.0)))
        history['crossover_gate_alpha_std'].append(float(species_info.get('crossover_gate_alpha_std', 0.0)))
        history['crossover_gate_alpha_chunkstd_mean'].append(float(species_info.get('crossover_gate_alpha_chunkstd_mean', 0.0)))
        history['crossover_gate_alpha_chunkstd_std'].append(float(species_info.get('crossover_gate_alpha_chunkstd_std', 0.0)))
        thresholds = [float(getattr(g, 'compat_threshold', compat_threshold)) for g in population]
        history['mean_compat_threshold'].append(float(np.mean(thresholds)))
        history['std_compat_threshold'].append(float(np.std(thresholds)))
        history['compat_guardrail_bias'].append(float(species_info.get('compat_bias', 0.0)))
        history['compat_effective_gate_mean'].append(float(species_info.get('compat_gate_mean', 0.0)))
        history['compat_effective_gate_std'].append(float(species_info.get('compat_gate_std', 0.0)))
        history['compat_effective_gate_min'].append(float(species_info.get('compat_gate_min', 0.0)))
        history['compat_effective_gate_max'].append(float(species_info.get('compat_gate_max', 0.0)))
        history['mutator_affinity_cos_mean'].append(float(species_info.get('mutator_affinity_cos_mean', 0.0)))
        history['mutator_affinity_cos_std'].append(float(species_info.get('mutator_affinity_cos_std', 0.0)))
        history['mutator_affinity_cos_p10'].append(float(species_info.get('mutator_affinity_cos_p10', 0.0)))
        history['mutator_affinity_cos_p50'].append(float(species_info.get('mutator_affinity_cos_p50', 0.0)))
        history['mutator_affinity_cos_p90'].append(float(species_info.get('mutator_affinity_cos_p90', 0.0)))
        history['mutator_affinity_pass_rate'].append(float(species_info.get('mutator_affinity_pass_rate', 0.0)))
        history['pair_head_pool_mean'].append(float(species_info.get('pair_head_pool_mean', 0.0)))
        history['pair_head_pool_std'].append(float(species_info.get('pair_head_pool_std', 0.0)))
        history['pair_head_pool_p10'].append(float(species_info.get('pair_head_pool_p10', 0.0)))
        history['pair_head_pool_p50'].append(float(species_info.get('pair_head_pool_p50', 0.0)))
        history['pair_head_pool_p90'].append(float(species_info.get('pair_head_pool_p90', 0.0)))
        history['pair_head_selected_mean'].append(float(species_info.get('pair_head_selected_mean', 0.0)))
        history['pair_head_selected_std'].append(float(species_info.get('pair_head_selected_std', 0.0)))
        history['pair_head_selected_p10'].append(float(species_info.get('pair_head_selected_p10', 0.0)))
        history['pair_head_selected_p50'].append(float(species_info.get('pair_head_selected_p50', 0.0)))
        history['pair_head_selected_p90'].append(float(species_info.get('pair_head_selected_p90', 0.0)))
        history['pair_head_selected_minus_pool_mean'].append(float(species_info.get('pair_head_selected_minus_pool_mean', 0.0)))
        history['pair_head_unique_parent_rate'].append(float(species_info.get('pair_head_unique_parent_rate', 0.0)))
        history['pair_head_unique_pair_rate'].append(float(species_info.get('pair_head_unique_pair_rate', 0.0)))
        history['pair_head_parent_species_entropy'].append(float(species_info.get('pair_head_parent_species_entropy', 0.0)))
        history['mate_choice_accept_rate'].append(float(species_info.get('mate_choice_accept_rate', 0.0)))
        history['mate_choice_fallback_rate'].append(float(species_info.get('mate_choice_fallback_rate', 0.0)))
        history['lineage_gate_attempts'].append(float(species_info.get('lineage_gate_attempts', 0.0)))
        history['lineage_gate_blocked'].append(float(species_info.get('lineage_gate_blocked', 0.0)))
        history['lineage_gate_block_rate'].append(float(species_info.get('lineage_gate_block_rate', 0.0)))
        history['lineage_assign_checks'].append(float(species_info.get('lineage_assign_checks', 0.0)))
        history['lineage_assign_filtered'].append(float(species_info.get('lineage_assign_filtered', 0.0)))
        history['lineage_assign_filter_rate'].append(float(species_info.get('lineage_assign_filter_rate', 0.0)))
        history['behavioral_components'].append(float(species_info.get('behavioral_components', 0.0)))
        history['behavioral_largest_component'].append(float(species_info.get('behavioral_largest_component', 0.0)))
        history['behavioral_largest_component_frac'].append(float(species_info.get('behavioral_largest_component_frac', 0.0)))
        history['behavioral_isolated_nodes'].append(float(species_info.get('behavioral_isolated_nodes', 0.0)))
        history['behavioral_isolated_frac'].append(float(species_info.get('behavioral_isolated_frac', 0.0)))
        history['behavioral_edge_density'].append(float(species_info.get('behavioral_edge_density', 0.0)))

        if speciation and compat_rate_guardrail:
            compat_rate_bias = _update_compat_rate_bias(
                compat_rate_bias,
                compat_rate=compat_rate,
                target_low=compat_rate_target_low,
                target_high=compat_rate_target_high,
                adjust=compat_rate_adjust,
            )

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
            if learn_compat_threshold:
                thr_text = f"Threshold μ/σ: {history['mean_compat_threshold'][-1]:.4f}/{history['std_compat_threshold'][-1]:.4f}"
            else:
                thr_text = f"Threshold: {float(current_compat):.4f}"
            print(f"         Species: {species_info['num_species']} | "
                  f"Entropy: {species_info.get('species_entropy', 0.0):.3f} | "
                  f"TagDist μ/σ: {species_info.get('tag_dist_mean', 0.0):.3f}/{species_info.get('tag_dist_std', 0.0):.3f} | "
                  f"Compat rate: {compat_rate:.2f} | "
                  f"Gate μ/σ/min/max: {species_info.get('compat_gate_mean', 0.0):.4f}/"
                  f"{species_info.get('compat_gate_std', 0.0):.4f}/"
                  f"{species_info.get('compat_gate_min', 0.0):.4f}/"
                  f"{species_info.get('compat_gate_max', 0.0):.4f} | "
                  f"Bias(now→next): {species_info.get('compat_bias', 0.0):+.4f}→{compat_rate_bias:+.4f} | "
                  f"{thr_text} | "
                  f"MutAff cos μ/p50/p90/pass: {species_info.get('mutator_affinity_cos_mean', 0.0):.3f}/"
                  f"{species_info.get('mutator_affinity_cos_p50', 0.0):.3f}/"
                  f"{species_info.get('mutator_affinity_cos_p90', 0.0):.3f}/"
                  f"{species_info.get('mutator_affinity_pass_rate', 0.0):.2f} | "
                  f"PairHead pool/sel p50: {species_info.get('pair_head_pool_p50', 0.0):.3f}/"
                  f"{species_info.get('pair_head_selected_p50', 0.0):.3f} Δμ={species_info.get('pair_head_selected_minus_pool_mean', 0.0):+.3f} "
                  f"uniqParent={species_info.get('pair_head_unique_parent_rate', 0.0):.2f} "
                  f"uniqPair={species_info.get('pair_head_unique_pair_rate', 0.0):.2f} | "
                  f"Rej(ph/compat/lin): {int(species_info.get('crossover_rejected_pair_head', 0))}/"
                  f"{int(species_info.get('crossover_rejected_compat', 0))}/"
                  f"{int(species_info.get('crossover_rejected_lineage', 0))} "
                  f"GateAlpha μ/σ: {species_info.get('crossover_gate_alpha_mean', 0.0):.3f}/"
                  f"{species_info.get('crossover_gate_alpha_std', 0.0):.3f} "
                  f"ExecRate: {float(species_info.get('crossover_executed', 0))/float(max(species_info.get('crossover_attempts', 1), 1)):.2f} | "
                  f"BhvComp: {int(species_info.get('behavioral_components', 0))} "
                  f"BhvLargest: {float(species_info.get('behavioral_largest_component_frac', 0.0)):.2f} "
                  f"BhvIso: {float(species_info.get('behavioral_isolated_frac', 0.0)):.2f} | "
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

            genome_cb = progress_callback.get_genome_callback() if hasattr(progress_callback, 'get_genome_callback') and progress_callback is not None else None

            if gen == 0:
                # First gen: synchronous (nothing to overlap with)
                raw_fitnesses, eval_profile = evaluate_population(
                    population, env_id, n_eval_episodes, n_workers=n_workers, fleet=fleet,
                    genome_callback=genome_cb
                )
            else:
                # Collect results dispatched at end of previous iteration
                raw_fitnesses, eval_profile = fleet.collect(genome_callback=genome_cb)

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
            genome_cb = progress_callback.get_genome_callback() if hasattr(progress_callback, 'get_genome_callback') and progress_callback is not None else None

            raw_fitnesses, eval_profile = evaluate_population(
                population, env_id, n_eval_episodes, n_workers=n_workers, fleet=fleet,
                genome_callback=genome_cb
            )
            population = _process_gen(gen, population, raw_fitnesses, eval_profile, gen_t0)

    if output_dir is not None:
        lineage_path, lineage_meta_path = lineage_tracker.save(output_dir)
        history['lineage_artifacts'] = {
            'lineage_jsonl': lineage_path,
            'lineage_meta_json': lineage_meta_path,
        }
        print(f"Lineage saved: {lineage_path}")
        print(f"Lineage metadata saved: {lineage_meta_path}")

    return history
