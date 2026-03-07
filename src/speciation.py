"""Speciation utilities and diagnostics.

Extracted from evolution loop so species assignment and metrics are testable
and easier to iterate independently.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch

from .genome import CompatibilityNet, Genome


def _species_entropy(species_sizes: Dict[int, int]) -> float:
    total = float(sum(species_sizes.values()))
    if total <= 0:
        return 0.0
    probs = np.array([v / total for v in species_sizes.values()], dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(max(0.0, -(probs * np.log(probs)).sum()))


def _pairwise_tag_distances(tags: List[torch.Tensor]) -> np.ndarray:
    n = len(tags)
    if n < 2:
        return np.array([], dtype=np.float64)
    dists = []
    for i in range(n):
        ti = tags[i]
        for j in range(i + 1, n):
            dists.append(float(torch.norm(ti - tags[j]).item()))
    return np.array(dists, dtype=np.float64)


def assign_species(population: List[Genome],
                   compat_threshold: float | None = 0.5,
                   lineage_tracker: Any | None = None,
                   lineage_assign_require_common_ancestor: bool = False,
                   lineage_assign_depth: int = 32) -> Dict:
    """Assign species via learned tag-distance clustering and return diagnostics.

    If compat_threshold is None, use per-genome learned thresholds.
    """
    mode0 = str(getattr(population[0], 'compat_mode', 'distance'))
    if population[0].compat_net is None and mode0 not in ('mutator_affinity', 'mutator_pair_head'):
        for g in population:
            g.species_id = 0
        species_sizes = {0: len(population)}
        return {
            'num_species': 1,
            'species_sizes': species_sizes,
            'species_entropy': _species_entropy(species_sizes),
            'tag_dist_mean': 0.0,
            'tag_dist_std': 0.0,
            'tag_dist_p50': 0.0,
            'crossover_attempts': 0,
            'crossover_compatible': 0,
        }

    if mode0 == 'mutator_pair_head':
        tags = [g.get_species_tag() for g in population]
    else:
        tags = [g.get_species_tag() for g in population] if population[0].compat_net is not None else [torch.zeros(CompatibilityNet.EMBED_DIM) for _ in population]

    if mode0 == 'mutator_pair_head':
        # Threshold-free graph clustering from mutual pair-head affinity.
        n = len(population)
        if n <= 1:
            for g in population:
                g.species_id = 0
            species_sizes = {0: n}
            return {
                'num_species': len(species_sizes),
                'species_sizes': species_sizes,
                'species_entropy': _species_entropy(species_sizes),
                'tag_dist_mean': 0.0,
                'tag_dist_std': 0.0,
                'tag_dist_p50': 0.0,
                'crossover_attempts': 0,
                'crossover_compatible': 0,
            }

        mutual = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                pij = float(population[i].pair_head_probability(population[j]))
                pji = float(population[j].pair_head_probability(population[i]))
                m = float(np.clip(pij * pji, 0.0, 1.0))
                mutual[i, j] = m
                mutual[j, i] = m

        k = max(1, int(np.sqrt(n)))
        k = min(k, n - 1)
        adj: list[set[int]] = [set() for _ in range(n)]
        for i in range(n):
            row = mutual[i].copy()
            row[i] = -np.inf
            nbr_idx = np.argpartition(row, -k)[-k:] if k > 0 else np.array([], dtype=int)
            nbr_idx = [int(j) for j in nbr_idx if np.isfinite(row[j])]
            for j in nbr_idx:
                adj[i].add(j)
                adj[j].add(i)

        sid = 0
        seen = [False] * n
        for i in range(n):
            if seen[i]:
                continue
            stack = [i]
            seen[i] = True
            while stack:
                u = stack.pop()
                population[u].species_id = sid
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            sid += 1

        species_sizes = dict(Counter(g.species_id for g in population))
        tag_dists = _pairwise_tag_distances(tags)
        return {
            'num_species': len(species_sizes),
            'species_sizes': species_sizes,
            'species_entropy': _species_entropy(species_sizes),
            'tag_dist_mean': float(tag_dists.mean()) if tag_dists.size else 0.0,
            'tag_dist_std': float(tag_dists.std()) if tag_dists.size else 0.0,
            'tag_dist_p50': float(np.percentile(tag_dists, 50)) if tag_dists.size else 0.0,
            'crossover_attempts': 0,
            'crossover_compatible': 0,
        }

    # Species assignment without sticky single-representative first-match:
    # evaluate against all current members of each species and choose best match.
    species_members: dict[int, list[int]] = {}
    species_counter = 0
    lineage_assign_checks = 0
    lineage_assign_filtered = 0

    for i, (g, tag) in enumerate(zip(population, tags)):
        mode = str(getattr(g, 'compat_mode', 'distance'))
        best_sid: int | None = None
        best_score = -1e18

        for sid, member_idxs in species_members.items():
            if not member_idxs:
                continue

            if lineage_assign_require_common_ancestor and lineage_tracker is not None:
                filtered_member_idxs = []
                for j in member_idxs:
                    lineage_assign_checks += 1
                    try:
                        d = lineage_tracker.lineage_distance(
                            g,
                            population[j],
                            max_depth=max(int(lineage_assign_depth), 1),
                        )
                    except Exception:
                        d = None
                    if d is not None:
                        filtered_member_idxs.append(j)
                    else:
                        lineage_assign_filtered += 1
                member_idxs_eff = filtered_member_idxs
            else:
                member_idxs_eff = member_idxs

            if not member_idxs_eff:
                continue

            if mode in ('binary_strict', 'mutator_pair_head'):
                passes = 0
                for j in member_idxs_eff:
                    m = population[j]
                    if g.is_compatible(m, None) and m.is_compatible(g, None):
                        passes += 1
                if passes <= 0:
                    continue
                score = float(passes) / float(len(member_idxs_eff))
                if score > best_score:
                    best_score = score
                    best_sid = sid
                continue

            if mode == 'mutator_affinity':
                passes = 0
                for j in member_idxs_eff:
                    m = population[j]
                    if g.is_compatible(m, None):
                        passes += 1
                if passes <= 0:
                    continue
                score = float(passes) / float(len(member_idxs_eff))
                if score > best_score:
                    best_score = score
                    best_sid = sid
                continue

            # distance mode: assign to the species with smallest normalized distance
            # among compatible members.
            best_ratio = None
            for j in member_idxs_eff:
                m = population[j]
                m_tag = tags[j]
                if compat_threshold is None:
                    thresh = 0.5 * (float(g.compat_threshold) + float(getattr(m, 'compat_threshold', 0.5)))
                else:
                    thresh = float(compat_threshold)
                max_dist = thresh * (CompatibilityNet.EMBED_DIM ** 0.5)
                if max_dist <= 1e-12:
                    continue
                ratio = float(torch.norm(tag - m_tag).item() / max_dist)
                if ratio < 1.0 and (best_ratio is None or ratio < best_ratio):
                    best_ratio = ratio
            if best_ratio is None:
                continue
            score = 1.0 - float(best_ratio)
            if score > best_score:
                best_score = score
                best_sid = sid

        if best_sid is None:
            g.species_id = species_counter
            species_members[species_counter] = [i]
            species_counter += 1
        else:
            g.species_id = best_sid
            species_members.setdefault(best_sid, []).append(i)

    species_sizes = dict(Counter(g.species_id for g in population))
    tag_dists = _pairwise_tag_distances(tags)

    return {
        'num_species': len(species_sizes),
        'species_sizes': species_sizes,
        'species_entropy': _species_entropy(species_sizes),
        'tag_dist_mean': float(tag_dists.mean()) if tag_dists.size else 0.0,
        'tag_dist_std': float(tag_dists.std()) if tag_dists.size else 0.0,
        'tag_dist_p50': float(np.percentile(tag_dists, 50)) if tag_dists.size else 0.0,
        'lineage_assign_require_common_ancestor': bool(lineage_assign_require_common_ancestor),
        'lineage_assign_depth': int(lineage_assign_depth),
        'lineage_assign_checks': int(lineage_assign_checks),
        'lineage_assign_filtered': int(lineage_assign_filtered),
        'lineage_assign_filter_rate': (
            float(lineage_assign_filtered) / float(max(lineage_assign_checks, 1))
        ),
        'crossover_attempts': 0,
        'crossover_compatible': 0,
    }
