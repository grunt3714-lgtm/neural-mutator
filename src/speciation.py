"""Speciation utilities and diagnostics.

Extracted from evolution loop so species assignment and metrics are testable
and easier to iterate independently.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

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


def assign_species(population: List[Genome], compat_threshold: float = 0.5) -> Dict:
    """Assign species via learned tag-distance clustering and return diagnostics."""
    if population[0].compat_net is None:
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

    tags = [g.get_species_tag() for g in population]
    max_dist = compat_threshold * (CompatibilityNet.EMBED_DIM ** 0.5)

    representatives: list[tuple[int, torch.Tensor]] = []
    species_counter = 0

    for g, tag in zip(population, tags):
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
