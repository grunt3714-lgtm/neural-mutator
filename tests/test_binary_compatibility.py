"""Tests for binary compatibility classifier and graph-based speciation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evolution import create_population, evolve_generation
from src.genome import Genome, Policy, CompatibilityNet, create_mutator
from src.speciation import assign_species


def _make_genome_with_compat() -> Genome:
    policy = Policy(4, 2, 16)
    mutator = create_mutator('dualmixture', chunk_size=32)
    policy_params = sum(p.numel() for p in policy.parameters())
    mutator_params = sum(p.numel() for p in mutator.parameters())
    tmp = CompatibilityNet(policy_params + mutator_params)
    compat_params = sum(p.numel() for p in tmp.parameters())
    compat = CompatibilityNet(policy_params + mutator_params + compat_params)
    return Genome(policy, mutator, mutator_type='dualmixture', compat_net=compat)


def test_binary_compatibility_decision_uses_bidirectional_agreement(monkeypatch):
    g1 = _make_genome_with_compat()
    g2 = _make_genome_with_compat()

    monkeypatch.setattr(g1.compat_net, 'predict_pair_proba', lambda *_: 0.9)
    monkeypatch.setattr(g2.compat_net, 'predict_pair_proba', lambda *_: 0.6)
    details = g1.compatibility_details(g2)
    assert details['mean_prob'] == pytest.approx(0.75)
    assert g1.is_compatible(g2) is True

    monkeypatch.setattr(g2.compat_net, 'predict_pair_proba', lambda *_: 0.4)
    details = g1.compatibility_details(g2)
    assert details['mean_prob'] == pytest.approx(0.65)
    assert g1.is_compatible(g2) is False


def test_initial_calibration_targets_half_acceptance(tmp_path):
    pop = create_population(
        pop_size=14,
        obs_dim=4,
        act_dim=2,
        mutator_type='dualmixture',
        hidden=16,
        chunk_size=32,
        speciation=True,
        compat_pretrain_steps=40,
        compat_pretrain_batch_size=32,
        compat_calibration_target=0.5,
        compat_calibration_pairs=256,
        output_dir=str(tmp_path),
    )
    info = assign_species(pop, compat_threshold=0.5)
    assert 0.35 <= info['pair_accept_rate'] <= 0.65
    assert abs(info['compat_calibration_drift']) <= 0.20


def test_assign_species_uses_compatibility_graph_components(monkeypatch):
    pop = [_make_genome_with_compat() for _ in range(4)]
    index = {id(g): i for i, g in enumerate(pop)}

    adjacency = {
        (0, 1): True,
        (0, 2): False,
        (0, 3): False,
        (1, 2): False,
        (1, 3): False,
        (2, 3): True,
    }

    def _details(self, other):
        a, b = sorted((index[id(self)], index[id(other)]))
        ok = adjacency[(a, b)]
        p = 0.9 if ok else 0.1
        return {'p_ab': p, 'p_ba': p, 'mean_prob': p, 'compatible': float(1.0 if ok else 0.0)}

    monkeypatch.setattr(Genome, 'compatibility_details', _details)

    info = assign_species(pop, compat_threshold=0.5)
    assert info['num_species'] == 2
    assert sorted(info['species_sizes'].values()) == [2, 2]


def test_evolution_flow_still_runs_without_speciation(monkeypatch):
    pop = []
    for i in range(6):
        g = Genome(
            Policy(4, 2, 16),
            create_mutator('dualmixture', chunk_size=32),
            mutator_type='dualmixture',
            compat_net=None,
        )
        g.fitness = float(6 - i)
        pop.append(g)

    monkeypatch.setattr('src.evolution.np.random.random', lambda: 0.5)

    new_pop, info = evolve_generation(pop, crossover_rate=0.3, elitism=2)
    assert len(new_pop) == len(pop)
    assert info['num_species'] == 1
    assert info['crossover_attempts'] >= info['crossover_compatible']
