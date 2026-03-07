"""Regression tests for genome reproduce/crossover/load across mutators."""

import io
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.genome import (
    CompatibilityNet,
    Genome,
    Policy,
    create_mutator,
    available_mutator_types,
)


@pytest.mark.parametrize('mutator_type', available_mutator_types())
def test_reproduce_works_for_all_mutators(mutator_type):
    policy = Policy(4, 2, 32)
    mutator = create_mutator(mutator_type, chunk_size=64)
    g = Genome(policy, mutator, mutator_type=mutator_type)

    child = g.reproduce(generation=1, max_generations=10)

    assert isinstance(child, Genome)
    assert child.mutator_type == mutator_type
    assert child.num_policy_params() == g.num_policy_params()
    assert child.num_mutator_params() == g.num_mutator_params()
    assert torch.isfinite(child.get_flat_weights()).all()


@pytest.mark.parametrize('mutator_type', available_mutator_types())
def test_crossover_works_for_all_mutators(mutator_type):
    g1 = Genome(Policy(4, 2, 32), create_mutator(mutator_type, chunk_size=64), mutator_type)
    g2 = Genome(Policy(4, 2, 32), create_mutator(mutator_type, chunk_size=64), mutator_type)
    g1.fitness = 2.0
    g2.fitness = 1.0

    child = g1.crossover(g2, generation=1, max_generations=10)

    assert isinstance(child, Genome)
    assert child.mutator_type == mutator_type
    assert child.num_policy_params() == g1.num_policy_params()
    assert child.num_mutator_params() == g1.num_mutator_params()
    assert torch.isfinite(child.get_flat_weights()).all()


@pytest.mark.parametrize('mutator_type', available_mutator_types())
def test_save_load_roundtrip_preserves_mutator_type(mutator_type):
    g = Genome(Policy(4, 2, 32), create_mutator(mutator_type, chunk_size=64), mutator_type)
    g.fitness = 12.34

    buf = io.BytesIO()
    g.save(buf)
    buf.seek(0)

    loaded = Genome.load(buf)

    assert loaded.mutator_type == mutator_type
    assert loaded.num_policy_params() == g.num_policy_params()
    assert loaded.num_mutator_params() == g.num_mutator_params()
    assert pytest.approx(loaded.fitness, rel=1e-6) == g.fitness


def test_load_legacy_class_name_mutator_type_still_works():
    # Simulate old checkpoint format where mutator_type stored class name.
    g = Genome(Policy(4, 2, 32), create_mutator('dualmixture', chunk_size=64), 'dualmixture')
    buf = io.BytesIO()
    g.save(buf)
    buf.seek(0)

    data = torch.load(buf, weights_only=False)
    data['mutator_type'] = type(g.mutator).__name__  # legacy style

    buf2 = io.BytesIO()
    torch.save(data, buf2)
    buf2.seek(0)

    loaded = Genome.load(buf2)
    assert loaded.mutator_type == 'dualmixture'


def test_create_mutator_dualmixture_kwargs_applied():
    m = create_mutator('dualmixture', chunk_size=64, p_gauss_policy=0.37, gauss_scale_policy=0.051)
    assert float(m.p_gauss_policy.item()) == pytest.approx(0.37)
    assert float(m.gauss_scale_policy.item()) == pytest.approx(0.051)


def test_create_mutator_dualmixture_v2_kwargs_applied():
    m = create_mutator(
        'dualmixture_v2',
        chunk_size=32,
        ref_dim=20,
        hidden=40,
        lowrank_rank=5,
        max_policy_groups=6,
        policy_corr_scale=0.021,
        policy_noise_scale=0.009,
        meta_corr_scale=0.007,
        meta_noise_scale=0.003,
    )
    assert m.chunk_size == 32
    assert m.ref_dim == 20
    assert m.lowrank_rank == 5
    assert m.max_policy_groups == 6
    assert float(m.policy_corr_scale.item()) == pytest.approx(0.021)
    assert float(m.policy_noise_scale.item()) == pytest.approx(0.009)
    assert float(m.meta_corr_scale.item()) == pytest.approx(0.007)
    assert float(m.meta_noise_scale.item()) == pytest.approx(0.003)


def test_reproduce_learned_compat_threshold_mutates_with_stronger_step(monkeypatch):
    policy = Policy(4, 2, 32)
    mutator = create_mutator('dualmixture', chunk_size=64)
    compat = CompatibilityNet(128)
    g = Genome(
        policy,
        mutator,
        mutator_type='dualmixture',
        compat_net=compat,
        compat_threshold=0.5,
        evolve_compat_threshold=True,
    )
    monkeypatch.setattr('src.genome.np.random.randn', lambda: 5.0)

    child = g.reproduce(generation=1, max_generations=10)
    expected = 0.5 * np.exp(0.10 * 5.0 - 0.5 * (0.10 ** 2))
    assert child.compat_threshold == pytest.approx(expected)
