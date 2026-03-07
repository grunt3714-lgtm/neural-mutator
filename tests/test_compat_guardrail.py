"""Tests for optional compatibility-rate guardrail behavior."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evolution import evolve_generation, _update_compat_rate_bias
from src.genome import Genome, Policy, create_mutator


def _make_genome(threshold: float = 0.5) -> Genome:
    return Genome(
        Policy(4, 2, 16),
        create_mutator('dualmixture', chunk_size=32),
        mutator_type='dualmixture',
        compat_net=None,
        compat_threshold=threshold,
        evolve_compat_threshold=True,
    )


def test_update_compat_rate_bias_moves_toward_target_band_and_is_bounded():
    assert _update_compat_rate_bias(0.0, compat_rate=0.20, target_low=0.35, target_high=0.75, adjust=0.03) == pytest.approx(0.03)
    assert _update_compat_rate_bias(0.0, compat_rate=0.90, target_low=0.35, target_high=0.75, adjust=0.03) == pytest.approx(-0.03)
    assert _update_compat_rate_bias(0.20, compat_rate=0.50, target_low=0.35, target_high=0.75, adjust=0.03) == pytest.approx(0.18)
    assert _update_compat_rate_bias(0.34, compat_rate=0.10, target_low=0.35, target_high=0.75, adjust=0.03) == pytest.approx(0.35)
    assert _update_compat_rate_bias(-0.34, compat_rate=0.95, target_low=0.35, target_high=0.75, adjust=0.03) == pytest.approx(-0.35)


def test_guardrail_applies_bias_to_fixed_gate(monkeypatch):
    g1 = _make_genome(threshold=0.5)
    g2 = _make_genome(threshold=0.5)
    g1.fitness = 2.0
    g2.fitness = 1.0
    population = [g1, g2]
    calls = []

    def _select(population, k=3):
        return population[0]

    def _compat(self, other, threshold=0.5):
        calls.append(threshold)
        return float(threshold) >= 0.60

    monkeypatch.setattr('src.evolution.tournament_select', _select)
    monkeypatch.setattr('src.evolution.np.random.random', lambda: 0.0)
    monkeypatch.setattr(Genome, 'is_compatible', _compat)

    _, info = evolve_generation(
        population,
        crossover_rate=1.0,
        elitism=0,
        compat_threshold=0.5,
        learn_compat_threshold=False,
        compat_rate_guardrail=True,
        compat_rate_bias=0.10,
    )

    assert info['crossover_attempts'] > 0
    assert info['crossover_compatible'] == info['crossover_attempts']
    assert all(float(t) == pytest.approx(0.60) for t in calls)
    assert info['compat_gate_mean'] == pytest.approx(0.60)
    assert info['compat_bias'] == pytest.approx(0.10)


def test_guardrail_bias_stacks_on_learned_thresholds(monkeypatch):
    g1 = _make_genome(threshold=0.40)
    g2 = _make_genome(threshold=0.80)
    g1.fitness = 2.0
    g2.fitness = 1.0
    population = [g1, g2]
    calls = []
    picks = iter([0, 1, 0, 1, 0, 1, 0, 1])

    def _select(population, k=3):
        return population[next(picks)]

    def _compat(self, other, threshold=0.5):
        calls.append(float(threshold))
        return True

    monkeypatch.setattr('src.evolution.tournament_select', _select)
    monkeypatch.setattr('src.evolution.np.random.random', lambda: 0.0)
    monkeypatch.setattr(Genome, 'is_compatible', _compat)

    _, info = evolve_generation(
        population,
        crossover_rate=1.0,
        elitism=0,
        compat_threshold=None,
        learn_compat_threshold=True,
        compat_rate_guardrail=True,
        compat_rate_bias=0.10,
    )

    assert info['crossover_attempts'] > 0
    assert any(v == pytest.approx(0.50) for v in calls)
    assert any(v == pytest.approx(0.90) for v in calls)
    assert info['compat_gate_mean'] == pytest.approx(0.70)


def test_guardrail_off_keeps_learned_mode_threshold_call_signature(monkeypatch):
    g1 = _make_genome(threshold=0.40)
    g2 = _make_genome(threshold=0.80)
    g1.fitness = 2.0
    g2.fitness = 1.0
    population = [g1, g2]
    calls = []

    def _select(population, k=3):
        return population[0]

    def _compat(self, other, threshold=0.5):
        calls.append(threshold)
        return True

    monkeypatch.setattr('src.evolution.tournament_select', _select)
    monkeypatch.setattr('src.evolution.np.random.random', lambda: 0.0)
    monkeypatch.setattr(Genome, 'is_compatible', _compat)

    evolve_generation(
        population,
        crossover_rate=1.0,
        elitism=0,
        compat_threshold=None,
        learn_compat_threshold=True,
        compat_rate_guardrail=False,
        compat_rate_bias=0.10,
    )

    assert all(t is None for t in calls)
