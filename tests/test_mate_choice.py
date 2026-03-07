"""Tests for experimental mate-choice parent selection."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evolution import evolve_generation
from src.genome import Genome, Policy, create_mutator


def _make_population(n: int = 4) -> list[Genome]:
    pop = []
    for i in range(n):
        g = Genome(
            Policy(4, 2, 16),
            create_mutator('dualmixture', chunk_size=32),
            mutator_type='dualmixture',
        )
        g.fitness = float(n - i)
        pop.append(g)
    return pop


def test_mate_choice_off_keeps_legacy_parent_selection(monkeypatch):
    pop = _make_population(3)
    calls = {'n': 0}

    def _select(population, k=3):
        calls['n'] += 1
        return population[0]

    monkeypatch.setattr('src.evolution.tournament_select', _select)
    monkeypatch.setattr('src.evolution.np.random.random', lambda: 0.0)

    _, info = evolve_generation(
        pop,
        crossover_rate=1.0,
        elitism=0,
        mate_choice=False,
    )

    assert calls['n'] > 0
    assert info['mate_choice_enabled'] is False
    assert info['mate_choice_attempts'] == 0
    assert info['mate_choice_accepts'] == 0
    assert info['mate_choice_fallbacks'] == 0
    assert info['mate_choice_accept_rate'] == 0.0
    assert info['mate_choice_fallback_rate'] == 0.0


def test_mate_choice_smoke_one_generation(monkeypatch):
    pop = _make_population(4)
    monkeypatch.setattr('src.evolution.np.random.random', lambda: 0.0)

    new_pop, info = evolve_generation(
        pop,
        crossover_rate=1.0,
        elitism=0,
        mate_choice=True,
        mate_choice_topk=3,
        mate_choice_threshold=0.0,
        mate_choice_temperature=1.0,
    )

    assert len(new_pop) == len(pop)
    assert info['mate_choice_enabled'] is True
    assert info['mate_choice_attempts'] > 0
    assert info['mate_choice_accepts'] + info['mate_choice_fallbacks'] == info['mate_choice_attempts']
    assert 0.0 <= info['mate_choice_accept_rate'] <= 1.0
    assert 0.0 <= info['mate_choice_fallback_rate'] <= 1.0
