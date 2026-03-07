"""Tests for lineage tracking and Graphviz artifact generation."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evolution import evolve_generation
from src.genome import Genome, Policy, create_mutator
from src.lineage import LineageTracker, generate_lineage_plots


def _make_population(n: int = 2) -> list[Genome]:
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


def test_lineage_records_births_and_crossover_parent_links(monkeypatch):
    pop = _make_population(2)
    tracker = LineageTracker(run_metadata={'env': 'CartPole-v1'})
    tracker.register_initial_population(pop, generation=0)

    picks = iter([0, 1, 0, 1, 0, 1])

    def _select(population, k=3):
        return population[next(picks)]

    monkeypatch.setattr('src.evolution.tournament_select', _select)
    monkeypatch.setattr('src.evolution.np.random.random', lambda: 0.0)

    new_pop, _ = evolve_generation(
        pop,
        crossover_rate=1.0,
        elitism=1,
        generation=0,
        lineage_tracker=tracker,
    )

    assert len(new_pop) == len(pop)

    events = tracker.summary()
    assert events['num_events'] == 4  # 2 initial + 1 elite + 1 produced child
    assert events['births_by_method']['initial'] == 2
    assert events['births_by_method']['elite'] == 1
    assert events['births_by_method']['crossover'] == 1

    # Find the crossover event directly from saved events in insertion order
    crossover_event = [e for e in tracker.events() if e['method'] == 'crossover'][0]
    assert crossover_event['generation'] == 1
    assert len(crossover_event['parent_ids']) == 2
    assert crossover_event['child_id'] not in crossover_event['parent_ids']


def test_plot_generation_writes_dot_without_graphviz(tmp_path, monkeypatch):
    lineage_path = tmp_path / 'lineage.jsonl'
    meta_path = tmp_path / 'lineage_meta.json'

    events = [
        {'event': 'birth', 'generation': 0, 'child_id': 'g0', 'parent_ids': [], 'method': 'initial'},
        {'event': 'birth', 'generation': 0, 'child_id': 'g1', 'parent_ids': [], 'method': 'initial'},
        {'event': 'birth', 'generation': 1, 'child_id': 'g2', 'parent_ids': ['g0', 'g1'], 'method': 'crossover'},
    ]
    lineage_path.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
    meta_path.write_text(json.dumps({'species_snapshots': {'1': {'g2': 3}}}), encoding='utf-8')

    monkeypatch.setattr('src.lineage.shutil.which', lambda _: None)

    result = generate_lineage_plots(
        lineage_path=str(lineage_path),
        meta_path=str(meta_path),
        out_prefix=str(tmp_path / 'lineage_tree'),
        max_generation=0,
        color_by='species',
        formats=('png', 'svg'),
    )

    assert os.path.exists(result['dot_path'])
    dot_text = (tmp_path / 'lineage_tree.dot').read_text(encoding='utf-8')
    assert 'g0' in dot_text
    assert 'g1' in dot_text
    assert 'g2' not in dot_text  # excluded by max_generation
    assert result['rendered_paths'] == []
    assert result['warning'] is not None
