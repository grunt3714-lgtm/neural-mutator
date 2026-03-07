import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


class LineageTracker:
    """Track genome lineage across generations with stable genome IDs."""

    def __init__(self, run_metadata: dict[str, Any] | None = None):
        self._next_id = 0
        self._events: list[dict[str, Any]] = []
        self._species_snapshots: dict[str, dict[str, int | None]] = {}
        self._fitness_snapshots: dict[str, dict[str, float | None]] = {}
        self._parents: dict[str, list[str]] = {}
        self.run_metadata = dict(run_metadata or {})

    def _new_genome_id(self) -> str:
        gid = f"g{self._next_id}"
        self._next_id += 1
        return gid

    def ensure_genome_id(self, genome) -> str:
        gid = getattr(genome, 'genome_id', None)
        if gid is None:
            gid = self._new_genome_id()
            setattr(genome, 'genome_id', gid)
        gid = str(gid)
        if gid.startswith('g'):
            suffix = gid[1:]
            if suffix.isdigit():
                self._next_id = max(self._next_id, int(suffix) + 1)
        return gid

    def register_initial_population(self, population, generation: int = 0) -> None:
        for genome in population:
            child_id = self.ensure_genome_id(genome)
            self._parents[child_id] = []
            self._events.append({
                'event': 'birth',
                'generation': int(generation),
                'child_id': child_id,
                'parent_ids': [],
                'method': 'initial',
            })

    def register_child(self, child, parents: list | tuple, generation: int, method: str) -> str:
        child_id = self.ensure_genome_id(child)
        parent_ids: list[str] = []
        for parent in parents:
            if parent is None:
                continue
            parent_ids.append(self.ensure_genome_id(parent))
        self._parents[child_id] = list(parent_ids)
        self._events.append({
            'event': 'birth',
            'generation': int(generation),
            'child_id': child_id,
            'parent_ids': parent_ids,
            'method': str(method),
        })
        return child_id

    def record_population_snapshot(self, population, generation: int) -> None:
        species: dict[str, int | None] = {}
        fitness: dict[str, float | None] = {}
        for genome in population:
            gid = self.ensure_genome_id(genome)
            sid = getattr(genome, 'species_id', None)
            species[gid] = int(sid) if sid is not None else None
            fit = getattr(genome, 'fitness', None)
            fitness[gid] = float(fit) if fit is not None else None
        self._species_snapshots[str(int(generation))] = species
        self._fitness_snapshots[str(int(generation))] = fitness

    def _ancestor_distances(self, genome_id: str, max_depth: int = 32) -> dict[str, int]:
        """Return {ancestor_id: edge-distance} within max_depth hops (includes self:0)."""
        start = str(genome_id)
        distances: dict[str, int] = {start: 0}
        queue: list[tuple[str, int]] = [(start, 0)]
        i = 0
        while i < len(queue):
            gid, depth = queue[i]
            i += 1
            if depth >= int(max_depth):
                continue
            for pid in self._parents.get(gid, []):
                if pid in distances:
                    continue
                nd = depth + 1
                distances[pid] = nd
                queue.append((pid, nd))
        return distances

    def lineage_distance(self, genome_a, genome_b, max_depth: int = 32) -> int | None:
        """Generational distance via nearest common ancestor; None if none found within max_depth."""
        ga = self.ensure_genome_id(genome_a)
        gb = self.ensure_genome_id(genome_b)
        da = self._ancestor_distances(ga, max_depth=max_depth)
        db = self._ancestor_distances(gb, max_depth=max_depth)
        common = set(da.keys()) & set(db.keys())
        if not common:
            return None
        return int(min(da[c] + db[c] for c in common))

    def summary(self) -> dict[str, Any]:
        method_counts: dict[str, int] = defaultdict(int)
        for ev in self._events:
            method_counts[str(ev.get('method', 'unknown'))] += 1
        return {
            'created_at': _utc_now_iso(),
            'run_metadata': self.run_metadata,
            'num_events': len(self._events),
            'num_genomes': len({ev['child_id'] for ev in self._events if 'child_id' in ev}),
            'max_genome_id': self._next_id - 1,
            'births_by_method': dict(sorted(method_counts.items())),
            'species_snapshots': self._species_snapshots,
            'fitness_snapshots': self._fitness_snapshots,
        }

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def save(self, output_dir: str | os.PathLike) -> tuple[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        lineage_path = out / 'lineage.jsonl'
        meta_path = out / 'lineage_meta.json'
        with lineage_path.open('w', encoding='utf-8') as f:
            for ev in self._events:
                f.write(json.dumps(ev, sort_keys=True) + '\n')
        with meta_path.open('w', encoding='utf-8') as f:
            json.dump(self.summary(), f, indent=2, sort_keys=True)
        return str(lineage_path), str(meta_path)


def read_lineage_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def read_lineage_meta(path: str | os.PathLike | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def _species_for_birth(meta: dict[str, Any], generation: int, genome_id: str) -> int | None:
    snapshots = meta.get('species_snapshots', {})
    gen_map = snapshots.get(str(int(generation)), {})
    value = gen_map.get(genome_id)
    return int(value) if value is not None else None


def _build_palette() -> list[str]:
    return [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]


def lineage_to_dot(events: list[dict[str, Any]],
                   meta: dict[str, Any] | None = None,
                   max_generation: int | None = None,
                   color_by: str = 'none') -> str:
    meta = meta or {}
    selected = []
    for ev in events:
        gen = int(ev.get('generation', 0))
        if max_generation is not None and gen > int(max_generation):
            continue
        selected.append(ev)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, str]] = []
    for ev in selected:
        child = str(ev['child_id'])
        method = str(ev.get('method', 'unknown'))
        gen = int(ev.get('generation', 0))
        nodes.setdefault(child, {'generation': gen, 'method': method})
        for pid in ev.get('parent_ids', []):
            pid = str(pid)
            nodes.setdefault(pid, {'generation': max(0, gen - 1), 'method': 'unknown'})
            edges.append((pid, child, method))

    palette = _build_palette()
    lines = [
        'digraph lineage {',
        '  rankdir=LR;',
        '  graph [bgcolor="white"];',
        '  node [shape=circle, style=filled, fillcolor="#f7f7f7", color="#333333", fontname="Helvetica", fontsize=10];',
        '  edge [color="#666666", arrowsize=0.6, penwidth=1.0];',
    ]

    for gid, info in sorted(nodes.items(), key=lambda x: x[0]):
        gen = int(info.get('generation', 0))
        label = f"{gid}\\nG{gen}"
        attrs = [f'label="{label}"']
        if color_by == 'method':
            method = str(info.get('method', 'unknown'))
            method_colors = {
                'initial': '#dddddd',
                'elite': '#9ecae1',
                'mutation': '#a1d99b',
                'mutation_fallback': '#fdae6b',
                'mutation_error': '#fdd0a2',
                'mutation_incompatible': '#fcbba1',
                'crossover': '#c994c7',
                'crossover_fallback': '#fdd49e',
            }
            attrs.append(f'fillcolor="{method_colors.get(method, "#f7f7f7")}"')
        elif color_by == 'species':
            sid = _species_for_birth(meta, gen, gid)
            if sid is not None:
                attrs.append(f'fillcolor="{palette[sid % len(palette)]}"')
        lines.append(f'  "{gid}" [{", ".join(attrs)}];')

    for parent, child, method in edges:
        edge_attr = []
        if color_by == 'method':
            method_edge_colors = {
                'crossover': '#756bb1',
                'mutation': '#31a354',
                'elite': '#3182bd',
            }
            edge_attr.append(f'color="{method_edge_colors.get(method, "#666666")}"')
        attr_text = f" [{', '.join(edge_attr)}]" if edge_attr else ''
        lines.append(f'  "{parent}" -> "{child}"{attr_text};')

    lines.append('}')
    return '\n'.join(lines) + '\n'


def render_dot(dot_path: str | os.PathLike,
               formats: list[str] | tuple[str, ...] = ('png', 'svg')) -> tuple[list[str], str | None]:
    dot_bin = shutil.which('dot')
    if dot_bin is None:
        return [], 'Graphviz binary `dot` not found; wrote DOT only.'

    generated: list[str] = []
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        out_path = str(Path(dot_path).with_suffix(f'.{fmt}'))
        try:
            subprocess.run([dot_bin, f'-T{fmt}', str(dot_path), '-o', out_path], check=True, capture_output=True)
            generated.append(out_path)
        except Exception as exc:
            print(f"[lineage] Warning: failed to render {fmt}: {exc}")
    return generated, None


def generate_lineage_plots(lineage_path: str | os.PathLike,
                           meta_path: str | os.PathLike | None,
                           out_prefix: str | os.PathLike,
                           max_generation: int | None = None,
                           color_by: str = 'none',
                           formats: list[str] | tuple[str, ...] = ('png', 'svg')) -> dict[str, Any]:
    events = read_lineage_jsonl(lineage_path)
    meta = read_lineage_meta(meta_path)
    dot_text = lineage_to_dot(events, meta=meta, max_generation=max_generation, color_by=color_by)
    dot_path = Path(str(out_prefix)).with_suffix('.dot')
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(dot_text, encoding='utf-8')
    rendered, warning = render_dot(dot_path, formats=formats)
    return {
        'dot_path': str(dot_path),
        'rendered_paths': rendered,
        'warning': warning,
    }
