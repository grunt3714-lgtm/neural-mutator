import copy
import math
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


ACTIVATIONS = ("tanh", "sin", "gauss", "relu", "identity", "abs", "step")


def _apply_activation(name: str, x: float) -> float:
    if name == "tanh":
        return float(np.tanh(x))
    if name == "sin":
        return float(np.sin(x))
    if name == "gauss":
        return float(np.exp(-(x * x)))
    if name == "relu":
        return float(x if x > 0.0 else 0.0)
    if name == "identity":
        return float(x)
    if name == "abs":
        return float(abs(x))
    if name == "step":
        return 1.0 if x > 0.0 else 0.0
    return float(np.tanh(x))


@dataclass
class NodeGene:
    node_id: int
    node_type: str  # input|hidden|output
    activation: str
    order: float
    bias: float = 0.0
    response: float = 1.0


@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int


class CPPN:
    """NEAT-style CPPN used by HyperNEAT.

    Tuned to match reference implementations (pureples / neat-python):
    - High topology mutation rates (conn_add=0.5, node_add=0.2)
    - Per-node bias + response parameters
    - Weight range ±30, bias range ±30
    - Connection deletion
    """

    _next_node_id: int = 0
    _next_innovation: int = 0
    _innovation_map: Dict[Tuple[int, int], int] = {}

    def __init__(
        self,
        n_inputs: int = 8,
        n_outputs: int = 1,
        use_expression_output: bool = False,
        weight_scale: float = 1.0,
        initialize: bool = True,
    ):
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.use_expression_output = bool(use_expression_output)
        self.weight_scale = float(weight_scale)

        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.input_ids: List[int] = []
        self.output_ids: List[int] = []

        if initialize:
            self._init_minimal()

    @classmethod
    def _new_node_id(cls) -> int:
        nid = cls._next_node_id
        cls._next_node_id += 1
        return nid

    @classmethod
    def _get_innovation(cls, in_node: int, out_node: int) -> int:
        key = (int(in_node), int(out_node))
        if key in cls._innovation_map:
            return cls._innovation_map[key]
        innov = cls._next_innovation
        cls._next_innovation += 1
        cls._innovation_map[key] = innov
        return innov

    def _init_minimal(self) -> None:
        for _ in range(self.n_inputs):
            node_id = self._new_node_id()
            self.nodes[node_id] = NodeGene(
                node_id=node_id, node_type="input", activation="identity",
                order=0.0, bias=0.0, response=1.0,
            )
            self.input_ids.append(node_id)

        out_count = self.n_outputs + (1 if self.use_expression_output else 0)
        for _ in range(out_count):
            node_id = self._new_node_id()
            self.nodes[node_id] = NodeGene(
                node_id=node_id, node_type="output", activation="tanh",
                order=1.0,
                bias=float(np.random.normal(0.0, 1.0)),
                response=1.0,
            )
            self.output_ids.append(node_id)

        for src in self.input_ids:
            for dst in self.output_ids:
                innov = self._get_innovation(src, dst)
                self.connections[innov] = ConnectionGene(
                    in_node=src, out_node=dst,
                    weight=float(np.random.normal(0.0, 1.0)),
                    enabled=True, innovation=innov,
                )

    def copy(self) -> "CPPN":
        return copy.deepcopy(self)

    def _enabled_connections(self) -> List[ConnectionGene]:
        return [c for c in self.connections.values() if c.enabled]

    def _topological_ids(self) -> List[int]:
        return sorted(self.nodes.keys(), key=lambda nid: (self.nodes[nid].order, nid))

    def query(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float32)
        if x.shape[0] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} CPPN inputs, got {x.shape[0]}")

        values: Dict[int, float] = {}
        for i, nid in enumerate(self.input_ids):
            values[nid] = float(x[i])

        incoming: Dict[int, List[ConnectionGene]] = {}
        for conn in self._enabled_connections():
            incoming.setdefault(conn.out_node, []).append(conn)

        for nid in self._topological_ids():
            node = self.nodes[nid]
            if node.node_type == "input":
                continue
            total = node.bias
            for conn in incoming.get(nid, []):
                total += values.get(conn.in_node, 0.0) * conn.weight
            values[nid] = _apply_activation(node.activation, total * node.response)

        outs = [values.get(oid, 0.0) for oid in self.output_ids]
        return np.asarray(outs, dtype=np.float32)

    def query_weight_and_expression(
        self,
        src_coord: Tuple[float, float, float],
        dst_coord: Tuple[float, float, float],
    ) -> Tuple[float, Optional[float]]:
        x1, y1, z1 = src_coord
        x2, y2, z2 = dst_coord
        dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        inp = np.asarray([x1, y1, z1, x2, y2, z2, 1.0, dist], dtype=np.float32)
        out = self.query(inp)

        w = float(np.tanh(out[0]) * self.weight_scale)
        gate = None
        if self.use_expression_output:
            gate = float(out[1])
        return w, gate

    # ── Mutations ─────────────────────────────────────────────────

    def mutate_weights(
        self,
        mutate_rate: float = 0.8,
        perturb_scale: float = 0.5,
        replace_rate: float = 0.1,
    ) -> None:
        for innov, conn in list(self.connections.items()):
            if np.random.random() < mutate_rate:
                if np.random.random() < replace_rate:
                    conn.weight = float(np.random.normal(0.0, 1.0))
                else:
                    conn.weight += float(np.random.normal(0.0, perturb_scale))
                conn.weight = float(np.clip(conn.weight, -30.0, 30.0))
                self.connections[innov] = conn

    def mutate_biases(
        self,
        mutate_rate: float = 0.7,
        perturb_scale: float = 0.5,
        replace_rate: float = 0.1,
    ) -> None:
        for nid, node in list(self.nodes.items()):
            if node.node_type == "input":
                continue
            if np.random.random() < mutate_rate:
                if np.random.random() < replace_rate:
                    node.bias = float(np.random.normal(0.0, 1.0))
                else:
                    node.bias += float(np.random.normal(0.0, perturb_scale))
                node.bias = float(np.clip(node.bias, -30.0, 30.0))
                self.nodes[nid] = node

    def mutate_activations(self, prob: float = 0.5) -> None:
        mutable = [nid for nid, n in self.nodes.items() if n.node_type != "input"]
        for nid in mutable:
            if np.random.random() < prob:
                node = self.nodes[nid]
                choices = [a for a in ACTIVATIONS if a != node.activation]
                node.activation = random.choice(choices)
                self.nodes[nid] = node

    def add_connection(self, max_attempts: int = 40) -> bool:
        node_ids = list(self.nodes.keys())
        existing = {(c.in_node, c.out_node) for c in self.connections.values()}
        for _ in range(max_attempts):
            a = random.choice(node_ids)
            b = random.choice(node_ids)
            if a == b:
                continue
            na, nb = self.nodes[a], self.nodes[b]
            if na.order >= nb.order:
                continue
            if na.node_type == "output":
                continue
            if nb.node_type == "input":
                continue
            if (a, b) in existing:
                continue
            innov = self._get_innovation(a, b)
            self.connections[innov] = ConnectionGene(
                in_node=a, out_node=b,
                weight=float(np.random.normal(0.0, 1.0)),
                enabled=True, innovation=innov,
            )
            return True
        return False

    def delete_connection(self) -> bool:
        enabled = [i for i, c in self.connections.items() if c.enabled]
        if not enabled:
            return False
        innov = random.choice(enabled)
        del self.connections[innov]
        return True

    def add_node(self) -> bool:
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return False
        conn = random.choice(enabled)
        self.connections[conn.innovation].enabled = False

        new_id = self._new_node_id()
        src_order = self.nodes[conn.in_node].order
        dst_order = self.nodes[conn.out_node].order
        new_order = (src_order + dst_order) / 2.0
        if abs(dst_order - src_order) < 1e-6:
            new_order = src_order + 1e-3

        self.nodes[new_id] = NodeGene(
            node_id=new_id, node_type="hidden",
            activation=random.choice(ACTIVATIONS),
            order=new_order,
            bias=0.0, response=1.0,
        )

        innov_a = self._get_innovation(conn.in_node, new_id)
        innov_b = self._get_innovation(new_id, conn.out_node)
        self.connections[innov_a] = ConnectionGene(
            in_node=conn.in_node, out_node=new_id,
            weight=1.0, enabled=True, innovation=innov_a,
        )
        self.connections[innov_b] = ConnectionGene(
            in_node=new_id, out_node=conn.out_node,
            weight=conn.weight, enabled=True, innovation=innov_b,
        )
        return True

    def delete_node(self) -> bool:
        hidden = [nid for nid, n in self.nodes.items() if n.node_type == "hidden"]
        if not hidden:
            return False
        victim = random.choice(hidden)
        # Remove all connections involving this node
        to_remove = [i for i, c in self.connections.items()
                     if c.in_node == victim or c.out_node == victim]
        for i in to_remove:
            del self.connections[i]
        del self.nodes[victim]
        return True

    def mutate(
        self,
        p_add_node: float = 0.2,
        p_del_node: float = 0.2,
        p_add_conn: float = 0.5,
        p_del_conn: float = 0.5,
        p_mutate_act: float = 0.5,
    ) -> None:
        # Structural mutations
        if np.random.random() < p_add_conn:
            self.add_connection()
        if np.random.random() < p_del_conn:
            self.delete_connection()
        if np.random.random() < p_add_node:
            self.add_node()
        if np.random.random() < p_del_node:
            self.delete_node()
        # Parameter mutations
        self.mutate_weights()
        self.mutate_biases()
        self.mutate_activations(prob=p_mutate_act)

    # ── Crossover ─────────────────────────────────────────────────

    def crossover(self, other: "CPPN", self_fitness: float = 0.0, other_fitness: float = 0.0) -> "CPPN":
        if self.n_inputs != other.n_inputs or self.use_expression_output != other.use_expression_output:
            raise ValueError("Incompatible CPPNs for crossover")

        if (other_fitness > self_fitness) or (
            abs(other_fitness - self_fitness) < 1e-12 and len(other.connections) < len(self.connections)
        ):
            fitter, weaker = other, self
        else:
            fitter, weaker = self, other

        child = CPPN(
            n_inputs=fitter.n_inputs, n_outputs=fitter.n_outputs,
            use_expression_output=fitter.use_expression_output,
            weight_scale=fitter.weight_scale, initialize=False,
        )

        all_innov = set(fitter.connections.keys()) | set(weaker.connections.keys())
        for innov in sorted(all_innov):
            g1 = fitter.connections.get(innov)
            g2 = weaker.connections.get(innov)
            if g1 is not None and g2 is not None:
                chosen = copy.deepcopy(g1 if np.random.random() < 0.5 else g2)
                if (not g1.enabled or not g2.enabled) and np.random.random() < 0.75:
                    chosen.enabled = False
            elif g1 is not None:
                chosen = copy.deepcopy(g1)
            else:
                continue
            child.connections[innov] = chosen

        node_ids = set(fitter.input_ids + fitter.output_ids)
        for conn in child.connections.values():
            node_ids.add(conn.in_node)
            node_ids.add(conn.out_node)

        for nid in sorted(node_ids):
            if nid in fitter.nodes and nid in weaker.nodes:
                # Average bias/response from matching nodes
                f_node = copy.deepcopy(fitter.nodes[nid])
                w_node = weaker.nodes[nid]
                if np.random.random() < 0.5:
                    f_node.bias = w_node.bias
                if np.random.random() < 0.5:
                    f_node.activation = w_node.activation
                child.nodes[nid] = f_node
            elif nid in fitter.nodes:
                child.nodes[nid] = copy.deepcopy(fitter.nodes[nid])
            elif nid in weaker.nodes:
                child.nodes[nid] = copy.deepcopy(weaker.nodes[nid])

        child.input_ids = list(fitter.input_ids)
        child.output_ids = list(fitter.output_ids)
        return child

    # ── Distance ──────────────────────────────────────────────────

    def distance(
        self,
        other: "CPPN",
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.5,
    ) -> float:
        set1 = set(self.connections.keys())
        set2 = set(other.connections.keys())
        if not set1 and not set2:
            return 0.0

        max1 = max(set1) if set1 else -1
        max2 = max(set2) if set2 else -1

        matching = set1 & set2
        disjoint = 0
        excess = 0
        for i in (set1 - set2):
            excess += 1 if i > max2 else 0
            disjoint += 1 if i <= max2 else 0
        for i in (set2 - set1):
            excess += 1 if i > max1 else 0
            disjoint += 1 if i <= max1 else 0

        if matching:
            wdiff = float(np.mean([
                abs(self.connections[i].weight - other.connections[i].weight)
                for i in matching
            ]))
        else:
            wdiff = 0.0

        n = max(len(set1), len(set2), 1)
        return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * wdiff)

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "use_expression_output": self.use_expression_output,
            "weight_scale": self.weight_scale,
            "nodes": [
                {
                    "node_id": n.node_id, "node_type": n.node_type,
                    "activation": n.activation, "order": n.order,
                    "bias": n.bias, "response": n.response,
                }
                for n in self.nodes.values()
            ],
            "connections": [
                {
                    "in_node": c.in_node, "out_node": c.out_node,
                    "weight": c.weight, "enabled": c.enabled,
                    "innovation": c.innovation,
                }
                for c in self.connections.values()
            ],
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CPPN":
        obj = cls(
            n_inputs=int(data["n_inputs"]),
            n_outputs=int(data["n_outputs"]),
            use_expression_output=bool(data["use_expression_output"]),
            weight_scale=float(data.get("weight_scale", 1.0)),
            initialize=False,
        )

        obj.nodes = {}
        for nd in data["nodes"]:
            n = NodeGene(
                node_id=int(nd["node_id"]),
                node_type=str(nd["node_type"]),
                activation=str(nd["activation"]),
                order=float(nd["order"]),
                bias=float(nd.get("bias", 0.0)),
                response=float(nd.get("response", 1.0)),
            )
            obj.nodes[n.node_id] = n

        obj.connections = {}
        for cd in data["connections"]:
            c = ConnectionGene(
                in_node=int(cd["in_node"]),
                out_node=int(cd["out_node"]),
                weight=float(cd["weight"]),
                enabled=bool(cd["enabled"]),
                innovation=int(cd["innovation"]),
            )
            obj.connections[c.innovation] = c
            key = (c.in_node, c.out_node)
            if key not in cls._innovation_map:
                cls._innovation_map[key] = c.innovation

        obj.input_ids = [int(v) for v in data["input_ids"]]
        obj.output_ids = [int(v) for v in data["output_ids"]]

        if obj.nodes:
            cls._next_node_id = max(cls._next_node_id, max(obj.nodes.keys()) + 1)
        if obj.connections:
            cls._next_innovation = max(cls._next_innovation, max(obj.connections.keys()) + 1)

        return obj

    def to_bytes(self) -> bytes:
        return pickle.dumps(self.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, blob: bytes) -> "CPPN":
        data = pickle.loads(blob)
        return cls.from_dict(data)
