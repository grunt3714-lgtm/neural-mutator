from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .cppn import CPPN


Coord = Tuple[float, float, float]


@dataclass
class Substrate:
    input_coords: List[Coord]
    hidden1_coords: List[Coord]
    hidden2_coords: List[Coord]
    output_coords: List[Coord]


def _line_coords(n: int, y: float, z: float) -> List[Coord]:
    if n <= 1:
        xs = [0.0]
    else:
        xs = np.linspace(-1.0, 1.0, n)
    return [(float(x), float(y), float(z)) for x in xs]


def lunar_lander_substrate(
    input_size: int = 8,
    hidden1_size: int = 16,
    hidden2_size: int = 16,
    output_size: int = 4,
) -> Substrate:
    return Substrate(
        input_coords=_line_coords(input_size, y=-1.0, z=0.0),
        hidden1_coords=_line_coords(hidden1_size, y=-0.33, z=0.0),
        hidden2_coords=_line_coords(hidden2_size, y=0.33, z=0.0),
        output_coords=_line_coords(output_size, y=1.0, z=0.0),
    )


def generate_layer_weights(
    cppn: CPPN,
    src_coords: List[Coord],
    dst_coords: List[Coord],
    expression_threshold: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    w = np.zeros((len(dst_coords), len(src_coords)), dtype=np.float32)
    b = np.zeros(len(dst_coords), dtype=np.float32)

    for j, dst in enumerate(dst_coords):
        for i, src in enumerate(src_coords):
            weight, gate = cppn.query_weight_and_expression(src, dst)
            if cppn.use_expression_output and expression_threshold is not None:
                if gate is None or gate < expression_threshold:
                    weight = 0.0
            w[j, i] = float(weight)

        # Bias is generated from a virtual bias node at the same x,y with z=-1.
        bx, by, _ = dst
        bias_src = (bx, by, -1.0)
        b_w, b_gate = cppn.query_weight_and_expression(bias_src, dst)
        if cppn.use_expression_output and expression_threshold is not None:
            if b_gate is None or b_gate < expression_threshold:
                b_w = 0.0
        b[j] = float(b_w)

    return w, b


class HyperNEATPolicy(nn.Module):
    """Torch policy with the same .act(obs)->np.ndarray behavior as Policy in genome.py."""

    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, w3: np.ndarray, b3: np.ndarray):
        super().__init__()
        in_dim = w1.shape[1]
        h1_dim = w1.shape[0]
        h2_dim = w2.shape[0]
        out_dim = w3.shape[0]

        self.fc1 = nn.Linear(in_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, out_dim)
        self.act_fn = nn.Tanh()

        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(w1))
            self.fc1.bias.copy_(torch.from_numpy(b1))
            self.fc2.weight.copy_(torch.from_numpy(w2))
            self.fc2.bias.copy_(torch.from_numpy(b2))
            self.fc3.weight.copy_(torch.from_numpy(w3))
            self.fc3.bias.copy_(torch.from_numpy(b3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        return self.fc3(x)

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x).squeeze(0)
        return logits.cpu().numpy()


def build_hyperneat_policy(
    cppn: CPPN,
    substrate: Substrate | None = None,
    expression_threshold: float | None = None,
) -> HyperNEATPolicy:
    sub = substrate or lunar_lander_substrate()
    w1, b1 = generate_layer_weights(cppn, sub.input_coords, sub.hidden1_coords, expression_threshold)
    w2, b2 = generate_layer_weights(cppn, sub.hidden1_coords, sub.hidden2_coords, expression_threshold)
    w3, b3 = generate_layer_weights(cppn, sub.hidden2_coords, sub.output_coords, expression_threshold)
    return HyperNEATPolicy(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
