"""
Genome: Policy + Mutator bundled together.

The mutator takes the full genome (policy + mutator weights flattened)
and outputs the next generation's genome.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class Policy(nn.Module):
    """Simple MLP policy for RL environments."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0)
            action = self.net(x).squeeze(0)
        return action.numpy()


class ChunkMutator(nn.Module):
    """
    Chunk-based MLP mutator.

    Reads weights in fixed-size chunks, outputs a perturbation per chunk.
    The perturbation is added to the original weights (residual mutation).

    Input: chunk of weights (size chunk_size)
    Output: delta for that chunk (same size)
    """

    def __init__(self, chunk_size: int = 64, hidden: int = 128):
        super().__init__()
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(chunk_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, chunk_size),
            nn.Tanh(),  # Bound perturbations to [-1, 1]
        )
        self.mutation_scale = nn.Parameter(torch.tensor(0.01))  # Learnable scale

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """Apply mutation to a chunk of weights."""
        delta = self.net(chunk) * self.mutation_scale
        return chunk + delta

    def mutate_genome(self, flat_weights: torch.Tensor) -> torch.Tensor:
        """
        Mutate an entire flattened weight vector chunk by chunk.
        Pads the last chunk if needed.
        """
        n = flat_weights.shape[0]
        cs = self.chunk_size
        # Pad to multiple of chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len)])

        chunks = padded.view(-1, cs)
        mutated_chunks = self.net(chunks) * self.mutation_scale + chunks
        mutated = mutated_chunks.view(-1)[:n]  # Remove padding
        return mutated


class TransformerMutator(nn.Module):
    """
    Transformer-based mutator.

    Treats weight chunks as tokens, applies self-attention to capture
    long-range dependencies between different parts of the network.
    """

    def __init__(self, chunk_size: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_model: int = 128):
        super().__init__()
        self.chunk_size = chunk_size
        self.embed = nn.Linear(chunk_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.project = nn.Linear(d_model, chunk_size)
        self.mutation_scale = nn.Parameter(torch.tensor(0.01))

    def mutate_genome(self, flat_weights: torch.Tensor) -> torch.Tensor:
        n = flat_weights.shape[0]
        cs = self.chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len)])

        chunks = padded.view(1, -1, cs)  # (1, n_chunks, chunk_size)
        embedded = self.embed(chunks)
        transformed = self.transformer(embedded)
        delta = self.project(transformed) * self.mutation_scale

        mutated = (chunks + delta).view(-1)[:n]
        return mutated


class CPPNMutator(nn.Module):
    """
    CPPN-style mutator (Compositional Pattern-Producing Network).

    Maps weight coordinates (layer_idx, row, col, bias_flag) → weight value.
    The CPPN is an indirect encoding: instead of storing weights directly,
    it generates them from a compact coordinate description.
    """

    def __init__(self, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(4, hidden), nn.Tanh()]  # Input: (layer, row, col, bias)
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))  # Output: single weight value
        self.net = nn.Sequential(*layers)
        self.mutation_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 4) → weight values (N, 1)"""
        return self.net(coords) * self.mutation_scale

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor) -> torch.Tensor:
        """
        Generate weight perturbations from coordinates.
        weight_coords: (n_weights, 4) normalized coordinates for each weight.
        """
        delta = self.net(weight_coords).squeeze(-1) * self.mutation_scale
        return flat_weights + delta


class Genome:
    """
    A complete genome: policy + mutator.

    The mutator can reproduce the entire genome (policy + itself).
    """

    def __init__(self, policy: Policy, mutator: nn.Module,
                 mutator_type: str = 'chunk'):
        self.policy = policy
        self.mutator = mutator
        self.mutator_type = mutator_type
        self.fitness = 0.0
        self._weight_coords = None  # Cached for CPPN

    def get_flat_weights(self) -> torch.Tensor:
        """Flatten all weights (policy + mutator) into a single vector."""
        policy_params = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        mutator_params = torch.cat([p.data.view(-1) for p in self.mutator.parameters()])
        return torch.cat([policy_params, mutator_params])

    def set_flat_weights(self, flat: torch.Tensor):
        """Restore weights from a flat vector."""
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].view(p.shape))
            idx += n
        for p in self.mutator.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].view(p.shape))
            idx += n

    def num_policy_params(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())

    def num_mutator_params(self) -> int:
        return sum(p.numel() for p in self.mutator.parameters())

    def num_total_params(self) -> int:
        return self.num_policy_params() + self.num_mutator_params()

    @torch.no_grad()
    def reproduce(self) -> 'Genome':
        """
        Self-replication: feed own weights through mutator to produce offspring.
        The mutator modifies BOTH the policy AND itself.
        """
        flat = self.get_flat_weights()

        if self.mutator_type == 'cppn':
            coords = self._get_weight_coords()
            new_flat = self.mutator.mutate_genome(flat, coords)
        else:
            new_flat = self.mutator.mutate_genome(flat)

        # Create offspring with same architecture
        child = Genome(
            policy=self._clone_policy(),
            mutator=self._clone_mutator(),
            mutator_type=self.mutator_type,
        )
        child.set_flat_weights(new_flat)
        return child

    @torch.no_grad()
    def crossover(self, other: 'Genome') -> 'Genome':
        """
        Crossover: feed THIS genome's policy into OTHER's mutator.
        Other's mutator processes [self.policy_weights, other.mutator_weights].
        """
        my_policy = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        other_mutator = torch.cat([p.data.view(-1) for p in other.mutator.parameters()])
        combined = torch.cat([my_policy, other_mutator])

        if other.mutator_type == 'cppn':
            coords = other._get_weight_coords()
            new_flat = other.mutator.mutate_genome(combined, coords)
        else:
            new_flat = other.mutator.mutate_genome(combined)

        child = Genome(
            policy=self._clone_policy(),
            mutator=other._clone_mutator(),
            mutator_type=other.mutator_type,
        )
        child.set_flat_weights(new_flat)
        return child

    def _clone_policy(self) -> Policy:
        """Create a new policy with same architecture."""
        first_layer = self.policy.net[0]
        last_layer = self.policy.net[-1]
        hidden = first_layer.out_features
        return Policy(first_layer.in_features, last_layer.out_features, hidden)

    def _clone_mutator(self) -> nn.Module:
        """Create a new mutator with same architecture."""
        import copy
        m = copy.deepcopy(self.mutator)
        return m

    def _get_weight_coords(self) -> torch.Tensor:
        """Generate normalized coordinates for each weight (for CPPN)."""
        if self._weight_coords is not None:
            return self._weight_coords

        coords = []
        layer_idx = 0
        for name, p in list(self.policy.named_parameters()) + \
                        list(self.mutator.named_parameters()):
            is_bias = 'bias' in name or p.dim() == 1
            shape = p.shape
            if is_bias:
                for i in range(shape[0]):
                    coords.append([layer_idx / 10.0, i / max(shape[0], 1), 0.0, 1.0])
            elif p.dim() >= 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        coords.append([
                            layer_idx / 10.0,
                            i / max(shape[0], 1),
                            j / max(shape[1], 1),
                            0.0,
                        ])
            else:
                # Scalar or other 1D non-bias param
                for i in range(p.numel()):
                    coords.append([layer_idx / 10.0, i / max(p.numel(), 1), 0.0, 0.5])
            if not is_bias:
                layer_idx += 1

        self._weight_coords = torch.FloatTensor(coords)
        return self._weight_coords
