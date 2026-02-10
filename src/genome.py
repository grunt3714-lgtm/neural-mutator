"""
Genome: Policy + Mutator bundled together.

The mutator takes the full genome (policy + mutator weights flattened)
and outputs the next generation's genome.

Stability improvements:
- Mutation scale decay over generations
- Separate mutator protection (smaller perturbations for mutator weights)
- Weight clipping on mutator output
- Gaussian noise baseline mutator
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


class GaussianMutator(nn.Module):
    """
    Baseline: simple Gaussian noise mutation (ES-style).
    No learned parameters — just adds scaled noise.
    """

    def __init__(self, mutation_scale: float = 0.02):
        super().__init__()
        self.base_scale = mutation_scale
        # Register a dummy parameter so param counting works
        self.dummy = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None) -> torch.Tensor:
        noise = torch.randn_like(flat_weights) * self.base_scale
        return flat_weights + noise


class ChunkMutator(nn.Module):
    """
    Chunk-based MLP mutator.
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
            nn.Tanh(),
        )
        self.mutation_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        delta = self.net(chunk) * self.mutation_scale
        return chunk + delta

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None) -> torch.Tensor:
        n = flat_weights.shape[0]
        cs = self.chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len)])

        chunks = padded.view(-1, cs)
        delta = self.net(chunks) * self.mutation_scale
        # Clip delta to prevent catastrophic changes
        delta = torch.clamp(delta, -0.1, 0.1)
        mutated = chunks + delta
        return mutated.view(-1)[:n]


class TransformerMutator(nn.Module):
    """Transformer-based mutator."""

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

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None) -> torch.Tensor:
        n = flat_weights.shape[0]
        cs = self.chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len)])

        chunks = padded.view(1, -1, cs)
        embedded = self.embed(chunks)
        transformed = self.transformer(embedded)
        delta = self.project(transformed) * self.mutation_scale
        delta = torch.clamp(delta, -0.1, 0.1)

        mutated = (chunks + delta).view(-1)[:n]
        return mutated


class CPPNMutator(nn.Module):
    """CPPN-style mutator."""

    def __init__(self, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(4, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.mutation_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords) * self.mutation_scale

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None) -> torch.Tensor:
        if weight_coords is None:
            raise ValueError("CPPN requires weight_coords")
        delta = self.net(weight_coords).squeeze(-1) * self.mutation_scale
        delta = torch.clamp(delta, -0.1, 0.1)
        return flat_weights + delta


class Genome:
    """
    A complete genome: policy + mutator.

    Stability features:
    - Mutator weights receive 10x smaller perturbations than policy weights
    - Mutation scale decays over generations
    - Output deltas are clamped to [-0.1, 0.1]
    - Global weight clipping after mutation
    """

    def __init__(self, policy: Policy, mutator: nn.Module,
                 mutator_type: str = 'chunk'):
        self.policy = policy
        self.mutator = mutator
        self.mutator_type = mutator_type
        self.fitness = 0.0
        self._weight_coords = None

    def get_flat_weights(self) -> torch.Tensor:
        policy_params = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        mutator_params = torch.cat([p.data.view(-1) for p in self.mutator.parameters()])
        return torch.cat([policy_params, mutator_params])

    def get_flat_policy_weights(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.policy.parameters()])

    def get_flat_mutator_weights(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.mutator.parameters()])

    def set_flat_weights(self, flat: torch.Tensor):
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
    def reproduce(self, generation: int = 0, max_generations: int = 100) -> 'Genome':
        """
        Self-replication with stability improvements:
        1. Only mutate policy weights through the mutator
        2. Apply small Gaussian noise to mutator weights (protective)
        3. Decay mutation scale over generations
        4. Clip all weights after mutation
        """
        n_policy = self.num_policy_params()
        policy_flat = self.get_flat_policy_weights()
        mutator_flat = self.get_flat_mutator_weights()

        # Decay factor: starts at 1.0, decays to 0.2 over generations
        decay = max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))

        # Mutate policy weights through the mutator network
        if self.mutator_type == 'cppn':
            # For CPPN, only use policy coords
            coords = self._get_policy_weight_coords()
            mutated_policy = self.mutator.mutate_genome(policy_flat, coords)
        elif self.mutator_type == 'gaussian':
            mutated_policy = self.mutator.mutate_genome(policy_flat)
        else:
            mutated_policy = self.mutator.mutate_genome(policy_flat)

        # Apply decay to the delta
        delta = mutated_policy - policy_flat
        mutated_policy = policy_flat + delta * decay

        # Small protective noise on mutator weights (not through the mutator itself)
        mutator_noise = torch.randn_like(mutator_flat) * 0.002 * decay
        mutated_mutator = mutator_flat + mutator_noise

        new_flat = torch.cat([mutated_policy, mutated_mutator])

        # Global weight clipping to prevent explosion
        new_flat = torch.clamp(new_flat, -5.0, 5.0)

        child = Genome(
            policy=self._clone_policy(),
            mutator=self._clone_mutator(),
            mutator_type=self.mutator_type,
        )
        child.set_flat_weights(new_flat)
        return child

    @torch.no_grad()
    def crossover(self, other: 'Genome', generation: int = 0,
                  max_generations: int = 100) -> 'Genome':
        """Crossover with same stability protections."""
        my_policy = self.get_flat_policy_weights()

        decay = max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))

        if other.mutator_type == 'cppn':
            coords = other._get_policy_weight_coords()
            mutated_policy = other.mutator.mutate_genome(my_policy, coords)
        elif other.mutator_type == 'gaussian':
            mutated_policy = other.mutator.mutate_genome(my_policy)
        else:
            mutated_policy = other.mutator.mutate_genome(my_policy)

        delta = mutated_policy - my_policy
        mutated_policy = my_policy + delta * decay

        # Keep other's mutator with small noise
        other_mutator = other.get_flat_mutator_weights()
        mutator_noise = torch.randn_like(other_mutator) * 0.002 * decay
        mutated_mutator = other_mutator + mutator_noise

        new_flat = torch.cat([mutated_policy, mutated_mutator])
        new_flat = torch.clamp(new_flat, -5.0, 5.0)

        child = Genome(
            policy=self._clone_policy(),
            mutator=other._clone_mutator(),
            mutator_type=other.mutator_type,
        )
        child.set_flat_weights(new_flat)
        return child

    def _clone_policy(self) -> Policy:
        first_layer = self.policy.net[0]
        last_layer = self.policy.net[-1]
        hidden = first_layer.out_features
        return Policy(first_layer.in_features, last_layer.out_features, hidden)

    def _clone_mutator(self) -> nn.Module:
        import copy
        return copy.deepcopy(self.mutator)

    def _get_policy_weight_coords(self) -> torch.Tensor:
        """Generate coordinates only for policy weights (for CPPN stability)."""
        coords = []
        layer_idx = 0
        for name, p in self.policy.named_parameters():
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
                for i in range(p.numel()):
                    coords.append([layer_idx / 10.0, i / max(p.numel(), 1), 0.0, 0.5])
            if not is_bias:
                layer_idx += 1
        return torch.FloatTensor(coords)

    def _get_weight_coords(self) -> torch.Tensor:
        """Generate normalized coordinates for all weights (for CPPN)."""
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
                for i in range(p.numel()):
                    coords.append([layer_idx / 10.0, i / max(p.numel(), 1), 0.0, 0.5])
            if not is_bias:
                layer_idx += 1

        self._weight_coords = torch.FloatTensor(coords)
        return self._weight_coords
