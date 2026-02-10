"""
Genome: Policy + Mutator bundled together with TRUE self-replication.

The mutator processes the ENTIRE genome (policy + mutator weights) and outputs
a new full genome — including its own weights. This is true self-replication.

Stabilization: asymmetric mutation rates (0.1x for mutator weights vs 1x for policy).
Natural selection handles the rest — if a mutator destroys itself, it dies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import copy


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
    """Baseline: simple Gaussian noise mutation (ES-style)."""

    def __init__(self, mutation_scale: float = 0.02):
        super().__init__()
        self.base_scale = mutation_scale
        self.dummy = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None) -> torch.Tensor:
        noise = torch.randn_like(flat_weights) * self.base_scale
        return flat_weights + noise


class ChunkMutator(nn.Module):
    """Chunk-based MLP mutator."""

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

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None) -> torch.Tensor:
        n = flat_weights.shape[0]
        cs = self.chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len)])

        chunks = padded.view(-1, cs)
        delta = self.net(chunks) * self.mutation_scale
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


class Genome:
    """
    A complete genome: policy + mutator with TRUE self-replication.

    The mutator processes the ENTIRE genome (policy + its own weights)
    and produces a new full genome. Stabilized only by asymmetric mutation
    rates — natural selection handles the rest.
    """

    # Asymmetric rate: mutator weights get this fraction of mutation
    MUTATOR_SELF_RATE = 0.1

    def __init__(self, policy: Policy, mutator: nn.Module,
                 mutator_type: str = 'chunk'):
        self.policy = policy
        self.mutator = mutator
        self.mutator_type = mutator_type
        self.fitness = 0.0
        self.self_replication_fidelity = 0.0  # how well mutator preserves itself
        self.mutator_delta_norm = 0.0  # magnitude of mutator weight change

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
        TRUE self-replication: feed the ENTIRE genome (policy + mutator weights)
        through the mutator, producing a new full genome.

        Stabilization: asymmetric rates (mutator delta scaled by 0.1x) + decay + clipping.
        """
        n_policy = self.num_policy_params()
        n_mutator = self.num_mutator_params()
        parent_full = self.get_flat_weights()
        parent_mutator = parent_full[n_policy:]

        # Decay factor
        decay = max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))

        # Feed FULL genome through the mutator
        if self.mutator_type == 'gaussian':
            mutated_full = self.mutator.mutate_genome(parent_full)
        else:
            mutated_full = self.mutator.mutate_genome(parent_full)

        # Compute deltas
        full_delta = mutated_full - parent_full
        policy_delta = full_delta[:n_policy]
        mutator_delta = full_delta[n_policy:]

        # Apply decay to policy delta
        policy_delta = policy_delta * decay

        # ASYMMETRIC MUTATION: scale down mutator self-modification
        mutator_delta = mutator_delta * decay * self.MUTATOR_SELF_RATE
        mutator_delta_norm = torch.norm(mutator_delta).item()

        # Reconstruct
        new_policy = parent_full[:n_policy] + policy_delta
        new_mutator = parent_mutator + mutator_delta
        new_flat = torch.cat([new_policy, new_mutator])

        # Global weight clipping
        new_flat = torch.clamp(new_flat, -5.0, 5.0)

        # Compute self-replication fidelity (L2 distance, lower = better preservation)
        fidelity = torch.norm(new_mutator - parent_mutator).item()

        child = Genome(
            policy=self._clone_policy(),
            mutator=self._clone_mutator(),
            mutator_type=self.mutator_type,
        )
        child.set_flat_weights(new_flat)
        child.self_replication_fidelity = fidelity
        child.mutator_delta_norm = mutator_delta_norm
        return child

    @torch.no_grad()
    def crossover(self, other: 'Genome', generation: int = 0,
                  max_generations: int = 100) -> 'Genome':
        """Crossover with same self-replication stabilization."""
        n_policy = self.num_policy_params()
        # Use self's policy + other's mutator, feed full genome through other's mutator
        my_policy = self.get_flat_policy_weights()
        other_mutator = other.get_flat_mutator_weights()
        combined = torch.cat([my_policy, other_mutator])

        decay = max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))

        if other.mutator_type == 'gaussian':
            mutated_full = other.mutator.mutate_genome(combined)
        else:
            mutated_full = other.mutator.mutate_genome(combined)

        full_delta = mutated_full - combined
        policy_delta = full_delta[:n_policy] * decay
        mutator_delta = full_delta[n_policy:] * decay * self.MUTATOR_SELF_RATE
        mutator_delta_norm = torch.norm(mutator_delta).item()

        new_policy = my_policy + policy_delta
        new_mutator = other_mutator + mutator_delta
        new_flat = torch.cat([new_policy, new_mutator])
        new_flat = torch.clamp(new_flat, -5.0, 5.0)

        fidelity = torch.norm(new_mutator - other_mutator).item()

        child = Genome(
            policy=self._clone_policy(),
            mutator=other._clone_mutator(),
            mutator_type=other.mutator_type,
        )
        child.set_flat_weights(new_flat)
        child.self_replication_fidelity = fidelity
        child.mutator_delta_norm = mutator_delta_norm
        return child

    def _clone_policy(self) -> Policy:
        first_layer = self.policy.net[0]
        last_layer = self.policy.net[-1]
        hidden = first_layer.out_features
        return Policy(first_layer.in_features, last_layer.out_features, hidden)

    def _clone_mutator(self) -> nn.Module:
        return copy.deepcopy(self.mutator)
