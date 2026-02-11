"""
Genome: Policy + Mutator bundled together with TRUE self-replication.

The mutator processes the ENTIRE genome (policy + mutator weights) and outputs
a new full genome — including its own weights. This is true self-replication.

Stabilization: asymmetric mutation rates (0.1x for mutator weights vs 1x for policy).
Natural selection handles the rest — if a mutator destroys itself, it dies.
"""

import io
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import copy
import random
from enum import Enum


class StructuralMutation(str, Enum):
    ADD_NEURON = 'add_neuron'
    REMOVE_NEURON = 'remove_neuron'
    ADD_LAYER = 'add_layer'
    REMOVE_LAYER = 'remove_layer'


# Constraints for flexible architectures
MIN_NEURONS = 8
MAX_NEURONS = 256
MIN_LAYERS = 1
MAX_LAYERS = 5


class Policy(nn.Module):
    """Simple MLP policy for RL environments (fixed architecture, backward compat)."""

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


# Alias for backward compatibility
FixedPolicy = Policy


class FlexiblePolicy(nn.Module):
    """MLP policy with variable number of layers and neurons per layer."""

    def __init__(self, obs_dim: int, act_dim: int, layer_sizes: List[int] = None):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [64, 64]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layer_sizes = list(layer_sizes)
        self._build_net()

    def _build_net(self):
        layers = []
        prev = self.obs_dim
        for size in self.layer_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.Tanh())
            prev = size
        layers.append(nn.Linear(prev, self.act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0)
            action = self.net(x).squeeze(0)
        return action.numpy()

    def get_topology(self) -> List[int]:
        """Return list of hidden layer sizes."""
        return list(self.layer_sizes)

    def _get_layer_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (weight, bias) for a linear layer by hidden-layer index."""
        # Linear layers are at positions 0, 2, 4, ... in self.net
        linear_idx = layer_idx * 2  # skip Tanh layers
        layer = self.net[linear_idx]
        return layer.weight.data.clone(), layer.bias.data.clone()

    def _get_output_layer(self) -> nn.Linear:
        """Get the output linear layer."""
        return self.net[-1]

    @torch.no_grad()
    def add_neuron(self, layer_idx: int = None) -> bool:
        """Add a neuron to a hidden layer. Returns True if successful."""
        if layer_idx is None:
            layer_idx = random.randint(0, len(self.layer_sizes) - 1)
        if layer_idx >= len(self.layer_sizes):
            return False
        if self.layer_sizes[layer_idx] >= MAX_NEURONS:
            return False

        old_size = self.layer_sizes[layer_idx]
        new_size = old_size + 1
        self.layer_sizes[layer_idx] = new_size

        # Save all weights
        old_params = [p.data.clone() for p in self.parameters()]
        self._build_net()

        # Restore weights, extending the modified layer
        param_iter = iter(self.parameters())
        old_iter = iter(old_params)
        linear_count = 0
        for module in self.net:
            if not isinstance(module, nn.Linear):
                continue
            w_new = next(param_iter)  # weight
            b_new = next(param_iter)  # bias
            w_old = next(old_iter)
            b_old = next(old_iter)

            if linear_count == layer_idx:
                # This layer gained an output neuron
                w_new.data[:old_size, :] = w_old
                b_new.data[:old_size] = b_old
                # New neuron: small random init
                w_new.data[old_size, :] = torch.randn(w_old.shape[1]) * 0.01
                b_new.data[old_size] = 0.01 * torch.randn(1).item()
            elif linear_count == layer_idx + 1:
                # Next layer gained an input
                w_new.data[:, :old_size] = w_old
                b_new.data[:] = b_old
                # New input column: small random
                w_new.data[:, old_size] = torch.randn(w_old.shape[0]) * 0.01
            else:
                w_new.data.copy_(w_old)
                b_new.data.copy_(b_old)
            linear_count += 1
        return True

    @torch.no_grad()
    def remove_neuron(self, layer_idx: int = None) -> bool:
        """Remove the neuron with smallest L2 norm from a hidden layer."""
        if layer_idx is None:
            layer_idx = random.randint(0, len(self.layer_sizes) - 1)
        if layer_idx >= len(self.layer_sizes):
            return False
        if self.layer_sizes[layer_idx] <= MIN_NEURONS:
            return False

        # Find neuron with smallest L2 norm in outgoing weights
        linear_idx = layer_idx * 2
        layer = self.net[linear_idx]
        norms = torch.norm(layer.weight.data, dim=1)
        remove_idx = torch.argmin(norms).item()

        old_size = self.layer_sizes[layer_idx]
        new_size = old_size - 1
        self.layer_sizes[layer_idx] = new_size

        old_params = [p.data.clone() for p in self.parameters()]
        self._build_net()

        param_iter = iter(self.parameters())
        old_iter = iter(old_params)
        keep = [i for i in range(old_size) if i != remove_idx]
        linear_count = 0
        for module in self.net:
            if not isinstance(module, nn.Linear):
                continue
            w_new = next(param_iter)
            b_new = next(param_iter)
            w_old = next(old_iter)
            b_old = next(old_iter)

            if linear_count == layer_idx:
                w_new.data[:] = w_old[keep, :]
                b_new.data[:] = b_old[keep]
            elif linear_count == layer_idx + 1:
                w_new.data[:] = w_old[:, keep]
                b_new.data[:] = b_old
            else:
                w_new.data.copy_(w_old)
                b_new.data.copy_(b_old)
            linear_count += 1
        return True

    @torch.no_grad()
    def add_layer(self, insert_idx: int = None) -> bool:
        """Add a near-identity hidden layer."""
        if len(self.layer_sizes) >= MAX_LAYERS:
            return False
        if insert_idx is None:
            insert_idx = random.randint(0, len(self.layer_sizes))

        # New layer size = size of the layer it's inserted after (or obs_dim if at start)
        if insert_idx == 0:
            new_layer_size = self.layer_sizes[0] if self.layer_sizes else 32
        elif insert_idx >= len(self.layer_sizes):
            new_layer_size = self.layer_sizes[-1]
        else:
            new_layer_size = self.layer_sizes[insert_idx - 1]

        old_params = [p.data.clone() for p in self.parameters()]
        old_layer_sizes = list(self.layer_sizes)
        self.layer_sizes.insert(insert_idx, new_layer_size)
        self._build_net()

        # Restore old weights, init new layer as near-identity
        param_iter = iter(self.parameters())
        old_iter = iter(old_params)
        linear_count = 0
        for module in self.net:
            if not isinstance(module, nn.Linear):
                continue
            w_new = next(param_iter)
            b_new = next(param_iter)

            if linear_count == insert_idx:
                # Near-identity init
                nn.init.eye_(w_new.data[:min(w_new.shape[0], w_new.shape[1]),
                                        :min(w_new.shape[0], w_new.shape[1])])
                if w_new.shape[0] != w_new.shape[1]:
                    # Zero out non-square parts, add small noise
                    w_new.data += torch.randn_like(w_new.data) * 0.01
                else:
                    w_new.data += torch.randn_like(w_new.data) * 0.01
                b_new.data.zero_()
            elif linear_count == insert_idx + 1:
                # The layer after the new one now has different input size
                # We need to handle dimension mismatch
                w_old = next(old_iter)
                b_old = next(old_iter)
                min_in = min(w_new.shape[1], w_old.shape[1])
                min_out = min(w_new.shape[0], w_old.shape[0])
                w_new.data.zero_()
                w_new.data[:min_out, :min_in] = w_old[:min_out, :min_in]
                b_new.data[:min_out] = b_old[:min_out]
            else:
                w_old = next(old_iter)
                b_old = next(old_iter)
                w_new.data.copy_(w_old)
                b_new.data.copy_(b_old)
            linear_count += 1
        return True

    @torch.no_grad()
    def remove_layer(self, layer_idx: int = None) -> bool:
        """Remove a hidden layer, approximately merging with adjacent."""
        if len(self.layer_sizes) <= MIN_LAYERS:
            return False
        if layer_idx is None:
            layer_idx = random.randint(0, len(self.layer_sizes) - 1)
        if layer_idx >= len(self.layer_sizes):
            return False

        # Get the two linear layers that will be merged:
        # layer at layer_idx (linear_idx = layer_idx*2) and the one after (linear_idx+2)
        li = layer_idx * 2
        W1 = self.net[li].weight.data.clone()      # (layer_size, prev_size)
        b1 = self.net[li].bias.data.clone()
        W2 = self.net[li + 2].weight.data.clone()   # (next_size, layer_size)
        b2 = self.net[li + 2].bias.data.clone()

        # Approximate merge: W_merged = W2 @ W1, b_merged = W2 @ b1 + b2
        # (ignores the tanh nonlinearity — approximate)
        W_merged = W2 @ W1
        b_merged = W2 @ b1 + b2

        old_params = [p.data.clone() for p in self.parameters()]
        self.layer_sizes.pop(layer_idx)
        self._build_net()

        # Restore weights
        param_iter = iter(self.parameters())
        old_iter = iter(old_params)
        linear_count = 0
        merged = False
        for module in self.net:
            if not isinstance(module, nn.Linear):
                continue
            w_new = next(param_iter)
            b_new = next(param_iter)

            if linear_count == layer_idx and not merged:
                # This is where the merged layer goes
                w_new.data.copy_(W_merged)
                b_new.data.copy_(b_merged)
                # Skip two old layers
                next(old_iter); next(old_iter)  # W1, b1
                next(old_iter); next(old_iter)  # W2, b2
                merged = True
            else:
                w_old = next(old_iter)
                b_old = next(old_iter)
                w_new.data.copy_(w_old)
                b_new.data.copy_(b_old)
            linear_count += 1
        return True

    @torch.no_grad()
    def apply_structural_mutation(self, mutation_type: StructuralMutation = None) -> Optional[StructuralMutation]:
        """Apply a random structural mutation. Returns the mutation applied, or None if failed."""
        if mutation_type is None:
            candidates = list(StructuralMutation)
            random.shuffle(candidates)
            for m in candidates:
                if self._can_apply(m):
                    mutation_type = m
                    break
            if mutation_type is None:
                return None

        dispatch = {
            StructuralMutation.ADD_NEURON: self.add_neuron,
            StructuralMutation.REMOVE_NEURON: self.remove_neuron,
            StructuralMutation.ADD_LAYER: self.add_layer,
            StructuralMutation.REMOVE_LAYER: self.remove_layer,
        }
        success = dispatch[mutation_type]()
        return mutation_type if success else None

    def _can_apply(self, m: StructuralMutation) -> bool:
        if m == StructuralMutation.ADD_NEURON:
            return any(s < MAX_NEURONS for s in self.layer_sizes)
        elif m == StructuralMutation.REMOVE_NEURON:
            return any(s > MIN_NEURONS for s in self.layer_sizes)
        elif m == StructuralMutation.ADD_LAYER:
            return len(self.layer_sizes) < MAX_LAYERS
        elif m == StructuralMutation.REMOVE_LAYER:
            return len(self.layer_sizes) > MIN_LAYERS
        return False


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


class ErrorCorrectorMutator(nn.Module):
    """
    Learned error-correction mutator.
    
    Instead of blind transformation, this mutator maintains a learned "reference"
    for what good weights look like. It compares the current genome to the reference
    and outputs targeted corrections — pushing weights toward the learned ideal.
    
    Architecture:
    - Reference: a learned parameter vector (compressed genome representation)
    - Encoder: compresses current weights to same space as reference
    - Corrector: takes (current_encoding, reference, difference) → correction delta
    
    If weights drift due to bad crossover or mutation, the corrector steers them back.
    Evolution selects for mutators whose reference leads to high fitness.
    """

    def __init__(self, chunk_size: int = 64, ref_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.chunk_size = chunk_size
        self.ref_dim = ref_dim
        
        # Learned reference: the mutator's internal "ideal" (evolved, not trained)
        self.reference = nn.Parameter(torch.randn(ref_dim) * 0.1)
        
        # Encoder: compress each chunk to reference space
        self.encoder = nn.Sequential(
            nn.Linear(chunk_size, ref_dim),
            nn.Tanh(),
        )
        
        # Corrector: takes current encoding + reference + diff → correction
        self.corrector = nn.Sequential(
            nn.Linear(ref_dim * 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, chunk_size),
            nn.Tanh(),
        )
        
        # Learned correction scale
        self.correction_scale = nn.Parameter(torch.tensor(0.02))
        
        # Exploration noise scale
        self.explore_scale = nn.Parameter(torch.tensor(0.005))

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None) -> torch.Tensor:
        n = flat_weights.shape[0]
        cs = self.chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len)])
        
        chunks = padded.view(-1, cs)
        num_chunks = chunks.shape[0]
        
        # Encode each chunk
        encoded = self.encoder(chunks)  # (num_chunks, ref_dim)
        
        # Broadcast reference to all chunks
        ref = self.reference.unsqueeze(0).expand(num_chunks, -1)
        
        # Compute difference: how far is each chunk from the "ideal"?
        diff = encoded - ref
        
        # Corrector input: current state + ideal + error signal
        corrector_input = torch.cat([encoded, ref, diff], dim=1)
        
        # Compute targeted correction
        correction = self.corrector(corrector_input) * self.correction_scale
        correction = torch.clamp(correction, -0.1, 0.1)
        
        # Add small exploration noise (prevents collapse to pure fixed point)
        noise = torch.randn_like(chunks) * self.explore_scale
        
        mutated = chunks + correction + noise
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


class CompatibilityNet(nn.Module):
    """
    Learned compatibility function for speciation.
    
    Uses random projection to compress variable-length genomes into fixed-size
    summaries, then a learned scorer decides compatibility.
    Each genome carries its own compat net — both must "agree" for crossover.
    The mutator modifies compat net weights alongside everything else.
    """
    
    SUMMARY_DIM = 64   # fixed-size genome summary via random projection
    EMBED_DIM = 16     # compatibility embedding dimension
    
    def __init__(self, genome_dim: int):
        super().__init__()
        self.genome_dim = genome_dim
        # Random projection matrix (NOT learned — frozen, just for compression)
        # Registered as buffer so it's not in parameters()
        proj = torch.randn(genome_dim, self.SUMMARY_DIM) / (genome_dim ** 0.5)
        self.register_buffer('projection', proj)
        
        # Learned encoder: summary → embedding
        self.encoder = nn.Sequential(
            nn.Linear(self.SUMMARY_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, self.EMBED_DIM),
        )
        # Scores compatibility between own embedding and other's embedding
        self.scorer = nn.Sequential(
            nn.Linear(self.EMBED_DIM * 2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Weights start random — call pretrain() to warm-start before evolution
    
    def summarize(self, genome_flat: torch.Tensor) -> torch.Tensor:
        """Compress genome to fixed-size summary via random projection."""
        # Handle genomes of different sizes than expected
        if genome_flat.shape[0] < self.genome_dim:
            padded = torch.zeros(self.genome_dim)
            padded[:genome_flat.shape[0]] = genome_flat
            genome_flat = padded
        elif genome_flat.shape[0] > self.genome_dim:
            genome_flat = genome_flat[:self.genome_dim]
        return genome_flat @ self.projection
    
    def embed(self, genome_flat: torch.Tensor) -> torch.Tensor:
        """Produce compatibility embedding for a genome."""
        summary = self.summarize(genome_flat)
        return self.encoder(summary)
    
    def score(self, my_embed: torch.Tensor, other_embed: torch.Tensor) -> float:
        """Score compatibility using learned scorer."""
        combined = torch.cat([my_embed, other_embed])
        return self.scorer(combined).item()

    def pretrain(self, genomes_flat: list, steps: int = 300, lr: float = 0.01):
        """
        Pre-train compat net so its embeddings preserve genome distance structure.
        
        Contrastive loss: similar genomes → close embeddings, dissimilar → far.
        This gives evolution a meaningful species tag space to start from.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        n = len(genomes_flat)
        # Pad to max length for distance computation (flex genomes have different sizes)
        max_len = max(g.shape[0] for g in genomes_flat)
        padded = []
        for g in genomes_flat:
            if g.shape[0] < max_len:
                padded.append(torch.cat([g, torch.zeros(max_len - g.shape[0])]))
            else:
                padded.append(g)
        
        # Precompute pairwise genome distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(padded[i] - padded[j]).item()
                distances.append((i, j, d))
        
        if not distances:
            return
        
        median_dist = sorted([d for _, _, d in distances])[len(distances) // 2]
        
        for step in range(steps):
            # Sample random pair
            idx = np.random.randint(len(distances))
            i, j, genome_dist = distances[idx]
            
            ei = self.embed(padded[i])
            ej = self.embed(padded[j])
            embed_dist = torch.norm(ei - ej)
            
            # Contrastive: if genomes are similar, embeddings should be close
            # if dissimilar, embeddings should be far (margin=2.0)
            margin = 2.0
            if genome_dist < median_dist:
                loss = embed_dist ** 2  # pull together
            else:
                loss = torch.clamp(margin - embed_dist, min=0) ** 2  # push apart
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class Genome:
    """
    A complete genome: policy + mutator + compatibility net with TRUE self-replication.

    The mutator processes the ENTIRE genome (policy + mutator + compat weights)
    and produces a new full genome. Stabilized only by asymmetric mutation
    rates — natural selection handles the rest.
    
    Learned speciation: each genome carries a CompatibilityNet that decides
    who it can crossover with. Both genomes must agree (score > threshold).
    """

    # Asymmetric rates: mutator self-modifies slowly, compat evolves freely
    MUTATOR_SELF_RATE = 0.1
    COMPAT_RATE = 1.0  # compat net mutates at full rate — speciation needs agility
    DEFAULT_STRUCTURAL_RATE = 0.05  # 5% base probability of structural mutation

    def __init__(self, policy, mutator: nn.Module,
                 mutator_type: str = 'chunk',
                 compat_net: 'CompatibilityNet | None' = None):
        self.policy = policy
        self.mutator = mutator
        self.mutator_type = mutator_type
        self.compat_net = compat_net  # None = no speciation (backward compat)
        self.fitness = 0.0
        self.self_replication_fidelity = 0.0  # how well mutator preserves itself
        self.mutator_delta_norm = 0.0  # magnitude of mutator weight change
        self.species_id = -1  # assigned during speciation
        # Learned structural mutation rate (only meaningful for FlexiblePolicy)
        self.structural_rate = self.DEFAULT_STRUCTURAL_RATE
        self.last_structural_mutation = None  # track what happened

    def get_flat_weights(self) -> torch.Tensor:
        """Get all weights: policy + mutator + compat (if present)."""
        policy_params = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        mutator_params = torch.cat([p.data.view(-1) for p in self.mutator.parameters()])
        parts = [policy_params, mutator_params]
        if self.compat_net is not None:
            compat_params = torch.cat([p.data.view(-1) for p in self.compat_net.parameters()])
            parts.append(compat_params)
        return torch.cat(parts)

    def get_flat_policy_weights(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.policy.parameters()])

    def get_flat_mutator_weights(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.mutator.parameters()])

    def get_flat_compat_weights(self) -> torch.Tensor:
        if self.compat_net is None:
            return torch.tensor([])
        return torch.cat([p.data.view(-1) for p in self.compat_net.parameters()])

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
        if self.compat_net is not None:
            for p in self.compat_net.parameters():
                n = p.numel()
                p.data.copy_(flat[idx:idx + n].view(p.shape))
                idx += n

    def num_policy_params(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())

    def num_mutator_params(self) -> int:
        return sum(p.numel() for p in self.mutator.parameters())

    def num_compat_params(self) -> int:
        if self.compat_net is None:
            return 0
        return sum(p.numel() for p in self.compat_net.parameters())

    def num_total_params(self) -> int:
        return self.num_policy_params() + self.num_mutator_params() + self.num_compat_params()

    def get_species_tag(self) -> torch.Tensor:
        """Get learned species tag — a low-dim embedding from the compat net."""
        if self.compat_net is None:
            return torch.zeros(CompatibilityNet.EMBED_DIM)
        with torch.no_grad():
            return self.compat_net.embed(self.get_flat_weights())
    
    def is_compatible(self, other: 'Genome', threshold: float = 0.5) -> bool:
        """Check compatibility via L2 distance between species tags."""
        if self.compat_net is None or other.compat_net is None:
            return True
        
        with torch.no_grad():
            my_tag = self.get_species_tag()
            other_tag = other.get_species_tag()
            dist = torch.norm(my_tag - other_tag).item()
            # Threshold on distance — closer = more compatible
            # Use threshold as max distance (scaled by embed dim)
            max_dist = threshold * (CompatibilityNet.EMBED_DIM ** 0.5)
            return dist < max_dist

    @property
    def is_flexible(self) -> bool:
        return isinstance(self.policy, FlexiblePolicy)

    @torch.no_grad()
    def reproduce(self, generation: int = 0, max_generations: int = 100) -> 'Genome':
        """
        TRUE self-replication: feed the ENTIRE genome (policy + mutator + compat weights)
        through the mutator, producing a new full genome.

        For FlexiblePolicy genomes: may apply a structural mutation first (controlled
        by learned structural_rate), then mutate weights on the new architecture.

        Stabilization: asymmetric rates (mutator/compat delta scaled by 0.1x) + decay + clipping.
        """
        # Step 1: Clone child with same architecture
        child = Genome(
            policy=self._clone_policy(),
            mutator=self._clone_mutator(),
            mutator_type=self.mutator_type,
            compat_net=self._clone_compat() if self.compat_net else None,
        )
        child.set_flat_weights(self.get_flat_weights())
        child.structural_rate = self.structural_rate
        child.last_structural_mutation = None

        # Step 2: Possibly apply structural mutation (FlexiblePolicy only)
        if self.is_flexible and random.random() < self.structural_rate:
            mutation = child.policy.apply_structural_mutation()
            child.last_structural_mutation = mutation
            # Mutate structural_rate itself (log-normal walk, asymmetric like mutator)
            child.structural_rate = np.clip(
                self.structural_rate * np.exp(np.random.randn() * 0.1 * self.MUTATOR_SELF_RATE),
                0.001, 0.5
            )

        # Step 3: Weight mutation on the (possibly new) architecture
        n_policy = child.num_policy_params()
        n_mutator = child.num_mutator_params()
        n_compat = child.num_compat_params()

        # If structural mutation changed genome size, we need parent mutator on parent genome
        parent_full = self.get_flat_weights()
        parent_meta = self.get_flat_weights()[self.num_policy_params():]  # mutator + compat weights

        decay = max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))

        # Feed parent's full genome through parent's mutator
        mutated_full = self.mutator.mutate_genome(parent_full)
        full_delta = mutated_full - parent_full

        # Extract policy delta — may need to resize if architecture changed
        parent_n_policy = self.num_policy_params()
        policy_delta_raw = full_delta[:parent_n_policy]

        if n_policy == parent_n_policy:
            policy_delta = policy_delta_raw * decay
        else:
            # Architecture changed: truncate or pad the delta
            policy_delta = torch.zeros(n_policy)
            copy_len = min(n_policy, parent_n_policy)
            policy_delta[:copy_len] = policy_delta_raw[:copy_len] * decay

        # ASYMMETRIC MUTATION: mutator slow, compat fast
        parent_n_mutator = self.num_mutator_params()
        parent_n_compat = self.num_compat_params()
        meta_delta = full_delta[parent_n_policy:parent_n_policy + parent_n_mutator + parent_n_compat]
        if meta_delta.shape[0] < parent_n_mutator + parent_n_compat:
            meta_delta = torch.cat([meta_delta, torch.zeros(parent_n_mutator + parent_n_compat - meta_delta.shape[0])])
        meta_delta = meta_delta[:parent_n_mutator + parent_n_compat]
        
        mutator_delta = meta_delta[:n_mutator] if meta_delta.shape[0] >= n_mutator else torch.cat([meta_delta, torch.zeros(n_mutator - meta_delta.shape[0])])
        mutator_delta = mutator_delta[:n_mutator] * decay * self.MUTATOR_SELF_RATE
        mutator_delta_norm = torch.norm(mutator_delta).item()
        
        if n_compat > 0:
            compat_noise = torch.randn(n_compat) * 0.05 * decay
            meta_delta_final = torch.cat([mutator_delta, compat_noise])
        else:
            meta_delta_final = mutator_delta

        # Apply deltas to child
        child_flat = child.get_flat_weights()
        new_policy = child_flat[:n_policy] + policy_delta
        child_meta = child_flat[n_policy:]
        new_meta = child_meta + meta_delta_final[:child_meta.shape[0]]
        new_flat = torch.cat([new_policy, new_meta])
        new_flat = torch.clamp(new_flat, -5.0, 5.0)

        fidelity = torch.norm(mutator_delta).item()

        child.set_flat_weights(new_flat)
        child.self_replication_fidelity = fidelity
        child.mutator_delta_norm = mutator_delta_norm
        return child

    @torch.no_grad()
    def crossover(self, other: 'Genome', generation: int = 0,
                  max_generations: int = 100) -> 'Genome':
        """Crossover with learned speciation check and self-replication stabilization.
        
        For FlexiblePolicy: uses the fitter parent's architecture (self is assumed fitter).
        """
        # For flexible policies with different architectures, use self's architecture
        child_policy = self._clone_policy()
        n_policy = sum(p.numel() for p in child_policy.parameters())
        n_mutator = other.num_mutator_params()
        n_compat = other.num_compat_params()
        
        # Use self's policy weights (possibly resized) + other's mutator/compat
        my_policy = self.get_flat_policy_weights()
        # Resize if needed
        if my_policy.shape[0] != n_policy:
            resized = torch.zeros(n_policy)
            copy_len = min(my_policy.shape[0], n_policy)
            resized[:copy_len] = my_policy[:copy_len]
            my_policy = resized

        other_meta = torch.cat([other.get_flat_mutator_weights(), other.get_flat_compat_weights()])
        combined = torch.cat([my_policy, other_meta])

        decay = max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))

        mutated_full = other.mutator.mutate_genome(combined)

        full_delta = mutated_full - combined
        policy_delta = full_delta[:n_policy] * decay
        meta_delta = full_delta[n_policy:n_policy + n_mutator + n_compat]
        if meta_delta.shape[0] < n_mutator + n_compat:
            meta_delta = torch.cat([meta_delta, torch.zeros(n_mutator + n_compat - meta_delta.shape[0])])
        meta_delta = meta_delta[:n_mutator + n_compat]
        
        mutator_d = meta_delta[:n_mutator] * decay * self.MUTATOR_SELF_RATE
        mutator_delta_norm = torch.norm(mutator_d).item()
        
        if n_compat > 0:
            compat_noise = torch.randn(n_compat) * 0.05 * decay
            meta_delta = torch.cat([mutator_d, compat_noise])
        else:
            meta_delta = mutator_d

        new_policy = my_policy + policy_delta
        new_meta = other_meta + meta_delta
        new_flat = torch.cat([new_policy, new_meta])
        new_flat = torch.clamp(new_flat, -5.0, 5.0)

        fidelity = torch.norm(new_meta[:n_mutator] - other_meta[:n_mutator]).item()

        child = Genome(
            policy=child_policy,
            mutator=other._clone_mutator(),
            mutator_type=other.mutator_type,
            compat_net=other._clone_compat() if other.compat_net else None,
        )
        child.set_flat_weights(new_flat)
        child.self_replication_fidelity = fidelity
        child.mutator_delta_norm = mutator_delta_norm
        child.structural_rate = (self.structural_rate + other.structural_rate) / 2
        return child

    def _clone_policy(self):
        if isinstance(self.policy, FlexiblePolicy):
            return FlexiblePolicy(self.policy.obs_dim, self.policy.act_dim,
                                  list(self.policy.layer_sizes))
        first_layer = self.policy.net[0]
        last_layer = self.policy.net[-1]
        hidden = first_layer.out_features
        return Policy(first_layer.in_features, last_layer.out_features, hidden)

    def _clone_mutator(self) -> nn.Module:
        return copy.deepcopy(self.mutator)

    def _clone_compat(self) -> 'CompatibilityNet | None':
        if self.compat_net is None:
            return None
        return copy.deepcopy(self.compat_net)

    def save(self, path):
        """Save genome to disk or file-like object."""
        data = {
            'policy_state': self.policy.state_dict(),
            'mutator_state': self.mutator.state_dict(),
            'obs_dim': self.policy.obs_dim if hasattr(self.policy, 'obs_dim') else self.policy.net[0].in_features,
            'act_dim': self.policy.act_dim if hasattr(self.policy, 'act_dim') else self.policy.net[-1].out_features,
            'mutator_type': type(self.mutator).__name__,
            'fitness': self.fitness,
        }
        if isinstance(self.policy, FlexiblePolicy):
            data['flex'] = True
            data['layer_sizes'] = list(self.policy.layer_sizes)
        else:
            data['flex'] = False
            data['hidden'] = self.policy.net[0].out_features
        torch.save(data, path)

    @staticmethod
    def load(path) -> 'Genome':
        """Load genome from disk or file-like object."""
        data = torch.load(path, weights_only=False)
        obs_dim, act_dim = data['obs_dim'], data['act_dim']
        if data.get('flex'):
            policy = FlexiblePolicy(obs_dim, act_dim, data['layer_sizes'])
        else:
            policy = Policy(obs_dim, act_dim, data.get('hidden', 64))
        policy.load_state_dict(data['policy_state'])
        # Reconstruct mutator
        mut_map = {
            'GaussianMutator': GaussianMutator,
            'ChunkMutator': ChunkMutator,
            'TransformerMutator': TransformerMutator,
            'ErrorCorrectorMutator': ErrorCorrectorMutator,
        }
        mut_cls = mut_map.get(data['mutator_type'], GaussianMutator)
        if mut_cls == GaussianMutator:
            mutator = mut_cls()
        else:
            n_params = sum(p.numel() for p in policy.parameters())
            mutator = mut_cls(n_params)
        mutator.load_state_dict(data['mutator_state'])
        genome = Genome(policy, mutator)
        genome.fitness = data.get('fitness', 0.0)
        return genome

    def to_bytes(self) -> bytes:
        """Serialize genome to bytes for multiprocessing."""
        buf = io.BytesIO()
        self.save(buf)
        return buf.getvalue()

    @staticmethod
    def load_bytes(data: bytes) -> 'Genome':
        """Deserialize genome from bytes."""
        buf = io.BytesIO(data)
        return Genome.load(buf)
