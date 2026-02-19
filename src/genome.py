"""
Genome: Policy + Mutator bundled together with TRUE self-replication.

The mutator processes the ENTIRE genome (policy + mutator weights) and outputs
a new full genome — including its own weights. This is true self-replication.

Stabilization: asymmetric mutation rates (0.1x for mutator weights vs 1x for policy).
Natural selection handles the rest — if a mutator destroys itself, it dies.
"""

import io
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import copy
import random
from enum import Enum

try:
    import lunar_torch_eval_rs as _rust_eval
except Exception:
    _rust_eval = None


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


class PolicyCNNLarge(nn.Module):
    """CNN policy for high-resolution image environments (e.g. CarRacing 96x96x3)."""

    def __init__(self, obs_shape: tuple, act_dim: int):
        super().__init__()
        self.obs_shape = obs_shape  # (H, W, C)
        self.obs_dim = int(np.prod(obs_shape))  # flattened, for compatibility
        self.act_dim = act_dim
        h, w, c = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 8, kernel_size=4, stride=2, padding=1),   # -> 48x48
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # -> 24x24
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), # -> 12x12
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=4, stride=2, padding=1),  # -> 6x6
            nn.ReLU(),
        )
        conv_out = 8 * (h // 16) * (w // 16)  # 8*6*6 = 288 for 96x96
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 32),
            nn.Tanh(),
            nn.Linear(32, act_dim),
            nn.Tanh(),  # bound outputs to [-1, 1] for continuous control
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h, w, c = self.obs_shape
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).float() / 255.0
        y = self.conv(x)
        y = y.reshape(b, -1)
        return self.fc(y)

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.FloatTensor(obs.flatten()).unsqueeze(0)
            action = self.forward(x).squeeze(0)
        return action.numpy()


class PolicyCNN(nn.Module):
    """Small CNN policy for Snake semantic grids (empty/snake/food flattened HWC)."""

    def __init__(self, obs_dim: int, act_dim: int, grid_size: int = 16, channels: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.grid_size = grid_size
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * grid_size * grid_size, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected flattened HWC [B, H*W*C]
        b = x.shape[0]
        x = x.view(b, self.grid_size, self.grid_size, self.channels).permute(0, 3, 1, 2)
        y = self.conv(x)
        y = y.reshape(b, -1)
        return self.fc(y)

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0)
            action = self.forward(x).squeeze(0)
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


class DualHeadCorrectorMutator(nn.Module):
    """Two-head corrector mutator: separate correction/exploration for policy vs meta (mutator+compat).

    Goal: keep replication (meta) stable while still allowing larger/more exploratory policy updates.

    Uses a shared encoder+corrector trunk, but two references + two scale/noise parameter sets.

    Expected weight_coords: 1D tensor where weight_coords[0] = n_policy (split index).
    """

    def __init__(self, chunk_size: int = 64, ref_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.chunk_size = chunk_size
        self.ref_dim = ref_dim

        self.reference_policy = nn.Parameter(torch.randn(ref_dim) * 0.1)
        self.reference_meta = nn.Parameter(torch.randn(ref_dim) * 0.1)

        self.encoder = nn.Sequential(
            nn.Linear(chunk_size, ref_dim),
            nn.Tanh(),
        )
        self.corrector = nn.Sequential(
            nn.Linear(ref_dim * 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, chunk_size),
            nn.Tanh(),
        )

        # Policy head explores more; meta head stays conservative.
        self.correction_scale_policy = nn.Parameter(torch.tensor(0.03))
        self.correction_scale_meta = nn.Parameter(torch.tensor(0.01))
        self.explore_scale_policy = nn.Parameter(torch.tensor(0.01))
        self.explore_scale_meta = nn.Parameter(torch.tensor(0.002))

    def _split_index(self, flat_weights: torch.Tensor, weight_coords: torch.Tensor | None) -> int:
        if weight_coords is None or weight_coords.numel() < 1:
            return int(flat_weights.shape[0] // 2)
        return int(weight_coords.view(-1)[0].item())

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None,
                      other_weights: torch.Tensor = None) -> torch.Tensor:
        n = flat_weights.shape[0]
        cs = self.chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len)])

        split = self._split_index(flat_weights, weight_coords)
        n_chunks = padded.shape[0] // cs
        starts = torch.arange(n_chunks) * cs
        is_policy_chunk = starts < split

        def correction_for(chunks: torch.Tensor, ref_vec: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            encoded = self.encoder(chunks)
            ref = ref_vec.unsqueeze(0).expand(encoded.shape[0], -1)
            diff = encoded - ref
            x = torch.cat([encoded, ref, diff], dim=1)
            corr = self.corrector(x) * scale
            return torch.clamp(corr, -0.1, 0.1)

        if other_weights is not None:
            other_pad = torch.cat([
                other_weights,
                torch.zeros(max(0, padded.shape[0] - other_weights.shape[0])),
            ])
            other_pad = other_pad[:padded.shape[0]]
            chunks_a = padded.view(-1, cs)
            chunks_b = other_pad.view(-1, cs)
            base = (chunks_a + chunks_b) / 2
        else:
            base = padded.view(-1, cs)

        corr = torch.zeros_like(base)
        if is_policy_chunk.any():
            corr[is_policy_chunk] = correction_for(
                base[is_policy_chunk], self.reference_policy, self.correction_scale_policy
            )
        if (~is_policy_chunk).any():
            corr[~is_policy_chunk] = correction_for(
                base[~is_policy_chunk], self.reference_meta, self.correction_scale_meta
            )

        noise = torch.zeros_like(base)
        if is_policy_chunk.any():
            noise[is_policy_chunk] = torch.randn_like(base[is_policy_chunk]) * self.explore_scale_policy
        if (~is_policy_chunk).any():
            noise[~is_policy_chunk] = torch.randn_like(base[~is_policy_chunk]) * self.explore_scale_meta

        mutated = base + corr + noise
        return mutated.view(-1)[:n]


class DualMixtureCorrectorMutator(DualHeadCorrectorMutator):
    """Dual-head corrector with an explicit Gaussian escape hatch on the *policy* slice.

    The meta slice (mutator+compat) remains conservative (corrector+small noise).

    Policy slice behavior:
    - Always applies corrector+noise
    - Additionally, with probability p_gauss_policy, adds Gaussian noise with scale gauss_scale_policy

    This keeps it a NN mutator, but prevents getting trapped in a fixed-point attractor.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        ref_dim: int = 16,
        hidden: int = 64,
        p_gauss_policy: float = 0.20,
        gauss_scale_policy: float = 0.03,
    ):
        super().__init__(chunk_size=chunk_size, ref_dim=ref_dim, hidden=hidden)
        # Store as buffers/tensors so they serialize nicely.
        self.p_gauss_policy = nn.Parameter(torch.tensor(float(p_gauss_policy)), requires_grad=False)
        self.gauss_scale_policy = nn.Parameter(torch.tensor(float(gauss_scale_policy)), requires_grad=False)

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None,
                      other_weights: torch.Tensor = None) -> torch.Tensor:
        # Start with the base dual-head corrector mutation
        mutated = super().mutate_genome(flat_weights, weight_coords=weight_coords, other_weights=other_weights)

        # Add Gaussian escape noise only on policy slice (and only sometimes)
        split = self._split_index(flat_weights, weight_coords)
        if split > 0 and float(self.p_gauss_policy.item()) > 0:
            if torch.rand(()) < self.p_gauss_policy:
                noise = torch.randn_like(mutated[:split]) * self.gauss_scale_policy
                mutated[:split] = mutated[:split] + noise
        return mutated


MUTATOR_REGISTRY = {
    'dualcorrector': DualHeadCorrectorMutator,
    'dualmixture': DualMixtureCorrectorMutator,
}

# Legacy keys that map to dualmixture for backward compatibility
_LEGACY_MUTATOR_KEYS = {'gaussian', 'chunk', 'transformer', 'corrector'}

MUTATOR_CLASS_TO_KEY = {cls.__name__: key for key, cls in MUTATOR_REGISTRY.items()}


def available_mutator_types() -> list[str]:
    return list(MUTATOR_REGISTRY.keys())


def create_mutator(mutator_type: str, **kwargs) -> nn.Module:
    # Map legacy mutator types to dualmixture
    if mutator_type in _LEGACY_MUTATOR_KEYS:
        mutator_type = 'dualmixture'

    if mutator_type not in MUTATOR_REGISTRY:
        raise ValueError(f"Unknown mutator type: {mutator_type}. "
                         f"Available: {list(MUTATOR_REGISTRY.keys())}")

    mut_cls = MUTATOR_REGISTRY[mutator_type]

    if mutator_type == 'dualmixture':
        ctor = {}
        if 'chunk_size' in kwargs:
            ctor['chunk_size'] = kwargs['chunk_size']
        if 'p_gauss_policy' in kwargs:
            ctor['p_gauss_policy'] = kwargs['p_gauss_policy']
        if 'gauss_scale_policy' in kwargs:
            ctor['gauss_scale_policy'] = kwargs['gauss_scale_policy']
        return mut_cls(**ctor)

    if 'chunk_size' in kwargs:
        return mut_cls(chunk_size=kwargs['chunk_size'])
    return mut_cls()


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
    COMPAT_RATE = 0.1  # compat net mutates slowly — prevents speciation fragmentation
    DEFAULT_STRUCTURAL_RATE = 0.05  # 5% base probability of structural mutation
    _RUST_WARNED = False

    @staticmethod
    def _rust_mutator_path() -> str | None:
        return os.environ.get('NM_RUST_MUTATOR_TS')

    @staticmethod
    def _rust_compat_path() -> str | None:
        return os.environ.get('NM_RUST_COMPAT_TS')

    @staticmethod
    def _rust_mutator_pair_path() -> str | None:
        return os.environ.get('NM_RUST_MUTATOR_PAIR_TS')

    def __init__(self, policy, mutator: nn.Module,
                 mutator_type: str = 'dualmixture',
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
        """Check compatibility via learned tags; optionally use Rust compat scorer if configured."""
        if self.compat_net is None or other.compat_net is None:
            return True

        rust_path = self._rust_compat_path()
        if _rust_eval is not None and rust_path:
            try:
                my_flat = self.get_flat_weights().detach().cpu().float().tolist()
                other_flat = other.get_flat_weights().detach().cpu().float().tolist()
                score = float(_rust_eval.compat_score(rust_path, my_flat, other_flat))
                return score > threshold
            except Exception as e:
                if not Genome._RUST_WARNED:
                    print(f"[rust] compat fallback to python: {e}")
                    Genome._RUST_WARNED = True

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

        # Extract policy size (split index for dual-head mutators)
        parent_n_policy = self.num_policy_params()

        # Feed parent's full genome through parent's mutator (optionally Rust TorchScript)
        rust_mut_path = self._rust_mutator_path()
        if _rust_eval is not None and rust_mut_path:
            try:
                out = _rust_eval.mutator_mutate(
                    rust_mut_path,
                    parent_full.detach().cpu().float().tolist(),
                    int(parent_n_policy),
                    None,
                )
                mutated_full = torch.tensor(out, dtype=parent_full.dtype)
            except Exception as e:
                if not Genome._RUST_WARNED:
                    print(f"[rust] mutator fallback to python: {e}")
                    Genome._RUST_WARNED = True
                mutated_full = self.mutator.mutate_genome(
                    parent_full,
                    weight_coords=torch.tensor([parent_n_policy]),
                )
        else:
            mutated_full = self.mutator.mutate_genome(
                parent_full,
                weight_coords=torch.tensor([parent_n_policy]),
            )
        full_delta = mutated_full - parent_full

        # Extract policy delta — may need to resize if architecture changed
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
            # Use mutator-generated delta for compat weights, scaled by COMPAT_RATE
            compat_delta_raw = meta_delta[n_mutator:n_mutator + n_compat] if meta_delta.shape[0] > n_mutator else torch.zeros(n_compat)
            compat_delta_raw = compat_delta_raw[:n_compat] if compat_delta_raw.shape[0] >= n_compat else torch.cat([compat_delta_raw, torch.zeros(n_compat - compat_delta_raw.shape[0])])
            compat_delta = compat_delta_raw * decay * self.COMPAT_RATE
            meta_delta_final = torch.cat([mutator_delta, compat_delta])
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
        """Crossover where the mutator sees BOTH parents and decides how to combine them.
        
        The mutator receives both parents' full weight vectors and outputs
        a child genome. This lets the mutator learn crossover strategies
        rather than using fixed rules.
        
        For FlexiblePolicy: uses the fitter parent's architecture (self is assumed fitter).
        """
        # Use self's (fitter parent) architecture for the child
        child_policy = self._clone_policy()
        n_policy = sum(p.numel() for p in child_policy.parameters())
        
        # Get both parents' full flat weights
        my_flat = self.get_flat_weights()
        other_flat = other.get_flat_weights()
        
        # Pad to same size (flex genomes may differ)
        max_len = max(my_flat.shape[0], other_flat.shape[0])
        if my_flat.shape[0] < max_len:
            my_flat = torch.cat([my_flat, torch.zeros(max_len - my_flat.shape[0])])
        if other_flat.shape[0] < max_len:
            other_flat = torch.cat([other_flat, torch.zeros(max_len - other_flat.shape[0])])
        
        decay = max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))
        
        # Let the fitter parent's mutator see both genomes and decide
        rust_pair_path = self._rust_mutator_pair_path()
        if _rust_eval is not None and rust_pair_path:
            try:
                out = _rust_eval.mutator_crossover(
                    rust_pair_path,
                    my_flat.detach().cpu().float().tolist(),
                    other_flat.detach().cpu().float().tolist(),
                    int(n_policy),
                    None,
                )
                mutated_full = torch.tensor(out, dtype=my_flat.dtype)
            except Exception as e:
                if not Genome._RUST_WARNED:
                    print(f"[rust] crossover fallback to python: {e}")
                    Genome._RUST_WARNED = True
                mutated_full = self.mutator.mutate_genome(my_flat, weight_coords=torch.tensor([n_policy]), other_weights=other_flat)
        else:
            mutated_full = self.mutator.mutate_genome(my_flat, weight_coords=torch.tensor([n_policy]), other_weights=other_flat)
        
        # Extract child weights from mutator output
        n_mutator = self.num_mutator_params()
        n_compat = self.num_compat_params()
        
        # Policy weights from mutator output
        child_policy_weights = mutated_full[:n_policy]
        
        # Apply decay to the delta from self's weights
        my_policy = self.get_flat_policy_weights()
        if my_policy.shape[0] != n_policy:
            resized = torch.zeros(n_policy)
            copy_len = min(my_policy.shape[0], n_policy)
            resized[:copy_len] = my_policy[:copy_len]
            my_policy = resized
        policy_delta = (child_policy_weights - my_policy) * decay
        new_policy = my_policy + policy_delta
        
        # Mutator weights: blend from both parents with asymmetric rate
        my_mutator = self.get_flat_mutator_weights()
        other_mutator = other.get_flat_mutator_weights()
        # Resize if needed
        n_mut = min(my_mutator.shape[0], other_mutator.shape[0])
        mutator_blend = (my_mutator[:n_mut] + other_mutator[:n_mut]) / 2
        mutator_noise = torch.randn(n_mut) * 0.02 * decay * self.MUTATOR_SELF_RATE
        new_mutator = mutator_blend + mutator_noise
        if n_mut < n_mutator:
            new_mutator = torch.cat([new_mutator, my_mutator[n_mut:]])
        
        mutator_delta_norm = torch.norm(new_mutator - my_mutator[:new_mutator.shape[0]]).item()
        
        # Compat weights
        if n_compat > 0:
            my_compat = self.get_flat_compat_weights()
            other_compat = other.get_flat_compat_weights()
            n_c = min(my_compat.shape[0], other_compat.shape[0])
            new_compat = (my_compat[:n_c] + other_compat[:n_c]) / 2
            compat_noise = torch.randn(n_c) * 0.05 * decay
            new_compat = new_compat + compat_noise
            if n_c < n_compat:
                new_compat = torch.cat([new_compat, my_compat[n_c:]])
        else:
            new_compat = torch.tensor([])
        
        new_flat = torch.cat([new_policy, new_mutator, new_compat])
        new_flat = torch.clamp(new_flat, -5.0, 5.0)
        
        fidelity = torch.norm(mutator_noise).item()
        
        child = Genome(
            policy=child_policy,
            mutator=self._clone_mutator(),
            mutator_type=self.mutator_type,
            compat_net=self._clone_compat() if self.compat_net else None,
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
        if isinstance(self.policy, PolicyCNNLarge):
            return PolicyCNNLarge(self.policy.obs_shape, self.policy.act_dim)
        if isinstance(self.policy, PolicyCNN):
            return PolicyCNN(self.policy.obs_dim, self.policy.act_dim,
                             grid_size=self.policy.grid_size,
                             channels=self.policy.channels)
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
            'mutator_type': self.mutator_type,
            'mutator_class': type(self.mutator).__name__,
            'fitness': self.fitness,
            'policy_arch': 'mlp',
        }
        if isinstance(self.policy, FlexiblePolicy):
            data['policy_arch'] = 'flex'
            data['flex'] = True
            data['layer_sizes'] = list(self.policy.layer_sizes)
        elif isinstance(self.policy, PolicyCNNLarge):
            data['policy_arch'] = 'cnn-large'
            data['flex'] = False
            data['obs_shape'] = self.policy.obs_shape
        elif isinstance(self.policy, PolicyCNN):
            data['policy_arch'] = 'cnn'
            data['flex'] = False
            data['grid_size'] = self.policy.grid_size
            data['channels'] = self.policy.channels
        else:
            data['flex'] = False
            data['hidden'] = self.policy.net[0].out_features
        if self.compat_net is not None:
            data['compat_state'] = self.compat_net.state_dict()
            data['compat_genome_dim'] = self.compat_net.genome_dim
        torch.save(data, path)

    @staticmethod
    def load(path) -> 'Genome':
        """Load genome from disk or file-like object."""
        data = torch.load(path, weights_only=False)
        obs_dim, act_dim = data['obs_dim'], data['act_dim']
        arch = data.get('policy_arch')
        if arch == 'cnn-large':
            policy = PolicyCNNLarge(tuple(data['obs_shape']), act_dim)
        elif arch == 'cnn':
            policy = PolicyCNN(obs_dim, act_dim,
                               grid_size=data.get('grid_size', 16),
                               channels=data.get('channels', 3))
        elif data.get('flex') or arch == 'flex':
            policy = FlexiblePolicy(obs_dim, act_dim, data['layer_sizes'])
        else:
            policy = Policy(obs_dim, act_dim, data.get('hidden', 64))
        policy.load_state_dict(data['policy_state'])
        # Reconstruct mutator from registry (supports legacy class-name saves)
        saved_type = data.get('mutator_type', 'dualmixture')
        mutator_key = MUTATOR_CLASS_TO_KEY.get(saved_type, saved_type.lower())
        if mutator_key not in MUTATOR_REGISTRY:
            # Legacy mutator type — fall back to dualmixture (cannot load old weights)
            mutator_key = 'dualmixture'
        mutator = create_mutator(mutator_key)
        mutator.load_state_dict(data['mutator_state'])
        compat_net = None
        if 'compat_state' in data:
            compat_net = CompatibilityNet(data['compat_genome_dim'])
            compat_net.load_state_dict(data['compat_state'])
        genome = Genome(policy, mutator, mutator_type=mutator_key, compat_net=compat_net)
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
