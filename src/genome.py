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
import torch.nn.functional as F
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

    def __init__(
        self,
        chunk_size: int = 64,
        ref_dim: int = 16,
        hidden: int = 64,
        crossover_gated_blend: bool = True,
        crossover_gate_mode: str = 'sigmoid',
        crossover_gumbel_tau: float = 1.0,
        crossover_gumbel_hard: bool = False,
        crossover_gate_clamp: float = 0.0,
    ):
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
        self.gate_head = nn.Sequential(
            nn.Linear(ref_dim * 4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

        # Policy head explores more; meta head stays conservative.
        self.correction_scale_policy = nn.Parameter(torch.tensor(0.03))
        self.correction_scale_meta = nn.Parameter(torch.tensor(0.01))
        self.explore_scale_policy = nn.Parameter(torch.tensor(0.01))
        self.explore_scale_meta = nn.Parameter(torch.tensor(0.002))
        self.register_buffer(
            'crossover_gated_blend',
            torch.tensor(1.0 if bool(crossover_gated_blend) else 0.0),
        )
        gate_mode = str(crossover_gate_mode).lower().strip()
        if gate_mode not in {'sigmoid', 'gumbel'}:
            gate_mode = 'sigmoid'
        self.register_buffer(
            'crossover_gate_mode_id',
            torch.tensor(1.0 if gate_mode == 'gumbel' else 0.0),
        )
        self.register_buffer('crossover_gumbel_tau', torch.tensor(float(crossover_gumbel_tau)))
        self.register_buffer(
            'crossover_gumbel_hard',
            torch.tensor(1.0 if bool(crossover_gumbel_hard) else 0.0),
        )
        self.register_buffer('crossover_gate_clamp', torch.tensor(float(crossover_gate_clamp)))
        self.last_crossover_gate_stats: dict | None = None

    def _split_index(self, flat_weights: torch.Tensor, weight_coords: torch.Tensor | None) -> int:
        if weight_coords is None or weight_coords.numel() < 1:
            return int(flat_weights.shape[0] // 2)
        return int(weight_coords.view(-1)[0].item())

    def _crossover_base_chunks(
        self,
        chunks_a: torch.Tensor,
        chunks_b: torch.Tensor,
    ) -> torch.Tensor:
        if float(self.crossover_gated_blend.item()) <= 0.5:
            self.last_crossover_gate_stats = {
                'enabled': False,
                'mode': 'average',
                'alpha_mean': 0.5,
                'alpha_std': 0.0,
            }
            return (chunks_a + chunks_b) / 2

        encoded_a = self.encoder(chunks_a)
        encoded_b = self.encoder(chunks_b)
        gate_in = torch.cat([encoded_a, encoded_b, encoded_a - encoded_b, encoded_a * encoded_b], dim=1)
        logits = self.gate_head(gate_in)

        if float(self.crossover_gate_mode_id.item()) >= 0.5:
            tau = max(float(self.crossover_gumbel_tau.item()), 1e-6)
            hard = float(self.crossover_gumbel_hard.item()) >= 0.5
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
            alpha = probs[:, 0]
            mode = 'gumbel'
        else:
            alpha = torch.sigmoid(logits[:, 0] - logits[:, 1])
            eps = float(self.crossover_gate_clamp.item())
            if eps > 0.0:
                eps = min(max(eps, 0.0), 0.499999)
                alpha = torch.clamp(alpha, eps, 1.0 - eps)
            mode = 'sigmoid'

        self.last_crossover_gate_stats = {
            'enabled': True,
            'mode': mode,
            'alpha_mean': float(alpha.mean().item()),
            'alpha_std': float(alpha.std(unbiased=False).item()),
        }
        return alpha.unsqueeze(1) * chunks_a + (1.0 - alpha).unsqueeze(1) * chunks_b

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None,
                      other_weights: torch.Tensor = None) -> torch.Tensor:
        n = flat_weights.shape[0]
        cs = self.chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len, device=flat_weights.device, dtype=flat_weights.dtype)])

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
                other_weights.to(flat_weights.device, dtype=flat_weights.dtype),
                torch.zeros(max(0, padded.shape[0] - other_weights.shape[0]), device=flat_weights.device, dtype=flat_weights.dtype),
            ])
            other_pad = other_pad[:padded.shape[0]]
            chunks_a = padded.view(-1, cs)
            chunks_b = other_pad.view(-1, cs)
            base = self._crossover_base_chunks(chunks_a, chunks_b)
        else:
            self.last_crossover_gate_stats = None
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
        crossover_gated_blend: bool = True,
        crossover_gate_mode: str = 'sigmoid',
        crossover_gumbel_tau: float = 1.0,
        crossover_gumbel_hard: bool = False,
        crossover_gate_clamp: float = 0.0,
        p_gauss_policy: float = 0.20,
        gauss_scale_policy: float = 0.03,
    ):
        super().__init__(
            chunk_size=chunk_size,
            ref_dim=ref_dim,
            hidden=hidden,
            crossover_gated_blend=crossover_gated_blend,
            crossover_gate_mode=crossover_gate_mode,
            crossover_gumbel_tau=crossover_gumbel_tau,
            crossover_gumbel_hard=crossover_gumbel_hard,
            crossover_gate_clamp=crossover_gate_clamp,
        )
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


class DualMixtureV2Mutator(nn.Module):
    """DualMixture v2 with group-wise low-rank policy deltas and conservative meta mutation.

    Policy mutation is structured:
    - Chunk encoder builds per-chunk features.
    - Policy chunks are partitioned into groups from policy boundaries when available.
    - Each group yields a low-rank basis; chunk coefficients project through this basis.

    Meta mutation remains conservative (corrector + low noise) to preserve self-replication.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        ref_dim: int = 16,
        hidden: int = 64,
        lowrank_rank: int = 4,
        max_policy_groups: int = 8,
        policy_corr_scale: float = 0.025,
        policy_noise_scale: float = 0.008,
        meta_corr_scale: float = 0.01,
        meta_noise_scale: float = 0.002,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.ref_dim = int(ref_dim)
        self.lowrank_rank = int(lowrank_rank)
        self.max_policy_groups = int(max(1, max_policy_groups))

        self.reference_policy = nn.Parameter(torch.randn(ref_dim) * 0.1)
        self.reference_meta = nn.Parameter(torch.randn(ref_dim) * 0.1)

        self.encoder = nn.Sequential(
            nn.Linear(self.chunk_size, self.ref_dim),
            nn.Tanh(),
        )
        self.policy_corrector = nn.Sequential(
            nn.Linear(self.ref_dim * 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.chunk_size),
            nn.Tanh(),
        )
        self.meta_corrector = nn.Sequential(
            nn.Linear(self.ref_dim * 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.chunk_size),
            nn.Tanh(),
        )
        self.coeff_head = nn.Sequential(
            nn.Linear(self.ref_dim * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.lowrank_rank),
        )
        self.basis_head = nn.Sequential(
            nn.Linear(self.ref_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.lowrank_rank * self.chunk_size),
        )

        self.policy_corr_scale = nn.Parameter(torch.tensor(float(policy_corr_scale)))
        self.policy_noise_scale = nn.Parameter(torch.tensor(float(policy_noise_scale)))
        self.meta_corr_scale = nn.Parameter(torch.tensor(float(meta_corr_scale)))
        self.meta_noise_scale = nn.Parameter(torch.tensor(float(meta_noise_scale)))
        self.group_log_scales = nn.Parameter(torch.zeros(self.max_policy_groups))

    def _split_index(self, flat_weights: torch.Tensor, weight_coords: torch.Tensor | None) -> int:
        if weight_coords is None or weight_coords.numel() < 1:
            return int(flat_weights.shape[0] // 2)
        return int(weight_coords.view(-1)[0].item())

    def _policy_boundaries(self, split: int, weight_coords: torch.Tensor | None) -> List[int]:
        if weight_coords is None or weight_coords.numel() <= 1:
            return []
        boundaries = []
        for b in weight_coords.view(-1)[1:].tolist():
            bi = int(b)
            if 0 < bi < split:
                boundaries.append(bi)
        boundaries = sorted(set(boundaries))
        if len(boundaries) > self.max_policy_groups - 1:
            idx = torch.linspace(0, len(boundaries) - 1, steps=self.max_policy_groups - 1).long().tolist()
            boundaries = [boundaries[i] for i in idx]
        return boundaries

    def _chunk_groups(self, policy_chunk_starts: torch.Tensor, boundaries: List[int]) -> torch.Tensor:
        n_policy_chunks = policy_chunk_starts.shape[0]
        if n_policy_chunks == 0:
            return torch.zeros(0, dtype=torch.long)

        if boundaries:
            b = torch.tensor(boundaries, dtype=policy_chunk_starts.dtype, device=policy_chunk_starts.device)
            # bucketize gives group id in [0, len(boundaries)]
            groups = torch.bucketize(policy_chunk_starts, b)
            return groups.to(torch.long)

        # Fallback: coarse groups by index range
        g = min(3, self.max_policy_groups, max(1, n_policy_chunks))
        pos = torch.arange(n_policy_chunks, device=policy_chunk_starts.device)
        groups = (pos * g) // max(1, n_policy_chunks)
        groups = torch.clamp(groups, 0, g - 1)
        return groups.to(torch.long)

    def mutate_genome(
        self,
        flat_weights: torch.Tensor,
        weight_coords: torch.Tensor = None,
        other_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        n = flat_weights.shape[0]
        cs = self.chunk_size
        pad_len = (cs - n % cs) % cs
        padded = torch.cat([flat_weights, torch.zeros(pad_len, device=flat_weights.device, dtype=flat_weights.dtype)])

        split = self._split_index(flat_weights, weight_coords)
        n_chunks = padded.shape[0] // cs
        starts = torch.arange(n_chunks, device=flat_weights.device) * cs
        is_policy_chunk = starts < split

        if other_weights is not None:
            other_pad = torch.cat([
                other_weights.to(flat_weights.device, dtype=flat_weights.dtype),
                torch.zeros(max(0, padded.shape[0] - other_weights.shape[0]), device=flat_weights.device, dtype=flat_weights.dtype),
            ])[:padded.shape[0]]
            base = (padded.view(-1, cs) + other_pad.view(-1, cs)) / 2
        else:
            base = padded.view(-1, cs)

        encoded = self.encoder(base)
        mutated = base.clone()

        if is_policy_chunk.any():
            p_idx = torch.where(is_policy_chunk)[0]
            policy_encoded = encoded[p_idx]
            policy_starts = starts[p_idx]
            boundaries = self._policy_boundaries(split, weight_coords)
            groups = self._chunk_groups(policy_starts, boundaries)
            n_groups = int(groups.max().item()) + 1
            n_groups = min(n_groups, self.max_policy_groups)
            groups = torch.clamp(groups, 0, n_groups - 1)

            group_ctx = torch.zeros(n_groups, self.ref_dim, device=flat_weights.device, dtype=flat_weights.dtype)
            for g in range(n_groups):
                mask = groups == g
                if mask.any():
                    group_ctx[g] = policy_encoded[mask].mean(dim=0)
                else:
                    group_ctx[g] = self.reference_policy

            local_ref = self.reference_policy.unsqueeze(0).expand(policy_encoded.shape[0], -1)
            local_diff = policy_encoded - local_ref
            corr_in = torch.cat([policy_encoded, local_ref, local_diff], dim=1)
            local_corr = self.policy_corrector(corr_in)

            chunk_group_ctx = group_ctx[groups]
            coeff = self.coeff_head(torch.cat([policy_encoded, chunk_group_ctx], dim=1))
            basis = self.basis_head(chunk_group_ctx).view(-1, self.lowrank_rank, cs)
            lowrank = torch.einsum('br,brc->bc', coeff, basis)

            group_scale = torch.exp(self.group_log_scales[:n_groups])[groups].unsqueeze(1)
            policy_noise = torch.randn_like(lowrank) * (torch.abs(self.policy_noise_scale) * group_scale)
            policy_delta = torch.clamp((local_corr * self.policy_corr_scale + lowrank) * group_scale + policy_noise, -0.2, 0.2)
            mutated[p_idx] = mutated[p_idx] + policy_delta

        if (~is_policy_chunk).any():
            m_idx = torch.where(~is_policy_chunk)[0]
            meta_encoded = encoded[m_idx]
            meta_ref = self.reference_meta.unsqueeze(0).expand(meta_encoded.shape[0], -1)
            meta_diff = meta_encoded - meta_ref
            meta_corr = self.meta_corrector(torch.cat([meta_encoded, meta_ref, meta_diff], dim=1))
            meta_noise = torch.randn_like(meta_corr) * torch.abs(self.meta_noise_scale)
            meta_delta = torch.clamp(meta_corr * self.meta_corr_scale + meta_noise, -0.05, 0.05)
            mutated[m_idx] = mutated[m_idx] + meta_delta

        return mutated.view(-1)[:n]


class GlobalLowRankMutator(nn.Module):
    """Global pooled low-rank mutator over fixed-size blocks (no chunk grouping semantics)."""

    def __init__(
        self,
        block_size: int = 64,
        ref_dim: int = 16,
        hidden: int = 64,
        rank: int = 4,
        policy_scale: float = 0.02,
        meta_scale: float = 0.006,
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.ref_dim = int(ref_dim)
        self.rank = int(rank)

        self.reference = nn.Parameter(torch.randn(self.ref_dim) * 0.1)
        self.encoder = nn.Sequential(
            nn.Linear(self.block_size, self.ref_dim),
            nn.Tanh(),
        )
        self.context = nn.Sequential(
            nn.Linear(self.ref_dim * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.ref_dim),
            nn.Tanh(),
        )
        self.coeff_head = nn.Sequential(
            nn.Linear(self.ref_dim * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.rank),
        )
        self.basis_head = nn.Sequential(
            nn.Linear(self.ref_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.rank * self.block_size),
        )
        self.policy_scale = nn.Parameter(torch.tensor(float(policy_scale)))
        self.meta_scale = nn.Parameter(torch.tensor(float(meta_scale)))

    def _split_index(self, flat_weights: torch.Tensor, weight_coords: torch.Tensor | None) -> int:
        if weight_coords is None or weight_coords.numel() < 1:
            return int(flat_weights.shape[0] // 2)
        return int(weight_coords.view(-1)[0].item())

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None,
                      other_weights: torch.Tensor = None) -> torch.Tensor:
        n = flat_weights.shape[0]
        bs = self.block_size
        pad_len = (bs - n % bs) % bs
        padded = torch.cat([flat_weights, torch.zeros(pad_len, device=flat_weights.device, dtype=flat_weights.dtype)])

        if other_weights is not None:
            other_pad = torch.cat([
                other_weights.to(flat_weights.device, dtype=flat_weights.dtype),
                torch.zeros(max(0, padded.shape[0] - other_weights.shape[0]), device=flat_weights.device, dtype=flat_weights.dtype),
            ])[:padded.shape[0]]
            blocks = (padded.view(-1, bs) + other_pad.view(-1, bs)) / 2
        else:
            blocks = padded.view(-1, bs)

        encoded = self.encoder(blocks)
        pooled = encoded.mean(dim=0, keepdim=True)
        global_ctx = self.context(torch.cat([pooled, self.reference.unsqueeze(0)], dim=1))
        ctx = global_ctx.expand(encoded.shape[0], -1)

        coeff = self.coeff_head(torch.cat([encoded, ctx], dim=1))
        basis = self.basis_head(ctx).view(-1, self.rank, bs)
        lowrank_delta = torch.einsum('br,brc->bc', coeff, basis)

        split = self._split_index(flat_weights, weight_coords)
        starts = torch.arange(blocks.shape[0], device=flat_weights.device) * bs
        is_policy = starts < split
        scales = torch.where(is_policy, torch.abs(self.policy_scale), torch.abs(self.meta_scale)).unsqueeze(1)

        noise = torch.randn_like(blocks) * scales * 0.2
        delta = torch.clamp(lowrank_delta * scales + noise, -0.15, 0.15)
        mutated = blocks + delta
        return mutated.view(-1)[:n]


class PerceiverLiteMutator(nn.Module):
    """Per-block cross-attention to latent array, then decode low-amplitude deltas."""

    def __init__(
        self,
        block_size: int = 64,
        ref_dim: int = 16,
        hidden: int = 64,
        latent_count: int = 8,
        policy_scale: float = 0.018,
        meta_scale: float = 0.004,
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.ref_dim = int(ref_dim)
        self.latent_count = int(latent_count)

        self.encoder = nn.Sequential(
            nn.Linear(self.block_size, self.ref_dim),
            nn.Tanh(),
        )
        self.latents = nn.Parameter(torch.randn(self.latent_count, self.ref_dim) * 0.1)
        self.to_query = nn.Linear(self.ref_dim, self.ref_dim)
        self.to_key = nn.Linear(self.ref_dim, self.ref_dim)
        self.to_value = nn.Linear(self.ref_dim, self.ref_dim)
        self.decode = nn.Sequential(
            nn.Linear(self.ref_dim * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.block_size),
            nn.Tanh(),
        )
        self.policy_scale = nn.Parameter(torch.tensor(float(policy_scale)))
        self.meta_scale = nn.Parameter(torch.tensor(float(meta_scale)))

    def _split_index(self, flat_weights: torch.Tensor, weight_coords: torch.Tensor | None) -> int:
        if weight_coords is None or weight_coords.numel() < 1:
            return int(flat_weights.shape[0] // 2)
        return int(weight_coords.view(-1)[0].item())

    def mutate_genome(self, flat_weights: torch.Tensor,
                      weight_coords: torch.Tensor = None,
                      other_weights: torch.Tensor = None) -> torch.Tensor:
        n = flat_weights.shape[0]
        bs = self.block_size
        pad_len = (bs - n % bs) % bs
        padded = torch.cat([flat_weights, torch.zeros(pad_len, device=flat_weights.device, dtype=flat_weights.dtype)])

        if other_weights is not None:
            other_pad = torch.cat([
                other_weights.to(flat_weights.device, dtype=flat_weights.dtype),
                torch.zeros(max(0, padded.shape[0] - other_weights.shape[0]), device=flat_weights.device, dtype=flat_weights.dtype),
            ])[:padded.shape[0]]
            blocks = (padded.view(-1, bs) + other_pad.view(-1, bs)) / 2
        else:
            blocks = padded.view(-1, bs)

        token = self.encoder(blocks)
        q = self.to_query(token)
        k = self.to_key(self.latents)
        v = self.to_value(self.latents)
        attn = torch.softmax((q @ k.T) / (self.ref_dim ** 0.5), dim=1)
        context = attn @ v
        delta = self.decode(torch.cat([token, context], dim=1))

        split = self._split_index(flat_weights, weight_coords)
        starts = torch.arange(blocks.shape[0], device=flat_weights.device) * bs
        is_policy = starts < split
        scales = torch.where(is_policy, torch.abs(self.policy_scale), torch.abs(self.meta_scale)).unsqueeze(1)
        noise = torch.randn_like(delta) * scales * 0.15
        block_delta = torch.clamp(delta * scales + noise, -0.12, 0.12)

        mutated = blocks + block_delta
        return mutated.view(-1)[:n]


MUTATOR_REGISTRY = {
    'dualcorrector': DualHeadCorrectorMutator,
    'dualmixture': DualMixtureCorrectorMutator,
    'dualmixture_v2': DualMixtureV2Mutator,
    'global_lowrank': GlobalLowRankMutator,
    'perceiver_lite': PerceiverLiteMutator,
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

    if mutator_type == 'dualcorrector':
        ctor = {}
        for key in (
            'chunk_size',
            'ref_dim',
            'hidden',
            'crossover_gated_blend',
            'crossover_gate_mode',
            'crossover_gumbel_tau',
            'crossover_gumbel_hard',
            'crossover_gate_clamp',
        ):
            if key in kwargs:
                ctor[key] = kwargs[key]
        return mut_cls(**ctor)

    if mutator_type == 'dualmixture':
        ctor = {}
        for key in (
            'chunk_size',
            'ref_dim',
            'hidden',
            'crossover_gated_blend',
            'crossover_gate_mode',
            'crossover_gumbel_tau',
            'crossover_gumbel_hard',
            'crossover_gate_clamp',
            'p_gauss_policy',
            'gauss_scale_policy',
        ):
            if key in kwargs:
                ctor[key] = kwargs[key]
        return mut_cls(**ctor)

    if mutator_type == 'dualmixture_v2':
        ctor = {}
        for key in (
            'chunk_size',
            'ref_dim',
            'hidden',
            'lowrank_rank',
            'max_policy_groups',
            'policy_corr_scale',
            'policy_noise_scale',
            'meta_corr_scale',
            'meta_noise_scale',
        ):
            if key in kwargs:
                ctor[key] = kwargs[key]
        return mut_cls(**ctor)

    if mutator_type == 'global_lowrank':
        ctor = {}
        for key in ('block_size', 'ref_dim', 'hidden', 'rank', 'policy_scale', 'meta_scale'):
            if key in kwargs:
                ctor[key] = kwargs[key]
        return mut_cls(**ctor)

    if mutator_type == 'perceiver_lite':
        ctor = {}
        for key in ('block_size', 'ref_dim', 'hidden', 'latent_count', 'policy_scale', 'meta_scale'):
            if key in kwargs:
                ctor[key] = kwargs[key]
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
        # Scores compatibility using symmetric pair features for stricter,
        # order-invariant pairwise classification.
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
    
    def _pair_features(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Symmetric pair features for order-invariant compatibility scoring."""
        return torch.cat([torch.abs(a - b), a * b])

    def score(self, my_embed: torch.Tensor, other_embed: torch.Tensor) -> float:
        """Score compatibility using learned scorer (symmetric features)."""
        combined = self._pair_features(my_embed, other_embed)
        return self.scorer(combined).item()

    def pretrain(self, genomes_flat: list, steps: int = 300, lr: float = 0.01,
                 mode: str = 'contrastive'):
        """
        Pre-train compatibility model.

        modes:
          - contrastive (default): preserve distance structure in embedding space
          - binary: train scorer as binary classifier (compatible vs incompatible)
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
        mode = str(mode).lower().strip()

        if mode == 'binary':
            # Balanced binary pretraining with margin bands:
            # - positive pairs: close in weight space (<= low quantile)
            # - negative pairs: far in weight space (>= high quantile)
            # - ignore ambiguous middle to sharpen classifier boundary
            bce = nn.BCELoss()
            dvals = np.array([d for _, _, d in distances], dtype=np.float64)
            q_low = float(np.quantile(dvals, 0.35))
            q_high = float(np.quantile(dvals, 0.65))

            pos_pairs = [(i, j, d) for (i, j, d) in distances if d <= q_low]
            neg_pairs = [(i, j, d) for (i, j, d) in distances if d >= q_high]

            # Fallback if quantile split degenerates on tiny/flat populations
            if not pos_pairs or not neg_pairs:
                pos_pairs = [(i, j, d) for (i, j, d) in distances if d < median_dist]
                neg_pairs = [(i, j, d) for (i, j, d) in distances if d >= median_dist]

            # Hard negatives: bias toward negatives closer to the decision margin.
            neg_pairs_sorted = sorted(neg_pairs, key=lambda t: t[2])
            hard_neg_pool = neg_pairs_sorted[:max(1, len(neg_pairs_sorted) // 2)]

            for _ in range(steps):
                use_pos = bool(np.random.rand() < 0.5)
                if use_pos and pos_pairs:
                    i, j, _ = pos_pairs[np.random.randint(len(pos_pairs))]
                    target_val = 1.0
                else:
                    src = hard_neg_pool if hard_neg_pool else neg_pairs
                    i, j, _ = src[np.random.randint(len(src))]
                    target_val = 0.0

                ei = self.embed(padded[i])
                ej = self.embed(padded[j])
                pred = self.scorer(self._pair_features(ei, ej))
                target = torch.tensor([target_val], dtype=pred.dtype)
                loss = bce(pred.view(-1), target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return

        # contrastive mode (legacy/default)
        for _ in range(steps):
            idx = np.random.randint(len(distances))
            i, j, genome_dist = distances[idx]

            ei = self.embed(padded[i])
            ej = self.embed(padded[j])
            embed_dist = torch.norm(ei - ej)

            margin = 2.0
            if genome_dist < median_dist:
                loss = embed_dist ** 2  # pull together
            else:
                loss = torch.clamp(margin - embed_dist, min=0) ** 2  # push apart

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _mutator_core_named_parameters(mutator: nn.Module):
    """Yield mutator parameters excluding the optional mutator-owned compat head."""
    for name, param in mutator.named_parameters():
        if name.startswith('compat_head.'):
            continue
        yield name, param


def _mutator_core_param_count(mutator: nn.Module) -> int:
    return sum(p.numel() for _, p in _mutator_core_named_parameters(mutator))


def _mutator_core_flat_weights(mutator: nn.Module) -> torch.Tensor:
    parts = [p.data.view(-1) for _, p in _mutator_core_named_parameters(mutator)]
    if not parts:
        return torch.tensor([])
    return torch.cat(parts)


def _attach_mutator_compat_head(mutator: nn.Module, core_dim: int | None = None) -> CompatibilityNet | None:
    """Ensure mutator has a mutator-owned compatibility head module."""
    head = getattr(mutator, 'compat_head', None)
    if isinstance(head, CompatibilityNet):
        return head
    dim = int(core_dim) if core_dim is not None else int(_mutator_core_param_count(mutator))
    if dim <= 0:
        return None
    mutator.compat_head = CompatibilityNet(dim)
    return mutator.compat_head


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
    # Keep threshold evolvable but stable (drift-neutral log-normal step)
    COMPAT_THRESHOLD_MUTATION_STD_REPRO = 0.10
    COMPAT_THRESHOLD_MUTATION_STD_CROSS = 0.08
    DEFAULT_STRUCTURAL_RATE = 0.05  # 5% base probability of structural mutation

    def __init__(self, policy, mutator: nn.Module,
                 mutator_type: str = 'dualmixture',
                 compat_net: 'CompatibilityNet | None' = None,
                 compat_threshold: float = 0.5,
                 evolve_compat_threshold: bool = False,
                 compat_mode: str = 'distance'):
        self.policy = policy
        self.mutator = mutator
        self.mutator_type = mutator_type
        self.compat_net = compat_net  # None = no speciation (backward compat)
        self.compat_threshold = float(compat_threshold)
        self.evolve_compat_threshold = bool(evolve_compat_threshold)
        self.compat_mode = str(compat_mode)
        self.fitness = 0.0
        self.self_replication_fidelity = 0.0  # how well mutator preserves itself
        self.mutator_delta_norm = 0.0  # magnitude of mutator weight change
        self.crossover_gate_alpha_mean = 0.0
        self.crossover_gate_alpha_std = 0.0
        self.species_id = -1  # assigned during speciation
        # Learned structural mutation rate (only meaningful for FlexiblePolicy)
        self.structural_rate = self.DEFAULT_STRUCTURAL_RATE
        self.last_structural_mutation = None  # track what happened
        if self.compat_mode == 'mutator_pair_head':
            _attach_mutator_compat_head(self.mutator)

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

    def get_flat_mutator_core_weights(self) -> torch.Tensor:
        """Mutator weights excluding optional mutator-owned compat head params."""
        return _mutator_core_flat_weights(self.mutator)

    def ensure_mutator_pair_head(self) -> CompatibilityNet | None:
        return _attach_mutator_compat_head(self.mutator)

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
        """Get learned species tag — embedding source depends on compat mode."""
        with torch.no_grad():
            mode = str(getattr(self, 'compat_mode', 'distance'))
            if mode == 'mutator_pair_head':
                head = _attach_mutator_compat_head(self.mutator)
                if head is None:
                    return torch.zeros(CompatibilityNet.EMBED_DIM)
                return head.embed(self.get_flat_mutator_core_weights())
            if self.compat_net is None:
                return torch.zeros(CompatibilityNet.EMBED_DIM)
            return self.compat_net.embed(self.get_flat_weights())

    def pair_head_probability(self, other: 'Genome') -> float:
        """Probability P(compatible | self, other) for mutator-owned pair head."""
        with torch.no_grad():
            my_head = _attach_mutator_compat_head(self.mutator)
            other_head = _attach_mutator_compat_head(other.mutator)
            if my_head is not None and other_head is not None:
                my_embed = my_head.embed(self.get_flat_mutator_core_weights())
                other_embed = other_head.embed(other.get_flat_mutator_core_weights())
                prob = my_head.scorer(my_head._pair_features(my_embed, other_embed)).item()
                return float(np.clip(prob, 1e-6, 1.0 - 1e-6))

            # Legacy fallback for old checkpoints/runs that used separate compat_net.
            if self.compat_net is None or other.compat_net is None:
                return 0.5
            my_mut = self.get_flat_mutator_weights()
            other_mut = other.get_flat_mutator_weights()
            my_embed = self.compat_net.embed(my_mut)
            other_embed = other.compat_net.embed(other_mut)
            prob = self.compat_net.scorer(self.compat_net._pair_features(my_embed, other_embed)).item()
            return float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    
    def is_compatible(self, other: 'Genome', threshold: float | None = 0.5) -> bool:
        """Check compatibility using configured mode.

        - distance (legacy): thresholded L2 distance between species tags
        - binary_strict: pairwise classifier probability P(compatible|g1,g2) >= 0.5
        - mutator_affinity: cosine similarity of mutator weights against gate
        - mutator_pair_head: learned pairwise decision over mutator-weight embeddings
        """
        mode = str(getattr(self, 'compat_mode', 'distance'))
        with torch.no_grad():
            if mode == 'mutator_affinity':
                a = self.get_flat_mutator_weights()
                b = other.get_flat_mutator_weights()
                denom = (torch.norm(a) * torch.norm(b)).item()
                if denom <= 1e-12:
                    return True
                cos = float(torch.dot(a, b).item() / denom)
                # Use threshold argument (0..1 gate) mapped to cosine space (-1..1).
                # gate=0.5 -> cos>=0.0 (legacy behavior)
                if threshold is None:
                    threshold = self.compat_threshold
                gate01 = float(np.clip(float(threshold), 0.0, 1.0))
                cos_gate = float(np.clip((2.0 * gate01) - 1.0, -0.99, 0.99))
                return cos >= cos_gate

            if self.compat_net is None or other.compat_net is None:
                return True

            if mode == 'mutator_pair_head':
                # Classifier-owned binary decision with fixed p>=0.5 boundary.
                return self.pair_head_probability(other) >= 0.5

            if mode == 'binary_strict':
                my_flat = self.get_flat_weights()
                other_flat = other.get_flat_weights()
                my_embed = self.compat_net.embed(my_flat)
                other_embed = other.compat_net.embed(other_flat)
                prob = self.compat_net.scorer(self.compat_net._pair_features(my_embed, other_embed)).item()
                return prob >= 0.5

            if threshold is None:
                threshold = self.compat_threshold

            my_tag = self.get_species_tag()
            other_tag = other.get_species_tag()
            dist = torch.norm(my_tag - other_tag).item()
            max_dist = threshold * (CompatibilityNet.EMBED_DIM ** 0.5)
            return dist < max_dist


    @property
    def is_flexible(self) -> bool:
        return isinstance(self.policy, FlexiblePolicy)

    @torch.no_grad()
    def reproduce(self, generation: int = 0, max_generations: int = 100,
                  mutation_decay: bool = False) -> 'Genome':
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
            compat_threshold=self.compat_threshold,
            evolve_compat_threshold=self.evolve_compat_threshold,
            compat_mode=self.compat_mode,
        )
        child.set_flat_weights(self.get_flat_weights())
        child.structural_rate = self.structural_rate
        child.last_structural_mutation = None
        if self.evolve_compat_threshold and self.compat_net is not None:
            sigma = self.COMPAT_THRESHOLD_MUTATION_STD_REPRO
            z = np.random.randn()
            # drift-neutral log-normal mutation: E[exp(sigma*z - 0.5*sigma^2)] ~= 1
            factor = np.exp(sigma * z - 0.5 * (sigma ** 2))
            child.compat_threshold = float(np.clip(
                self.compat_threshold * factor,
                0.05,
                2.0,
            ))

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

        decay = (max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))
                 if mutation_decay else 1.0)

        # Extract policy size + optional policy segment boundaries for group-wise mutators
        parent_n_policy = self.num_policy_params()
        policy_coords = torch.tensor(self._policy_weight_coords(parent_n_policy), dtype=torch.long)

        # Feed parent's full genome through parent's mutator
        mutated_full = self.mutator.mutate_genome(
            parent_full,
            weight_coords=policy_coords,
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
                  max_generations: int = 100,
                  mutation_decay: bool = False) -> 'Genome':
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
        
        decay = (max(0.2, 1.0 - 0.8 * (generation / max(max_generations, 1)))
                 if mutation_decay else 1.0)
        
        # Let the fitter parent's mutator see both genomes and decide
        policy_coords = torch.tensor(self._policy_weight_coords(n_policy), dtype=torch.long)
        mutated_full = self.mutator.mutate_genome(my_flat, weight_coords=policy_coords, other_weights=other_flat)
        gate_stats = getattr(self.mutator, 'last_crossover_gate_stats', None)
        
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
            compat_threshold=(self.compat_threshold + other.compat_threshold) / 2,
            evolve_compat_threshold=(self.evolve_compat_threshold or other.evolve_compat_threshold),
            compat_mode=self.compat_mode,
        )
        child.set_flat_weights(new_flat)
        child.self_replication_fidelity = fidelity
        child.mutator_delta_norm = mutator_delta_norm
        child.structural_rate = (self.structural_rate + other.structural_rate) / 2
        if gate_stats:
            child.crossover_gate_alpha_mean = float(gate_stats.get('alpha_mean', 0.5))
            child.crossover_gate_alpha_std = float(gate_stats.get('alpha_std', 0.0))
        else:
            child.crossover_gate_alpha_mean = 0.5
            child.crossover_gate_alpha_std = 0.0
        if child.evolve_compat_threshold and child.compat_net is not None:
            sigma = self.COMPAT_THRESHOLD_MUTATION_STD_CROSS
            z = np.random.randn()
            # drift-neutral log-normal mutation
            factor = np.exp(sigma * z - 0.5 * (sigma ** 2))
            child.compat_threshold = float(np.clip(
                child.compat_threshold * factor,
                0.05,
                2.0,
            ))
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

    def _policy_weight_coords(self, n_policy: int) -> List[int]:
        """Coordinate vector for mutators: [policy_split, policy_param_boundaries...]."""
        coords = [int(n_policy)]
        offset = 0
        boundaries = []
        for p in self.policy.parameters():
            offset += p.numel()
            if 0 < offset < n_policy:
                boundaries.append(int(offset))
        coords.extend(boundaries)
        return coords

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
            'compat_threshold': float(self.compat_threshold),
            'evolve_compat_threshold': bool(self.evolve_compat_threshold),
            'compat_mode': str(self.compat_mode),
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
        mut_state = data['mutator_state']
        mut_kwargs = {}
        # Infer constructor shapes from saved state to support non-default configs
        # across heterogeneous workers.
        if mutator_key in {'dualcorrector', 'dualmixture'}:
            if 'encoder.0.weight' in mut_state:
                mut_kwargs['chunk_size'] = int(mut_state['encoder.0.weight'].shape[1])
                mut_kwargs['ref_dim'] = int(mut_state['encoder.0.weight'].shape[0])
            if 'corrector.0.weight' in mut_state:
                mut_kwargs['hidden'] = int(mut_state['corrector.0.weight'].shape[0])
        elif mutator_key == 'dualmixture_v2' and 'group_log_scales' in mut_state:
            mut_kwargs['max_policy_groups'] = int(mut_state['group_log_scales'].numel())
        elif mutator_key == 'global_lowrank':
            if 'encoder.0.weight' in mut_state:
                mut_kwargs['block_size'] = int(mut_state['encoder.0.weight'].shape[1])
                mut_kwargs['ref_dim'] = int(mut_state['encoder.0.weight'].shape[0])
            if 'pool_mlp.0.weight' in mut_state:
                mut_kwargs['hidden'] = int(mut_state['pool_mlp.0.weight'].shape[0])
            if 'coeff_head.2.weight' in mut_state:
                mut_kwargs['rank'] = int(mut_state['coeff_head.2.weight'].shape[0])
        elif mutator_key == 'perceiver_lite':
            if 'encoder.0.weight' in mut_state:
                mut_kwargs['block_size'] = int(mut_state['encoder.0.weight'].shape[1])
                mut_kwargs['ref_dim'] = int(mut_state['encoder.0.weight'].shape[0])
            if 'query_proj.weight' in mut_state:
                mut_kwargs['hidden'] = int(mut_state['query_proj.weight'].shape[0])
            if 'latents' in mut_state:
                mut_kwargs['latent_count'] = int(mut_state['latents'].shape[0])

        mutator = create_mutator(mutator_key, **mut_kwargs)
        has_head_state = any(str(k).startswith('compat_head.') for k in mut_state.keys())
        if has_head_state and 'compat_head.projection' in mut_state:
            head_dim = int(mut_state['compat_head.projection'].shape[0])
            _attach_mutator_compat_head(mutator, core_dim=head_dim)
        strict_mutator_load = True
        if mutator_key in {'dualcorrector', 'dualmixture'}:
            has_gate_state = any(str(k).startswith('gate_head.') for k in mut_state.keys())
            if not has_gate_state:
                strict_mutator_load = False
        mutator.load_state_dict(mut_state, strict=strict_mutator_load)
        compat_net = None
        if 'compat_state' in data:
            compat_net = CompatibilityNet(data['compat_genome_dim'])
            compat_net.load_state_dict(data['compat_state'])
        genome = Genome(
            policy,
            mutator,
            mutator_type=mutator_key,
            compat_net=compat_net,
            compat_threshold=float(data.get('compat_threshold', 0.5)),
            evolve_compat_threshold=bool(data.get('evolve_compat_threshold', False)),
            compat_mode=str(data.get('compat_mode', 'distance')),
        )
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
