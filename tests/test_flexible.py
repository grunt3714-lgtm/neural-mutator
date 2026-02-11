"""Tests for variable-length genome encoding and structural mutations."""

import torch
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.genome import (
    FlexiblePolicy, Policy, StructuralMutation, Genome,
    ChunkMutator, GaussianMutator, MIN_NEURONS, MAX_NEURONS, MIN_LAYERS, MAX_LAYERS
)


class TestFlexiblePolicy:
    """Test FlexiblePolicy with various architectures."""

    def test_single_layer(self):
        p = FlexiblePolicy(4, 2, [32])
        obs = np.random.randn(4).astype(np.float32)
        action = p.act(obs)
        assert action.shape == (2,)

    def test_two_layers(self):
        p = FlexiblePolicy(4, 2, [64, 64])
        obs = np.random.randn(4).astype(np.float32)
        action = p.act(obs)
        assert action.shape == (2,)

    def test_three_layers_varied(self):
        p = FlexiblePolicy(8, 3, [128, 64, 32])
        obs = np.random.randn(8).astype(np.float32)
        action = p.act(obs)
        assert action.shape == (3,)

    def test_default_architecture(self):
        p = FlexiblePolicy(4, 2)
        assert p.layer_sizes == [64, 64]

    def test_topology(self):
        p = FlexiblePolicy(4, 2, [32, 16])
        assert p.get_topology() == [32, 16]

    def test_param_count_varies(self):
        p1 = FlexiblePolicy(4, 2, [32])
        p2 = FlexiblePolicy(4, 2, [64, 64])
        assert sum(p.numel() for p in p1.parameters()) < sum(p.numel() for p in p2.parameters())

    def test_forward(self):
        p = FlexiblePolicy(4, 2, [32, 16])
        x = torch.randn(1, 4)
        out = p(x)
        assert out.shape == (1, 2)


class TestStructuralMutations:
    """Test structural mutations preserve function approximately."""

    def _make_policy(self, sizes=None):
        if sizes is None:
            sizes = [32, 32]
        return FlexiblePolicy(4, 2, sizes)

    def _eval_policy(self, policy, obs):
        with torch.no_grad():
            return policy(torch.FloatTensor(obs).unsqueeze(0)).squeeze(0).numpy()

    def test_add_neuron(self):
        p = self._make_policy([32, 32])
        obs = np.random.randn(4).astype(np.float32)
        out_before = self._eval_policy(p, obs)
        success = p.add_neuron(0)
        assert success
        assert p.layer_sizes[0] == 33
        out_after = self._eval_policy(p, obs)
        # Output should be similar (new neuron has small weights)
        assert np.allclose(out_before, out_after, atol=0.5)

    def test_remove_neuron(self):
        p = self._make_policy([32, 32])
        obs = np.random.randn(4).astype(np.float32)
        out_before = self._eval_policy(p, obs)
        success = p.remove_neuron(0)
        assert success
        assert p.layer_sizes[0] == 31
        out_after = self._eval_policy(p, obs)
        # Removing smallest-norm neuron should have small effect
        assert np.allclose(out_before, out_after, atol=1.0)

    def test_remove_neuron_min_constraint(self):
        p = self._make_policy([MIN_NEURONS])
        success = p.remove_neuron(0)
        assert not success
        assert p.layer_sizes[0] == MIN_NEURONS

    def test_add_neuron_max_constraint(self):
        p = self._make_policy([MAX_NEURONS])
        success = p.add_neuron(0)
        assert not success

    def test_add_layer(self):
        p = self._make_policy([32])
        obs = np.random.randn(4).astype(np.float32)
        out_before = self._eval_policy(p, obs)
        success = p.add_layer(1)
        assert success
        assert len(p.layer_sizes) == 2
        out_after = self._eval_policy(p, obs)
        # Near-identity init should preserve function approximately
        assert np.allclose(out_before, out_after, atol=2.0)

    def test_remove_layer(self):
        p = self._make_policy([32, 32])
        success = p.remove_layer(0)
        assert success
        assert len(p.layer_sizes) == 1

    def test_remove_layer_min_constraint(self):
        p = self._make_policy([32])
        success = p.remove_layer(0)
        assert not success
        assert len(p.layer_sizes) == 1

    def test_add_layer_max_constraint(self):
        p = self._make_policy([32] * MAX_LAYERS)
        success = p.add_layer()
        assert not success

    def test_apply_structural_mutation_random(self):
        p = self._make_policy([32, 32])
        mutation = p.apply_structural_mutation()
        assert mutation is not None
        assert isinstance(mutation, StructuralMutation)


class TestVariableLengthGenomeSerialization:
    """Test genome serialization with variable-length policies."""

    def test_flexible_genome_flat_roundtrip(self):
        policy = FlexiblePolicy(4, 2, [32, 16])
        mutator = ChunkMutator(chunk_size=64)
        g = Genome(policy, mutator, 'chunk')
        flat = g.get_flat_weights()
        
        # Clone and restore
        policy2 = FlexiblePolicy(4, 2, [32, 16])
        mutator2 = ChunkMutator(chunk_size=64)
        g2 = Genome(policy2, mutator2, 'chunk')
        g2.set_flat_weights(flat)
        
        flat2 = g2.get_flat_weights()
        assert torch.allclose(flat, flat2)

    def test_different_architectures_different_sizes(self):
        g1 = Genome(FlexiblePolicy(4, 2, [32]), ChunkMutator(), 'chunk')
        g2 = Genome(FlexiblePolicy(4, 2, [64, 64]), ChunkMutator(), 'chunk')
        assert g1.num_policy_params() != g2.num_policy_params()

    def test_is_flexible(self):
        g1 = Genome(FlexiblePolicy(4, 2, [32]), ChunkMutator(), 'chunk')
        g2 = Genome(Policy(4, 2, 64), ChunkMutator(), 'chunk')
        assert g1.is_flexible
        assert not g2.is_flexible


class TestCrossoverDifferentArchitectures:
    """Test crossover between genomes with different architectures."""

    def test_crossover_same_arch(self):
        g1 = Genome(FlexiblePolicy(4, 2, [32, 32]), ChunkMutator(), 'chunk')
        g2 = Genome(FlexiblePolicy(4, 2, [32, 32]), ChunkMutator(), 'chunk')
        g1.fitness = 10.0
        g2.fitness = 5.0
        child = g1.crossover(g2)
        assert child.is_flexible
        assert child.policy.layer_sizes == [32, 32]

    def test_crossover_different_arch(self):
        g1 = Genome(FlexiblePolicy(4, 2, [64, 64]), ChunkMutator(), 'chunk')
        g2 = Genome(FlexiblePolicy(4, 2, [32]), ChunkMutator(), 'chunk')
        g1.fitness = 10.0
        g2.fitness = 5.0
        # g1 is fitter, so child should have g1's architecture
        child = g1.crossover(g2)
        assert child.policy.layer_sizes == [64, 64]

    def test_crossover_fixed_still_works(self):
        g1 = Genome(Policy(4, 2, 64), ChunkMutator(), 'chunk')
        g2 = Genome(Policy(4, 2, 64), ChunkMutator(), 'chunk')
        child = g1.crossover(g2)
        assert not child.is_flexible


class TestReproduceWithStructuralMutation:
    """Test reproduce with structural mutations."""

    def test_reproduce_no_structural(self):
        """With structural_rate=0, no structural mutation should occur."""
        g = Genome(FlexiblePolicy(4, 2, [32, 32]), ChunkMutator(), 'chunk')
        g.structural_rate = 0.0
        child = g.reproduce()
        assert child.policy.layer_sizes == [32, 32]
        assert child.last_structural_mutation is None

    def test_reproduce_forced_structural(self):
        """With structural_rate=1.0, structural mutation should always occur."""
        np.random.seed(42)
        torch.manual_seed(42)
        g = Genome(FlexiblePolicy(4, 2, [32, 32]), GaussianMutator(), 'gaussian')
        g.structural_rate = 1.0
        child = g.reproduce()
        assert child.last_structural_mutation is not None

    def test_reproduce_preserves_fixed_policy(self):
        """Fixed policy genomes should never get structural mutations."""
        g = Genome(Policy(4, 2, 64), ChunkMutator(), 'chunk')
        child = g.reproduce()
        assert not child.is_flexible

    def test_structural_rate_evolves(self):
        """Structural rate should change slightly in offspring."""
        np.random.seed(42)
        g = Genome(FlexiblePolicy(4, 2, [32, 32]), GaussianMutator(), 'gaussian')
        g.structural_rate = 1.0  # force structural mutation
        child = g.reproduce()
        # Rate should have changed (log-normal walk)
        # It might be the same by chance, so just check it's valid
        assert 0.001 <= child.structural_rate <= 0.5

    def test_child_functional_after_structural(self):
        """Child should produce valid actions after structural mutation."""
        np.random.seed(42)
        g = Genome(FlexiblePolicy(4, 2, [32, 32]), GaussianMutator(), 'gaussian')
        g.structural_rate = 1.0
        child = g.reproduce()
        obs = np.random.randn(4).astype(np.float32)
        action = child.policy.act(obs)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
