#!/usr/bin/env python3
"""Mechanistic interpretability analysis of neural mutator genomes.

Inspects:
1. What mutation patterns each mutator type produces
2. Weight distributions before/after mutation
3. Mutator internal activations & learned features
4. Whether mutators learn structured vs random perturbations
5. Mutation scale parameter evolution
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import gymnasium as gym

from src.genome import Genome, Policy, ChunkMutator, TransformerMutator, CPPNMutator
from src.evolution import create_population, evaluate_genome, evolve_generation


def analyze_mutation_patterns(save_dir: Path):
    """Compare what each mutator type does to a genome's weights."""
    obs_dim, act_dim = 4, 2  # CartPole
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    for col, (mut_type, MutClass) in enumerate([
        ('chunk', ChunkMutator),
        ('transformer', TransformerMutator),
        ('cppn', CPPNMutator),
    ]):
        # Create a genome
        policy = Policy(obs_dim, act_dim, hidden=64)
        mutator = MutClass(chunk_size=64) if mut_type != 'cppn' else CPPNMutator()
        genome = Genome(policy, mutator, mut_type)
        
        parent_weights = genome.get_flat_weights().clone()
        
        # Apply mutation multiple times to see patterns
        deltas = []
        for _ in range(50):
            # Reset to parent each time
            genome.set_flat_weights(parent_weights.clone())
            child = genome.reproduce()
            child_weights = child.get_flat_weights()
            delta = child_weights - parent_weights
            deltas.append(delta.numpy())
        
        deltas = np.array(deltas)  # (50, n_params)
        
        # Row 0: Heatmap of deltas (params × trials)
        ax = axes[0, col]
        n_show = min(200, deltas.shape[1])
        im = ax.imshow(deltas[:, :n_show].T, aspect='auto', cmap='RdBu_r', 
                       vmin=-0.05, vmax=0.05)
        ax.set_title(f'{mut_type.upper()} — Delta Heatmap')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Weight index')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 1: Distribution of delta magnitudes
        ax = axes[1, col]
        flat_deltas = deltas.flatten()
        ax.hist(flat_deltas, bins=100, density=True, alpha=0.7, color='steelblue')
        ax.set_title(f'{mut_type.upper()} — Delta Distribution')
        ax.set_xlabel('Delta magnitude')
        ax.set_ylabel('Density')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        stats_text = f'μ={np.mean(flat_deltas):.4f}\nσ={np.std(flat_deltas):.4f}\n|max|={np.max(np.abs(flat_deltas)):.4f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, va='top', ha='right',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 2: Spatial correlation — are nearby weights mutated similarly?
        ax = axes[2, col]
        mean_delta = np.mean(deltas, axis=0)
        std_delta = np.std(deltas, axis=0)
        ax.plot(mean_delta[:n_show], label='Mean delta', alpha=0.7, linewidth=0.5)
        ax.fill_between(range(n_show), 
                        mean_delta[:n_show] - std_delta[:n_show],
                        mean_delta[:n_show] + std_delta[:n_show],
                        alpha=0.2)
        ax.set_title(f'{mut_type.upper()} — Spatial Pattern')
        ax.set_xlabel('Weight index')
        ax.set_ylabel('Delta (mean ± std)')
        ax.axhline(0, color='red', linestyle='--', alpha=0.3)
    
    # Column 3: Gaussian baseline for comparison
    for row in range(3):
        ax = axes[row, 3]
        gaussian_deltas = np.random.randn(50, parent_weights.shape[0]) * 0.02
        
        if row == 0:
            im = ax.imshow(gaussian_deltas[:, :n_show].T, aspect='auto', cmap='RdBu_r',
                          vmin=-0.05, vmax=0.05)
            ax.set_title('GAUSSIAN — Delta Heatmap')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Weight index')
            plt.colorbar(im, ax=ax, fraction=0.046)
        elif row == 1:
            flat_g = gaussian_deltas.flatten()
            ax.hist(flat_g, bins=100, density=True, alpha=0.7, color='steelblue')
            ax.set_title('GAUSSIAN — Delta Distribution')
            ax.set_xlabel('Delta magnitude')
            ax.set_ylabel('Density')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            stats_text = f'μ={np.mean(flat_g):.4f}\nσ={np.std(flat_g):.4f}\n|max|={np.max(np.abs(flat_g)):.4f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, va='top', ha='right',
                    fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            mean_g = np.mean(gaussian_deltas, axis=0)
            std_g = np.std(gaussian_deltas, axis=0)
            ax.plot(mean_g[:n_show], label='Mean delta', alpha=0.7, linewidth=0.5)
            ax.fill_between(range(n_show),
                           mean_g[:n_show] - std_g[:n_show],
                           mean_g[:n_show] + std_g[:n_show], alpha=0.2)
            ax.set_title('GAUSSIAN — Spatial Pattern')
            ax.set_xlabel('Weight index')
            ax.set_ylabel('Delta (mean ± std)')
            ax.axhline(0, color='red', linestyle='--', alpha=0.3)
    
    plt.suptitle('Mechanistic Analysis: What Do Mutators Actually Do?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'mutation_patterns.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'mutation_patterns.png'}")


def analyze_mutator_internals(save_dir: Path):
    """Probe what the chunk mutator's internal layers learn."""
    obs_dim, act_dim = 4, 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Create evolved population (run a few gens to get non-random mutators)
    pop = create_population(30, obs_dim, act_dim, 'chunk', hidden=64, chunk_size=64)
    
    # Evaluate and evolve for 20 gens
    for gen in range(20):
        for g in pop:
            g.fitness = evaluate_genome(g, 'CartPole-v1', n_episodes=3, max_steps=200)
        pop = evolve_generation(pop, crossover_rate=0.3, elitism=5)
    
    # Get best genome
    for g in pop:
        g.fitness = evaluate_genome(g, 'CartPole-v1', n_episodes=3, max_steps=200)
    best = max(pop, key=lambda g: g.fitness)
    worst = min(pop, key=lambda g: g.fitness)
    
    print(f"Best fitness: {best.fitness:.1f}, Worst: {worst.fitness:.1f}")
    
    # 1. Weight matrices of the chunk mutator (best vs worst)
    for idx, (genome, label) in enumerate([(best, 'Best'), (worst, 'Worst')]):
        mutator = genome.mutator
        # Get the first layer's weight matrix
        w1 = list(mutator.net.parameters())[0].data.numpy()  # (hidden, chunk_size)
        
        ax = axes[0, idx]
        im = ax.imshow(w1, aspect='auto', cmap='viridis')
        ax.set_title(f'{label} Mutator — Layer 1 Weights')
        ax.set_xlabel('Input (chunk)')
        ax.set_ylabel('Hidden unit')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 2. Mutation scale parameter across population
    ax = axes[0, 2]
    scales = [g.mutator.mutation_scale.item() for g in pop]
    ax.bar(range(len(scales)), sorted(scales, reverse=True), color='coral')
    ax.set_title('Mutation Scale (sorted, population)')
    ax.set_xlabel('Individual')
    ax.set_ylabel('Scale value')
    ax.axhline(0.01, color='blue', linestyle='--', label='Initial (0.01)')
    ax.legend()
    
    # 3. Activation patterns: feed different input chunks through best mutator
    ax = axes[1, 0]
    test_chunks = torch.randn(20, 64)  # Random chunks
    with torch.no_grad():
        # Get intermediate activations
        x = test_chunks
        activations = []
        for layer in best.mutator.net:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x.numpy().copy())
    
    if activations:
        act1 = activations[0]  # First hidden layer activations
        im = ax.imshow(act1.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('Best Mutator — Hidden Activations')
        ax.set_xlabel('Input chunk')
        ax.set_ylabel('Hidden unit')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 4. Singular value decomposition of mutations
    ax = axes[1, 1]
    parent_flat = best.get_flat_weights()
    mutations = []
    for _ in range(30):
        best.set_flat_weights(parent_flat.clone())
        child = best.reproduce()
        delta = (child.get_flat_weights() - parent_flat).numpy()
        mutations.append(delta)
    
    mutations = np.array(mutations)
    U, S, Vt = np.linalg.svd(mutations, full_matrices=False)
    
    ax.plot(S[:20] / S.sum() * 100, 'o-', color='darkgreen')
    ax.set_title('SVD of Mutation Matrix')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('% variance explained')
    ax.set_ylim(0, None)
    
    # If top few SVs explain most variance → structured mutations
    cumvar = np.cumsum(S**2) / np.sum(S**2) * 100
    ax2 = ax.twinx()
    ax2.plot(cumvar[:20], 's--', color='gray', alpha=0.5)
    ax2.set_ylabel('Cumulative %', color='gray')
    
    # 5. Compare policy weight distributions: evolved vs random
    ax = axes[1, 2]
    evolved_policy_w = torch.cat([p.data.view(-1) for p in best.policy.parameters()]).numpy()
    random_policy = Policy(obs_dim, act_dim, hidden=64)
    random_policy_w = torch.cat([p.data.view(-1) for p in random_policy.parameters()]).numpy()
    
    ax.hist(random_policy_w, bins=80, density=True, alpha=0.5, label='Random init', color='gray')
    ax.hist(evolved_policy_w, bins=80, density=True, alpha=0.5, label='Evolved (best)', color='green')
    ax.set_title('Policy Weight Distribution')
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Density')
    ax.legend()
    
    plt.suptitle('Mutator Internals — Chunk Mutator on CartPole (20 gens)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'mutator_internals.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'mutator_internals.png'}")


def analyze_mutation_autocorrelation(save_dir: Path):
    """Do successive mutations from the same mutator correlate? 
    i.e., does the mutator produce consistent direction or just noise?"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    obs_dim, act_dim = 4, 2
    
    for col, mut_type in enumerate(['chunk', 'transformer', 'cppn']):
        policy = Policy(obs_dim, act_dim, hidden=64)
        if mut_type == 'chunk':
            mutator = ChunkMutator(chunk_size=64)
        elif mut_type == 'transformer':
            mutator = TransformerMutator(chunk_size=64)
        else:
            mutator = CPPNMutator()
        
        genome = Genome(policy, mutator, mut_type)
        parent_flat = genome.get_flat_weights().clone()
        
        # Generate many mutations from same parent
        deltas = []
        for _ in range(100):
            genome.set_flat_weights(parent_flat.clone())
            child = genome.reproduce()
            delta = (child.get_flat_weights() - parent_flat).numpy()
            deltas.append(delta)
        
        deltas = np.array(deltas)
        
        # Compute pairwise cosine similarity between mutations
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = deltas / norms
        cos_sim = normalized @ normalized.T
        
        ax = axes[col]
        im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'{mut_type.upper()}\nMean cos_sim={np.mean(cos_sim[np.triu_indices(100, k=1)]):.3f}')
        ax.set_xlabel('Mutation i')
        ax.set_ylabel('Mutation j')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Mutation Consistency: Cosine Similarity Between Mutations from Same Parent', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'mutation_consistency.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'mutation_consistency.png'}")


if __name__ == '__main__':
    save_dir = Path('results/interpretability')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("1. Mutation Patterns Analysis")
    print("="*60)
    analyze_mutation_patterns(save_dir)
    
    print("\n" + "="*60)
    print("2. Mutator Internals Analysis")  
    print("="*60)
    analyze_mutator_internals(save_dir)
    
    print("\n" + "="*60)
    print("3. Mutation Consistency Analysis")
    print("="*60)
    analyze_mutation_autocorrelation(save_dir)
    
    print("\nAll analyses complete!")
