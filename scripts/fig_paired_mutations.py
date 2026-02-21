#!/usr/bin/env python3
"""Publication figure: How mutator mutations pair with network activity.

Shows the relationship between which FC hidden neurons are active during
inference and how the mutator's mutation deltas are distributed and correlated
across those neurons.

Key questions:
1. Do co-active neurons get co-mutated? (activity correlation vs mutation correlation)
2. Does the mutator target active neurons differently than dead ones?
3. Are mutation deltas for a single neuron's weights internally coherent?

Usage:
    python scripts/fig_paired_mutations.py [--genome PATH] [--output PATH]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from src.genome import Genome
import gymnasium as gym


def collect_fc_activations(genome, env_id="CarRacing-v3", n_steps=500, seed=42):
    """Run policy and record FC hidden layer activations per timestep."""
    env = gym.make(env_id, render_mode=None)
    obs, _ = env.reset(seed=seed)
    
    policy = genome.policy
    policy.eval()
    
    activations = []  # (n_steps, 64) — ReLU outputs of FC hidden
    
    # Hook into FC hidden layer
    hook_data = []
    def hook_fn(module, input, output):
        hook_data.append(output.detach().cpu().numpy().flatten())
    
    # Find the FC hidden layer (after conv stack, first Linear + ReLU)
    # In PolicyCNNLarge: fc = Sequential(Linear, ReLU, Linear)
    # fc[0] is the hidden linear, fc[1] is ReLU
    handle = policy.fc[1].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for step in range(n_steps):
            if isinstance(obs, np.ndarray):
                t_obs = torch.from_numpy(obs.copy()).unsqueeze(0)
            action = policy(t_obs).squeeze().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    
    handle.remove()
    env.close()
    
    activations = np.array(hook_data)  # (n_steps, hidden_dim)
    return activations


def collect_neuron_mutations(genome, n_mutations=200):
    """Collect mutation deltas reshaped per FC hidden neuron.
    
    Returns:
        neuron_deltas: (n_mutations, n_neurons, weights_per_neuron) — delta for each neuron's input weights
        neuron_bias_deltas: (n_mutations, n_neurons) — delta for each neuron's bias
        full_deltas: (n_mutations, n_params) — full flat delta
    """
    parent_flat = genome.get_flat_weights().clone()
    
    # Find FC hidden layer weight indices in flat param vector
    policy = genome.policy
    offset = 0
    fc_weight_offset = None
    fc_weight_shape = None
    fc_bias_offset = None
    fc_bias_shape = None
    
    for name, param in policy.named_parameters():
        size = param.numel()
        if name == "fc.0.weight":
            fc_weight_offset = offset
            fc_weight_shape = param.shape  # (hidden, in_features)
        elif name == "fc.0.bias":
            fc_bias_offset = offset
            fc_bias_shape = param.shape  # (hidden,)
        offset += size
    
    assert fc_weight_offset is not None, "Could not find fc.0.weight"
    n_neurons = fc_weight_shape[0]
    in_features = fc_weight_shape[1]
    
    print(f"  FC hidden: {n_neurons} neurons × {in_features} input features")
    print(f"  Weight offset: {fc_weight_offset}, Bias offset: {fc_bias_offset}")
    
    full_deltas = []
    for _ in range(n_mutations):
        genome.set_flat_weights(parent_flat.clone())
        child = genome.reproduce()
        delta = (child.get_flat_weights() - parent_flat).detach().numpy()
        full_deltas.append(delta)
    
    genome.set_flat_weights(parent_flat)
    full_deltas = np.array(full_deltas)  # (n_mut, n_params)
    
    # Extract per-neuron weight deltas
    w_start = fc_weight_offset
    w_end = w_start + n_neurons * in_features
    neuron_deltas = full_deltas[:, w_start:w_end].reshape(len(full_deltas), n_neurons, in_features)
    
    # Extract per-neuron bias deltas
    b_start = fc_bias_offset
    neuron_bias_deltas = full_deltas[:, b_start:b_start + n_neurons]
    
    return neuron_deltas, neuron_bias_deltas, full_deltas, n_neurons, in_features


def make_figure(genome_path, output_path="results/paired_mutations_fig.png",
                n_mutations=200, n_steps=500):
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading genome: {genome_path}")
    genome = Genome.load(genome_path)
    print(f"  Fitness: {genome.fitness:.1f}")
    
    # 1. Collect activations
    print(f"Collecting {n_steps} steps of FC activations...")
    activations = collect_fc_activations(genome, n_steps=n_steps)
    n_neurons_act = activations.shape[1]
    print(f"  Activation matrix: {activations.shape}")
    
    # 2. Collect per-neuron mutations
    print(f"Collecting {n_mutations} mutation samples...")
    neuron_deltas, bias_deltas, full_deltas, n_neurons, in_features = \
        collect_neuron_mutations(genome, n_mutations)
    
    assert n_neurons == n_neurons_act, f"Neuron count mismatch: {n_neurons} vs {n_neurons_act}"
    
    # --- Derived metrics ---
    
    # Per-neuron mean activation (across timesteps)
    mean_act = np.mean(activations, axis=0)  # (n_neurons,)
    # Fraction of time each neuron is active (>0 after ReLU)
    frac_active = np.mean(activations > 0, axis=0)  # (n_neurons,)
    
    # Per-neuron mean absolute mutation magnitude
    mean_mut_mag = np.mean(np.abs(neuron_deltas), axis=(0, 2))  # (n_neurons,)
    # Per-neuron mutation std (spread of deltas across samples)
    mut_std = np.std(neuron_deltas.reshape(n_mutations, n_neurons, -1), axis=(0, 2))  # (n_neurons,)
    
    # Activity correlation matrix (n_neurons × n_neurons) — how neurons co-activate
    act_corr = np.corrcoef(activations.T)  # (n, n)
    
    # Mutation correlation matrix — how neurons' deltas co-vary across mutation samples
    # For each mutation sample, each neuron has a vector of weight deltas
    # We compute: for neuron i and j, correlation of their mean |delta| across samples
    neuron_mut_means = np.mean(np.abs(neuron_deltas), axis=2)  # (n_mutations, n_neurons)
    mut_corr = np.corrcoef(neuron_mut_means.T)  # (n, n)
    
    # Extract upper triangles for scatter
    triu_idx = np.triu_indices(n_neurons, k=1)
    act_corr_flat = act_corr[triu_idx]
    mut_corr_flat = mut_corr[triu_idx]
    valid = np.isfinite(act_corr_flat) & np.isfinite(mut_corr_flat)
    act_corr_flat = act_corr_flat[valid]
    mut_corr_flat = mut_corr_flat[valid]
    
    # Classify neurons
    dead_mask = frac_active < 0.01
    active_mask = frac_active > 0.5
    partial_mask = ~dead_mask & ~active_mask
    
    print(f"  Dead neurons: {np.sum(dead_mask)}, Active: {np.sum(active_mask)}, Partial: {np.sum(partial_mask)}")
    
    # --- Random baseline ---
    rnd_deltas = np.random.randn(*neuron_deltas.shape) * np.std(neuron_deltas)
    rnd_mut_means = np.mean(np.abs(rnd_deltas), axis=2)
    rnd_mut_corr = np.corrcoef(rnd_mut_means.T)
    rnd_corr_flat = rnd_mut_corr[triu_idx][valid]
    
    # --- FIGURE ---
    fig = plt.figure(figsize=(16, 11), dpi=200)
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35,
                  left=0.06, right=0.97, top=0.91, bottom=0.07)
    
    c_active = "#2166ac"
    c_dead = "#b2182b"
    c_partial = "#999999"
    c_mut = "#2166ac"
    c_rnd = "#b2182b"
    
    # ── Panel A: Neuron activation profile + mutation magnitude ──
    ax_a = fig.add_subplot(gs[0, 0])
    order = np.argsort(mean_act)[::-1]  # sort by activation
    x = np.arange(n_neurons)
    
    ax_a.bar(x, mean_act[order], color=[c_active if frac_active[i] > 0.5 
             else c_dead if frac_active[i] < 0.01 else c_partial for i in order],
             alpha=0.7, width=1.0, label="Mean activation")
    
    ax_a2 = ax_a.twinx()
    ax_a2.plot(x, mean_mut_mag[order], color="orange", linewidth=1.2, alpha=0.9, label="Mutation |δ|")
    ax_a2.set_ylabel("Mean |mutation δ|", fontsize=10, color="orange")
    ax_a2.tick_params(axis='y', labelcolor='orange')
    
    ax_a.set_xlabel("Neuron (sorted by activation)", fontsize=10)
    ax_a.set_ylabel("Mean activation", fontsize=10)
    ax_a.set_title("A   Neuron Activity vs Mutation Targeting", fontsize=11, fontweight="bold", loc="left")
    
    # Combined legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(fc=c_active, alpha=0.7, label=f"Active ({np.sum(active_mask)})"),
               Patch(fc=c_dead, alpha=0.7, label=f"Dead ({np.sum(dead_mask)})"),
               Patch(fc=c_partial, alpha=0.7, label=f"Partial ({np.sum(partial_mask)})"),
               Line2D([0], [0], color="orange", label="Mutation |δ|")]
    ax_a.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9)
    
    # ── Panel B: Scatter — activity fraction vs mutation magnitude per neuron ──
    ax_b = fig.add_subplot(gs[0, 1])
    colors = [c_active if frac_active[i] > 0.5 else c_dead if frac_active[i] < 0.01 
              else c_partial for i in range(n_neurons)]
    ax_b.scatter(frac_active, mean_mut_mag, c=colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)
    
    # Regression line
    slope, intercept, r_val, p_val, _ = stats.linregress(frac_active, mean_mut_mag)
    x_line = np.linspace(0, 1, 100)
    ax_b.plot(x_line, slope * x_line + intercept, "--", color="gray", linewidth=1.5, alpha=0.7)
    
    ax_b.set_xlabel("Fraction of time active", fontsize=10)
    ax_b.set_ylabel("Mean |mutation δ|", fontsize=10)
    ax_b.set_title("B   Activity vs Mutation Scale", fontsize=11, fontweight="bold", loc="left")
    ax_b.annotate(f"r = {r_val:.3f}\np = {p_val:.2e}",
                  xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                  fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    # ── Panel C: Co-activation vs co-mutation scatter ──
    ax_c = fig.add_subplot(gs[0, 2])
    # Subsample for readability
    n_pairs = len(act_corr_flat)
    if n_pairs > 3000:
        idx = np.random.choice(n_pairs, 3000, replace=False)
    else:
        idx = np.arange(n_pairs)
    
    ax_c.scatter(act_corr_flat[idx], mut_corr_flat[idx], s=3, alpha=0.15, color=c_mut, label="Mutator")
    ax_c.scatter(act_corr_flat[idx], rnd_corr_flat[idx], s=3, alpha=0.15, color=c_rnd, label="Random")
    
    r_mut, p_mut = pearsonr(act_corr_flat, mut_corr_flat)
    r_rnd, p_rnd = pearsonr(act_corr_flat, rnd_corr_flat) if len(rnd_corr_flat) == len(act_corr_flat) else (0, 1)
    
    ax_c.set_xlabel("Co-activation correlation", fontsize=10)
    ax_c.set_ylabel("Co-mutation correlation", fontsize=10)
    ax_c.set_title("C   Co-Active → Co-Mutated?", fontsize=11, fontweight="bold", loc="left")
    ax_c.legend(fontsize=8, markerscale=4, framealpha=0.9)
    ax_c.annotate(f"Mutator r = {r_mut:.3f} (p={p_mut:.1e})\nRandom r = {r_rnd:.3f}",
                  xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                  fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    # ── Panel D: Distribution of mutation magnitude by neuron class ──
    ax_d = fig.add_subplot(gs[1, 0])
    
    active_mags = mean_mut_mag[active_mask]
    dead_mags = mean_mut_mag[dead_mask]
    partial_mags = mean_mut_mag[partial_mask]
    
    parts = ax_d.violinplot([active_mags, partial_mags, dead_mags] if len(dead_mags) > 0 
                            else [active_mags, partial_mags],
                            positions=[1, 2, 3] if len(dead_mags) > 0 else [1, 2],
                            showmeans=True, showmedians=True)
    
    colors_v = [c_active, c_partial, c_dead]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_v[i])
        pc.set_alpha(0.6)
    
    labels = ["Active", "Partial", "Dead"] if len(dead_mags) > 0 else ["Active", "Partial"]
    ax_d.set_xticks([1, 2, 3] if len(dead_mags) > 0 else [1, 2])
    ax_d.set_xticklabels(labels, fontsize=10)
    ax_d.set_ylabel("Mean |mutation δ|", fontsize=10)
    ax_d.set_title("D   Mutation Scale by Neuron Class", fontsize=11, fontweight="bold", loc="left")
    
    # Mann-Whitney test
    if len(active_mags) > 1 and len(dead_mags) > 1:
        U, p_mw = stats.mannwhitneyu(active_mags, dead_mags, alternative='two-sided')
        ax_d.annotate(f"Active vs Dead\nMW U={U:.0f}, p={p_mw:.2e}",
                      xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                      fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    # ── Panel E: Mutation internal coherence per neuron ──
    # For each neuron, compute mean pairwise cosine similarity of its weight deltas across mutations
    ax_e = fig.add_subplot(gs[1, 1])
    
    neuron_coherence = []
    for ni in range(n_neurons):
        # neuron_deltas[:, ni, :] is (n_mutations, in_features)
        nd = neuron_deltas[:, ni, :]
        norms = np.linalg.norm(nd, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = nd / norms
        # Mean pairwise cosine
        cos_mat = normed @ normed.T
        triu = cos_mat[np.triu_indices(n_mutations, k=1)]
        neuron_coherence.append(np.mean(triu))
    
    neuron_coherence = np.array(neuron_coherence)
    
    # Random baseline coherence
    rnd_coherence = []
    for ni in range(n_neurons):
        rd = rnd_deltas[:, ni, :]
        norms = np.linalg.norm(rd, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = rd / norms
        cos_mat = normed @ normed.T
        triu = cos_mat[np.triu_indices(n_mutations, k=1)]
        rnd_coherence.append(np.mean(triu))
    rnd_coherence = np.array(rnd_coherence)
    
    ax_e.scatter(frac_active, neuron_coherence, s=25, alpha=0.7, color=c_mut,
                 edgecolors="white", linewidth=0.3, label="Mutator", zorder=3)
    ax_e.scatter(frac_active, rnd_coherence, s=15, alpha=0.4, color=c_rnd,
                 edgecolors="none", label="Random", zorder=2)
    
    ax_e.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax_e.set_xlabel("Fraction of time active", fontsize=10)
    ax_e.set_ylabel("Mutation direction coherence\n(mean pairwise cosine)", fontsize=10)
    ax_e.set_title("E   Mutation Coherence per Neuron", fontsize=11, fontweight="bold", loc="left")
    ax_e.legend(fontsize=8, framealpha=0.9)
    
    r_coh, p_coh = spearmanr(frac_active, neuron_coherence)
    ax_e.annotate(f"Spearman ρ = {r_coh:.3f}\np = {p_coh:.2e}",
                  xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                  fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    # ── Panel F: Heatmap — neurons sorted by activity, showing mutation correlation matrix ──
    ax_f = fig.add_subplot(gs[1, 2])
    order_act = np.argsort(frac_active)[::-1]
    sorted_mut_corr = mut_corr[np.ix_(order_act, order_act)]
    
    im = ax_f.imshow(sorted_mut_corr, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax_f.set_xlabel("Neuron (sorted by activity)", fontsize=10)
    ax_f.set_ylabel("Neuron (sorted by activity)", fontsize=10)
    ax_f.set_title("F   Mutation Correlation Matrix", fontsize=11, fontweight="bold", loc="left")
    
    # Mark boundary between active and inactive
    n_active_sorted = np.sum(active_mask)
    if n_active_sorted > 0 and n_active_sorted < n_neurons:
        ax_f.axhline(n_active_sorted - 0.5, color="lime", linewidth=1, linestyle="--", alpha=0.8)
        ax_f.axvline(n_active_sorted - 0.5, color="lime", linewidth=1, linestyle="--", alpha=0.8)
        ax_f.text(n_active_sorted + 1, 2, "← active | partial/dead →", fontsize=7, color="lime")
    
    plt.colorbar(im, ax=ax_f, fraction=0.046, label="Correlation")
    
    # ── Suptitle ──
    fig.suptitle(
        "Learned Mutator Pairs Mutations with Network Activity\n"
        f"CarRacing-v3  ·  {n_neurons} FC hidden neurons  ·  {n_mutations} mutations  ·  {n_steps} inference steps",
        fontsize=13, fontweight="bold", y=0.98
    )
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.0f} KB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--genome", default="results/carracing_fleet_p100_g400_s42_e3_pull1_clean_py/best_ever_genome.pt")
    p.add_argument("--n-mutations", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--output", default="results/paired_mutations_fig.png")
    args = p.parse_args()
    make_figure(args.genome, args.output, args.n_mutations, args.n_steps)
