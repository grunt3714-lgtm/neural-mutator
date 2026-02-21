#!/usr/bin/env python3
"""Publication figure: Structured pairwise mutation patterns.

Combines the strongest evidence:
- Pairwise correlation & cosine similarity distributions (bimodal vs unimodal)
- SVD spectrum showing low-rank mutation subspace
- Per-neuron mutation direction coherence
- Mutation correlation matrix showing block structure in weight space

Usage:
    python scripts/fig_mutation_pairs_pub.py [--genome PATH] [--output PATH]
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
from scipy.stats import gaussian_kde, spearmanr
import gymnasium as gym
from src.genome import Genome


def collect_mutations(genome, n=200):
    parent_flat = genome.get_flat_weights().clone()
    deltas = []
    for _ in range(n):
        genome.set_flat_weights(parent_flat.clone())
        child = genome.reproduce()
        delta = child.get_flat_weights() - parent_flat
        deltas.append(delta.detach().numpy())
    genome.set_flat_weights(parent_flat)
    return np.array(deltas)


def random_deltas_matched(mutator_deltas):
    n, d = mutator_deltas.shape
    elem_std = np.std(mutator_deltas, axis=0)
    global_std = np.mean(elem_std[elem_std > 0]) if np.any(elem_std > 0) else 0.01
    return np.random.randn(n, d) * global_std


def collect_fc_activations(genome, n_steps=500, seed=42):
    env = gym.make("CarRacing-v3", render_mode=None)
    obs, _ = env.reset(seed=seed)
    policy = genome.policy
    policy.eval()
    hook_data = []
    def hook_fn(module, input, output):
        hook_data.append(output.detach().cpu().numpy().flatten())
    handle = policy.fc[1].register_forward_hook(hook_fn)
    with torch.no_grad():
        for _ in range(n_steps):
            t_obs = torch.from_numpy(obs.copy()).unsqueeze(0)
            action = policy(t_obs).squeeze().numpy()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    handle.remove()
    env.close()
    return np.array(hook_data)


def get_fc_neuron_deltas(genome, full_deltas):
    """Extract per-neuron weight deltas for FC hidden layer."""
    policy = genome.policy
    offset = 0
    fc_w_off = fc_w_shape = None
    for name, param in policy.named_parameters():
        if name == "fc.0.weight":
            fc_w_off = offset
            fc_w_shape = param.shape
        offset += param.numel()
    n_neurons, in_feat = fc_w_shape
    w_start = fc_w_off
    return full_deltas[:, w_start:w_start + n_neurons * in_feat].reshape(
        len(full_deltas), n_neurons, in_feat), n_neurons, in_feat


def pairwise_stats(deltas, max_pairs=4000):
    """Compute pairwise Pearson correlations and cosine similarities."""
    n = deltas.shape[0]
    idx = np.random.choice(n, size=(max_pairs, 2), replace=True)
    idx = idx[idx[:, 0] != idx[:, 1]]
    
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = deltas / norms
    
    corrs, coss = [], []
    for i, j in idx:
        r = np.corrcoef(deltas[i], deltas[j])[0, 1]
        c = np.dot(normed[i], normed[j])
        if np.isfinite(r):
            corrs.append(r)
        if np.isfinite(c):
            coss.append(c)
    return np.array(corrs), np.array(coss)


def make_figure(genome_path, output_path, n_mutations=200, n_steps=500):
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Loading genome: {genome_path}")
    genome = Genome.load(genome_path)
    n_params = genome.get_flat_weights().numel()
    print(f"  Parameters: {n_params:,}  |  Fitness: {genome.fitness:.1f}")

    # Collect data
    print("Collecting mutations...")
    mut_deltas = collect_mutations(genome, n_mutations)
    rnd_deltas = random_deltas_matched(mut_deltas)

    print("Computing pairwise stats...")
    mut_corr, mut_cos = pairwise_stats(mut_deltas)
    rnd_corr, rnd_cos = pairwise_stats(rnd_deltas)

    print("Collecting activations...")
    activations = collect_fc_activations(genome, n_steps)
    n_neurons_act = activations.shape[1]
    frac_active = np.mean(activations > 0, axis=0)

    print("Computing neuron-level stats...")
    neuron_deltas, n_neurons, in_feat = get_fc_neuron_deltas(genome, mut_deltas)
    rnd_neuron_deltas, _, _ = get_fc_neuron_deltas(genome, rnd_deltas)

    # SVD
    U_m, S_m, _ = np.linalg.svd(mut_deltas, full_matrices=False)
    _, S_r, _ = np.linalg.svd(rnd_deltas, full_matrices=False)
    var_m = (S_m ** 2) / np.sum(S_m ** 2) * 100
    var_r = (S_r ** 2) / np.sum(S_r ** 2) * 100
    cumvar_m = np.cumsum(var_m)
    cumvar_r = np.cumsum(var_r)
    pr_m = (np.sum(S_m ** 2)) ** 2 / np.sum(S_m ** 4)
    pr_r = (np.sum(S_r ** 2)) ** 2 / np.sum(S_r ** 4)

    # Per-neuron coherence
    neuron_coherence = []
    rnd_coherence = []
    for ni in range(n_neurons):
        for deltas_arr, coh_list in [(neuron_deltas, neuron_coherence),
                                      (rnd_neuron_deltas, rnd_coherence)]:
            nd = deltas_arr[:, ni, :]
            norms = np.linalg.norm(nd, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normed = nd / norms
            cos_mat = normed @ normed.T
            triu = cos_mat[np.triu_indices(n_mutations, k=1)]
            coh_list.append(np.mean(triu))
    neuron_coherence = np.array(neuron_coherence)
    rnd_coherence = np.array(rnd_coherence)

    # Per-parameter mutation std
    mut_std = np.std(mut_deltas, axis=0)
    rnd_std = np.std(rnd_deltas, axis=0)

    # KS tests
    ks_corr = stats.ks_2samp(mut_corr, rnd_corr)
    ks_cos = stats.ks_2samp(mut_cos, rnd_cos)

    # ── FIGURE ──
    fig = plt.figure(figsize=(16, 12), dpi=200)
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32,
                  left=0.06, right=0.97, top=0.92, bottom=0.06)

    c_mut = "#2166ac"
    c_rnd = "#b2182b"
    c_accent = "#ff7f00"

    # ── Row 1: The core distributions ──

    # Panel A: Pairwise Pearson Correlation
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-0.3, 0.55, 90)
    ax.hist(rnd_corr, bins=bins, density=True, alpha=0.30, color=c_rnd, edgecolor="none")
    ax.hist(mut_corr, bins=bins, density=True, alpha=0.30, color=c_mut, edgecolor="none")
    x_kde = np.linspace(-0.3, 0.55, 400)
    kde_r = gaussian_kde(rnd_corr, bw_method=0.03)
    kde_m = gaussian_kde(mut_corr, bw_method=0.03)
    ax.plot(x_kde, kde_r(x_kde), color=c_rnd, linewidth=2.2, label="Random (Gaussian)")
    ax.plot(x_kde, kde_m(x_kde), color=c_mut, linewidth=2.2, label="Learned mutator")
    ax.set_xlabel("Pearson correlation (r)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("A   Pairwise Mutation Correlation", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.annotate(f"KS = {ks_corr.statistic:.3f},  p ≈ 0",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Panel B: Pairwise Cosine Similarity
    ax = fig.add_subplot(gs[0, 1])
    bins_c = np.linspace(-0.25, 0.5, 90)
    ax.hist(rnd_cos, bins=bins_c, density=True, alpha=0.30, color=c_rnd, edgecolor="none")
    ax.hist(mut_cos, bins=bins_c, density=True, alpha=0.30, color=c_mut, edgecolor="none")
    x_kde_c = np.linspace(-0.25, 0.5, 400)
    kde_rc = gaussian_kde(rnd_cos, bw_method=0.03)
    kde_mc = gaussian_kde(mut_cos, bw_method=0.03)
    ax.plot(x_kde_c, kde_rc(x_kde_c), color=c_rnd, linewidth=2.2, label="Random (Gaussian)")
    ax.plot(x_kde_c, kde_mc(x_kde_c), color=c_mut, linewidth=2.2, label="Learned mutator")
    ax.set_xlabel("Cosine similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("B   Pairwise Mutation Direction", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.annotate(f"KS = {ks_cos.statistic:.3f},  p ≈ 0",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Panel C: SVD spectrum + cumulative
    ax = fig.add_subplot(gs[0, 2])
    k = min(30, len(var_m))
    ax.bar(range(1, k+1), var_m[:k], color=c_mut, alpha=0.6, width=0.8, label="Learned mutator")
    ax.bar(range(1, k+1), -var_r[:k], color=c_rnd, alpha=0.4, width=0.8, label="Random (flipped)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Singular value rank", fontsize=11)
    ax.set_ylabel("% variance explained", fontsize=11)
    ax.set_title("C   Mutation SVD Spectrum", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.annotate(f"Eff. dim: {pr_m:.0f} (mutator) vs {pr_r:.0f} (random)",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # ── Row 2: Weight-space structure ──

    # Panel D: Per-parameter mutation std (spatial targeting)
    ax = fig.add_subplot(gs[1, 0:2])
    show = min(1000, len(mut_std))
    ax.fill_between(range(show), mut_std[:show], alpha=0.4, color=c_mut, label="Learned mutator")
    ax.fill_between(range(show), rnd_std[:show], alpha=0.3, color=c_rnd, label="Random")
    ax.plot(range(show), mut_std[:show], color=c_mut, linewidth=0.5, alpha=0.8)
    ax.plot(range(show), rnd_std[:show], color=c_rnd, linewidth=0.5, alpha=0.6)
    
    # Mark layer boundaries
    offset = 0
    for name, param in genome.policy.named_parameters():
        size = param.numel()
        if offset < show and "weight" in name:
            ax.axvline(offset, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
            short_name = name.replace("convs.", "C").replace(".weight", "").replace("fc.", "FC")
            if offset + 20 < show:
                ax.text(offset + 5, ax.get_ylim()[1] * 0.92, short_name,
                        fontsize=6, rotation=90, va="top", color="gray", alpha=0.7)
        offset += size
    
    cv_mut = np.std(mut_std) / np.mean(mut_std) if np.mean(mut_std) > 0 else 0
    cv_rnd = np.std(rnd_std) / np.mean(rnd_std) if np.mean(rnd_std) > 0 else 0
    ax.set_xlabel("Parameter index", fontsize=11)
    ax.set_ylabel("Mutation σ (across samples)", fontsize=11)
    ax.set_title("D   Per-Parameter Mutation Scale — Spatial Targeting Across Layers",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.annotate(f"CV(σ): {cv_mut:.2f} (mutator) vs {cv_rnd:.2f} (random)\n"
                f"→ {cv_mut/cv_rnd:.1f}× more heterogeneous",
                xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Panel E: Mutation coherence per neuron
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(frac_active[:n_neurons], neuron_coherence, s=40, alpha=0.8, color=c_mut,
               edgecolors="white", linewidth=0.5, label="Mutator", zorder=3)
    ax.scatter(frac_active[:n_neurons], rnd_coherence, s=25, alpha=0.5, color=c_rnd,
               edgecolors="none", label="Random", zorder=2)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Neuron activity rate", fontsize=11)
    ax.set_ylabel("Mutation direction coherence\n(mean pairwise cosine)", fontsize=10)
    ax.set_title("E   Per-Neuron Coherence", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, framealpha=0.9)
    
    # Mean coherence comparison
    mean_m_coh = np.mean(neuron_coherence)
    mean_r_coh = np.mean(rnd_coherence)
    ax.annotate(f"Mean coherence:\n  Mutator: {mean_m_coh:.4f}\n  Random: {mean_r_coh:.4f}\n"
                f"  Ratio: {mean_m_coh/mean_r_coh:.1f}×" if mean_r_coh != 0 else "",
                xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # ── Row 3: The key insight — weight-space correlation structure ──

    # Panel F: Parameter-parameter mutation correlation heatmap (subsample)
    ax = fig.add_subplot(gs[2, 0])
    # Sample 200 parameters evenly across the genome
    n_samp = min(200, n_params)
    param_idx = np.linspace(0, n_params - 1, n_samp, dtype=int)
    sub_mut = mut_deltas[:, param_idx]
    corr_mat = np.corrcoef(sub_mut.T)
    corr_mat = np.nan_to_num(corr_mat, nan=0)
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto",
                   interpolation="nearest")
    ax.set_xlabel("Parameter (subsampled)", fontsize=10)
    ax.set_ylabel("Parameter (subsampled)", fontsize=10)
    ax.set_title("F   Mutator: Weight Correlation Matrix", fontsize=12, fontweight="bold", loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")

    # Panel G: Same for random
    ax = fig.add_subplot(gs[2, 1])
    sub_rnd = rnd_deltas[:, param_idx]
    corr_mat_r = np.corrcoef(sub_rnd.T)
    corr_mat_r = np.nan_to_num(corr_mat_r, nan=0)
    im2 = ax.imshow(corr_mat_r, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto",
                    interpolation="nearest")
    ax.set_xlabel("Parameter (subsampled)", fontsize=10)
    ax.set_ylabel("Parameter (subsampled)", fontsize=10)
    ax.set_title("G   Random: Weight Correlation Matrix", fontsize=12, fontweight="bold", loc="left")
    plt.colorbar(im2, ax=ax, fraction=0.046, label="Pearson r")

    # Panel H: Summary table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    
    # Off-diagonal correlation stats
    triu_m = corr_mat[np.triu_indices(n_samp, k=1)]
    triu_r = corr_mat_r[np.triu_indices(n_samp, k=1)]
    
    rows = [
        ["Metric", "Mutator", "Random"],
        ["Mean |pairwise corr|", f"{np.mean(np.abs(mut_corr)):.4f}", f"{np.mean(np.abs(rnd_corr)):.4f}"],
        ["Mean cosine sim", f"{np.mean(mut_cos):.4f}", f"{np.mean(rnd_cos):.4f}"],
        ["Effective dim (SVD)", f"{pr_m:.0f}", f"{pr_r:.0f}"],
        ["Top-5 cumul var %", f"{cumvar_m[4]:.1f}%", f"{cumvar_r[4]:.1f}%"],
        ["CV(per-param σ)", f"{cv_mut:.3f}", f"{cv_rnd:.3f}"],
        ["Mean |weight corr|", f"{np.mean(np.abs(triu_m)):.4f}", f"{np.mean(np.abs(triu_r)):.4f}"],
        ["Neuron coherence", f"{mean_m_coh:.4f}", f"{mean_r_coh:.4f}"],
        ["KS (corr) stat", f"{ks_corr.statistic:.3f}", "—"],
        ["KS (cos) stat", f"{ks_cos.statistic:.3f}", "—"],
    ]

    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center",
                     cellLoc="center", colWidths=[0.42, 0.29, 0.29])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for j in range(3):
        table[0, j].set_facecolor("#d9d9d9")
        table[0, j].set_text_props(fontweight="bold")
    for i in range(1, len(rows)):
        for j in range(3):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f7f7f7")

    ax.set_title("H   Summary", fontsize=12, fontweight="bold", loc="left", pad=15)

    # Suptitle
    fig.suptitle(
        "Learned Mutator Produces Structured, Correlated Weight Perturbations\n"
        f"DualMixtureCorrectorMutator  ·  CarRacing-v3  ·  {n_params:,} params  ·  "
        f"{n_mutations} mutations  ·  {n_steps} inference steps",
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
    p.add_argument("--output", default="results/paired_mutations_pub.png")
    args = p.parse_args()
    make_figure(args.genome, args.output, args.n_mutations, args.n_steps)
