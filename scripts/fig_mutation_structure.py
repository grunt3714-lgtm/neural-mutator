#!/usr/bin/env python3
"""Publication-quality figure: Learned mutator produces structured (non-random) mutations.

Generates a multi-panel figure comparing pairwise correlation distributions,
cosine similarity distributions, and SVD spectra between DualMixtureCorrectorMutator
and isotropic Gaussian noise applied to the same genome.

Usage:
    python scripts/fig_mutation_structure.py [--genome PATH] [--n-mutations 100] [--output PATH]
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
from src.genome import Genome


def collect_mutations(genome, n=100):
    """Collect n mutation delta vectors from genome.reproduce()."""
    parent_flat = genome.get_flat_weights().clone()
    deltas = []
    for _ in range(n):
        genome.set_flat_weights(parent_flat.clone())
        child = genome.reproduce()
        delta = child.get_flat_weights() - parent_flat
        deltas.append(delta.detach().numpy())
    genome.set_flat_weights(parent_flat)  # restore
    return np.array(deltas)


def random_deltas_matched(mutator_deltas):
    """Generate Gaussian noise matched to mutator's per-element std."""
    n, d = mutator_deltas.shape
    elem_std = np.std(mutator_deltas, axis=0)
    global_std = np.mean(elem_std[elem_std > 0]) if np.any(elem_std > 0) else 0.01
    return np.random.randn(n, d) * global_std


def pairwise_correlations(deltas, max_pairs=2000):
    """Compute Pearson correlations between random pairs of mutation vectors."""
    n = deltas.shape[0]
    idx = np.random.choice(n, size=(max_pairs, 2), replace=True)
    idx = idx[idx[:, 0] != idx[:, 1]]
    corrs = []
    for i, j in idx:
        r = np.corrcoef(deltas[i], deltas[j])[0, 1]
        if np.isfinite(r):
            corrs.append(r)
    return np.array(corrs)


def pairwise_cosine(deltas, max_pairs=2000):
    """Compute cosine similarities between random pairs of mutation vectors."""
    n = deltas.shape[0]
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = deltas / norms
    idx = np.random.choice(n, size=(max_pairs, 2), replace=True)
    idx = idx[idx[:, 0] != idx[:, 1]]
    cos = np.array([np.dot(normed[i], normed[j]) for i, j in idx])
    return cos[np.isfinite(cos)]


def make_figure(genome_path, n_mutations=100, output_path="results/mutation_structure_fig.png"):
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Loading genome: {genome_path}")
    genome = Genome.load(genome_path)
    n_params = genome.get_flat_weights().numel()
    print(f"  Parameters: {n_params:,}  |  Fitness: {genome.fitness:.1f}")

    print(f"Collecting {n_mutations} mutator mutations...")
    mut_deltas = collect_mutations(genome, n_mutations)
    print(f"Generating matched random baseline...")
    rnd_deltas = random_deltas_matched(mut_deltas)

    print("Computing pairwise statistics...")
    mut_corr = pairwise_correlations(mut_deltas)
    rnd_corr = pairwise_correlations(rnd_deltas)
    mut_cos = pairwise_cosine(mut_deltas)
    rnd_cos = pairwise_cosine(rnd_deltas)

    # SVD
    print("Computing SVD...")
    U_m, S_m, _ = np.linalg.svd(mut_deltas, full_matrices=False)
    U_r, S_r, _ = np.linalg.svd(rnd_deltas, full_matrices=False)
    var_m = (S_m ** 2) / np.sum(S_m ** 2) * 100
    var_r = (S_r ** 2) / np.sum(S_r ** 2) * 100
    cumvar_m = np.cumsum(var_m)
    cumvar_r = np.cumsum(var_r)

    # Statistical tests
    ks_corr = stats.ks_2samp(mut_corr, rnd_corr)
    ks_cos = stats.ks_2samp(mut_cos, rnd_cos)

    # Effective dimensionality (participation ratio)
    pr_m = (np.sum(S_m ** 2)) ** 2 / np.sum(S_m ** 4)
    pr_r = (np.sum(S_r ** 2)) ** 2 / np.sum(S_r ** 4)

    # --- Figure ---
    fig = plt.figure(figsize=(14, 10), dpi=200)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.07, right=0.97, top=0.91, bottom=0.08)

    # Color scheme
    c_mut = "#2166ac"   # blue
    c_rnd = "#b2182b"   # red
    alpha_hist = 0.65

    # ── Panel A: Pairwise Pearson Correlation ──
    ax_a = fig.add_subplot(gs[0, 0])
    bins_corr = np.linspace(-0.4, 0.6, 80)
    ax_a.hist(rnd_corr, bins=bins_corr, density=True, alpha=0.35,
              color=c_rnd, edgecolor="none")
    ax_a.hist(mut_corr, bins=bins_corr, density=True, alpha=0.35,
              color=c_mut, edgecolor="none")
    # KDE overlays
    from scipy.stats import gaussian_kde
    x_corr = np.linspace(-0.4, 0.6, 300)
    kde_rnd = gaussian_kde(rnd_corr, bw_method=0.04)
    kde_mut = gaussian_kde(mut_corr, bw_method=0.04)
    ax_a.plot(x_corr, kde_rnd(x_corr), color=c_rnd, linewidth=2, label="Random")
    ax_a.plot(x_corr, kde_mut(x_corr), color=c_mut, linewidth=2, label="Learned mutator")
    ax_a.set_xlabel("Pearson correlation", fontsize=11)
    ax_a.set_ylabel("Density", fontsize=11)
    ax_a.set_title("A   Pairwise Correlation", fontsize=12, fontweight="bold", loc="left")
    ax_a.legend(fontsize=9, framealpha=0.9)
    ax_a.annotate(f"KS = {ks_corr.statistic:.3f}\np < {max(ks_corr.pvalue, 1e-300):.1e}",
                  xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                  fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # ── Panel B: Pairwise Cosine Similarity ──
    ax_b = fig.add_subplot(gs[0, 1])
    bins_cos = np.linspace(-0.3, 0.5, 80)
    ax_b.hist(rnd_cos, bins=bins_cos, density=True, alpha=0.35,
              color=c_rnd, edgecolor="none")
    ax_b.hist(mut_cos, bins=bins_cos, density=True, alpha=0.35,
              color=c_mut, edgecolor="none")
    x_cos = np.linspace(-0.3, 0.5, 300)
    kde_rnd_c = gaussian_kde(rnd_cos, bw_method=0.04)
    kde_mut_c = gaussian_kde(mut_cos, bw_method=0.04)
    ax_b.plot(x_cos, kde_rnd_c(x_cos), color=c_rnd, linewidth=2, label="Random")
    ax_b.plot(x_cos, kde_mut_c(x_cos), color=c_mut, linewidth=2, label="Learned mutator")
    ax_b.set_xlabel("Cosine similarity", fontsize=11)
    ax_b.set_ylabel("Density", fontsize=11)
    ax_b.set_title("B   Pairwise Cosine Similarity", fontsize=12, fontweight="bold", loc="left")
    ax_b.legend(fontsize=9, framealpha=0.9)
    ax_b.annotate(f"KS = {ks_cos.statistic:.3f}\np < {max(ks_cos.pvalue, 1e-300):.1e}",
                  xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                  fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # ── Panel C: SVD Spectrum ──
    ax_c = fig.add_subplot(gs[0, 2])
    k = min(30, len(var_m))
    ax_c.plot(range(1, k+1), var_m[:k], "o-", color=c_mut, markersize=4, linewidth=1.5,
              label="Learned mutator")
    ax_c.plot(range(1, k+1), var_r[:k], "s--", color=c_rnd, markersize=4, linewidth=1.5,
              label="Random")
    ax_c.set_xlabel("Singular value rank", fontsize=11)
    ax_c.set_ylabel("% variance explained", fontsize=11)
    ax_c.set_title("C   SVD Spectrum", fontsize=12, fontweight="bold", loc="left")
    ax_c.legend(fontsize=9, framealpha=0.9)
    ax_c.annotate(f"Eff. dim: {pr_m:.1f} (mut) vs {pr_r:.1f} (rnd)",
                  xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                  fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # ── Panel D: Cumulative variance ──
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.plot(range(1, k+1), cumvar_m[:k], "o-", color=c_mut, markersize=3, linewidth=1.5,
              label="Learned mutator")
    ax_d.plot(range(1, k+1), cumvar_r[:k], "s--", color=c_rnd, markersize=3, linewidth=1.5,
              label="Random")
    ax_d.axhline(90, color="gray", linestyle=":", alpha=0.5)
    ax_d.text(k, 90, "90%", ha="right", va="bottom", fontsize=8, color="gray")
    ax_d.set_xlabel("Number of components", fontsize=11)
    ax_d.set_ylabel("Cumulative variance (%)", fontsize=11)
    ax_d.set_title("D   Cumulative SVD Variance", fontsize=12, fontweight="bold", loc="left")
    ax_d.legend(fontsize=9, framealpha=0.9)

    # ── Panel E: Per-element mutation std (spatial structure) ──
    ax_e = fig.add_subplot(gs[1, 1])
    mut_std = np.std(mut_deltas, axis=0)
    rnd_std = np.std(rnd_deltas, axis=0)
    # Show first 500 params for readability
    show = min(500, len(mut_std))
    ax_e.plot(range(show), mut_std[:show], color=c_mut, alpha=0.7, linewidth=0.6,
              label="Learned mutator")
    ax_e.plot(range(show), rnd_std[:show], color=c_rnd, alpha=0.5, linewidth=0.6,
              label="Random")
    ax_e.set_xlabel("Parameter index", fontsize=11)
    ax_e.set_ylabel("Mutation σ", fontsize=11)
    ax_e.set_title("E   Per-Parameter Mutation Scale", fontsize=12, fontweight="bold", loc="left")
    ax_e.legend(fontsize=9, framealpha=0.9)
    # Coefficient of variation
    cv_mut = np.std(mut_std) / np.mean(mut_std) if np.mean(mut_std) > 0 else 0
    cv_rnd = np.std(rnd_std) / np.mean(rnd_std) if np.mean(rnd_std) > 0 else 0
    ax_e.annotate(f"CV(σ): {cv_mut:.2f} (mut) vs {cv_rnd:.2f} (rnd)",
                  xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                  fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # ── Panel F: Summary statistics table ──
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    rows = [
        ["Metric", "Mutator", "Random"],
        ["Mean |corr|", f"{np.mean(np.abs(mut_corr)):.4f}", f"{np.mean(np.abs(rnd_corr)):.4f}"],
        ["Mean cos sim", f"{np.mean(mut_cos):.4f}", f"{np.mean(rnd_cos):.4f}"],
        ["Eff. dimensionality", f"{pr_m:.1f}", f"{pr_r:.1f}"],
        ["Top-1 SV var %", f"{var_m[0]:.1f}%", f"{var_r[0]:.1f}%"],
        ["Top-5 cumul var %", f"{cumvar_m[4]:.1f}%", f"{cumvar_r[4]:.1f}%"],
        ["CV(per-param σ)", f"{cv_mut:.3f}", f"{cv_rnd:.3f}"],
        ["KS (corr) p-val", f"{ks_corr.pvalue:.1e}", "—"],
        ["KS (cos) p-val", f"{ks_cos.pvalue:.1e}", "—"],
    ]

    table = ax_f.table(cellText=rows[1:], colLabels=rows[0], loc="center",
                       cellLoc="center", colWidths=[0.45, 0.28, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor("#d9d9d9")
        table[0, j].set_text_props(fontweight="bold")
    # Alternate row shading
    for i in range(1, len(rows)):
        for j in range(3):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f7f7f7")

    ax_f.set_title("F   Summary Statistics", fontsize=12, fontweight="bold", loc="left",
                    pad=15)

    # ── Suptitle ──
    fig.suptitle(
        "Learned Mutator Produces Structured, Non-Random Mutations\n"
        f"DualMixtureCorrectorMutator  ·  CarRacing-v3  ·  {n_params:,} parameters  ·  {n_mutations} samples",
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
    p.add_argument("--output", default="results/mutation_structure_fig.png")
    args = p.parse_args()
    make_figure(args.genome, args.n_mutations, args.output)
