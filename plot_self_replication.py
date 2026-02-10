#!/usr/bin/env python3
"""Generate comparison plots for self-replication experiments."""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "results/self_replication"

def load_results():
    results = {}
    for f in sorted(glob.glob(os.path.join(OUTDIR, "*.json"))):
        with open(f) as fh:
            data = json.load(fh)
        key = f"{data['mutator']}_ql{data.get('quine_lambda', 0.0)}"
        results[key] = data
    return results

def main():
    results = load_results()
    if not results:
        print("No results found!")
        return

    # 1. Fitness comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for key, data in results.items():
        h = data['history']
        ax.plot(h['best'], label=f"{key} (best)")
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Fitness Curves (Best)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Self-replication fidelity
    ax = axes[1]
    for key, data in results.items():
        h = data['history']
        if 'mean_fidelity' in h:
            ax.plot(h['mean_fidelity'], label=key)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Fidelity (L2, lower=better)')
    ax.set_title('Self-Replication Fidelity')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Mutator drift
    ax = axes[2]
    for key, data in results.items():
        h = data['history']
        if 'mean_mutator_drift' in h:
            ax.plot(h['mean_mutator_drift'], label=key)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Mutator Drift (L2)')
    ax.set_title('Mutator Weight Drift')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTDIR, "comparison.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()
