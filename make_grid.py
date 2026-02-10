#!/usr/bin/env python3
"""Generate comparison grid plot from all run results."""
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

ENVS = ['CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0', 'Pendulum-v1', 'LunarLander-v3']
MUTATORS = ['chunk', 'transformer', 'cppn', 'gaussian']
COLORS = {'chunk': '#e74c3c', 'transformer': '#3498db', 'cppn': '#2ecc71', 'gaussian': '#95a5a6'}
RESULTS_DIR = 'results'

fig, axes = plt.subplots(len(ENVS), 1, figsize=(12, 4 * len(ENVS)), sharex=False)
if len(ENVS) == 1:
    axes = [axes]

for i, env in enumerate(ENVS):
    ax = axes[i]
    for mut in MUTATORS:
        json_path = os.path.join(RESULTS_DIR, f"{env}_{mut}_s42.json")
        if not os.path.exists(json_path):
            print(f"Missing: {json_path}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        history = data['history']
        gens = range(len(history['best']))
        ax.plot(gens, history['best'], label=f'{mut} (best)', color=COLORS[mut], linewidth=2)
        ax.plot(gens, history['mean'], label=f'{mut} (mean)', color=COLORS[mut], linewidth=1, linestyle='--', alpha=0.7)
    
    ax.set_title(env, fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle('Neural Mutator Comparison: Mutator Types × Environments', fontsize=16, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'comparison_grid.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR}/comparison_grid.png")
plt.close()
