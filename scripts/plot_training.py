#!/usr/bin/env python3
"""Standardized training plot generator for all environments."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Consistent style
COLORS = {
    'best': '#2ecc71',
    'mean': '#3498db',
    'worst': '#e74c3c',
    'fill': '#3498db',
    'fidelity_mean': '#9b59b6',
    'fidelity_best': '#e67e22',
    'drift_mean': '#8e44ad',
    'drift_max': '#c0392b',
    'species': '#1abc9c',
    'entropy': '#e74c3c',
    'compat': '#f39c12',
}

SOLVE_THRESHOLDS = {
    'CartPole-v1': 500,
    'LunarLander-v3': 200,
}


def rolling_mean(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training(json_path, output_path):
    with open(json_path) as f:
        d = json.load(f)
    
    h = d['history']
    env = d['env']
    gens = np.arange(len(h['best']))
    best_ever = max(h['best'])
    best_gen = int(np.argmax(h['best']))
    
    # Title
    title_parts = [env]
    if d.get('mutator'): title_parts.append(d['mutator'])
    if d.get('speciation'): title_parts.append('speciation')
    title_parts.append(f"pop {d.get('pop_size', '?')}")
    title_parts.append(f"seed {d.get('seed', '?')}")
    title_parts.append(f"{d.get('generations', len(gens))} gens")
    if d.get('elapsed_sec'):
        hrs = d['elapsed_sec'] / 3600
        title_parts.append(f"{hrs:.1f}h" if hrs >= 1 else f"{d['elapsed_sec']/60:.0f}m")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(' · '.join(title_parts), fontsize=14, fontweight='bold')
    
    # === [0,0] Fitness ===
    ax = axes[0, 0]
    ax.plot(gens, h['best'], color=COLORS['best'], linewidth=2, label='Best', zorder=3)
    ax.plot(gens, h['mean'], color=COLORS['mean'], linewidth=1.5, label='Mean')
    ax.fill_between(gens, h['worst'], h['best'], alpha=0.08, color='gray')
    ax.plot(gens, h['worst'], color=COLORS['worst'], linewidth=0.6, alpha=0.5, label='Worst')
    ax.annotate(f'{best_ever:.1f} @ gen {best_gen}', xy=(best_gen, best_ever),
                xytext=(best_gen + len(gens)*0.05, best_ever),
                fontsize=9, color=COLORS['best'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['best'], lw=1.2))
    if env in SOLVE_THRESHOLDS:
        ax.axhline(SOLVE_THRESHOLDS[env], color='green', linestyle='--', alpha=0.5, label=f'Solve ({SOLVE_THRESHOLDS[env]})')
    ax.set_title('Fitness', fontsize=11, fontweight='bold')
    ax.set_xlabel('Generation'); ax.set_ylabel('Reward')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # === [0,1] Rolling Average ===
    ax = axes[0, 1]
    window = max(5, len(gens) // 20)
    rm_best = rolling_mean(h['best'], window)
    rm_mean = rolling_mean(h['mean'], window)
    x_rm = np.arange(window-1, len(gens))[:len(rm_best)]
    ax.plot(x_rm, rm_best, color=COLORS['best'], linewidth=2, label=f'Best (rolling {window})')
    ax.plot(x_rm, rm_mean, color=COLORS['mean'], linewidth=1.5, label=f'Mean (rolling {window})')
    if env in SOLVE_THRESHOLDS:
        ax.axhline(SOLVE_THRESHOLDS[env], color='green', linestyle='--', alpha=0.5)
    ax.set_title(f'Rolling Average (window={window})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Generation'); ax.set_ylabel('Reward')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # === [0,2] Self-Replication Fidelity ===
    ax = axes[0, 2]
    if 'mean_fidelity' in h and any(v != 0 for v in h['mean_fidelity']):
        ax.plot(gens, h['mean_fidelity'], color=COLORS['fidelity_mean'], linewidth=1.5, label='Mean Fidelity')
    if 'best_fidelity' in h and any(v != 0 for v in h['best_fidelity']):
        ax.plot(gens, h['best_fidelity'], color=COLORS['fidelity_best'], linewidth=1.2, alpha=0.7, label='Best Genome Fidelity')
    ax.set_title('Self-Replication Fidelity', fontsize=11, fontweight='bold')
    ax.set_xlabel('Generation'); ax.set_ylabel('Fidelity (cosine sim)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # === [1,0] Mutator Weight Drift ===
    ax = axes[1, 0]
    if 'mean_mutator_drift' in h:
        ax.plot(gens, h['mean_mutator_drift'], color=COLORS['drift_mean'], linewidth=1.5, label='Mean Drift')
    if 'max_mutator_drift' in h:
        ax.plot(gens, h['max_mutator_drift'], color=COLORS['drift_max'], linewidth=1, alpha=0.6, label='Max Drift')
    ax.set_title('Mutator Weight Drift', fontsize=11, fontweight='bold')
    ax.set_xlabel('Generation'); ax.set_ylabel('L2 distance from init')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # === [1,1] Speciation ===
    ax = axes[1, 1]
    if 'num_species' in h:
        ax.plot(gens, h['num_species'], color=COLORS['species'], linewidth=2, label='Species')
        ax.set_ylabel('Species Count', color=COLORS['species'])
    if 'species_entropy' in h:
        ax2 = ax.twinx()
        ax2.plot(gens, h['species_entropy'], color=COLORS['entropy'], linewidth=1.2, alpha=0.7, label='Entropy')
        ax2.set_ylabel('Shannon Entropy', color=COLORS['entropy'])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=8)
    if 'crossover_compat_rate' in h:
        # Show as subtle background if available
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 50))
        ax3.plot(gens, h['crossover_compat_rate'], color=COLORS['compat'], linewidth=0.8, alpha=0.5, linestyle='--', label='Compat Rate')
        ax3.set_ylabel('Compat Rate', color=COLORS['compat'], fontsize=8)
        ax3.set_ylim(0, 1.1)
    ax.set_title('Speciation', fontsize=11, fontweight='bold')
    ax.set_xlabel('Generation'); ax.grid(True, alpha=0.2)
    
    # === [1,2] Architecture / Run Summary ===
    ax = axes[1, 2]
    has_arch = ('mean_layers' in h and len(h['mean_layers']) > 0 and any(v != 0 for v in h.get('mean_neurons', [])))
    
    if has_arch:
        ax.plot(gens[:len(h['mean_layers'])], h['mean_layers'], color='#16a085', linewidth=1.5, label='Mean Layers')
        ax.set_ylabel('Layers', color='#16a085')
        if 'mean_neurons' in h and len(h['mean_neurons']) > 0:
            ax2 = ax.twinx()
            ax2.plot(gens[:len(h['mean_neurons'])], h['mean_neurons'], color='#2c3e50', linewidth=1.5, label='Mean Neurons')
            ax2.set_ylabel('Neurons', color='#2c3e50')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1+lines2, labels1+labels2, fontsize=8)
        ax.set_title('Architecture Evolution', fontsize=11, fontweight='bold')
    else:
        # Run summary text box
        ax.axis('off')
        summary_lines = [
            f"Environment: {env}",
            f"Best Reward: {best_ever:.1f} (gen {best_gen})",
            f"Final Mean: {h['mean'][-1]:.1f}",
            f"Population: {d.get('pop_size', '?')}",
            f"Generations: {d.get('generations', len(gens))}",
            f"Mutator: {d.get('mutator', '?')}",
            f"Speciation: {d.get('speciation', False)}",
        ]
        if d.get('elapsed_sec'):
            summary_lines.append(f"Time: {d['elapsed_sec']/3600:.1f}h")
        if 'mean_fidelity' in h:
            summary_lines.append(f"Final Fidelity: {h['mean_fidelity'][-1]:.4f}")
        if 'mean_policy_params' in h and len(h['mean_policy_params']) > 0:
            summary_lines.append(f"Policy Params: {int(h['mean_policy_params'][-1]):,}")
        
        text = '\n'.join(summary_lines)
        ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        ax.set_title('Run Summary', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Generation'); ax.grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path}')


if __name__ == '__main__':
    runs = [
        ('results/cartpole_dm_flex_s45_50g/CartPole-v1_dualmixture_spec_s45.json',
         'results/cartpole_dm_flex_s45_50g/training_plot.png'),
        ('results/acrobot_dm_flex_s45_300g/Acrobot-v1_dualmixture_spec_s45.json',
         'results/acrobot_dm_flex_s45_300g/training_plot.png'),
        ('results/pendulum_dm_flex_s45_1000g/Pendulum-v1_dualmixture_spec_s45.json',
         'results/pendulum_dm_flex_s45_1000g/training_plot.png'),
        ('results/lunar_s45_300g_fleet/LunarLander-v3_dualmixture_spec_s45.json',
         'results/lunar_s45_300g_fleet/training_plot.png'),
        ('results/carracing_dm_s45_100g_fleet/CarRacing-v3_dualmixture_spec_s45.json',
         'results/carracing_dm_s45_100g_fleet/training_plot.png'),
    ]
    
    for json_path, output_path in runs:
        if os.path.exists(json_path):
            plot_training(json_path, output_path)
        else:
            print(f'SKIP: {json_path} not found')
    
    print('\nAll done!')
