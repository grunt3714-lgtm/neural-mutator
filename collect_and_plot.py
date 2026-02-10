#!/usr/bin/env python3
"""
Collect results from log files (gateway + remote nodes) and generate comparison grid.
Parses generation lines from logs to extract fitness history.
"""
import json
import os
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

ENVS = ['CartPole-v1', 'Acrobot-v1', 'Pendulum-v1', 'LunarLander-v3']
# Excluding MountainCarContinuous-v0 (only reached gen 4)
MUTATORS = ['chunk', 'transformer', 'cppn', 'gaussian']
COLORS = {'chunk': '#e74c3c', 'transformer': '#3498db', 'cppn': '#2ecc71', 'gaussian': '#7f8c8d'}

NODES = {
    'Acrobot-v1': '192.168.1.95',
    'Pendulum-v1': '192.168.1.112',
    'LunarLander-v3': '192.168.1.100',
}

def parse_log(log_text):
    """Parse generation lines from log to extract best/mean/worst."""
    history = {'best': [], 'mean': [], 'worst': []}
    for line in log_text.strip().split('\n'):
        m = re.match(r'Gen\s+(\d+)\s+\|\s+Best:\s+([-\d.]+)\s+\|\s+Mean:\s+([-\d.]+)\s+\|\s+Worst:\s+([-\d.]+)', line.strip())
        if m:
            history['best'].append(float(m.group(2)))
            history['mean'].append(float(m.group(3)))
            history['worst'].append(float(m.group(4)))
    return history

def get_log(env, mut):
    """Get log text from gateway or remote node."""
    if env == 'CartPole-v1':
        path = f'results/logs/{env}_{mut}.log'
        if os.path.exists(path):
            return open(path).read()
    elif env in NODES:
        node = NODES[env]
        try:
            result = subprocess.run(
                ['ssh', f'grunt@{node}', f'cat /tmp/neural-mutator/results/logs/{env}_{mut}.log'],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout
        except:
            pass
    return ''

# Collect all results
all_results = {}
for env in ENVS:
    for mut in MUTATORS:
        # First check if JSON already exists locally
        json_path = os.path.join(RESULTS_DIR, f'{env}_{mut}_s42.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            all_results[(env, mut)] = data['history']
            print(f"  {env}/{mut}: {len(data['history']['best'])} gens (from JSON)")
            continue
        
        # Parse from log
        log_text = get_log(env, mut)
        history = parse_log(log_text)
        if history['best']:
            all_results[(env, mut)] = history
            # Save as JSON
            result = {
                'env': env, 'mutator': mut, 'pop_size': 30,
                'generations': len(history['best']), 'seed': 42,
                'history': history,
            }
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  {env}/{mut}: {len(history['best'])} gens (from log)")
        else:
            print(f"  {env}/{mut}: NO DATA")

# Generate comparison grid
fig, axes = plt.subplots(len(ENVS), 1, figsize=(14, 4.5 * len(ENVS)))
if len(ENVS) == 1:
    axes = [axes]

for i, env in enumerate(ENVS):
    ax = axes[i]
    for mut in MUTATORS:
        key = (env, mut)
        if key not in all_results:
            continue
        h = all_results[key]
        gens = range(len(h['best']))
        ax.plot(gens, h['best'], label=f'{mut} (best)', color=COLORS[mut], linewidth=2)
        ax.plot(gens, h['mean'], label=f'{mut} (mean)', color=COLORS[mut],
                linewidth=1, linestyle='--', alpha=0.6)
    
    ax.set_title(env, fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle('Neural Mutator: Fitness Curves by Mutator Type',
             fontsize=16, fontweight='bold', y=1.01)
fig.tight_layout()
out = os.path.join(RESULTS_DIR, 'comparison_grid.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()

# Print summary
print("\n=== SUMMARY ===")
for env in ENVS:
    print(f"\n{env}:")
    for mut in MUTATORS:
        key = (env, mut)
        if key in all_results:
            h = all_results[key]
            print(f"  {mut:12s}: {len(h['best']):3d} gens, best={max(h['best']):.2f}, final_mean={h['mean'][-1]:.2f}")
        else:
            print(f"  {mut:12s}: NO DATA")
