# Neural Mutator — Self-Replicating Neuroevolution

Evolve neural networks that **mutate their own weights**. Each genome contains a **policy** (plays the environment), a **mutator** (rewrites the entire genome, including itself), and a **compatibility network** (decides who can crossover). Natural selection is the only filter — mutators that destroy themselves die, those that improve themselves thrive.

## Architecture

![Architecture](docs/plots/architecture.png)

The system has five key components:

- **(a) Genome** — Three co-evolved networks: policy θ_p, mutator θ_m, and compatibility θ_c
- **(b) DualMixture Reproduction** — 80% learned corrections via encoder→corrector pipeline, 20% Gaussian escape hatch (ratio evolved independently across environments, converges to ~20% universally)
- **(c) Evolutionary Loop** — Evaluate → select → speciate → reproduce → structural mutation
- **(d) Learned Speciation** — Compatibility network scores pairwise genome similarity to form breeding groups
- **(e) Learned Crossover** — Mutator network sees both parents and decides how to combine them

The mutator is **self-referential**: since θ_m ⊂ θ, the mutator rewrites its own weights through its own output.

## Results

All runs use the DualMixture mutator with flexible architecture and learned speciation.

### CartPole-v1

**Solved in 2 generations.** Best: 500 (maximum). Mean converges to ~484 by gen 50.

| Training | Gameplay (with activations) |
|---|---|
| ![CartPole Training](results/cartpole_dm_flex_s45_50g/training_plot.png) | <video src="results/cartpole_dm_flex_s45_50g/best_gameplay.mp4" width="400"> |

- **Pop:** 80 · **Gens:** 50 · **Episodes:** 5 · **Seed:** 45
- **Final architecture:** 4→64→2 (Tanh), 386 params
- **Mutator:** 8,374 params (21.7× larger than the policy it evolves)

### LunarLander-v3

**Best: 291.06** — well above the 200 solve threshold. Fleet-trained across 4 nodes (24 parallel workers).

| Training | Gameplay (with activations) |
|---|---|
| ![LunarLander Training](results/lunar_s45_300g_fleet/training_plot.png) | <video src="results/lunar_s45_300g_fleet/best_gameplay.mp4" width="400"> |

- **Pop:** 160 · **Gens:** 300 · **Episodes:** 10 · **Seed:** 45 · **Fleet:** 4 nodes × 6 workers
- **Final architecture:** 8→51→4 (Tanh), 667 params
- **Weight analysis:** Angular velocity and angle dominate input importance — the network learned that orientation control is key for landing

### Acrobot-v1

**Best: -64.0** (previous baseline: -70.7). Swing-up solved efficiently.

| Training | Gameplay (with activations) |
|---|---|
| ![Acrobot Training](results/acrobot_dm_flex_s45_300g/training_plot.png) | <video src="results/acrobot_dm_flex_s45_300g/best_gameplay.mp4" width="400"> |

- **Pop:** 80 · **Gens:** 300 · **Episodes:** 10 · **Seed:** 45
- **Final architecture:** 6→64→64→3 (Tanh), 2 layers, 112 neurons
- **Fidelity:** Climbed to 0.038 — mutator learned stable self-replication

### Pendulum-v1

**Best: -112.3** with discrete action mapping (continuous control is inherently harder for neuroevolution). Architecture self-simplified from 128→48 neurons over 1000 generations.

| Training | Gameplay (with activations) |
|---|---|
| ![Pendulum Training](results/pendulum_dm_flex_s45_1000g/training_plot.png) | <video src="results/pendulum_dm_flex_s45_1000g/best_gameplay.mp4" width="400"> |

- **Pop:** 80 · **Gens:** 1000 · **Episodes:** 10 · **Seed:** 45
- **Final architecture:** 3→32→32→1 (Tanh), 2 layers — evolved continuous output
- **Fidelity:** 0.00 → 0.23 over 1000 gens — mutator never stopped improving self-replication
- **Best test episode:** -10.01 (near-perfect upright balance)

### CarRacing-v3

**Best: 808.9** — first vision-based environment. CNN policy processes raw pixels. Fleet-trained across 4 nodes.

| Training | Gameplay (with activations) |
|---|---|
| ![CarRacing Training](results/carracing_dm_s45_100g_fleet/CarRacing-v3_dualmixture_spec_s45.png) | <video src="results/carracing_dm_s45_100g_fleet/best_gameplay.mp4" width="400"> |

- **Pop:** 30 · **Gens:** 100 · **Episodes:** 3 · **Seed:** 45 · **Fleet:** 4 nodes
- **Final architecture:** CNN — 3 conv layers (8→16→16→8 channels) + FC (288→32→3), 17,971 params
- **Mutator:** 8,374 params (0.47× the policy — first time mutator is **smaller** than the policy it evolves)
- **Training time:** ~13.2 hours
- **Multi-seed eval:** seed 42 = 769, seed 7 = 242, seed 123 = 412 (avg ~474)
- **Note:** Random track layouts each evaluation force genuine generalization — no memorization possible

### Summary

| Environment | Best Reward | Architecture | Gens | Status |
|-------------|-----------|--------------|------|--------|
| CartPole-v1 | **500** | 4→64→2, 1 layer | 50 | ✅ Solved (gen 2) |
| LunarLander-v3 | **291** | 8→51→4, 1 layer | 300 | ✅ Solved |
| Acrobot-v1 | **-64** | 6→64→64→3, 2 layers | 300 | ✅ Beat baseline |
| Pendulum-v1 | **-112** | 3→32→32→1, 2 layers | 1000 | 📈 Improving |
| CarRacing-v3 | **809** | CNN 3conv+FC, 17,971 params | 100 | 🏎️ First vision env |

### Cross-Environment Findings

The DualMixture mutator adapts its learned parameters per environment:

- **p_gauss converges to ~20% across all environments** — this ratio appears to be a universal sweet spot
- **Correction scales specialize**: CartPole (0.033) > Pendulum (0.028) > LunarLander (0.020) — harder problems demand finer precision
- **Mutator-to-policy ratio varies dramatically**: 21.7× for CartPole (386-param policy) down to 0.47× for CarRacing (17,971-param policy) — larger policies contain enough structure that the mutator doesn't need to be bigger
- **Flexible architecture works**: networks self-compress to minimal viable size (CartPole: 128→64, Pendulum: 128→48)

## Mutator Analysis

Visualizing what the mutator networks actually learn across environments.

### Weight Distributions

The mutator's learned weights reveal how each environment shapes the mutation strategy:

| | |
|---|---|
| ![CartPole Weights](docs/plots/mutator_weights_cartpole_annotated.png) | ![Acrobot Weights](docs/plots/mutator_weights_acrobot_annotated.png) |
| ![Pendulum Weights](docs/plots/mutator_weights_pendulum_annotated.png) | ![LunarLander Weights](docs/plots/mutator_weights_lunarlander_annotated.png) |
| ![CarRacing Weights](docs/plots/mutator_weights_carracing_annotated.png) | |

### Mutation Deltas Across Environments

How the magnitude and distribution of mutations compare across all five environments:

![Mutation Deltas](docs/plots/mutation_delta_all_envs.png)

## Mutator Architecture

The **DualMixtureCorrectorMutator** (8,374 params) is the sole mutator architecture. It processes policy weights in 64-element chunks through a learned encoder→corrector pipeline:

1. **Encoder** (64→16): Compresses each weight chunk to a latent representation
2. **Corrector** (48→64→64): Compares encoded chunk against a learned reference vector, outputs targeted weight deltas (clamped to ±0.1)
3. **Gaussian escape** (~20%): With evolved probability, adds pure Gaussian noise to prevent fixed-point collapse
4. **Dual-head**: Separate scales for policy weights (larger mutations) vs mutator self-weights (conservative)

## Usage

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install torch gymnasium matplotlib numpy

# Basic run
python -m src.train --env CartPole-v1 --generations 100

# Full featured run
python -m src.train \
    --env LunarLander-v3 \
    --mutator dualmixture \
    --flex --speciation \
    --pop-size 80 \
    --generations 300 \
    --episodes 10 \
    --workers 6

# Fleet training (distributed across nodes)
python -m src.train \
    --env LunarLander-v3 \
    --mutator dualmixture \
    --flex --speciation \
    --pop-size 160 --generations 300 --episodes 10 \
    --fleet --fleet-port 5611 --fleet-workers 4
# Then on each worker node:
python -m fleet.worker --host <manager-ip> --port 5611 --workers 6 --name node1
```

## Key Properties

- **Self-referential**: The mutator modifies itself through its own output
- **Evolvable variation operator**: Natural selection acts on the mutation strategy, not just the policy
- **Learned recombination**: Crossover is performed by the mutator network, not fixed rules
- **Adaptive precision**: Correction scales evolve per-environment — tight for hard, loose for easy
- **Parsimony pressure**: Flex architecture + selection drives evolution toward minimal networks

## Related Work

- Neural Network Quine (Chang & Lipson, 2018) — self-replicating networks
- HyperNEAT (Stanley et al., 2009) — CPPNs for indirect weight encoding
- Adaptive RL through Evolving Self-Modifying NNs (Schmidgall, 2020)
- Self-Referential Meta Learning (Kirsch & Schmidhuber, 2022)

## License

MIT
