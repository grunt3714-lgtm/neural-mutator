# Neural Mutator — Self-Modifying Evolutionary Networks

Evolve neural networks that mutate their own weights. Each genome contains a **policy** (plays the environment) and a **mutator** (rewrites the genome). The mutator must reproduce both the policy and itself.

## Architecture

```
Genome = [Policy θ_p, Mutator θ_m]

Evaluation:
  reward = Environment(Policy(θ_p))

Reproduction (mutation):
  θ' = Mutator(θ_m)(concat(θ_p, θ_m))
  → θ'_p = θ'[:len(θ_p)]     # new policy
  → θ'_m = θ'[len(θ_p):]     # new mutator (self-replication!)

Crossover:
  offspring = Mutator_B(θ_m_B)(θ_p_A, θ_m_B)
  → Feed genome A's policy into genome B's mutator
```

## Mutator Architectures

### 1. Chunk-Based MLP
Read weights in fixed-size chunks, output perturbation per chunk.
Simple, fast, limited receptive field.

### 2. Transformer
Self-attention over weight segments. Can capture long-range dependencies
between different layers/regions of the network.

### 3. CPPN (Compositional Pattern-Producing Network)
Encode weight positions as (layer, row, col) coordinates → weight value.
Like HyperNEAT: the mutator is an indirect encoding of the entire genome.
Naturally produces structured, symmetric weight patterns.

## Evolution

- **Selection:** Tournament or truncation
- **Mutation:** Mutator network applied to own genome
- **Crossover:** Feed one genome's policy into another's mutator
- **Population:** Standard generational EA
- **Fitness:** Environment reward (CartPole → Ant → custom)

## Key Properties

- **Self-referential:** The mutator modifies itself through its own output
- **Evolvable variation operator:** Natural selection acts on the mutation strategy, not just the policy
- **Open-ended potential:** Mutators that produce better mutators get selected

## Environments

Start simple, scale up:
1. **CartPole-v1** — sanity check (4 obs, 2 actions)
2. **LunarLander-v3** — medium difficulty (8 obs, 4 actions)
3. **Ant-v5** — hard (111 obs, 8 actions, same as SM-NN project)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch gymnasium matplotlib numpy
```

## Related Work

- Neural Network Quine (Chang & Lipson, 2018) — self-replicating networks
- HyperNEAT (Stanley et al., 2009) — CPPNs for indirect weight encoding
- Adaptive RL through Evolving Self-Modifying NNs (Schmidgall, 2020) — neuromodulated plasticity
- Meta-learning via learned loss functions and update rules

## License

MIT
