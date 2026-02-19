# Rust Eval Engine

## Goal
Replace Python multiprocessing eval loop with a Rust-native batch evaluator.
Eliminate pickle/IPC overhead (~5000x speedup potential for simple envs).

## Architecture

```
Python (evolution.py)          Rust (PyO3 module)
─────────────────────         ──────────────────────
population weights ──────────► batch_evaluate(
  (flat f32 arrays)              env_id,
                                 weights: Vec<Vec<f32>>,
                                 n_episodes: usize,
                                 policy_arch: str,
                                 policy_shape: tuple,
                              ) -> Vec<f32>  // fitnesses
                              
                              Internally:
                              - Reconstruct policy (CNN forward pass)
                              - Run episodes in parallel threads
                              - Return fitness scores
```

## Phase 1: Snake (easiest — env already in Rust)
- [ ] Add CNN forward pass to `snake-gpu-core` crate (conv2d + relu + flatten + fc)
- [ ] Add `batch_evaluate(weights, n_episodes, seeds)` PyO3 function
- [ ] Wire into evolution.py as alternative eval backend
- [ ] Benchmark vs Python multiprocessing

## Phase 2: Classic Control (trivial physics)
- [ ] CartPole-v1: ~30 lines of Euler integration
- [ ] Acrobot-v1: ~50 lines (2-link pendulum)
- [ ] Pendulum-v1: ~20 lines (single pendulum + torque)
- [ ] MLP forward pass (linear + tanh, no conv)
- [ ] Shared `batch_evaluate` with env selector

## Phase 3: LunarLander
- [ ] 2D rigid body: position, velocity, angle, angular velocity
- [ ] Two thrusters (main + side) with force application
- [ ] Leg contact detection (two legs, ground plane)
- [ ] Terrain generation (flat with landing pad)
- [ ] Reward function (match gym exactly for comparison)
- [ ] ~200-300 lines of physics

## Phase 4: CarRacing (hardest)
- [ ] Procedural track generation
- [ ] Tire friction model
- [ ] Pixel rendering for CNN observation (96×96×3)
- [ ] ~500-800 lines

## Non-goals
- Not a general Rust gym replacement
- Not rewriting mutation/speciation/selection (keep in Python)
- Not GPU compute (CPU threads are fast enough for these envs)

## Key constraint
CNN forward pass must be **exact match** with PyTorch output for weight compatibility.
Use same conv2d formula: no padding tricks, same stride/kernel conventions.
