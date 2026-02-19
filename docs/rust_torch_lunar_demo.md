# Rust + Torch Threaded LunarLander Demo

This branch adds a proof-of-concept Rust evaluator using `tch-rs` (libtorch) and Rayon threading.

## What it demonstrates

- Rust-side custom LunarLander-like environment (`CustomLunarLander`)
- TorchScript policy loading in Rust via `tch::CModule`
- Inference with grad disabled (`no_grad_guard`)
- Parallel episode rollouts with configurable thread counts
- Optional Discord progress pings from Rust (`webhook`, `progress_every`)
- Rust bindings for mutator + compat TorchScript modules (`mutator_mutate`, `compat_score`)

## Files

- `native/lunar_torch_eval_rs/Cargo.toml`
- `native/lunar_torch_eval_rs/src/lib.rs`
- `scripts/demo_rust_torch_lunar.py`

## Build

```bash
cd native/lunar_torch_eval_rs
maturin develop --release
```

If libtorch is not auto-detected, set one of:

```bash
export LIBTORCH=/path/to/libtorch
# or use Python torch lib dir via:
export LIBTORCH_USE_PYTORCH=1
```

## Run demo

```bash
python scripts/demo_rust_torch_lunar.py
```

Expected output format:

```text
threads= 1 | mean_reward=... | ... episodes/s
threads= 2 | mean_reward=... | ... episodes/s
threads= 4 | mean_reward=... | ... episodes/s
threads= 8 | mean_reward=... | ... episodes/s
```

## Notes

- This is a PoC for threaded Rust evaluation, not a full Gym-compatible LunarLander clone.
- Next step would be integrating this evaluator behind a Python flag in `src/evolution.py`.
