#!/usr/bin/env python3
"""Demo: threaded Rust+tch evaluation on a custom LunarLander env.

Usage:
  python scripts/demo_rust_torch_lunar.py

Prereqs:
  1) Build extension (example):
     cd native/lunar_torch_eval_rs && maturin develop --release
  2) Ensure torch is installed in Python env.
"""
from pathlib import Path
import time
import torch
import torch.nn as nn


def ensure_model(path: Path):
    if path.exists():
        return
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 4),
            )

        def forward(self, x):
            return self.net(x)

    model = Policy().eval()
    example = torch.zeros(1, 8)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(path))
    print(f"Saved demo TorchScript model -> {path}")


def main():
    model_path = Path("/tmp/lunar_policy_demo.pt")
    ensure_model(model_path)

    import lunar_torch_eval_rs as rt  # noqa: E402

    episodes = 512
    max_steps = 500

    for threads in [1, 2, 4, 8]:
        t0 = time.time()
        score = rt.evaluate_model_threaded(str(model_path), episodes, max_steps, threads, 42, None, None)
        dt = time.time() - t0
        eps_per_sec = episodes / dt
        print(f"threads={threads:>2} | mean_reward={score:8.3f} | {eps_per_sec:8.1f} episodes/s")


if __name__ == "__main__":
    main()
