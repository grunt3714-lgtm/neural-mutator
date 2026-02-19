#!/usr/bin/env python3
"""Export policy/mutator/compat TorchScript modules for Rust (tch-rs) use.

Usage:
  .venv/bin/python scripts/export_torchscript_stack.py \
    --genome results/lunar_s45_300g_fleet/best_ever_genome.pt \
    --out-dir /tmp/ts_stack
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn

from src.genome import Genome


class MutatorWrapper(nn.Module):
    def __init__(self, mutator: nn.Module):
        super().__init__()
        self.mutator = mutator

    def forward(self, flat_weights: torch.Tensor, split_idx: torch.Tensor) -> torch.Tensor:
        # Rust passes scalar tensor for split index
        split = split_idx.to(dtype=torch.float32).reshape(1)
        return self.mutator.mutate_genome(flat_weights, weight_coords=split)


class CompatWrapper(nn.Module):
    def __init__(self, compat: nn.Module):
        super().__init__()
        self.compat = compat

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea = self.compat.embed(a)
        eb = self.compat.embed(b)
        c = torch.cat([ea, eb])
        # return scalar tensor
        return self.compat.scorer(c).reshape(())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    g = Genome.load(args.genome)
    g.policy.eval(); g.mutator.eval(); g.compat_net.eval()

    # Policy
    policy_ex = torch.zeros(1, g.obs_dim, dtype=torch.float32)
    ts_policy = torch.jit.trace(g.policy, policy_ex)
    ts_policy.save(str(out / "policy.pt"))

    # Mutator
    flat = g.get_flat_weights().detach().float()
    split = torch.tensor([g.num_policy_params()], dtype=torch.float32)
    mw = MutatorWrapper(g.mutator).eval()
    ts_mut = torch.jit.trace(mw, (flat, split))
    ts_mut.save(str(out / "mutator.pt"))

    # Compat
    cw = CompatWrapper(g.compat_net).eval()
    ts_comp = torch.jit.trace(cw, (flat, flat.clone()))
    ts_comp.save(str(out / "compat.pt"))

    print(f"Exported TorchScript stack to {out}")


if __name__ == "__main__":
    main()
