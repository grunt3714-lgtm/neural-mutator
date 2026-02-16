"""Typed runtime config for training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
import argparse


@dataclass(frozen=True)
class MutatorConfig:
    mutator_type: str
    chunk_size: int = 64
    dualmix_p_gauss_policy: float = 0.20
    dualmix_gauss_scale_policy: float = 0.03

    def to_kwargs(self) -> dict:
        if self.mutator_type == 'dualmixture':
            return {
                'chunk_size': self.chunk_size,
                'p_gauss_policy': self.dualmix_p_gauss_policy,
                'gauss_scale_policy': self.dualmix_gauss_scale_policy,
            }
        return {'chunk_size': self.chunk_size}


@dataclass(frozen=True)
class TrainConfig:
    env: str
    pop_size: int
    generations: int
    episodes: int
    crossover_rate: float
    hidden: int
    policy_arch: str
    seed: int
    output: str
    speciation: bool
    compat_threshold: float
    flex: bool
    complexity_cost: float
    workers: int


def build_configs(args: argparse.Namespace) -> tuple[TrainConfig, MutatorConfig]:
    train_cfg = TrainConfig(
        env=args.env,
        pop_size=args.pop_size,
        generations=args.generations,
        episodes=args.episodes,
        crossover_rate=args.crossover_rate,
        hidden=args.hidden,
        policy_arch=args.policy_arch,
        seed=args.seed,
        output=args.output,
        speciation=args.speciation,
        compat_threshold=args.compat_threshold,
        flex=args.flex,
        complexity_cost=args.complexity_cost,
        workers=args.workers,
    )
    mut_cfg = MutatorConfig(
        mutator_type=args.mutator,
        chunk_size=args.chunk_size,
        dualmix_p_gauss_policy=args.dualmix_p_gauss_policy,
        dualmix_gauss_scale_policy=args.dualmix_gauss_scale_policy,
    )
    return train_cfg, mut_cfg
