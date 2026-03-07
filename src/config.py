"""Typed runtime config for training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
import argparse


@dataclass(frozen=True)
class MutatorConfig:
    mutator_type: str
    chunk_size: int = 64
    crossover_gated_blend: bool = True
    crossover_gate_mode: str = 'sigmoid'
    crossover_gumbel_tau: float = 1.0
    crossover_gumbel_hard: bool = False
    crossover_gate_clamp: float = 0.0
    dualmix_p_gauss_policy: float = 0.20
    dualmix_gauss_scale_policy: float = 0.03
    dualmix_v2_ref_dim: int = 16
    dualmix_v2_hidden: int = 64
    dualmix_v2_lowrank_rank: int = 4
    dualmix_v2_max_policy_groups: int = 8
    dualmix_v2_policy_corr_scale: float = 0.025
    dualmix_v2_policy_noise_scale: float = 0.008
    dualmix_v2_meta_corr_scale: float = 0.01
    dualmix_v2_meta_noise_scale: float = 0.002
    global_lowrank_block_size: int = 64
    global_lowrank_ref_dim: int = 16
    global_lowrank_hidden: int = 64
    global_lowrank_rank: int = 4
    global_lowrank_policy_scale: float = 0.02
    global_lowrank_meta_scale: float = 0.006
    perceiver_lite_block_size: int = 64
    perceiver_lite_ref_dim: int = 16
    perceiver_lite_hidden: int = 64
    perceiver_lite_latent_count: int = 8
    perceiver_lite_policy_scale: float = 0.018
    perceiver_lite_meta_scale: float = 0.004

    def to_kwargs(self) -> dict:
        crossover_cfg = {
            'crossover_gated_blend': self.crossover_gated_blend,
            'crossover_gate_mode': self.crossover_gate_mode,
            'crossover_gumbel_tau': self.crossover_gumbel_tau,
            'crossover_gumbel_hard': self.crossover_gumbel_hard,
            'crossover_gate_clamp': self.crossover_gate_clamp,
        }
        if self.mutator_type == 'dualcorrector':
            return {
                'chunk_size': self.chunk_size,
                **crossover_cfg,
            }
        if self.mutator_type == 'dualmixture':
            return {
                'chunk_size': self.chunk_size,
                **crossover_cfg,
                'p_gauss_policy': self.dualmix_p_gauss_policy,
                'gauss_scale_policy': self.dualmix_gauss_scale_policy,
            }
        if self.mutator_type == 'dualmixture_v2':
            return {
                'chunk_size': self.chunk_size,
                'ref_dim': self.dualmix_v2_ref_dim,
                'hidden': self.dualmix_v2_hidden,
                'lowrank_rank': self.dualmix_v2_lowrank_rank,
                'max_policy_groups': self.dualmix_v2_max_policy_groups,
                'policy_corr_scale': self.dualmix_v2_policy_corr_scale,
                'policy_noise_scale': self.dualmix_v2_policy_noise_scale,
                'meta_corr_scale': self.dualmix_v2_meta_corr_scale,
                'meta_noise_scale': self.dualmix_v2_meta_noise_scale,
            }
        if self.mutator_type == 'global_lowrank':
            return {
                'block_size': self.global_lowrank_block_size,
                'ref_dim': self.global_lowrank_ref_dim,
                'hidden': self.global_lowrank_hidden,
                'rank': self.global_lowrank_rank,
                'policy_scale': self.global_lowrank_policy_scale,
                'meta_scale': self.global_lowrank_meta_scale,
            }
        if self.mutator_type == 'perceiver_lite':
            return {
                'block_size': self.perceiver_lite_block_size,
                'ref_dim': self.perceiver_lite_ref_dim,
                'hidden': self.perceiver_lite_hidden,
                'latent_count': self.perceiver_lite_latent_count,
                'policy_scale': self.perceiver_lite_policy_scale,
                'meta_scale': self.perceiver_lite_meta_scale,
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
    compat_rate_guardrail: bool
    compat_rate_target_low: float
    compat_rate_target_high: float
    compat_rate_adjust: float
    flex: bool
    complexity_cost: float
    workers: int
    mate_choice: bool
    mate_choice_topk: int
    mate_choice_threshold: float
    mate_choice_temperature: float


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
        compat_rate_guardrail=args.compat_rate_guardrail,
        compat_rate_target_low=args.compat_rate_target_low,
        compat_rate_target_high=args.compat_rate_target_high,
        compat_rate_adjust=args.compat_rate_adjust,
        flex=args.flex,
        complexity_cost=args.complexity_cost,
        workers=args.workers,
        mate_choice=args.mate_choice,
        mate_choice_topk=args.mate_choice_topk,
        mate_choice_threshold=args.mate_choice_threshold,
        mate_choice_temperature=args.mate_choice_temperature,
    )
    mut_cfg = MutatorConfig(
        mutator_type=args.mutator,
        chunk_size=args.chunk_size,
        crossover_gated_blend=args.crossover_gated_blend,
        crossover_gate_mode=args.crossover_gate_mode,
        crossover_gumbel_tau=args.crossover_gumbel_tau,
        crossover_gumbel_hard=args.crossover_gumbel_hard,
        crossover_gate_clamp=args.crossover_gate_clamp,
        dualmix_p_gauss_policy=args.dualmix_p_gauss_policy,
        dualmix_gauss_scale_policy=args.dualmix_gauss_scale_policy,
        dualmix_v2_ref_dim=args.dualmix_v2_ref_dim,
        dualmix_v2_hidden=args.dualmix_v2_hidden,
        dualmix_v2_lowrank_rank=args.dualmix_v2_lowrank_rank,
        dualmix_v2_max_policy_groups=args.dualmix_v2_max_policy_groups,
        dualmix_v2_policy_corr_scale=args.dualmix_v2_policy_corr_scale,
        dualmix_v2_policy_noise_scale=args.dualmix_v2_policy_noise_scale,
        dualmix_v2_meta_corr_scale=args.dualmix_v2_meta_corr_scale,
        dualmix_v2_meta_noise_scale=args.dualmix_v2_meta_noise_scale,
        global_lowrank_block_size=args.global_lowrank_block_size,
        global_lowrank_ref_dim=args.global_lowrank_ref_dim,
        global_lowrank_hidden=args.global_lowrank_hidden,
        global_lowrank_rank=args.global_lowrank_rank,
        global_lowrank_policy_scale=args.global_lowrank_policy_scale,
        global_lowrank_meta_scale=args.global_lowrank_meta_scale,
        perceiver_lite_block_size=args.perceiver_lite_block_size,
        perceiver_lite_ref_dim=args.perceiver_lite_ref_dim,
        perceiver_lite_hidden=args.perceiver_lite_hidden,
        perceiver_lite_latent_count=args.perceiver_lite_latent_count,
        perceiver_lite_policy_scale=args.perceiver_lite_policy_scale,
        perceiver_lite_meta_scale=args.perceiver_lite_meta_scale,
    )
    return train_cfg, mut_cfg
