#!/usr/bin/env python3
import random
import time

from src import snake_env as se


def bench_env(env, steps=20000):
    obs, info = env.reset()
    t0 = time.perf_counter()
    done = False
    for _ in range(steps):
        if done:
            obs, info = env.reset()
            done = False
        a = random.randint(0, 3)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
    dt = time.perf_counter() - t0
    return steps / dt


def main():
    # GPU/Rust path
    env_gpu = se.SnakePixelsEnv(grid_size=16, max_steps=300)
    _, info_gpu = env_gpu.reset()
    gpu_sps = bench_env(env_gpu)

    # Force CPU fallback by disabling Rust core import
    old_cls = se.SnakeGpuCore
    se.SnakeGpuCore = None
    env_cpu = se.SnakePixelsEnv(grid_size=16, max_steps=300, device="cpu")
    _, info_cpu = env_cpu.reset()
    cpu_sps = bench_env(env_cpu)
    se.SnakeGpuCore = old_cls

    print("SnakePixels-v0 benchmark")
    print(f"GPU env info: {info_gpu}")
    print(f"CPU env info: {info_cpu}")
    print(f"GPU path steps/sec: {gpu_sps:,.1f}")
    print(f"CPU fallback steps/sec: {cpu_sps:,.1f}")
    if cpu_sps > 0:
        print(f"Speedup (GPU/CPU): {gpu_sps/cpu_sps:.2f}x")


if __name__ == "__main__":
    main()
