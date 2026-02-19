"""Snake environment for neuroevolution.

Fitness = number of food eaten. That's it.
Obs: 12 head-relative features (danger, food dir, position, direction).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class SnakeShapedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}
    DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

    def __init__(self, grid_size: int = 10, max_steps: int = 200, render_mode: str | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

    def _place_food(self):
        while True:
            r = self.np_random.integers(0, self.grid_size)
            c = self.np_random.integers(0, self.grid_size)
            if (r, c) not in self.body_set:
                self.food = (r, c)
                return

    def _get_obs(self):
        hr, hc = self.head
        fr, fc = self.food
        gs = self.grid_size

        danger = np.zeros(4, dtype=np.float32)
        for i, (dr, dc) in enumerate(self.DIRS):
            nr, nc = hr + dr, hc + dc
            if nr < 0 or nr >= gs or nc < 0 or nc >= gs or (nr, nc) in self.body_set:
                danger[i] = 1.0

        food_dr = (fr - hr) / gs
        food_dc = (fc - hc) / gs
        head_r = (hr / (gs - 1)) * 2 - 1
        head_c = (hc / (gs - 1)) * 2 - 1
        length_norm = len(self.body) / gs
        time_norm = (self.max_steps - self._step_count) / self.max_steps
        dr, dc = self.DIRS[self.direction]

        return np.array([
            danger[0], danger[1], danger[2], danger[3],
            food_dr, food_dc, head_r, head_c,
            length_norm, time_norm, float(dr), float(dc),
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        c = self.grid_size // 2
        self.head = (c, c)
        self.body = [(c, c), (c, c - 1), (c, c - 2)]
        self.body_set = set(self.body)
        self.direction = 1
        self._place_food()
        self._food_eaten = 0
        self._steps_since_food = 0
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        if action == (self.direction + 2) % 4:
            action = self.direction
        self.direction = action

        dr, dc = self.DIRS[action]
        hr, hc = self.head
        nr, nc = hr + dr, hc + dc
        self._step_count += 1
        self._steps_since_food += 1

        # Wall collision
        if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
            return self._get_obs(), 0.0, True, False, {"length": len(self.body), "food_eaten": self._food_eaten}

        # Self collision
        tail = self.body[-1]
        ate = (nr, nc) == self.food
        if (nr, nc) in self.body_set and not ((nr, nc) == tail and not ate):
            return self._get_obs(), 0.0, True, False, {"length": len(self.body), "food_eaten": self._food_eaten}

        # Move
        self.head = (nr, nc)
        self.body.insert(0, self.head)
        self.body_set.add(self.head)

        reward = 0.0
        if ate:
            reward = 1.0
            self._food_eaten += 1
            self._steps_since_food = 0
            self._place_food()
        else:
            old_tail = self.body.pop()
            self.body_set.discard(old_tail)

        # Starvation: no food in grid_size^2 steps
        truncated = self._steps_since_food > self.grid_size * self.grid_size
        done = self._step_count >= self.max_steps

        return self._get_obs(), reward, done, truncated, {"length": len(self.body), "food_eaten": self._food_eaten}

    def render(self):
        gs = self.grid_size
        img = np.zeros((gs, gs, 3), dtype=np.uint8)
        for r, c in self.body:
            img[r, c] = [0, 255, 0]
        hr, hc = self.head
        img[hr, hc] = [255, 255, 255]
        fr, fc = self.food
        img[fr, fc] = [255, 0, 0]
        return img


_registered = False


def ensure_snake_shaped_registered():
    global _registered
    if _registered:
        return
    register(
        id="SnakeShaped-v0",
        entry_point="src.snake_shaped_env:SnakeShapedEnv",
        max_episode_steps=200,
    )
    _registered = True
