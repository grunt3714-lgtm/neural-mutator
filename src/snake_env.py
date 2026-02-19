import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

try:
    from snake_gpu_rs import SnakeGpuCore  # type: ignore
except Exception:
    SnakeGpuCore = None


class SnakePixelsEnv(gym.Env):
    """
    Single-snake pixel-grid environment.

    Internal state updates run in torch tensors and can run on CUDA.
    Observation can be semantic channels (empty/snake/food) or raw RGB (flattened).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    # 0=up,1=right,2=down,3=left
    DIRS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=torch.long)

    def __init__(self, grid_size: int = 16, max_steps: int = None, device: str | None = None,
                 obs_encoding: str = "semantic", render_mode: str | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps if max_steps is not None else grid_size * grid_size
        self.render_mode = render_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_encoding = obs_encoding  # "semantic" (empty/snake/food) or "rgb"

        self.action_space = spaces.Discrete(4)
        # Image observation: (H, W, 4) uint8 — empty/snake/food/direction
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(grid_size, grid_size, 4),
            dtype=np.uint8,
        )
        self._direction = 1  # current direction for 4th channel

        self._dirs = self.DIRS.to(self.device)
        self._grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int8, device=self.device)
        self._step_count = 0

        self.head = None
        self.body: list[tuple[int, int]] = []
        self.direction = 1
        self.foods: list[tuple[int, int]] = []
        self.num_foods = 1
        self.bonus_steps_per_food = grid_size * grid_size  # extend episode on eating

        self._rust_core = None
        self.gpu_backend = "python"
        self.gpu_adapter = "none"
        self._last_obs = None  # encoded obs last returned by step/reset
        self._last_rgb = None  # rgb frame for render
        self.compute_mode = "cpu"
        # Rust core: init with very high max_steps, Python handles truncation with bonus steps
        if SnakeGpuCore is not None:
            try:
                self._rust_core = SnakeGpuCore(self.grid_size, 999999, self.num_foods)  # Python handles truncation
                b, a = self._rust_core.gpu_info()
                self.gpu_backend = b
                self.gpu_adapter = a
                if hasattr(self._rust_core, "execution_mode"):
                    self.compute_mode = str(self._rust_core.execution_mode())
            except Exception:
                self._rust_core = None

    def _spawn_food(self, count: int = 1):
        """Spawn `count` food items on empty cells (min Manhattan distance 4 from head)."""
        for _ in range(count):
            free = torch.nonzero(self._grid == 0, as_tuple=False)
            if free.shape[0] == 0:
                break
            if self.head is not None:
                hr, hc = self.head
                dists = torch.abs(free[:, 0] - hr) + torch.abs(free[:, 1] - hc)
                far_mask = dists >= 4
                if far_mask.any():
                    free = free[far_mask]
            idx = torch.randint(0, free.shape[0], (1,), device=self.device).item()
            r, c = free[idx].tolist()
            self.foods.append((r, c))
            self._grid[r, c] = 2

    def _obs_pixels(self) -> np.ndarray:
        """RGB observation for rendering/backward compatibility."""
        # palette: empty black, snake green, food red, head white
        # grid codes: 0 empty, 1 snake body, 2 food, 3 head
        g = self._grid
        H, W = g.shape
        pix = torch.zeros((H, W, 3), dtype=torch.uint8, device=self.device)
        pix[g == 1] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=self.device)
        pix[g == 2] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=self.device)
        pix[g == 3] = torch.tensor([255, 255, 255], dtype=torch.uint8, device=self.device)
        return pix.detach().cpu().numpy()

    def _obs_semantic(self) -> np.ndarray:
        """Semantic one-hot channels: empty/snake/food (head included in snake)."""
        g = self._grid
        H, W = g.shape
        sem = torch.zeros((H, W, 3), dtype=torch.uint8, device=self.device)
        sem[g == 0, 0] = 1  # empty
        sem[(g == 1) | (g == 3), 1] = 1  # snake (body+head)
        sem[g == 2, 2] = 1  # food
        return sem.detach().cpu().numpy()

    def _add_direction_channel(self, obs_3ch: np.ndarray) -> np.ndarray:
        """Append 4th channel: direction at head pixel.
        0=up, 1=right, 2=down, 3=left encoded as (dir+1)*63 → 63/126/189/252."""
        H, W = obs_3ch.shape[:2]
        dir_ch = np.zeros((H, W, 1), dtype=np.uint8)
        if self.head is not None:
            hr, hc = self.head
            if 0 <= hr < H and 0 <= hc < W:
                dir_ch[hr, hc, 0] = (self._direction + 1) * 63
        return np.concatenate([obs_3ch, dir_ch], axis=2)

    @staticmethod
    def _rgb_flat_to_semantic(rgb_flat: np.ndarray) -> np.ndarray:
        """Convert flattened RGB obs to semantic channels empty/snake/food."""
        arr = np.asarray(rgb_flat, dtype=np.uint8).reshape(-1, 3)
        sem = np.zeros_like(arr, dtype=np.uint8)
        # empty = black
        empty = np.all(arr == np.array([0, 0, 0], dtype=np.uint8), axis=1)
        # snake = green body or white head
        snake = np.all(arr == np.array([0, 255, 0], dtype=np.uint8), axis=1) | \
                np.all(arr == np.array([255, 255, 255], dtype=np.uint8), axis=1)
        # food = red
        food = np.all(arr == np.array([255, 0, 0], dtype=np.uint8), axis=1)
        sem[empty, 0] = 1
        sem[snake, 1] = 1
        sem[food, 2] = 1
        gs = int(np.sqrt(len(arr)))
        return sem.reshape(gs, gs, 3)

    @staticmethod
    def _semantic_flat_to_rgb(sem_flat: np.ndarray) -> np.ndarray:
        sem = np.asarray(sem_flat, dtype=np.uint8).reshape(-1, 3)
        rgb = np.zeros_like(sem, dtype=np.uint8)
        empty = sem[:, 0] > 0
        snake = sem[:, 1] > 0
        food = sem[:, 2] > 0
        rgb[empty] = np.array([0, 0, 0], dtype=np.uint8)
        rgb[snake] = np.array([0, 255, 0], dtype=np.uint8)
        rgb[food] = np.array([255, 0, 0], dtype=np.uint8)
        gs = int(np.sqrt(len(sem)))
        return rgb.reshape(gs, gs, 3)

    def _encode_obs(self, rgb_flat: np.ndarray | None = None) -> np.ndarray:
        if self.obs_encoding == "semantic":
            if rgb_flat is not None:
                return self._rgb_flat_to_semantic(rgb_flat)
            return self._obs_semantic()
        if rgb_flat is not None:
            return np.asarray(rgb_flat, dtype=np.uint8)
        return self._obs_pixels()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._rust_core is not None:
            raw_obs = np.array(self._rust_core.reset(), dtype=np.uint8)
            raw_obs = raw_obs.reshape(self.grid_size, self.grid_size, 3)
            # Track head/direction for 4th channel
            c = self.grid_size // 2
            self.head = (c, c)
            self._direction = 1  # initial direction: right
            self._food_eaten = 0
            self._step_count = 0
            self.max_steps = self.grid_size * self.grid_size  # reset to base
            obs = self._add_direction_channel(raw_obs)
            self._last_rgb = self._semantic_flat_to_rgb(raw_obs.reshape(-1))
            self._last_obs = obs
            return obs, {"backend": "rust", "gpu_backend": self.gpu_backend, "gpu_adapter": self.gpu_adapter, "compute_mode": self.compute_mode}

        self._step_count = 0
        self.max_steps = self.grid_size * self.grid_size  # reset to base, bonus extends it
        self._grid.zero_()

        c = self.grid_size // 2
        self.head = (c, c)
        self.body = [(c, c), (c, c - 1), (c, c - 2)]
        self.direction = 1
        self._direction = 1
        self._food_eaten = 0

        for r, col in self.body[1:]:
            self._grid[r, col] = 1
        self._grid[self.head[0], self.head[1]] = 3

        self.foods = []
        self._spawn_food(count=self.num_foods)
        sem_obs = self._obs_semantic()
        obs = self._add_direction_channel(sem_obs)
        self._last_rgb = self._obs_pixels()
        self._last_obs = obs
        return obs, {"backend": "python"}

    def step(self, action):
        if self._rust_core is not None:
            action = int(action)
            # Block reverse (mirror rust logic)
            if action == (self._direction + 2) % 4:
                action = self._direction
            self._direction = action
            # Move head in Python to track position
            DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            dr, dc = DIRS[action]
            hr, hc = self.head
            self.head = (hr + dr, hc + dc)

            self._step_count += 1
            obs, rust_reward, terminated, truncated, length = self._rust_core.step(action)
            raw_obs = np.array(obs, dtype=np.uint8).reshape(self.grid_size, self.grid_size, 3)
            # Override reward: pure food count (+1 per food, 0 otherwise)
            food_now = int(length) - 3  # initial length is 3
            reward = 0.0
            if food_now > self._food_eaten:
                reward = 1.0
                self._food_eaten = food_now
                self.max_steps += self.bonus_steps_per_food  # extend episode
            # Python-side truncation (rust has 999999 max_steps)
            if self._step_count >= self.max_steps:
                truncated = True
            obs_arr = self._add_direction_channel(raw_obs)
            self._last_rgb = self._semantic_flat_to_rgb(raw_obs.reshape(-1))
            self._last_obs = obs_arr
            return obs_arr, reward, bool(terminated), bool(truncated), {
                "length": int(length),
                "food_eaten": self._food_eaten,
                "backend": "rust",
            }

        action = int(action)
        # block instant reverse
        if action == (self.direction + 2) % 4:
            action = self.direction
        self.direction = action
        self._direction = action

        dr, dc = self._dirs[action].tolist()
        hr, hc = self.head
        nr, nc = hr + dr, hc + dc

        self._step_count += 1
        done = False

        # wall hit
        if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
            done = True
            sem_obs = self._obs_semantic()
            obs = self._add_direction_channel(sem_obs)
            self._last_rgb = self._obs_pixels()
            self._last_obs = obs
            return obs, 0.0, done, False, {"food_eaten": self._food_eaten}

        ate = (nr, nc) in self.foods

        # self hit
        tail = self.body[-1]
        occupied = set(self.body)
        if (nr, nc) in occupied and not ((nr, nc) == tail and not ate):
            done = True
            sem_obs = self._obs_semantic()
            obs = self._add_direction_channel(sem_obs)
            self._last_rgb = self._obs_pixels()
            self._last_obs = obs
            return obs, 0.0, done, False, {"food_eaten": self._food_eaten}

        # clear old head to body
        self._grid[hr, hc] = 1
        self.head = (nr, nc)
        self.body.insert(0, self.head)
        self._grid[nr, nc] = 3

        reward = 0.0
        if ate:
            reward = 1.0
            self._food_eaten += 1
            self.max_steps += self.bonus_steps_per_food  # extend episode
            self.foods.remove((nr, nc))
            self._spawn_food(count=1)
        else:
            tr, tc = self.body.pop()
            if (tr, tc) != self.head:
                self._grid[tr, tc] = 0

        if self._step_count >= self.max_steps:
            done = True

        sem_obs = self._obs_semantic()
        obs = self._add_direction_channel(sem_obs)
        self._last_rgb = self._obs_pixels()
        self._last_obs = obs
        return obs, reward, done, False, {"length": len(self.body), "food_eaten": self._food_eaten}

    def render(self):
        if self._last_rgb is not None:
            rgb = np.asarray(self._last_rgb)
            if rgb.ndim == 1:
                return rgb.reshape(self.grid_size, self.grid_size, 3)
            return rgb
        return np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)


_registered = False


def ensure_snake_registered():
    global _registered
    if _registered:
        return
    register(
        id="SnakePixels-v0",
        entry_point="src.snake_env:SnakePixelsEnv",
        max_episode_steps=300,
    )
    _registered = True
