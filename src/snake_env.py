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

    def __init__(self, grid_size: int = 16, max_steps: int = 300, device: str | None = None,
                 obs_encoding: str = "semantic"):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_encoding = obs_encoding  # "semantic" (empty/snake/food) or "rgb"

        self.action_space = spaces.Discrete(4)
        # Flattened observation: (H, W, 3) uint8 -> [H*W*3]
        # semantic channels are one-hot uint8 in {0,1}; rgb in {0..255}
        obs_high = 1 if self.obs_encoding == "semantic" else 255
        self.observation_space = spaces.Box(
            low=0,
            high=obs_high,
            shape=(grid_size * grid_size * 3,),
            dtype=np.uint8,
        )

        self._dirs = self.DIRS.to(self.device)
        self._grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int8, device=self.device)
        self._step_count = 0

        self.head = None
        self.body: list[tuple[int, int]] = []
        self.direction = 1
        self.foods: list[tuple[int, int]] = []
        self.num_foods = 10

        self._rust_core = None
        self.gpu_backend = "python"
        self.gpu_adapter = "none"
        self._last_obs = None  # encoded obs last returned by step/reset
        self._last_rgb = None  # rgb frame for render
        self.compute_mode = "cpu"
        if SnakeGpuCore is not None:
            try:
                self._rust_core = SnakeGpuCore(self.grid_size, self.max_steps, self.num_foods)
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
        return pix.flatten().detach().cpu().numpy()

    def _obs_semantic(self) -> np.ndarray:
        """Semantic one-hot channels: empty/snake/food (head included in snake)."""
        g = self._grid
        H, W = g.shape
        sem = torch.zeros((H, W, 3), dtype=torch.uint8, device=self.device)
        sem[g == 0, 0] = 1  # empty
        sem[(g == 1) | (g == 3), 1] = 1  # snake (body+head)
        sem[g == 2, 2] = 1  # food
        return sem.flatten().detach().cpu().numpy()

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
        return sem.reshape(-1)

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
        return rgb.reshape(-1)

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
            # Rust now returns semantic channels (empty/snake/food)
            if self.obs_encoding == "semantic":
                obs = raw_obs
                self._last_rgb = self._semantic_flat_to_rgb(raw_obs)
            else:
                obs = self._semantic_flat_to_rgb(raw_obs)
                self._last_rgb = obs
            self._last_obs = obs
            return obs, {"backend": "rust", "gpu_backend": self.gpu_backend, "gpu_adapter": self.gpu_adapter, "compute_mode": self.compute_mode, "obs_encoding": self.obs_encoding}

        self._step_count = 0
        self._grid.zero_()

        c = self.grid_size // 2
        self.head = (c, c)
        self.body = [(c, c), (c, c - 1), (c, c - 2)]
        self.direction = 1

        for r, col in self.body[1:]:
            self._grid[r, col] = 1
        self._grid[self.head[0], self.head[1]] = 3

        self.foods = []
        self._spawn_food(count=self.num_foods)
        rgb_obs = self._obs_pixels()
        obs = self._encode_obs(rgb_obs)
        self._last_rgb = rgb_obs
        self._last_obs = obs
        return obs, {"backend": "python", "gpu_backend": self.gpu_backend, "gpu_adapter": self.gpu_adapter, "obs_encoding": self.obs_encoding}

    def step(self, action):
        if self._rust_core is not None:
            obs, reward, terminated, truncated, length = self._rust_core.step(int(action))
            raw_obs = np.array(obs, dtype=np.uint8)
            if self.obs_encoding == "semantic":
                obs_arr = raw_obs
                self._last_rgb = self._semantic_flat_to_rgb(raw_obs)
            else:
                obs_arr = self._semantic_flat_to_rgb(raw_obs)
                self._last_rgb = obs_arr
            self._last_obs = obs_arr
            return obs_arr, float(reward), bool(terminated), bool(truncated), {
                "length": int(length),
                "backend": "rust",
                "gpu_backend": self.gpu_backend,
                "gpu_adapter": self.gpu_adapter,
                "compute_mode": self.compute_mode,
                "obs_encoding": self.obs_encoding,
            }

        action = int(action)
        # block instant reverse
        if action == (self.direction + 2) % 4:
            action = self.direction
        self.direction = action

        dr, dc = self._dirs[action].tolist()
        hr, hc = self.head
        nr, nc = hr + dr, hc + dc

        self._step_count += 1
        reward = -0.01
        done = False

        # wall hit
        if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
            reward = -1.0
            done = True
            rgb_obs = self._obs_pixels()
            obs = self._encode_obs(rgb_obs)
            self._last_rgb = rgb_obs
            self._last_obs = obs
            return obs, reward, done, False, {"obs_encoding": self.obs_encoding}

        ate = (nr, nc) in self.foods

        # self hit (allow stepping into tail only if not growing)
        tail = self.body[-1]
        occupied = set(self.body)
        if (nr, nc) in occupied and not ((nr, nc) == tail and not ate):
            reward = -1.0
            done = True
            rgb_obs = self._obs_pixels()
            obs = self._encode_obs(rgb_obs)
            self._last_rgb = rgb_obs
            self._last_obs = obs
            return obs, reward, done, False, {"obs_encoding": self.obs_encoding}

        # clear old head to body
        self._grid[hr, hc] = 1
        # add new head
        self.head = (nr, nc)
        self.body.insert(0, self.head)
        self._grid[nr, nc] = 3

        if ate:
            reward = 1.0
            self.foods.remove((nr, nc))
            self._spawn_food(count=1)
        else:
            tr, tc = self.body.pop()
            # if tail wasn't overwritten by new head, clear it
            if (tr, tc) != self.head:
                self._grid[tr, tc] = 0

        if self._step_count >= self.max_steps:
            done = True

        rgb_obs = self._obs_pixels()
        obs = self._encode_obs(rgb_obs)
        self._last_rgb = rgb_obs
        self._last_obs = obs
        return obs, reward, done, False, {"length": len(self.body), "backend": "python", "obs_encoding": self.obs_encoding}

    def render(self):
        if self._last_rgb is not None:
            return self._last_rgb.reshape(self.grid_size, self.grid_size, 3)
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
