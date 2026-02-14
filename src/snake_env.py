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
    Observation is raw RGB pixels (flattened) so policy sees the board directly.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    # 0=up,1=right,2=down,3=left
    DIRS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=torch.long)

    def __init__(self, grid_size: int = 16, max_steps: int = 300, device: str | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.action_space = spaces.Discrete(4)
        # Flattened RGB pixels: (H, W, 3) uint8 -> [H*W*3]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
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
        self._last_obs = None
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
        # palette: empty black, snake green, food red, head white
        # grid codes: 0 empty, 1 snake body, 2 food, 3 head
        g = self._grid
        H, W = g.shape
        pix = torch.zeros((H, W, 3), dtype=torch.uint8, device=self.device)
        pix[g == 1] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=self.device)
        pix[g == 2] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=self.device)
        pix[g == 3] = torch.tensor([255, 255, 255], dtype=torch.uint8, device=self.device)
        return pix.flatten().detach().cpu().numpy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._rust_core is not None:
            obs = np.array(self._rust_core.reset(), dtype=np.uint8)
            self._last_obs = obs
            return obs, {"backend": "rust", "gpu_backend": self.gpu_backend, "gpu_adapter": self.gpu_adapter, "compute_mode": self.compute_mode}

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
        obs = self._obs_pixels()
        self._last_obs = obs
        return obs, {"backend": "python", "gpu_backend": self.gpu_backend, "gpu_adapter": self.gpu_adapter}

    def step(self, action):
        if self._rust_core is not None:
            obs, reward, terminated, truncated, length = self._rust_core.step(int(action))
            obs_arr = np.array(obs, dtype=np.uint8)
            self._last_obs = obs_arr
            return obs_arr, float(reward), bool(terminated), bool(truncated), {
                "length": int(length),
                "backend": "rust",
                "gpu_backend": self.gpu_backend,
                "gpu_adapter": self.gpu_adapter,
                "compute_mode": self.compute_mode,
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
            return self._obs_pixels(), reward, done, False, {}

        ate = (nr, nc) in self.foods

        # self hit (allow stepping into tail only if not growing)
        tail = self.body[-1]
        occupied = set(self.body)
        if (nr, nc) in occupied and not ((nr, nc) == tail and not ate):
            reward = -1.0
            done = True
            return self._obs_pixels(), reward, done, False, {}

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

        return self._obs_pixels(), reward, done, False, {"length": len(self.body), "backend": "python"}

    def render(self):
        if self._rust_core is not None:
            if self._last_obs is None:
                return np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            return self._last_obs.reshape(self.grid_size, self.grid_size, 3)
        pix = self._obs_pixels().reshape(self.grid_size, self.grid_size, 3)
        return pix


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
