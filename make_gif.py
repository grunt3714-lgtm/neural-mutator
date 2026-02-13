"""Generate a GIF of the best genome playing Snake."""
import sys, os
os.environ["SNAKE_NO_RUST"] = "1"  # Force Python backend
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from src.snake_env import SnakePixelsEnv, ensure_snake_registered
from src.genome import Genome

ensure_snake_registered()

genome_path = sys.argv[1] if len(sys.argv) > 1 else "results/fixed_food_300g/best_genome.pt"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/fixed_food_300g/best_snake.gif"

genome = Genome.load(genome_path)
env = SnakePixelsEnv(grid_size=16, device="cpu")

obs, _ = env.reset(seed=42)
frames = []
done = False
total_reward = 0

while not done:
    frame = env.render()
    # Upscale for visibility
    img = Image.fromarray(frame).resize((240, 240), Image.NEAREST)
    frames.append(img)
    
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = genome.policy(obs_t).argmax(dim=-1).item()
    
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

# Add last frame
frame = env.render()
img = Image.fromarray(frame).resize((240, 240), Image.NEAREST)
frames.append(img)

frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
print(f"GIF saved: {out_path} ({len(frames)} frames, reward={total_reward})")
