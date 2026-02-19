#!/usr/bin/env python3
"""Render gameplay videos with neural activation overlays for all environments."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import gymnasium as gym
import imageio
from PIL import Image, ImageDraw
from matplotlib import cm
from src.genome import Genome
from src.snake_env import ensure_snake_registered
from src.snake_shaped_env import ensure_snake_shaped_registered

ENV_INFO = {
    'CartPole-v1': {
        'obs_names': ['Cart Pos','Cart Vel','Pole Angle','Pole Vel'],
        'act_names': ['Left','Right'], 'discrete': True, 'max_steps': 500,
    },
    'Acrobot-v1': {
        'obs_names': ['cos θ1','sin θ1','cos θ2','sin θ2','θ1 vel','θ2 vel'],
        'act_names': ['+torque','0','−torque'], 'discrete': True, 'max_steps': 500,
    },
    'Pendulum-v1': {
        'obs_names': ['cos θ','sin θ','θ vel'],
        'act_names': ['Torque'], 'discrete': False, 'max_steps': 200,
    },
    'LunarLander-v3': {
        'obs_names': ['x','y','vx','vy','θ','ω','leg_L','leg_R'],
        'act_names': ['Noop','Left','Main','Right'], 'discrete': True, 'max_steps': 1000,
    },
    'CarRacing-v3': {
        'obs_names': None,  # CNN - no named obs
        'act_names': ['Steer','Gas','Brake'], 'discrete': False, 'max_steps': 1000,
    },
    'SnakeShaped-v0': {
        'obs_names': ['danger↑','danger→','danger↓','danger←','food_dr','food_dc','head_r','head_c','length','time','dir_r','dir_c'],
        'act_names': ['Up','Right','Down','Left'], 'discrete': True, 'max_steps': 200,
    },
    'SnakePixels-v0': {
        'obs_names': None,  # CNN
        'act_names': ['Up','Right','Down','Left'], 'discrete': True, 'max_steps': 300,
    },
}

PANEL_W = 220

def draw_mlp_panel(activations, obs, action, info, game_h):
    """Draw MLP activation panel: obs bars → hidden heatmap → action bars."""
    parts = []
    
    # Observation inputs
    lbl = Image.new('RGB', (PANEL_W, 14), (20,20,20))
    ImageDraw.Draw(lbl).text((4,1), 'input (observations)', fill=(180,180,180))
    parts.append(np.array(lbl))
    
    obs_flat = obs.flatten()
    n_obs = len(obs_flat)
    obs_h = max(60, n_obs * 14 + 4)
    obs_img = Image.new('RGB', (PANEL_W, obs_h), (30,30,30))
    d = ImageDraw.Draw(obs_img)
    obs_names = info.get('obs_names', [f'obs[{i}]' for i in range(n_obs)])
    max_abs = max(abs(obs_flat.max()), abs(obs_flat.min()), 1e-6)
    for i in range(min(n_obs, len(obs_names))):
        y = 2 + i * 14
        v = obs_flat[i]
        norm = v / max_abs
        bw = int(norm * 60)
        cx = 100
        color = (100, 180, 255) if v >= 0 else (255, 130, 80)
        if bw > 0: d.rectangle([cx, y, cx+bw, y+11], fill=color)
        elif bw < 0: d.rectangle([cx+bw, y, cx, y+11], fill=color)
        else: d.rectangle([cx, y, cx+1, y+11], fill=color)
        d.text((4, y), f'{obs_names[i]}: {v:.2f}', fill=(200,200,200))
    parts.append(np.array(obs_img))
    
    # Hidden layers
    for name in sorted(activations.keys()):
        if name.startswith('output'): continue
        act = activations[name][0].numpy()
        n_neurons = len(act)
        
        lbl = Image.new('RGB', (PANEL_W, 14), (20,20,20))
        ImageDraw.Draw(lbl).text((4,1), f'{name} ({n_neurons} neurons)', fill=(180,180,180))
        parts.append(np.array(lbl))
        
        # Heatmap bar
        act_norm = (act - act.min()) / (act.max() - act.min() + 1e-8)
        bar_w = PANEL_W - 10
        bar_h = max(16, min(30, 400 // max(1, n_neurons // 8)))
        rows = max(1, (n_neurons + (bar_w // 6) - 1) // (bar_w // 6))
        cols = min(n_neurons, bar_w // 6)
        cell = max(4, min(12, bar_w // cols))
        
        grid_h = rows * (cell + 1) + 1
        grid_w = cols * (cell + 1) + 1
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for idx in range(n_neurons):
            r, c = divmod(idx, cols)
            color = (np.array(cm.viridis(act_norm[idx])[:3]) * 255).astype(np.uint8)
            y0, x0 = r * (cell+1) + 1, c * (cell+1) + 1
            grid[y0:y0+cell, x0:x0+cell] = color
        
        padded = Image.new('RGB', (PANEL_W, grid_h), (10,10,10))
        padded.paste(Image.fromarray(grid), (5, 0))
        parts.append(np.array(padded))
    
    # Output / action
    lbl = Image.new('RGB', (PANEL_W, 14), (20,20,20))
    ImageDraw.Draw(lbl).text((4,1), 'output (actions)', fill=(180,180,180))
    parts.append(np.array(lbl))
    
    action_arr = np.atleast_1d(action)
    act_names = info.get('act_names', [f'a[{i}]' for i in range(len(action_arr))])
    act_h = max(40, len(act_names) * 17 + 6)
    act_img = Image.new('RGB', (PANEL_W, act_h), (30,30,30))
    d = ImageDraw.Draw(act_img)
    
    if info.get('discrete'):
        chosen = int(action_arr[0]) if len(action_arr) == 1 and action_arr[0] == int(action_arr[0]) else int(np.argmax(action_arr))
        if np.isscalar(action): chosen = int(action)
        out = activations.get('output', None)
        out_vals = out[0].numpy() if out is not None else np.zeros(len(act_names))
        for i, aname in enumerate(act_names):
            y = 3 + i * 17
            v = out_vals[i] if i < len(out_vals) else 0
            cx = 100; bw = int(v * 50)
            is_chosen = (i == chosen)
            color = (50, 255, 50) if is_chosen else (100, 100, 100)
            if bw > 0: d.rectangle([cx, y, cx+bw, y+13], fill=color)
            elif bw < 0: d.rectangle([cx+bw, y, cx, y+13], fill=color)
            else: d.rectangle([cx, y, cx+1, y+13], fill=color)
            marker = ' ◄' if is_chosen else ''
            d.text((4, y), f'{aname}: {v:.2f}{marker}', fill=(255,255,255) if is_chosen else (180,180,180))
    else:
        out = activations.get('output', None)
        out_vals = out[0].numpy() if out is not None else (action if not np.isscalar(action) else np.array([action]))
        colors_pos = [(100,200,100), (100,180,255), (255,160,80)]
        for i, aname in enumerate(act_names):
            y = 3 + i * 17
            v = out_vals[i] if i < len(out_vals) else 0
            cx = 100; bw = int(v * 50)
            color = colors_pos[i % len(colors_pos)] if v >= 0 else (200,100,100)
            if bw > 0: d.rectangle([cx, y, cx+bw, y+13], fill=color)
            elif bw < 0: d.rectangle([cx+bw, y, cx, y+13], fill=color)
            else: d.rectangle([cx, y, cx+1, y+13], fill=color)
            d.text((4, y), f'{aname}: {v:+.2f}', fill=(220,220,220))
    
    parts.append(np.array(act_img))
    
    panel = np.vstack(parts)
    if panel.shape[0] < game_h:
        panel = np.vstack([panel, np.zeros((game_h - panel.shape[0], PANEL_W, 3), dtype=np.uint8)])
    else:
        panel = panel[:game_h]
    return panel


def draw_cnn_panel(activations, action, info, game_h):
    """Draw CNN activation panel for CarRacing."""
    parts = []
    
    conv_keys = [k for k in sorted(activations.keys()) if k.startswith('conv')]
    
    # Pair conv layers
    for i in range(0, len(conv_keys), 2):
        pair = conv_keys[i:i+2]
        pair_name = ' | '.join(pair)
        
        grids = []
        for k in pair:
            act = activations[k][0]
            c, h, w = act.shape
            act = act - act.min()
            mx = act.max()
            if mx > 0: act = act / mx
            act = (act * 255).byte().numpy()
            cols = min(c, 4); rows = (c + cols - 1) // cols; pad = 1
            cell = min(18, (PANEL_W - 20) // cols - pad)
            grid = np.zeros((rows*(cell+pad)+pad, cols*(cell+pad)+pad), dtype=np.uint8)
            for idx in range(c):
                r, col = divmod(idx, cols)
                fm = np.array(Image.fromarray(act[idx]).resize((cell, cell), Image.NEAREST))
                grid[r*(cell+pad)+pad:r*(cell+pad)+pad+cell, col*(cell+pad)+pad:col*(cell+pad)+pad+cell] = fm
            grids.append((cm.viridis(grid/255.0)[:,:,:3]*255).astype(np.uint8))
        
        max_h = max(g.shape[0] for g in grids)
        rp = []
        for gr in grids:
            if gr.shape[0] < max_h: gr = np.vstack([gr, np.zeros((max_h-gr.shape[0], gr.shape[1], 3), dtype=np.uint8)])
            rp.append(gr)
        row = np.hstack([rp[0], np.zeros((max_h, 4, 3), dtype=np.uint8)] + ([rp[1]] if len(rp) > 1 else []))
        
        lbl = Image.new('RGB', (PANEL_W, 12), (20,20,20))
        ImageDraw.Draw(lbl).text((4,0), pair_name, fill=(180,180,180))
        pil = Image.fromarray(row)
        pil = pil.resize((PANEL_W-6, int(pil.height*(PANEL_W-6)/pil.width)), Image.NEAREST)
        padded = Image.new('RGB', (PANEL_W, pil.height), (10,10,10))
        padded.paste(pil, (3, 0))
        parts.append(np.array(lbl)); parts.append(np.array(padded))
    
    # FC hidden
    if 'fc_hidden' in activations:
        fc1 = activations['fc_hidden'][0].numpy()
        fc1_norm = (fc1 - fc1.min()) / (fc1.max() - fc1.min() + 1e-8)
        bar_w = PANEL_W - 10; bar_h = 16
        fc_img = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
        cw = max(1, bar_w // len(fc1_norm))
        for i, v in enumerate(fc1_norm):
            fc_img[:, i*cw:i*cw+cw] = (np.array(cm.viridis(v)[:3])*255).astype(np.uint8)
        lbl = Image.new('RGB', (PANEL_W, 12), (20,20,20))
        ImageDraw.Draw(lbl).text((4,0), f'fc hidden ({len(fc1)} neurons)', fill=(180,180,180))
        padded = Image.new('RGB', (PANEL_W, bar_h), (10,10,10))
        padded.paste(Image.fromarray(fc_img), (5, 0))
        parts.append(np.array(lbl)); parts.append(np.array(padded))
    
    # Action output
    act_names = info['act_names']
    out_vals = activations.get('output', None)
    out_np = out_vals[0].numpy() if out_vals is not None else action
    
    lbl = Image.new('RGB', (PANEL_W, 12), (20,20,20))
    ImageDraw.Draw(lbl).text((4,0), 'output (actions)', fill=(180,180,180))
    parts.append(np.array(lbl))
    
    act_h = max(54, len(act_names) * 17 + 6)
    act_img = Image.new('RGB', (PANEL_W, act_h), (30,30,30))
    d = ImageDraw.Draw(act_img)
    colors_pos = [(100,200,100), (100,180,255), (255,160,80)]
    for i, (aname, aval) in enumerate(zip(act_names, out_np)):
        y = 3 + i * 17; cx = 100; bw = int(aval * 50)
        color = colors_pos[i % 3] if aval >= 0 else (200,100,100)
        if bw > 0: d.rectangle([cx, y, cx+bw, y+13], fill=color)
        elif bw < 0: d.rectangle([cx+bw, y, cx, y+13], fill=color)
        else: d.rectangle([cx, y, cx+1, y+13], fill=color)
        d.text((4, y), f'{aname}: {aval:+.2f}', fill=(220,220,220))
    parts.append(np.array(act_img))
    
    panel = np.vstack(parts)
    if panel.shape[0] < game_h:
        panel = np.vstack([panel, np.zeros((game_h - panel.shape[0], PANEL_W, 3), dtype=np.uint8)])
    else:
        panel = panel[:game_h]
    return panel


def render_env(env_name, genome_path, output_path, seed=42):
    g = Genome.load(genome_path)
    g.policy.eval()
    info = ENV_INFO[env_name]
    from src.genome import PolicyCNNLarge, PolicyCNNSmall
    is_cnn = isinstance(g.policy, (PolicyCNNLarge, PolicyCNNSmall))
    
    # Register hooks
    activations = {}
    def make_hook(name):
        def hook(mod, inp, out):
            activations[name] = out.detach().cpu()
        return hook
    
    if is_cnn:
        if isinstance(g.policy, PolicyCNNSmall):
            # 2 conv layers: conv[0]=Conv2d, conv[1]=ReLU, conv[2]=Conv2d, conv[3]=ReLU
            g.policy.conv[1].register_forward_hook(make_hook('conv1'))
            g.policy.conv[3].register_forward_hook(make_hook('conv2'))
            g.policy.fc[1].register_forward_hook(make_hook('fc_hidden'))
            g.policy.fc[3].register_forward_hook(make_hook('output'))
        else:
            g.policy.conv[1].register_forward_hook(make_hook('conv1'))
            g.policy.conv[3].register_forward_hook(make_hook('conv2'))
            g.policy.conv[5].register_forward_hook(make_hook('conv3'))
            g.policy.conv[7].register_forward_hook(make_hook('conv4'))
            g.policy.fc[1].register_forward_hook(make_hook('fc_hidden'))
            g.policy.fc[3].register_forward_hook(make_hook('output'))
    else:
        # MLP — hook after each Tanh
        layer_idx = 0
        for i, mod in enumerate(g.policy.net):
            if isinstance(mod, torch.nn.Tanh):
                if i == len(g.policy.net) - 1 or (i+1 < len(g.policy.net) and isinstance(g.policy.net[i], torch.nn.Tanh)):
                    # Last tanh or hidden
                    pass
                name = f'hidden_{layer_idx}' if layer_idx < len(list(g.policy.net)) - 2 else 'output'
                mod.register_forward_hook(make_hook(name))
                layer_idx += 1
        # Also hook the final layer output
        last_idx = len(g.policy.net) - 1
        g.policy.net[last_idx].register_forward_hook(make_hook('output'))
    
    ensure_snake_registered()
    ensure_snake_shaped_registered()
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset(seed=seed)
    
    writer = imageio.get_writer(output_path, fps=30, macro_block_size=1,
                                output_params=['-pix_fmt', 'yuv420p'])
    total_reward = 0
    step = 0
    done = False
    max_steps = info['max_steps']
    
    while not done and step < max_steps:
        frame = env.render()
        # Upscale tiny grids (e.g. Snake 10x10)
        if frame.shape[0] < 100:
            scale = 400 // frame.shape[0]
            frame = np.kron(frame, np.ones((scale, scale, 1))).astype(np.uint8)
        
        if is_cnn:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        
        with torch.no_grad():
            raw_out = g.policy(obs_t).squeeze()
            if info['discrete']:
                action = int(torch.argmax(raw_out).item())
            elif is_cnn:
                action = raw_out.numpy()
            else:
                action = raw_out.numpy()
        
        if is_cnn:
            panel = draw_cnn_panel(activations, action, info, frame.shape[0])
        else:
            act_for_panel = raw_out.numpy() if not np.isscalar(action) else action
            panel = draw_mlp_panel(activations, obs, act_for_panel, info, frame.shape[0])
        
        sep = np.ones((frame.shape[0], 2, 3), dtype=np.uint8) * 60
        combined = np.hstack([frame, sep, panel])
        h, w = combined.shape[:2]
        if w % 2: combined = np.pad(combined, ((0,0),(0,1),(0,0)))
        if h % 2: combined = np.pad(combined, ((0,1),(0,0),(0,0)))
        writer.append_data(combined)
        
        if info['discrete'] and np.isscalar(action):
            obs, reward, terminated, truncated, _ = env.step(action)
        elif not info['discrete'] and env_name != 'CarRacing-v3':
            act_clipped = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(act_clipped)
        else:
            obs, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        done = terminated or truncated
        step += 1
    
    writer.close()
    env.close()
    print(f'{env_name}: {step} steps, reward={total_reward:.1f} → {output_path}')
    return total_reward


if __name__ == '__main__':
    runs = [
        ('CartPole-v1', 'results/cartpole_dm_flex_s45_50g/best_ever_genome.pt', 'results/cartpole_dm_flex_s45_50g/best_gameplay.mp4'),
        ('Acrobot-v1', 'results/acrobot_dm_flex_s45_300g/best_ever_genome.pt', 'results/acrobot_dm_flex_s45_300g/best_gameplay.mp4'),
        ('Pendulum-v1', 'results/pendulum_dm_flex_s45_1000g/best_ever_genome.pt', 'results/pendulum_dm_flex_s45_1000g/best_gameplay.mp4'),
        ('LunarLander-v3', 'results/lunar_s45_300g_fleet/best_ever_genome.pt', 'results/lunar_s45_300g_fleet/best_gameplay.mp4'),
        ('CarRacing-v3', 'results/carracing_dm_s45_100g_fleet/best_ever_genome.pt', 'results/carracing_dm_s45_100g_fleet/best_gameplay.mp4'),
    ]
    
    for env_name, gpath, opath in runs:
        render_env(env_name, gpath, opath, seed=42)
    
    print('\nAll done!')
