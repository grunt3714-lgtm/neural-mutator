# Task: Get distributed fleet training running with multi-food snake

## What to build

### 1. Multi-food snake environment
- Modify `src/snake_env.py`: spawn 5 food items at a time instead of 1
- Each food gives +1 reward when eaten, respawn a new food immediately (maintain 5 on grid)
- All food must respect min Manhattan distance 4 from head
- Update the Rust core (`native/snake_gpu_rs/src/lib.rs`) to match — 5 foods, same distance rule, randomized placement
- The WGSL shader and Rust CPU path both need updating
- After modifying Rust core, rebuild with `maturin develop --release`

### 2. Fix fleet reliability
- `fleet_zmq/worker.py`: wrap job execution in try/except — on error return fitness -999.0, don't crash
- `fleet_zmq/evaluator.py`: handle missing/failed results gracefully
- Workers should reconnect automatically if they lose connection

### 3. Deploy and run
- Use OpenClaw RPC (NOT SSH) to manage nodes. The tool is called `nodes` with action `run`:
  - Nodes: "Inspiron 5570" (192.168.1.95), "gruntnode2" (192.168.1.99), "gruntnode3" (192.168.1.112), "Inspiron 5570 Node 4" (192.168.1.100)
  - RPC has 30s timeout — keep commands short
- ALL nodes must use the SAME Rust `snake_gpu_rs` core
- Build wheel on gateway, then SCP + pip install on each node
- The wheel filename MUST keep its full name: `snake_gpu_rs-0.1.0-cp312-cp312-manylinux_2_35_x86_64.whl`
- Verify each node: `cd ~/neural-mutator && .venv/bin/python -c 'from snake_gpu_rs import SnakeGpuCore; print("ok")'`
- Kill old python processes on nodes, launch fresh workers, then launch master on gateway
- Training: 1000 gens, pop 20, episodes 3, seed 42, gaussian mutator, no speciation, --fleet tcp://*:5555

### 4. Verify
- Confirm Gen lines appearing in train log
- All workers connecting and returning results

## File locations
- Snake env: `src/snake_env.py`
- Rust core: `native/snake_gpu_rs/src/lib.rs`
- Fleet evaluator: `fleet_zmq/evaluator.py`
- Fleet worker: `fleet_zmq/worker.py`
- Train script: `src/train.py`
- Venv: `.venv/bin/python` (gateway), `~/neural-mutator/.venv/bin/python` (nodes)

## Do NOT change
- Training algorithm, genome.py, observation space (768-dim pixels)
- Don't use SSH — use OpenClaw RPC (`nodes` tool with `system.run`)
