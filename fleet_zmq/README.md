# fleet_zmq

ZeroMQ-based master/worker evaluation layer (no OpenClaw dependency).

## Ports
- Master binds: `tcp://*:5555` (default)
- Workers connect to: `tcp://<node1-ip>:5555`

## Run

On **node1 (master)**:
```bash
cd Projects/neural-mutator
source .venv/bin/activate
python fleet_zmq/master.py --bind tcp://*:5555
```

On each **worker node** (gateway/node2/node3/node4):
```bash
cd Projects/neural-mutator
source .venv/bin/activate
python fleet_zmq/worker.py --connect tcp://192.168.1.95:5555 --name gateway --local-workers 4
```

This is currently a standalone eval service; integration into `src.train` comes next.
