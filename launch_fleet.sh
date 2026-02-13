#!/bin/bash
set -e

NODES="192.168.1.95 192.168.1.99 192.168.1.112 192.168.1.100"
COORD_ADDR="tcp://192.168.1.94:5555"
RESULTS_DIR="results/fleet_gaussian_1000g"

mkdir -p "$RESULTS_DIR"

# Launch workers on nodes (background SSH)
for node in $NODES; do
  name="node-$(echo $node | awk -F. '{print $4}')"
  echo "Launching worker on $node ($name)..."
  ssh -o ConnectTimeout=5 -o BatchMode=yes -f grunt@$node \
    "cd ~/neural-mutator && nohup .venv/bin/python -u fleet_zmq/worker.py --connect $COORD_ADDR --name $name --local-workers 4 > /tmp/worker.log 2>&1 &"
done

echo "Workers launched, waiting 3s..."
sleep 3

# Launch coordinator
echo "Starting coordinator..."
exec .venv/bin/python -u -m src.train \
  --env SnakePixels-v0 \
  --generations 1000 \
  --pop-size 20 \
  --mutator gaussian \
  --fleet tcp://*:5555 \
  --fleet-workers 3 \
  --fleet-batch 5 \
  --seed 42 \
  --output "$RESULTS_DIR"
