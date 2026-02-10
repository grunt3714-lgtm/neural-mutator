#!/bin/bash
set -e
cd /home/grunt/.openclaw/workspace/Projects/neural-mutator
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p results/logs

ENVS="CartPole-v1 Acrobot-v1 MountainCarContinuous-v0 Pendulum-v1 LunarLander-v3"
MUTATORS="chunk transformer cppn gaussian"

for env in $ENVS; do
  for mut in $MUTATORS; do
    logfile="results/logs/${env}_${mut}.log"
    echo "=== Starting $env / $mut ==="
    python -m src.train --env "$env" --mutator "$mut" --generations 100 --pop-size 30 \
      > "$logfile" 2>&1 &
  done
  # Run 4 at a time (one per env batch), wait between env batches
  wait
  echo "=== Completed all mutators for $env ==="
done

echo "=== ALL RUNS COMPLETE ==="
