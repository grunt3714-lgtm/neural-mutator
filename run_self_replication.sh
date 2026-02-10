#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

OUTDIR="results/self_replication"
mkdir -p "$OUTDIR"

# Gaussian baseline
echo "=== Gaussian baseline ==="
python -m src.train --mutator gaussian --generations 100 --pop-size 30 --output "$OUTDIR" --quine-lambda 0.0 2>&1 | tee "$OUTDIR/log_gaussian.txt"

# Chunk mutator with different quine lambdas
for ql in 0.0 0.1 0.5; do
    echo "=== Chunk ql=$ql ==="
    python -m src.train --mutator chunk --generations 100 --pop-size 30 --quine-lambda $ql --output "$OUTDIR" 2>&1 | tee "$OUTDIR/log_chunk_ql${ql}.txt"
done

# Transformer mutator with different quine lambdas
for ql in 0.0 0.1 0.5; do
    echo "=== Transformer ql=$ql ==="
    python -m src.train --mutator transformer --generations 100 --pop-size 30 --quine-lambda $ql --output "$OUTDIR" 2>&1 | tee "$OUTDIR/log_transformer_ql${ql}.txt"
done

echo "=== All runs complete ==="
echo "Generating comparison plots..."
python plot_self_replication.py
