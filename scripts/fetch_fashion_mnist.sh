#!/usr/bin/env bash
# Fetch Fashion-MNIST (60k train + 10k test, 28x28, 10 clothing classes).
# Source: zalandoresearch/fashion-mnist. IDX ubyte format, same as MNIST.
# Data lands in tuplet/data/fashion/ which is gitignored.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/fashion"
BASE="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"

mkdir -p "$DEST"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ -f "$DEST/$f" ]; then
    echo "have $f"
  else
    echo "fetching $f.gz"
    curl -sSL --fail -o "$DEST/$f.gz" "$BASE/$f.gz"
    gunzip -f "$DEST/$f.gz"
  fi
done
echo "done -> $DEST"
