# Scripts

This directory contains small utilities used for local benchmarking and validation.

## PyTorch forward benchmark

`scripts/pytorch_forward_bench.py` runs a CPU-only, batched-forward benchmark for a
simple MLP (Linear + Tanh + ... + Linear).

Example:

```bash
python3 scripts/pytorch_forward_bench.py \
  --batch-size 128 --iters 2000 --warmup 200 \
  --in-dim 128 --hidden 256 --layers 2 --out-dim 10 --threads 1
```

Notes:

- This is a throughput sanity check, not a "winner" claim.
- For a fair comparison, keep dtype/shapes the same and control CPU threads.
