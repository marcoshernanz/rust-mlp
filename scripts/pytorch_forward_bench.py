#!/usr/bin/env python3

import argparse
import importlib
import os
import time
import warnings


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Failed to initialize NumPy:.*",
        category=UserWarning,
    )

    parser = argparse.ArgumentParser(
        description="PyTorch batched forward benchmark (MLP: Linear + Tanh + ... + Linear)."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    # CPU-only benchmark for apples-to-apples comparison with this crate.
    args = parser.parse_args()

    # Keep PyTorch stable and comparable.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))

    try:
        torch = importlib.import_module("torch")
    except Exception as e:
        raise SystemExit(
            "PyTorch is required for this benchmark. Install it first, e.g. `pip install torch`. "
            f"Import error: {e}"
        )

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except Exception:
        # Not all builds allow setting interop threads at runtime.
        pass
    torch.manual_seed(args.seed)

    batch_size = args.batch_size
    iters = args.iters
    warmup = args.warmup
    in_dim = args.in_dim
    hidden = args.hidden
    layers = args.layers
    out_dim = args.out_dim

    if batch_size <= 0 or iters <= 0:
        raise ValueError("batch-size and iters must be > 0")
    if in_dim <= 0 or hidden <= 0 or out_dim <= 0:
        raise ValueError("in-dim/hidden/out-dim must be > 0")

    mods = []
    d = in_dim
    for _ in range(layers):
        mods.append(torch.nn.Linear(d, hidden, bias=True))
        mods.append(torch.nn.Tanh())
        d = hidden
    mods.append(torch.nn.Linear(d, out_dim, bias=True))
    model = torch.nn.Sequential(*mods)
    model.eval()

    # Deterministic, non-constant inputs.
    x = torch.arange(batch_size * in_dim, dtype=torch.float32).reshape(
        batch_size, in_dim
    )
    x = x * 1e-3

    # Warmup.
    with torch.inference_mode():
        y = None
        for _ in range(warmup):
            y = model(x)
        assert y is not None

        t0 = time.perf_counter()
        y = None
        for _ in range(iters):
            y = model(x)
        t1 = time.perf_counter()
        assert y is not None
        checksum = float(y[0, 0])

    elapsed_s = t1 - t0
    total_samples = iters * batch_size
    samples_per_s = total_samples / elapsed_s
    iters_per_s = iters / elapsed_s

    print(
        "pytorch forward_batch "
        f"batch_size={batch_size} iters={iters} warmup={warmup} "
        f"in_dim={in_dim} hidden={hidden} layers={layers} out_dim={out_dim} "
        f"threads={args.threads} elapsed_s={elapsed_s:.6f} "
        f"iters_per_s={iters_per_s:.2f} samples_per_s={samples_per_s:.2f} "
        f"checksum={checksum}"
    )


if __name__ == "__main__":
    main()
