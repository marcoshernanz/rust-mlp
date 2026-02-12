# rust-mlp

[![CI](https://github.com/marcoshernanz/rust-mlp/actions/workflows/ci.yml/badge.svg)](https://github.com/marcoshernanz/rust-mlp/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/rust-mlp.svg)](https://crates.io/crates/rust-mlp)
[![docs.rs](https://img.shields.io/docsrs/rust-mlp)](https://docs.rs/rust-mlp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small, from-scratch multi-layer perceptron (MLP) library for Rust.

This crate is intentionally "small-core": dense layers, common activations/losses, a predictable training loop, and a performance model you can reason about.

Highlights:

- Allocation-free hot path: reuse `Scratch` / `Gradients`.
- Batched compute: `forward_batch` / `backward_batch` with an optional faster GEMM backend.
- Determinism: seeded init + deterministic shuffling.
- Practical API: `fit`, `evaluate`, `predict`, `predict_into`.
- Optional JSON model I/O (feature: `serde`).

## Install

```toml
[dependencies]
rust-mlp = "0.1"
```

Optional features:

```toml
[dependencies]
rust-mlp = { version = "0.1", features = ["matrixmultiply", "serde"] }
```

## Quick start (train + evaluate)

```rust
use rust_mlp::{Activation, Dataset, FitConfig, Loss, Metric, MlpBuilder, Optimizer, Shuffle};

fn main() -> rust_mlp::Result<()> {
    // XOR toy dataset.
    let xs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let ys = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    let train = Dataset::from_rows(&xs, &ys)?;

    let mut mlp = MlpBuilder::new(2)?
        .add_layer(8, Activation::ReLU)?
        .add_layer(1, Activation::Sigmoid)?
        .build_with_seed(0)?;

    let report = mlp.fit(
        &train,
        None,
        FitConfig {
            epochs: 200,
            lr: 0.1,
            batch_size: 4,
            shuffle: Shuffle::Seeded(0),
            optimizer: Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
            loss: Loss::Mse,
            metrics: vec![Metric::Accuracy],
            ..FitConfig::default()
        },
    )?;

    let last = report.epochs.last().unwrap();
    println!("final train loss={}", last.train.loss);

    let eval = mlp.evaluate(&train, Loss::Mse, &[Metric::Accuracy])?;
    println!("evaluate: loss={} metrics={:?}", eval.loss, eval.metrics);
    Ok(())
}
```

## Allocation-free inference

`Mlp::forward` is allocation-free if you reuse `Scratch`. For shape-checked inference (returns `Result`), use `predict_into`.

```rust
use rust_mlp::{Activation, MlpBuilder};

fn main() -> rust_mlp::Result<()> {
    let mlp = MlpBuilder::new(3)?
        .add_layer(8, Activation::Tanh)?
        .add_layer(2, Activation::Identity)?
        .build_with_seed(0)?;

    let mut scratch = mlp.scratch();
    let mut out = vec![0.0_f32; mlp.output_dim()];

    let x = [0.1_f32, -0.2, 0.3];
    mlp.predict_into(&x, &mut scratch, &mut out)?;
    println!("y={out:?}");
    Ok(())
}
```

## Feature flags

- `matrixmultiply`: use the `matrixmultiply` crate as a faster GEMM backend for batched compute.
- `serde`: enable JSON model serialization via `serde` + `serde_json`.

## Data layout and shapes

- All scalars are `f32`.
- `Dataset` / `Inputs` store samples contiguously in row-major layout.
  - inputs shape: `(len, input_dim)` stored as `len * input_dim` flat
  - targets shape: `(len, target_dim)` stored as `len * target_dim` flat

## Performance

- `fit` allocates its buffers once and reuses them across steps.
- When `batch_size > 1`, `fit` uses a batched GEMM-based path for full-size batches.
- For meaningful batched performance, enable the `matrixmultiply` feature (otherwise a simple scalar GEMM is used).
- Benchmarks:

```bash
cargo bench
cargo bench --features matrixmultiply
```

## Examples

```bash
cargo run --example tanh_sum
cargo run --example xor_relu
cargo run --example softmax_3class
cargo run --example save_load_json --features serde
```

## MSRV

MSRV is specified in `Cargo.toml`.

## License

MIT. See `LICENSE`.
