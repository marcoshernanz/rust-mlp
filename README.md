# rust-mlp

A small, from-scratch multi-layer perceptron (MLP) library in Rust.

The crate is intentionally "small-core": it focuses on dense feed-forward networks, a simple training loop, and a predictable performance model.

## Goals

- Clear API and data layout.
- Allocation-free hot path for per-sample forward/backward.
- Deterministic initialization (seeded RNG) and strong tests (including numerical gradient checks).

## Non-goals (for now)

- GPU support.
- Autodiff graph engine.
- Large model zoo / high-level framework features.

## Quick start

Run the example training loop:

```bash
cargo run --example tanh_sum
```

Run tests (includes gradient checks):

```bash
cargo test
```

## Library tour

The main public types are:

- `Mlp`: the model (stack of dense layers).
- `Scratch`: reusable activation buffers for `Mlp::forward`.
- `Gradients`: reusable gradient buffers for `Mlp::backward`.
- `Trainer`: convenience wrapper holding `Scratch + Gradients`.
- `Dataset` / `Inputs`: validated contiguous row-major data holders.
- `FitConfig` / `FitReport`: high-level training configuration and output.

## API overview

Two layers of API exist:

1) Low-level, allocation-free, "hot path":

- `Mlp::forward(input, &mut scratch) -> &[f32]`
- `Mlp::backward(input, &scratch, &mut grads) -> &[f32]`
- `Mlp::sgd_step(&grads, lr)`

These methods treat shape mismatches as programmer error and will panic via `assert!`.

2) High-level convenience:

- `Mlp::fit(&Dataset, Option<&Dataset>, FitConfig) -> Result<FitReport>`
- `Mlp::evaluate(&Dataset, Loss, &[Metric]) -> Result<EvalReport>`
- `Mlp::predict(&Dataset) -> Result<Vec<f32>>`
- `Mlp::predict_inputs(&Inputs) -> Result<Vec<f32>>`
- `Mlp::evaluate_mse(&Dataset) -> Result<f32>`

These validate inputs and return `Result`.

## Example: train + evaluate

```rust
use rust_mlp::{Activation, Dataset, FitConfig, Loss, Metric, MlpBuilder};

fn main() -> rust_mlp::Result<()> {
    // XOR-ish tiny dataset (for demonstration).
    let xs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let ys = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    let train = Dataset::from_rows(&xs, &ys)?;

    // 2 -> 8 (tanh) -> 1 (identity) MLP with deterministic initialization.
    let mut mlp = MlpBuilder::new(2)?
        .add_layer(8, Activation::Tanh)?
        .add_layer(1, Activation::Identity)?
        .build_with_seed(123)?;

    let report = mlp.fit(
        &train,
        None,
        FitConfig {
            epochs: 200,
            lr: 0.1,
            batch_size: 4,
            shuffle: rust_mlp::Shuffle::Seeded(0),
            lr_schedule: rust_mlp::LrSchedule::Constant,
            optimizer: rust_mlp::Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
            weight_decay: 0.0,
            grad_clip_norm: None,
            loss: Loss::Mse,
            metrics: vec![Metric::Mse],
        },
    )?;

    let mse = mlp.evaluate_mse(&train)?;
    let last = report.epochs.last().unwrap();
    println!("final train loss (from fit): {}", last.train.loss);
    println!("train MSE: {mse}");
    Ok(())
}
```

## Example: allocation-free inference

If you want to avoid allocating the output buffer on each call, reuse `Scratch` and copy the returned slice.

```rust
use rust_mlp::{Activation, MlpBuilder};

fn main() -> rust_mlp::Result<()> {
    let mlp = MlpBuilder::new(3)?
        .add_layer(5, Activation::Tanh)?
        .add_layer(2, Activation::Identity)?
        .build_with_seed(0)?;
    let mut scratch = mlp.scratch();

    let x = [0.1_f32, -0.2, 0.3];
    let y = mlp.forward(&x, &mut scratch);
    let y_owned = y.to_vec();
    println!("y = {y_owned:?}");
    Ok(())
}
```

## Performance model

- The per-sample `forward`/`backward` path does not allocate if you reuse `Scratch`/`Gradients`.
- `Dataset`/`Inputs` store data contiguously (row-major) to keep slice access cheap.
- Training (`fit`) allocates buffers once and reuses them across all steps.

## Determinism

- Use `MlpBuilder::build_with_seed` for deterministic initialization.

## Panics vs. `Result`

- Constructors and high-level helpers validate and return `Result`.
- Low-level per-sample methods (`forward`, `backward`, `sgd_step`, and loss helpers) panic on shape mismatches.

## Roadmap

See `ROADMAP.md` for the production-readiness plan.
