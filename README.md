# rust-mlp

A small MLP (multi-layer perceptron) library built from scratch in Rust.

This is a learning project with an emphasis on:
- clear data/layout decisions
- allocation-free hot paths (forward/backward)
- tests and numerical gradient checks

## Quick start

Run the example training loop:

```bash
cargo run --example tanh_sum
```

Run tests (includes gradient checks):

```bash
cargo test
```

Run benchmarks:

```bash
cargo bench
```

## Design conventions

- `f32` everywhere.
- Forward/backward operate on slices (`&[f32]`, `&mut [f32]`), not `Vec`, to avoid hidden allocations.
- `Scratch` stores forward activations.
- `Gradients` stores parameter gradients and backprop buffers.

## Roadmap

See `PLAN.md`.
