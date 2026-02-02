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

This uses the high-level API:
- build `Dataset`
- `mlp.fit(...)`
- `mlp.evaluate_mse(...)`

If you only have inputs (no targets), build `Inputs` and call `mlp.predict_inputs(...)`.

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

See `ROADMAP.md` for the production-readiness plan and `PLAN.md` for the learning checklist.
