# Agent Notes (rust-mlp)

This file is for agentic coding tools operating in this repository.

Repo snapshot:
- Rust edition: 2024 (`Cargo.toml`)
- Crate: `rust-mlp` (library-first; `src/main.rs` is a tiny helper binary)
- Core focus: small MLP (dense layers + tanh), allocation-free per-sample hot path

## Commands (Build / Lint / Test)

CI runs (see `.github/workflows/ci.yml`):

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

Common local commands:

```bash
# Fast compile check
cargo check

# Build
cargo build
cargo build --release

# Format
cargo fmt

# Lint (match CI)
cargo clippy --all-targets --all-features -- -D warnings

# Tests
cargo test

# Docs
cargo doc --open
```

Run a single unit test (recommended patterns):

```bash
# Substring match (fastest to type)
cargo test backward_matches_numeric_gradients

# Exact match (best for disambiguation)
cargo test mlp::tests::backward_matches_numeric_gradients -- --exact

# Show stdout/stderr
cargo test mlp::tests::backward_matches_numeric_gradients -- --exact --nocapture

# Run tests matching a module/file
cargo test layer::tests::
```

Note: `cargo test --all-targets` will also build/run benches and examples as test targets.
Use plain `cargo test` for the tight inner loop.

Examples: `cargo run --example tanh_sum`

Benchmarks (Criterion):

```bash
cargo bench
cargo bench --bench mlp -- mlp_forward
```

## Code Style and Conventions

### Formatting

- Use `rustfmt` (no custom config in this repo). Run `cargo fmt` before finalizing changes.
- Keep lines readable; let rustfmt handle wrapping.

### Imports

- Prefer grouping imports by origin (typical order): `std::...`, external crates, then `crate::...`.
- Prefer `{}` import lists for multiple items: `use crate::{Error, Result};`.
- `use super::*;` is fine inside `#[cfg(test)] mod tests`.

### Types and Numerics

- Scalars are `f32` throughout the crate.
- Dimensions/indices use `usize` (`in_dim`, `out_dim`, `input_dim`, `target_dim`, `len`).
- Deterministic seeds use `u64` and `StdRng::seed_from_u64`.
- Prefer `mul_add` in inner loops where it is already used (dot products).
- Keep weight/data layout decisions consistent:
  - Layer weights: row-major `(out_dim, in_dim)` contiguous `Vec<f32>`.
  - Dataset/inputs: row-major contiguous buffers.

### Naming

- Types: `PascalCase` (`Mlp`, `Layer`, `Scratch`, `Gradients`, `FitConfig`).
- Methods/vars: `snake_case`.
- Gradients use `d_` prefix (`d_weights`, `d_biases`, `d_input`, `d_output`).
- Use `idx` for indices in loops; use `len()` for counts.

### API Layers (Panics vs Result)

The crate intentionally has two layers of API:

- Low-level, allocation-free hot path:
  - `Mlp::forward`, `Mlp::backward`, `Mlp::sgd_step`
  - `Layer::forward`, `Layer::backward`, `Layer::sgd_step`
  - loss helpers (e.g. `loss::mse_backward`)
  These treat shape mismatches as programmer error and MUST panic on misuse.
  Use `assert!` / `assert_eq!` with clear messages (expected vs actual).

- High-level convenience APIs:
  - `Mlp::fit`, `Mlp::predict`, `Mlp::predict_inputs`, `Mlp::evaluate_mse`
  These validate inputs and return `Result` with `Error::InvalidData` / `Error::InvalidConfig`.

Avoid adding duplicate "try_*" APIs; keep one obvious way to do things.

### Error Handling

- Use the crate error types: `crate::Result<T>` and `crate::Error` (`src/error.rs`).
- Use:
  - `Error::InvalidConfig` for hyperparameters/model configuration (e.g. `epochs == 0`, `lr <= 0`).
  - `Error::InvalidData` for dataset/inputs issues (empty sets, dimension mismatch, bad shapes).
- Prefer actionable error strings that include the offending values.

### Performance Guidelines

- Hot paths should be allocation-free when buffers are reused.
  - Use `Scratch` and `Gradients` from `Mlp::scratch()` / `Mlp::gradients()`.
  - Avoid creating temporary `Vec`s inside per-sample loops.
- Use `Vec::with_capacity` when building contiguous buffers.

### Tests

- Keep tests deterministic (use fixed seeds).
- Numerical gradient checks exist; keep tolerances reasonable and avoid flakiness.
- Panic tests (`#[should_panic]`) should be minimal and fast.

### Documentation

- Each module uses a short `//!` module doc at the top.
- Public types/functions should have rustdoc comments, especially describing:
  - shape contracts
  - overwrite vs accumulation semantics
  - panic vs `Result` behavior

## Cursor / Copilot Instructions

No Cursor rules were found (`.cursor/rules/` or `.cursorrules`).
No Copilot instructions were found (`.github/copilot-instructions.md`).
