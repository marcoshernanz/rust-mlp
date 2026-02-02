# Roadmap

This document is the execution plan to move `rust-mlp` from a learning implementation into a small, reliable, production-grade Rust crate for feed-forward MLP inference and training.

Scope note: this crate intentionally stays "small-core" (dense layers + common activations + common losses + SGD/Adam + evaluation + serialization). It is not intended to compete with full deep learning frameworks.

## North Star

- Ergonomics: end users can `fit`, `evaluate`, and `predict` with minimal ceremony.
- Correctness: shape safety, numerically stable loss/activations, reproducibility, and strong tests.
- Performance: avoid per-step allocations; support batched compute; provide a clear path to SIMD/BLAS.
- Maintenance: clear API boundaries, semver discipline, CI, docs, and benchmarks.

## Current Baseline (What Exists)

- Dense `Layer` and `Mlp` with `tanh` activation.
- Per-sample training loop with MSE and SGD.
- Reusable `Scratch`/`Gradients` to avoid per-step allocation.
- Dataset helpers (`Dataset`, `Inputs`) with validation.
- Gradient-check tests and determinism tests.

## Decisions To Lock In Early

These decisions reduce churn and clarify API boundaries.

1. Floating point type
   - Decision: keep `f32` as the primary type for v1.
   - Rationale: speed, memory footprint; most learning examples use `f32`.
   - Follow-up: design internals so adding `f64` later is possible (but not promised).

2. Memory layout
   - Decision: weights stored row-major as `[out][in]` contiguous `Vec<f32>`.
   - Rationale: simple indexing; compatible with many GEMM backends.

3. Activation configuration
   - Decision: activation is per-layer and part of the model definition.
   - Rationale: fixed `tanh` is too limiting; per-layer is the most common need.

4. Training semantics
   - Decision: v1 supports both per-sample and mini-batch training.
   - Rationale: production training generally requires batching for throughput and stability.
   - Note: keep the per-sample path (useful for teaching and tiny data).

5. Error handling policy
   - Decision: constructors and high-level APIs validate and return `Result`.
   - Decision: low-level hot paths (per-sample forward/backward/step) panic on misuse
     (shape mismatches) via `assert!` / `assert_eq!`.

6. Determinism
   - Decision: provide deterministic init and deterministic training order when requested.
   - Rationale: reproducible experiments and stable tests.

## Milestones

Each milestone should be deliverable, tested, and documented.

### M0: Hardening The Existing Core

Goal: make the current API safe and predictable without changing the learning-oriented structure.

- Public API audit
  - Keep constructors and high-level APIs validating and returning `Error`.
  - Keep low-level per-sample APIs panicking on misuse with clear `assert!` messages.

- Numeric stability and invariants
  - Ensure `mse_backward` uses a consistent normalization convention (document it).
  - Document gradient overwrite semantics vs accumulation semantics.

- Documentation
  - Add crate-level docs describing the data layout, training flow, and performance model.
  - Ensure examples compile and run as part of CI.

Deliverables
- Updated docs and error messages.
- CI running `cargo test`, `cargo fmt --check`, `cargo clippy`.

Status: completed.

### M1: Model Definition 2.0 (Activations, Shapes, Builders)

Goal: a clean, explicit model definition that supports common networks.

- Activation enum
  - `Activation::Tanh`, `ReLU`, `LeakyReLU{alpha}`, `Sigmoid`, `Identity`.
  - Forward/backward implementations (derivative expressed in terms of cached outputs where possible).
  - Decide caching strategy: store post-activation outputs (current approach) and compute derivatives from them.

- Layer spec and model builder
  - Add `MlpBuilder` or an `Mlp::from_sizes` constructor that takes a list of layer sizes plus activations.
  - Provide a "sane defaults" initializer for each activation:
    - Tanh/Sigmoid: Xavier/Glorot.
    - ReLU/LeakyReLU: He/Kaiming.
    - Identity: Xavier or small uniform.
  - Keep `new_with_seed` for deterministic init.

- Shape-safe inference API
  - Add a non-allocating prediction API:
    - `predict_one_into(&self, input: &[f32], scratch: &mut Scratch, out: &mut [f32]) -> Result<()>`.
  - Keep an ergonomic allocating wrapper for convenience.

- Documentation polish
  - Add rustdoc examples for common flows (`fit`, `predict`, low-level `forward`/`backward`).
  - Add a clear "Panics vs Result" section in crate docs and README.

Deliverables
- Per-layer activations in `Mlp`.
- Builder with clear error messages.
- Updated tests and at least one new example (e.g. XOR with ReLU).

### M2: Losses + Metrics + Evaluation API

Goal: support classification and standard reporting.

- Losses
  - Regression: MSE (existing), MAE (optional).
  - Binary classification:
    - `BinaryCrossEntropyWithLogits` (preferred) for numerical stability.
  - Multi-class classification:
    - `SoftmaxCrossEntropy` (stable log-sum-exp).
  - Decision: for classification, expose "with logits" losses to avoid separate sigmoid/softmax layers.

- Metrics
  - Regression: MSE, MAE.
  - Binary: accuracy, AUROC (optional; can be future).
  - Multi-class: accuracy, top-k accuracy.

- Reporting
  - `FitReport` should include:
    - train loss over epochs
    - optional validation loss
    - metric values if configured
    - early stopping reason if triggered

Deliverables
- Loss trait or enum used by the training loop.
- Stable softmax/xent implementation with tests.
- Example: simple 3-class classification.

### M3: Mini-batching + Accumulation

Goal: throughput and realistic training.

- Batch data model
  - Add `Batch<'a>` views over contiguous X/Y.
  - Ensure batch dimensions are validated once.

- Batched forward/backward
  - Add `forward_batch` and `backward_batch`:
    - Inputs: `[batch, in]`, outputs: `[batch, out]`.
  - Update `Scratch`/`Gradients` to support batch-shaped buffers.
  - Add gradient accumulation over batch and normalization conventions (sum vs mean).
  - Provide both:
    - per-sample update (current)
    - batch update (preferred)

- Data shuffling
  - Deterministic shuffling via seeded RNG.
  - Options: no shuffle / shuffle each epoch.

Deliverables
- `fit` supports `batch_size`.
- Benchmarks show improved throughput with batching.

### M4: Optimizers + Regularization

Goal: standard training knobs.

- Optimizers
  - SGD with momentum
  - Adam (with bias correction)
  - Decision: store optimizer state separately from `Mlp` (e.g. `AdamState`).

- Regularization
  - L2 weight decay (decoupled for AdamW).
  - Gradient clipping (global norm).
  - Dropout (training-only; requires RNG and mask caching) (optional; can be M5 if too large).

- Learning rate schedules
  - Constant (default)
  - Step decay
  - Cosine annealing (optional)

Deliverables
- `FitConfig` supports optimizer selection and common options.
- Tests for optimizer updates (small deterministic cases).

### M5: Serialization + Model Portability

Goal: save/load models safely and reproducibly.

- `serde` support
  - Add `serde` feature flag.
  - Serialize: layer sizes, activations, weights, biases, init metadata (optional).

- Versioning
  - Include a model format version field.
  - Validate on load; provide meaningful errors.

- Interop (future)
  - Optional: simple CSV/NPY export of weights.

Deliverables
- `Mlp::save_json`, `Mlp::load_json` (or `serde_json` example).
- Golden-file tests for stable format.

### M6: Performance Track (CPU)

Goal: reach "fast enough" CPU performance for small-to-medium MLPs.

- Allocation-free hot path
  - Verify: `fit` does not allocate per step (use benchmarks/profiling).
  - Ensure `Scratch` and `Gradients` can be reused across calls.

- Batched matmul backend
  - Add a backend abstraction:
    - naive scalar
    - optional `matrixmultiply` crate
    - optional BLAS (feature-gated)
  - Choose default: naive first; upgrade behind features.

- Parallelism (optional)
  - `rayon` for batch-level parallelism (feature-gated).
  - Ensure deterministic option disables parallel non-determinism.

Deliverables
- Benchmarks for forward/backward batch sizes.
- Clear feature flags and documentation.

### M7: API Polish + Stability For 1.0

Goal: finalize a stable surface.

- Public API review
  - Reduce the number of public types needed for common use.
  - Keep low-level APIs available, but document them as "advanced".

- Consistency
  - Naming: `fit`, `evaluate`, `predict`, `predict_into`.
  - Config structs: avoid long argument lists; validate with good errors.

- MSRV policy
  - Decide MSRV and enforce in CI.

Deliverables
- `CHANGELOG.md` and a semver plan.
- `1.0.0` checklist completed.

## Engineering Workstreams (Cross-Cutting)

These are not single milestones; they should be addressed incrementally.

### Testing Strategy

- Unit tests for:
  - activations forward/backward
  - each loss forward/backward
  - optimizer step correctness

- Numeric gradient checking
  - Keep existing checks; extend to classification losses.
  - Use small random models with deterministic seeds.

- Property-based tests (future)
  - Shape validation invariants.
  - Determinism with fixed seeds.

- Fuzzing (future)
  - Input validation and panics in debug/release.

### Documentation and Examples

- Examples should mirror typical user goals:
  - regression (current tanh_sum)
  - binary classification
  - multi-class classification
  - save/load model

- Include a "Performance" section:
  - how to reuse scratch buffers
  - batch sizing guidance
  - feature flags for faster backends

### CI / Quality Gates

- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test`
- Benchmarks are not gating by default, but keep a local benchmarking guide.

### Error Design

- Expand error variants carefully:
  - `InvalidConfig { what }`
  - `InvalidData { what }`
  - `Serde { what }` behind feature
  - Avoid exposing internal crate types in errors.

## Proposed "1.0" Feature Set

- Dense MLP with per-layer activations
- Regression and classification losses (stable implementations)
- Per-sample and mini-batch training
- Optimizers: SGD, SGD+momentum, Adam
- Metrics: MSE/MAE, accuracy
- Serialization behind `serde` feature
- Strong docs/examples + CI

## Non-Goals (For 1.0)

- GPU support
- Convolutional / recurrent architectures
- Autodiff graph engine
- Mixed precision training
- Distributed training

## Tracking

If you want this roadmap to become actionable in the repo, the next step is to turn milestones into GitHub issues with:

- acceptance criteria
- API sketch (function signatures)
- tests to add
- benchmark expectations (if applicable)
