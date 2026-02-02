# Plan: Build a Small, Professional MLP Library (Rust)

This repository is a learning project. The goal is to build a small but serious multi-layer perceptron (MLP) library with a clean API, correct math, and good performance characteristics.

We intentionally take small steps. Each step should compile, be testable, and teach one idea.

## Guiding principles

- Correctness first, then performance, then features.
- Avoid hidden allocations in hot paths.
- Make shapes explicit and checked (at least in debug builds).
- Prefer simple, explicit data layouts (contiguous `Vec<f32>` buffers).
- Keep modules small and responsibility-driven.

## Step 0: Project scaffold (what we just did)

Goal: Create a library layout we can grow without churn.

- `src/neuron.rs`: a single neuron (affine + activation).
- `src/layer.rs`: placeholder for a full dense layer.
- `src/mlp.rs`: placeholder for a multi-layer model.
- `src/lib.rs`: crate root that exports modules.

Why: In professional codebases, the executable (`main.rs`) is not where core ML logic lives. The library layout makes it easy to add tests/benches and evolve APIs.

## Step 1: Define the core math contracts (shapes and invariants)

Goal: Decide what every component promises.

- Choose conventions:
  - Inputs are `&[f32]` (borrowed slice).
  - Parameters live in contiguous `Vec<f32>`.
  - Debug-time shape assertions (`debug_assert_eq!`).
- Decide activation story:
  - Start with `tanh` baked in (simple), later generalize.

Background: Backprop is just the chain rule applied to a computation graph. Most bugs come from shape mismatches and unclear ownership of buffers.

Deliverables:
- Document invariants for `Neuron`, then reuse those patterns for `Layer` and `Mlp`.

## Step 2: Implement `Layer` as a dense matrix-vector transform

Goal: Move from “neuron-per-neuron” to a professional, efficient layer.

- Implement `Layer` with:
  - `weights: Vec<f32>` storing an `(out_dim x in_dim)` matrix in a defined order (row-major is typical).
  - `biases: Vec<f32>` of length `out_dim`.
- Implement `forward(&self, inputs: &[f32], outputs: &mut [f32])`:
  - No allocations.
  - Shape checks in debug.
  - Apply activation (`tanh`) elementwise.

Background:
- A real MLP should not store a `Vec<Neuron>` for performance. A single contiguous weight matrix is better for cache locality and vectorization.

## Step 3: Backward for `Layer` (single-sample)

Goal: Implement correct gradients for weights, biases, and inputs.

- API design:
  - Backward writes into caller-provided buffers (`&mut [f32]`) to avoid allocations.
  - Choose gradient semantics:
    - Overwrite: each call writes gradients for exactly one sample.
    - Accumulate: each call adds into existing gradient buffers (for batching).
- Minimal caching:
  - Backward needs either pre-activation `z` or post-activation `y`.
  - We can store `y` from forward (or recompute with a cost). For learning, start explicit: pass `outputs` from forward into backward.

Background:
- For `tanh`: `d/dz tanh(z) = 1 - tanh(z)^2`. Using the forward output `y` makes it cheap: `1 - y*y`.

### Overwrite vs accumulate (when to use each)

- Overwrite gradients:
  - Best when you train with pure SGD on single samples, or when you want the simplest mental model.
  - The caller controls whether to sum/average across samples by looping and storing the result.
  - Easy to test and reason about.

- Accumulate gradients:
  - Best for minibatch training.
  - You zero gradients once per batch, then call backward for each sample with `+=` updates.
  - Lets you apply a single optimizer step using the batch-average gradient.

Professional note: many libraries implement accumulation in the module/layer itself (grads stored with params). For learning and performance, both approaches are valid; explicit buffers make allocation behavior obvious.

## Step 4: `Mlp` as a stack of layers

Goal: Compose layers into a model.

- `Mlp` holds `Vec<Layer>`.
- Provide:
  - `forward(&self, input: &[f32], scratch: &mut Scratch) -> &[f32]`
  - `backward(&self, input: &[f32], scratch: &Scratch, d_output: &[f32], grads: &mut Gradients) -> &[f32]`

Design notes:
- `Scratch` stores intermediate activations (layer outputs) from `forward`.
- `Gradients` stores parameter gradients (and any needed intermediate gradient buffers) so `backward` can run without allocations.

Background:
- For performance, you typically want reusable scratch buffers for intermediate activations/gradients. Rust ownership makes it natural to model this as an explicit `Scratch` struct.

## Step 5: Initialization

Goal: Avoid “all zeros” weights.

- Add an initialization strategy:
  - Default to Xavier/Glorot uniform init for `tanh`.
  - Keep biases at 0.0.
- Choose randomness implementation:
  - Use the `rand` crate (professional, well-tested, ergonomic).
  - Provide a deterministic constructor (seeded RNG) for reproducibility and tests.

Deliverables:
- `Init` enum (at least `Zeros` and `XavierTanh`).
- `Layer::new(...)` uses Xavier by default.
- `Layer::new_with_seed(...)` and `Mlp::new_with_seed(...)` for reproducible initialization.

Background:
- Symmetry breaking is required. With identical weights, neurons learn identical features.

## Step 6: Loss functions and training loop

Goal: Make it train end-to-end.

- Start with MSE loss for regression.
- Add a simple optimizer:
  - SGD with learning rate.
  - Optionally momentum later.

Deliverables:
- `loss::mse` and `loss::mse_backward` (writes dL/d(pred) into a caller-provided buffer).
- `optim::Sgd` (or equivalent) that applies `param -= lr * grad`.
- A small example training loop that shows the full flow:
  - `forward` -> `loss backward` -> `model backward` -> `optimizer step`

Background:
- Separating “model” from “optimizer” keeps responsibilities clear and matches most ML systems.

## Step 7: Tests, numerical gradient checks, and benchmarks

Goal: Confidence and performance proof.

- Unit tests for shape handling and deterministic outputs.
- Numerical gradient checking on small networks:
  - Compare analytical gradients vs finite differences.
- Benchmarks:
  - Forward throughput.
  - Backward throughput.

Deliverables:
- Determinism tests (seeded init produces identical outputs).
- Gradient-check tests using central differences:
  - Choose a small `epsilon` (e.g. `1e-3` for `f32`).
  - Compare `dL/d(param)` and `dL/d(input)` from backprop vs `(L(theta+e)-L(theta-e)) / (2e)`.
  - Use reasonable tolerances for `f32` (relative + absolute).
- Bench harness (Criterion) for forward and backward.

Why central differences:
- Forward difference is simpler but less accurate (error O(e)).
- Central difference is still simple and much more accurate (error O(e^2)).

Background:
- Backprop code can be “plausible but wrong”. Gradient checks are the fastest way to catch subtle errors.

## Step 8: API polish and ergonomics

Goal: Make it feel like a real library.

- Expose types from `lib.rs`.
- Add documentation and examples.
- Make feature decisions explicit (batching, f32 vs f64, activations).

Deliverables:
- Keep a clear public API surface:
  - Re-export core types (`Mlp`, `Layer`, `Gradients`, `Scratch`, `Sgd`, `Init`).
- Provide examples that compile and demonstrate intended usage.
- Tighten docs:
  - Document forward/backward contracts and required call order.
  - Document initialization defaults.
- Add sanity tests for utility modules (loss/optimizer).

## Step 9: Accumulating gradients + batching

Goal: Train efficiently with minibatches.

- Add `*_accum` variants (or a flag) for backward that uses `+=` into `d_weights/d_biases/d_inputs`.
- Define a `Batch` convention:
  - Inputs stored as `&[f32]` with shape `(batch, in_dim)` in a contiguous buffer, or as `&[&[f32]]` for simplicity.
  - Outputs/gradients stored similarly.
- Implement:
  - `zero_grads()` for the buffers (once per batch).
  - `backward_batch(...)` that loops samples and accumulates.
  - A final “scale gradients” step to compute the mean gradient (divide by batch size) before the optimizer step.

Background:
- Batch training improves throughput and reduces gradient noise.
- Accumulation avoids allocating per sample and matches how optimizers are usually written.

Note:
- This step is optional. A production-ready API can still be simple and correct
  with single-sample SGD (what we have now). If/when you want minibatches for
  throughput and stability, implement this step.

Status:
- Deferred. We keep the internal design compatible with adding batching later,
  but the public API focuses on simplicity (`fit`, `predict`, `evaluate_*`).
