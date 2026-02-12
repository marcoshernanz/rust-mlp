//! A small MLP (multi-layer perceptron) crate.
//!
//! `rust-mlp` is a small-core, from-scratch implementation of a dense feed-forward network.
//! It is designed to be easy to read while keeping the per-sample hot path allocation-free.
//!
//! # Design goals
//!
//! - Predictable performance: reuse buffers (`Scratch` / `Gradients`) instead of allocating.
//! - Clear contracts: shapes are explicit and validated at the API boundary.
//! - Practical training loop: `fit` supports mini-batches, shuffling, LR schedules, and common
//!   optimizers.
//!
//! # Panics vs `Result`
//!
//! This crate intentionally exposes two layers of API:
//!
//! - Low-level hot path (panics on misuse):
//!   - [`mlp::Mlp::forward`], [`mlp::Mlp::backward`]
//!   - [`mlp::Mlp::forward_batch`], [`mlp::Mlp::backward_batch`]
//!     Shape mismatches are treated as programmer error and will panic via `assert!`.
//!
//! - High-level convenience APIs (shape-checked):
//!   - [`crate::Mlp::fit`], [`crate::Mlp::evaluate`]
//!   - [`mlp::Mlp::predict_into`]
//!     These validate inputs and return [`Result`].
//!
//! # Data layout and shapes
//!
//! - Scalars are `f32`.
//! - [`Dataset`] and [`Inputs`] store samples contiguously in row-major layout.
//! - Layer weights are row-major with shape `(out_dim, in_dim)`.
//! - Batched inputs/outputs are passed as flat row-major buffers:
//!   - inputs: `(batch_size, input_dim)` as `batch_size * input_dim` scalars
//!   - outputs: `(batch_size, output_dim)` as `batch_size * output_dim` scalars
//!
//! # MSRV
//!
//! This crate's minimum supported Rust version (MSRV) is specified in `Cargo.toml`.
//!
//! See `ROADMAP.md` for the production-readiness plan.

//! # Quick start
//!
//! ```rust
//! use rust_mlp::{Activation, FitConfig, Loss, Metric, MlpBuilder};
//!
//! # fn main() -> rust_mlp::Result<()> {
//! let xs = vec![
//!     vec![0.0, 0.0],
//!     vec![0.0, 1.0],
//!     vec![1.0, 0.0],
//!     vec![1.0, 1.0],
//! ];
//! let ys = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
//! let train = rust_mlp::Dataset::from_rows(&xs, &ys)?;
//!
//! let mut mlp = MlpBuilder::new(2)?
//!     .add_layer(8, Activation::ReLU)?
//!     .add_layer(1, Activation::Sigmoid)?
//!     .build_with_seed(0)?;
//!
//! let _report = mlp.fit(
//!     &train,
//!     None,
//!     FitConfig {
//!         epochs: 200,
//!         lr: 0.1,
//!         batch_size: 4,
//!         shuffle: rust_mlp::Shuffle::Seeded(0),
//!         lr_schedule: rust_mlp::LrSchedule::Constant,
//!         optimizer: rust_mlp::Optimizer::Adam {
//!             beta1: 0.9,
//!             beta2: 0.999,
//!             eps: 1e-8,
//!         },
//!         weight_decay: 0.0,
//!         grad_clip_norm: None,
//!         loss: Loss::Mse,
//!         metrics: vec![Metric::Accuracy],
//!     },
//! )?;
//! Ok(())
//! # }
//! ```

//! # Allocation-free training (advanced)
//!
//! If you want to drive training yourself (e.g. custom loop), allocate buffers once and reuse
//! them across steps:
//!
//! ```rust
//! use rust_mlp::{Activation, Loss, MlpBuilder};
//!
//! # fn main() -> rust_mlp::Result<()> {
//! let mut mlp = MlpBuilder::new(3)?
//!     .add_layer(8, Activation::Tanh)?
//!     .add_layer(2, Activation::Identity)?
//!     .build_with_seed(0)?;
//!
//! let mut trainer = mlp.trainer();
//! let x = [0.1_f32, -0.2, 0.3];
//! let t = [0.0_f32, 1.0];
//!
//! let y = mlp.forward(&x, &mut trainer.scratch);
//! let _loss = Loss::Mse.backward(y, &t, trainer.grads.d_output_mut());
//! mlp.backward(&x, &trainer.scratch, &mut trainer.grads);
//! mlp.sgd_step(&trainer.grads, 1e-2);
//! Ok(())
//! # }
//! ```

pub mod activation;
pub mod builder;
pub mod data;
pub mod error;
pub mod layer;
pub mod loss;
pub(crate) mod matmul;
pub mod metrics;
pub mod mlp;
pub mod optim;
pub mod train;

#[cfg(feature = "serde")]
pub mod serde_model;

pub use activation::Activation;
pub use builder::MlpBuilder;
pub use data::{Dataset, Inputs};
pub use error::{Error, Result};
pub use layer::{Init, Layer};
pub use loss::Loss;
pub use metrics::Metric;
pub use mlp::Trainer;
pub use mlp::{BatchBackpropScratch, BatchScratch, Gradients, Mlp, Scratch};
pub use optim::{Optimizer, OptimizerState, Sgd};
pub use train::Shuffle;
pub use train::{EpochReport, EvalReport, FitConfig, FitReport, LrSchedule};

/// Shape-safe, non-allocating inference.
///
/// Thin wrapper around [`Mlp::predict_into`].
pub fn predict_into(
    mlp: &Mlp,
    input: &[f32],
    scratch: &mut Scratch,
    out: &mut [f32],
) -> Result<()> {
    mlp.predict_into(input, scratch, out)
}
