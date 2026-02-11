//! A small MLP (multi-layer perceptron) crate.
//!
//! This crate is built to be easy to understand while keeping the hot path allocation-free.
//!
//! Conventions:
//! - Scalars are `f32`.
//! - Inputs/outputs are passed as slices (`&[f32]`, `&mut [f32]`) to avoid allocations.
//! - Low-level APIs (`forward`, `backward`) panic on shape mismatches.
//! - High-level APIs (`fit`, `predict`, `evaluate_*`) validate shapes and return `Result`.
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

pub mod activation;
pub mod builder;
pub mod data;
pub mod error;
pub mod layer;
pub mod loss;
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
pub use mlp::{BatchScratch, Gradients, Mlp, Scratch};
pub use optim::{Optimizer, OptimizerState, Sgd};
pub use train::Shuffle;
pub use train::{EpochReport, EvalReport, FitConfig, FitReport, LrSchedule};
