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

pub mod data;
pub mod error;
pub mod layer;
pub mod loss;
pub mod mlp;
pub mod optim;
pub mod train;

pub use data::{Dataset, Inputs};
pub use error::{Error, Result};
pub use layer::{Init, Layer};
pub use mlp::Trainer;
pub use mlp::{Gradients, Mlp, Scratch};
pub use optim::Sgd;
pub use train::{FitConfig, FitReport};
