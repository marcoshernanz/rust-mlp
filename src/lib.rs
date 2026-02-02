//! A small MLP library built for learning.
//!
//! Conventions used across the crate:
//! - Scalars are `f32`.
//! - Inputs/outputs are passed as slices (`&[f32]`, `&mut [f32]`) to avoid
//!   allocations in hot paths.
//! - Shape checks use `debug_assert_eq!` (checked in debug builds).

pub mod data;
pub mod error;
pub mod layer;
pub mod loss;
pub mod mlp;
pub mod neuron;
pub mod optim;
pub mod train;

pub use data::{Dataset, Inputs};
pub use error::{Error, Result};
pub use layer::{Init, Layer};
pub use mlp::Trainer;
pub use mlp::{Gradients, Mlp, Scratch};
pub use neuron::Neuron;
pub use optim::Sgd;
pub use train::{FitConfig, FitReport};
