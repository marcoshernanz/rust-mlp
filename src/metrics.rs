//! Metrics.
//!
//! Metrics are evaluation helpers (they do not participate in backprop).
//!
//! In this crate, metrics are computed sample-by-sample during evaluation/training
//! without allocating per step.

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Supported evaluation metrics.
pub enum Metric {
    /// Mean squared error.
    Mse,
    /// Mean absolute error.
    Mae,
    /// Classification accuracy.
    ///
    /// - For `output_dim == 1`: binary accuracy.
    /// - For `output_dim > 1`: multiclass accuracy (argmax).
    Accuracy,
    /// Top-k accuracy for multiclass classification.
    ///
    /// This metric requires `output_dim > 1` and `k <= output_dim`.
    TopKAccuracy { k: usize },
}

impl Metric {
    /// Validate metric parameters.
    pub fn validate(self) -> Result<()> {
        match self {
            Metric::TopKAccuracy { k } => {
                if k == 0 {
                    return Err(Error::InvalidConfig(
                        "TopKAccuracy requires k > 0".to_owned(),
                    ));
                }
            }
            Metric::Mse | Metric::Mae | Metric::Accuracy => {}
        }
        Ok(())
    }
}
