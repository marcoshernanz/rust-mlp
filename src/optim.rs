//! Optimizers.
//!
//! Optimizers are small, stateless helpers that apply parameter updates to an `Mlp` given
//! a set of `Gradients`.

use crate::{Error, Result};
use crate::{Gradients, Mlp};

#[derive(Debug, Clone, Copy)]
/// Stochastic gradient descent with a fixed learning rate.
pub struct Sgd {
    lr: f32,
}

impl Sgd {
    #[inline]
    /// Construct an SGD optimizer.
    ///
    /// Returns an error if `lr` is not finite or `lr <= 0`.
    pub fn new(lr: f32) -> Result<Self> {
        if !(lr.is_finite() && lr > 0.0) {
            return Err(Error::InvalidConfig(
                "learning rate must be finite and > 0".to_owned(),
            ));
        }
        Ok(Self { lr })
    }

    #[inline]
    /// Returns the learning rate.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    #[inline]
    /// Apply one optimizer step: `param -= lr * d_param`.
    pub fn step(&self, model: &mut Mlp, grads: &Gradients) {
        model.sgd_step(grads, self.lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_requires_positive_finite_lr() {
        assert!(Sgd::new(0.0).is_err());
        assert!(Sgd::new(-1.0).is_err());
        assert!(Sgd::new(f32::NAN).is_err());
    }
}
