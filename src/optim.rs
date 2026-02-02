use crate::{Error, Result};
use crate::{Gradients, Mlp};

#[derive(Debug, Clone, Copy)]
pub struct Sgd {
    lr: f32,
}

impl Sgd {
    #[inline]
    pub fn new(lr: f32) -> Self {
        assert!(
            lr.is_finite() && lr > 0.0,
            "learning rate must be finite and > 0"
        );
        Self { lr }
    }

    pub fn try_new(lr: f32) -> Result<Self> {
        if !(lr.is_finite() && lr > 0.0) {
            return Err(Error::InvalidConfig(
                "learning rate must be finite and > 0".to_owned(),
            ));
        }
        Ok(Self { lr })
    }

    #[inline]
    pub fn lr(&self) -> f32 {
        self.lr
    }

    #[inline]
    pub fn step(&self, model: &mut Mlp, grads: &Gradients) {
        model.sgd_step(grads, self.lr);
    }

    pub fn try_step(&self, model: &mut Mlp, grads: &Gradients) -> Result<()> {
        model.try_sgd_step(grads, self.lr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_requires_positive_finite_lr() {
        assert!(std::panic::catch_unwind(|| Sgd::new(0.0)).is_err());
        assert!(std::panic::catch_unwind(|| Sgd::new(-1.0)).is_err());
        assert!(std::panic::catch_unwind(|| Sgd::new(f32::NAN)).is_err());
    }
}
