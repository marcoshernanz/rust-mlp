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

    #[inline]
    pub fn lr(&self) -> f32 {
        self.lr
    }

    #[inline]
    pub fn step(&self, model: &mut Mlp, grads: &Gradients) {
        model.sgd_step(grads, self.lr);
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
