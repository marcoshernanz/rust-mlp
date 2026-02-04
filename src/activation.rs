//! Activation functions.
//!
//! A dense layer computes a pre-activation value `z = W x + b` and then applies an
//! activation function element-wise: `y = activation(z)`.
//!
//! In this crate we cache the *post-activation* outputs `y` in `Scratch`. During
//! backprop we compute `dL/dz` from `dL/dy` using `y` (when possible). This keeps
//! the per-sample hot path allocation-free without needing a separate `z` buffer.

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Element-wise activation function.
pub enum Activation {
    Tanh,
    ReLU,
    LeakyReLU { alpha: f32 },
    Sigmoid,
    Identity,
}

impl Activation {
    /// Validate activation parameters.
    pub fn validate(self) -> Result<()> {
        match self {
            Activation::LeakyReLU { alpha } => {
                if !(alpha.is_finite() && alpha >= 0.0) {
                    return Err(Error::InvalidConfig(format!(
                        "leaky ReLU alpha must be finite and >= 0, got {alpha}"
                    )));
                }
            }
            Activation::Tanh | Activation::ReLU | Activation::Sigmoid | Activation::Identity => {}
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn forward(self, x: f32) -> f32 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU { alpha } => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Activation::Sigmoid => sigmoid(x),
            Activation::Identity => x,
        }
    }

    /// Derivative of the activation with respect to its input, expressed in terms
    /// of the cached post-activation output `y`.
    #[inline]
    pub(crate) fn grad_from_output(self, y: f32) -> f32 {
        match self {
            Activation::Tanh => 1.0 - y * y,
            Activation::ReLU => {
                if y > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyReLU { alpha } => {
                if y > 0.0 {
                    1.0
                } else {
                    alpha
                }
            }
            Activation::Sigmoid => y * (1.0 - y),
            Activation::Identity => 1.0,
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Numerically stable sigmoid.
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaky_relu_alpha_must_be_finite_and_non_negative() {
        assert!(
            Activation::LeakyReLU { alpha: f32::NAN }
                .validate()
                .is_err()
        );
        assert!(Activation::LeakyReLU { alpha: -0.1 }.validate().is_err());
        assert!(Activation::LeakyReLU { alpha: 0.1 }.validate().is_ok());
    }

    #[test]
    fn sigmoid_basic_values() {
        let y0 = Activation::Sigmoid.forward(0.0);
        assert!((y0 - 0.5).abs() < 1e-6);

        let y_pos = Activation::Sigmoid.forward(10.0);
        let y_neg = Activation::Sigmoid.forward(-10.0);
        assert!(y_pos > 0.999);
        assert!(y_neg < 0.001);
    }

    #[test]
    fn relu_and_leaky_relu_shapes() {
        assert_eq!(Activation::ReLU.forward(-2.0), 0.0);
        assert_eq!(Activation::ReLU.forward(3.0), 3.0);

        let act = Activation::LeakyReLU { alpha: 0.1 };
        assert_eq!(act.forward(-2.0), -0.2);
        assert_eq!(act.forward(3.0), 3.0);

        // Gradients expressed via cached outputs.
        assert_eq!(Activation::ReLU.grad_from_output(0.0), 0.0);
        assert_eq!(Activation::ReLU.grad_from_output(1.0), 1.0);
        assert_eq!(act.grad_from_output(-0.2), 0.1);
        assert_eq!(act.grad_from_output(3.0), 1.0);
    }

    #[test]
    fn tanh_and_sigmoid_gradients_from_output() {
        let y_tanh = Activation::Tanh.forward(0.3);
        let g_tanh = Activation::Tanh.grad_from_output(y_tanh);
        assert!((g_tanh - (1.0 - y_tanh * y_tanh)).abs() < 1e-6);

        let y_sig = Activation::Sigmoid.forward(0.0);
        let g_sig = Activation::Sigmoid.grad_from_output(y_sig);
        assert!((g_sig - 0.25).abs() < 1e-6);
    }
}
