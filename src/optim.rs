//! Optimizers.
//!
//! This module provides small, allocation-free-per-step optimizers that update an `Mlp`
//! given a set of `Gradients`.
//!
//! Design notes:
//! - Optimizer *state* (momentum/Adam moments) lives outside the model.
//! - The training loop owns the optimizer state and reuses it across steps.

use crate::{Error, Gradients, Mlp, Result};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
/// Optimizer choice for training.
pub enum Optimizer {
    /// Plain SGD.
    #[default]
    Sgd,
    /// SGD with momentum.
    SgdMomentum { momentum: f32 },
    /// Adam (bias-corrected).
    Adam { beta1: f32, beta2: f32, eps: f32 },
}

impl Optimizer {
    /// Validate optimizer hyperparameters.
    pub fn validate(self) -> Result<()> {
        match self {
            Optimizer::Sgd => Ok(()),
            Optimizer::SgdMomentum { momentum } => {
                if !(momentum.is_finite() && (0.0..1.0).contains(&momentum)) {
                    return Err(Error::InvalidConfig(format!(
                        "momentum must be finite and in [0,1), got {momentum}"
                    )));
                }
                Ok(())
            }
            Optimizer::Adam { beta1, beta2, eps } => {
                if !(beta1.is_finite() && (0.0..1.0).contains(&beta1)) {
                    return Err(Error::InvalidConfig(format!(
                        "adam beta1 must be finite and in [0,1), got {beta1}"
                    )));
                }
                if !(beta2.is_finite() && (0.0..1.0).contains(&beta2)) {
                    return Err(Error::InvalidConfig(format!(
                        "adam beta2 must be finite and in [0,1), got {beta2}"
                    )));
                }
                if !(eps.is_finite() && eps > 0.0) {
                    return Err(Error::InvalidConfig(format!(
                        "adam eps must be finite and > 0, got {eps}"
                    )));
                }
                Ok(())
            }
        }
    }

    /// Allocate optimizer state for `model`.
    pub fn state(self, model: &Mlp) -> Result<OptimizerState> {
        self.validate()?;

        match self {
            Optimizer::Sgd => Ok(OptimizerState::Sgd),
            Optimizer::SgdMomentum { momentum } => {
                let (vw, vb) = zeros_like_params(model);
                Ok(OptimizerState::SgdMomentum {
                    momentum,
                    v_weights: vw,
                    v_biases: vb,
                })
            }
            Optimizer::Adam { beta1, beta2, eps } => {
                let (mw, mb) = zeros_like_params(model);
                let (vw, vb) = zeros_like_params(model);
                Ok(OptimizerState::Adam {
                    beta1,
                    beta2,
                    eps,
                    t: 0,
                    beta1_pow: 1.0,
                    beta2_pow: 1.0,
                    m_weights: mw,
                    m_biases: mb,
                    v_weights: vw,
                    v_biases: vb,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
/// Owned optimizer state.
pub enum OptimizerState {
    /// Plain SGD (no state).
    #[default]
    Sgd,
    /// SGD with momentum state.
    SgdMomentum {
        momentum: f32,
        v_weights: Vec<Vec<f32>>,
        v_biases: Vec<Vec<f32>>,
    },
    /// Adam state.
    Adam {
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: u64,
        beta1_pow: f32,
        beta2_pow: f32,
        m_weights: Vec<Vec<f32>>,
        m_biases: Vec<Vec<f32>>,
        v_weights: Vec<Vec<f32>>,
        v_biases: Vec<Vec<f32>>,
    },
}

impl OptimizerState {
    /// Apply one optimizer step.
    ///
    /// `lr` is passed in from the training loop to support learning rate schedules.
    pub fn step(&mut self, model: &mut Mlp, grads: &mut Gradients, lr: f32) {
        assert!(lr.is_finite() && lr > 0.0, "lr must be finite and > 0");

        match self {
            OptimizerState::Sgd => {
                model.sgd_step(grads, lr);
            }
            OptimizerState::SgdMomentum {
                momentum,
                v_weights,
                v_biases,
            } => {
                debug_assert_eq!(v_weights.len(), model.num_layers());
                debug_assert_eq!(v_biases.len(), model.num_layers());

                for layer_idx in 0..model.num_layers() {
                    let dw = grads.d_weights(layer_idx);
                    let db = grads.d_biases(layer_idx);

                    let vw = &mut v_weights[layer_idx];
                    let vb = &mut v_biases[layer_idx];

                    debug_assert_eq!(vw.len(), dw.len());
                    debug_assert_eq!(vb.len(), db.len());

                    for (v, &g) in vw.iter_mut().zip(dw) {
                        *v = (*momentum) * *v + g;
                    }
                    for (v, &g) in vb.iter_mut().zip(db) {
                        *v = (*momentum) * *v + g;
                    }

                    let layer = model.layer_mut(layer_idx).expect("layer idx must be valid");
                    layer.sgd_step(vw, vb, lr);
                }
            }
            OptimizerState::Adam {
                beta1,
                beta2,
                eps,
                t,
                beta1_pow,
                beta2_pow,
                m_weights,
                m_biases,
                v_weights,
                v_biases,
            } => {
                *t += 1;
                *beta1_pow *= *beta1;
                *beta2_pow *= *beta2;

                let one_minus_beta1 = 1.0 - *beta1;
                let one_minus_beta2 = 1.0 - *beta2;
                let corr1 = 1.0 - *beta1_pow;
                let corr2 = 1.0 - *beta2_pow;

                // Overwrite `grads` with the Adam update direction and then reuse `sgd_step`.
                for layer_idx in 0..model.num_layers() {
                    let mw = &mut m_weights[layer_idx];
                    let mb = &mut m_biases[layer_idx];
                    let vw = &mut v_weights[layer_idx];
                    let vb = &mut v_biases[layer_idx];

                    debug_assert_eq!(mw.len(), vw.len());
                    debug_assert_eq!(mb.len(), vb.len());

                    {
                        let upd_w = grads.d_weights_mut(layer_idx);
                        for i in 0..upd_w.len() {
                            let g = upd_w[i];
                            mw[i] = (*beta1) * mw[i] + one_minus_beta1 * g;
                            vw[i] = (*beta2) * vw[i] + one_minus_beta2 * (g * g);

                            let m_hat = mw[i] / corr1;
                            let v_hat = vw[i] / corr2;
                            upd_w[i] = m_hat / (v_hat.sqrt() + *eps);
                        }
                    }
                    {
                        let upd_b = grads.d_biases_mut(layer_idx);
                        for i in 0..upd_b.len() {
                            let g = upd_b[i];
                            mb[i] = (*beta1) * mb[i] + one_minus_beta1 * g;
                            vb[i] = (*beta2) * vb[i] + one_minus_beta2 * (g * g);

                            let m_hat = mb[i] / corr1;
                            let v_hat = vb[i] / corr2;
                            upd_b[i] = m_hat / (v_hat.sqrt() + *eps);
                        }
                    }
                }

                model.sgd_step(grads, lr);
            }
        }
    }
}

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

fn zeros_like_params(model: &Mlp) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut ws = Vec::with_capacity(model.num_layers());
    let mut bs = Vec::with_capacity(model.num_layers());
    for i in 0..model.num_layers() {
        let layer = model.layer(i).expect("layer idx must be valid");
        ws.push(vec![0.0; layer.in_dim() * layer.out_dim()]);
        bs.push(vec![0.0; layer.out_dim()]);
    }
    (ws, bs)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{Activation, MlpBuilder};

    #[test]
    fn sgd_requires_positive_finite_lr() {
        assert!(Sgd::new(0.0).is_err());
        assert!(Sgd::new(-1.0).is_err());
        assert!(Sgd::new(f32::NAN).is_err());
    }

    #[test]
    fn optimizer_validation_rejects_bad_hyperparams() {
        assert!(Optimizer::SgdMomentum { momentum: 1.0 }.validate().is_err());
        assert!(
            Optimizer::SgdMomentum { momentum: -0.1 }
                .validate()
                .is_err()
        );
        assert!(
            Optimizer::Adam {
                beta1: 1.0,
                beta2: 0.999,
                eps: 1e-8
            }
            .validate()
            .is_err()
        );
        assert!(
            Optimizer::Adam {
                beta1: 0.9,
                beta2: 1.0,
                eps: 1e-8
            }
            .validate()
            .is_err()
        );
        assert!(
            Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 0.0
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn sgd_momentum_updates_like_sgd_on_first_step() {
        let mut mlp = MlpBuilder::new(1)
            .unwrap()
            .add_layer(1, Activation::Identity)
            .unwrap()
            .build_with_seed(0)
            .unwrap();

        // Force parameters to known values.
        {
            let layer = mlp.layer_mut(0).unwrap();
            layer.weights_mut()[0] = 1.0;
            layer.biases_mut()[0] = 2.0;
        }

        let mut grads = mlp.gradients();
        grads.d_weights_mut(0)[0] = 3.0;
        grads.d_biases_mut(0)[0] = 4.0;

        let mut opt = Optimizer::SgdMomentum { momentum: 0.9 }
            .state(&mlp)
            .unwrap();
        opt.step(&mut mlp, &mut grads, 0.1);

        let (w, b) = {
            let layer = mlp.layer_mut(0).unwrap();
            (layer.weights_mut()[0], layer.biases_mut()[0])
        };
        assert!((w - (1.0 - 0.1 * 3.0)).abs() < 1e-6);
        assert!((b - (2.0 - 0.1 * 4.0)).abs() < 1e-6);
    }

    #[test]
    fn adam_first_step_matches_expected_direction_for_unit_grad() {
        let mut mlp = MlpBuilder::new(1)
            .unwrap()
            .add_layer(1, Activation::Identity)
            .unwrap()
            .build_with_seed(0)
            .unwrap();

        {
            let layer = mlp.layer_mut(0).unwrap();
            layer.weights_mut()[0] = 1.0;
            layer.biases_mut()[0] = 1.0;
        }

        let mut grads = mlp.gradients();
        grads.d_weights_mut(0)[0] = 1.0;
        grads.d_biases_mut(0)[0] = 1.0;

        let mut opt = Optimizer::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0,
        }
        .state(&mlp)
        .unwrap();
        opt.step(&mut mlp, &mut grads, 0.1);

        // With eps=1.0 and unit grad, the first bias-corrected step has update ~= 1/(1+eps) = 0.5.
        let (w, b) = {
            let layer = mlp.layer_mut(0).unwrap();
            (layer.weights_mut()[0], layer.biases_mut()[0])
        };
        assert!((w - (1.0 - 0.1 * 0.5)).abs() < 1e-6);
        assert!((b - (1.0 - 0.1 * 0.5)).abs() < 1e-6);
    }
}
