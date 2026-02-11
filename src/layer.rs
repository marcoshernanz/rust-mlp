//! Dense layer implementation.
//!
//! A `Layer` is a dense affine transform followed by an element-wise activation:
//!
//! - `z = W x + b`
//! - `y = activation(z)`
//!
//! The activation is stored in the layer so an `Mlp` can mix activation functions
//! across layers.
//!
//! Shape mismatches are treated as programmer error and will panic via `assert!`.

use rand::Rng;
use rand::distributions::{Distribution, Uniform};

use crate::Activation;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Initialization scheme for layer weights.
pub enum Init {
    Zeros,
    /// Xavier/Glorot uniform init.
    ///
    /// This is a good default for `tanh`, `sigmoid`, and `identity` activations.
    Xavier,
    /// He/Kaiming uniform init.
    ///
    /// This is a good default for `ReLU`-family activations.
    He,
}

#[derive(Debug, Clone)]
/// A dense layer: `y = activation(Wx + b)`.
///
/// Weights use row-major layout with shape `(out_dim, in_dim)`.
pub struct Layer {
    in_dim: usize,
    out_dim: usize,
    activation: Activation,
    /// Row-major matrix with shape (out_dim, in_dim).
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl Layer {
    #[inline]
    /// Returns this layer's activation.
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

impl Layer {
    pub fn new_with_rng<R: Rng + ?Sized>(
        in_dim: usize,
        out_dim: usize,
        init: Init,
        activation: Activation,
        rng: &mut R,
    ) -> Result<Self> {
        if in_dim == 0 || out_dim == 0 {
            return Err(Error::InvalidConfig("layer dims must be > 0".to_owned()));
        }

        activation.validate()?;

        let mut weights = vec![0.0; in_dim * out_dim];
        match init {
            Init::Zeros => {}
            Init::Xavier => {
                let fan_in = in_dim as f32;
                let fan_out = out_dim as f32;
                let limit = (6.0 / (fan_in + fan_out)).sqrt();
                let dist = Uniform::new(-limit, limit);
                for w in &mut weights {
                    *w = dist.sample(rng);
                }
            }
            Init::He => {
                let fan_in = in_dim as f32;
                let limit = (6.0 / fan_in).sqrt();
                let dist = Uniform::new(-limit, limit);
                for w in &mut weights {
                    *w = dist.sample(rng);
                }
            }
        }

        let biases = vec![0.0; out_dim];

        Ok(Self {
            in_dim,
            out_dim,
            activation,
            weights,
            biases,
        })
    }

    #[inline]
    /// Returns the input dimension.
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    #[inline]
    /// Returns the output dimension.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn biases_mut(&mut self) -> &mut [f32] {
        &mut self.biases
    }

    /// Forward pass for a single sample.
    ///
    /// Computes:
    /// - `z = W * inputs + b`
    /// - `outputs = activation(z)`
    ///
    /// Shape contract:
    /// - `inputs.len() == self.in_dim`
    /// - `outputs.len() == self.out_dim`
    #[inline]
    pub fn forward(&self, inputs: &[f32], outputs: &mut [f32]) {
        assert_eq!(
            inputs.len(),
            self.in_dim,
            "inputs len {} does not match layer in_dim {}",
            inputs.len(),
            self.in_dim
        );
        assert_eq!(
            outputs.len(),
            self.out_dim,
            "outputs len {} does not match layer out_dim {}",
            outputs.len(),
            self.out_dim
        );

        let activation = self.activation;

        for (o, out) in outputs.iter_mut().enumerate() {
            let mut sum = self.biases[o];
            let row = o * self.in_dim;
            for (i, &x) in inputs.iter().enumerate() {
                sum = self.weights[row + i].mul_add(x, sum);
            }
            *out = activation.forward(sum);
        }
    }

    /// Backward pass for a single sample.
    ///
    /// This uses overwrite semantics:
    /// - `d_inputs` is overwritten (and internally zeroed before accumulation)
    /// - `d_weights` is overwritten
    /// - `d_biases` is overwritten
    ///
    /// Inputs:
    /// - `inputs`: the same inputs passed to `forward`
    /// - `outputs`: the outputs previously produced by `forward` (post-activation)
    /// - `d_outputs`: upstream gradient dL/d(outputs)
    ///
    /// Shape contract:
    /// - `inputs.len() == self.in_dim`
    /// - `outputs.len() == self.out_dim`
    /// - `d_outputs.len() == self.out_dim`
    /// - `d_inputs.len() == self.in_dim`
    /// - `d_weights.len() == self.weights.len()`
    /// - `d_biases.len() == self.out_dim`
    #[inline]
    pub fn backward(
        &self,
        inputs: &[f32],
        outputs: &[f32],
        d_outputs: &[f32],
        d_inputs: &mut [f32],
        d_weights: &mut [f32],
        d_biases: &mut [f32],
    ) {
        assert_eq!(
            inputs.len(),
            self.in_dim,
            "inputs len {} does not match layer in_dim {}",
            inputs.len(),
            self.in_dim
        );
        assert_eq!(
            outputs.len(),
            self.out_dim,
            "outputs len {} does not match layer out_dim {}",
            outputs.len(),
            self.out_dim
        );
        assert_eq!(
            d_outputs.len(),
            self.out_dim,
            "d_outputs len {} does not match layer out_dim {}",
            d_outputs.len(),
            self.out_dim
        );
        assert_eq!(
            d_inputs.len(),
            self.in_dim,
            "d_inputs len {} does not match layer in_dim {}",
            d_inputs.len(),
            self.in_dim
        );
        assert_eq!(
            d_weights.len(),
            self.weights.len(),
            "d_weights len {} does not match weights len {}",
            d_weights.len(),
            self.weights.len()
        );
        assert_eq!(
            d_biases.len(),
            self.out_dim,
            "d_biases len {} does not match layer out_dim {}",
            d_biases.len(),
            self.out_dim
        );

        // d_inputs accumulates contributions from all outputs.
        d_inputs.fill(0.0);

        let activation = self.activation;

        for o in 0..self.out_dim {
            let d_z = d_outputs[o] * activation.grad_from_output(outputs[o]);
            d_biases[o] = d_z;

            let row = o * self.in_dim;
            for i in 0..self.in_dim {
                let w = self.weights[row + i];
                d_weights[row + i] = d_z * inputs[i];
                d_inputs[i] = w.mul_add(d_z, d_inputs[i]);
            }
        }
    }

    /// Backward pass for a single sample (parameter accumulation semantics).
    ///
    /// This is identical to `backward` except that parameter gradients are *accumulated*:
    /// - `d_inputs` is overwritten (and internally zeroed before accumulation)
    /// - `d_weights` is accumulated into (`+=`)
    /// - `d_biases` is accumulated into (`+=`)
    ///
    /// This is useful for mini-batch training where you sum gradients over multiple samples.
    ///
    /// Shape contract:
    /// - `inputs.len() == self.in_dim`
    /// - `outputs.len() == self.out_dim`
    /// - `d_outputs.len() == self.out_dim`
    /// - `d_inputs.len() == self.in_dim`
    /// - `d_weights.len() == self.weights.len()`
    /// - `d_biases.len() == self.out_dim`
    #[inline]
    pub fn backward_accumulate(
        &self,
        inputs: &[f32],
        outputs: &[f32],
        d_outputs: &[f32],
        d_inputs: &mut [f32],
        d_weights: &mut [f32],
        d_biases: &mut [f32],
    ) {
        assert_eq!(
            inputs.len(),
            self.in_dim,
            "inputs len {} does not match layer in_dim {}",
            inputs.len(),
            self.in_dim
        );
        assert_eq!(
            outputs.len(),
            self.out_dim,
            "outputs len {} does not match layer out_dim {}",
            outputs.len(),
            self.out_dim
        );
        assert_eq!(
            d_outputs.len(),
            self.out_dim,
            "d_outputs len {} does not match layer out_dim {}",
            d_outputs.len(),
            self.out_dim
        );
        assert_eq!(
            d_inputs.len(),
            self.in_dim,
            "d_inputs len {} does not match layer in_dim {}",
            d_inputs.len(),
            self.in_dim
        );
        assert_eq!(
            d_weights.len(),
            self.weights.len(),
            "d_weights len {} does not match weights len {}",
            d_weights.len(),
            self.weights.len()
        );
        assert_eq!(
            d_biases.len(),
            self.out_dim,
            "d_biases len {} does not match layer out_dim {}",
            d_biases.len(),
            self.out_dim
        );

        // d_inputs accumulates contributions from all outputs.
        d_inputs.fill(0.0);

        let activation = self.activation;

        for o in 0..self.out_dim {
            let d_z = d_outputs[o] * activation.grad_from_output(outputs[o]);
            d_biases[o] += d_z;

            let row = o * self.in_dim;
            for i in 0..self.in_dim {
                let w = self.weights[row + i];
                d_weights[row + i] += d_z * inputs[i];
                d_inputs[i] = w.mul_add(d_z, d_inputs[i]);
            }
        }
    }

    /// Applies an SGD update: `param -= lr * d_param`.
    ///
    /// Shape contract:
    /// - `d_weights.len() == self.weights.len()`
    /// - `d_biases.len() == self.biases.len()`
    #[inline]
    pub fn sgd_step(&mut self, d_weights: &[f32], d_biases: &[f32], lr: f32) {
        assert_eq!(
            d_weights.len(),
            self.weights.len(),
            "d_weights len {} does not match weights len {}",
            d_weights.len(),
            self.weights.len()
        );
        assert_eq!(
            d_biases.len(),
            self.biases.len(),
            "d_biases len {} does not match biases len {}",
            d_biases.len(),
            self.biases.len()
        );

        for (w, &dw) in self.weights.iter_mut().zip(d_weights) {
            *w -= lr * dw;
        }
        for (b, &db) in self.biases.iter_mut().zip(d_biases) {
            *b -= lr * db;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn loss_for_layer(layer: &Layer, input: &[f32], target: &[f32], out: &mut [f32]) -> f32 {
        layer.forward(input, out);
        crate::loss::mse(out, target)
    }

    fn assert_close(analytic: f32, numeric: f32, abs_tol: f32, rel_tol: f32) {
        let diff = (analytic - numeric).abs();
        let scale = analytic.abs().max(numeric.abs()).max(1.0);
        assert!(
            diff <= abs_tol || diff / scale <= rel_tol,
            "analytic={analytic} numeric={numeric} diff={diff}"
        );
    }

    #[test]
    fn seeded_init_is_deterministic() {
        let mut rng_a = StdRng::seed_from_u64(123);
        let mut rng_b = StdRng::seed_from_u64(123);
        let a = Layer::new_with_rng(3, 2, Init::Xavier, Activation::Tanh, &mut rng_a).unwrap();
        let b = Layer::new_with_rng(3, 2, Init::Xavier, Activation::Tanh, &mut rng_b).unwrap();
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.biases, b.biases);
    }

    #[test]
    fn backward_matches_numeric_gradients() {
        let in_dim = 3;
        let out_dim = 2;
        let mut rng = StdRng::seed_from_u64(0);
        let mut layer =
            Layer::new_with_rng(in_dim, out_dim, Init::Xavier, Activation::Tanh, &mut rng).unwrap();

        let mut input = vec![0.3_f32, -0.7_f32, 0.1_f32];
        let target = vec![0.2_f32, -0.1_f32];

        let mut outputs = vec![0.0_f32; out_dim];
        layer.forward(&input, &mut outputs);

        let mut d_outputs = vec![0.0_f32; out_dim];
        let _loss = crate::loss::mse_backward(&outputs, &target, &mut d_outputs);

        let mut d_inputs = vec![0.0_f32; in_dim];
        let mut d_weights = vec![0.0_f32; in_dim * out_dim];
        let mut d_biases = vec![0.0_f32; out_dim];

        layer.backward(
            &input,
            &outputs,
            &d_outputs,
            &mut d_inputs,
            &mut d_weights,
            &mut d_biases,
        );

        let eps = 1e-3_f32;
        let abs_tol = 1e-3_f32;
        let rel_tol = 1e-2_f32;

        // Weights.
        let mut out_tmp = vec![0.0_f32; out_dim];
        for (p, &analytic) in d_weights.iter().enumerate() {
            let orig = layer.weights[p];

            layer.weights[p] = orig + eps;
            let loss_plus = loss_for_layer(&layer, &input, &target, &mut out_tmp);

            layer.weights[p] = orig - eps;
            let loss_minus = loss_for_layer(&layer, &input, &target, &mut out_tmp);

            layer.weights[p] = orig;

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert_close(analytic, numeric, abs_tol, rel_tol);
        }

        // Biases.
        for (p, &analytic) in d_biases.iter().enumerate() {
            let orig = layer.biases[p];

            layer.biases[p] = orig + eps;
            let loss_plus = loss_for_layer(&layer, &input, &target, &mut out_tmp);

            layer.biases[p] = orig - eps;
            let loss_minus = loss_for_layer(&layer, &input, &target, &mut out_tmp);

            layer.biases[p] = orig;

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert_close(analytic, numeric, abs_tol, rel_tol);
        }

        // Inputs.
        for i in 0..input.len() {
            let orig = input[i];

            input[i] = orig + eps;
            let loss_plus = loss_for_layer(&layer, &input, &target, &mut out_tmp);

            input[i] = orig - eps;
            let loss_minus = loss_for_layer(&layer, &input, &target, &mut out_tmp);

            input[i] = orig;

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            let analytic = d_inputs[i];
            assert_close(analytic, numeric, abs_tol, rel_tol);
        }
    }

    #[test]
    #[should_panic]
    fn forward_panics_on_input_shape_mismatch() {
        let mut rng = StdRng::seed_from_u64(0);
        let layer = Layer::new_with_rng(3, 2, Init::Xavier, Activation::Tanh, &mut rng).unwrap();
        let input = vec![0.0_f32; 2];
        let mut out = vec![0.0_f32; 2];
        layer.forward(&input, &mut out);
    }

    #[test]
    #[should_panic]
    fn forward_panics_on_output_shape_mismatch() {
        let mut rng = StdRng::seed_from_u64(0);
        let layer = Layer::new_with_rng(3, 2, Init::Xavier, Activation::Tanh, &mut rng).unwrap();
        let input = vec![0.0_f32; 3];
        let mut out = vec![0.0_f32; 1];
        layer.forward(&input, &mut out);
    }
}
