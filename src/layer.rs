use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Init {
    Zeros,
    /// Xavier/Glorot uniform init, a good default for `tanh`.
    XavierTanh,
}

#[derive(Debug, Clone)]
pub struct Layer {
    in_dim: usize,
    out_dim: usize,
    /// Row-major matrix with shape (out_dim, in_dim).
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl Layer {
    #[inline]
    pub fn new_with_seed(in_dim: usize, out_dim: usize, init: Init, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new_with_rng(in_dim, out_dim, init, &mut rng)
    }

    pub fn new_with_rng<R: Rng + ?Sized>(
        in_dim: usize,
        out_dim: usize,
        init: Init,
        rng: &mut R,
    ) -> Result<Self> {
        if in_dim == 0 || out_dim == 0 {
            return Err(Error::InvalidConfig("layer dims must be > 0".to_owned()));
        }

        let mut weights = vec![0.0; in_dim * out_dim];
        match init {
            Init::Zeros => {}
            Init::XavierTanh => {
                let fan_in = in_dim as f32;
                let fan_out = out_dim as f32;
                let limit = (6.0 / (fan_in + fan_out)).sqrt();
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
            weights,
            biases,
        })
    }

    #[inline]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    #[inline]
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
    /// - `outputs = tanh(z)`
    ///
    /// Shape contract:
    /// - `inputs.len() == self.in_dim`
    /// - `outputs.len() == self.out_dim`
    #[inline]
    pub fn forward(&self, inputs: &[f32], outputs: &mut [f32]) {
        debug_assert_eq!(inputs.len(), self.in_dim);
        debug_assert_eq!(outputs.len(), self.out_dim);

        for (o, out) in outputs.iter_mut().enumerate() {
            let mut sum = self.biases[o];
            let row = o * self.in_dim;
            for (i, &x) in inputs.iter().enumerate() {
                sum = self.weights[row + i].mul_add(x, sum);
            }
            *out = sum.tanh();
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
        debug_assert_eq!(inputs.len(), self.in_dim);
        debug_assert_eq!(outputs.len(), self.out_dim);
        debug_assert_eq!(d_outputs.len(), self.out_dim);
        debug_assert_eq!(d_inputs.len(), self.in_dim);
        debug_assert_eq!(d_weights.len(), self.weights.len());
        debug_assert_eq!(d_biases.len(), self.out_dim);

        // d_inputs accumulates contributions from all outputs.
        d_inputs.fill(0.0);

        for o in 0..self.out_dim {
            // tanh'(z) = 1 - tanh(z)^2 = 1 - outputs[o]^2
            let d_z = d_outputs[o] * (1.0 - outputs[o] * outputs[o]);
            d_biases[o] = d_z;

            let row = o * self.in_dim;
            for i in 0..self.in_dim {
                let w = self.weights[row + i];
                d_weights[row + i] = d_z * inputs[i];
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
        debug_assert_eq!(d_weights.len(), self.weights.len());
        debug_assert_eq!(d_biases.len(), self.biases.len());

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
        let a = Layer::new_with_seed(3, 2, Init::XavierTanh, 123).unwrap();
        let b = Layer::new_with_seed(3, 2, Init::XavierTanh, 123).unwrap();
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.biases, b.biases);
    }

    #[test]
    fn backward_matches_numeric_gradients() {
        let in_dim = 3;
        let out_dim = 2;
        let mut layer = Layer::new_with_seed(in_dim, out_dim, Init::XavierTanh, 0).unwrap();

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

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn forward_panics_on_input_shape_mismatch() {
        let layer = Layer::new_with_seed(3, 2, Init::XavierTanh, 0).unwrap();
        let input = vec![0.0_f32; 2];
        let mut out = vec![0.0_f32; 2];
        layer.forward(&input, &mut out);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn forward_panics_on_output_shape_mismatch() {
        let layer = Layer::new_with_seed(3, 2, Init::XavierTanh, 0).unwrap();
        let input = vec![0.0_f32; 3];
        let mut out = vec![0.0_f32; 1];
        layer.forward(&input, &mut out);
    }
}
