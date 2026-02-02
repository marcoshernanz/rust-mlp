use crate::{Error, Result};
use crate::{Init, Layer};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct Mlp {
    layers: Vec<Layer>,
}

/// Reusable buffers for `Mlp::forward`.
///
/// The output of the most recent forward pass lives inside `Scratch`.
#[derive(Debug, Clone)]
pub struct Scratch {
    layer_outputs: Vec<Vec<f32>>,
}

/// Parameter gradients for an `Mlp` (overwrite semantics).
///
/// Allocate once via `Mlp::gradients()` and reuse across training steps.
#[derive(Debug, Clone)]
pub struct Gradients {
    d_weights: Vec<Vec<f32>>,
    d_biases: Vec<Vec<f32>>,

    // Backprop intermediate: gradient w.r.t each layer output.
    // This includes the final layer output; `Mlp::backward` copies the provided
    // `d_output` into this buffer so it can uniformly backprop layer-by-layer.
    d_layer_outputs: Vec<Vec<f32>>,

    d_input: Vec<f32>,
}

impl Mlp {
    pub fn new_with_seed(sizes: &[usize], seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new_with_rng(sizes, &mut rng)
    }

    pub fn new_with_rng<R: Rng + ?Sized>(sizes: &[usize], rng: &mut R) -> Result<Self> {
        if sizes.len() < 2 {
            return Err(Error::InvalidConfig(
                "sizes must include input and output dims".to_owned(),
            ));
        }
        if sizes.contains(&0) {
            return Err(Error::InvalidConfig(
                "all layer sizes must be > 0".to_owned(),
            ));
        }

        let mut layers = Vec::with_capacity(sizes.len() - 1);
        for w in sizes.windows(2) {
            let in_dim = w[0];
            let out_dim = w[1];
            layers.push(Layer::new_with_rng(in_dim, out_dim, Init::XavierTanh, rng)?);
        }
        Ok(Self { layers })
    }

    #[inline]
    pub fn input_dim(&self) -> usize {
        self.layers
            .first()
            .expect("mlp must have at least one layer")
            .in_dim()
    }

    #[inline]
    pub fn output_dim(&self) -> usize {
        self.layers
            .last()
            .expect("mlp must have at least one layer")
            .out_dim()
    }

    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn scratch(&self) -> Scratch {
        Scratch::new(self)
    }

    pub fn gradients(&self) -> Gradients {
        Gradients::new(self)
    }

    /// Convenience constructor: allocate all training buffers.
    #[inline]
    pub fn trainer(&self) -> Trainer {
        Trainer::new(self)
    }

    /// Forward pass for a single sample.
    ///
    /// Writes intermediate activations into `scratch` and returns the final output slice.
    ///
    /// Shape contract:
    /// - `input.len() == self.input_dim()`
    /// - `scratch` must be built for this `Mlp` (same layer count and output sizes)
    pub fn forward<'a>(&self, input: &[f32], scratch: &'a mut Scratch) -> &'a [f32] {
        assert_eq!(
            input.len(),
            self.input_dim(),
            "input len {} does not match model input_dim {}",
            input.len(),
            self.input_dim()
        );
        assert_eq!(
            scratch.layer_outputs.len(),
            self.layers.len(),
            "scratch has {} layer outputs, model has {} layers",
            scratch.layer_outputs.len(),
            self.layers.len()
        );

        for (idx, layer) in self.layers.iter().enumerate() {
            if idx == 0 {
                let out = &mut scratch.layer_outputs[0];
                assert_eq!(
                    out.len(),
                    layer.out_dim(),
                    "scratch layer 0 output len {} does not match layer out_dim {}",
                    out.len(),
                    layer.out_dim()
                );
                layer.forward(input, out);
            } else {
                // Borrow the previous output immutably and the current output mutably.
                let (left, right) = scratch.layer_outputs.split_at_mut(idx);
                let prev = &left[idx - 1];
                let out = &mut right[0];
                assert_eq!(
                    out.len(),
                    layer.out_dim(),
                    "scratch layer {idx} output len {} does not match layer out_dim {}",
                    out.len(),
                    layer.out_dim()
                );
                layer.forward(prev, out);
            }
        }

        scratch.output()
    }

    /// Backward pass for a single sample, using the internal `d_output` buffer.
    ///
    /// You must call `forward` first using the same `input` and `scratch`.
    ///
    /// Before calling this, write the upstream gradient `dL/d(output)` into
    /// `grads.d_output_mut()`.
    ///
    /// Overwrite semantics:
    /// - `grads` is overwritten with gradients for this sample.
    ///
    /// Returns dL/d(input).
    pub fn backward<'a>(
        &self,
        input: &[f32],
        scratch: &Scratch,
        grads: &'a mut Gradients,
    ) -> &'a [f32] {
        assert_eq!(
            input.len(),
            self.input_dim(),
            "input len {} does not match model input_dim {}",
            input.len(),
            self.input_dim()
        );
        assert_eq!(
            scratch.layer_outputs.len(),
            self.layers.len(),
            "scratch has {} layer outputs, model has {} layers",
            scratch.layer_outputs.len(),
            self.layers.len()
        );

        assert_eq!(
            grads.d_weights.len(),
            self.layers.len(),
            "grads has {} d_weights entries, model has {} layers",
            grads.d_weights.len(),
            self.layers.len()
        );
        assert_eq!(
            grads.d_biases.len(),
            self.layers.len(),
            "grads has {} d_biases entries, model has {} layers",
            grads.d_biases.len(),
            self.layers.len()
        );
        assert_eq!(
            grads.d_layer_outputs.len(),
            self.layers.len(),
            "grads has {} d_layer_outputs entries, model has {} layers",
            grads.d_layer_outputs.len(),
            self.layers.len()
        );
        assert_eq!(
            grads.d_input.len(),
            self.input_dim(),
            "grads d_input len {} does not match model input_dim {}",
            grads.d_input.len(),
            self.input_dim()
        );

        let last = self.layers.len() - 1;
        assert_eq!(
            grads.d_layer_outputs[last].len(),
            self.output_dim(),
            "grads d_output len {} does not match model output_dim {}",
            grads.d_layer_outputs[last].len(),
            self.output_dim()
        );

        for idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[idx];

            let layer_input: &[f32] = if idx == 0 {
                input
            } else {
                &scratch.layer_outputs[idx - 1]
            };

            let layer_output: &[f32] = &scratch.layer_outputs[idx];
            assert_eq!(
                layer_output.len(),
                layer.out_dim(),
                "scratch layer {idx} output len {} does not match layer out_dim {}",
                layer_output.len(),
                layer.out_dim()
            );

            if idx == 0 {
                let d_outputs = &grads.d_layer_outputs[0];
                layer.backward(
                    layer_input,
                    layer_output,
                    d_outputs,
                    &mut grads.d_input,
                    &mut grads.d_weights[0],
                    &mut grads.d_biases[0],
                );
            } else {
                // We need two different layer gradient buffers:
                // - `d_outputs` for the current layer (read-only)
                // - `d_inputs` for the current layer, which becomes `d_outputs` of the previous
                let (left, right) = grads.d_layer_outputs.split_at_mut(idx);
                let d_inputs_prev = &mut left[idx - 1];
                let d_outputs = &right[0];
                layer.backward(
                    layer_input,
                    layer_output,
                    d_outputs,
                    d_inputs_prev,
                    &mut grads.d_weights[idx],
                    &mut grads.d_biases[idx],
                );
            }
        }

        &grads.d_input
    }

    /// Applies an SGD update to all layers.
    #[inline]
    pub fn sgd_step(&mut self, grads: &Gradients, lr: f32) {
        assert!(
            lr.is_finite() && lr > 0.0,
            "learning rate must be finite and > 0"
        );
        assert_eq!(
            self.layers.len(),
            grads.d_weights.len(),
            "grads has {} d_weights entries, model has {} layers",
            grads.d_weights.len(),
            self.layers.len()
        );
        assert_eq!(
            self.layers.len(),
            grads.d_biases.len(),
            "grads has {} d_biases entries, model has {} layers",
            grads.d_biases.len(),
            self.layers.len()
        );

        for i in 0..self.layers.len() {
            self.layers[i].sgd_step(&grads.d_weights[i], &grads.d_biases[i], lr);
        }
    }
}

/// Reusable buffers for training a specific `Mlp`.
///
/// This is the ergonomic wrapper around `Scratch` + `Gradients`.
#[derive(Debug, Clone)]
pub struct Trainer {
    pub scratch: Scratch,
    pub grads: Gradients,
}

impl Trainer {
    pub fn new(mlp: &Mlp) -> Self {
        Self {
            scratch: Scratch::new(mlp),
            grads: Gradients::new(mlp),
        }
    }
}

impl Scratch {
    pub fn new(mlp: &Mlp) -> Self {
        let mut layer_outputs = Vec::with_capacity(mlp.layers.len());
        for layer in &mlp.layers {
            layer_outputs.push(vec![0.0; layer.out_dim()]);
        }
        Self { layer_outputs }
    }

    #[inline]
    pub fn output(&self) -> &[f32] {
        self.layer_outputs
            .last()
            .expect("scratch must have at least one layer output")
            .as_slice()
    }
}

impl Gradients {
    pub fn new(mlp: &Mlp) -> Self {
        let mut d_weights = Vec::with_capacity(mlp.layers.len());
        let mut d_biases = Vec::with_capacity(mlp.layers.len());
        let mut d_layer_outputs = Vec::with_capacity(mlp.layers.len());

        for layer in &mlp.layers {
            d_weights.push(vec![0.0; layer.in_dim() * layer.out_dim()]);
            d_biases.push(vec![0.0; layer.out_dim()]);
            d_layer_outputs.push(vec![0.0; layer.out_dim()]);
        }

        let d_input = vec![0.0; mlp.input_dim()];

        Self {
            d_weights,
            d_biases,
            d_layer_outputs,
            d_input,
        }
    }

    /// Mutable view of the upstream gradient buffer for the final model output.
    ///
    /// Typical training flow:
    /// - `mlp.forward(input, &mut scratch)`
    /// - loss writes `dL/d(output)` into `grads.d_output_mut()`
    /// - `mlp.backward(input, &scratch, &mut grads)`
    #[inline]
    pub fn d_output_mut(&mut self) -> &mut [f32] {
        self.d_layer_outputs
            .last_mut()
            .expect("mlp must have at least one layer")
            .as_mut_slice()
    }

    #[inline]
    pub fn d_output(&self) -> &[f32] {
        self.d_layer_outputs
            .last()
            .expect("mlp must have at least one layer")
            .as_slice()
    }

    #[inline]
    pub fn d_input(&self) -> &[f32] {
        &self.d_input
    }

    #[inline]
    pub fn d_weights(&self, layer_idx: usize) -> &[f32] {
        &self.d_weights[layer_idx]
    }

    #[inline]
    pub fn d_biases(&self, layer_idx: usize) -> &[f32] {
        &self.d_biases[layer_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loss_for_mlp(mlp: &Mlp, input: &[f32], target: &[f32], scratch: &mut Scratch) -> f32 {
        mlp.forward(input, scratch);
        crate::loss::mse(scratch.output(), target)
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
        let a = Mlp::new_with_seed(&[2, 3, 1], 123).unwrap();
        let b = Mlp::new_with_seed(&[2, 3, 1], 123).unwrap();

        let mut scratch_a = a.scratch();
        let mut scratch_b = b.scratch();
        let input = [0.3_f32, -0.7_f32];

        let out_a = a.forward(&input, &mut scratch_a).to_vec();
        let out_b = b.forward(&input, &mut scratch_b).to_vec();
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn backward_matches_numeric_gradients() {
        let mut mlp = Mlp::new_with_seed(&[2, 3, 1], 0).unwrap();
        let mut scratch = mlp.scratch();
        let mut grads = mlp.gradients();

        let input = [0.3_f32, -0.7_f32];
        let target = [0.2_f32];

        mlp.forward(&input, &mut scratch);
        let _loss = crate::loss::mse_backward(scratch.output(), &target, grads.d_output_mut());
        let d_input = mlp.backward(&input, &scratch, &mut grads).to_vec();

        let eps = 1e-3_f32;
        let abs_tol = 1e-3_f32;
        let rel_tol = 1e-2_f32;

        let mut scratch_tmp = mlp.scratch();

        // Parameters.
        for layer_idx in 0..mlp.layers.len() {
            // Weights.
            let w_len = mlp.layers[layer_idx].in_dim() * mlp.layers[layer_idx].out_dim();
            debug_assert_eq!(w_len, grads.d_weights(layer_idx).len());

            for p in 0..w_len {
                let orig = {
                    let w = mlp.layers[layer_idx].weights_mut();
                    let orig = w[p];
                    w[p] = orig + eps;
                    orig
                };
                let loss_plus = loss_for_mlp(&mlp, &input, &target, &mut scratch_tmp);

                {
                    let w = mlp.layers[layer_idx].weights_mut();
                    w[p] = orig - eps;
                }
                let loss_minus = loss_for_mlp(&mlp, &input, &target, &mut scratch_tmp);

                {
                    let w = mlp.layers[layer_idx].weights_mut();
                    w[p] = orig;
                }

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let analytic = grads.d_weights(layer_idx)[p];
                assert_close(analytic, numeric, abs_tol, rel_tol);
            }

            // Biases.
            let b_len = mlp.layers[layer_idx].out_dim();
            debug_assert_eq!(b_len, grads.d_biases(layer_idx).len());

            for p in 0..b_len {
                let orig = {
                    let b = mlp.layers[layer_idx].biases_mut();
                    let orig = b[p];
                    b[p] = orig + eps;
                    orig
                };
                let loss_plus = loss_for_mlp(&mlp, &input, &target, &mut scratch_tmp);

                {
                    let b = mlp.layers[layer_idx].biases_mut();
                    b[p] = orig - eps;
                }
                let loss_minus = loss_for_mlp(&mlp, &input, &target, &mut scratch_tmp);

                {
                    let b = mlp.layers[layer_idx].biases_mut();
                    b[p] = orig;
                }

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let analytic = grads.d_biases(layer_idx)[p];
                assert_close(analytic, numeric, abs_tol, rel_tol);
            }
        }

        // Inputs.
        let mut input_var = input;
        for i in 0..input_var.len() {
            let orig = input_var[i];

            input_var[i] = orig + eps;
            let loss_plus = loss_for_mlp(&mlp, &input_var, &target, &mut scratch_tmp);

            input_var[i] = orig - eps;
            let loss_minus = loss_for_mlp(&mlp, &input_var, &target, &mut scratch_tmp);

            input_var[i] = orig;

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            let analytic = d_input[i];
            assert_close(analytic, numeric, abs_tol, rel_tol);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn forward_panics_on_input_shape_mismatch() {
        let mlp = Mlp::new_with_seed(&[2, 3, 1], 0).unwrap();
        let mut scratch = mlp.scratch();
        let input = [0.0_f32; 3];
        mlp.forward(&input, &mut scratch);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn forward_panics_on_scratch_mismatch() {
        let a = Mlp::new_with_seed(&[2, 3, 1], 0).unwrap();
        let b = Mlp::new_with_seed(&[2, 4, 1], 0).unwrap();
        let mut scratch_b = b.scratch();
        let input = [0.0_f32; 2];
        a.forward(&input, &mut scratch_b);
    }
}
