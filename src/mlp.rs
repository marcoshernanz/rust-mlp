//! Multi-layer perceptron (MLP) core.
//!
//! The low-level API is intentionally allocation-free:
//! - `Mlp::forward` writes activations into a reusable `Scratch` and returns a slice.
//! - `Mlp::backward` writes gradients into a reusable `Gradients`.
//!
//! Shape mismatches are treated as programmer error and will panic via `assert!`.

use crate::Layer;
use crate::{Error, Result};

/// A feed-forward multi-layer perceptron composed of dense layers.
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

/// Reusable buffers for `Mlp::forward_batch`.
///
/// This stores per-layer outputs for all samples in the batch in flat row-major layout:
/// - each layer buffer has shape `(batch_size, out_dim)`.
#[derive(Debug, Clone)]
pub struct BatchScratch {
    batch_size: usize,
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
    pub(crate) fn from_layers(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    /// Returns the expected input dimension.
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.layers
            .first()
            .expect("mlp must have at least one layer")
            .in_dim()
    }

    /// Returns the produced output dimension.
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.layers
            .last()
            .expect("mlp must have at least one layer")
            .out_dim()
    }

    /// Returns the number of layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns a reference to a layer by index.
    ///
    /// This is primarily useful for inspection and debugging.
    #[inline]
    pub fn layer(&self, idx: usize) -> Option<&Layer> {
        self.layers.get(idx)
    }

    /// Allocate a `Scratch` buffer suitable for this model.
    pub fn scratch(&self) -> Scratch {
        Scratch::new(self)
    }

    /// Allocate a `BatchScratch` buffer suitable for this model and a fixed batch size.
    pub fn scratch_batch(&self, batch_size: usize) -> BatchScratch {
        BatchScratch::new(self, batch_size)
    }

    /// Allocate a `Gradients` buffer suitable for this model.
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

    /// Forward pass for a contiguous batch.
    ///
    /// Writes intermediate activations into `scratch` and returns the final output buffer
    /// for the whole batch (flat row-major).
    ///
    /// Shape contract:
    /// - `inputs.len() == batch_size * self.input_dim()`
    /// - `scratch` must be built for this `Mlp` and the same `batch_size`
    pub fn forward_batch<'a>(&self, inputs: &[f32], scratch: &'a mut BatchScratch) -> &'a [f32] {
        let batch_size = scratch.batch_size;
        assert!(batch_size > 0, "batch_size must be > 0");
        assert_eq!(
            inputs.len(),
            batch_size * self.input_dim(),
            "inputs len {} does not match batch_size * input_dim ({} * {})",
            inputs.len(),
            batch_size,
            self.input_dim()
        );
        assert_eq!(
            scratch.layer_outputs.len(),
            self.layers.len(),
            "batch scratch has {} layer outputs, model has {} layers",
            scratch.layer_outputs.len(),
            self.layers.len()
        );

        for (idx, layer) in self.layers.iter().enumerate() {
            let out_dim = layer.out_dim();

            if idx == 0 {
                let out = &mut scratch.layer_outputs[0];
                assert_eq!(
                    out.len(),
                    batch_size * out_dim,
                    "batch scratch layer 0 output len {} does not match batch_size * out_dim ({} * {})",
                    out.len(),
                    batch_size,
                    out_dim
                );

                for b in 0..batch_size {
                    let x0 = b * self.input_dim();
                    let x = &inputs[x0..x0 + self.input_dim()];
                    let y0 = b * out_dim;
                    let y = &mut out[y0..y0 + out_dim];
                    layer.forward(x, y);
                }
            } else {
                // Borrow the previous output immutably and the current output mutably.
                let (left, right) = scratch.layer_outputs.split_at_mut(idx);
                let prev = &left[idx - 1];
                let out = &mut right[0];

                let in_dim = layer.in_dim();
                assert_eq!(
                    prev.len(),
                    batch_size * in_dim,
                    "batch scratch layer {} input len {} does not match batch_size * in_dim ({} * {})",
                    idx - 1,
                    prev.len(),
                    batch_size,
                    in_dim
                );
                assert_eq!(
                    out.len(),
                    batch_size * out_dim,
                    "batch scratch layer {idx} output len {} does not match batch_size * out_dim ({} * {})",
                    out.len(),
                    batch_size,
                    out_dim
                );

                for b in 0..batch_size {
                    let x0 = b * in_dim;
                    let x = &prev[x0..x0 + in_dim];
                    let y0 = b * out_dim;
                    let y = &mut out[y0..y0 + out_dim];
                    layer.forward(x, y);
                }
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

    /// Backward pass for a single sample (parameter accumulation semantics).
    ///
    /// This is identical to `backward` except that parameter gradients are *accumulated*:
    /// - `grads.d_weights` and `grads.d_biases` are accumulated into (`+=`)
    /// - `grads.d_layer_outputs` and `grads.d_input` are overwritten
    ///
    /// This is useful for mini-batch training.
    ///
    /// You must call `forward` first using the same `input` and `scratch`.
    /// Before calling this, write the upstream gradient `dL/d(output)` into
    /// `grads.d_output_mut()`.
    pub fn backward_accumulate<'a>(
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
                layer.backward_accumulate(
                    layer_input,
                    layer_output,
                    d_outputs,
                    &mut grads.d_input,
                    &mut grads.d_weights[0],
                    &mut grads.d_biases[0],
                );
            } else {
                let (left, right) = grads.d_layer_outputs.split_at_mut(idx);
                let d_inputs_prev = &mut left[idx - 1];
                let d_outputs = &right[0];
                layer.backward_accumulate(
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

    /// Backward pass for a contiguous batch.
    ///
    /// This overwrites `grads` with the *mean* parameter gradients over the batch.
    ///
    /// Inputs:
    /// - `inputs`: flat row-major with shape `(batch_size, input_dim)`
    /// - `scratch`: activations from `forward_batch`
    /// - `d_outputs`: flat row-major upstream gradients with shape `(batch_size, output_dim)`
    pub fn backward_batch(
        &self,
        inputs: &[f32],
        scratch: &BatchScratch,
        d_outputs: &[f32],
        grads: &mut Gradients,
    ) {
        let batch_size = scratch.batch_size;
        assert!(batch_size > 0, "batch_size must be > 0");
        assert_eq!(
            inputs.len(),
            batch_size * self.input_dim(),
            "inputs len {} does not match batch_size * input_dim ({} * {})",
            inputs.len(),
            batch_size,
            self.input_dim()
        );
        assert_eq!(
            d_outputs.len(),
            batch_size * self.output_dim(),
            "d_outputs len {} does not match batch_size * output_dim ({} * {})",
            d_outputs.len(),
            batch_size,
            self.output_dim()
        );
        assert_eq!(
            scratch.layer_outputs.len(),
            self.layers.len(),
            "batch scratch has {} layer outputs, model has {} layers",
            scratch.layer_outputs.len(),
            self.layers.len()
        );

        for (idx, (buf, layer)) in scratch.layer_outputs.iter().zip(&self.layers).enumerate() {
            assert_eq!(
                buf.len(),
                batch_size * layer.out_dim(),
                "batch scratch layer {idx} output len {} does not match batch_size * out_dim ({} * {})",
                buf.len(),
                batch_size,
                layer.out_dim()
            );
        }

        grads.zero_params();

        let out_dim = self.output_dim();
        let in_dim0 = self.input_dim();

        for b in 0..batch_size {
            let dy0 = b * out_dim;
            grads
                .d_output_mut()
                .copy_from_slice(&d_outputs[dy0..dy0 + out_dim]);

            let x0 = b * in_dim0;
            let input = &inputs[x0..x0 + in_dim0];

            for idx in (0..self.layers.len()).rev() {
                let layer = &self.layers[idx];

                let layer_input: &[f32] = if idx == 0 {
                    input
                } else {
                    let in_dim = layer.in_dim();
                    let start = b * in_dim;
                    &scratch.layer_outputs[idx - 1][start..start + in_dim]
                };

                let out_dim = layer.out_dim();
                let start = b * out_dim;
                let layer_output: &[f32] = &scratch.layer_outputs[idx][start..start + out_dim];

                if idx == 0 {
                    let d_outputs = &grads.d_layer_outputs[0];
                    layer.backward_accumulate(
                        layer_input,
                        layer_output,
                        d_outputs,
                        &mut grads.d_input,
                        &mut grads.d_weights[0],
                        &mut grads.d_biases[0],
                    );
                } else {
                    let (left, right) = grads.d_layer_outputs.split_at_mut(idx);
                    let d_inputs_prev = &mut left[idx - 1];
                    let d_outputs = &right[0];
                    layer.backward_accumulate(
                        layer_input,
                        layer_output,
                        d_outputs,
                        d_inputs_prev,
                        &mut grads.d_weights[idx],
                        &mut grads.d_biases[idx],
                    );
                }
            }
        }

        grads.scale_params(1.0 / batch_size as f32);
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

    /// Shape-safe, non-allocating inference for a single input.
    ///
    /// This validates shapes and returns `Result` instead of panicking.
    /// Internally it uses the low-level `forward` hot path.
    pub fn predict_one_into(
        &self,
        input: &[f32],
        scratch: &mut Scratch,
        out: &mut [f32],
    ) -> Result<()> {
        if input.len() != self.input_dim() {
            return Err(Error::InvalidData(format!(
                "input len {} does not match model input_dim {}",
                input.len(),
                self.input_dim()
            )));
        }
        if out.len() != self.output_dim() {
            return Err(Error::InvalidData(format!(
                "out len {} does not match model output_dim {}",
                out.len(),
                self.output_dim()
            )));
        }
        if scratch.layer_outputs.len() != self.layers.len() {
            return Err(Error::InvalidData(format!(
                "scratch has {} layer outputs, model has {} layers",
                scratch.layer_outputs.len(),
                self.layers.len()
            )));
        }
        for (idx, (buf, layer)) in scratch.layer_outputs.iter().zip(&self.layers).enumerate() {
            if buf.len() != layer.out_dim() {
                return Err(Error::InvalidData(format!(
                    "scratch layer {idx} output len {} does not match layer out_dim {}",
                    buf.len(),
                    layer.out_dim()
                )));
            }
        }

        let y = self.forward(input, scratch);
        out.copy_from_slice(y);
        Ok(())
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
    /// Allocate a `Trainer` (scratch + gradients) for `mlp`.
    pub fn new(mlp: &Mlp) -> Self {
        Self {
            scratch: Scratch::new(mlp),
            grads: Gradients::new(mlp),
        }
    }
}

impl Scratch {
    /// Allocate a scratch buffer suitable for `mlp`.
    pub fn new(mlp: &Mlp) -> Self {
        let mut layer_outputs = Vec::with_capacity(mlp.layers.len());
        for layer in &mlp.layers {
            layer_outputs.push(vec![0.0; layer.out_dim()]);
        }
        Self { layer_outputs }
    }

    #[inline]
    /// Returns the final model output slice from the last `forward` call.
    pub fn output(&self) -> &[f32] {
        self.layer_outputs
            .last()
            .expect("scratch must have at least one layer output")
            .as_slice()
    }
}

impl BatchScratch {
    /// Allocate a batch scratch buffer suitable for `mlp` and `batch_size`.
    pub fn new(mlp: &Mlp, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");

        let mut layer_outputs = Vec::with_capacity(mlp.layers.len());
        for layer in &mlp.layers {
            layer_outputs.push(vec![0.0; batch_size * layer.out_dim()]);
        }
        Self {
            batch_size,
            layer_outputs,
        }
    }

    #[inline]
    /// Returns the fixed batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[inline]
    /// Returns the final model output buffer from the last `forward_batch` call.
    ///
    /// Shape: `(batch_size * output_dim,)`.
    pub fn output(&self) -> &[f32] {
        self.layer_outputs
            .last()
            .expect("batch scratch must have at least one layer output")
            .as_slice()
    }

    #[inline]
    /// Returns the `idx`-th output row (shape: `(output_dim,)`).
    ///
    /// Panics if `idx >= batch_size`.
    pub fn output_row(&self, idx: usize) -> &[f32] {
        assert!(idx < self.batch_size, "batch index out of bounds");

        let out = self
            .layer_outputs
            .last()
            .expect("batch scratch must have at least one layer output");
        let out_dim = out.len() / self.batch_size;
        let start = idx * out_dim;
        &out[start..start + out_dim]
    }
}

impl Gradients {
    /// Allocate gradient buffers suitable for `mlp`.
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
    /// Immutable view of the final upstream gradient buffer.
    pub fn d_output(&self) -> &[f32] {
        self.d_layer_outputs
            .last()
            .expect("mlp must have at least one layer")
            .as_slice()
    }

    #[inline]
    /// Returns dL/d(input) computed by the most recent `backward` call.
    pub fn d_input(&self) -> &[f32] {
        &self.d_input
    }

    /// Returns the weight gradient for the given layer (row-major `(out_dim, in_dim)`).
    #[inline]
    pub fn d_weights(&self, layer_idx: usize) -> &[f32] {
        &self.d_weights[layer_idx]
    }

    /// Returns the bias gradient for the given layer (length `out_dim`).
    #[inline]
    pub fn d_biases(&self, layer_idx: usize) -> &[f32] {
        &self.d_biases[layer_idx]
    }

    /// Zero the parameter gradient buffers (`d_weights` and `d_biases`).
    #[inline]
    pub fn zero_params(&mut self) {
        for w in &mut self.d_weights {
            w.fill(0.0);
        }
        for b in &mut self.d_biases {
            b.fill(0.0);
        }
    }

    /// Scale parameter gradients (`d_weights` and `d_biases`) in place.
    #[inline]
    pub fn scale_params(&mut self, scale: f32) {
        assert!(scale.is_finite(), "scale must be finite");

        for w in &mut self.d_weights {
            for v in w.iter_mut() {
                *v *= scale;
            }
        }
        for b in &mut self.d_biases {
            for v in b.iter_mut() {
                *v *= scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::{Activation, MlpBuilder};

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
    fn predict_one_into_validates_shapes() {
        let mlp = MlpBuilder::new(2)
            .unwrap()
            .add_layer(3, Activation::Tanh)
            .unwrap()
            .add_layer(1, Activation::Identity)
            .unwrap()
            .build_with_seed(0)
            .unwrap();

        let mut scratch = mlp.scratch();
        let mut out = [0.0_f32; 1];

        let ok = mlp.predict_one_into(&[0.1, 0.2], &mut scratch, &mut out);
        assert!(ok.is_ok());

        let err = mlp.predict_one_into(&[0.1_f32], &mut scratch, &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn backward_matches_numeric_gradients_for_tanh() {
        let mut mlp = MlpBuilder::new(2)
            .unwrap()
            .add_layer(3, Activation::Tanh)
            .unwrap()
            .add_layer(1, Activation::Tanh)
            .unwrap()
            .build_with_seed(0)
            .unwrap();

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

    #[test]
    #[should_panic]
    fn forward_panics_on_input_shape_mismatch() {
        let mut rng = StdRng::seed_from_u64(0);
        let mlp = MlpBuilder::new(2)
            .unwrap()
            .add_layer(3, Activation::Tanh)
            .unwrap()
            .add_layer(1, Activation::Tanh)
            .unwrap()
            .build_with_rng(&mut rng)
            .unwrap();
        let mut scratch = mlp.scratch();
        let input = [0.0_f32; 3];
        mlp.forward(&input, &mut scratch);
    }
}
