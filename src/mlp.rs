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

/// Reusable buffers for `Mlp::backward_batch`.
///
/// This stores two work buffers sized for the maximum layer dimension.
#[derive(Debug, Clone)]
pub struct BatchBackpropScratch {
    batch_size: usize,
    max_dim: usize,
    buf0: Vec<f32>,
    buf1: Vec<f32>,
}

impl BatchBackpropScratch {
    /// Allocate a backprop scratch buffer suitable for `mlp` and `batch_size`.
    pub fn new(mlp: &Mlp, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");

        let mut max_dim = mlp.input_dim();
        for layer in &mlp.layers {
            max_dim = max_dim.max(layer.in_dim());
            max_dim = max_dim.max(layer.out_dim());
        }

        let len = batch_size * max_dim;
        Self {
            batch_size,
            max_dim,
            buf0: vec![0.0; len],
            buf1: vec![0.0; len],
        }
    }
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

    #[inline]
    pub(crate) fn layer_mut(&mut self, idx: usize) -> Option<&mut Layer> {
        self.layers.get_mut(idx)
    }

    /// Allocate a `Scratch` buffer suitable for this model.
    pub fn scratch(&self) -> Scratch {
        Scratch::new(self)
    }

    /// Allocate a `BatchScratch` buffer suitable for this model and a fixed batch size.
    pub fn scratch_batch(&self, batch_size: usize) -> BatchScratch {
        BatchScratch::new(self, batch_size)
    }

    /// Allocate a `BatchBackpropScratch` buffer suitable for this model and a fixed batch size.
    pub fn backprop_scratch_batch(&self, batch_size: usize) -> BatchBackpropScratch {
        BatchBackpropScratch::new(self, batch_size)
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
            let in_dim = layer.in_dim();

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

                // out = inputs * weights^T
                // inputs: (batch_size, in_dim) row-major
                // weights: (out_dim, in_dim) row-major, so weights^T is represented by strides.
                crate::matmul::gemm_f32(
                    batch_size,
                    out_dim,
                    in_dim,
                    1.0,
                    inputs,
                    in_dim,
                    1,
                    layer.weights(),
                    1,
                    in_dim,
                    0.0,
                    out,
                    out_dim,
                    1,
                );

                let activation = layer.activation();
                let b = layer.biases();
                debug_assert_eq!(b.len(), out_dim);
                for row in 0..batch_size {
                    let o0 = row * out_dim;
                    for o in 0..out_dim {
                        let z = out[o0 + o] + b[o];
                        out[o0 + o] = activation.forward(z);
                    }
                }
            } else {
                // Borrow the previous output immutably and the current output mutably.
                let (left, right) = scratch.layer_outputs.split_at_mut(idx);
                let prev = &left[idx - 1];
                let out = &mut right[0];

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

                crate::matmul::gemm_f32(
                    batch_size,
                    out_dim,
                    in_dim,
                    1.0,
                    prev,
                    in_dim,
                    1,
                    layer.weights(),
                    1,
                    in_dim,
                    0.0,
                    out,
                    out_dim,
                    1,
                );

                let activation = layer.activation();
                let b = layer.biases();
                debug_assert_eq!(b.len(), out_dim);
                for row in 0..batch_size {
                    let o0 = row * out_dim;
                    for o in 0..out_dim {
                        let z = out[o0 + o] + b[o];
                        out[o0 + o] = activation.forward(z);
                    }
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
        backprop_scratch: &mut BatchBackpropScratch,
    ) {
        let batch_size = scratch.batch_size;
        assert!(batch_size > 0, "batch_size must be > 0");
        assert_eq!(
            backprop_scratch.batch_size, batch_size,
            "backprop scratch batch_size {} does not match scratch batch_size {}",
            backprop_scratch.batch_size, batch_size
        );
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

        // Overwrite semantics.
        for (idx, layer) in self.layers.iter().enumerate() {
            let out_dim = layer.out_dim();
            let in_dim = layer.in_dim();
            grads.d_weights[idx].fill(0.0);
            grads.d_biases[idx].fill(0.0);
            debug_assert_eq!(grads.d_weights[idx].len(), out_dim * in_dim);
            debug_assert_eq!(grads.d_biases[idx].len(), out_dim);
        }

        // Ensure work buffers are large enough.
        let needed = batch_size * backprop_scratch.max_dim;
        assert!(
            backprop_scratch.buf0.len() >= needed && backprop_scratch.buf1.len() >= needed,
            "backprop scratch buffers are too small"
        );

        let inv_batch = 1.0 / batch_size as f32;

        // d_cur holds dL/dy for the current layer output, then overwritten to dL/dz.
        let mut cur_dim = self.output_dim();
        let cur_len = batch_size * cur_dim;
        backprop_scratch.buf0[..cur_len].copy_from_slice(d_outputs);
        let mut cur_in_buf0 = true;

        for idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[idx];
            let out_dim = layer.out_dim();
            let in_dim = layer.in_dim();

            debug_assert_eq!(cur_dim, out_dim);

            let (cur_buf, other_buf) = if cur_in_buf0 {
                (&mut backprop_scratch.buf0, &mut backprop_scratch.buf1)
            } else {
                (&mut backprop_scratch.buf1, &mut backprop_scratch.buf0)
            };

            let d_cur: &mut [f32] = &mut cur_buf[..batch_size * out_dim];

            let y = &scratch.layer_outputs[idx];
            debug_assert_eq!(y.len(), batch_size * out_dim);

            // dZ = dY * activation'(y)
            let activation = layer.activation();
            for i in 0..d_cur.len() {
                d_cur[i] *= activation.grad_from_output(y[i]);
            }

            // db = mean over batch of dZ
            let db = &mut grads.d_biases[idx];
            assert_eq!(db.len(), out_dim);
            db.fill(0.0);
            for b in 0..batch_size {
                let row0 = b * out_dim;
                for o in 0..out_dim {
                    db[o] += d_cur[row0 + o];
                }
            }
            for v in db.iter_mut() {
                *v *= inv_batch;
            }

            // dW = mean over batch of dZ^T * X
            let x: &[f32] = if idx == 0 {
                inputs
            } else {
                &scratch.layer_outputs[idx - 1]
            };
            assert_eq!(x.len(), batch_size * in_dim);

            let dw = &mut grads.d_weights[idx];
            assert_eq!(dw.len(), out_dim * in_dim);
            crate::matmul::gemm_f32(
                out_dim, in_dim, batch_size, inv_batch, d_cur, 1, out_dim, x, in_dim, 1, 0.0, dw,
                in_dim, 1,
            );

            if idx == 0 {
                break;
            }

            // dX = dZ * W into the other buffer.
            let d_x: &mut [f32] = &mut other_buf[..batch_size * in_dim];
            crate::matmul::gemm_f32(
                batch_size,
                in_dim,
                out_dim,
                1.0,
                d_cur,
                out_dim,
                1,
                layer.weights(),
                in_dim,
                1,
                0.0,
                d_x,
                in_dim,
                1,
            );

            cur_in_buf0 = !cur_in_buf0;
            cur_dim = in_dim;
        }
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

    /// Apply decoupled weight decay to all layer weights.
    ///
    /// This updates weights only (biases are not decayed): `w -= lr * weight_decay * w`.
    pub(crate) fn apply_weight_decay(&mut self, lr: f32, weight_decay: f32) {
        assert!(
            lr.is_finite() && lr > 0.0,
            "learning rate must be finite and > 0"
        );
        assert!(
            weight_decay.is_finite() && weight_decay >= 0.0,
            "weight_decay must be finite and >= 0"
        );

        if weight_decay == 0.0 {
            return;
        }

        for layer in &mut self.layers {
            layer.apply_weight_decay(lr, weight_decay);
        }
    }

    /// Shape-safe, non-allocating inference.
    ///
    /// This validates shapes and returns `Result` instead of panicking.
    /// Internally it uses the low-level `forward` hot path.
    pub fn predict_into(
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

    /// Shape-safe, non-allocating inference for a single input.
    ///
    /// Alias of [`Mlp::predict_into`].
    #[inline]
    pub fn predict_one_into(
        &self,
        input: &[f32],
        scratch: &mut Scratch,
        out: &mut [f32],
    ) -> Result<()> {
        self.predict_into(input, scratch, out)
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

    /// Mutable weight gradient for the given layer.
    #[inline]
    pub fn d_weights_mut(&mut self, layer_idx: usize) -> &mut [f32] {
        &mut self.d_weights[layer_idx]
    }

    /// Returns the bias gradient for the given layer (length `out_dim`).
    #[inline]
    pub fn d_biases(&self, layer_idx: usize) -> &[f32] {
        &self.d_biases[layer_idx]
    }

    /// Mutable bias gradient for the given layer.
    #[inline]
    pub fn d_biases_mut(&mut self, layer_idx: usize) -> &mut [f32] {
        &mut self.d_biases[layer_idx]
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

    /// Compute the global L2 norm of parameter gradients.
    pub fn global_l2_norm_params(&self) -> f32 {
        let mut sum_sq = 0.0_f32;
        for w in &self.d_weights {
            for &v in w {
                sum_sq = v.mul_add(v, sum_sq);
            }
        }
        for b in &self.d_biases {
            for &v in b {
                sum_sq = v.mul_add(v, sum_sq);
            }
        }
        sum_sq.sqrt()
    }

    /// Clip parameter gradients by global norm.
    ///
    /// Returns the pre-clip norm.
    pub fn clip_global_norm_params(&mut self, max_norm: f32) -> f32 {
        assert!(
            max_norm.is_finite() && max_norm > 0.0,
            "max_norm must be finite and > 0"
        );

        let norm = self.global_l2_norm_params();
        if norm > max_norm && norm > 0.0 {
            self.scale_params(max_norm / norm);
        }
        norm
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
    fn predict_into_validates_shapes() {
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

        let ok = mlp.predict_into(&[0.1, 0.2], &mut scratch, &mut out);
        assert!(ok.is_ok());

        let err = mlp.predict_into(&[0.1_f32], &mut scratch, &mut out);
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
