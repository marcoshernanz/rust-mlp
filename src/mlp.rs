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
    /// Builds an MLP from a list of sizes.
    ///
    /// Example: `sizes = [in, hidden1, hidden2, out]`.
    pub fn new(sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        Self::new_with_rng(sizes, &mut rng)
    }

    pub fn new_with_seed(sizes: &[usize], seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new_with_rng(sizes, &mut rng)
    }

    pub fn new_with_rng<R: Rng + ?Sized>(sizes: &[usize], rng: &mut R) -> Self {
        assert!(sizes.len() >= 2, "sizes must include input and output dims");
        assert!(sizes.iter().all(|&d| d > 0), "all layer sizes must be > 0");

        let mut layers = Vec::with_capacity(sizes.len() - 1);
        for w in sizes.windows(2) {
            let in_dim = w[0];
            let out_dim = w[1];
            layers.push(Layer::new_with_rng(in_dim, out_dim, Init::XavierTanh, rng));
        }
        Self { layers }
    }

    #[inline]
    pub fn input_dim(&self) -> usize {
        self.layers.first().map(|l| l.in_dim()).unwrap_or(0)
    }

    #[inline]
    pub fn output_dim(&self) -> usize {
        self.layers.last().map(|l| l.out_dim()).unwrap_or(0)
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

    /// Forward pass for a single sample.
    ///
    /// Writes intermediate activations into `scratch` and returns the final output slice.
    ///
    /// Shape contract:
    /// - `input.len() == self.input_dim()`
    /// - `scratch` must be built for this `Mlp` (same layer count and output sizes)
    pub fn forward<'a>(&self, input: &[f32], scratch: &'a mut Scratch) -> &'a [f32] {
        debug_assert_eq!(input.len(), self.input_dim());
        debug_assert_eq!(scratch.layer_outputs.len(), self.layers.len());

        for (idx, layer) in self.layers.iter().enumerate() {
            if idx == 0 {
                let out = &mut scratch.layer_outputs[0];
                debug_assert_eq!(out.len(), layer.out_dim());
                layer.forward(input, out);
            } else {
                // Borrow the previous output immutably and the current output mutably.
                let (left, right) = scratch.layer_outputs.split_at_mut(idx);
                let prev = &left[idx - 1];
                let out = &mut right[0];
                debug_assert_eq!(out.len(), layer.out_dim());
                layer.forward(prev, out);
            }
        }

        scratch.output()
    }

    /// Backward pass for a single sample.
    ///
    /// You must call `forward` first using the same `input` and `scratch`.
    ///
    /// Overwrite semantics:
    /// - `grads` is overwritten with gradients for this sample.
    ///
    /// Returns dL/d(input).
    pub fn backward<'a>(
        &self,
        input: &[f32],
        scratch: &Scratch,
        d_output: &[f32],
        grads: &'a mut Gradients,
    ) -> &'a [f32] {
        debug_assert_eq!(input.len(), self.input_dim());
        debug_assert_eq!(scratch.layer_outputs.len(), self.layers.len());
        debug_assert_eq!(d_output.len(), self.output_dim());

        debug_assert_eq!(grads.d_weights.len(), self.layers.len());
        debug_assert_eq!(grads.d_biases.len(), self.layers.len());
        debug_assert_eq!(grads.d_layer_outputs.len(), self.layers.len());
        debug_assert_eq!(grads.d_input.len(), self.input_dim());

        if self.layers.is_empty() {
            grads.d_input.fill(0.0);
            return &grads.d_input;
        }

        let last = self.layers.len() - 1;
        debug_assert_eq!(grads.d_layer_outputs[last].len(), self.output_dim());
        grads.d_layer_outputs[last].copy_from_slice(d_output);

        for idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[idx];

            let layer_input: &[f32] = if idx == 0 {
                input
            } else {
                &scratch.layer_outputs[idx - 1]
            };

            let layer_output: &[f32] = &scratch.layer_outputs[idx];
            debug_assert_eq!(layer_output.len(), layer.out_dim());

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
        self.layer_outputs.last().map(Vec::as_slice).unwrap_or(&[])
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
