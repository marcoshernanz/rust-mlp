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
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = vec![0.0; in_dim * out_dim];
        let biases = vec![0.0; out_dim];
        Self {
            in_dim,
            out_dim,
            weights,
            biases,
        }
    }

    #[inline]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    #[inline]
    pub fn out_dim(&self) -> usize {
        self.out_dim
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

        for o in 0..self.out_dim {
            let mut sum = self.biases[o];
            let row = o * self.in_dim;
            for i in 0..self.in_dim {
                sum = self.weights[row + i].mul_add(inputs[i], sum);
            }
            outputs[o] = sum.tanh();
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
}
