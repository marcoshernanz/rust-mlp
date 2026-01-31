#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

impl Neuron {
    /// Creates a neuron with `num_inputs` weights and a bias.
    ///
    /// Invariants:
    /// - `weights.len() == num_inputs`
    /// - `bias` is a scalar parameter
    ///
    /// Note: initialization is currently all zeros for simplicity.
    /// For training, you'll typically want a random initializer (e.g. Xavier).
    #[inline]
    pub fn new(num_inputs: usize) -> Self {
        let weights = vec![0.0; num_inputs];
        Self { weights, bias: 0.0 }
    }

    /// Forward pass for a single sample.
    ///
    /// Computes:
    /// - `z = dot(weights, inputs) + bias`
    /// - `y = tanh(z)`
    ///
    /// Shape contract:
    /// - `inputs.len() == self.weights.len()`
    #[inline]
    pub fn forward(&self, inputs: &[f32]) -> f32 {
        debug_assert_eq!(inputs.len(), self.weights.len());

        let mut sum = self.bias;
        for (&w, &x) in self.weights.iter().zip(inputs) {
            sum = w.mul_add(x, sum);
        }
        sum.tanh()
    }

    /// Backward pass for a single sample.
    ///
    /// You provide the upstream gradient `d_output = dL/dy` and the previous
    /// forward output `output = y` (so we can compute `tanh'(z)` cheaply).
    ///
    /// Computes and writes:
    /// - `d_weights[i] = dL/dw_i`
    /// - `d_inputs[i] = dL/dx_i` (for propagating gradients to the previous layer)
    ///
    /// Returns:
    /// - `d_bias = dL/db`
    ///
    /// Shape contract:
    /// - `inputs.len() == self.weights.len()`
    /// - `d_inputs.len() == self.weights.len()`
    /// - `d_weights.len() == self.weights.len()`
    ///
    /// This function overwrites `d_inputs`/`d_weights`. For minibatch training,
    /// an outer loop typically accumulates (`+=`) into gradient buffers.
    #[inline]
    pub fn backward(
        &self,
        inputs: &[f32],
        output: f32,
        d_output: f32,
        d_inputs: &mut [f32],
        d_weights: &mut [f32],
    ) -> f32 {
        debug_assert_eq!(inputs.len(), self.weights.len());
        debug_assert_eq!(d_inputs.len(), self.weights.len());
        debug_assert_eq!(d_weights.len(), self.weights.len());

        // tanh'(z) = 1 - tanh(z)^2 = 1 - output^2
        let d_z = d_output * (1.0 - output * output);

        for i in 0..self.weights.len() {
            d_weights[i] = d_z * inputs[i];
            d_inputs[i] = d_z * self.weights[i];
        }

        d_z
    }
}
