use crate::Layer;

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

impl Mlp {
    /// Builds an MLP from a list of sizes.
    ///
    /// Example: `sizes = [in, hidden1, hidden2, out]`.
    pub fn new(sizes: &[usize]) -> Self {
        debug_assert!(sizes.len() >= 2, "sizes must include input and output dims");

        let mut layers = Vec::with_capacity(sizes.len() - 1);
        for w in sizes.windows(2) {
            let in_dim = w[0];
            let out_dim = w[1];
            layers.push(Layer::new(in_dim, out_dim));
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
