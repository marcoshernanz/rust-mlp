struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let weights = vec![0.0; num_inputs];
        Self { weights, bias: 0.0 }
    }

    fn forward(&self, inputs: &[f32]) -> f32 {
        debug_assert_eq!(inputs.len(), self.weights.len());

        let mut sum = self.bias;
        for (&w, &x) in self.weights.iter().zip(inputs) {
            sum = w.mul_add(x, sum);
        }
        sum.tanh()
    }
}

fn main() {
    println!("Hello, world!");
}
