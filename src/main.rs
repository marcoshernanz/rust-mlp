struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let weights = vec![0.0; num_inputs];
        Self { weights, bias: 0.0 }
    }
}

fn main() {
    println!("Hello, world!");
}
