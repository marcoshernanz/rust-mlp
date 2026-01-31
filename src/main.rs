struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

impl Neuron {
    fn new(num_inputs: usize) -> Neuron {
        let mut weights = Vec::new();
        for _ in 0..num_inputs {
            weights.push(0.0);
        }
        return Neuron { weights, bias: 0.0 };
    }
}

fn main() {
    println!("Hello, world!");
}
