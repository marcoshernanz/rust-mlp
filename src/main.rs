fn main() {
    // Temporary smoke check while the library scaffold is built out.
    let neuron = rust_mlp::Neuron::new(3);
    let y = neuron.forward(&[0.0, 0.0, 0.0]);
    println!("{y}");
}
