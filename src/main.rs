fn main() {
    // Temporary smoke check while the library scaffold is built out.
    let neuron = rust_mlp::Neuron::new(3);
    let y = neuron.forward(&[0.0, 0.0, 0.0]);

    let layer = rust_mlp::Layer::new_with_seed(3, 2, rust_mlp::Init::XavierTanh, 0);
    let mut out = [0.0_f32; 2];
    layer.forward(&[0.0, 0.0, 0.0], &mut out);

    // Backward smoke check (shapes and API wiring).
    let inputs = [0.0_f32; 3];
    let outputs = out;
    let d_outputs = [1.0_f32; 2];
    let mut d_inputs = [0.0_f32; 3];
    let mut d_weights = vec![0.0_f32; 3 * 2];
    let mut d_biases = [0.0_f32; 2];
    layer.backward(
        &inputs,
        &outputs,
        &d_outputs,
        &mut d_inputs,
        &mut d_weights,
        &mut d_biases,
    );

    let mlp = rust_mlp::Mlp::new_with_seed(&[3, 4, 2], 0);
    let mut scratch = mlp.scratch();
    let input = [0.0_f32; 3];
    mlp.forward(&input, &mut scratch);

    let mut grads = mlp.gradients();
    let d_input = mlp.backward(&input, &scratch, &[1.0, 1.0], &mut grads);

    println!("{y} {out:?} {:?} {d_input:?}", scratch.output());
}
