use rand::SeedableRng;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;

fn main() {
    // Task: learn y = tanh(x0 + x1)
    // This is representable by a single layer: y = tanh(w0*x0 + w1*x1 + b).
    let mut mlp = rust_mlp::Mlp::new_with_seed(&[2, 1], 0);
    let mut scratch = mlp.scratch();
    let mut grads = mlp.gradients();

    let opt = rust_mlp::Sgd::new(0.1);

    let mut rng = StdRng::seed_from_u64(1);
    let dist = Uniform::new(-1.0_f32, 1.0_f32);

    for step in 0..5_000 {
        let x0 = dist.sample(&mut rng);
        let x1 = dist.sample(&mut rng);
        let input = [x0, x1];
        let target = [(x0 + x1).tanh()];

        mlp.forward(&input, &mut scratch);

        let pred = scratch.output();
        let loss = rust_mlp::loss::mse_backward(pred, &target, grads.d_output_mut());
        mlp.backward_in_place(&input, &scratch, &mut grads);
        opt.step(&mut mlp, &grads);

        if step % 500 == 0 {
            println!("step={step} loss={loss:.6}");
        }
    }

    // Quick evaluation.
    let test = [0.2_f32, -0.7_f32];
    let target = (test[0] + test[1]).tanh();
    mlp.forward(&test, &mut scratch);
    let pred = scratch.output()[0];
    println!("test pred={pred:.6} target={target:.6}");
}
