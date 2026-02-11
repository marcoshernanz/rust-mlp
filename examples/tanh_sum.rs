use rand::SeedableRng;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;

fn main() {
    // Task: learn y = tanh(x0 + x1)
    // This is representable by a single layer: y = tanh(w0*x0 + w1*x1 + b).
    let mut mlp = rust_mlp::MlpBuilder::new(2)
        .unwrap()
        .add_layer(1, rust_mlp::Activation::Tanh)
        .unwrap()
        .build_with_seed(0)
        .unwrap();

    let mut rng = StdRng::seed_from_u64(1);
    let dist = Uniform::new(-1.0_f32, 1.0_f32);

    // Build train set.
    let mut train_x = Vec::with_capacity(2 * 256);
    let mut train_y = Vec::with_capacity(256);
    for _ in 0..256 {
        let x0 = dist.sample(&mut rng);
        let x1 = dist.sample(&mut rng);
        train_x.extend_from_slice(&[x0, x1]);
        train_y.push((x0 + x1).tanh());
    }

    // Build test set.
    let mut test_x = Vec::with_capacity(2 * 64);
    let mut test_y = Vec::with_capacity(64);
    for _ in 0..64 {
        let x0 = dist.sample(&mut rng);
        let x1 = dist.sample(&mut rng);
        test_x.extend_from_slice(&[x0, x1]);
        test_y.push((x0 + x1).tanh());
    }

    let train = rust_mlp::Dataset::from_flat(train_x, train_y, 2, 1).unwrap();
    let test = rust_mlp::Dataset::from_flat(test_x, test_y, 2, 1).unwrap();

    let report = mlp
        .fit(
            &train,
            None,
            rust_mlp::FitConfig {
                epochs: 200,
                lr: 0.1,
                loss: rust_mlp::Loss::Mse,
                metrics: vec![],
            },
        )
        .unwrap();

    let test_mse = mlp.evaluate_mse(&test).unwrap();
    let _test_preds = mlp.predict_inputs(test.inputs()).unwrap();
    println!(
        "train_mse={:.6} test_mse={:.6}",
        report.epochs.last().unwrap().train.loss,
        test_mse
    );
}
