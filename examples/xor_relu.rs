use rust_mlp::{Activation, Dataset, FitConfig, Loss, MlpBuilder};

fn main() -> rust_mlp::Result<()> {
    // Classic XOR dataset.
    let xs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let ys = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    let train = Dataset::from_rows(&xs, &ys)?;

    // 2 -> 8 -> 1 network.
    // ReLU hidden layer, sigmoid output for a probability-like output.
    let mut mlp = MlpBuilder::new(2)?
        .add_layer(8, Activation::ReLU)?
        .add_layer(1, Activation::Sigmoid)?
        .build_with_seed(0)?;

    let report = mlp.fit(
        &train,
        None,
        FitConfig {
            epochs: 2_000,
            lr: 0.1,
            batch_size: 4,
            shuffle: rust_mlp::Shuffle::Seeded(0),
            lr_schedule: rust_mlp::LrSchedule::Constant,
            optimizer: rust_mlp::Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
            weight_decay: 0.0,
            grad_clip_norm: None,
            loss: Loss::Mse,
            metrics: vec![],
        },
    )?;

    let mse = mlp.evaluate_mse(&train)?;
    let last = report
        .epochs
        .last()
        .expect("fit must report at least 1 epoch");
    println!("final_loss_from_fit={} train_mse={}", last.train.loss, mse);

    let mut scratch = mlp.scratch();
    let mut out = [0.0_f32; 1];
    for x in xs {
        mlp.predict_into(&x, &mut scratch, &mut out)?;
        println!("x={x:?} y={:?}", out[0]);
    }

    Ok(())
}
