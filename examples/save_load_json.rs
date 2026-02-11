#[cfg(not(feature = "serde"))]
fn main() {
    println!("enable the `serde` feature: cargo run --example save_load_json --features serde");
}

#[cfg(feature = "serde")]
fn main() -> rust_mlp::Result<()> {
    use rust_mlp::{Activation, FitConfig, Loss, MlpBuilder, Optimizer, Shuffle};

    let xs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let ys = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    let train = rust_mlp::Dataset::from_rows(&xs, &ys)?;

    let mut mlp = MlpBuilder::new(2)?
        .add_layer(8, Activation::ReLU)?
        .add_layer(1, Activation::Sigmoid)?
        .build_with_seed(0)?;

    mlp.fit(
        &train,
        None,
        FitConfig {
            epochs: 200,
            lr: 0.1,
            batch_size: 4,
            shuffle: Shuffle::Seeded(0),
            lr_schedule: rust_mlp::LrSchedule::Constant,
            optimizer: Optimizer::Adam {
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

    let path = "target/tmp_mlp.json";
    mlp.save_json(path)?;

    let loaded = rust_mlp::Mlp::load_json(path)?;
    let _ = loaded;
    println!("saved and loaded model: {path}");
    Ok(())
}
