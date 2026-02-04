use rust_mlp::{Activation, Dataset, FitConfig, MlpBuilder};

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
        FitConfig {
            epochs: 2_000,
            lr: 0.1,
        },
    )?;

    let mse = mlp.evaluate_mse(&train)?;
    println!(
        "final_loss_from_fit={} train_mse={}",
        report.final_loss, mse
    );

    let mut scratch = mlp.scratch();
    let mut out = [0.0_f32; 1];
    for x in xs {
        mlp.predict_one_into(&x, &mut scratch, &mut out)?;
        println!("x={x:?} y={:?}", out[0]);
    }

    Ok(())
}
