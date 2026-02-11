use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rust_mlp::{Activation, Dataset, FitConfig, Loss, Metric, MlpBuilder};

fn main() -> rust_mlp::Result<()> {
    // Tiny synthetic 3-class dataset in 2D.
    // Each class is a Gaussian-ish blob around a different center.
    let mut rng = StdRng::seed_from_u64(0);

    let centers = [[-1.0_f32, -1.0], [1.0, -1.0], [0.0, 1.0]];
    let n_per_class = 128;
    let mut xs = Vec::with_capacity(3 * n_per_class);
    let mut ys = Vec::with_capacity(3 * n_per_class);

    for (class, center) in centers.iter().enumerate() {
        for _ in 0..n_per_class {
            // Uniform noise is good enough for a learning example.
            let x0 = center[0] + rng.gen_range(-0.3..0.3);
            let x1 = center[1] + rng.gen_range(-0.3..0.3);
            xs.push(vec![x0, x1]);

            let mut one_hot = vec![0.0_f32; 3];
            one_hot[class] = 1.0;
            ys.push(one_hot);
        }
    }

    let train = Dataset::from_rows(&xs, &ys)?;

    // Output activation is Identity because SoftmaxCrossEntropy expects logits.
    let mut mlp = MlpBuilder::new(2)?
        .add_layer(16, Activation::ReLU)?
        .add_layer(3, Activation::Identity)?
        .build_with_seed(0)?;

    let report = mlp.fit(
        &train,
        None,
        FitConfig {
            epochs: 200,
            lr: 0.05,
            batch_size: 32,
            shuffle: rust_mlp::Shuffle::Seeded(0),
            loss: Loss::SoftmaxCrossEntropy,
            metrics: vec![Metric::Accuracy, Metric::TopKAccuracy { k: 2 }],
        },
    )?;

    let last = report.epochs.last().unwrap();
    println!(
        "train_loss={} metrics={:?}",
        last.train.loss, last.train.metrics
    );

    let eval = mlp.evaluate(&train, Loss::SoftmaxCrossEntropy, &[Metric::Accuracy])?;
    println!("evaluate: loss={} metrics={:?}", eval.loss, eval.metrics);

    Ok(())
}
