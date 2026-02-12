use criterion::{Criterion, black_box, criterion_group, criterion_main};

use rust_mlp::{Activation, MlpBuilder, loss};

fn mlp_forward_bench(c: &mut Criterion) {
    let mlp = MlpBuilder::new(128)
        .unwrap()
        .add_layer(256, Activation::Tanh)
        .unwrap()
        .add_layer(256, Activation::Tanh)
        .unwrap()
        .add_layer(10, Activation::Identity)
        .unwrap()
        .build_with_seed(0)
        .unwrap();
    let mut scratch = mlp.scratch();
    let input = vec![0.1_f32; mlp.input_dim()];

    c.bench_function("mlp_forward_128_256_256_10", |b| {
        b.iter(|| {
            let out = mlp.forward(black_box(&input), &mut scratch);
            black_box(out);
        })
    });
}

fn mlp_backward_bench(c: &mut Criterion) {
    let mlp = MlpBuilder::new(128)
        .unwrap()
        .add_layer(256, Activation::Tanh)
        .unwrap()
        .add_layer(256, Activation::Tanh)
        .unwrap()
        .add_layer(10, Activation::Identity)
        .unwrap()
        .build_with_seed(0)
        .unwrap();
    let mut scratch = mlp.scratch();
    let mut grads = mlp.gradients();
    let input = vec![0.1_f32; mlp.input_dim()];
    let target = vec![0.0_f32; mlp.output_dim()];

    mlp.forward(&input, &mut scratch);
    loss::mse_backward(scratch.output(), &target, grads.d_output_mut());

    c.bench_function("mlp_backward_128_256_256_10", |b| {
        b.iter(|| {
            let d_input = mlp.backward(black_box(&input), black_box(&scratch), &mut grads);
            black_box(d_input);
        })
    });
}

fn mlp_forward_batch_bench(c: &mut Criterion) {
    let mlp = MlpBuilder::new(128)
        .unwrap()
        .add_layer(256, Activation::Tanh)
        .unwrap()
        .add_layer(256, Activation::Tanh)
        .unwrap()
        .add_layer(10, Activation::Identity)
        .unwrap()
        .build_with_seed(0)
        .unwrap();

    for &bs in &[8_usize, 32, 128] {
        let mut scratch = mlp.scratch_batch(bs);
        let inputs = vec![0.1_f32; bs * mlp.input_dim()];

        c.bench_function(&format!("mlp_forward_batch_{bs}_128_256_256_10"), |b| {
            b.iter(|| {
                let out = mlp.forward_batch(black_box(&inputs), &mut scratch);
                black_box(out);
            })
        });
    }
}

fn mlp_backward_batch_bench(c: &mut Criterion) {
    let mlp = MlpBuilder::new(128)
        .unwrap()
        .add_layer(256, Activation::Tanh)
        .unwrap()
        .add_layer(256, Activation::Tanh)
        .unwrap()
        .add_layer(10, Activation::Identity)
        .unwrap()
        .build_with_seed(0)
        .unwrap();

    for &bs in &[8_usize, 32, 128] {
        let mut scratch = mlp.scratch_batch(bs);
        let mut backprop_scratch = mlp.backprop_scratch_batch(bs);
        let mut grads = mlp.gradients();
        let inputs = vec![0.1_f32; bs * mlp.input_dim()];

        // Compute activations once.
        mlp.forward_batch(&inputs, &mut scratch);

        // Use a fixed upstream gradient.
        let d_outputs = vec![0.01_f32; bs * mlp.output_dim()];

        c.bench_function(&format!("mlp_backward_batch_{bs}_128_256_256_10"), |b| {
            b.iter(|| {
                mlp.backward_batch(
                    black_box(&inputs),
                    black_box(&scratch),
                    black_box(&d_outputs),
                    &mut grads,
                    &mut backprop_scratch,
                );
                black_box(&grads);
            })
        });
    }
}

criterion_group!(
    benches,
    mlp_forward_bench,
    mlp_backward_bench,
    mlp_forward_batch_bench,
    mlp_backward_batch_bench
);
criterion_main!(benches);
