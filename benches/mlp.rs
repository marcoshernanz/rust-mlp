use criterion::{Criterion, black_box, criterion_group, criterion_main};

use rust_mlp::{Mlp, loss};

fn mlp_forward_bench(c: &mut Criterion) {
    let mlp = Mlp::new_with_seed(&[128, 256, 256, 10], 0);
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
    let mlp = Mlp::new_with_seed(&[128, 256, 256, 10], 0);
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

criterion_group!(benches, mlp_forward_bench, mlp_backward_bench);
criterion_main!(benches);
