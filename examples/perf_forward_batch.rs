use std::time::Instant;

use rust_mlp::{Activation, MlpBuilder};

fn parse_usize(args: &[String], key: &str, default: usize) -> usize {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == key {
            return args[i + 1]
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("failed to parse {key} as usize"));
        }
        i += 1;
    }
    default
}

fn parse_u64(args: &[String], key: &str, default: u64) -> u64 {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == key {
            return args[i + 1]
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("failed to parse {key} as u64"));
        }
        i += 1;
    }
    default
}

fn main() -> rust_mlp::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let batch_size = parse_usize(&args, "--batch-size", 128);
    let iters = parse_usize(&args, "--iters", 2_000);
    let warmup = parse_usize(&args, "--warmup", 200);
    let in_dim = parse_usize(&args, "--in-dim", 128);
    let hidden = parse_usize(&args, "--hidden", 256);
    let layers = parse_usize(&args, "--layers", 2);
    let out_dim = parse_usize(&args, "--out-dim", 10);
    let seed = parse_u64(&args, "--seed", 0);

    if batch_size == 0 || iters == 0 {
        panic!("batch_size and iters must be > 0");
    }
    if in_dim == 0 || hidden == 0 || out_dim == 0 {
        panic!("in_dim/hidden/out_dim must be > 0");
    }

    let mut builder = MlpBuilder::new(in_dim)?;
    for _ in 0..layers {
        builder = builder.add_layer(hidden, Activation::Tanh)?;
    }
    builder = builder.add_layer(out_dim, Activation::Identity)?;
    let mlp = builder.build_with_seed(seed)?;

    let backend = if cfg!(feature = "matrixmultiply") {
        "matrixmultiply"
    } else {
        "naive"
    };

    let mut scratch = mlp.scratch_batch(batch_size);

    // Deterministic, non-constant inputs.
    let mut inputs = vec![0.0_f32; batch_size * in_dim];
    for (i, v) in inputs.iter_mut().enumerate() {
        // Keep the values small to avoid extreme activations.
        *v = ((i % 997) as f32) * 1e-3;
    }

    // Warmup.
    for _ in 0..warmup {
        let out = mlp.forward_batch(std::hint::black_box(&inputs), &mut scratch);
        std::hint::black_box(out[0]);
    }

    // Timed run.
    let start = Instant::now();
    let mut checksum = 0.0_f32;
    for _ in 0..iters {
        let out = mlp.forward_batch(std::hint::black_box(&inputs), &mut scratch);
        checksum += out[0];
    }
    let elapsed = start.elapsed();
    std::hint::black_box(checksum);

    let elapsed_s = elapsed.as_secs_f64();
    let total_samples = (iters as f64) * (batch_size as f64);
    let samples_per_s = total_samples / elapsed_s;
    let iters_per_s = (iters as f64) / elapsed_s;

    println!(
        "rust-mlp perf_forward_batch backend={backend} batch_size={batch_size} iters={iters} warmup={warmup} in_dim={in_dim} hidden={hidden} layers={layers} out_dim={out_dim} elapsed_s={elapsed_s:.6} iters_per_s={iters_per_s:.2} samples_per_s={samples_per_s:.2} checksum={checksum}",
    );

    Ok(())
}
