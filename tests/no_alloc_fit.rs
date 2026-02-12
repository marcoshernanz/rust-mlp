use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rust_mlp::{Activation, Dataset, FitConfig, Loss, LrSchedule, MlpBuilder, Optimizer, Shuffle};

struct CountingAlloc {
    allocs: AtomicUsize,
    reallocs: AtomicUsize,
    deallocs: AtomicUsize,
    bytes: AtomicUsize,
}

impl CountingAlloc {
    const fn new() -> Self {
        Self {
            allocs: AtomicUsize::new(0),
            reallocs: AtomicUsize::new(0),
            deallocs: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.allocs.store(0, Ordering::Relaxed);
        self.reallocs.store(0, Ordering::Relaxed);
        self.deallocs.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> AllocSnapshot {
        AllocSnapshot {
            allocs: self.allocs.load(Ordering::Relaxed),
            reallocs: self.reallocs.load(Ordering::Relaxed),
            deallocs: self.deallocs.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
        }
    }

    fn alloc_events(&self) -> usize {
        self.allocs.load(Ordering::Relaxed) + self.reallocs.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AllocSnapshot {
    allocs: usize,
    reallocs: usize,
    deallocs: usize,
    bytes: usize,
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocs.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.allocs.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocs.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.reallocs.fetch_add(1, Ordering::Relaxed);
        // Approximate accounting: record the new size.
        self.bytes.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc::new();

fn make_dataset(len: usize, input_dim: usize, target_dim: usize) -> Dataset {
    let inputs = vec![0.1_f32; len * input_dim];
    let targets = vec![0.0_f32; len * target_dim];
    Dataset::from_flat(inputs, targets, input_dim, target_dim).unwrap()
}

#[test]
fn fit_does_not_allocate_per_step_for_batched_training() {
    if cfg!(feature = "matrixmultiply") {
        // The `matrixmultiply` backend may allocate internal scratch buffers.
        // This test focuses on the crate's own training loop behavior.
        return;
    }

    let input_dim = 32;
    let hidden = 64;
    let output_dim = 8;
    let batch_size = 16;

    let base = MlpBuilder::new(input_dim)
        .unwrap()
        .add_layer(hidden, Activation::Tanh)
        .unwrap()
        .add_layer(output_dim, Activation::Identity)
        .unwrap()
        .build_with_seed(0)
        .unwrap();

    let train_small = make_dataset(batch_size, input_dim, output_dim);
    let train_large = make_dataset(batch_size * 64, input_dim, output_dim);

    let cfg = FitConfig {
        epochs: 1,
        lr: 1e-2,
        batch_size,
        shuffle: Shuffle::None,
        lr_schedule: LrSchedule::Constant,
        optimizer: Optimizer::Sgd,
        weight_decay: 0.0,
        grad_clip_norm: None,
        loss: Loss::Mse,
        metrics: vec![],
    };

    let mut mlp_small = base.clone();
    ALLOC.reset();
    let before_small = ALLOC.snapshot();
    mlp_small.fit(&train_small, None, cfg.clone()).unwrap();
    let alloc_small = ALLOC.alloc_events();
    let after_small = ALLOC.snapshot();

    let mut mlp_large = base;
    ALLOC.reset();
    let before_large = ALLOC.snapshot();
    mlp_large.fit(&train_large, None, cfg).unwrap();
    let alloc_large = ALLOC.alloc_events();
    let after_large = ALLOC.snapshot();

    assert_eq!(
        alloc_small, alloc_large,
        "expected allocation event count to be independent of steps.\n\
small: before={before_small:?} after={after_small:?}\n\
large: before={before_large:?} after={after_large:?}"
    );
}
