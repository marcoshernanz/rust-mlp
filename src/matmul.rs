//! Small GEMM wrapper used by batched training.
//!
//! This module provides a single abstraction over matrix multiplication:
//! - default: a simple, safe triple-loop implementation
//! - optional: a faster backend via the `matrixmultiply` feature

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    rsb: usize,
    csb: usize,
    beta: f32,
    c: &mut [f32],
    rsc: usize,
    csc: usize,
) {
    debug_assert!(m > 0 && n > 0 && k > 0);
    debug_assert!(rsa > 0 || m <= 1);
    debug_assert!(csa > 0 || k <= 1);
    debug_assert!(rsb > 0 || k <= 1);
    debug_assert!(csb > 0 || n <= 1);
    debug_assert!(rsc > 0 || m <= 1);
    debug_assert!(csc > 0 || n <= 1);

    // Bounds are validated by callers in performance-sensitive code.
    // Keep this function minimal and inlineable.

    #[cfg(feature = "matrixmultiply")]
    {
        let (m, n, k) = (m, n, k);
        let (rsa, csa, rsb, csb, rsc, csc) = (rsa, csa, rsb, csb, rsc, csc);

        // matrixmultiply supports arbitrary strides.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa as isize,
                csa as isize,
                b.as_ptr(),
                rsb as isize,
                csb as isize,
                beta,
                c.as_mut_ptr(),
                rsc as isize,
                csc as isize,
            );
        }
        // Use an explicit block + `else` to keep the non-feature path unreachable.
    }

    #[cfg(not(feature = "matrixmultiply"))]
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            let a0 = i * rsa;
            let b0 = j * csb;

            for p in 0..k {
                let av = a[a0 + p * csa];
                let bv = b[p * rsb + b0];
                acc = av.mul_add(bv, acc);
            }

            let idx = i * rsc + j * csc;
            c[idx] = alpha * acc + beta * c[idx];
        }
    }
}
