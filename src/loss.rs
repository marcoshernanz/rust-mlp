//! Loss functions.
//!
//! These are small, allocation-free helpers intended to be used like:
//!
//! - run `model.forward(...)`
//! - compute `d_output` via a loss (e.g. `mse_backward`)
//! - run `model.backward(...)`
//! - update parameters with an optimizer

/// Mean squared error (MSE) loss.
///
/// Returns `0.5 * mean((pred - target)^2)`.
#[inline]
pub fn mse(pred: &[f32], target: &[f32]) -> f32 {
    debug_assert_eq!(pred.len(), target.len());

    if pred.is_empty() {
        return 0.0;
    }

    let inv_n = 1.0 / pred.len() as f32;
    let mut sum_sq = 0.0_f32;
    for i in 0..pred.len() {
        let diff = pred[i] - target[i];
        sum_sq = diff.mul_add(diff, sum_sq);
    }
    0.5 * sum_sq * inv_n
}

/// MSE loss + gradient w.r.t. `pred`.
///
/// Writes `d_pred = dL/d(pred)` into `d_pred` and returns the loss.
///
/// With `L = 0.5 * mean((pred - target)^2)`, the gradient is:
/// - `d_pred[i] = (pred[i] - target[i]) / N`
#[inline]
pub fn mse_backward(pred: &[f32], target: &[f32], d_pred: &mut [f32]) -> f32 {
    debug_assert_eq!(pred.len(), target.len());
    debug_assert_eq!(pred.len(), d_pred.len());

    if pred.is_empty() {
        return 0.0;
    }

    let inv_n = 1.0 / pred.len() as f32;
    let mut sum_sq = 0.0_f32;

    for i in 0..pred.len() {
        let diff = pred[i] - target[i];
        sum_sq = diff.mul_add(diff, sum_sq);
        d_pred[i] = diff * inv_n;
    }

    0.5 * sum_sq * inv_n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_is_zero_when_equal() {
        let pred = [1.0_f32, -2.0, 0.5];
        let target = pred;
        assert_eq!(mse(&pred, &target), 0.0);
    }

    #[test]
    fn mse_backward_matches_expected_gradient() {
        let pred = [1.0_f32, 3.0];
        let target = [2.0_f32, 1.0];
        let mut d_pred = [0.0_f32; 2];
        let loss = mse_backward(&pred, &target, &mut d_pred);

        // L = 0.5 * mean([(-1)^2, (2)^2]) = 0.5 * (1 + 4)/2 = 1.25
        assert!((loss - 1.25).abs() < 1e-6);
        // dL/dpred = (pred - target) / N
        assert!((d_pred[0] - (-0.5)).abs() < 1e-6);
        assert!((d_pred[1] - (1.0)).abs() < 1e-6);
    }
}
