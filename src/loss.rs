//! Loss functions.
//!
//! These are small, allocation-free helpers intended to be used like:
//!
//! - run `model.forward(...)`
//! - compute `d_output` via a loss (e.g. `mse_backward`)
//! - run `model.backward(...)`
//! - update parameters with an optimizer

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Supported loss functions.
pub enum Loss {
    /// Mean squared error.
    Mse,
    /// Mean absolute error.
    Mae,
    /// Binary cross-entropy with logits.
    ///
    /// This expects raw logits as predictions and targets in `[0, 1]`.
    /// In most cases you should use an `Identity` activation on the output layer.
    BinaryCrossEntropyWithLogits,
    /// Softmax cross-entropy.
    ///
    /// This expects raw logits as predictions and a one-hot target vector.
    /// In most cases you should use an `Identity` activation on the output layer.
    SoftmaxCrossEntropy,
}

impl Loss {
    /// Validate a loss configuration.
    pub fn validate(self) -> crate::Result<()> {
        // No parameters today.
        Ok(())
    }

    /// Compute a loss value.
    ///
    /// Shape contract: `pred.len() == target.len()`.
    #[inline]
    pub fn forward(self, pred: &[f32], target: &[f32]) -> f32 {
        match self {
            Loss::Mse => mse(pred, target),
            Loss::Mae => mae(pred, target),
            Loss::BinaryCrossEntropyWithLogits => bce_with_logits(pred, target),
            Loss::SoftmaxCrossEntropy => softmax_cross_entropy(pred, target),
        }
    }

    /// Compute loss + gradient w.r.t `pred`.
    ///
    /// Writes `d_pred = dL/d(pred)` into `d_pred` and returns the loss.
    ///
    /// Shape contract:
    /// - `pred.len() == target.len()`
    /// - `pred.len() == d_pred.len()`
    #[inline]
    pub fn backward(self, pred: &[f32], target: &[f32], d_pred: &mut [f32]) -> f32 {
        match self {
            Loss::Mse => mse_backward(pred, target, d_pred),
            Loss::Mae => mae_backward(pred, target, d_pred),
            Loss::BinaryCrossEntropyWithLogits => bce_with_logits_backward(pred, target, d_pred),
            Loss::SoftmaxCrossEntropy => softmax_cross_entropy_backward(pred, target, d_pred),
        }
    }
}

/// Mean squared error (MSE) loss.
///
/// Returns `0.5 * mean((pred - target)^2)`.
#[inline]
pub fn mse(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(
        pred.len(),
        target.len(),
        "pred len {} does not match target len {}",
        pred.len(),
        target.len()
    );

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
    assert_eq!(
        pred.len(),
        target.len(),
        "pred len {} does not match target len {}",
        pred.len(),
        target.len()
    );
    assert_eq!(
        pred.len(),
        d_pred.len(),
        "pred len {} does not match d_pred len {}",
        pred.len(),
        d_pred.len()
    );

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

/// Mean absolute error (MAE) loss.
///
/// Returns `mean(|pred - target|)`.
#[inline]
pub fn mae(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(
        pred.len(),
        target.len(),
        "pred len {} does not match target len {}",
        pred.len(),
        target.len()
    );

    if pred.is_empty() {
        return 0.0;
    }

    let inv_n = 1.0 / pred.len() as f32;
    let mut sum = 0.0_f32;
    for i in 0..pred.len() {
        sum += (pred[i] - target[i]).abs();
    }
    sum * inv_n
}

/// MAE loss + gradient w.r.t `pred`.
///
/// Gradient is a subgradient at `pred == target`.
#[inline]
pub fn mae_backward(pred: &[f32], target: &[f32], d_pred: &mut [f32]) -> f32 {
    assert_eq!(
        pred.len(),
        target.len(),
        "pred len {} does not match target len {}",
        pred.len(),
        target.len()
    );
    assert_eq!(
        pred.len(),
        d_pred.len(),
        "pred len {} does not match d_pred len {}",
        pred.len(),
        d_pred.len()
    );

    if pred.is_empty() {
        return 0.0;
    }

    let inv_n = 1.0 / pred.len() as f32;
    let mut sum = 0.0_f32;
    for i in 0..pred.len() {
        let diff = pred[i] - target[i];
        sum += diff.abs();
        d_pred[i] = if diff > 0.0 {
            inv_n
        } else if diff < 0.0 {
            -inv_n
        } else {
            0.0
        };
    }
    sum * inv_n
}

/// Binary cross-entropy loss with logits.
///
/// Per element (with `t` in [0,1]):
///
/// - `L = max(x, 0) - x * t + ln(1 + exp(-|x|))`
///
/// This is numerically stable for large |x|.
#[inline]
pub fn bce_with_logits(logits: &[f32], target: &[f32]) -> f32 {
    assert_eq!(
        logits.len(),
        target.len(),
        "pred len {} does not match target len {}",
        logits.len(),
        target.len()
    );

    if logits.is_empty() {
        return 0.0;
    }

    let inv_n = 1.0 / logits.len() as f32;
    let mut sum = 0.0_f32;
    for i in 0..logits.len() {
        let x = logits[i];
        let t = target[i];
        let abs_x = x.abs();
        let loss = x.max(0.0) - x * t + (1.0 + (-abs_x).exp()).ln();
        sum += loss;
    }
    sum * inv_n
}

/// BCE-with-logits loss + gradient w.r.t logits.
///
/// Gradient: `dL/dx = (sigmoid(x) - t) / N`.
#[inline]
pub fn bce_with_logits_backward(logits: &[f32], target: &[f32], d_logits: &mut [f32]) -> f32 {
    assert_eq!(
        logits.len(),
        target.len(),
        "pred len {} does not match target len {}",
        logits.len(),
        target.len()
    );
    assert_eq!(
        logits.len(),
        d_logits.len(),
        "pred len {} does not match d_pred len {}",
        logits.len(),
        d_logits.len()
    );

    if logits.is_empty() {
        return 0.0;
    }

    let inv_n = 1.0 / logits.len() as f32;
    let mut sum = 0.0_f32;

    for i in 0..logits.len() {
        let x = logits[i];
        let t = target[i];
        let abs_x = x.abs();
        let loss = x.max(0.0) - x * t + (1.0 + (-abs_x).exp()).ln();
        sum += loss;

        let s = sigmoid(x);
        d_logits[i] = (s - t) * inv_n;
    }

    sum * inv_n
}

/// Softmax cross-entropy over a single sample.
///
/// `logits` is a length-K vector. `target` is a one-hot length-K vector.
#[inline]
pub fn softmax_cross_entropy(logits: &[f32], target: &[f32]) -> f32 {
    assert_eq!(
        logits.len(),
        target.len(),
        "pred len {} does not match target len {}",
        logits.len(),
        target.len()
    );
    assert!(
        !logits.is_empty(),
        "softmax_cross_entropy requires at least 1 class"
    );

    let (log_sum_exp, _max) = log_sum_exp_and_max(logits);

    // Cross entropy: -sum_i t_i * log softmax_i
    // log softmax_i = logits[i] - log_sum_exp
    let mut sum = 0.0_f32;
    for i in 0..logits.len() {
        let t = target[i];
        if t != 0.0 {
            sum -= t * (logits[i] - log_sum_exp);
        }
    }

    // Mean over classes (matches the crate's "mean over pred.len()" convention).
    sum / logits.len() as f32
}

/// Softmax cross-entropy + gradient w.r.t logits.
///
/// Writes `d_logits = (softmax(logits) - target) / K`.
///
/// This function is allocation-free: it computes softmax into `d_logits` and then
/// turns it into a gradient in place.
#[inline]
pub fn softmax_cross_entropy_backward(logits: &[f32], target: &[f32], d_logits: &mut [f32]) -> f32 {
    assert_eq!(
        logits.len(),
        target.len(),
        "pred len {} does not match target len {}",
        logits.len(),
        target.len()
    );
    assert_eq!(
        logits.len(),
        d_logits.len(),
        "pred len {} does not match d_pred len {}",
        logits.len(),
        d_logits.len()
    );
    assert!(
        !logits.is_empty(),
        "softmax_cross_entropy_backward requires at least 1 class"
    );

    let k = logits.len();
    let inv_k = 1.0 / k as f32;

    let (log_sum_exp, max_logit) = log_sum_exp_and_max(logits);

    // Softmax into d_logits.
    for i in 0..k {
        d_logits[i] = (logits[i] - max_logit).exp();
    }
    let mut sum_exp = 0.0_f32;
    for &v in d_logits.iter() {
        sum_exp += v;
    }
    let inv_sum = 1.0 / sum_exp;
    for v in d_logits.iter_mut() {
        *v *= inv_sum;
    }

    // Loss.
    let mut loss = 0.0_f32;
    for i in 0..k {
        let t = target[i];
        if t != 0.0 {
            loss -= t * (logits[i] - log_sum_exp);
        }
    }
    loss *= inv_k;

    // Gradient: (softmax - target) / K.
    for i in 0..k {
        d_logits[i] = (d_logits[i] - target[i]) * inv_k;
    }

    loss
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Stable sigmoid (duplicated here to keep loss module self-contained).
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[inline]
fn log_sum_exp_and_max(xs: &[f32]) -> (f32, f32) {
    let mut max_x = xs[0];
    for &x in xs.iter().skip(1) {
        if x > max_x {
            max_x = x;
        }
    }
    let mut sum_exp = 0.0_f32;
    for &x in xs {
        sum_exp += (x - max_x).exp();
    }
    (max_x + sum_exp.ln(), max_x)
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

    #[test]
    fn bce_with_logits_is_reasonable_for_extreme_logits() {
        let logits = [100.0_f32, -100.0];
        let target = [1.0_f32, 0.0];
        let loss = bce_with_logits(&logits, &target);
        assert!(loss.is_finite());
        assert!(loss < 1e-3);
    }

    #[test]
    fn bce_with_logits_backward_matches_sigmoid_minus_target() {
        let logits = [0.0_f32];
        let target = [1.0_f32];
        let mut d = [0.0_f32];
        let loss = bce_with_logits_backward(&logits, &target, &mut d);
        assert!((loss - std::f32::consts::LN_2).abs() < 1e-5);
        // sigmoid(0) - 1 = -0.5
        assert!((d[0] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn softmax_cross_entropy_prefers_correct_class() {
        let logits_good = [5.0_f32, 0.0, -1.0];
        let logits_bad = [-1.0_f32, 0.0, 5.0];
        let target = [1.0_f32, 0.0, 0.0];
        let loss_good = softmax_cross_entropy(&logits_good, &target);
        let loss_bad = softmax_cross_entropy(&logits_bad, &target);
        assert!(loss_good < loss_bad);
    }
}
