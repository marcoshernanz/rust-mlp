//! High-level training and evaluation APIs.
//!
//! These methods validate dataset shapes and return `Result`, while internally using
//! allocation-free per-sample forward/backward passes.

use crate::{Activation, Dataset, Error, Layer, Loss, Metric, Mlp, Result, Sgd, Trainer, loss};

#[derive(Debug, Clone)]
/// Configuration for `Mlp::fit`.
pub struct FitConfig {
    pub epochs: usize,
    pub lr: f32,
    pub loss: Loss,
    pub metrics: Vec<Metric>,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            lr: 1e-2,
            loss: Loss::Mse,
            metrics: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
/// Output of a training run.
pub struct FitReport {
    pub epochs: Vec<EpochReport>,
}

#[derive(Debug, Clone)]
/// Report for a single epoch.
pub struct EpochReport {
    pub train: EvalReport,
    pub val: Option<EvalReport>,
}

#[derive(Debug, Clone)]
/// Output of `Mlp::evaluate`.
pub struct EvalReport {
    pub loss: f32,
    pub metrics: Vec<(Metric, f32)>,
}

impl EvalReport {
    fn new(loss: f32, metrics: Vec<(Metric, f32)>) -> Self {
        Self { loss, metrics }
    }
}

impl Mlp {
    /// Evaluate a dataset with a loss and optional metrics.
    pub fn evaluate(
        &self,
        data: &Dataset,
        loss_fn: Loss,
        metrics: &[Metric],
    ) -> Result<EvalReport> {
        validate_dataset_shapes(self, data)?;
        validate_loss_compat(self, loss_fn, data.target_dim())?;
        for &m in metrics {
            m.validate()?;
        }

        let mut scratch = self.scratch();
        let mut out_buf = vec![0.0_f32; self.output_dim()];

        let mut total_loss = 0.0_f32;
        let mut metric_acc = MetricsAccum::new(self.output_dim(), metrics)?;

        for idx in 0..data.len() {
            let x = data.input(idx);
            let t = data.target(idx);

            self.predict_one_into(x, &mut scratch, &mut out_buf)?;
            total_loss += loss_fn.forward(&out_buf, t);
            metric_acc.update(&out_buf, t)?;
        }

        let inv_n = 1.0 / data.len() as f32;
        Ok(EvalReport::new(
            total_loss * inv_n,
            metric_acc.finish(data.len()),
        ))
    }

    /// Train the model on a dataset.
    ///
    /// This is a "batteries included" API intended to be easy to use.
    /// Internally it still uses allocation-free forward/backward via scratch buffers.
    pub fn fit(
        &mut self,
        train: &Dataset,
        val: Option<&Dataset>,
        cfg: FitConfig,
    ) -> Result<FitReport> {
        if train.is_empty() {
            return Err(Error::InvalidData(
                "train dataset must not be empty".to_owned(),
            ));
        }
        validate_dataset_shapes(self, train)?;
        validate_loss_compat(self, cfg.loss, train.target_dim())?;
        for &m in &cfg.metrics {
            m.validate()?;
        }

        if let Some(val) = val {
            if val.is_empty() {
                return Err(Error::InvalidData(
                    "val dataset must not be empty".to_owned(),
                ));
            }
            validate_dataset_shapes(self, val)?;
            validate_loss_compat(self, cfg.loss, val.target_dim())?;
        }

        if cfg.epochs == 0 {
            return Err(Error::InvalidConfig("epochs must be > 0".to_owned()));
        }
        if !(cfg.lr.is_finite() && cfg.lr > 0.0) {
            return Err(Error::InvalidConfig("lr must be finite and > 0".to_owned()));
        }

        let opt = Sgd::new(cfg.lr)?;
        let mut trainer = Trainer::new(self);
        let mut reports = Vec::with_capacity(cfg.epochs);

        for _epoch in 0..cfg.epochs {
            let mut epoch_loss = 0.0_f32;
            let mut metric_acc = MetricsAccum::new(self.output_dim(), &cfg.metrics)?;

            for idx in 0..train.len() {
                let input = train.input(idx);
                let target = train.target(idx);

                self.forward(input, &mut trainer.scratch);
                let pred = trainer.scratch.output();

                let loss_val = cfg
                    .loss
                    .backward(pred, target, trainer.grads.d_output_mut());
                epoch_loss += loss_val;
                metric_acc.update(pred, target)?;

                self.backward(input, &trainer.scratch, &mut trainer.grads);
                opt.step(self, &trainer.grads);
            }

            let inv_n = 1.0 / train.len() as f32;
            let train_report = EvalReport::new(epoch_loss * inv_n, metric_acc.finish(train.len()));
            let val_report = match val {
                Some(v) => Some(self.evaluate(v, cfg.loss, &cfg.metrics)?),
                None => None,
            };

            reports.push(EpochReport {
                train: train_report,
                val: val_report,
            });
        }

        Ok(FitReport { epochs: reports })
    }

    /// Predict outputs for all inputs in `data`.
    ///
    /// Returns a flat buffer with shape `(len, output_dim)`.
    pub fn predict(&self, data: &Dataset) -> Result<Vec<f32>> {
        if data.is_empty() {
            return Err(Error::InvalidData("dataset must not be empty".to_owned()));
        }
        if data.input_dim() != self.input_dim() {
            return Err(Error::InvalidData(format!(
                "dataset input_dim {} does not match model input_dim {}",
                data.input_dim(),
                self.input_dim()
            )));
        }

        let mut scratch = self.scratch();
        let out_dim = self.output_dim();
        let mut preds = vec![0.0_f32; data.len() * out_dim];

        for idx in 0..data.len() {
            let input = data.input(idx);
            let y = self.forward(input, &mut scratch);
            let start = idx * out_dim;
            preds[start..start + out_dim].copy_from_slice(y);
        }

        Ok(preds)
    }

    /// Predict outputs for inputs (X).
    ///
    /// Returns a flat buffer with shape `(len, output_dim)`.
    pub fn predict_inputs(&self, inputs: &crate::Inputs) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            return Err(Error::InvalidData("inputs must not be empty".to_owned()));
        }
        if inputs.input_dim() != self.input_dim() {
            return Err(Error::InvalidData(format!(
                "inputs input_dim {} does not match model input_dim {}",
                inputs.input_dim(),
                self.input_dim()
            )));
        }

        let mut scratch = self.scratch();
        let out_dim = self.output_dim();
        let mut preds = vec![0.0_f32; inputs.len() * out_dim];

        for idx in 0..inputs.len() {
            let x = inputs.input(idx);
            let y = self.forward(x, &mut scratch);
            let start = idx * out_dim;
            preds[start..start + out_dim].copy_from_slice(y);
        }

        Ok(preds)
    }

    /// Evaluate mean MSE over a dataset.
    ///
    /// This is a convenience wrapper around `evaluate`.
    pub fn evaluate_mse(&self, data: &Dataset) -> Result<f32> {
        if data.is_empty() {
            return Err(Error::InvalidData("dataset must not be empty".to_owned()));
        }
        Ok(self.evaluate(data, Loss::Mse, &[])?.loss)
    }
}

fn validate_dataset_shapes(model: &Mlp, data: &Dataset) -> Result<()> {
    if data.input_dim() != model.input_dim() {
        return Err(Error::InvalidData(format!(
            "dataset input_dim {} does not match model input_dim {}",
            data.input_dim(),
            model.input_dim()
        )));
    }
    if data.target_dim() != model.output_dim() {
        return Err(Error::InvalidData(format!(
            "dataset target_dim {} does not match model output_dim {}",
            data.target_dim(),
            model.output_dim()
        )));
    }
    Ok(())
}

fn validate_loss_compat(model: &Mlp, loss_fn: Loss, target_dim: usize) -> Result<()> {
    loss_fn.validate()?;

    match loss_fn {
        Loss::Mse | Loss::Mae => Ok(()),
        Loss::BinaryCrossEntropyWithLogits => {
            if target_dim != 1 {
                return Err(Error::InvalidConfig(format!(
                    "BinaryCrossEntropyWithLogits requires output_dim == 1, got {target_dim}"
                )));
            }
            let last = last_layer_activation(model);
            if last != Activation::Identity {
                return Err(Error::InvalidConfig(
                    "BinaryCrossEntropyWithLogits expects raw logits; set the output layer activation to Identity"
                        .to_owned(),
                ));
            }
            Ok(())
        }
        Loss::SoftmaxCrossEntropy => {
            if target_dim < 2 {
                return Err(Error::InvalidConfig(format!(
                    "SoftmaxCrossEntropy requires output_dim >= 2, got {target_dim}"
                )));
            }
            let last = last_layer_activation(model);
            if last != Activation::Identity {
                return Err(Error::InvalidConfig(
                    "SoftmaxCrossEntropy expects raw logits; set the output layer activation to Identity".to_owned(),
                ));
            }
            Ok(())
        }
    }
}

fn last_layer_activation(model: &Mlp) -> Activation {
    // `Mlp` is guaranteed to have at least one layer when constructed via `MlpBuilder`.
    last_layer(model)
        .expect("mlp must have at least one layer")
        .activation()
}

fn last_layer(model: &Mlp) -> Option<&Layer> {
    // We intentionally keep `Mlp`'s internal layout private. This helper uses a
    // public accessor to inspect the last layer when validating logits-based losses.
    model.layer(model.num_layers().checked_sub(1)?)
}

struct MetricsAccum {
    output_dim: usize,
    metrics: Vec<Metric>,
    sums: Vec<f32>,
}

impl MetricsAccum {
    fn new(output_dim: usize, metrics: &[Metric]) -> Result<Self> {
        let mut ms = Vec::with_capacity(metrics.len());
        for &m in metrics {
            m.validate()?;
            ms.push(m);
        }
        Ok(Self {
            output_dim,
            metrics: ms,
            sums: vec![0.0; metrics.len()],
        })
    }

    fn update(&mut self, pred: &[f32], target: &[f32]) -> Result<()> {
        if self.metrics.is_empty() {
            return Ok(());
        }
        if pred.len() != target.len() {
            return Err(Error::InvalidData(format!(
                "pred/target length mismatch: {} vs {}",
                pred.len(),
                target.len()
            )));
        }
        if pred.len() != self.output_dim {
            return Err(Error::InvalidData(format!(
                "pred len {} does not match expected output_dim {}",
                pred.len(),
                self.output_dim
            )));
        }

        for (idx, &m) in self.metrics.iter().enumerate() {
            self.sums[idx] += metric_value(m, pred, target)?;
        }
        Ok(())
    }

    fn finish(self, n: usize) -> Vec<(Metric, f32)> {
        if self.metrics.is_empty() {
            return Vec::new();
        }

        let inv_n = 1.0 / n as f32;
        self.metrics
            .into_iter()
            .zip(self.sums)
            .map(|(m, s)| (m, s * inv_n))
            .collect()
    }
}

fn metric_value(metric: Metric, pred: &[f32], target: &[f32]) -> Result<f32> {
    match metric {
        Metric::Mse => Ok(loss::mse(pred, target)),
        Metric::Mae => Ok(loss::mae(pred, target)),
        Metric::Accuracy => Ok(accuracy(pred, target)?),
        Metric::TopKAccuracy { k } => Ok(topk_accuracy(pred, target, k)?),
    }
}

fn accuracy(pred: &[f32], target: &[f32]) -> Result<f32> {
    if pred.len() != target.len() {
        return Err(Error::InvalidData(format!(
            "pred/target length mismatch: {} vs {}",
            pred.len(),
            target.len()
        )));
    }
    if pred.is_empty() {
        return Ok(0.0);
    }

    if pred.len() == 1 {
        // Binary accuracy.
        let y = pred[0];
        let t = target[0];
        let pred_label = if y >= 0.5 { 1 } else { 0 };
        let true_label = if t >= 0.5 { 1 } else { 0 };
        Ok(if pred_label == true_label { 1.0 } else { 0.0 })
    } else {
        // Multiclass (argmax).
        let pred_idx = argmax(pred);
        let true_idx = argmax(target);
        Ok(if pred_idx == true_idx { 1.0 } else { 0.0 })
    }
}

fn topk_accuracy(pred: &[f32], target: &[f32], k: usize) -> Result<f32> {
    if pred.len() != target.len() {
        return Err(Error::InvalidData(format!(
            "pred/target length mismatch: {} vs {}",
            pred.len(),
            target.len()
        )));
    }
    if pred.len() <= 1 {
        return Err(Error::InvalidConfig(
            "TopKAccuracy requires output_dim > 1".to_owned(),
        ));
    }
    if k == 0 || k > pred.len() {
        return Err(Error::InvalidConfig(format!(
            "TopKAccuracy requires 1 <= k <= output_dim, got k={k} output_dim={}",
            pred.len()
        )));
    }

    let true_idx = argmax(target);

    // Find if true_idx is in top-k of pred without allocating:
    // Count how many logits are strictly greater than pred[true_idx].
    let true_score = pred[true_idx];
    let mut num_greater = 0_usize;
    for (i, &v) in pred.iter().enumerate() {
        if i != true_idx && v > true_score {
            num_greater += 1;
        }
    }
    Ok(if num_greater < k { 1.0 } else { 0.0 })
}

fn argmax(xs: &[f32]) -> usize {
    debug_assert!(!xs.is_empty());
    let mut best_idx = 0;
    let mut best_val = xs[0];
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use crate::{Activation, Dataset, Loss, Metric, MlpBuilder};

    #[test]
    fn evaluate_computes_accuracy_for_multiclass_one_hot() {
        // Make a tiny dataset where the model is forced to output logits we can control.
        // We'll build a 2 -> 3 identity-ish model.
        let mlp = MlpBuilder::new(2)
            .unwrap()
            .add_layer(3, Activation::Identity)
            .unwrap()
            .build_with_seed(0)
            .unwrap();

        // Create data: two samples, one-hot targets.
        let xs = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let ys = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let data = Dataset::from_rows(&xs, &ys).unwrap();

        // We cannot currently mutate parameters through the public API.
        // This test focuses on metric shape handling; we still should be able to run evaluate.
        let report = mlp
            .evaluate(&data, Loss::SoftmaxCrossEntropy, &[Metric::Accuracy])
            .unwrap();
        assert_eq!(report.metrics.len(), 1);
    }
}
