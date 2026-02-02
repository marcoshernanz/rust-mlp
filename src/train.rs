use crate::{Dataset, Error, Mlp, Result, Sgd, Trainer, loss};

#[derive(Debug, Clone, Copy)]
pub struct FitConfig {
    pub epochs: usize,
    pub lr: f32,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            lr: 1e-2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FitReport {
    pub final_loss: f32,
}

impl Mlp {
    /// Train the model on a dataset using MSE + SGD.
    ///
    /// This is a "batteries included" API intended to be easy to use.
    /// Internally it still uses allocation-free forward/backward via scratch buffers.
    pub fn fit(&mut self, train: &Dataset, cfg: FitConfig) -> Result<FitReport> {
        if train.is_empty() {
            return Err(Error::InvalidData(
                "train dataset must not be empty".to_owned(),
            ));
        }
        if train.input_dim() != self.input_dim() {
            return Err(Error::InvalidData(format!(
                "train input_dim {} does not match model input_dim {}",
                train.input_dim(),
                self.input_dim()
            )));
        }
        if train.target_dim() != self.output_dim() {
            return Err(Error::InvalidData(format!(
                "train target_dim {} does not match model output_dim {}",
                train.target_dim(),
                self.output_dim()
            )));
        }
        if cfg.epochs == 0 {
            return Err(Error::InvalidConfig("epochs must be > 0".to_owned()));
        }
        if !(cfg.lr.is_finite() && cfg.lr > 0.0) {
            return Err(Error::InvalidConfig("lr must be finite and > 0".to_owned()));
        }

        let opt = Sgd::new(cfg.lr)?;
        let mut trainer = Trainer::new(self);
        let mut epoch_loss = 0.0_f32;

        for _ in 0..cfg.epochs {
            epoch_loss = 0.0;
            for idx in 0..train.len() {
                let input = train.input(idx);
                let target = train.target(idx);

                self.forward(input, &mut trainer.scratch);
                let pred = trainer.scratch.output();

                let loss = loss::mse_backward(pred, target, trainer.grads.d_output_mut());
                epoch_loss += loss;

                self.backward(input, &trainer.scratch, &mut trainer.grads);
                opt.step(self, &trainer.grads);
            }
        }

        Ok(FitReport {
            final_loss: epoch_loss / train.len() as f32,
        })
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
    pub fn evaluate_mse(&self, data: &Dataset) -> Result<f32> {
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
        if data.target_dim() != self.output_dim() {
            return Err(Error::InvalidData(format!(
                "dataset target_dim {} does not match model output_dim {}",
                data.target_dim(),
                self.output_dim()
            )));
        }

        let mut scratch = self.scratch();
        let mut total = 0.0_f32;
        for idx in 0..data.len() {
            let input = data.input(idx);
            let target = data.target(idx);
            self.forward(input, &mut scratch);
            total += loss::mse(scratch.output(), target);
        }
        Ok(total / data.len() as f32)
    }
}
