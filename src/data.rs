//! Contiguous dataset helpers.
//!
//! The training loop operates on slices to avoid per-step allocations. `Inputs` and
//! `Dataset` provide validated, row-major storage for feature/target matrices.

use crate::{Error, Result};

/// A collection of input samples (X).
///
/// Stored as a contiguous buffer with row-major layout:
/// - `inputs.len() == len * input_dim`
#[derive(Debug, Clone)]
pub struct Inputs {
    inputs: Vec<f32>,
    len: usize,
    input_dim: usize,
}

impl Inputs {
    /// Build inputs from a flat buffer with shape `(len, input_dim)`.
    pub fn from_flat(inputs: Vec<f32>, input_dim: usize) -> Result<Self> {
        if input_dim == 0 {
            return Err(Error::InvalidData("input_dim must be > 0".to_owned()));
        }
        if !inputs.len().is_multiple_of(input_dim) {
            return Err(Error::InvalidData(format!(
                "inputs length {} is not divisible by input_dim {}",
                inputs.len(),
                input_dim
            )));
        }

        let len = inputs.len() / input_dim;

        Ok(Self {
            inputs,
            len,
            input_dim,
        })
    }

    /// Build inputs from per-sample rows.
    ///
    /// This is a convenience constructor (it copies into contiguous storage).
    pub fn from_rows(inputs: &[Vec<f32>]) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::InvalidData("inputs must not be empty".to_owned()));
        }

        let input_dim = inputs[0].len();
        if input_dim == 0 {
            return Err(Error::InvalidData("input_dim must be > 0".to_owned()));
        }

        for (i, row) in inputs.iter().enumerate() {
            if row.len() != input_dim {
                return Err(Error::InvalidData(format!(
                    "input row {i} has len {}, expected {input_dim}",
                    row.len()
                )));
            }
        }

        let len = inputs.len();
        let mut inputs_flat = Vec::with_capacity(len * input_dim);
        for row in inputs {
            inputs_flat.extend_from_slice(row);
        }

        Ok(Self {
            inputs: inputs_flat,
            len,
            input_dim,
        })
    }

    #[inline]
    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    /// Returns true if there are no samples.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    /// Returns the per-sample input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[inline]
    /// Returns the `idx`-th input row (shape: `(input_dim,)`).
    ///
    /// Panics if `idx >= len`.
    pub fn input(&self, idx: usize) -> &[f32] {
        let start = idx * self.input_dim;
        &self.inputs[start..start + self.input_dim]
    }
}

/// A supervised dataset: inputs (X) and targets (Y).
///
/// Stored as contiguous buffers with row-major layout:
/// - `inputs.len() == len * input_dim`
/// - `targets.len() == len * target_dim`
#[derive(Debug, Clone)]
pub struct Dataset {
    inputs: Inputs,
    targets: Vec<f32>,
    target_dim: usize,
}

impl Dataset {
    /// Build a dataset from flat buffers.
    ///
    /// `inputs` is `(len, input_dim)` and `targets` is `(len, target_dim)`.
    pub fn from_flat(
        inputs: Vec<f32>,
        targets: Vec<f32>,
        input_dim: usize,
        target_dim: usize,
    ) -> Result<Self> {
        let inputs = Inputs::from_flat(inputs, input_dim)?;
        if target_dim == 0 {
            return Err(Error::InvalidData("target_dim must be > 0".to_owned()));
        }

        if targets.len() != inputs.len() * target_dim {
            return Err(Error::InvalidData(format!(
                "targets length {} does not match len * target_dim ({} * {})",
                targets.len(),
                inputs.len(),
                target_dim
            )));
        }

        Ok(Self {
            inputs,
            targets,
            target_dim,
        })
    }

    /// Build a dataset from per-sample rows.
    ///
    /// This is a convenience constructor (it copies into contiguous storage).
    pub fn from_rows(inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> Result<Self> {
        if inputs.len() != targets.len() {
            return Err(Error::InvalidData(format!(
                "inputs/targets length mismatch: {} vs {}",
                inputs.len(),
                targets.len()
            )));
        }

        let inputs = Inputs::from_rows(inputs)?;
        let target_dim = targets.first().map(|t| t.len()).unwrap_or(0);
        if target_dim == 0 {
            return Err(Error::InvalidData("target_dim must be > 0".to_owned()));
        }
        for (i, row) in targets.iter().enumerate() {
            if row.len() != target_dim {
                return Err(Error::InvalidData(format!(
                    "target row {i} has len {}, expected {target_dim}",
                    row.len()
                )));
            }
        }

        let len = inputs.len();
        let mut targets_flat = Vec::with_capacity(len * target_dim);
        for row in targets {
            targets_flat.extend_from_slice(row);
        }

        Ok(Self {
            inputs,
            targets: targets_flat,
            target_dim,
        })
    }

    #[inline]
    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    #[inline]
    /// Returns true if there are no samples.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    #[inline]
    /// Returns the per-sample input dimension.
    pub fn input_dim(&self) -> usize {
        self.inputs.input_dim()
    }

    #[inline]
    /// Returns the per-sample target dimension.
    pub fn target_dim(&self) -> usize {
        self.target_dim
    }

    #[inline]
    /// Returns a view of the inputs (X).
    pub fn inputs(&self) -> &Inputs {
        &self.inputs
    }

    #[inline]
    /// Returns the `idx`-th input row (shape: `(input_dim,)`).
    ///
    /// Panics if `idx >= len`.
    pub fn input(&self, idx: usize) -> &[f32] {
        self.inputs.input(idx)
    }

    #[inline]
    /// Returns the `idx`-th target row (shape: `(target_dim,)`).
    ///
    /// Panics if `idx >= len`.
    pub fn target(&self, idx: usize) -> &[f32] {
        let start = idx * self.target_dim;
        &self.targets[start..start + self.target_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_from_flat_validates_shapes() {
        let ok = Dataset::from_flat(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0], 2, 1);
        assert!(ok.is_ok());

        let err = Dataset::from_flat(vec![0.0, 1.0, 2.0], vec![0.0], 2, 1);
        assert!(err.is_err());
    }
}
