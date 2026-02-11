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
    /// Returns the underlying contiguous buffer.
    ///
    /// Shape: `(len * input_dim,)`.
    pub fn as_flat(&self) -> &[f32] {
        &self.inputs
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
    /// Returns the underlying contiguous inputs buffer.
    ///
    /// Shape: `(len * input_dim,)`.
    pub fn inputs_flat(&self) -> &[f32] {
        self.inputs.as_flat()
    }

    #[inline]
    /// Returns the underlying contiguous targets buffer.
    ///
    /// Shape: `(len * target_dim,)`.
    pub fn targets_flat(&self) -> &[f32] {
        &self.targets
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

    /// Returns a contiguous batch view.
    ///
    /// Panics if the requested range is out of bounds.
    pub fn batch(&self, start: usize, len: usize) -> Batch<'_> {
        assert!(len > 0, "batch len must be > 0");
        assert!(start < self.len(), "batch start out of bounds");
        assert!(
            start + len <= self.len(),
            "batch range out of bounds: start={start} len={len} dataset_len={}",
            self.len()
        );

        let in_dim = self.input_dim();
        let t_dim = self.target_dim();
        let x0 = start * in_dim;
        let x1 = (start + len) * in_dim;
        let y0 = start * t_dim;
        let y1 = (start + len) * t_dim;
        Batch {
            inputs: &self.inputs_flat()[x0..x1],
            targets: &self.targets_flat()[y0..y1],
            len,
            input_dim: in_dim,
            target_dim: t_dim,
        }
    }

    /// Iterate contiguous batch views.
    ///
    /// Panics if `batch_size == 0`.
    pub fn batches(&self, batch_size: usize) -> Batches<'_> {
        assert!(batch_size > 0, "batch_size must be > 0");
        Batches {
            data: self,
            batch_size,
            pos: 0,
        }
    }
}

/// A contiguous dataset batch view.
///
/// `inputs` and `targets` are flat row-major buffers.
#[derive(Debug, Clone, Copy)]
pub struct Batch<'a> {
    inputs: &'a [f32],
    targets: &'a [f32],
    len: usize,
    input_dim: usize,
    target_dim: usize,
}

impl<'a> Batch<'a> {
    #[inline]
    /// Returns the number of samples in this batch.
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    /// Returns true if this batch is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    /// Returns the per-sample input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[inline]
    /// Returns the per-sample target dimension.
    pub fn target_dim(&self) -> usize {
        self.target_dim
    }

    #[inline]
    /// Returns the contiguous flat inputs buffer.
    ///
    /// Shape: `(len * input_dim,)`.
    pub fn inputs_flat(&self) -> &'a [f32] {
        self.inputs
    }

    #[inline]
    /// Returns the contiguous flat targets buffer.
    ///
    /// Shape: `(len * target_dim,)`.
    pub fn targets_flat(&self) -> &'a [f32] {
        self.targets
    }

    #[inline]
    /// Returns the `idx`-th input row (shape: `(input_dim,)`).
    ///
    /// Panics if `idx >= len`.
    pub fn input(&self, idx: usize) -> &'a [f32] {
        let start = idx * self.input_dim;
        &self.inputs[start..start + self.input_dim]
    }

    #[inline]
    /// Returns the `idx`-th target row (shape: `(target_dim,)`).
    ///
    /// Panics if `idx >= len`.
    pub fn target(&self, idx: usize) -> &'a [f32] {
        let start = idx * self.target_dim;
        &self.targets[start..start + self.target_dim]
    }
}

/// Iterator over contiguous batches.
#[derive(Debug, Clone)]
pub struct Batches<'a> {
    data: &'a Dataset,
    batch_size: usize,
    pos: usize,
}

impl<'a> Iterator for Batches<'a> {
    type Item = Batch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        let start = self.pos;
        let end = (start + self.batch_size).min(self.data.len());
        self.pos = end;
        Some(self.data.batch(start, end - start))
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

    #[test]
    fn batches_cover_all_samples_in_order() {
        // len=5, input_dim=2, target_dim=1
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let data = Dataset::from_flat(x, y, 2, 1).unwrap();

        let batches: Vec<_> = data.batches(2).collect();
        assert_eq!(batches.len(), 3);

        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[0].inputs_flat(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(batches[0].targets_flat(), &[10.0, 11.0]);

        assert_eq!(batches[1].len(), 2);
        assert_eq!(batches[1].inputs_flat(), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(batches[1].targets_flat(), &[12.0, 13.0]);

        assert_eq!(batches[2].len(), 1);
        assert_eq!(batches[2].inputs_flat(), &[8.0, 9.0]);
        assert_eq!(batches[2].targets_flat(), &[14.0]);
    }
}
