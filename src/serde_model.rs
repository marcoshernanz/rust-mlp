//! Model serialization/deserialization (feature: `serde`).
//!
//! This module defines a versioned, stable on-disk format for `Mlp`.
//!
//! Design notes:
//! - We do NOT directly serialize internal `Mlp`/`Layer` structs, to keep the
//!   file format stable even if internal representation changes.
//! - All deserialization validates dimensions, parameter lengths, and that
//!   all parameters are finite.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Activation, Error, Layer, Mlp, Result};

#[cfg(feature = "serde")]
use std::path::Path;

pub const MODEL_FORMAT_VERSION: u32 = 1;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SerializedMlp {
    pub format_version: u32,
    pub layers: Vec<SerializedLayer>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SerializedLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: SerializedActivation,
    /// Row-major (out_dim, in_dim).
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SerializedActivation {
    Tanh,
    Relu,
    LeakyRelu { alpha: f32 },
    Sigmoid,
    Identity,
}

impl From<Activation> for SerializedActivation {
    fn from(value: Activation) -> Self {
        match value {
            Activation::Tanh => SerializedActivation::Tanh,
            Activation::ReLU => SerializedActivation::Relu,
            Activation::LeakyReLU { alpha } => SerializedActivation::LeakyRelu { alpha },
            Activation::Sigmoid => SerializedActivation::Sigmoid,
            Activation::Identity => SerializedActivation::Identity,
        }
    }
}

impl SerializedActivation {
    fn into_activation(self) -> Activation {
        match self {
            SerializedActivation::Tanh => Activation::Tanh,
            SerializedActivation::Relu => Activation::ReLU,
            SerializedActivation::LeakyRelu { alpha } => Activation::LeakyReLU { alpha },
            SerializedActivation::Sigmoid => Activation::Sigmoid,
            SerializedActivation::Identity => Activation::Identity,
        }
    }
}

impl SerializedMlp {
    pub fn validate(&self) -> Result<()> {
        if self.format_version != MODEL_FORMAT_VERSION {
            return Err(Error::InvalidData(format!(
                "unsupported model format_version {}; expected {}",
                self.format_version, MODEL_FORMAT_VERSION
            )));
        }
        if self.layers.is_empty() {
            return Err(Error::InvalidData(
                "serialized model must have at least one layer".to_owned(),
            ));
        }

        for (i, layer) in self.layers.iter().enumerate() {
            layer.validate()?;

            if i > 0 {
                let prev_out = self.layers[i - 1].out_dim;
                if layer.in_dim != prev_out {
                    return Err(Error::InvalidData(format!(
                        "layer {i} in_dim {} does not match previous out_dim {}",
                        layer.in_dim, prev_out
                    )));
                }
            }
        }

        Ok(())
    }
}

impl SerializedLayer {
    fn validate(&self) -> Result<()> {
        if self.in_dim == 0 || self.out_dim == 0 {
            return Err(Error::InvalidData(format!(
                "layer dims must be > 0, got in_dim={} out_dim={}",
                self.in_dim, self.out_dim
            )));
        }

        let expected_w = self
            .in_dim
            .checked_mul(self.out_dim)
            .ok_or_else(|| Error::InvalidData("layer weight shape overflow".to_owned()))?;
        if self.weights.len() != expected_w {
            return Err(Error::InvalidData(format!(
                "weights length {} does not match out_dim * in_dim ({} * {})",
                self.weights.len(),
                self.out_dim,
                self.in_dim
            )));
        }
        if self.biases.len() != self.out_dim {
            return Err(Error::InvalidData(format!(
                "biases length {} does not match out_dim {}",
                self.biases.len(),
                self.out_dim
            )));
        }

        let act = self.activation.into_activation();
        act.validate()
            .map_err(|e| Error::InvalidData(format!("invalid activation: {e}")))?;

        if self.weights.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidData(
                "weights must contain only finite values".to_owned(),
            ));
        }
        if self.biases.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidData(
                "biases must contain only finite values".to_owned(),
            ));
        }

        Ok(())
    }
}

impl From<&Mlp> for SerializedMlp {
    fn from(model: &Mlp) -> Self {
        let mut layers = Vec::with_capacity(model.num_layers());
        for i in 0..model.num_layers() {
            let layer = model.layer(i).expect("layer idx must be valid");
            layers.push(SerializedLayer::from(layer));
        }
        Self {
            format_version: MODEL_FORMAT_VERSION,
            layers,
        }
    }
}

impl From<&Layer> for SerializedLayer {
    fn from(layer: &Layer) -> Self {
        Self {
            in_dim: layer.in_dim(),
            out_dim: layer.out_dim(),
            activation: SerializedActivation::from(layer.activation()),
            weights: layer.weights().to_vec(),
            biases: layer.biases().to_vec(),
        }
    }
}

impl TryFrom<SerializedMlp> for Mlp {
    type Error = Error;

    fn try_from(value: SerializedMlp) -> std::result::Result<Self, Self::Error> {
        value.validate()?;

        let mut layers = Vec::with_capacity(value.layers.len());
        for (i, layer) in value.layers.into_iter().enumerate() {
            let act = layer.activation.into_activation();

            // Layer::from_parts performs shape validation and finiteness checks.
            let l = Layer::from_parts(
                layer.in_dim,
                layer.out_dim,
                act,
                layer.weights,
                layer.biases,
            )
            .map_err(|e| Error::InvalidData(format!("layer {i} invalid: {e}")))?;
            layers.push(l);
        }

        Ok(Mlp::from_layers(layers))
    }
}

#[cfg(feature = "serde")]
impl Mlp {
    /// Serialize the model to a pretty-printed JSON string.
    pub fn to_json_string_pretty(&self) -> Result<String> {
        let ser = SerializedMlp::from(self);
        serde_json::to_string_pretty(&ser)
            .map_err(|e| Error::InvalidData(format!("failed to serialize model: {e}")))
    }

    /// Serialize the model to a compact JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        let ser = SerializedMlp::from(self);
        serde_json::to_string(&ser)
            .map_err(|e| Error::InvalidData(format!("failed to serialize model: {e}")))
    }

    /// Parse a model from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let ser: SerializedMlp = serde_json::from_str(s)
            .map_err(|e| Error::InvalidData(format!("failed to parse model json: {e}")))?;
        ser.try_into()
    }

    /// Save the model to a JSON file (pretty-printed).
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let s = self.to_json_string_pretty()?;
        let p = path.as_ref();
        std::fs::write(p, s)
            .map_err(|e| Error::InvalidData(format!("failed to write {}: {e}", p.display())))?;
        Ok(())
    }

    /// Load a model from a JSON file.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let s = std::fs::read_to_string(p)
            .map_err(|e| Error::InvalidData(format!("failed to read {}: {e}", p.display())))?;
        Self::from_json_str(&s)
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;

    #[test]
    fn golden_json_is_stable_and_roundtrips() {
        let l1 = Layer::from_parts(
            2,
            3,
            Activation::Tanh,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![0.1, 0.2, 0.3],
        )
        .unwrap();
        let l2 =
            Layer::from_parts(3, 1, Activation::Identity, vec![7.0, 8.0, 9.0], vec![0.4]).unwrap();

        let mlp = Mlp::from_layers(vec![l1, l2]);
        let json = mlp.to_json_string_pretty().unwrap();

        let golden = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/mlp_v1.json"
        ))
        .trim_end();
        assert_eq!(json, golden);

        // Round-trip via JSON.
        let loaded = Mlp::from_json_str(golden).unwrap();
        let json2 = loaded.to_json_string_pretty().unwrap();
        assert_eq!(json2, golden);
    }

    #[test]
    fn rejects_unknown_version() {
        let bad = r#"{"format_version":999,"layers":[]}"#;
        let err = Mlp::from_json_str(bad).unwrap_err();
        assert!(format!("{err}").contains("format_version"));
    }
}
