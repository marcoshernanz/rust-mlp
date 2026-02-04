//! Model builder.
//!
//! `MlpBuilder` is the recommended way to define a model.
//!
//! It makes model structure explicit (layer sizes + activations) and chooses a
//! reasonable default weight initializer for each activation:
//!
//! - `tanh` / `sigmoid` / `identity`: Xavier/Glorot
//! - `relu` / `leaky relu`: He/Kaiming
//!
//! The resulting `Mlp` still supports the low-level, allocation-free hot path:
//! reuse `Scratch` / `Gradients` for per-sample forward/backward.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{Activation, Error, Init, Layer, Mlp, Result};

#[derive(Debug, Clone, Copy)]
struct LayerSpec {
    out_dim: usize,
    activation: Activation,
}

#[derive(Debug, Clone)]
/// Builder for an `Mlp`.
///
/// Example:
///
/// ```rust
/// use rust_mlp::{Activation, MlpBuilder};
///
/// # fn main() -> rust_mlp::Result<()> {
/// let mlp = MlpBuilder::new(2)?
///     .add_layer(8, Activation::ReLU)?
///     .add_layer(1, Activation::Sigmoid)?
///     .build_with_seed(0)?;
/// # Ok(())
/// # }
/// ```
pub struct MlpBuilder {
    input_dim: usize,
    layers: Vec<LayerSpec>,
}

impl MlpBuilder {
    /// Start building an MLP that accepts inputs of length `input_dim`.
    pub fn new(input_dim: usize) -> Result<Self> {
        if input_dim == 0 {
            return Err(Error::InvalidConfig("input_dim must be > 0".to_owned()));
        }
        Ok(Self {
            input_dim,
            layers: Vec::new(),
        })
    }

    /// Convenience constructor from a sizes list + activations.
    ///
    /// `sizes` includes input and output dimensions, so its length must be at least 2.
    /// `activations` must have length `sizes.len() - 1`.
    pub fn from_sizes(sizes: &[usize], activations: &[Activation]) -> Result<Self> {
        if sizes.len() < 2 {
            return Err(Error::InvalidConfig(
                "sizes must include input and output dims".to_owned(),
            ));
        }
        if sizes.contains(&0) {
            return Err(Error::InvalidConfig(
                "all layer sizes must be > 0".to_owned(),
            ));
        }
        if activations.len() != sizes.len() - 1 {
            return Err(Error::InvalidConfig(format!(
                "activations length {} does not match sizes.len() - 1 ({})",
                activations.len(),
                sizes.len() - 1
            )));
        }

        let mut b = Self::new(sizes[0])?;
        for (out_dim, &act) in sizes[1..].iter().zip(activations) {
            b = b.add_layer(*out_dim, act)?;
        }
        Ok(b)
    }

    /// Add a dense layer.
    ///
    /// The layer will have `out_dim` outputs and uses `activation`.
    pub fn add_layer(mut self, out_dim: usize, activation: Activation) -> Result<Self> {
        if out_dim == 0 {
            return Err(Error::InvalidConfig("layer out_dim must be > 0".to_owned()));
        }
        activation.validate()?;

        self.layers.push(LayerSpec {
            out_dim,
            activation,
        });
        Ok(self)
    }

    /// Build using a deterministic seed.
    pub fn build_with_seed(self, seed: u64) -> Result<Mlp> {
        let mut rng = StdRng::seed_from_u64(seed);
        self.build_with_rng(&mut rng)
    }

    /// Build using the provided RNG.
    pub fn build_with_rng<R: Rng + ?Sized>(self, rng: &mut R) -> Result<Mlp> {
        if self.layers.is_empty() {
            return Err(Error::InvalidConfig(
                "mlp must have at least one layer".to_owned(),
            ));
        }

        let mut layers = Vec::with_capacity(self.layers.len());
        let mut in_dim = self.input_dim;
        for spec in self.layers {
            let init = default_init_for_activation(spec.activation);
            let layer = Layer::new_with_rng(in_dim, spec.out_dim, init, spec.activation, rng)?;
            layers.push(layer);
            in_dim = spec.out_dim;
        }

        Ok(Mlp::from_layers(layers))
    }
}

#[inline]
fn default_init_for_activation(act: Activation) -> Init {
    match act {
        Activation::Tanh | Activation::Sigmoid | Activation::Identity => Init::Xavier,
        Activation::ReLU | Activation::LeakyReLU { .. } => Init::He,
    }
}
