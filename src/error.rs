//! Error and `Result` types.
//!
//! This crate uses a split error-handling policy:
//! - Configuration/data validation at the API boundary returns `Result`.
//! - Low-level hot-path methods (e.g. per-sample forward/backward) panic on misuse
//!   (shape mismatches) via `assert!` / `assert_eq!`.

use std::fmt;

#[derive(Debug, Clone)]
/// Errors returned by fallible constructors and high-level APIs.
pub enum Error {
    /// The provided dataset/inputs/targets are invalid.
    InvalidData(String),
    /// The provided configuration is invalid (e.g. zero-sized layer, non-finite lr).
    InvalidConfig(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidData(msg) => write!(f, "invalid data: {msg}"),
            Error::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
