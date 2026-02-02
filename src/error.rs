use std::fmt;

#[derive(Debug, Clone)]
pub enum Error {
    InvalidData(String),
    InvalidConfig(String),
    InvalidShape(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidData(msg) => write!(f, "invalid data: {msg}"),
            Error::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Error::InvalidShape(msg) => write!(f, "invalid shape: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
