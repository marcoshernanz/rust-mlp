# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- Batched training via GEMM (`forward_batch`/`backward_batch`) with an optional `matrixmultiply` backend.
- Allocation stability test for `fit` to ensure no per-step allocations.
- Production-ready metadata (license, docs.rs config) and a practical README.

### Changed
- Training (`fit`) uses the batched path for full-size batches when `batch_size > 1`.
- MSRV is now declared and enforced in CI.
- Added `Mlp::predict_into` (and `predict_one_into` as an alias) for shape-checked, non-allocating inference.

## [0.1.0] - 2026-02-12

Initial release.
