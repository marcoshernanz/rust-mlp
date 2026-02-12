# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- Batched training via GEMM (`forward_batch`/`backward_batch`) with an optional `matrixmultiply` backend.
- Allocation stability test for `fit` to ensure no per-step allocations.

### Changed
- Training (`fit`) uses the batched path for full-size batches when `batch_size > 1`.

## [0.1.0] - 2026-02-12

Initial release.
