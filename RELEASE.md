# Release checklist

This is a lightweight checklist for publishing `rust-mlp`.

## Before publishing

- Update `CHANGELOG.md`:
  - Move entries from `[Unreleased]` into a new version section.
  - Add the release date.
- Ensure `Cargo.toml` is correct:
  - `version`
  - `license`, `repository`, `documentation`, `readme`, `description`
  - `rust-version` (MSRV)
- Run local quality gates (match CI):

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

## Sanity checks

- Build release artifacts:

```bash
cargo build --release
```

- (Optional) Benchmarks:

```bash
cargo bench
cargo bench --features matrixmultiply
```

## Publish

- Dry run:

```bash
cargo publish --dry-run
```

- Publish:

```bash
cargo publish
```

## After publishing

- Create a git tag and GitHub release (example for v0.1.1):

```bash
git tag v0.1.1
git push origin v0.1.1
```

- Verify docs.rs finished building: https://docs.rs/rust-mlp
