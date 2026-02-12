# Publishing instructions (rust-mlp)

This is a practical checklist to publish this crate to crates.io and make the GitHub repo look polished.

## 0) One-time prerequisites

- Create a crates.io account: https://crates.io
- Create an API token: https://crates.io/settings/tokens
- Log in locally:

```bash
cargo login <YOUR_CRATES_IO_TOKEN>
```

## 1) Final preflight (local)

From the repo root:

```bash
# Format
cargo fmt

# Lint (match CI)
cargo clippy --all-targets --all-features -- -D warnings

# Tests (match CI)
cargo test --all-targets --all-features

# Build release artifacts
cargo build --release
```

Optional (but recommended):

```bash
cargo doc --no-deps --all-features
```

## 2) Verify package metadata

Check `Cargo.toml`:

- `[package] name`, `version`, `edition`, `rust-version`
- `license`, `readme`, `repository`, `documentation`, `description`, `keywords`, `categories`

Make sure these files exist in the repo root:

- `README.md`
- `LICENSE`
- `CHANGELOG.md`

## 3) Update changelog for the release

Edit `CHANGELOG.md`:

- Move relevant entries from `[Unreleased]` into a new version section (or update the existing section).
- Add the release date.

## 4) Dry-run the publish

This verifies what will be uploaded and catches missing files early:

```bash
cargo publish --dry-run
```

If the dry run fails due to missing files or metadata, fix it and retry.

## 5) Publish to crates.io

```bash
cargo publish
```

If you use feature flags and want docs.rs to build with all features, ensure this is present in `Cargo.toml`:

```toml
[package.metadata.docs.rs]
all-features = true
```

## 6) Tag the release on GitHub

After `cargo publish` succeeds:

```bash
git tag v<version>
git push origin v<version>
```

Then create a GitHub Release for the tag and paste the relevant `CHANGELOG.md` entries.

## 7) Post-publish checks

- Confirm the crate page looks good: https://crates.io/crates/rust-mlp
- Confirm docs.rs built successfully: https://docs.rs/rust-mlp
- Confirm the README renders well on GitHub.
