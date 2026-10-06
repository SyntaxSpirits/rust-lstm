# Contributing

Thank you for helping to improve rust-lstm.

## Questions and bug reports

Open an issue at <https://github.com/SyntaxSpirits/rust-lstm/issues>. For bugs, include
the crate version, a minimal program that reproduces the problem and the output you
expected. Issues are usually answered within a week.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request focused on one change and describe why it is needed.
3. Make sure the checks that run in CI pass locally:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --all-features
   ```

4. Any new layer or change to a backward pass needs a case in
   `tests/gradient_check.rs`. Changes to LSTM or GRU numerics should also pass the
   PyTorch comparison described in `validation/README.md`.
5. Add an entry to `CHANGELOG.md` under the next version.

## Releases

Versions follow semantic versioning. Releases are tagged on GitHub, published to
crates.io and archived on Zenodo.

## Code of conduct

Be respectful and constructive. Harassment of any kind is not tolerated; report it to
alexandrkholodniak@gmail.com.
