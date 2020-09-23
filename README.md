# libprio-rs
[![Build Status]][actions] [![Latest Version]][crates.io]

[Build Status]: https://github.com/abetterinternet/libprio-rs/workflows/ci-build/badge.svg
[actions]: https://github.com/abetterinternet/libprio-rs/actions?query=branch%3Amain
[Latest Version]: https://img.shields.io/crates/v/prio.svg
[crates.io]: https://crates.io/crates/prio

Pure Rust implementation of [Prio](https://crypto.stanford.edu/prio/), a system for Private, Robust, and Scalable Computation of Aggregate Statistics.

## Releases

We use a GitHub Action to publish a crate named `prio` to [crates.io](https://crates.io). To cut a release and publish:

- Bump the version number in `Cargo.toml` to e.g. `1.2.3` and merge that change to `main`
- Tag that commit on main as `v1.2.3`, either in `git` or in [GitHub's releases UI](https://github.com/abetterinternet/libprio-rs/releases/new).
- Publish a release in [GitHub's releases UI](https://github.com/abetterinternet/libprio-rs/releases/new).

Publishing the release will automatically publish the updated `prio` crate to `crates.io`.
