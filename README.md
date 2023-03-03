# libprio-rs
[![Build Status]][actions] [![Latest Version]][crates.io] [![Docs badge]][docs.rs]

[Build Status]: https://github.com/divviup/libprio-rs/workflows/ci-build/badge.svg
[actions]: https://github.com/divviup/libprio-rs/actions?query=branch%3Amain
[Latest Version]: https://img.shields.io/crates/v/prio.svg
[crates.io]: https://crates.io/crates/prio
[Docs badge]: https://img.shields.io/badge/docs.rs-rustdoc-green
[docs.rs]: https://docs.rs/prio/

Pure Rust implementation of [Prio](https://crypto.stanford.edu/prio/), a system for Private, Robust,
and Scalable Computation of Aggregate Statistics.

## Exposure Notifications Private Analytics

This crate is used in the [Exposure Notifications Private Analytics][enpa] system. This is supported
by the interfaces in modules `server` and `client` and is referred to in various places as Prio v2.
See [`prio-server`][prio-server] or the [ENPA whitepaper][enpa-whitepaper] for more details.

## Verifiable Distributed Aggregation Function (EXPERIMENTAL)

Crate `prio` also implements a [Verifiable Distributed Aggregation Function
(VDAF)][vdaf] called "Prio3", implemented in the `vdaf` module, allowing Prio to
be used in the [Distributed Aggregation Protocol][dap] protocol being developed
in the PPM working group at the IETF. This support is still experimental, and is
evolving along with the DAP and VDAF specifications. Formal security analysis is
also forthcoming. Prio3 should not yet be used in production applications.

### Draft versions and release branches

The `main` branch is under continuous development and will usually be partway between VDAF drafts.
libprio uses stable release branches to maintain implementations of different VDAF draft versions.
Crate `prio` version `x.y.z` is released from a corresponding `release/x.y` branch. We try to
maintain [Rust SemVer][semver] compatibility, meaning that API breaks only happen on minor version
increases (e.g., 0.10 to 0.11).

| Git branch | Draft version | Conforms to specification? | Status |
| ---------- | ------------- | --------------------- | ------ |
| `release/0.8` | [`draft-irtf-cfrg-vdaf-01`][vdaf-01] | Yes | Supported |
| `release/0.9` | [`draft-irtf-cfrg-vdaf-03`][vdaf-03] | Yes | Unmaintained as of September 22, 2022 |
| `release/0.10` | [`draft-irtf-cfrg-vdaf-03`][vdaf-03] | Yes | Supported |
| `release/0.11` | [`draft-irtf-cfrg-vdaf-04`][vdaf-04] | Yes | Unmaintained |
| `main` | draft-irtf-cfrg-vdaf-05 (forthcoming) | No | Supported, unstable |

[vdaf-01]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/01/
[vdaf-03]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/
[vdaf-04]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/04/
[enpa]: https://www.abetterinternet.org/post/prio-services-for-covid-en/
[enpa-whitepaper]: https://covid19-static.cdn-apple.com/applications/covid19/current/static/contact-tracing/pdf/ENPA_White_Paper.pdf
[prio-server]: https://github.com/divviup/prio-server
[vdaf]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/
[dap]: https://datatracker.ietf.org/doc/draft-ietf-ppm-dap/
[semver]: https://doc.rust-lang.org/cargo/reference/semver.html

## Cargo Features

This crate defines the following feature flags:

|Name|Default feature?|Description|
|---|---|---|
|`crypto-dependencies`|Yes|Enables dependencies on various RustCrypto crates, and uses them to implement `PrgSha3` to support VDAFs.|
|`experimental`|No|Certain experimental APIs are guarded by this feature. They may undergo breaking changes in future patch releases, as an exception to semantic versioning.|
|`multithreaded`|No|Enables certain Prio3 VDAF implementations that use `rayon` for parallelization of gadget evaluations.|
|`prio2`|No|Enables the Prio v2 API, and a VDAF based on the Prio2 system.|
|`test-util`|No|For internal use only, to support the test suite and test vectors.|
