# libprio-rs
[![Build Status]][actions] [![Latest Version]][crates.io] [![Docs badge]][docs.rs]


[Build Status]: https://github.com/abetterinternet/libprio-rs/workflows/ci-build/badge.svg
[actions]: https://github.com/abetterinternet/libprio-rs/actions?query=branch%3Amain
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

Crate `prio` also implements a [Verifiable Distributed Aggregation Function (VDAF)][vdaf] called
"prio3", implemented in the `vdaf` module, allowing Prio to be used in the
[Privacy Preserving Measurements][ppm] protocol. This support is still experimental, and is evolving
along with the PPM and VDAF specifications. Formal security analysis is also forthcoming. prio3
should not yet be used in production applications.

[enpa]: https://www.abetterinternet.org/post/prio-services-for-covid-en/
[enpa-whitepaper]: https://covid19-static.cdn-apple.com/applications/covid19/current/static/contact-tracing/pdf/ENPA_White_Paper.pdf
[prio-server]: https://github.com/abetterinternet/prio-server
[vdaf]: https://cjpatton.github.io/vdaf/draft-patton-cfrg-vdaf.html
[ppm]: https://github.com/abetterinternet/ppm-specification
