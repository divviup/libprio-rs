// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

#![warn(missing_docs)]

//! Libprio-rs
//!
//! Implementation of the [Prio](https://crypto.stanford.edu/prio/) private data aggregation
//! protocol. For now we only support 0 / 1 vectors.

#[macro_use]
extern crate static_assertions;

pub mod benchmarked;
pub mod client;
pub mod encrypt;
pub mod fft;
pub mod field;
mod fp;
pub mod pcp;
mod polynomial;
mod prng;
pub mod server;
// Module test_vector depends on crate `rand` so we make it an optional feature
// to spare most clients the extra dependency.
#[cfg(any(feature = "test-vector", test))]
pub mod test_vector;
pub mod util;
pub mod vdaf;
