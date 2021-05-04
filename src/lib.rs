// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

#![warn(missing_docs)]

//! Libprio-rs
//!
//! Implementation of Prio: https://crypto.stanford.edu/prio/
//!
//! For now we only support 0 / 1 vectors.

pub mod benchmarked;
pub mod client;
pub mod encrypt;
pub mod fft;
pub mod field;
mod fp;
pub mod pcp;
mod polynomial;
mod prng;
pub mod proto;
pub mod server;
pub mod util;
