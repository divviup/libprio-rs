// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

#![warn(missing_docs)]

//! Libprio-rs
//!
//! Implementation of Prio: https://crypto.stanford.edu/prio/
//!
//! For now we only support 0 / 1 vectors.

pub mod client;
pub mod encrypt;
pub mod finite_field;
mod fp;
mod polynomial;
mod prng;
pub mod server;
pub mod util;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
