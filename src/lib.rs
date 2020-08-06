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
mod polynomial;
mod prng;
pub mod server;
pub mod util;
pub mod c_vec;
pub mod crypto_io;
pub mod client_io;
pub mod server_io;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
