// Copyright (c) 2025 ISRG
// SPDX-License-Identifier: MPL-2.0 AND Apache-2.0
//
// Portions of this file are derived from the num-bigint crate
// (https://docs.rs/num-bigint/0.4.6/)
// Copyright 2013-2014 The Rust Project Developers
// Licensed under the Apache 2.0 license
//
// This file contains code covered by the following copyright and permission notice
// and has been modified by ISRG and collaborators.
//
// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Helper functions to randomly generate big integers, sampled from uniform distributions.
//!
//! These were vendored from num-bigint. Unused functionality has been deleted, trait
//! implementations have been transformed into free functions, use of the private
//! `biguint_from_vec()` function has been replaced, use of the `SampleBorrow` trait has been
//! removed, method names and signatures have been updated to align with rand 0.9, and documentation
//! has been updated.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{ToPrimitive, Zero};
use rand::distr::uniform::Error;
use rand::Rng;

/// Use [`Rng::fill`] to generate random bits.
///
/// If `rem` is greater than zero, than only the lowest `rem` bits of the last u32 are filled with
/// random data.
fn random_bits<R>(rng: &mut R, data: &mut [u32], rem: u64)
where
    R: Rng + ?Sized,
{
    // `fill` is faster than many `random::<u32>` calls
    rng.fill(data);
    if rem > 0 {
        let last = data.len() - 1;
        data[last] >>= 32 - rem;
    }
}

/// Uniformly generate a random [`BigUint`] in the range \[0, 2^`bits`).
fn random_biguint<R>(rng: &mut R, bit_size: u64) -> BigUint
where
    R: Rng + ?Sized,
{
    let (digits, rem) = bit_size.div_rem(&32);
    let len = (digits + (rem > 0) as u64)
        .to_usize()
        .expect("capacity overflow");
    let mut data = vec![0u32; len];
    random_bits(rng, &mut data, rem);
    BigUint::new(data)
}

/// Uniformly generate a random [`BigUint`] in the range \[0, `bound`).
fn random_biguint_below<R>(rng: &mut R, bound: &BigUint) -> BigUint
where
    R: Rng + ?Sized,
{
    assert!(!bound.is_zero());
    let bits = bound.bits();
    loop {
        let n = random_biguint(rng, bits);
        if n < *bound {
            return n;
        }
    }
}

/// Uniform distribution producing [`BigUint`].
pub(super) struct UniformBigUint {
    base: BigUint,
    len: BigUint,
}

impl UniformBigUint {
    pub(super) fn new(low: &BigUint, high: &BigUint) -> Result<Self, Error> {
        if low >= high {
            return Err(Error::EmptyRange);
        }
        Ok(UniformBigUint {
            len: high - low,
            base: low.clone(),
        })
    }

    pub(super) fn new_inclusive(low: &BigUint, high: &BigUint) -> Result<Self, Error> {
        if low > high {
            return Err(Error::EmptyRange);
        }
        Self::new(low, &(high + 1u32))
    }

    pub(super) fn sample<R>(&self, rng: &mut R) -> BigUint
    where
        R: Rng + ?Sized,
    {
        &self.base + random_biguint_below(rng, &self.len)
    }
}
