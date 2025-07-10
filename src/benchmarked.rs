// SPDX-License-Identifier: MPL-2.0

#![doc(hidden)]

//! This module provides wrappers around internal components of this crate that we want to
//! benchmark, but which we don't want to expose in the public API.

use crate::field::NttFriendlyFieldElement;
use crate::flp::gadgets::Mul;
use crate::flp::FlpError;

/// Sets `outp` to `inp[0] * inp[1]`, where `inp[0]` and `inp[1]` are polynomials. This function
/// uses NTT for multiplication.
pub fn benchmarked_gadget_mul_call_poly_ntt<F: NttFriendlyFieldElement>(
    g: &mut Mul<F>,
    outp: &mut [F],
    inp: &[Vec<F>],
) -> Result<(), FlpError> {
    g.call_poly_ntt(outp, inp)
}

/// Sets `outp` to `inp[0] * inp[1]`, where `inp[0]` and `inp[1]` are polynomials. This function
/// does the multiplication directly.
pub fn benchmarked_gadget_mul_call_poly_direct<F: NttFriendlyFieldElement>(
    g: &mut Mul<F>,
    outp: &mut [F],
    inp: &[Vec<F>],
) -> Result<(), FlpError> {
    g.call_poly_direct(outp, inp)
}
