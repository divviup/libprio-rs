// SPDX-License-Identifier: MPL-2.0

//! This module provides wrappers around internal components of this crate that we want to
//! benchmark, but which we don't want to expose in the public API.

use crate::client::Client;
use crate::fft::discrete_fourier_transform;
use crate::field::FieldElement;
use crate::flp::gadgets::Mul;
use crate::flp::FlpError;
use crate::polynomial::{poly_fft, PolyAuxMemory};

/// Sets `outp` to the Discrete Fourier Transform (DFT) using an iterative FFT algorithm.
pub fn benchmarked_iterative_fft<F: FieldElement>(outp: &mut [F], inp: &[F]) {
    discrete_fourier_transform(outp, inp, inp.len()).unwrap();
}

/// Sets `outp` to the Discrete Fourier Transform (DFT) using a recursive FFT algorithm.
pub fn benchmarked_recursive_fft<F: FieldElement>(outp: &mut [F], inp: &[F]) {
    let mut mem = PolyAuxMemory::new(inp.len() / 2);
    poly_fft(
        outp,
        inp,
        &mem.roots_2n,
        inp.len(),
        false,
        &mut mem.fft_memory,
    )
}

/// Sets `outp` to `inp[0] * inp[1]`, where `inp[0]` and `inp[1]` are polynomials. This function
/// uses FFT for multiplication.
pub fn benchmarked_gadget_mul_call_poly_fft<F: FieldElement>(
    g: &mut Mul<F>,
    outp: &mut [F],
    inp: &[Vec<F>],
) -> Result<(), FlpError> {
    g.call_poly_fft(outp, inp)
}

/// Sets `outp` to `inp[0] * inp[1]`, where `inp[0]` and `inp[1]` are polynomials. This function
/// does the multiplication directly.
pub fn benchmarked_gadget_mul_call_poly_direct<F: FieldElement>(
    g: &mut Mul<F>,
    outp: &mut [F],
    inp: &[Vec<F>],
) -> Result<(), FlpError> {
    g.call_poly_direct(outp, inp)
}

/// Returns a Prio v2 proof that `data` is a valid boolean vector.
pub fn benchmarked_v2_prove<F: FieldElement>(data: &[F], client: &mut Client<F>) -> Vec<F> {
    let copy_data = |share_data: &mut [F]| {
        share_data[..].clone_from_slice(data);
    };
    client.prove_with(copy_data)
}
