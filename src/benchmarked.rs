// SPDX-License-Identifier: MPL-2.0

#![doc(hidden)]

//! This module provides wrappers around internal components of this crate that we want to
//! benchmark, but which we don't want to expose in the public API.

#[cfg(feature = "prio2")]
use crate::client::Client;
use crate::fft::discrete_fourier_transform;
use crate::field::FftFriendlyFieldElement;
use crate::flp::gadgets::Mul;
use crate::flp::FlpError;
use crate::polynomial::{fft_get_roots, poly_fft, PolyFFTTempMemory};

/// Sets `outp` to the Discrete Fourier Transform (DFT) using an iterative FFT algorithm.
pub fn benchmarked_iterative_fft<F: FftFriendlyFieldElement>(outp: &mut [F], inp: &[F]) {
    discrete_fourier_transform(outp, inp, inp.len()).unwrap();
}

/// Sets `outp` to the Discrete Fourier Transform (DFT) using a recursive FFT algorithm.
pub fn benchmarked_recursive_fft<F: FftFriendlyFieldElement>(outp: &mut [F], inp: &[F]) {
    let roots_2n = fft_get_roots(inp.len(), false);
    let mut fft_memory = PolyFFTTempMemory::new(inp.len());
    poly_fft(outp, inp, &roots_2n, inp.len(), false, &mut fft_memory)
}

/// Sets `outp` to `inp[0] * inp[1]`, where `inp[0]` and `inp[1]` are polynomials. This function
/// uses FFT for multiplication.
pub fn benchmarked_gadget_mul_call_poly_fft<F: FftFriendlyFieldElement>(
    g: &mut Mul<F>,
    outp: &mut [F],
    inp: &[Vec<F>],
) -> Result<(), FlpError> {
    g.call_poly_fft(outp, inp)
}

/// Sets `outp` to `inp[0] * inp[1]`, where `inp[0]` and `inp[1]` are polynomials. This function
/// does the multiplication directly.
pub fn benchmarked_gadget_mul_call_poly_direct<F: FftFriendlyFieldElement>(
    g: &mut Mul<F>,
    outp: &mut [F],
    inp: &[Vec<F>],
) -> Result<(), FlpError> {
    g.call_poly_direct(outp, inp)
}

/// Returns a Prio v2 proof that `data` is a valid boolean vector.
#[cfg(feature = "prio2")]
pub fn benchmarked_v2_prove<F: FftFriendlyFieldElement>(
    data: &[F],
    client: &mut Client<F>,
) -> Vec<F> {
    client.gen_proof(data)
}
