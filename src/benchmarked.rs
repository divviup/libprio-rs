// SPDX-License-Identifier: MPL-2.0

//! This package provides wrappers around internal components of this crate that we want to
//! benchmark, but which we don't want to expose in the public API.

use crate::fft::discrete_fourier_transform;
use crate::field::FieldElement;
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
