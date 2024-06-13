// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Primitives for the Prio2 client.

use crate::{
    codec::CodecError,
    field::FftFriendlyFieldElement,
    polynomial::{fft_get_roots, poly_fft, PolyFFTTempMemory},
    prng::{Prng, PrngError},
    vdaf::{
        xof::{Seed, SeedStreamAes128},
        VdafError,
    },
};

use std::convert::TryFrom;

/// Errors that might be emitted by the client.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ClientError {
    /// PRNG error
    #[error("prng error: {0}")]
    Prng(#[from] PrngError),
    /// VDAF error
    #[error("vdaf error: {0}")]
    Vdaf(#[from] VdafError),
    /// failure when calling getrandom().
    #[error("getrandom: {0}")]
    GetRandom(#[from] getrandom::Error),
}

/// Serialization errors
#[derive(Debug, thiserror::Error)]
pub enum SerializeError {
    /// Emitted by `unpack_proof[_mut]` if the serialized share+proof has the wrong length
    #[error("serialized input has wrong length")]
    UnpackInputSizeMismatch,
    /// Codec error.
    #[error(transparent)]
    Codec(#[from] CodecError),
}

#[derive(Debug)]
pub(crate) struct ClientMemory<F> {
    prng: Prng<F, SeedStreamAes128>,
    points_f: Vec<F>,
    points_g: Vec<F>,
    evals_f: Vec<F>,
    evals_g: Vec<F>,
    roots_2n: Vec<F>,
    roots_n_inverted: Vec<F>,
    fft_memory: PolyFFTTempMemory<F>,
    coeffs: Vec<F>,
}

impl<F: FftFriendlyFieldElement> ClientMemory<F> {
    pub(crate) fn new(dimension: usize) -> Result<Self, VdafError> {
        let n = (dimension + 1).next_power_of_two();
        if let Ok(size) = F::Integer::try_from(2 * n) {
            if size > F::generator_order() {
                return Err(VdafError::Uncategorized(
                    "input size exceeds field capacity".into(),
                ));
            }
        } else {
            return Err(VdafError::Uncategorized(
                "input size exceeds field capacity".into(),
            ));
        }

        Ok(Self {
            prng: Prng::from_prio2_seed(Seed::<32>::generate()?.as_ref()),
            points_f: vec![F::zero(); n],
            points_g: vec![F::zero(); n],
            evals_f: vec![F::zero(); 2 * n],
            evals_g: vec![F::zero(); 2 * n],
            roots_2n: fft_get_roots(2 * n, false),
            roots_n_inverted: fft_get_roots(n, true),
            fft_memory: PolyFFTTempMemory::new(2 * n),
            coeffs: vec![F::zero(); 2 * n],
        })
    }
}

impl<F: FftFriendlyFieldElement> ClientMemory<F> {
    pub(crate) fn prove_with<G>(&mut self, dimension: usize, init_function: G) -> Vec<F>
    where
        G: FnOnce(&mut [F]),
    {
        let mut proof = vec![F::zero(); proof_length(dimension)];
        // unpack one long vector to different subparts
        let unpacked = unpack_proof_mut(&mut proof, dimension).unwrap();
        // initialize the data part
        init_function(unpacked.data);
        // fill in the rest
        construct_proof(
            unpacked.data,
            dimension,
            unpacked.f0,
            unpacked.g0,
            unpacked.h0,
            unpacked.points_h_packed,
            self,
        );

        proof
    }
}

/// Returns the number of field elements in the proof for given dimension of
/// data elements
///
/// Proof is a vector, where the first `dimension` elements are the data
/// elements, the next 3 elements are the zero terms for polynomials f, g and h
/// and the remaining elements are non-zero points of h(x).
pub(crate) fn proof_length(dimension: usize) -> usize {
    // number of data items + number of zero terms + N
    dimension + 3 + (dimension + 1).next_power_of_two()
}

/// Unpacked proof with subcomponents
#[derive(Debug)]
pub(crate) struct UnpackedProof<'a, F: FftFriendlyFieldElement> {
    /// Data
    pub data: &'a [F],
    /// Zeroth coefficient of polynomial f
    pub f0: &'a F,
    /// Zeroth coefficient of polynomial g
    pub g0: &'a F,
    /// Zeroth coefficient of polynomial h
    pub h0: &'a F,
    /// Non-zero points of polynomial h
    pub points_h_packed: &'a [F],
}

/// Unpacked proof with mutable subcomponents
#[derive(Debug)]
pub(crate) struct UnpackedProofMut<'a, F: FftFriendlyFieldElement> {
    /// Data
    pub data: &'a mut [F],
    /// Zeroth coefficient of polynomial f
    pub f0: &'a mut F,
    /// Zeroth coefficient of polynomial g
    pub g0: &'a mut F,
    /// Zeroth coefficient of polynomial h
    pub h0: &'a mut F,
    /// Non-zero points of polynomial h
    pub points_h_packed: &'a mut [F],
}

/// Unpacks the proof vector into subcomponents
pub(crate) fn unpack_proof<F: FftFriendlyFieldElement>(
    proof: &[F],
    dimension: usize,
) -> Result<UnpackedProof<F>, SerializeError> {
    // check the proof length
    if proof.len() != proof_length(dimension) {
        return Err(SerializeError::UnpackInputSizeMismatch);
    }
    // split share into components
    let (data, rest) = proof.split_at(dimension);
    if let ([f0, g0, h0], points_h_packed) = rest.split_at(3) {
        Ok(UnpackedProof {
            data,
            f0,
            g0,
            h0,
            points_h_packed,
        })
    } else {
        Err(SerializeError::UnpackInputSizeMismatch)
    }
}

/// Unpacks a mutable proof vector into mutable subcomponents
pub(crate) fn unpack_proof_mut<F: FftFriendlyFieldElement>(
    proof: &mut [F],
    dimension: usize,
) -> Result<UnpackedProofMut<F>, SerializeError> {
    // check the share length
    if proof.len() != proof_length(dimension) {
        return Err(SerializeError::UnpackInputSizeMismatch);
    }
    // split share into components
    let (data, rest) = proof.split_at_mut(dimension);
    if let ([f0, g0, h0], points_h_packed) = rest.split_at_mut(3) {
        Ok(UnpackedProofMut {
            data,
            f0,
            g0,
            h0,
            points_h_packed,
        })
    } else {
        Err(SerializeError::UnpackInputSizeMismatch)
    }
}

/// Interpolate a polynomial at the nth roots of unity, and then evaluate it at the 2nth roots of
/// unity.
///
/// # Arguments
///
/// * `n` - The number of points to interpolate a polynomial through.
/// * `points_in` - The values that the polynomial must take on when evaluated at the nth roots of
///   unity. This must have length n.
/// * `evals_out` - The values that the polynomial takes on when evaluated at the 2nth roots of
///   unity. This must have length 2 * n.
/// * `roots_n_inverted` - Precomputed inverses of the nth roots of unity.
/// * `roots_2n` - Precomputed 2nth roots of unity.
/// * `fft_memory` - Scratch space for the FFT algorithm.
/// * `coeffs` - Scratch space. This must have length 2 * n.
fn interpolate_and_evaluate_at_2n<F: FftFriendlyFieldElement>(
    n: usize,
    points_in: &[F],
    evals_out: &mut [F],
    roots_n_inverted: &[F],
    roots_2n: &[F],
    fft_memory: &mut PolyFFTTempMemory<F>,
    coeffs: &mut [F],
) {
    // interpolate through roots of unity
    poly_fft(coeffs, points_in, roots_n_inverted, n, true, fft_memory);
    // evaluate at 2N roots of unity
    poly_fft(evals_out, coeffs, roots_2n, 2 * n, false, fft_memory);
}

/// Proof construction
///
/// Based on Theorem 2.3.3 from Henry Corrigan-Gibbs' dissertation
/// This constructs the output \pi by doing the necessesary calculations
fn construct_proof<F: FftFriendlyFieldElement>(
    data: &[F],
    dimension: usize,
    f0: &mut F,
    g0: &mut F,
    h0: &mut F,
    points_h_packed: &mut [F],
    mem: &mut ClientMemory<F>,
) {
    let n = (dimension + 1).next_power_of_two();

    // set zero terms to random
    *f0 = mem.prng.get();
    *g0 = mem.prng.get();
    mem.points_f[0] = *f0;
    mem.points_g[0] = *g0;

    // set zero term for the proof polynomial
    *h0 = *f0 * *g0;

    // set f_i = data_(i - 1)
    // set g_i = f_i - 1
    for ((f_coeff, g_coeff), data_val) in mem.points_f[1..1 + dimension]
        .iter_mut()
        .zip(mem.points_g[1..1 + dimension].iter_mut())
        .zip(data[..dimension].iter())
    {
        *f_coeff = *data_val;
        *g_coeff = *data_val - F::one();
    }

    // interpolate and evaluate at roots of unity
    interpolate_and_evaluate_at_2n(
        n,
        &mem.points_f,
        &mut mem.evals_f,
        &mem.roots_n_inverted,
        &mem.roots_2n,
        &mut mem.fft_memory,
        &mut mem.coeffs,
    );
    interpolate_and_evaluate_at_2n(
        n,
        &mem.points_g,
        &mut mem.evals_g,
        &mem.roots_n_inverted,
        &mem.roots_2n,
        &mut mem.fft_memory,
        &mut mem.coeffs,
    );

    // calculate the proof polynomial as evals_f(r) * evals_g(r)
    // only add non-zero points
    let mut j: usize = 0;
    let mut i: usize = 1;
    while i < 2 * n {
        points_h_packed[j] = mem.evals_f[i] * mem.evals_g[i];
        j += 1;
        i += 2;
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use crate::{
        field::{Field64, FieldPrio2},
        vdaf::prio2::client::{proof_length, unpack_proof, unpack_proof_mut, SerializeError},
    };

    #[test]
    fn test_unpack_share_mut() {
        let dim = 15;
        let len = proof_length(dim);

        let mut share = vec![FieldPrio2::from(0); len];
        let unpacked = unpack_proof_mut(&mut share, dim).unwrap();
        *unpacked.f0 = FieldPrio2::from(12);
        assert_eq!(share[dim], 12);

        let mut short_share = vec![FieldPrio2::from(0); len - 1];
        assert_matches!(
            unpack_proof_mut(&mut short_share, dim),
            Err(SerializeError::UnpackInputSizeMismatch)
        );
    }

    #[test]
    fn test_unpack_share() {
        let dim = 15;
        let len = proof_length(dim);

        let share = vec![Field64::from(0); len];
        unpack_proof(&share, dim).unwrap();

        let short_share = vec![Field64::from(0); len - 1];
        assert_matches!(
            unpack_proof(&short_share, dim),
            Err(SerializeError::UnpackInputSizeMismatch)
        );
    }
}
