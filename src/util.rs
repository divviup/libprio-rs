// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Utility functions for handling Prio stuff.

use crate::field::{FieldElement, FieldError};

/// Serialization errors
#[derive(Debug, thiserror::Error)]
pub enum SerializeError {
    /// Emitted by `deserialize()` if the last chunk of input is not long enough to encode an
    /// element of the field.
    #[error("last chunk of bytes is incomplete")]
    IncompleteChunk,
    /// Finite field operation error.
    #[error("finite field operation error")]
    Field(#[from] FieldError),
}

/// Returns the number of field elements in the proof for given dimension of
/// data elements
///
/// Proof is a vector, where the first `dimension` elements are the data
/// elements, the next 3 elements are the zero terms for polynomials f, g and h
/// and the remaining elements are non-zero points of h(x).
pub fn proof_length(dimension: usize) -> usize {
    // number of data items + number of zero terms + N
    dimension + 3 + (dimension + 1).next_power_of_two()
}

/// Convenience function for initializing fixed sized vectors of Field elements.
pub fn vector_with_length<F: FieldElement>(len: usize) -> Vec<F> {
    vec![F::zero(); len]
}

/// Unpacked proof with subcomponents
pub struct UnpackedProof<'a, F: FieldElement> {
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
pub struct UnpackedProofMut<'a, F: FieldElement> {
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
pub fn unpack_proof<F: FieldElement>(proof: &[F], dimension: usize) -> Option<UnpackedProof<F>> {
    // check the proof length
    if proof.len() != proof_length(dimension) {
        return None;
    }
    // split share into components
    let (data, rest) = proof.split_at(dimension);
    let (zero_terms, points_h_packed) = rest.split_at(3);
    if let [f0, g0, h0] = zero_terms {
        let unpacked = UnpackedProof {
            data,
            f0,
            g0,
            h0,
            points_h_packed,
        };
        Some(unpacked)
    } else {
        None
    }
}

/// Unpacks a mutable proof vector into mutable subcomponents
pub fn unpack_proof_mut<F: FieldElement>(
    proof: &mut [F],
    dimension: usize,
) -> Option<UnpackedProofMut<F>> {
    // check the share length
    if proof.len() != proof_length(dimension) {
        return None;
    }
    // split share into components
    let (data, rest) = proof.split_at_mut(dimension);
    let (zero_terms, points_h_packed) = rest.split_at_mut(3);
    if let [f0, g0, h0] = zero_terms {
        let unpacked = UnpackedProofMut {
            data,
            f0,
            g0,
            h0,
            points_h_packed,
        };
        Some(unpacked)
    } else {
        None
    }
}

/// Get a byte array from a slice of field elements
pub fn serialize<F: FieldElement>(data: &[F]) -> Vec<u8> {
    let mut vec = Vec::<u8>::with_capacity(data.len() * F::BYTES);
    for elem in data.iter() {
        elem.append_to(&mut vec);
    }
    vec
}

/// Get a vector of field elements from a byte slice
pub fn deserialize<F: FieldElement>(data: &[u8]) -> Result<Vec<F>, SerializeError> {
    if data.len() % F::BYTES != 0 {
        return Err(SerializeError::IncompleteChunk);
    }
    let mut vec = Vec::<F>::with_capacity(data.len() / F::BYTES);
    for chunk in data.chunks_exact(F::BYTES) {
        vec.push(F::read_from(chunk).or_else(|err| Err(SerializeError::Field(err)))?);
    }
    Ok(vec)
}

/// Add two field element arrays together elementwise.
///
/// Returns None, when array lengths are not equal.
pub fn reconstruct_shares<F: FieldElement>(share1: &[F], share2: &[F]) -> Option<Vec<F>> {
    if share1.len() != share2.len() {
        return None;
    }

    let mut reconstructed: Vec<F> = vector_with_length(share1.len());

    for (r, (s1, s2)) in reconstructed
        .iter_mut()
        .zip(share1.iter().zip(share2.iter()))
    {
        *r = *s1 + *s2;
    }

    Some(reconstructed)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::field::Field32;

    pub fn secret_share(share: &mut [Field32]) -> Vec<Field32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut random = vec![0u32; share.len()];
        let mut share2 = vec![Field32::zero(); share.len()];

        rng.fill(&mut random[..]);

        for (r, f) in random.iter().zip(share2.iter_mut()) {
            *f = Field32::from(*r);
        }

        for (f1, f2) in share.iter_mut().zip(share2.iter()) {
            *f1 -= *f2;
        }

        share2
    }

    #[test]
    fn test_unpack_share() {
        let dim = 15;
        let len = proof_length(dim);

        let mut share = vec![Field32::from(0); len];
        let unpacked = unpack_proof_mut(&mut share, dim).unwrap();
        *unpacked.f0 = Field32::from(12);
        assert_eq!(share[dim], 12);
    }

    #[test]
    fn secret_sharing() {
        let mut share1 = vec![Field32::zero(); 10];
        share1[3] = 21.into();
        share1[8] = 123.into();

        let original_data = share1.clone();

        let share2 = secret_share(&mut share1);

        let reconstructed = reconstruct_shares(&share1, &share2).unwrap();
        assert_eq!(reconstructed, original_data);
    }

    #[test]
    fn serialization() {
        let field = [Field32::from(1), Field32::from(0x99997)];
        let bytes = serialize(&field);
        let field_deserialized = deserialize::<Field32>(&bytes).unwrap();
        assert_eq!(field_deserialized, field);
    }
}
