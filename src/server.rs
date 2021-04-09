// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Prio server
use crate::{
    encrypt::{decrypt_share, EncryptError, PrivateKey},
    field::{merge_vector, FieldElement, FieldError},
    polynomial::{poly_interpret_eval, PolyAuxMemory},
    prng::extract_share_from_seed,
    util::{deserialize, proof_length, unpack_proof, vector_with_length, SerializeError},
};

/// Possible errors from server operations
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Encryption/decryption error
    #[error("encryption/decryption error")]
    Encrypt(#[from] EncryptError),
    /// Finite field operation error
    #[error("finite field operation error")]
    Field(#[from] FieldError),
    /// Serialization/deserialization error
    #[error("serialization/deserialization error")]
    Serialize(#[from] SerializeError),
}

/// Auxiliary memory for constructing a
/// [`VerificationMessage`](struct.VerificationMessage.html)
#[derive(Debug)]
pub struct ValidationMemory<F: FieldElement> {
    points_f: Vec<F>,
    points_g: Vec<F>,
    points_h: Vec<F>,
    poly_mem: PolyAuxMemory<F>,
}

impl<F: FieldElement> ValidationMemory<F> {
    /// Construct a new ValidationMemory object for validating proof shares of
    /// length `dimension`.
    pub fn new(dimension: usize) -> Self {
        let n: usize = (dimension + 1).next_power_of_two();
        ValidationMemory {
            points_f: vector_with_length(n),
            points_g: vector_with_length(n),
            points_h: vector_with_length(2 * n),
            poly_mem: PolyAuxMemory::new(n),
        }
    }
}

/// Main workhorse of the server.
#[derive(Debug)]
pub struct Server<F: FieldElement> {
    dimension: usize,
    is_first_server: bool,
    accumulator: Vec<F>,
    validation_mem: ValidationMemory<F>,
    private_key: PrivateKey,
}

impl<F: FieldElement> Server<F> {
    /// Construct a new server instance
    ///
    /// Params:
    ///  * `dimension`: the number of elements in the aggregation vector.
    ///  * `is_first_server`: only one of the servers should have this true.
    ///  * `private_key`: the private key for decrypting the share of the proof.
    pub fn new(dimension: usize, is_first_server: bool, private_key: PrivateKey) -> Server<F> {
        Server {
            dimension,
            is_first_server,
            accumulator: vector_with_length(dimension),
            validation_mem: ValidationMemory::new(dimension),
            private_key,
        }
    }

    /// Decrypt and deserialize
    fn deserialize_share(&self, encrypted_share: &[u8]) -> Result<Vec<F>, ServerError> {
        let share = decrypt_share(encrypted_share, &self.private_key)?;
        Ok(if self.is_first_server {
            deserialize(&share)?
        } else {
            let len = proof_length(self.dimension);
            extract_share_from_seed(len, &share)
        })
    }

    /// Generate verification message from an encrypted share
    ///
    /// This decrypts the share of the proof and constructs the
    /// [`VerificationMessage`](struct.VerificationMessage.html).
    /// The `eval_at` field should be generate by
    /// [choose_eval_at](#method.choose_eval_at).
    pub fn generate_verification_message(
        &mut self,
        eval_at: F,
        share: &[u8],
    ) -> Result<VerificationMessage<F>, ServerError> {
        let share_field = self.deserialize_share(share)?;
        generate_verification_message(
            self.dimension,
            eval_at,
            &share_field,
            self.is_first_server,
            &mut self.validation_mem,
        )
    }

    /// Add the content of the encrypted share into the accumulator
    ///
    /// This only changes the accumulator if the verification messages `v1` and
    /// `v2` indicate that the share passed validation.
    pub fn aggregate(
        &mut self,
        share: &[u8],
        v1: &VerificationMessage<F>,
        v2: &VerificationMessage<F>,
    ) -> Result<bool, ServerError> {
        let share_field = self.deserialize_share(share)?;
        let is_valid = is_valid_share(v1, v2);
        if is_valid {
            // Add to the accumulator. share_field also includes the proof
            // encoding, so we slice off the first dimension fields, which are
            // the actual data share.
            merge_vector(&mut self.accumulator, &share_field[..self.dimension])?;
        }

        Ok(is_valid)
    }

    /// Return the current accumulated shares.
    ///
    /// These can be merged together using
    /// [`reconstruct_shares`](../util/fn.reconstruct_shares.html).
    pub fn total_shares(&self) -> &[F] {
        &self.accumulator
    }

    /// Merge shares from another server.
    ///
    /// This modifies the current accumulator.
    ///
    /// # Errors
    ///
    /// Returns an error if `other_total_shares.len()` is not equal to this
    //// server's `dimension`.
    pub fn merge_total_shares(&mut self, other_total_shares: &[F]) -> Result<(), ServerError> {
        Ok(merge_vector(&mut self.accumulator, other_total_shares)?)
    }

    /// Choose a random point for polynomial evaluation
    ///
    /// The point returned is not one of the roots used for polynomial
    /// evaluation.
    pub fn choose_eval_at(&self) -> F {
        loop {
            let eval_at = F::rand();
            if !self.validation_mem.poly_mem.roots_2n.contains(&eval_at) {
                break eval_at;
            }
        }
    }
}

/// Verification message for proof validation
pub struct VerificationMessage<F: FieldElement> {
    /// f evaluated at random point
    pub f_r: F,
    /// g evaluated at random point
    pub g_r: F,
    /// h evaluated at random point
    pub h_r: F,
}

/// Given a proof and evaluation point, this constructs the verification
/// message.
pub fn generate_verification_message<F: FieldElement>(
    dimension: usize,
    eval_at: F,
    proof: &[F],
    is_first_server: bool,
    mem: &mut ValidationMemory<F>,
) -> Result<VerificationMessage<F>, ServerError> {
    let unpacked = unpack_proof(proof, dimension)?;
    let proof_length = 2 * (dimension + 1).next_power_of_two();

    // set zero terms
    mem.points_f[0] = *unpacked.f0;
    mem.points_g[0] = *unpacked.g0;
    mem.points_h[0] = *unpacked.h0;

    // set points_f and points_g
    for (i, x) in unpacked.data.iter().enumerate() {
        mem.points_f[i + 1] = *x;

        if is_first_server {
            // only one server needs to subtract one for point_g
            mem.points_g[i + 1] = *x - F::one();
        } else {
            mem.points_g[i + 1] = *x;
        }
    }

    // set points_h, skipping over elements that should be zero
    let mut i = 1;
    let mut j = 0;
    while i < proof_length {
        mem.points_h[i] = unpacked.points_h_packed[j];
        j += 1;
        i += 2;
    }

    // evaluate polynomials at random point
    let f_r = poly_interpret_eval(
        &mem.points_f,
        &mem.poly_mem.roots_n_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );
    let g_r = poly_interpret_eval(
        &mem.points_g,
        &mem.poly_mem.roots_n_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );
    let h_r = poly_interpret_eval(
        &mem.points_h,
        &mem.poly_mem.roots_2n_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );

    Ok(VerificationMessage { f_r, g_r, h_r })
}

/// Decides if the distributed proof is valid
pub fn is_valid_share<F: FieldElement>(
    v1: &VerificationMessage<F>,
    v2: &VerificationMessage<F>,
) -> bool {
    // reconstruct f_r, g_r, h_r
    let f_r = v1.f_r + v2.f_r;
    let g_r = v1.g_r + v2.g_r;
    let h_r = v1.h_r + v2.h_r;
    // validity check
    f_r * g_r == h_r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field32;
    use crate::util;

    #[test]
    fn test_validation() {
        let dim = 8;
        let proof_u32: Vec<u32> = vec![
            1, 0, 0, 0, 0, 0, 0, 0, 2052337230, 3217065186, 1886032198, 2533724497, 397524722,
            3820138372, 1535223968, 4291254640, 3565670552, 2447741959, 163741941, 335831680,
            2567182742, 3542857140, 124017604, 4201373647, 431621210, 1618555683, 267689149,
        ];

        let mut proof: Vec<Field32> = proof_u32.iter().map(|x| Field32::from(*x)).collect();
        let share2 = util::tests::secret_share(&mut proof);
        let eval_at = Field32::from(12313);

        let mut validation_mem = ValidationMemory::new(dim);

        let v1 =
            generate_verification_message(dim, eval_at, &proof, true, &mut validation_mem).unwrap();
        let v2 = generate_verification_message(dim, eval_at, &share2, false, &mut validation_mem)
            .unwrap();
        assert_eq!(is_valid_share(&v1, &v2), true);
    }
}
