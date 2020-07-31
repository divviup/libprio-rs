//! Prio server

use crate::encrypt::*;
use crate::finite_field::*;
use crate::polynomial::*;
use crate::prng;
use crate::util;
use crate::util::*;

/// Auxiliary memory for constructing a
/// [`VerificationMessage`](struct.VerificationMessage.html)
#[derive(Debug)]
pub struct ValidationMemory {
    points_f: Vec<Field>,
    points_g: Vec<Field>,
    points_h: Vec<Field>,
    poly_mem: PolyAuxMemory,
}

impl ValidationMemory {
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
pub struct Server {
    dimension: usize,
    is_first_server: bool,
    accumulator: Vec<Field>,
    validation_mem: ValidationMemory,
    private_key: PrivateKey,
}

impl Server {
    /// Construct a new server instance
    ///
    /// Params:
    ///  * `dimension`: the number of elements in the aggregation vector.
    ///  * `is_first_server`: only one of the servers should have this true.
    ///  * `private_key`: the private key for decrypting the share of the proof.
    pub fn new(dimension: usize, is_first_server: bool, private_key: PrivateKey) -> Server {
        Server {
            dimension,
            is_first_server,
            accumulator: vector_with_length(dimension),
            validation_mem: ValidationMemory::new(dimension),
            private_key,
        }
    }

    /// Decrypt and deserialize
    fn deserialize_share(&self, encrypted_share: &[u8]) -> Result<Vec<Field>, EncryptError> {
        let share = decrypt_share(encrypted_share, &self.private_key)?;
        Ok(if self.is_first_server {
            util::deserialize(&share)
        } else {
            let len = util::proof_length(self.dimension);
            prng::extract_share_from_seed(len, &share)
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
        eval_at: Field,
        share: &[u8],
    ) -> Option<VerificationMessage> {
        let share_field = self.deserialize_share(share).ok()?;
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
        v1: &VerificationMessage,
        v2: &VerificationMessage,
    ) -> Result<bool, EncryptError> {
        let share_field = self.deserialize_share(share)?;
        let is_valid = is_valid_share(v1, v2);
        if is_valid {
            // add to the accumulator
            for (a, s) in self.accumulator.iter_mut().zip(share_field.iter()) {
                *a += *s;
            }
        }

        Ok(is_valid)
    }

    /// Return the current accumulated shares.
    ///
    /// These can be merged together using
    /// [`reconstruct_shares`](../util/fn.reconstruct_shares.html).
    pub fn total_shares(&self) -> &[Field] {
        &self.accumulator
    }

    /// Merge shares from another server.
    ///
    /// This modifies the current accumulator
    pub fn merge_total_shares(&mut self, other_total_shares: &[Field]) {
        for (a, o) in self.accumulator.iter_mut().zip(other_total_shares.iter()) {
            *a += *o;
        }
    }

    /// Choose a random point for polynomial evaluation
    ///
    /// The point returned is not one of the roots used for polynomial
    /// evaluation.
    pub fn choose_eval_at(&self) -> Field {
        loop {
            let eval_at = Field::from(rand::random::<u32>());
            if !self.validation_mem.poly_mem.roots_2n.contains(&eval_at) {
                break eval_at;
            }
        }
    }
}

/// Verification message for proof validation
pub struct VerificationMessage {
    /// f evaluated at random point
    pub f_r: Field,
    /// g evaluated at random point
    pub g_r: Field,
    /// h evaluated at random point
    pub h_r: Field,
}

/// Given a proof and evaluation point, this constructs the verification
/// message.
pub fn generate_verification_message(
    dimension: usize,
    eval_at: Field,
    proof: &[Field],
    is_first_server: bool,
    mem: &mut ValidationMemory,
) -> Option<VerificationMessage> {
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
            mem.points_g[i + 1] = *x - 1.into();
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

    let vm = VerificationMessage { f_r, g_r, h_r };
    Some(vm)
}

/// Decides if the distributed proof is valid
pub fn is_valid_share(v1: &VerificationMessage, v2: &VerificationMessage) -> bool {
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

    #[test]
    fn test_validation() {
        let dim = 8;
        let proof_u32: Vec<u32> = vec![
            1, 0, 0, 0, 0, 0, 0, 0, 2052337230, 3217065186, 1886032198, 2533724497, 397524722,
            3820138372, 1535223968, 4291254640, 3565670552, 2447741959, 163741941, 335831680,
            2567182742, 3542857140, 124017604, 4201373647, 431621210, 1618555683, 267689149,
        ];

        let mut proof: Vec<Field> = proof_u32.iter().map(|x| Field::from(*x)).collect();
        let share2 = util::tests::secret_share(&mut proof);
        let eval_at = Field::from(12313);

        let mut validation_mem = ValidationMemory::new(dim);

        let v1 =
            generate_verification_message(dim, eval_at, &proof, true, &mut validation_mem).unwrap();
        let v2 = generate_verification_message(dim, eval_at, &share2, false, &mut validation_mem)
            .unwrap();
        assert_eq!(is_valid_share(&v1, &v2), true);
    }
}
