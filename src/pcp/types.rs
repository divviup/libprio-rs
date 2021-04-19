// SPDX-License-Identifier: MPL-2.0

//! A collection of data types.

use crate::field::FieldElement;
use crate::pcp::gadgets::Mul;
use crate::pcp::{GadgetCallOnly, PcpError, Value};
use crate::polynomial::poly_range_check;

/// Values of this type encode a simple boolean (either `true` or `false`).
pub struct Boolean<F: FieldElement> {
    data: Vec<F>,  // The encoded input
    range: Vec<F>, // A range check polynomial for [0, 2)
}

impl<F: FieldElement> Boolean<F> {
    /// Encodes a boolean as a value of this type.
    pub fn new(b: bool) -> Self {
        Self {
            range: poly_range_check(0, 2),
            data: vec![match b {
                true => F::one(),
                false => F::zero(),
            }],
        }
    }

    /// Construct a boolean from an arbitrary field element. (The result may be invalid.)
    pub fn from(val: F) -> Self {
        Self {
            range: poly_range_check(0, 2),
            data: vec![val],
        }
    }
}

impl<F: FieldElement> Value<F, Mul<F>> for Boolean<F> {
    fn valid(&self, g: &mut dyn GadgetCallOnly<F>, _rand: &[F]) -> Result<F, PcpError> {
        if self.data.len() != 1 {
            return Err(PcpError::CircuitInLen);
        }

        let mut inp = [self.data[0], self.data[0]];
        let mut v = self.range[0];
        for c in &self.range[1..] {
            v += *c * inp[0];
            inp[0] = g.call(&inp)?;
        }

        Ok(v)
    }

    fn valid_gadget_calls(&self) -> usize {
        2
    }

    fn valid_rand_len(&self) -> usize {
        0
    }

    fn gadget(&self, in_len: usize) -> Mul<F> {
        Mul::new(in_len)
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn new_with(&self, data: Vec<F>) -> Self {
        Self {
            data,
            range: poly_range_check(0, 2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{split, Field64 as TestField};
    use crate::pcp::{decide, prove, query, rand_vec, Gadget, Proof, Value, Verifier};

    use std::convert::TryFrom;

    // Number of shares to split input and proofs into in `pcp_test`.
    const NUM_SHARES: usize = 3;

    #[test]
    fn test_boolean() {
        // Test PCP on valid input.
        pcp_validity_test(
            &Boolean::<TestField>::new(true),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 9,
            },
        );
        pcp_validity_test(
            &Boolean::<TestField>::new(false),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 9,
            },
        );

        // Test PCP on invalid input.
        pcp_validity_test(
            &Boolean {
                data: vec![TestField::rand()],
                range: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 9,
            },
        );

        // Test PCP on a malformed proofs.
        pcp_mutant_test(&Boolean::<TestField>::new(true));

        // Try running the validity circuit on an input that's too short.
        let malformed_x = Boolean::<TestField> {
            data: vec![],
            range: poly_range_check(0, 2),
        };
        assert_eq!(
            malformed_x.valid(&mut malformed_x.gadget(0), &[]).err(),
            Some(PcpError::CircuitInLen),
        );

        // Try running the validity circuit on an input that's too large.
        let malformed_x = Boolean::<TestField> {
            data: vec![TestField::zero(), TestField::zero()],
            range: poly_range_check(0, 2),
        };
        assert_eq!(
            malformed_x.valid(&mut malformed_x.gadget(0), &[]).err(),
            Some(PcpError::CircuitInLen),
        );
    }

    struct ValidityTestCase {
        expect_valid: bool,
        expected_proof_len: usize,
    }

    fn pcp_validity_test<F, G, V>(x: &V, t: &ValidityTestCase)
    where
        F: FieldElement,
        G: Gadget<F>,
        V: Value<F, G>,
    {
        let rand = rand_vec(1 + x.valid_rand_len());

        // Run the validity circuit.
        let v = x
            .valid(&mut x.gadget(0), &rand_vec(x.valid_rand_len()))
            .unwrap();
        assert_eq!(
            v == F::zero(),
            t.expect_valid,
            "{:?} validity circuit output {}",
            x.as_slice(),
            v
        );

        // Generate and verify a PCP.
        let pf = prove(x).unwrap();
        let vf = query(x, &pf, &rand).unwrap();
        let res = decide(x, &vf).unwrap();
        assert_eq!(
            res,
            t.expect_valid,
            "{:?} query output {:?}",
            x.as_slice(),
            vf
        );

        // CHeck that the proof size is as expected.
        assert_eq!(pf.as_slice().len(), t.expected_proof_len);

        // Run distributed PCP.
        let x_shares: Vec<V> = split(x.as_slice(), NUM_SHARES)
            .into_iter()
            .map(|data| x.new_with(data))
            .collect();

        let pf_shares: Vec<Proof<F>> = split(pf.as_slice(), NUM_SHARES)
            .into_iter()
            .map(|data| Proof::from(data))
            .collect();

        let mut vf_shares: Vec<Verifier<F>> = Vec::with_capacity(NUM_SHARES);
        for i in 0..NUM_SHARES {
            vf_shares.push(query(&x_shares[i], &pf_shares[i], &rand).unwrap());
        }

        let vf = Verifier::try_from(vf_shares.as_slice()).unwrap();
        let res = decide(&x_shares[0], &vf).unwrap();
        assert_eq!(
            res,
            t.expect_valid,
            "{:?} sum of of verifier shares is {:?}",
            x.as_slice(),
            &vf_shares[0]
        );
    }

    fn pcp_mutant_test<F, G, V>(x: &V)
    where
        F: FieldElement,
        G: Gadget<F>,
        V: Value<F, G>,
    {
        let l = x.gadget(0).call_in_len();
        let rand = rand_vec(1 + x.valid_rand_len());
        let pf = prove(x).unwrap();

        // Try verifying a proof with an invalid seed for one of the intermediate polynomials.
        // Verification should fail regardless of whether the input is valid.
        let mut mutated_pf = pf.clone();
        mutated_pf.data[0] += F::one();
        assert_eq!(
            decide(x, &query(x, &mutated_pf, &rand).unwrap()).unwrap(),
            false,
            "{:?} proof mutant verified",
            x.as_slice(),
        );

        // Try verifying a proof with an invalid proof polynomial.
        let mut mutated_pf = pf.clone();
        mutated_pf.data[l + 1] += F::one();
        assert_eq!(
            decide(x, &query(x, &mutated_pf, &rand).unwrap()).unwrap(),
            false,
            "{:?} proof mutant verified",
            x.as_slice(),
        );

        // Try verifying a proof with a short proof polynomial.
        let mut mutated_pf = pf.clone();
        mutated_pf.data.truncate(l + 1);
        assert_eq!(
            decide(x, &query(x, &mutated_pf, &rand).unwrap()).unwrap(),
            false,
            "{:?} proof mutant verified",
            x.as_slice(),
        );

        // Try verifying a proof that is too short.
        let mut mutated_pf = pf.clone();
        mutated_pf.data.truncate(l - 1);
        assert_eq!(
            query(x, &mutated_pf, &rand).err(),
            Some(PcpError::QueryProofLen),
            "{:?} proof mutant verified",
            x.as_slice(),
        );

        // Try verifying a proof that is too long
        let mut mutated_pf = pf.clone();
        mutated_pf.data.extend_from_slice(&[F::one(); 4]);
        assert_eq!(
            query(x, &mutated_pf, &rand).err(),
            Some(PcpError::QueryProofLen),
            "{:?} proof mutant verified",
            x.as_slice(),
        );
    }
}
