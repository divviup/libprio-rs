// SPDX-License-Identifier: MPL-2.0

//! A collection of data types.

use crate::field::FieldElement;
use crate::pcp::gadgets::{Mul, PolyEval};
use crate::pcp::{Gadget, PcpError, Value};
use crate::polynomial::poly_range_check;

use std::convert::TryFrom;
use std::mem::size_of;

/// The counter data type. Each measurement is `0` or `1` and the aggregate result is the sum of
/// the measurements (i.e., the number of `1s`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Count<F: FieldElement> {
    data: Vec<F>,  // The encoded input
    range: Vec<F>, // A range check polynomial for [0, 2)
}

impl<F: FieldElement> Count<F> {
    /// Construct a new counter.
    pub fn new(value: u64) -> Result<Self, PcpError> {
        Ok(Self {
            range: poly_range_check(0, 2),
            data: vec![match value {
                1 => F::one(),
                0 => F::zero(),
                _ => {
                    return Err(PcpError::Value("Count value  must be 0 or 1".to_string()));
                }
            }],
        })
    }
}

impl<F: FieldElement> Value for Count<F> {
    type Field = F;
    type Param = ();

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        if rand.len() != self.joint_rand_len() {
            return Err(PcpError::Valid(format!(
                "unexpected joint randomness length: got {}; want {}",
                rand.len(),
                self.joint_rand_len()
            )));
        }

        if self.data.len() != 1 {
            return Err(PcpError::Valid(format!(
                "unexpected input length: got {}; want {}",
                self.data.len(),
                1
            )));
        }

        let mut inp = [self.data[0], self.data[0]];
        let mut v = self.range[0];
        for c in &self.range[1..] {
            v += *c * inp[0];
            inp[0] = g[0].call(&inp)?;
        }

        Ok(v)
    }

    fn valid_gadget_calls(&self) -> Vec<usize> {
        vec![2]
    }

    fn joint_rand_len(&self) -> usize {
        0
    }

    fn prove_rand_len(&self) -> usize {
        2
    }

    fn query_rand_len(&self) -> usize {
        1
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(Mul::new(2))]
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn param(&self) -> Self::Param {}
}

impl<F: FieldElement> TryFrom<((), &[F])> for Count<F> {
    type Error = PcpError;

    fn try_from(val: ((), &[F])) -> Result<Self, PcpError> {
        Ok(Self {
            data: val.1.to_vec(),
            range: poly_range_check(0, 2),
        })
    }
}

/// This sum type. Each measurement is a integer in `[0, 2^bits)` and the aggregate is the sum of the measurements.
///
/// The validity circuit is based on the SIMD circuit construction of [[BBCG+19], Theorem 5.3].
///
/// [BBCG+19]: https://ia.cr/2019/188
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sum<F: FieldElement> {
    data: Vec<F>,
    range_checker: Vec<F>,
}

impl<F: FieldElement> Sum<F> {
    /// Constructs a new summand. The value of `summand` must be in `[0, 2^bits)`.
    pub fn new(summand: u64, bits: u32) -> Result<Self, PcpError> {
        let summand = usize::try_from(summand).unwrap();
        let bits = usize::try_from(bits).unwrap();

        if bits > (size_of::<F::Integer>() << 3) {
            return Err(PcpError::Value(
                "bits exceeds bit length of the field's integer representation".to_string(),
            ));
        }

        let int = F::Integer::try_from(summand).map_err(|err| {
            PcpError::Value(format!("failed to convert summand to field: {:?}", err))
        })?;

        let max = F::Integer::try_from(1 << bits).unwrap();
        if int >= max {
            return Err(PcpError::Value(
                "value of summand exceeds bit length".to_string(),
            ));
        }

        let one = F::Integer::try_from(1).unwrap();
        let mut data: Vec<F> = Vec::with_capacity(bits);
        for l in 0..bits {
            let l = F::Integer::try_from(l).unwrap();
            let w = F::from((int >> l) & one);
            data.push(w);
        }

        Ok(Self {
            data,
            range_checker: poly_range_check(0, 2),
        })
    }
}

impl<F: FieldElement> Value for Sum<F> {
    type Field = F;
    type Param = u32;

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        if rand.len() != self.joint_rand_len() {
            return Err(PcpError::Valid(format!(
                "unexpected joint randomness length: got {}; want {}",
                rand.len(),
                self.joint_rand_len()
            )));
        }

        // Check that each element of `data` is a 0 or 1.
        let mut range_check = F::zero();
        let mut r = rand[0];
        for chunk in self.data.chunks(1) {
            range_check += r * g[0].call(chunk)?;
            r *= rand[0];
        }

        Ok(range_check)
    }

    fn valid_gadget_calls(&self) -> Vec<usize> {
        vec![self.data.len()]
    }

    fn joint_rand_len(&self) -> usize {
        1
    }

    fn prove_rand_len(&self) -> usize {
        1
    }

    fn query_rand_len(&self) -> usize {
        1
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(PolyEval::new(
            self.range_checker.clone(),
            self.data.len(),
        ))]
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn param(&self) -> Self::Param {
        self.data.len() as u32
    }
}

impl<F: FieldElement> TryFrom<(u32, &[F])> for Sum<F> {
    type Error = PcpError;

    fn try_from(val: (u32, &[F])) -> Result<Self, PcpError> {
        let bits = usize::try_from(val.0).unwrap();
        let data = val.1;

        if data.len() != bits {
            return Err(PcpError::Value(
                "data length does not match bit length".to_string(),
            ));
        }

        Ok(Self {
            data: data.to_vec(),
            range_checker: poly_range_check(0, 2),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{random_vector, split_vector, Field64 as TestField};
    use crate::pcp::{decide, prove, query, Proof, Value, Verifier};

    // Number of shares to split input and proofs into in `pcp_test`.
    const NUM_SHARES: usize = 3;

    struct ValidityTestCase {
        expect_valid: bool,
        expected_proof_len: usize,
    }

    #[test]
    fn test_count() {
        // Test PCP on valid input.
        pcp_validity_test(
            &Count::<TestField>::new(1).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 9,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Count::<TestField>::new(0).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 9,
            },
        )
        .unwrap();

        // Test PCP on invalid input.
        pcp_validity_test(
            &Count {
                data: vec![TestField::from(1337)],
                range: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 9,
            },
        )
        .unwrap();

        // Try running the validity circuit on an input that's too short.
        let malformed_x = Count::<TestField> {
            data: vec![],
            range: poly_range_check(0, 2),
        };
        malformed_x
            .valid(&mut malformed_x.gadget(), &[])
            .unwrap_err();

        // Try running the validity circuit on an input that's too large.
        let malformed_x = Count::<TestField> {
            data: vec![TestField::zero(), TestField::zero()],
            range: poly_range_check(0, 2),
        };
        malformed_x
            .valid(&mut malformed_x.gadget(), &[])
            .unwrap_err();
    }

    #[test]
    fn test_sum() {
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        // Test PCP on valid input.
        pcp_validity_test(
            &Sum::<TestField>::new(1337, 11).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 32,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 2,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![one, zero],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 8,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![one, zero, one, one, zero, one, one, one, zero],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 32,
            },
        )
        .unwrap();

        // Test PCP on invalid input.
        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![one, nine, zero],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 8,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![zero, zero, zero, zero, nine],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 16,
            },
        )
        .unwrap();
    }

    fn pcp_validity_test<V: Value>(input: &V, t: &ValidityTestCase) -> Result<(), PcpError> {
        let mut gadgets = input.gadget();
        let joint_rand = random_vector(input.joint_rand_len()).unwrap();
        let prove_rand = random_vector(input.prove_rand_len()).unwrap();
        let query_rand = random_vector(input.query_rand_len()).unwrap();

        if input.query_rand_len() != gadgets.len() {
            return Err(PcpError::Test(format!(
                "query rand length: got {}; want {}",
                input.query_rand_len(),
                gadgets.len()
            )));
        }

        // Ensure that the input can be constructed from its parameters and its encoding as a
        // sequence of field elements.
        let got = &V::try_from((input.param(), input.as_slice()))?;
        if got != input {
            return Err(PcpError::Test(format!(
                "input constructed from data and param does not match input: got {:?}; want {:?}",
                got, input
            )));
        }

        // Run the validity circuit.
        let v = input.valid(&mut gadgets, &joint_rand)?;
        if v != V::Field::zero() && t.expect_valid {
            return Err(PcpError::Test(format!(
                "expected valid input: valid() returned {}",
                v
            )));
        }
        if v == V::Field::zero() && !t.expect_valid {
            return Err(PcpError::Test(format!(
                "expected invalid input: valid() returned {}",
                v
            )));
        }

        // Generate the proof.
        let proof = prove(input, &prove_rand, &joint_rand)?;
        if proof.as_slice().len() != t.expected_proof_len {
            // TODO(cjpatton) Have `Value` compute the expected proof length and check that here
            // instead.
            return Err(PcpError::Test(format!(
                "unexpected proof length: got {}; want {}",
                proof.as_slice().len(),
                t.expected_proof_len
            )));
        }

        // Query the proof.
        let verifier = query(input, &proof, &query_rand, &joint_rand)?;
        // TODO(cjpatton) Have `Value` compute the expected verifier length and check that here.

        // Decide if the input is valid.
        let res = decide(input, &verifier)?;
        if res != t.expect_valid {
            return Err(PcpError::Test(format!(
                "decision is {}; want {}",
                res, t.expect_valid,
            )));
        }

        // Run distributed PCP.
        let input_shares: Vec<V> = split_vector(input.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, data)| {
                let mut input_share = V::try_from((input.param(), &data)).unwrap();
                input_share.set_leader(i == 0);
                input_share
            })
            .collect();

        let proof_shares: Vec<Proof<V::Field>> = split_vector(proof.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .map(Proof::from)
            .collect();

        let verifier: Verifier<V::Field> = (0..NUM_SHARES)
            .map(|i| query(&input_shares[i], &proof_shares[i], &query_rand, &joint_rand).unwrap())
            .reduce(|mut left, right| {
                for (x, y) in left.data.iter_mut().zip(right.data.iter()) {
                    *x += *y;
                }
                Verifier { data: left.data }
            })
            .unwrap();

        let res = decide(&input_shares[0], &verifier)?;
        if res != t.expect_valid {
            return Err(PcpError::Test(format!(
                "distributed decision is {}; want {}",
                res, t.expect_valid,
            )));
        }

        // Try verifying various proof mutants.
        for i in 0..proof.as_slice().len() {
            let mut mutated_proof = proof.clone();
            mutated_proof.data[i] += V::Field::one();
            let verifier = query(input, &mutated_proof, &query_rand, &joint_rand)?;
            if decide(input, &verifier)? {
                return Err(PcpError::Test(format!(
                    "decision for proof mutant {} is {}; want {}",
                    i, true, false,
                )));
            }
        }

        // Try verifying a proof that is too short.
        let mut mutated_proof = proof.clone();
        mutated_proof.data.truncate(gadgets[0].arity() - 1);
        if !query(input, &mutated_proof, &query_rand, &joint_rand).is_err() {
            return Err(PcpError::Test(format!(
                "query on short proof succeeded; want failure",
            )));
        }

        // Try verifying a proof that is too long.
        let mut mutated_proof = proof.clone();
        mutated_proof.data.extend_from_slice(&[V::Field::one(); 17]);
        if !query(input, &mutated_proof, &query_rand, &joint_rand).is_err() {
            return Err(PcpError::Test(format!(
                "query on short proof succeeded; want failure",
            )));
        }

        Ok(())
    }
}
