// SPDX-License-Identifier: MPL-2.0

//! A collection of data types.

use crate::field::FieldElement;
use crate::pcp::gadgets::{MeanVarUnsigned, Mul, PolyEval};
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

/// This type is used to compute the mean and variance of a sequence of integers in range `[0,
/// 2^bits)` for some length parameter `bits`.
///
/// Each integer is encoded using `bits + 1` field elements. The encoding has two parts: `x_vec`,
/// the representation of the integer as a boolean vector (i.e., a sequence of 0s and 1s); and
/// `xx`, which is the integer raised to the power of two. The validity circuit checks that the
/// integer represented by `x_vec` is the square root of `xx`. It is based on the SIMD circuit of
/// \[BBG+19, Theorem 5.3\].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeanVarUnsignedVector<F: FieldElement> {
    data: Vec<F>,
    /// The maximum length of each integer.
    bits: usize,
    /// The number of integers encoded by `data`.
    len: usize,
    /// Whether `data` corresponds to the leader's share.
    is_leader: bool,
}

impl<F: FieldElement> MeanVarUnsignedVector<F> {
    /// Encodes `measurement` as an instance of the MeanVarUnsignedVector type. `bits` specifies
    /// the maximum length of each integer in bits.
    pub fn new(bits: usize, measurement: &[F::Integer]) -> Result<Self, PcpError> {
        if bits > (size_of::<F::Integer>() << 3) {
            return Err(PcpError::Value(
                "MeanVarUnsignedVector: bits exceeds bit length of the field's integer representation".to_string(),
            ));
        }

        let one = F::Integer::try_from(1).unwrap();
        let max = F::Integer::try_from(1 << bits).unwrap();
        let mut data: Vec<F> = Vec::with_capacity((bits + 1) * measurement.len());
        for &int in measurement {
            if int >= max {
                return Err(PcpError::Value(
                    "MeanVarUnsignedVector: input overflow".to_string(),
                ));
            }

            for l in 0..bits {
                let l = F::Integer::try_from(l).unwrap();
                let w = F::from((int >> l) & one);
                data.push(w);
            }

            let x = F::from(int);
            data.push(x * x);
        }

        Ok(Self {
            data,
            bits,
            len: measurement.len(),
            is_leader: true,
        })
    }
}

impl<F: FieldElement> Value for MeanVarUnsignedVector<F> {
    type Field = F;
    type Param = usize; // Length of each integer in bits

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        let bits = self.bits;
        let mut inp = vec![F::zero(); 2 * bits + 1];
        let mut outp = F::zero();

        let r = rand[0];
        let mut pr = r;
        for chunk in self.data.chunks(bits + 1) {
            if chunk.len() < bits + 1 {
                return Err(PcpError::Valid(
                    "MeanVarUnsignedVector: length of data not divisible by chunk size".to_string(),
                ));
            }

            let x_vec = &chunk[..bits];
            let xx = chunk[bits];

            // Sets `x` to the `bits`-bit integer encoded by `x_vec`.
            let mut x = F::zero();
            for (l, bit) in x_vec.iter().enumerate() {
                let w = F::from(F::Integer::try_from(1 << l).unwrap());
                x += w * *bit;
            }

            // The first `bits` inputs to the gadget are are `r, r^2, ..., r^bits`.
            #[allow(clippy::needless_range_loop)]
            for l in 0..bits {
                // The prover and verifier use joint randomness to generate and verify the proof.
                // In order to ensure the gadget inputs are the same in the distributed setting,
                // only the leader will use the joint randomness here.
                inp[l] = if self.is_leader { pr } else { F::zero() };
                pr *= r;
            }

            // The next `bits` inputs to the gadget comprise the bit-vector representation of `x`.
            inp[bits..2 * bits].clone_from_slice(x_vec);

            // The last input to the gadget is `x`.
            inp[2 * bits] = x;

            // Sets `yy = x^2` if `x_vec` is a bit vector. Otherwise, if `x_vec` is not a bit
            // vector, then `yy != x^2` with high probability.
            let yy = g[0].call(&inp)?;

            outp += pr * (yy - xx);
            pr *= r;
        }

        Ok(outp)
    }

    fn valid_gadget_calls(&self) -> Vec<usize> {
        vec![self.len]
    }

    fn joint_rand_len(&self) -> usize {
        1
    }

    fn prove_rand_len(&self) -> usize {
        2 * self.bits + 1
    }

    fn query_rand_len(&self) -> usize {
        1
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(MeanVarUnsigned::new(self.bits, self.len))]
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn param(&self) -> Self::Param {
        self.bits
    }

    fn set_leader(&mut self, is_leader: bool) {
        self.is_leader = is_leader;
    }
}

impl<F: FieldElement> TryFrom<(usize, &[F])> for MeanVarUnsignedVector<F> {
    type Error = PcpError;

    fn try_from(val: (usize, &[F])) -> Result<Self, PcpError> {
        let bits = val.0;
        let data = val.1.to_vec();
        let len = data.len() / (bits + 1);

        if data.len() % (bits + 1) != 0 {
            return Err(PcpError::Value(
                "MeanVarUnsignedVector: length of data not divisible by chunk size".to_string(),
            ));
        }

        Ok(Self {
            data,
            bits,
            len,
            is_leader: true,
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
        );
        pcp_validity_test(
            &Count::<TestField>::new(0).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 9,
            },
        );

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
        );

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
        );
        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 2,
            },
        );
        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![one, zero],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 8,
            },
        );
        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![one, zero, one, one, zero, one, one, one, zero],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 32,
            },
        );

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
        );
        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![zero, zero, zero, zero, nine],
                range_checker: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 16,
            },
        );
    }

    #[test]
    fn test_mean_var_uint_vec() {
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        // Can't encode an integer that is larger than 2^bits.
        assert!(MeanVarUnsignedVector::<TestField>::new(8, &[256]).is_err());

        // Can't instantiate this type if bits > 64.
        assert!(MeanVarUnsignedVector::<TestField>::new(65, &[]).is_err());

        let bits = 7;
        let ints = [1, 61, 27, 17, 0];
        let x: MeanVarUnsignedVector<TestField> = MeanVarUnsignedVector::new(bits, &ints).unwrap();
        assert_eq!(x.data.len(), ints.len() * (1 + bits));
        for chunk in x.data.chunks(1 + bits) {
            let x_vec = &chunk[..bits];
            let xx = chunk[bits];

            let mut x = TestField::zero();
            for (l, bit) in x_vec.iter().enumerate() {
                x += TestField::from(1 << l) * *bit;
            }
            assert_eq!(x * x, xx);
        }

        // Test PCP on valid input.
        pcp_validity_test(
            &x,
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 37,
            },
        );
        pcp_validity_test(
            &MeanVarUnsignedVector::<TestField>::new(bits, &[]).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 16,
            },
        );
        pcp_validity_test(
            &MeanVarUnsignedVector::<TestField>::new(bits, &[0]).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 19,
            },
        );
        pcp_validity_test(
            &MeanVarUnsignedVector::<TestField>::new(bits, &vec![61; 100]).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 397,
            },
        );

        // Test PCP on invalid input.
        pcp_validity_test(
            &MeanVarUnsignedVector::<TestField> {
                // x_vec is incorrect
                data: vec![/* x_vec */ one, zero, one, /* xx */ one],
                len: 1,
                bits: 3,
                is_leader: true,
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 11,
            },
        );
        pcp_validity_test(
            &MeanVarUnsignedVector::<TestField> {
                // x_vec is malformed
                data: vec![/* x_vec */ nine, zero, zero, /* xx */ one],
                len: 1,
                bits: 3,
                is_leader: true,
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 11,
            },
        );
        pcp_validity_test(
            &MeanVarUnsignedVector::<TestField> {
                // xx is incorrect
                data: vec![/* x_vec */ one, zero, zero, /* xx */ nine],
                len: 1,
                bits: 3,
                is_leader: true,
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 11,
            },
        );
    }

    // TODO(cjpatton) Have this return an error and have the caller assert success (or failure).
    fn pcp_validity_test<V: Value>(input: &V, t: &ValidityTestCase) {
        let mut gadgets = input.gadget();
        let joint_rand = random_vector(input.joint_rand_len()).unwrap();
        let prove_rand = random_vector(input.prove_rand_len()).unwrap();
        let query_rand = random_vector(input.query_rand_len()).unwrap();

        assert_eq!(input.query_rand_len(), gadgets.len());

        // Ensure that the input can be constructed from its parameters and its encoding as a
        // sequence of field elements.
        assert_eq!(
            input,
            &V::try_from((input.param(), input.as_slice())).unwrap()
        );

        // Run the validity circuit.
        let v = input.valid(&mut gadgets, &joint_rand).unwrap();
        assert_eq!(
            v == V::Field::zero(),
            t.expect_valid,
            "{:?} validity circuit output {}",
            input.as_slice(),
            v
        );

        // Generate and verify a PCP.
        let proof = prove(input, &prove_rand, &joint_rand).unwrap();
        let verifier = query(input, &proof, &query_rand, &joint_rand).unwrap();
        let res = decide(input, &verifier).unwrap();
        assert_eq!(
            res,
            t.expect_valid,
            "{:?} query output {:?}",
            input.as_slice(),
            verifier
        );

        // Check that the proof size is as expected.
        assert_eq!(proof.as_slice().len(), t.expected_proof_len);

        // Run distributed PCP.
        let x_shares: Vec<V> = split_vector(input.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, data)| {
                let mut share = V::try_from((input.param(), &data)).unwrap();
                share.set_leader(i == 0);
                share
            })
            .collect();

        let proof_shares: Vec<Proof<V::Field>> = split_vector(proof.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .map(Proof::from)
            .collect();

        let verifier: Verifier<V::Field> = (0..NUM_SHARES)
            .map(|i| query(&x_shares[i], &proof_shares[i], &query_rand, &joint_rand).unwrap())
            .reduce(|mut left, right| {
                for (x, y) in left.data.iter_mut().zip(right.data.iter()) {
                    *x += *y;
                }
                Verifier { data: left.data }
            })
            .unwrap();
        let res = decide(&x_shares[0], &verifier).unwrap();
        assert_eq!(
            res,
            t.expect_valid,
            "{:?} sum of of verifier shares is {:?}",
            input.as_slice(),
            &verifier
        );

        // Try verifying a proof with an invalid seed for one of the intermediate polynomials.
        // Verification should fail regardless of whether the input is valid.
        let mut mutated_proof = proof.clone();
        mutated_proof.data[0] += V::Field::one();
        assert!(
            !decide(
                input,
                &query(input, &mutated_proof, &query_rand, &joint_rand).unwrap()
            )
            .unwrap(),
            "{:?} proof mutant verified",
            input.as_slice(),
        );

        // Try verifying a proof that is too short.
        let mut mutated_proof = proof.clone();
        mutated_proof.data.truncate(gadgets[0].arity() - 1);
        assert!(
            query(input, &mutated_proof, &query_rand, &joint_rand).is_err(),
            "{:?} proof mutant verified",
            input.as_slice(),
        );

        // Try verifying a proof that is too long.
        let mut mutated_proof = proof.clone();
        mutated_proof.data.extend_from_slice(&[V::Field::one(); 17]);
        assert!(
            query(input, &mutated_proof, &query_rand, &joint_rand).is_err(),
            "{:?} proof mutant verified",
            input.as_slice(),
        );

        if input.as_slice().len() > gadgets[0].arity() {
            // Try verifying a proof with an invalid proof polynomial.
            let mut mutated_proof = proof.clone();
            mutated_proof.data[gadgets[0].arity()] += V::Field::one();
            assert!(
                !decide(
                    input,
                    &query(input, &mutated_proof, &query_rand, &joint_rand).unwrap()
                )
                .unwrap(),
                "{:?} proof mutant verified",
                input.as_slice(),
            );

            // Try verifying a proof with a short proof polynomial.
            let mut mutated_proof = proof;
            mutated_proof.data.truncate(gadgets[0].arity());
            assert!(query(input, &mutated_proof, &query_rand, &joint_rand).is_err());
        }
    }
}
