// SPDX-License-Identifier: MPL-2.0

//! A collection of data types.

use crate::field::FieldElement;
use crate::pcp::gadgets::{MeanVarUnsigned, Mul, PolyEval};
use crate::pcp::{Gadget, PcpError, Value};
use crate::polynomial::poly_range_check;

use std::convert::{Infallible, TryFrom};
use std::mem::size_of;

/// Errors propagated by methods in this module.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum TypeError {
    /// Encountered an error while trying to construct an instance of some type.
    #[error("failed to instantiate type: {0}")]
    Instantiate(&'static str),
}

/// Values of this type encode a simple boolean (either `true` or `false`).
#[derive(Debug, PartialEq, Eq)]
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
}

impl<F: FieldElement> Value<F> for Boolean<F> {
    type Param = ();
    type TryFromError = Infallible;

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        if rand.len() != self.valid_rand_len() {
            return Err(PcpError::ValidRandLen);
        }

        if self.data.len() != 1 {
            return Err(PcpError::CircuitInLen);
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

    fn valid_rand_len(&self) -> usize {
        0
    }

    fn valid_gadget_len(&self) -> usize {
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

impl<F: FieldElement> TryFrom<((), Vec<F>)> for Boolean<F> {
    type Error = Infallible;

    fn try_from(val: ((), Vec<F>)) -> Result<Self, Infallible> {
        Ok(Self {
            data: val.1,
            range: poly_range_check(0, 2),
        })
    }
}

/// This type represents vectors for which `poly(x) == 0` holds for some polynomial `poly` and each
/// vector element `x`.
///
/// This type is "generic" in that it can be used to construct high level types. For example, a
/// boolean vector can be represented as follows:
///
/// ```
/// use prio::field::{Field64, FieldElement};
/// use prio::pcp::types::PolyCheckedVector;
///
/// let data = vec![Field64::zero(), Field64::one(), Field64::zero()];
/// let x = PolyCheckedVector::new_range_checked(data, 0, 2);
/// ```
/// The proof technique is based on the SIMD circuit construction of
/// \[[BBG+19](https://eprint.iacr.org/2019/188), Theorem 5.3\].
#[derive(Debug, PartialEq, Eq)]
pub struct PolyCheckedVector<F: FieldElement> {
    data: Vec<F>,
    poly: Vec<F>,
}

impl<F: FieldElement> PolyCheckedVector<F> {
    /// Returns a `poly`-checked vector where `poly(x) == 0` if and only if `x` is in range
    /// `[start, end)`. The degree of `poly` is equal to `end`.
    pub fn new_range_checked(data: Vec<F>, start: usize, end: usize) -> Self {
        Self {
            data,
            poly: poly_range_check(start, end),
        }
    }
}

impl<F: FieldElement> Value<F> for PolyCheckedVector<F> {
    type Param = Vec<F>; // A polynomial
    type TryFromError = Infallible;

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        if rand.len() != self.valid_rand_len() {
            return Err(PcpError::ValidRandLen);
        }

        let mut outp = F::zero();
        let mut r = rand[0];
        for chunk in self.data.chunks(1) {
            outp += r * g[0].call(chunk)?;
            r *= rand[0];
        }

        Ok(outp)
    }

    fn valid_gadget_calls(&self) -> Vec<usize> {
        vec![self.data.len()]
    }

    fn valid_rand_len(&self) -> usize {
        1
    }

    fn valid_gadget_len(&self) -> usize {
        1
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(PolyEval::new(self.poly.clone(), self.data.len()))]
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn param(&self) -> Self::Param {
        self.poly.clone()
    }
}

impl<F: FieldElement> TryFrom<(Vec<F>, Vec<F>)> for PolyCheckedVector<F> {
    type Error = Infallible;

    fn try_from(val: (Vec<F>, Vec<F>)) -> Result<Self, Infallible> {
        Ok(Self {
            data: val.1,
            poly: val.0,
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
#[derive(Debug, PartialEq, Eq)]
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
    pub fn new(bits: usize, measurement: &[F::Integer]) -> Result<Self, TypeError> {
        if bits > (size_of::<F::Integer>() << 3) {
            return Err(TypeError::Instantiate(
                "MeanVarUnsignedVector: bits exceeds bit length of the field's integer representation",
            ));
        }

        let one = F::Integer::try_from(1).unwrap();
        let max = F::Integer::try_from(1 << bits).unwrap();
        let mut data: Vec<F> = Vec::with_capacity((bits + 1) * measurement.len());
        for &int in measurement {
            if int >= max {
                return Err(TypeError::Instantiate(
                    "MeanVarUnsignedVector: input overflow",
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

impl<F: FieldElement> Value<F> for MeanVarUnsignedVector<F> {
    type Param = usize; // Length of each integer in bits
    type TryFromError = TypeError;

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        let bits = self.bits;
        let mut inp = vec![F::zero(); 2 * bits + 1];
        let mut outp = F::zero();

        let r = rand[0];
        let mut pr = r;
        for chunk in self.data.chunks(bits + 1) {
            if chunk.len() < bits + 1 {
                return Err(PcpError::Valid(
                    "MeanVarUnsignedVector: length of data not divisible by chunk size",
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

    fn valid_rand_len(&self) -> usize {
        1
    }

    fn valid_gadget_len(&self) -> usize {
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

impl<F: FieldElement> TryFrom<(usize, Vec<F>)> for MeanVarUnsignedVector<F> {
    type Error = TypeError;

    fn try_from(val: (usize, Vec<F>)) -> Result<Self, TypeError> {
        let bits = val.0;
        let data = val.1;
        let len = data.len() / (bits + 1);

        if data.len() % (bits + 1) != 0 {
            return Err(TypeError::Instantiate(
                "MeanVarUnsignedVector: length of data not divisible by chunk size",
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
    use crate::field::{rand, split, Field64 as TestField};
    use crate::pcp::{decide, prove, query, Proof, Value, Verifier};

    // Number of shares to split input and proofs into in `pcp_test`.
    const NUM_SHARES: usize = 3;

    struct ValidityTestCase {
        expect_valid: bool,
        expected_proof_len: usize,
    }

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
                data: vec![TestField::from(1337)],
                range: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 9,
            },
        );

        // Try running the validity circuit on an input that's too short.
        let malformed_x = Boolean::<TestField> {
            data: vec![],
            range: poly_range_check(0, 2),
        };
        assert_eq!(
            malformed_x.valid(&mut malformed_x.gadget(), &[]).err(),
            Some(PcpError::CircuitInLen),
        );

        // Try running the validity circuit on an input that's too large.
        let malformed_x = Boolean::<TestField> {
            data: vec![TestField::zero(), TestField::zero()],
            range: poly_range_check(0, 2),
        };
        assert_eq!(
            malformed_x.valid(&mut malformed_x.gadget(), &[]).err(),
            Some(PcpError::CircuitInLen),
        );
    }

    #[test]
    fn test_poly_checked_vec() {
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        // Test PCP on valid input.
        pcp_validity_test(
            &PolyCheckedVector::<TestField>::new_range_checked(
                vec![nine, one, one, one, one, one, one, one, one, one, one],
                1,
                10,
            ),
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 137,
            },
        );
        pcp_validity_test(
            &PolyCheckedVector::<TestField> {
                data: vec![],
                poly: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 2,
            },
        );
        pcp_validity_test(
            &PolyCheckedVector::<TestField> {
                data: vec![one],
                poly: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 4,
            },
        );
        pcp_validity_test(
            &PolyCheckedVector::<TestField> {
                data: vec![one, zero, one, one, zero, one, one, one, zero],
                poly: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_proof_len: 32,
            },
        );

        // Test PCP on invalid input.
        pcp_validity_test(
            &PolyCheckedVector::<TestField> {
                data: vec![one, nine, zero],
                poly: poly_range_check(0, 2),
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 8,
            },
        );
        pcp_validity_test(
            &PolyCheckedVector::<TestField> {
                data: vec![zero, zero, zero, zero, nine],
                poly: poly_range_check(0, 2),
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

    fn pcp_validity_test<F, V>(x: &V, t: &ValidityTestCase)
    where
        F: FieldElement,
        V: Value<F>,
    {
        let mut g = x.gadget();
        let joint_rand = rand(x.valid_rand_len()).unwrap();
        let query_rand = rand(x.valid_gadget_len()).unwrap();

        // Check that the output of valid_gadgets_len() is correct.
        assert_eq!(x.valid_gadget_len(), g.len());

        // Ensure that the input can be constructed from its parameters and its encoding as a
        // sequence of field elements.
        assert_eq!(x, &V::try_from((x.param(), x.as_slice().to_vec())).unwrap());

        // Run the validity circuit.
        let v = x.valid(&mut g, &joint_rand).unwrap();
        assert_eq!(
            v == F::zero(),
            t.expect_valid,
            "{:?} validity circuit output {}",
            x.as_slice(),
            v
        );

        // Generate and verify a PCP.
        let pf = prove(x, &joint_rand).unwrap();
        let vf = query(x, &pf, &query_rand, &joint_rand).unwrap();
        let res = decide(x, &vf).unwrap();
        assert_eq!(
            res,
            t.expect_valid,
            "{:?} query output {:?}",
            x.as_slice(),
            vf
        );

        // Check that the proof size is as expected.
        assert_eq!(pf.as_slice().len(), t.expected_proof_len);

        // Run distributed PCP.
        let x_shares: Vec<V> = split(x.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, data)| {
                let mut share = V::try_from((x.param(), data)).unwrap();
                share.set_leader(i == 0);
                share
            })
            .collect();

        let pf_shares: Vec<Proof<F>> = split(pf.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .map(Proof::from)
            .collect();

        let mut vf_shares: Vec<Verifier<F>> = Vec::with_capacity(NUM_SHARES);
        for i in 0..NUM_SHARES {
            vf_shares.push(query(&x_shares[i], &pf_shares[i], &query_rand, &joint_rand).unwrap());
        }

        let vf = Verifier::try_from(vf_shares.as_slice()).unwrap();
        let res = decide(&x_shares[0], &vf).unwrap();
        assert_eq!(
            res,
            t.expect_valid,
            "{:?} sum of of verifier shares is {:?}",
            x.as_slice(),
            &vf
        );

        // Try verifying a proof with an invalid seed for one of the intermediate polynomials.
        // Verification should fail regardless of whether the input is valid.
        let mut mutated_pf = pf.clone();
        mutated_pf.data[0] += F::one();
        assert!(
            !decide(x, &query(x, &mutated_pf, &query_rand, &joint_rand).unwrap()).unwrap(),
            "{:?} proof mutant verified",
            x.as_slice(),
        );

        // Try verifying a proof that is too short.
        let mut mutated_pf = pf.clone();
        mutated_pf.data.truncate(g[0].arity() - 1);
        assert!(
            query(x, &mutated_pf, &query_rand, &joint_rand).is_err(),
            "{:?} proof mutant verified",
            x.as_slice(),
        );

        // Try verifying a proof that is too long.
        let mut mutated_pf = pf.clone();
        mutated_pf.data.extend_from_slice(&[F::one(); 17]);
        assert!(
            query(x, &mutated_pf, &query_rand, &joint_rand).is_err(),
            "{:?} proof mutant verified",
            x.as_slice(),
        );

        let g_arity = g[0].arity();
        if x.as_slice().len() > g_arity {
            // Try verifying a proof with an invalid proof polynomial.
            let mut mutated_pf = pf.clone();
            mutated_pf.data[g_arity] += F::one();
            assert!(
                !decide(x, &query(x, &mutated_pf, &query_rand, &joint_rand).unwrap()).unwrap(),
                "{:?} proof mutant verified",
                x.as_slice(),
            );

            // Try verifying a proof with a short proof polynomial.
            let mut mutated_pf = pf;
            mutated_pf.data.truncate(g_arity);
            assert!(query(x, &mutated_pf, &query_rand, &joint_rand).is_err());
        }
    }
}
