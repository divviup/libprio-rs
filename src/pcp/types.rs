// SPDX-License-Identifier: MPL-2.0

//! A collection of data types.

use crate::field::FieldElement;
use crate::pcp::gadgets::{MeanVarUnsigned, Mul, PolyEval};
use crate::pcp::{GadgetCallOnly, PcpError, Value};
use crate::polynomial::poly_range_check;

use std::convert::TryFrom;
use std::mem::size_of;

/// Errors propagagted by methods in this module.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum TypeError {
    /// Encoding measurement as an input failed
    #[error("encoding error")]
    Encode(&'static str),

    /// XXX
    #[error("XXX")]
    Construct(&'static str),
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

    /// Construct a boolean from an arbitrary field element. (The result may be invalid.)
    pub fn from(val: F) -> Self {
        Self {
            range: poly_range_check(0, 2),
            data: vec![val],
        }
    }
}

impl<F: FieldElement> Value<F, Mul<F>> for Boolean<F> {
    fn valid(&self, g: &mut dyn GadgetCallOnly<F>, rand: &[F]) -> Result<F, PcpError> {
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

impl<F: FieldElement> Value<F, PolyEval<F>> for PolyCheckedVector<F> {
    fn valid(&self, g: &mut dyn GadgetCallOnly<F>, rand: &[F]) -> Result<F, PcpError> {
        if rand.len() != self.valid_rand_len() {
            return Err(PcpError::ValidRandLen);
        }

        let mut outp = F::zero();
        let mut r = rand[0];
        for chunk in self.data.chunks(1) {
            outp += r * g.call(chunk)?;
            r *= rand[0];
        }

        Ok(outp)
    }

    fn valid_gadget_calls(&self) -> usize {
        self.data.len()
    }

    fn valid_rand_len(&self) -> usize {
        1
    }

    fn gadget(&self, in_len: usize) -> PolyEval<F> {
        PolyEval::new(self.poly.clone(), in_len)
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn new_with(&self, data: Vec<F>) -> Self {
        Self {
            data,
            poly: self.poly.clone(),
        }
    }
}

/// XXX
#[derive(Debug, PartialEq, Eq)]
pub struct MeanVarUnsignedVector<F: FieldElement> {
    data: Vec<F>,
    bits: usize,
    len: usize,
    is_leader: bool,
}

impl<F: FieldElement> MeanVarUnsignedVector<F> {
    /// XXX
    pub fn new(bits: usize, ints: &[u64]) -> Result<Self, TypeError> {
        if bits > (size_of::<usize>() << 3) {
            return Err(TypeError::Construct(
                "can't instantiate MeanVarUnsignedVector with bits > bit length of usize",
            ));
        }

        if bits > 64 {
            return Err(TypeError::Construct(
                "can't instantiate MeanVarUnsignedVector with bits > 64",
            ));
        }

        let max = 1 << bits;
        let mut data: Vec<F> = Vec::with_capacity((bits + 2) * ints.len());
        for int in ints {
            let int = *int as usize;
            if int >= max {
                return Err(TypeError::Encode("input >= 2^bits"));
            }

            for l in 0..bits {
                let w = F::from(F::Integer::try_from((int >> l) & 1).unwrap());
                data.push(w);
            }

            let int = F::Integer::try_from(int);
            if int.is_err() {
                return Err(TypeError::Encode("field is too small to encode input"));
            }

            let x = F::from(int.unwrap());
            data.push(x);
            data.push(x * x);
        }

        Ok(Self {
            data,
            bits,
            len: ints.len(),
            is_leader: true,
        })
    }
}

impl<F: FieldElement> Value<F, MeanVarUnsigned<F>> for MeanVarUnsignedVector<F> {
    fn valid(&self, g: &mut dyn GadgetCallOnly<F>, rand: &[F]) -> Result<F, PcpError> {
        let bits = self.bits;
        let mut inp = vec![F::zero(); 2 * bits + 1];
        let mut outp = F::zero();

        for (i, chunk) in self.data.chunks(bits + 2).enumerate() {
            if chunk.len() < bits + 2 {
                panic!("XXX");
            }

            let mut pr = rand[i];
            let x_vec = &chunk[..bits];
            let x = chunk[bits];
            let xx = chunk[bits + 1];

            // The first bits inputs to the gadget are are `r, r^2, ..., r^bits`.
            for l in 0..bits {
                inp[l] = if self.is_leader { pr } else { F::zero() };
                pr *= rand[i];
            }

            // The next bits inputs to the gadget comprise the bit-vector representation of `x`.
            &inp[bits..2 * bits].clone_from_slice(x_vec);

            // The last input to the gadget is `x`.
            inp[2 * bits] = x;

            // Sets `yy = x^2` if `x_vec` is a bit vector. Otherwise, if `x_vec` is not a bit
            // vector, then `yy != x^2` with high probability.
            let yy = g.call(&inp)?;

            // Sets `y` to the `bits`-bit integer encoded by `x_vec`.
            let mut y = F::zero();
            for l in 0..bits {
                let w = F::from(F::Integer::try_from(1 << l).unwrap());
                y += w * x_vec[l];
            }

            outp += pr * (yy - xx);
            pr *= rand[i];

            outp += pr * (y - x);
            pr *= rand[i];
        }

        Ok(outp)
    }

    fn valid_gadget_calls(&self) -> usize {
        self.len
    }

    fn valid_rand_len(&self) -> usize {
        self.len
    }

    fn gadget(&self, in_len: usize) -> MeanVarUnsigned<F> {
        MeanVarUnsigned::new(self.bits, in_len)
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn new_with(&self, data: Vec<F>) -> Self {
        if data.len() % (self.bits + 2) != 0 {
            panic!("XXX");
        }
        let len = data.len() / (self.bits + 2);
        Self {
            data,
            bits: self.bits,
            len,
            is_leader: true,
        }
    }

    fn set_leader(&mut self, is_leader: bool) {
        self.is_leader = is_leader;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{rand_vec, split, Field64 as TestField};
    use crate::pcp::{decide, prove, query, Gadget, Proof, Value, Verifier};

    use std::convert::TryFrom;

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
                data: vec![TestField::rand()],
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

    #[test]
    fn test_poly_checked_vector() {
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
    fn test_mean_var_integer_vector() {
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
        assert_eq!(x.data.len(), ints.len() * (2 + bits));
        for chunk in x.data.chunks(2 + bits) {
            let x_vec = &chunk[..bits];
            let x = chunk[bits];
            let xx = chunk[bits + 1];

            assert_eq!(x * x, xx);

            let mut x_sum = TestField::zero();
            for l in 0..bits {
                x_sum += TestField::from(1 << l) * x_vec[l];
            }
            assert_eq!(x_sum, x);
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
                data: vec![
                    /* x_vec */ one, zero, one, /* x */ one, /* xx */ one,
                ],
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
                data: vec![
                    /* x_vec */ nine, zero, zero, /* x */ one, /* xx */ one,
                ],
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
                data: vec![
                    /* x_vec */ one, zero, zero, /* x */ one, /* xx */ nine,
                ],
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
                // x is incorrect
                data: vec![
                    /* x_vec */ zero,
                    zero,
                    zero,
                    /* x */ nine,
                    /* xx */ TestField::from(81),
                ],
                bits: 3,
                len: 1,
                is_leader: true,
            },
            &ValidityTestCase {
                expect_valid: false,
                expected_proof_len: 11,
            },
        );
    }

    fn pcp_validity_test<F, G, V>(x: &V, t: &ValidityTestCase)
    where
        F: FieldElement,
        G: Gadget<F>,
        V: Value<F, G>,
    {
        let l = x.gadget(0).call_in_len();
        let rand_len = x.valid_rand_len();
        let joint_rand = rand_vec(rand_len);
        let query_rand = vec![F::rand()];

        // Ensure that `new_with` properly clones the value's parameters.
        assert_eq!(x, &x.new_with(x.as_slice().to_vec()));

        // Run the validity circuit.
        let v = x.valid(&mut x.gadget(0), &joint_rand).unwrap();
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
            .into_iter()
            .enumerate()
            .map(|(i, data)| {
                let mut share = x.new_with(data);
                share.set_leader(i == 0);
                share
            })
            .collect();

        let pf_shares: Vec<Proof<F>> = split(pf.as_slice(), NUM_SHARES)
            .into_iter()
            .map(|data| Proof::from(data))
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
        assert_eq!(
            decide(x, &query(x, &mutated_pf, &query_rand, &joint_rand).unwrap()).unwrap(),
            false,
            "{:?} proof mutant verified",
            x.as_slice(),
        );

        // Try verifying a proof that is too short.
        let mut mutated_pf = pf.clone();
        mutated_pf.data.truncate(l - 1);
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

        if x.as_slice().len() > l {
            // Try verifying a proof with an invalid proof polynomial.
            let mut mutated_pf = pf.clone();
            mutated_pf.data[l] += F::one();
            assert_eq!(
                decide(x, &query(x, &mutated_pf, &query_rand, &joint_rand).unwrap()).unwrap(),
                false,
                "{:?} proof mutant verified",
                x.as_slice(),
            );

            // Try verifying a proof with a short proof polynomial.
            let mut mutated_pf = pf.clone();
            mutated_pf.data.truncate(l);
            assert_eq!(
                decide(x, &query(x, &mutated_pf, &query_rand, &joint_rand).unwrap()).unwrap(),
                false,
                "{:?} proof mutant verified",
                x.as_slice(),
            );
        }
    }
}
