// SPDX-License-Identifier: MPL-2.0

//! A collection of [`Type`](crate::pcp::Type) implementations.

use crate::field::FieldElement;
use crate::pcp::gadgets::{Mul, PolyEval};
use crate::pcp::{Gadget, PcpError, Type};
use crate::polynomial::poly_range_check;

use std::convert::TryFrom;

/// The counter data type. Each measurement is `0` or `1` and the aggregate result is the sum of
/// the measurements (i.e., the total number of `1s`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Count<F> {
    range_checker: Vec<F>,
}

impl<F: FieldElement> Count<F> {
    /// Return a new [`Count`] type instance.
    pub fn new() -> Self {
        Self {
            range_checker: poly_range_check(0, 2),
        }
    }
}

impl<F: FieldElement> Default for Count<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FieldElement> Type for Count<F> {
    type Measurement = F::Integer;
    type Field = F;

    fn encode(&self, value: &F::Integer) -> Result<Vec<F>, PcpError> {
        let max = F::Integer::try_from(1).unwrap();
        if *value > max {
            return Err(PcpError::Encode("Count value must be 0 or 1".to_string()));
        }

        Ok(vec![F::from(*value)])
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(Mul::new(2))]
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        _num_shares: usize,
    ) -> Result<F, PcpError> {
        valid_call_check(self, input, joint_rand)?;

        let mut inp = [input[0], input[0]];
        let mut v = self.range_checker[0];
        for c in &self.range_checker[1..] {
            v += *c * inp[0];
            inp[0] = g[0].call(&inp)?;
        }

        Ok(v)
    }

    fn truncate(&self, input: &[F]) -> Result<Vec<F>, PcpError> {
        truncate_call_check(self, input)?;
        Ok(input.to_vec())
    }

    fn input_len(&self) -> usize {
        1
    }

    fn proof_len(&self) -> usize {
        9
    }

    fn verifier_len(&self) -> usize {
        4
    }

    fn output_len(&self) -> usize {
        self.input_len()
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
}

/// This sum type. Each measurement is a integer in `[0, 2^bits)` and the aggregate is the sum of the measurements.
///
/// The validity circuit is based on the SIMD circuit construction of [[BBCG+19], Theorem 5.3].
///
/// [BBCG+19]: https://ia.cr/2019/188
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sum<F: FieldElement> {
    bits: usize,
    one: F::Integer,
    max_summand: F::Integer,
    range_checker: Vec<F>,
}

impl<F: FieldElement> Sum<F> {
    /// Return a new [`Sum`] type parameter. Each value of this type is an integer in range `[0,
    /// 2^bits)`.
    pub fn new(bits: usize) -> Result<Self, PcpError> {
        let bits_int = F::Integer::try_from(bits).map_err(|err| {
            PcpError::Encode(format!(
                "bit length ({}) cannot be represented as a field element: {:?}",
                bits, err,
            ))
        })?;

        if F::modulus() >> bits_int == F::Integer::from(F::zero()) {
            return Err(PcpError::Encode(format!(
                "bit length ({}) exceeds field modulus",
                bits,
            )));
        }

        let one = F::Integer::from(F::one());
        let max_summand = (one << bits_int) - one;

        Ok(Self {
            bits,
            one,
            max_summand,
            range_checker: poly_range_check(0, 2),
        })
    }
}

impl<F: FieldElement> Type for Sum<F> {
    type Measurement = F::Integer;
    type Field = F;

    fn encode(&self, summand: &F::Integer) -> Result<Vec<F>, PcpError> {
        if *summand > self.max_summand {
            return Err(PcpError::Encode(
                "value of summand exceeds bit length".to_string(),
            ));
        }

        let mut encoded: Vec<F> = Vec::with_capacity(self.bits);
        for l in 0..self.bits {
            let l = F::Integer::try_from(l).unwrap();
            let w = F::from((*summand >> l) & self.one);
            encoded.push(w);
        }

        Ok(encoded)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(PolyEval::new(
            self.range_checker.clone(),
            self.bits,
        ))]
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        _num_shares: usize,
    ) -> Result<F, PcpError> {
        valid_call_check(self, input, joint_rand)?;

        // Check that each element of `data` is a 0 or 1.
        let mut range_check = F::zero();
        let mut r = joint_rand[0];
        for chunk in input.chunks(1) {
            range_check += r * g[0].call(chunk)?;
            r *= joint_rand[0];
        }

        Ok(range_check)
    }

    fn truncate(&self, input: &[F]) -> Result<Vec<F>, PcpError> {
        truncate_call_check(self, input)?;

        let mut decoded = F::zero();
        for (l, bit) in input.iter().enumerate() {
            let w = F::from(F::Integer::try_from(1 << l).unwrap());
            decoded += w * *bit;
        }
        Ok(vec![decoded])
    }

    fn input_len(&self) -> usize {
        self.bits
    }

    fn proof_len(&self) -> usize {
        2 * ((1 + self.bits).next_power_of_two() - 1) + 2
    }

    fn verifier_len(&self) -> usize {
        3
    }

    fn output_len(&self) -> usize {
        1
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
}

/// The histogram type. Each measurement is a non-negative integer and the aggregate is a histogram
/// approximating the distribution of the measurements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram<F: FieldElement> {
    buckets: Vec<F::Integer>,
    range_checker: Vec<F>,
}

impl<F: FieldElement> Histogram<F> {
    /// Return a new [`Histogram`] type with the given buckets.
    pub fn new(buckets: Vec<F::Integer>) -> Result<Self, PcpError> {
        if buckets.len() >= u32::MAX as usize {
            return Err(PcpError::Encode(
                "invalid buckets: number of buckets exceeds maximum permitted".to_string(),
            ));
        }

        if !buckets.is_empty() {
            for i in 0..buckets.len() - 1 {
                if buckets[i + 1] <= buckets[i] {
                    return Err(PcpError::Encode(
                        "invalid buckets: out-of-order boundary".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            buckets,
            range_checker: poly_range_check(0, 2),
        })
    }
}

impl<F: FieldElement> Type for Histogram<F> {
    type Measurement = F::Integer;
    type Field = F;

    fn encode(&self, measurement: &F::Integer) -> Result<Vec<F>, PcpError> {
        let mut data = vec![F::zero(); self.buckets.len() + 1];

        let bucket = match self.buckets.binary_search(measurement) {
            Ok(i) => i,  // on a bucket boundary
            Err(i) => i, // smaller than the i-th bucket boundary
        };

        data[bucket] = F::one();
        Ok(data)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(PolyEval::new(
            self.range_checker.to_vec(),
            self.input_len(),
        ))]
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, PcpError> {
        valid_call_check(self, input, joint_rand)?;

        // Check that each element of `data` is a 0 or 1.
        let mut range_check = F::zero();
        let mut r = joint_rand[0];
        for chunk in input.chunks(1) {
            range_check += r * g[0].call(chunk)?;
            r *= joint_rand[0];
        }

        // Check that the elements of `data` sum to 1.
        let mut sum_check = -(F::one() / F::from(F::Integer::try_from(num_shares).unwrap()));
        for val in input.iter() {
            sum_check += *val;
        }

        // Take a random linear combination of both checks.
        let out = joint_rand[1] * range_check + (joint_rand[1] * joint_rand[1]) * sum_check;
        Ok(out)
    }

    fn truncate(&self, input: &[F]) -> Result<Vec<F>, PcpError> {
        truncate_call_check(self, input)?;
        Ok(input.to_vec())
    }

    fn input_len(&self) -> usize {
        self.buckets.len() + 1
    }

    fn proof_len(&self) -> usize {
        2 * ((1 + self.input_len()).next_power_of_two() - 1) + 2
    }

    fn verifier_len(&self) -> usize {
        3
    }

    fn output_len(&self) -> usize {
        self.input_len()
    }

    fn joint_rand_len(&self) -> usize {
        2
    }

    fn prove_rand_len(&self) -> usize {
        1
    }

    fn query_rand_len(&self) -> usize {
        1
    }
}

fn valid_call_check<T: Type>(
    typ: &T,
    input: &[T::Field],
    joint_rand: &[T::Field],
) -> Result<(), PcpError> {
    if input.len() != typ.input_len() {
        return Err(PcpError::Valid(format!(
            "unexpected input length: got {}; want {}",
            input.len(),
            typ.input_len(),
        )));
    }

    if joint_rand.len() != typ.joint_rand_len() {
        return Err(PcpError::Valid(format!(
            "unexpected joint randomness length: got {}; want {}",
            joint_rand.len(),
            typ.joint_rand_len()
        )));
    }

    Ok(())
}

fn truncate_call_check<T: Type>(typ: &T, input: &[T::Field]) -> Result<(), PcpError> {
    if input.len() != typ.input_len() {
        return Err(PcpError::Truncate(format!(
            "Unexpected input length: got {}; want {}",
            input.len(),
            typ.input_len()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{random_vector, split_vector, Field64 as TestField};

    // Number of shares to split input and proofs into in `pcp_test`.
    const NUM_SHARES: usize = 3;

    struct ValidityTestCase<F> {
        expect_valid: bool,
        expected_output: Option<Vec<F>>,
    }

    #[test]
    fn test_count() {
        let count: Count<TestField> = Count::new();
        let zero = TestField::zero();
        let one = TestField::one();

        // Test PCP on valid input.
        pcp_validity_test(
            &count,
            &count.encode(&1).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![one]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &count,
            &count.encode(&0).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![zero]),
            },
        )
        .unwrap();

        // Test PCP on invalid input.
        pcp_validity_test(
            &count,
            &[TestField::from(1337)],
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        // Try running the validity circuit on an input that's too short.
        count.valid(&mut count.gadget(), &[], &[], 1).unwrap_err();
        count
            .valid(&mut count.gadget(), &[1.into(), 2.into()], &[], 1)
            .unwrap_err();
    }

    #[test]
    fn test_sum() {
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        // TODO(cjpatton) Try encoding invalid measurements.

        // Test PCP on valid input.
        let sum = Sum::new(11).unwrap();
        pcp_validity_test(
            &sum,
            &sum.encode(&1337).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_output: Some(vec![TestField::from(1337)]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::new(0).unwrap(),
            &[],
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![zero]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::new(2).unwrap(),
            &[one, zero],
            &ValidityTestCase {
                expect_valid: true,
                expected_output: Some(vec![one]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::new(9).unwrap(),
            &[one, zero, one, one, zero, one, one, one, zero],
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![TestField::from(237)]),
            },
        )
        .unwrap();

        // Test PCP on invalid input.
        pcp_validity_test(
            &Sum::new(3).unwrap(),
            &[one, nine, zero],
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::new(5).unwrap(),
            &[zero, zero, zero, zero, nine],
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_histogram() {
        let hist = Histogram::new(vec![10, 20]).unwrap();
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        assert_eq!(&hist.encode(&7).unwrap(), &[one, zero, zero]);
        assert_eq!(&hist.encode(&10).unwrap(), &[one, zero, zero]);
        assert_eq!(&hist.encode(&17).unwrap(), &[zero, one, zero]);
        assert_eq!(&hist.encode(&20).unwrap(), &[zero, one, zero]);
        assert_eq!(&hist.encode(&27).unwrap(), &[zero, zero, one]);

        // Invalid bucket boundaries.
        Histogram::<TestField>::new(vec![10, 0]).unwrap_err();
        Histogram::<TestField>::new(vec![10, 10]).unwrap_err();

        // Test valid inputs.
        pcp_validity_test(
            &hist,
            &hist.encode(&0).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![one, zero, zero]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &hist,
            &hist.encode(&17).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![zero, one, zero]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &hist,
            &hist.encode(&1337).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![zero, zero, one]),
            },
        )
        .unwrap();

        // Test invalid inputs.
        pcp_validity_test(
            &hist,
            &[zero, zero, nine],
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        pcp_validity_test(
            &hist,
            &[zero, one, one],
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        pcp_validity_test(
            &hist,
            &[one, one, one],
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        pcp_validity_test(
            &hist,
            &[zero, zero, zero],
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();
    }

    fn pcp_validity_test<T: Type>(
        typ: &T,
        input: &[T::Field],
        t: &ValidityTestCase<T::Field>,
    ) -> Result<(), PcpError> {
        let mut gadgets = typ.gadget();

        if input.len() != typ.input_len() {
            return Err(PcpError::Test(format!(
                "unexpected input length: got {}; want {}",
                input.len(),
                typ.input_len()
            )));
        }

        if typ.query_rand_len() != gadgets.len() {
            return Err(PcpError::Test(format!(
                "query rand length: got {}; want {}",
                typ.query_rand_len(),
                gadgets.len()
            )));
        }

        let joint_rand = random_vector(typ.joint_rand_len()).unwrap();
        let prove_rand = random_vector(typ.prove_rand_len()).unwrap();
        let query_rand = random_vector(typ.query_rand_len()).unwrap();

        // Run the validity circuit.
        let v = typ.valid(&mut gadgets, input, &joint_rand, 1)?;
        if v != T::Field::zero() && t.expect_valid {
            return Err(PcpError::Test(format!(
                "expected valid input: valid() returned {}",
                v
            )));
        }
        if v == T::Field::zero() && !t.expect_valid {
            return Err(PcpError::Test(format!(
                "expected invalid input: valid() returned {}",
                v
            )));
        }

        // Generate the proof.
        let proof = typ.prove(input, &prove_rand, &joint_rand)?;
        if proof.len() != typ.proof_len() {
            return Err(PcpError::Test(format!(
                "unexpected proof length: got {}; want {}",
                proof.len(),
                typ.proof_len()
            )));
        }

        // Query the proof.
        let verifier = typ.query(input, &proof, &query_rand, &joint_rand, 1)?;
        if verifier.len() != typ.verifier_len() {
            return Err(PcpError::Test(format!(
                "unexpected verifier length: got {}; want {}",
                verifier.len(),
                typ.verifier_len()
            )));
        }

        // Decide if the input is valid.
        let res = typ.decide(&verifier)?;
        if res != t.expect_valid {
            return Err(PcpError::Test(format!(
                "decision is {}; want {}",
                res, t.expect_valid,
            )));
        }

        // Run distributed PCP.
        let input_shares: Vec<Vec<T::Field>> = split_vector(input, NUM_SHARES)
            .unwrap()
            .into_iter()
            .collect();

        let proof_shares: Vec<Vec<T::Field>> = split_vector(&proof, NUM_SHARES)
            .unwrap()
            .into_iter()
            .collect();

        let verifier: Vec<T::Field> = (0..NUM_SHARES)
            .map(|i| {
                typ.query(
                    &input_shares[i],
                    &proof_shares[i],
                    &query_rand,
                    &joint_rand,
                    NUM_SHARES,
                )
                .unwrap()
            })
            .reduce(|mut left, right| {
                for (x, y) in left.iter_mut().zip(right.iter()) {
                    *x += *y;
                }
                left
            })
            .unwrap();

        let res = typ.decide(&verifier)?;
        if res != t.expect_valid {
            return Err(PcpError::Test(format!(
                "distributed decision is {}; want {}",
                res, t.expect_valid,
            )));
        }

        // Try verifying various proof mutants.
        for i in 0..proof.len() {
            let mut mutated_proof = proof.clone();
            mutated_proof[i] += T::Field::one();
            let verifier = typ.query(input, &mutated_proof, &query_rand, &joint_rand, 1)?;
            if typ.decide(&verifier)? {
                return Err(PcpError::Test(format!(
                    "decision for proof mutant {} is {}; want {}",
                    i, true, false,
                )));
            }
        }

        // Try verifying a proof that is too short.
        let mut mutated_proof = proof.clone();
        mutated_proof.truncate(gadgets[0].arity() - 1);
        if typ
            .query(input, &mutated_proof, &query_rand, &joint_rand, 1)
            .is_ok()
        {
            return Err(PcpError::Test(
                "query on short proof succeeded; want failure".to_string(),
            ));
        }

        // Try verifying a proof that is too long.
        let mut mutated_proof = proof;
        mutated_proof.extend_from_slice(&[T::Field::one(); 17]);
        if typ
            .query(input, &mutated_proof, &query_rand, &joint_rand, 1)
            .is_ok()
        {
            return Err(PcpError::Test(
                "query on long proof succeeded; want failure".to_string(),
            ));
        }

        if let Some(ref want) = t.expected_output {
            let got = typ.truncate(input)?;

            if got.len() != typ.output_len() {
                return Err(PcpError::Test(format!(
                    "unexpected output length: got {}; want {}",
                    got.len(),
                    typ.output_len()
                )));
            }

            if &got != want {
                return Err(PcpError::Test(format!(
                    "unexpected output: got {:?}; want {:?}",
                    got, want
                )));
            }
        }

        Ok(())
    }
}
