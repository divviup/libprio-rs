// SPDX-License-Identifier: MPL-2.0

//! A collection of data types.

use crate::field::FieldElement;
use crate::pcp::gadgets::{Mul, PolyEval};
use crate::pcp::{Gadget, PcpError, Value, ValueParam};
use crate::polynomial::poly_range_check;

use std::convert::TryFrom;
use std::mem::size_of;

/// The counter data type. Each measurement is `0` or `1` and the aggregate result is the sum of
/// the measurements (i.e., the number of `1s`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Count<'a, F: FieldElement> {
    data: Vec<F>, // The encoded input
    param: &'a CountParam<F>,
}

impl<'a, F: FieldElement> Count<'a, F> {
    /// Construct a new counter.
    pub fn new(value: u64, param: &'a CountParam<F>) -> Result<Self, PcpError> {
        Ok(Self {
            data: vec![match value {
                1 => F::one(),
                0 => F::zero(),
                _ => {
                    return Err(PcpError::Value("Count value must be 0 or 1".to_string()));
                }
            }],
            param,
        })
    }
}

impl<'a, F: FieldElement> Value<'a> for Count<'a, F> {
    type Field = F;
    type Param = CountParam<F>;

    fn new_share(
        data: Vec<F>,
        param: &'a CountParam<F>,
        _num_shares: usize,
    ) -> Result<Self, PcpError> {
        Ok(Self { data, param })
    }

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        valid_call_check(self, rand)?;

        if self.data.len() != 1 {
            return Err(PcpError::Valid(format!(
                "unexpected input length: got {}; want {}",
                self.data.len(),
                1
            )));
        }

        let mut inp = [self.data[0], self.data[0]];
        let mut v = self.param.range_checker[0];
        for c in &self.param.range_checker[1..] {
            v += *c * inp[0];
            inp[0] = g[0].call(&inp)?;
        }

        Ok(v)
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn param(&self) -> &CountParam<F> {
        self.param
    }

    fn into_output(self) -> Vec<F> {
        self.data
    }
}

/// Parameters for the [`Count`] type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountParam<F> {
    range_checker: Vec<F>,
}

impl<F: FieldElement> CountParam<F> {
    /// Return a new [`Count`] type parameter.
    pub fn new() -> Self {
        Self {
            range_checker: poly_range_check(0, 2),
        }
    }
}

impl<F: FieldElement> ValueParam<F> for CountParam<F> {
    fn input_len(&self) -> usize {
        1
    }

    fn proof_len(&self) -> usize {
        9
    }

    fn verifier_len(&self) -> usize {
        4
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
}

/// This sum type. Each measurement is a integer in `[0, 2^bits)` and the aggregate is the sum of the measurements.
///
/// The validity circuit is based on the SIMD circuit construction of [[BBCG+19], Theorem 5.3].
///
/// [BBCG+19]: https://ia.cr/2019/188
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sum<'a, F: FieldElement> {
    data: Vec<F>,
    param: &'a SumParam<F>,
}

impl<'a, F: FieldElement> Sum<'a, F> {
    /// Constructs a new summand. The value of `summand` must be in `[0, 2^bits)`.
    pub fn new(summand: u64, param: &'a SumParam<F>) -> Result<Self, PcpError> {
        let summand = usize::try_from(summand).unwrap();
        let bits = usize::try_from(param.bits).unwrap();

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

        Ok(Self { data, param })
    }
}

impl<'a, F: FieldElement> Value<'a> for Sum<'a, F> {
    type Field = F;
    type Param = SumParam<F>;

    fn new_share(
        data: Vec<F>,
        param: &'a SumParam<F>,
        _num_shares: usize,
    ) -> Result<Self, PcpError> {
        if data.len() != param.bits {
            return Err(PcpError::Value(
                "data length does not match bit length".to_string(),
            ));
        }

        Ok(Self { data, param })
    }

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        valid_call_check(self, rand)?;

        // Check that each element of `data` is a 0 or 1.
        let mut range_check = F::zero();
        let mut r = rand[0];
        for chunk in self.data.chunks(1) {
            range_check += r * g[0].call(chunk)?;
            r *= rand[0];
        }

        Ok(range_check)
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn param(&self) -> &SumParam<F> {
        &self.param
    }

    fn into_output(self) -> Vec<F> {
        let mut decoded = F::zero();
        for (l, bit) in self.data.iter().enumerate() {
            let w = F::from(F::Integer::try_from(1 << l).unwrap());
            decoded += w * *bit;
        }
        vec![decoded]
    }
}

/// Parameters for the [`Sum`] type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumParam<F> {
    bits: usize,
    range_checker: Vec<F>,
}

impl<F: FieldElement> SumParam<F> {
    /// Return a new [`Sum`] type parameter. Each value of this type is an integer in range `[0,
    /// 2^bits)`.
    pub fn new(bits: usize) -> Self {
        Self {
            bits,
            range_checker: poly_range_check(0, 2),
        }
    }
}

impl<F: FieldElement> ValueParam<F> for SumParam<F> {
    fn input_len(&self) -> usize {
        self.bits
    }

    fn proof_len(&self) -> usize {
        2 * ((1 + self.bits).next_power_of_two() - 1) + 2
    }

    fn verifier_len(&self) -> usize {
        3
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
            self.bits,
        ))]
    }
}

/// The histogram type. Each measurement is a non-negative integer and the aggregate is a histogram
/// approximating the distribution of the measurements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram<'a, F> {
    data: Vec<F>,
    sum_check_share: F,
    param: &'a HistogramParam<F>,
}

impl<'a, F: FieldElement> Histogram<'a, F> {
    /// Constructs a new histogram input. The values of `buckets` must be strictly increasing.
    pub fn new(measurement: u64, param: &'a HistogramParam<F>) -> Result<Self, PcpError> {
        let buckets = &param.buckets;
        let mut data = vec![F::zero(); buckets.len() + 1];

        if buckets.len() >= u32::MAX as usize {
            return Err(PcpError::Value(
                "invalid buckets: number of buckets exceeds maximum permitted".to_string(),
            ));
        }

        if !buckets.is_empty() {
            for i in 0..buckets.len() - 1 {
                if buckets[i + 1] <= buckets[i] {
                    return Err(PcpError::Value(
                        "invalid buckets: out-of-order boundary".to_string(),
                    ));
                }
            }
        }

        let bucket = match buckets.binary_search(&measurement) {
            Ok(i) => i,  // on a bucket boundary
            Err(i) => i, // smaller than the i-th bucket boundary
        };

        data[bucket] = F::one();

        Ok(Self {
            data,
            sum_check_share: F::one(),
            param,
        })
    }
}

impl<'a, F: FieldElement> Value<'a> for Histogram<'a, F> {
    type Field = F;
    type Param = HistogramParam<F>;

    fn new_share(
        data: Vec<F>,
        param: &'a HistogramParam<F>,
        num_shares: usize,
    ) -> Result<Self, PcpError> {
        if data.len() != param.buckets.len() + 1 {
            return Err(PcpError::Value(
                "data length does not match buckets".to_string(),
            ));
        }

        let sum_check_share = F::one() / F::from(F::Integer::try_from(num_shares).unwrap());
        Ok(Self {
            data: data.to_vec(),
            sum_check_share,
            param,
        })
    }

    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, rand: &[F]) -> Result<F, PcpError> {
        valid_call_check(self, rand)?;

        // Check that each element of `data` is a 0 or 1.
        let mut range_check = F::zero();
        let mut r = rand[0];
        for chunk in self.data.chunks(1) {
            range_check += r * g[0].call(chunk)?;
            r *= rand[0];
        }

        // Check that the elements of `data` sum to 1.
        let mut sum_check = -self.sum_check_share;
        for val in self.data.iter() {
            sum_check += *val;
        }

        // Take a random linear combination of both checks.
        let out = rand[1] * range_check + (rand[1] * rand[1]) * sum_check;
        Ok(out)
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn param(&self) -> &HistogramParam<F> {
        self.param
    }

    fn into_output(self) -> Vec<F> {
        self.data
    }
}

/// Parameters for the [`Histogram`] type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistogramParam<F> {
    buckets: Vec<u64>,
    range_checker: Vec<F>,
}

impl<F: FieldElement> HistogramParam<F> {
    /// Return a new [`Histogram`] type parameter with the given buckets.
    pub fn new(buckets: Vec<u64>) -> Self {
        Self {
            buckets,
            range_checker: poly_range_check(0, 2),
        }
    }
}

impl<F: FieldElement> ValueParam<F> for HistogramParam<F> {
    fn input_len(&self) -> usize {
        self.buckets.len() + 1
    }

    fn proof_len(&self) -> usize {
        2 * ((1 + self.input_len()).next_power_of_two() - 1) + 2
    }

    fn verifier_len(&self) -> usize {
        3
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

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(PolyEval::new(
            self.range_checker.to_vec(),
            self.input_len(),
        ))]
    }
}

fn valid_call_check<'a, V: Value<'a>>(input: &V, joint_rand: &[V::Field]) -> Result<(), PcpError> {
    let param = input.param();

    if input.as_slice().len() != param.input_len() {
        return Err(PcpError::Valid(format!(
            "unexpected input length: got {}; want {}",
            input.as_slice().len(),
            param.input_len(),
        )));
    }

    if joint_rand.len() != param.joint_rand_len() {
        return Err(PcpError::Valid(format!(
            "unexpected joint randomness length: got {}; want {}",
            joint_rand.len(),
            param.joint_rand_len()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{random_vector, split_vector, Field64 as TestField};
    use crate::pcp::{decide, prove, query, Proof, Value, Verifier};

    // Number of shares to split input and proofs into in `pcp_test`.
    const NUM_SHARES: usize = 3;

    struct ValidityTestCase<F> {
        expect_valid: bool,
        expected_output: Option<Vec<F>>,
    }

    #[test]
    fn test_count() {
        let param = &CountParam::new();
        let zero = TestField::zero();
        let one = TestField::one();

        // Test PCP on valid input.
        pcp_validity_test(
            &Count::<TestField>::new(1, param).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![one]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Count::<TestField>::new(0, param).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![zero]),
            },
        )
        .unwrap();

        // Test PCP on invalid input.
        pcp_validity_test(
            &Count {
                data: vec![TestField::from(1337)],
                param,
            },
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        // Try running the validity circuit on an input that's too short.
        let malformed_x = Count::<TestField> {
            data: vec![],
            param,
        };
        malformed_x.valid(&mut param.gadget(), &[]).unwrap_err();

        // Try running the validity circuit on an input that's too large.
        let malformed_x = Count::<TestField> {
            data: vec![TestField::zero(), TestField::zero()],
            param,
        };
        malformed_x.valid(&mut param.gadget(), &[]).unwrap_err();
    }

    #[test]
    fn test_sum() {
        let param = &SumParam::new(11);
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        // Test PCP on valid input.
        pcp_validity_test(
            &Sum::<TestField>::new(1337, param).unwrap(),
            &ValidityTestCase {
                expect_valid: true,
                expected_output: Some(vec![TestField::from(1337)]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![],
                param: &SumParam::new(0),
            },
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![zero]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![one, zero],
                param: &SumParam::new(2),
            },
            &ValidityTestCase {
                expect_valid: true,
                expected_output: Some(vec![one]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![one, zero, one, one, zero, one, one, one, zero],
                param: &SumParam::new(9),
            },
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![TestField::from(237)]),
            },
        )
        .unwrap();

        // Test PCP on invalid input.
        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![one, nine, zero],
                param: &SumParam::new(3),
            },
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Sum::<TestField> {
                data: vec![zero, zero, zero, zero, nine],
                param: &SumParam::new(5),
            },
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_histogram() {
        let param = &HistogramParam::new(vec![10, 20]);
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        let input: Histogram<TestField> = Histogram::new(7, param).unwrap();
        assert_eq!(input.data, &[one, zero, zero]);

        let input: Histogram<TestField> = Histogram::new(10, param).unwrap();
        assert_eq!(input.data, &[one, zero, zero]);

        let input: Histogram<TestField> = Histogram::new(17, param).unwrap();
        assert_eq!(input.data, &[zero, one, zero]);

        let input: Histogram<TestField> = Histogram::new(20, param).unwrap();
        assert_eq!(input.data, &[zero, one, zero]);

        let input: Histogram<TestField> = Histogram::new(27, param).unwrap();
        assert_eq!(input.data, &[zero, zero, one]);

        // Invalid bucket boundaries.
        Histogram::<TestField>::new(27, &HistogramParam::new(vec![10, 0])).unwrap_err();
        Histogram::<TestField>::new(27, &HistogramParam::new(vec![10, 10])).unwrap_err();

        // Test valid inputs.
        pcp_validity_test(
            &Histogram::<TestField>::new(0, param).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![one, zero, zero]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Histogram::<TestField>::new(17, param).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![zero, one, zero]),
            },
        )
        .unwrap();

        pcp_validity_test(
            &Histogram::<TestField>::new(1337, param).unwrap(),
            &ValidityTestCase::<TestField> {
                expect_valid: true,
                expected_output: Some(vec![zero, zero, one]),
            },
        )
        .unwrap();

        // Test invalid inputs.
        pcp_validity_test(
            &Histogram::<TestField> {
                data: vec![zero, zero, nine],
                param,
                sum_check_share: one,
            },
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Histogram::<TestField> {
                data: vec![zero, one, one],
                param,
                sum_check_share: one,
            },
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Histogram::<TestField> {
                data: vec![one, one, one],
                param,
                sum_check_share: one,
            },
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();

        pcp_validity_test(
            &Histogram::<TestField> {
                data: vec![zero, zero, zero],
                param,
                sum_check_share: one,
            },
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: None,
            },
        )
        .unwrap();
    }

    fn pcp_validity_test<'a, V: Value<'a>>(
        input: &'a V,
        t: &ValidityTestCase<V::Field>,
    ) -> Result<(), PcpError> {
        let param = input.param();
        let mut gadgets = param.gadget();

        if input.as_slice().len() != param.input_len() {
            return Err(PcpError::Test(format!(
                "unexpected input length: got {}; want {}",
                input.as_slice().len(),
                param.input_len()
            )));
        }

        if param.query_rand_len() != gadgets.len() {
            return Err(PcpError::Test(format!(
                "query rand length: got {}; want {}",
                param.query_rand_len(),
                gadgets.len()
            )));
        }

        // Ensure that the input can be constructed from its parameters and its encoding as a
        // sequence of field elements.
        let got = &V::new_share(input.as_slice().to_vec(), input.param(), 1)?;
        if got != input {
            return Err(PcpError::Test(format!(
                "input constructed from data and param does not match input: got {:?}; want {:?}",
                got, input
            )));
        }

        let joint_rand = random_vector(param.joint_rand_len()).unwrap();
        let prove_rand = random_vector(param.prove_rand_len()).unwrap();
        let query_rand = random_vector(param.query_rand_len()).unwrap();

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
        if proof.as_slice().len() != param.proof_len() {
            return Err(PcpError::Test(format!(
                "unexpected proof length: got {}; want {}",
                proof.as_slice().len(),
                param.proof_len()
            )));
        }

        // Query the proof.
        let verifier = query(input, &proof, &query_rand, &joint_rand)?;
        if verifier.as_slice().len() != param.verifier_len() {
            return Err(PcpError::Test(format!(
                "unexpected verifier length: got {}; want {}",
                verifier.as_slice().len(),
                param.verifier_len()
            )));
        }

        // Decide if the input is valid.
        let res = decide(param, &verifier)?;
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
            .map(|data| V::new_share(data.clone(), input.param(), NUM_SHARES).unwrap())
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

        let res = decide(param, &verifier)?;
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
            if decide(param, &verifier)? {
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
                "query on long proof succeeded; want failure",
            )));
        }

        if let Some(ref want) = t.expected_output {
            let got = input.clone().into_output();
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
