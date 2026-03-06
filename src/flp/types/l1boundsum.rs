//! Implementation of the L1BoundSum FLP described in [draft-ietf-ppm-l1-bound-sum-01][1].
//!
//! [1]: https://www.ietf.org/archive/id/draft-ietf-ppm-l1-bound-sum-01.html

use crate::{
    field::{Integer, NttFriendlyFieldElement},
    flp::{
        gadgets::{Mul, ParallelSumGadget},
        types::{
            decode_range_checked_int, decode_result_vec, encode_range_checked_int,
            parallel_sum_range_checks,
        },
        Flp, FlpError, Gadget, Type,
    },
};
use std::{fmt::Debug, marker::PhantomData};

/// Implementation of the L1BoundSum FLP described in [draft-ietf-ppm-l1-bound-sum-01][1].
/// L1BoundSum is very similar to SumVec, except that it also checks that the L1 norm of the
/// measurement is within a bound, using extra information included in the secret shared input.
///
/// [1]: https://www.ietf.org/archive/id/draft-ietf-ppm-l1-bound-sum-01.html
#[derive(Copy, PartialEq, Eq)]
pub struct L1BoundSum<F: NttFriendlyFieldElement, S> {
    /// Number of field elements in a measurement. Does not include the elements representing the
    /// L1 norm.
    measurement_len: usize,
    /// Maximum allowed value for each element of the measurement, and for the L1 norm.
    pub(super) max_value: F::Integer,
    /// Size in bits of each element of the measurement.
    bits: usize,
    /// Length in bits of an encoded measurement, including the L1 norm.
    measurement_len_in_bits: usize,
    /// Chunk length for ParallelSum gadget.
    chunk_length: usize,
    /// Number of gadget calls needed to evaluate validity.
    gadget_calls: usize,
    /// Weight of the last digit in modified bit vector encoding.
    last_weight: F::Integer,
    /// Projection of `last_weight` into the field.
    last_weight_field: F,

    phantom: PhantomData<(S, F)>,
}

impl<F: NttFriendlyFieldElement, S> Debug for L1BoundSum<F, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L1BoundSum")
            .field("measurement_len", &self.measurement_len)
            .field("max_value", &self.max_value)
            .field("bits", &self.bits)
            .field("chunk_length", &self.chunk_length)
            .finish()
    }
}

impl<F: NttFriendlyFieldElement, S: ParallelSumGadget<F, Mul<F>>> L1BoundSum<F, S> {
    /// Create a new instance of the FLP.
    pub fn new(
        max_value: F::Integer,
        measurement_len: usize,
        chunk_length: usize,
    ) -> Result<Self, FlpError> {
        // Check for degenerate parameters.
        if measurement_len == 0 {
            return Err(FlpError::InvalidParameter(
                "measurement_len cannot be zero".to_string(),
            ));
        }
        if chunk_length == 0 {
            return Err(FlpError::InvalidParameter(
                "chunk_length cannot be zero".to_string(),
            ));
        }
        if max_value <= F::Integer::zero() {
            return Err(FlpError::InvalidParameter(
                "max_value must be positive".to_string(),
            ));
        }
        if max_value >= F::modulus() {
            return Err(FlpError::InvalidParameter(
                "max_value exceeds field modulus".to_string(),
            ));
        }

        // Number of bits needed to represent each value.
        let bits = max_value.checked_ilog2().unwrap() as usize + 1;

        let measurement_len_in_bits = bits.checked_mul(measurement_len + 1).ok_or_else(|| {
            FlpError::InvalidParameter(
                "bits*(measurement_len+1) overflows addressable memory".into(),
            )
        })?;

        let last_weight = max_value - ((F::Integer::one() << (bits - 1)) - F::Integer::one());
        let last_weight_field = F::from(last_weight);

        let mut gadget_calls = measurement_len_in_bits / chunk_length;
        if measurement_len_in_bits % chunk_length != 0 {
            gadget_calls += 1;
        }

        Ok(Self {
            measurement_len,
            max_value,
            bits,
            measurement_len_in_bits,
            chunk_length,
            gadget_calls,
            last_weight,
            last_weight_field,
            phantom: PhantomData,
        })
    }
}

impl<F: NttFriendlyFieldElement, S> Clone for L1BoundSum<F, S> {
    fn clone(&self) -> Self {
        Self {
            measurement_len: self.measurement_len,
            max_value: self.max_value,
            bits: self.bits,
            measurement_len_in_bits: self.measurement_len_in_bits,
            chunk_length: self.chunk_length,
            gadget_calls: self.gadget_calls,
            last_weight: self.last_weight,
            last_weight_field: self.last_weight_field,
            phantom: PhantomData,
        }
    }
}

impl<F, S> Flp for L1BoundSum<F, S>
where
    F: NttFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
{
    type Field = F;

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(S::new(
            Mul::new(self.gadget_calls),
            self.chunk_length,
        ))]
    }

    fn num_gadgets(&self) -> usize {
        1
    }

    fn valid(
        &self,
        gadget: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_randomness: &[F],
        num_shares: usize,
    ) -> Result<Vec<F>, FlpError> {
        self.valid_call_check(input, joint_randomness)?;

        let range_check = parallel_sum_range_checks(
            &mut gadget[0],
            input,
            joint_randomness,
            self.chunk_length,
            num_shares,
        )?;

        // The encoded input consists of the measurement with the claimed weight appended, with each
        // value encoded as a bits-length vector of field elements. The observed weight is obtained
        // by summing up all the input elements but the last one.
        let mut observed_weight = F::zero();
        let mut claimed_weight = F::zero();
        for (index, elements) in input.chunks(self.bits).enumerate() {
            let decoded = decode_range_checked_int(elements, self.last_weight_field)?;
            if index == self.measurement_len {
                // Last element: this is the claimed weight.
                claimed_weight += decoded;
            } else {
                observed_weight += decoded;
            }
        }

        Ok(vec![range_check, observed_weight - claimed_weight])
    }

    fn input_len(&self) -> usize {
        self.measurement_len_in_bits
    }

    fn proof_len(&self) -> usize {
        (self.chunk_length * 2) + 2 * ((1 + self.gadget_calls).next_power_of_two() - 1) + 1
    }

    fn verifier_len(&self) -> usize {
        2 + self.chunk_length * 2
    }

    fn joint_rand_len(&self) -> usize {
        self.gadget_calls
    }

    fn eval_output_len(&self) -> usize {
        2
    }

    fn prove_rand_len(&self) -> usize {
        self.chunk_length * 2
    }
}

impl<F, S> Type for L1BoundSum<F, S>
where
    F: NttFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
{
    /// The measurement is the input represented as a vector of integers, without any norm
    /// appended.
    type Measurement = Vec<F::Integer>;

    /// The aggregate result is the element-wise sum of the inputs represented as a vector of field
    /// elements, without any norm appended.
    type AggregateResult = Vec<F::Integer>;

    fn encode_measurement(
        &self,
        measurement: &Self::Measurement,
    ) -> Result<Vec<Self::Field>, FlpError> {
        // The measurement encoding is the elements of the measurement with the claimed weight (the
        // L1 norm) appended at the end.
        if measurement.len() != self.measurement_len {
            return Err(FlpError::Encode(format!(
                "unexpected measurement length: got {}; want {}",
                measurement.len(),
                self.measurement_len
            )));
        }

        let mut flattened = Vec::with_capacity(self.measurement_len_in_bits);
        let mut l1_norm = F::Integer::from(F::zero());
        for summand in measurement {
            encode_range_checked_int(*summand, self.bits, self.last_weight, &mut flattened)?;

            // Accumulate measurement elements into L1 norm.
            l1_norm = l1_norm
                .checked_add(*summand)
                .ok_or_else(|| FlpError::Encode("L1 norm of measurement overflowed".to_string()))?;
        }

        encode_range_checked_int(l1_norm, self.bits, self.last_weight, &mut flattened)?;

        Ok(flattened)
    }

    fn truncate(&self, input: Vec<Self::Field>) -> Result<Vec<Self::Field>, FlpError> {
        self.truncate_call_check(&input)?;

        // Truncation removes the L1 norm, so skip the last element of the input.
        let mut truncated = Vec::with_capacity(self.measurement_len);
        for chunk in input.chunks(self.bits).take(self.measurement_len) {
            truncated.push(decode_range_checked_int(chunk, self.last_weight_field)?);
        }

        Ok(truncated)
    }

    fn decode_result(
        &self,
        data: &[Self::Field],
        _num_measurements: usize,
    ) -> Result<Self::AggregateResult, FlpError> {
        // The claimed weight was removed during truncation, so there's no special handling to
        // decode the aggregation result: it consists solely of the element-wise sum of the input
        // vectors.
        decode_result_vec(data, self.measurement_len)
    }

    fn output_len(&self) -> usize {
        // Output length is the length of the measurement, without the claimed weight.
        self.measurement_len
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        field::{Field128, Field64, FieldElement, FieldElementWithIntegerExt},
        flp::{gadgets::ParallelSum, test_utils::TypeTest},
    };

    use super::*;

    fn roundtrip_encoding_with_field<F: NttFriendlyFieldElement>() {
        let l1_bound_sum =
            L1BoundSum::<F, ParallelSum<F, Mul<F>>>::new(7.try_into().unwrap(), 4, 3).unwrap();

        for measurement in [
            [7usize, 0, 0, 0],
            [0, 0, 0, 7],
            [1, 2, 2, 1],
            [2, 1, 2, 2],
            [5, 2, 0, 0],
            [0, 0, 0, 0],
        ] {
            let measurement: Vec<_> = measurement
                .into_iter()
                .map(|m| F::valid_integer_try_from(m).unwrap())
                .collect();
            let encoded = l1_bound_sum.encode_measurement(&measurement).unwrap();
            assert_eq!(encoded.len(), (measurement.len() + 1) * 3);

            let decoded = l1_bound_sum
                .decode_result(&l1_bound_sum.truncate(encoded).unwrap(), 1)
                .unwrap();
            assert_eq!(measurement, decoded);
        }

        let l1_bound_sum =
            L1BoundSum::<F, ParallelSum<F, Mul<F>>>::new(4.try_into().unwrap(), 4, 3).unwrap();
        for measurement in [
            [4usize, 0, 0, 0],
            [0, 0, 0, 4],
            [1, 1, 1, 1],
            [2, 0, 2, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ] {
            let measurement: Vec<_> = measurement
                .into_iter()
                .map(|m| F::valid_integer_try_from(m).unwrap())
                .collect();
            let encoded = l1_bound_sum.encode_measurement(&measurement).unwrap();
            assert_eq!(encoded.len(), (measurement.len() + 1) * 3);

            let decoded = l1_bound_sum
                .decode_result(&l1_bound_sum.truncate(encoded).unwrap(), 1)
                .unwrap();
            assert_eq!(measurement, decoded);
        }
    }

    #[test]
    fn roundtrip_encoding_field128() {
        roundtrip_encoding_with_field::<Field128>()
    }

    #[test]
    fn roundtrip_encoding_field64() {
        roundtrip_encoding_with_field::<Field64>()
    }

    #[test]
    fn valid_measurements() {
        let l1_bound_sum =
            L1BoundSum::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(7, 4, 3).unwrap();

        for measurement in [
            [7, 0, 0, 0],
            [0, 0, 0, 7],
            [2, 2, 2, 1],
            [0, 0, 0, 0],
            [2, 2, 2, 0],
            [1, 0, 2, 0],
        ] {
            TypeTest::expect_valid_no_output::<2>(
                &l1_bound_sum,
                &l1_bound_sum
                    .encode_measurement(&measurement.to_vec())
                    .unwrap(),
            );
        }

        let l1_bound_sum =
            L1BoundSum::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(4, 4, 3).unwrap();
        for measurement in [
            [4, 0, 0, 0],
            [0, 0, 0, 4],
            [1, 1, 1, 1],
            [2, 0, 2, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ] {
            TypeTest::expect_valid_no_output::<2>(
                &l1_bound_sum,
                &l1_bound_sum
                    .encode_measurement(&measurement.to_vec())
                    .unwrap(),
            );
        }
    }

    /// Manually construct some improperly encoded measurements, and confirm that the FLP rejects
    /// them.
    ///
    /// The test cases exercise different constraints imposed by the validity circuit.
    #[test]
    fn invalid_measurements() {
        let l1_bound_sum =
            L1BoundSum::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(6, 4, 3).unwrap();
        let bits = l1_bound_sum.bits;

        let mut measurement_1 = Vec::new();
        encode_range_checked_int(6, bits, l1_bound_sum.last_weight, &mut measurement_1).unwrap();
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_1).unwrap();
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_1).unwrap();
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_1).unwrap();
        // Claimed weight is inconsistent with the rest of the measurement.
        encode_range_checked_int(5, bits, l1_bound_sum.last_weight, &mut measurement_1).unwrap();

        let mut measurement_2 = Vec::new();
        // Improperly encoded measurement element
        measurement_2.push(Field128::from(2));
        measurement_2.push(Field128::zero());
        measurement_2.push(Field128::zero());
        // Rest of vector elements: 0
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_2).unwrap();
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_2).unwrap();
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_2).unwrap();
        // Weight: 2
        encode_range_checked_int(2, bits, l1_bound_sum.last_weight, &mut measurement_2).unwrap();

        let mut measurement_3 = Vec::new();
        // First vector element: 2
        encode_range_checked_int(2, bits, l1_bound_sum.last_weight, &mut measurement_3).unwrap();
        // Rest of vector elements: 0
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_3).unwrap();
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_3).unwrap();
        encode_range_checked_int(0, bits, l1_bound_sum.last_weight, &mut measurement_3).unwrap();
        // Improperly encoded weight
        measurement_3.push(Field128::from(2));
        measurement_3.push(Field128::zero());
        measurement_3.push(Field128::zero());

        for measurement in [measurement_1, measurement_2, measurement_3] {
            TypeTest::expect_invalid::<2>(&l1_bound_sum, &measurement);
        }
    }
}
