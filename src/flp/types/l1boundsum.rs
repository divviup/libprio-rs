//! Implementation of the L1BoundSum FLP described in [draft-ietf-ppm-l1-bound-sum][1].
//!
//! [1]: https://ietf-wg-ppm.github.io/draft-ietf-ppm-l1-bound-sum/draft-ietf-ppm-l1-bound-sum.html

use crate::{
    field::NttFriendlyFieldElement,
    flp::{
        gadgets::{Mul, ParallelSumGadget},
        types::{decode_result_vec, parallel_sum_range_checks},
        Flp, FlpError, Gadget, Type,
    },
};
use std::{fmt::Debug, marker::PhantomData};

/// Implementation of the L1BoundSum FLP described in [draft-ietf-ppm-l1-bound-sum][1]. L1BoundSum
/// is very similar to SumVec, except that it also checks that the L1 norm of the measurement is
/// within a bound included in the secret shared input.
///
/// [1]: https://ietf-wg-ppm.github.io/draft-ietf-ppm-l1-bound-sum/draft-ietf-ppm-l1-bound-sum.html
#[derive(Copy, PartialEq, Eq)]
pub struct L1BoundSum<F: NttFriendlyFieldElement, S> {
    /// Number of field elements in a measurement. Does not include the element representing the
    /// L1 norm.
    measurement_len: usize,
    /// Size in bits of each element of the measurement.
    bits: usize,
    /// Length in bits of an encoded measurement, including the L1 norm.
    measurement_len_in_bits: usize,
    /// Chunk length for ParallelSum gadget.
    chunk_length: usize,
    /// Number of gadget calls needed to evaluate validity.
    gadget_calls: usize,

    phantom: PhantomData<(S, F)>,
}

impl<F: NttFriendlyFieldElement, S> Debug for L1BoundSum<F, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L1BoundSum")
            .field("measurement_len", &self.measurement_len)
            .field("bits", &self.bits)
            .field("measurement_len_in_bits", &self.measurement_len_in_bits)
            .field("chunk_length", &self.chunk_length)
            .field("gadget_calls", &self.gadget_calls)
            .finish()
    }
}

impl<F: NttFriendlyFieldElement, S: ParallelSumGadget<F, Mul<F>>> L1BoundSum<F, S> {
    /// Create a new instance of the FLP.
    pub fn new(bits: usize, measurement_len: usize, chunk_length: usize) -> Result<Self, FlpError> {
        let measurement_len_in_bits = bits.checked_mul(measurement_len + 1).ok_or_else(|| {
            FlpError::InvalidParameter("bits*measurement_len overflows addressable memory".into())
        })?;

        // Check if the bit width is too large. This limit is defined to be one bit less than the
        // number of bits required to encode `F::Integer`. (One less so that we can compute `1 <<
        // bits` without overflowing.)
        let limit = std::mem::size_of::<F::Integer>() * 8 - 1;
        if bits > limit {
            return Err(FlpError::InvalidParameter(format!(
                "bit width exceeds limit of {limit}"
            )));
        }

        // Check for degenerate parameters.
        if bits == 0 {
            return Err(FlpError::InvalidParameter(
                "bits cannot be zero".to_string(),
            ));
        }
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

        let mut gadget_calls = measurement_len_in_bits / chunk_length;
        if measurement_len_in_bits % chunk_length != 0 {
            gadget_calls += 1;
        }

        Ok(Self {
            measurement_len,
            bits,
            measurement_len_in_bits,
            chunk_length,
            gadget_calls,
            phantom: PhantomData,
        })
    }

    /// Computes the largest field element for the chosen bit count.
    fn largest_field_element(&self) -> F::Integer {
        let one = F::Integer::from(F::one());
        (one << self.bits) - one
    }
}

impl<F: NttFriendlyFieldElement, S> Clone for L1BoundSum<F, S> {
    fn clone(&self) -> Self {
        Self {
            measurement_len: self.measurement_len,
            bits: self.bits,
            measurement_len_in_bits: self.measurement_len_in_bits,
            chunk_length: self.chunk_length,
            gadget_calls: self.gadget_calls,
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

        // The encoded input consists of the measurement with the claimed weight appended, as a
        // bits-length vector of field elements. The observed weight is obtained by summing up all
        // the input elements but the last one.
        let mut observed_weight = F::zero();
        let mut claimed_weight = F::zero();
        for (index, element) in input.chunks(self.bits).enumerate() {
            let decoded = F::decode_bitvector(element)?;
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
    /// The measurement is the input represented as a vector of field elements, without any norm
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
            if summand > &self.largest_field_element() {
                return Err(FlpError::Encode(format!(
                    "summand exceeds maximum of 2^{}-1",
                    self.bits
                )));
            }

            // Accumulate measurement elements into L1 norm.
            l1_norm = l1_norm + *summand;
            flattened.extend(F::encode_as_bitvector(*summand, self.bits)?);
        }

        flattened.extend(F::encode_as_bitvector(l1_norm, self.bits)?);

        Ok(flattened)
    }

    fn truncate(&self, input: Vec<Self::Field>) -> Result<Vec<Self::Field>, FlpError> {
        self.truncate_call_check(&input)?;

        // Truncation removes the L1 norm, so skip the last element of the input.
        let mut truncated = Vec::with_capacity(self.measurement_len);
        for chunk in input.chunks(self.bits).take(self.measurement_len) {
            truncated.push(F::decode_bitvector(chunk)?);
        }

        Ok(truncated)
    }

    fn decode_result(
        &self,
        data: &[Self::Field],
        _num_measurements: usize,
    ) -> Result<Self::AggregateResult, FlpError> {
        // The claimed weights were removed during truncation, so there's no special handling to
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
        field::{
            Field128, Field64, FieldElement, FieldElementWithInteger, FieldElementWithIntegerExt,
        },
        flp::{gadgets::ParallelSum, test_utils::TypeTest},
    };

    use super::*;

    fn roundtrip_encoding_with_field<F: NttFriendlyFieldElement>() {
        let l1_bound_sum = L1BoundSum::<F, ParallelSum<F, Mul<F>>>::new(3, 4, 3).unwrap();

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
            L1BoundSum::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(3, 4, 3).unwrap();

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
    }

    #[test]
    fn invalid_measurements() {
        let bits = 3;
        let l1_bound_sum =
            L1BoundSum::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(bits, 4, 3).unwrap();

        for measurement in [
            [
                Field128::encode_as_bitvector(7, bits).unwrap(),
                Field128::encode_as_bitvector(0, bits).unwrap(),
                Field128::encode_as_bitvector(0, bits).unwrap(),
                Field128::encode_as_bitvector(0, bits).unwrap(),
                Field128::encode_as_bitvector(6, bits).unwrap(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
            [
                vec![
                    Field128::from(2),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                    Field128::zero(),
                ],
                Field128::encode_as_bitvector(2, bits)
                    .unwrap()
                    .collect::<Vec<_>>(),
            ]
            .into_iter()
            .flatten()
            .collect(),
            vec![
                // First vector element: 2
                Field128::zero(),
                Field128::one(),
                Field128::zero(),
                // Rest of vector elements: 0
                Field128::zero(),
                Field128::zero(),
                Field128::zero(),
                Field128::zero(),
                Field128::zero(),
                Field128::zero(),
                Field128::zero(),
                Field128::zero(),
                Field128::zero(),
                // Improperly encoded weight
                Field128::from(2),
                Field128::zero(),
                Field128::zero(),
            ],
        ] {
            TypeTest::expect_invalid::<2>(&l1_bound_sum, &measurement);
        }
    }
}
