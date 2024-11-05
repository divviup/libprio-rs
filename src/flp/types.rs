// SPDX-License-Identifier: MPL-2.0

//! A collection of [`Type`] implementations.

use crate::field::{FftFriendlyFieldElement, FieldElementWithIntegerExt};
use crate::flp::gadgets::{Mul, ParallelSumGadget, PolyEval};
use crate::flp::{FlpError, Gadget, Type};
use crate::polynomial::poly_range_check;
use std::convert::TryInto;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use subtle::Choice;

#[cfg(feature = "experimental")]
mod dp;

/// The counter data type. Each measurement is `0` or `1` and the aggregate result is the sum of the
/// measurements (i.e., the total number of `1s`).
#[derive(Clone, PartialEq, Eq)]
pub struct Count<F> {
    _phantom: PhantomData<F>,
}

impl<F> Debug for Count<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Count").finish()
    }
}

impl<F: FftFriendlyFieldElement> Count<F> {
    /// Return a new [`Count`] type instance.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: FftFriendlyFieldElement> Default for Count<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FftFriendlyFieldElement> Type for Count<F> {
    type Measurement = bool;
    type AggregateResult = F::Integer;
    type Field = F;

    fn encode_measurement(&self, value: &bool) -> Result<Vec<F>, FlpError> {
        Ok(vec![F::conditional_select(
            &F::zero(),
            &F::one(),
            Choice::from(u8::from(*value)),
        )])
    }

    fn decode_result(&self, data: &[F], _num_measurements: usize) -> Result<F::Integer, FlpError> {
        decode_result(data)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(Mul::new(1))]
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        _num_shares: usize,
    ) -> Result<F, FlpError> {
        self.valid_call_check(input, joint_rand)?;
        Ok(g[0].call(&[input[0], input[0]])? - input[0])
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        Ok(input)
    }

    fn input_len(&self) -> usize {
        1
    }

    fn proof_len(&self) -> usize {
        5
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

/// This sum type. Each measurement is a integer in `[0, 2^bits)` and the aggregate is the sum of
/// the measurements.
///
/// The validity circuit is based on the SIMD circuit construction of [[BBCG+19], Theorem 5.3].
///
/// [BBCG+19]: https://ia.cr/2019/188
#[derive(Clone, PartialEq, Eq)]
pub struct Sum<F: FftFriendlyFieldElement> {
    bits: usize,
    range_checker: Vec<F>,
}

impl<F: FftFriendlyFieldElement> Debug for Sum<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sum").field("bits", &self.bits).finish()
    }
}

impl<F: FftFriendlyFieldElement> Sum<F> {
    /// Return a new [`Sum`] type parameter. Each value of this type is an integer in range `[0,
    /// 2^bits)`.
    pub fn new(bits: usize) -> Result<Self, FlpError> {
        if !F::valid_integer_bitlength(bits) {
            return Err(FlpError::Encode(
                "invalid bits: number of bits exceeds maximum number of bits in this field"
                    .to_string(),
            ));
        }
        Ok(Self {
            bits,
            range_checker: poly_range_check(0, 2),
        })
    }
}

impl<F: FftFriendlyFieldElement> Type for Sum<F> {
    type Measurement = F::Integer;
    type AggregateResult = F::Integer;
    type Field = F;

    fn encode_measurement(&self, summand: &F::Integer) -> Result<Vec<F>, FlpError> {
        let v = F::encode_as_bitvector(*summand, self.bits)?.collect();
        Ok(v)
    }

    fn decode_result(&self, data: &[F], _num_measurements: usize) -> Result<F::Integer, FlpError> {
        decode_result(data)
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
    ) -> Result<F, FlpError> {
        self.valid_call_check(input, joint_rand)?;
        call_gadget_on_vec_entries(&mut g[0], input, joint_rand[0])
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        let res = F::decode_bitvector(&input)?;
        Ok(vec![res])
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

/// The average type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and the
/// aggregate is the arithmetic average.
// This is just a `Sum` object under the hood. The only difference is that the aggregate result is
// an f64, which we get by dividing by `num_measurements`
#[derive(Clone, PartialEq, Eq)]
pub struct Average<F: FftFriendlyFieldElement> {
    summer: Sum<F>,
}

impl<F: FftFriendlyFieldElement> Debug for Average<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Average")
            .field("bits", &self.summer.bits)
            .finish()
    }
}

impl<F: FftFriendlyFieldElement> Average<F> {
    /// Return a new [`Average`] type parameter. Each value of this type is an integer in range `[0,
    /// 2^bits)`.
    pub fn new(bits: usize) -> Result<Self, FlpError> {
        let summer = Sum::new(bits)?;
        Ok(Average { summer })
    }
}

impl<F: FftFriendlyFieldElement> Type for Average<F> {
    type Measurement = F::Integer;
    type AggregateResult = f64;
    type Field = F;

    fn encode_measurement(&self, summand: &F::Integer) -> Result<Vec<F>, FlpError> {
        self.summer.encode_measurement(summand)
    }

    fn decode_result(&self, data: &[F], num_measurements: usize) -> Result<f64, FlpError> {
        // Compute the average from the aggregated sum.
        let sum = self.summer.decode_result(data, num_measurements)?;
        let data: u64 = sum
            .try_into()
            .map_err(|err| FlpError::Decode(format!("failed to convert {sum:?} to u64: {err}",)))?;
        let result = (data as f64) / (num_measurements as f64);
        Ok(result)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        self.summer.gadget()
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, FlpError> {
        self.summer.valid(g, input, joint_rand, num_shares)
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.summer.truncate(input)
    }

    fn input_len(&self) -> usize {
        self.summer.bits
    }

    fn proof_len(&self) -> usize {
        self.summer.proof_len()
    }

    fn verifier_len(&self) -> usize {
        self.summer.verifier_len()
    }

    fn output_len(&self) -> usize {
        self.summer.output_len()
    }

    fn joint_rand_len(&self) -> usize {
        self.summer.joint_rand_len()
    }

    fn prove_rand_len(&self) -> usize {
        self.summer.prove_rand_len()
    }

    fn query_rand_len(&self) -> usize {
        self.summer.query_rand_len()
    }
}

/// The histogram type. Each measurement is an integer in `[0, length)` and the aggregate is a
/// histogram counting the number of occurrences of each measurement.
#[derive(PartialEq, Eq)]
pub struct Histogram<F, S> {
    length: usize,
    chunk_length: usize,
    gadget_calls: usize,
    phantom: PhantomData<(F, S)>,
}

impl<F: FftFriendlyFieldElement, S> Debug for Histogram<F, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Histogram")
            .field("length", &self.length)
            .field("chunk_length", &self.chunk_length)
            .finish()
    }
}

impl<F: FftFriendlyFieldElement, S: ParallelSumGadget<F, Mul<F>>> Histogram<F, S> {
    /// Return a new [`Histogram`] type with the given number of buckets.
    pub fn new(length: usize, chunk_length: usize) -> Result<Self, FlpError> {
        if length >= u32::MAX as usize {
            return Err(FlpError::Encode(
                "invalid length: number of buckets exceeds maximum permitted".to_string(),
            ));
        }
        if length == 0 {
            return Err(FlpError::InvalidParameter(
                "length cannot be zero".to_string(),
            ));
        }
        if chunk_length == 0 {
            return Err(FlpError::InvalidParameter(
                "chunk_length cannot be zero".to_string(),
            ));
        }

        let mut gadget_calls = length / chunk_length;
        if length % chunk_length != 0 {
            gadget_calls += 1;
        }

        Ok(Self {
            length,
            chunk_length,
            gadget_calls,
            phantom: PhantomData,
        })
    }
}

impl<F, S> Clone for Histogram<F, S> {
    fn clone(&self) -> Self {
        Self {
            length: self.length,
            chunk_length: self.chunk_length,
            gadget_calls: self.gadget_calls,
            phantom: self.phantom,
        }
    }
}

impl<F, S> Type for Histogram<F, S>
where
    F: FftFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
{
    type Measurement = usize;
    type AggregateResult = Vec<F::Integer>;
    type Field = F;

    fn encode_measurement(&self, measurement: &usize) -> Result<Vec<F>, FlpError> {
        let mut data = vec![F::zero(); self.length];

        data[*measurement] = F::one();
        Ok(data)
    }

    fn decode_result(
        &self,
        data: &[F],
        _num_measurements: usize,
    ) -> Result<Vec<F::Integer>, FlpError> {
        decode_result_vec(data, self.length)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(S::new(
            Mul::new(self.gadget_calls),
            self.chunk_length,
        ))]
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, FlpError> {
        self.valid_call_check(input, joint_rand)?;

        // Check that each element of `input` is a 0 or 1.
        let range_check = parallel_sum_range_checks(
            &mut g[0],
            input,
            joint_rand[0],
            self.chunk_length,
            num_shares,
        )?;

        // Check that the elements of `input` sum to 1.
        let mut sum_check = -F::from(F::valid_integer_try_from(num_shares)?).inv();
        for val in input.iter() {
            sum_check += *val;
        }

        // Take a random linear combination of both checks.
        let out = joint_rand[1] * range_check + (joint_rand[1] * joint_rand[1]) * sum_check;
        Ok(out)
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        Ok(input)
    }

    fn input_len(&self) -> usize {
        self.length
    }

    fn proof_len(&self) -> usize {
        (self.chunk_length * 2) + 2 * ((1 + self.gadget_calls).next_power_of_two() - 1) + 1
    }

    fn verifier_len(&self) -> usize {
        2 + self.chunk_length * 2
    }

    fn output_len(&self) -> usize {
        self.input_len()
    }

    fn joint_rand_len(&self) -> usize {
        2
    }

    fn prove_rand_len(&self) -> usize {
        self.chunk_length * 2
    }

    fn query_rand_len(&self) -> usize {
        1
    }
}

/// The multihot counter data type. Each measurement is a list of booleans of length `length`, with
/// at most `max_weight` true values, and the aggregate is a histogram counting the number of true
/// values at each position across all measurements.
#[derive(PartialEq, Eq)]
pub struct MultihotCountVec<F, S> {
    // Parameters
    /// The number of elements in the list of booleans
    length: usize,
    /// The max number of permissible `true` values in the list of booleans
    max_weight: usize,
    /// The size of the chunks fed into our gadget calls
    chunk_length: usize,

    // Calculated from parameters
    gadget_calls: usize,
    bits_for_weight: usize,
    offset: usize,
    phantom: PhantomData<(F, S)>,
}

impl<F: FftFriendlyFieldElement, S> Debug for MultihotCountVec<F, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultihotCountVec")
            .field("length", &self.length)
            .field("max_weight", &self.max_weight)
            .field("chunk_length", &self.chunk_length)
            .finish()
    }
}

impl<F: FftFriendlyFieldElement, S: ParallelSumGadget<F, Mul<F>>> MultihotCountVec<F, S> {
    /// Return a new [`MultihotCountVec`] type with the given number of buckets.
    pub fn new(
        num_buckets: usize,
        max_weight: usize,
        chunk_length: usize,
    ) -> Result<Self, FlpError> {
        if num_buckets >= u32::MAX as usize {
            return Err(FlpError::Encode(
                "invalid num_buckets: exceeds maximum permitted".to_string(),
            ));
        }
        if num_buckets == 0 {
            return Err(FlpError::InvalidParameter(
                "num_buckets cannot be zero".to_string(),
            ));
        }
        if chunk_length == 0 {
            return Err(FlpError::InvalidParameter(
                "chunk_length cannot be zero".to_string(),
            ));
        }
        if max_weight == 0 {
            return Err(FlpError::InvalidParameter(
                "max_weight cannot be zero".to_string(),
            ));
        }

        // The bitlength of a measurement is the number of buckets plus the bitlength of the max
        // weight
        let bits_for_weight = max_weight.ilog2() as usize + 1;
        let meas_length = num_buckets + bits_for_weight;

        // Gadget calls is ⌈meas_length / chunk_length⌉
        let gadget_calls = (meas_length + chunk_length - 1) / chunk_length;
        // Offset is 2^max_weight.bitlen() - 1 - max_weight
        let offset = (1 << bits_for_weight) - 1 - max_weight;

        Ok(Self {
            length: num_buckets,
            max_weight,
            chunk_length,
            gadget_calls,
            bits_for_weight,
            offset,
            phantom: PhantomData,
        })
    }
}

// Cannot autoderive clone because it requires F and S to be Clone, which they're not in general
impl<F, S> Clone for MultihotCountVec<F, S> {
    fn clone(&self) -> Self {
        Self {
            length: self.length,
            max_weight: self.max_weight,
            chunk_length: self.chunk_length,
            bits_for_weight: self.bits_for_weight,
            offset: self.offset,
            gadget_calls: self.gadget_calls,
            phantom: self.phantom,
        }
    }
}

impl<F, S> Type for MultihotCountVec<F, S>
where
    F: FftFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
{
    type Measurement = Vec<bool>;
    type AggregateResult = Vec<F::Integer>;
    type Field = F;

    fn encode_measurement(&self, measurement: &Vec<bool>) -> Result<Vec<F>, FlpError> {
        let weight_reported: usize = measurement.iter().filter(|bit| **bit).count();

        if measurement.len() != self.length {
            return Err(FlpError::Encode(format!(
                "unexpected measurement length: got {}; want {}",
                measurement.len(),
                self.length
            )));
        }
        if weight_reported > self.max_weight {
            return Err(FlpError::Encode(format!(
                "unexpected measurement weight: got {}; want ≤{}",
                weight_reported, self.max_weight
            )));
        }

        // Convert bool vector to field elems
        let multihot_vec: Vec<F> = measurement
            .iter()
            // We can unwrap because any Integer type can cast from bool
            .map(|bit| F::from(F::valid_integer_try_from(*bit as usize).unwrap()))
            .collect();

        // Encode the measurement weight in binary (actually, the weight plus some offset)
        let offset_weight_bits = {
            let offset_weight_reported = F::valid_integer_try_from(self.offset + weight_reported)?;
            F::encode_as_bitvector(offset_weight_reported, self.bits_for_weight)?.collect()
        };

        // Report the concat of the two
        Ok([multihot_vec, offset_weight_bits].concat())
    }

    fn decode_result(
        &self,
        data: &[Self::Field],
        _num_measurements: usize,
    ) -> Result<Self::AggregateResult, FlpError> {
        // The aggregate is the same as the decoded result. Just convert to integers
        decode_result_vec(data, self.length)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(S::new(
            Mul::new(self.gadget_calls),
            self.chunk_length,
        ))]
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, FlpError> {
        self.valid_call_check(input, joint_rand)?;

        // Check that each element of `input` is a 0 or 1.
        let range_check = parallel_sum_range_checks(
            &mut g[0],
            input,
            joint_rand[0],
            self.chunk_length,
            num_shares,
        )?;

        // Check that the elements of `input` sum to at most `max_weight`.
        let count_vec = &input[..self.length];
        let weight = count_vec.iter().fold(F::zero(), |a, b| a + *b);
        let offset_weight_reported = F::decode_bitvector(&input[self.length..])?;

        // From spec: weight_check = self.offset*shares_inv + weight - weight_reported
        let weight_check = {
            let offset = F::from(F::valid_integer_try_from(self.offset)?);
            let shares_inv = F::from(F::valid_integer_try_from(num_shares)?).inv();
            offset * shares_inv + weight - offset_weight_reported
        };

        // Take a random linear combination of both checks.
        let out = joint_rand[1] * range_check + (joint_rand[1] * joint_rand[1]) * weight_check;
        Ok(out)
    }

    // Truncates the measurement, removing extra data that was necessary for validity (here, the
    // encoded weight), but not important for aggregation
    fn truncate(&self, input: Vec<Self::Field>) -> Result<Vec<Self::Field>, FlpError> {
        self.truncate_call_check(&input)?;
        // Cut off the encoded weight
        Ok(input[..self.length].to_vec())
    }

    // The length in field elements of the encoded input returned by [`Self::encode_measurement`].
    fn input_len(&self) -> usize {
        self.length + self.bits_for_weight
    }

    fn proof_len(&self) -> usize {
        (self.chunk_length * 2) + 2 * ((1 + self.gadget_calls).next_power_of_two() - 1) + 1
    }

    fn verifier_len(&self) -> usize {
        2 + self.chunk_length * 2
    }

    // The length of the truncated output (i.e., the output of [`Type::truncate`]).
    fn output_len(&self) -> usize {
        self.length
    }

    // The number of random values needed in the validity checks
    fn joint_rand_len(&self) -> usize {
        2
    }

    fn prove_rand_len(&self) -> usize {
        self.chunk_length * 2
    }

    fn query_rand_len(&self) -> usize {
        // TODO: this will need to be increase once draft-10 is implemented and more randomness is
        // necessary due to random linear combination computations
        1
    }
}

/// A sequence of integers in range `[0, 2^bits)`. This type uses a neat trick from [[BBCG+19],
/// Corollary 4.9] to reduce the proof size to roughly the square root of the input size.
///
/// [BBCG+19]: https://eprint.iacr.org/2019/188
#[derive(PartialEq, Eq)]
pub struct SumVec<F: FftFriendlyFieldElement, S> {
    len: usize,
    bits: usize,
    flattened_len: usize,
    max: F::Integer,
    chunk_length: usize,
    gadget_calls: usize,
    phantom: PhantomData<S>,
}

impl<F: FftFriendlyFieldElement, S> Debug for SumVec<F, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SumVec")
            .field("len", &self.len)
            .field("bits", &self.bits)
            .field("chunk_length", &self.chunk_length)
            .finish()
    }
}

impl<F: FftFriendlyFieldElement, S: ParallelSumGadget<F, Mul<F>>> SumVec<F, S> {
    /// Returns a new [`SumVec`] with the desired bit width and vector length.
    ///
    /// # Errors
    ///
    /// * The length of the encoded measurement, i.e., `bits * len`, overflows addressable memory.
    /// * The bit width cannot be encoded, i.e., `bits` is larger than or equal to the number of
    ///   bits required to encode field elements.
    /// * Any of `bits`, `len`, or `chunk_length` are zero.
    pub fn new(bits: usize, len: usize, chunk_length: usize) -> Result<Self, FlpError> {
        let flattened_len = bits.checked_mul(len).ok_or_else(|| {
            FlpError::InvalidParameter("`bits*len` overflows addressable memory".into())
        })?;

        // Check if the bit width is too large. This limit is defined to be one bit less than the
        // number of bits required to encode `F::Integer`. (One less so that we can compute `1 <<
        // bits` without overflowing.)
        let limit = std::mem::size_of::<F::Integer>() * 8 - 1;
        if bits > limit {
            return Err(FlpError::InvalidParameter(format!(
                "bit wdith exceeds limit of {limit}"
            )));
        }

        // Check for degenerate parameters.
        if bits == 0 {
            return Err(FlpError::InvalidParameter(
                "bits cannot be zero".to_string(),
            ));
        }
        if len == 0 {
            return Err(FlpError::InvalidParameter("len cannot be zero".to_string()));
        }
        if chunk_length == 0 {
            return Err(FlpError::InvalidParameter(
                "chunk_length cannot be zero".to_string(),
            ));
        }

        // Compute the largest encodable measurement.
        let one = F::Integer::from(F::one());
        let max = (one << bits) - one;

        let mut gadget_calls = flattened_len / chunk_length;
        if flattened_len % chunk_length != 0 {
            gadget_calls += 1;
        }

        Ok(Self {
            len,
            bits,
            flattened_len,
            max,
            chunk_length,
            gadget_calls,
            phantom: PhantomData,
        })
    }
}

impl<F: FftFriendlyFieldElement, S> Clone for SumVec<F, S> {
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            bits: self.bits,
            flattened_len: self.flattened_len,
            max: self.max,
            chunk_length: self.chunk_length,
            gadget_calls: self.gadget_calls,
            phantom: PhantomData,
        }
    }
}

impl<F, S> Type for SumVec<F, S>
where
    F: FftFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
{
    type Measurement = Vec<F::Integer>;
    type AggregateResult = Vec<F::Integer>;
    type Field = F;

    fn encode_measurement(&self, measurement: &Vec<F::Integer>) -> Result<Vec<F>, FlpError> {
        if measurement.len() != self.len {
            return Err(FlpError::Encode(format!(
                "unexpected measurement length: got {}; want {}",
                measurement.len(),
                self.len
            )));
        }

        let mut flattened = Vec::with_capacity(self.flattened_len);
        for summand in measurement.iter() {
            if summand > &self.max {
                return Err(FlpError::Encode(format!(
                    "summand exceeds maximum of 2^{}-1",
                    self.bits
                )));
            }
            flattened.extend(F::encode_as_bitvector(*summand, self.bits)?);
        }

        Ok(flattened)
    }

    fn decode_result(
        &self,
        data: &[F],
        _num_measurements: usize,
    ) -> Result<Vec<F::Integer>, FlpError> {
        decode_result_vec(data, self.len)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(S::new(
            Mul::new(self.gadget_calls),
            self.chunk_length,
        ))]
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, FlpError> {
        self.valid_call_check(input, joint_rand)?;

        parallel_sum_range_checks(
            &mut g[0],
            input,
            joint_rand[0],
            self.chunk_length,
            num_shares,
        )
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        let mut unflattened = Vec::with_capacity(self.len);
        for chunk in input.chunks(self.bits) {
            unflattened.push(F::decode_bitvector(chunk)?);
        }
        Ok(unflattened)
    }

    fn input_len(&self) -> usize {
        self.flattened_len
    }

    fn proof_len(&self) -> usize {
        (self.chunk_length * 2) + 2 * ((1 + self.gadget_calls).next_power_of_two() - 1) + 1
    }

    fn verifier_len(&self) -> usize {
        2 + self.chunk_length * 2
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn joint_rand_len(&self) -> usize {
        1
    }

    fn prove_rand_len(&self) -> usize {
        self.chunk_length * 2
    }

    fn query_rand_len(&self) -> usize {
        1
    }
}

/// Compute a random linear combination of the result of calls of `g` on each element of `input`.
///
/// # Arguments
///
/// * `g` - The gadget to be applied elementwise
/// * `input` - The vector on whose elements to apply `g`
/// * `rnd` - The randomness used for the linear combination
pub(crate) fn call_gadget_on_vec_entries<F: FftFriendlyFieldElement>(
    g: &mut Box<dyn Gadget<F>>,
    input: &[F],
    rnd: F,
) -> Result<F, FlpError> {
    let mut comb = F::zero();
    let mut r = rnd;
    for chunk in input.chunks(1) {
        comb += r * g.call(chunk)?;
        r *= rnd;
    }
    Ok(comb)
}

/// Given a vector `data` of field elements which should contain exactly one entry, return the
/// integer representation of that entry.
pub(crate) fn decode_result<F: FftFriendlyFieldElement>(
    data: &[F],
) -> Result<F::Integer, FlpError> {
    if data.len() != 1 {
        return Err(FlpError::Decode("unexpected input length".into()));
    }
    Ok(F::Integer::from(data[0]))
}

/// Given a vector `data` of field elements, return a vector containing the corresponding integer
/// representations, if the number of entries matches `expected_len`.
pub(crate) fn decode_result_vec<F: FftFriendlyFieldElement>(
    data: &[F],
    expected_len: usize,
) -> Result<Vec<F::Integer>, FlpError> {
    if data.len() != expected_len {
        return Err(FlpError::Decode("unexpected input length".into()));
    }
    Ok(data.iter().map(|elem| F::Integer::from(*elem)).collect())
}

/// This evaluates range checks on a slice of field elements, using a ParallelSum gadget evaluating
/// many multiplication gates.
///
/// # Arguments
///
/// * `gadget`: A `ParallelSumGadget<F, Mul<F>>` gadget, or a shim wrapping the same.
/// * `input`: A slice of inputs. This calculation will check that all inputs were zero or one
///   before secret sharing.
/// * `joint_randomness`: A joint randomness value, used to compute a random linear combination of
///   individual range checks.
/// * `chunk_length`: How many multiplication gates per ParallelSum gadget. This must match what the
///   gadget was constructed with.
/// * `num_shares`: The number of shares that the inputs were secret shared into. This is needed to
///   correct constant terms in the circuit.
///
/// # Returns
///
/// This returns (additive shares of) zero if all inputs were zero or one, and otherwise returns a
/// non-zero value with high probability.
pub(crate) fn parallel_sum_range_checks<F: FftFriendlyFieldElement>(
    gadget: &mut Box<dyn Gadget<F>>,
    input: &[F],
    joint_randomness: F,
    chunk_length: usize,
    num_shares: usize,
) -> Result<F, FlpError> {
    let f_num_shares = F::from(F::valid_integer_try_from::<usize>(num_shares)?);
    let num_shares_inverse = f_num_shares.inv();

    let mut output = F::zero();
    let mut r_power = joint_randomness;
    let mut padded_chunk = vec![F::zero(); 2 * chunk_length];

    for chunk in input.chunks(chunk_length) {
        // Construct arguments for the Mul subcircuits.
        for (input, args) in chunk.iter().zip(padded_chunk.chunks_exact_mut(2)) {
            args[0] = r_power * *input;
            args[1] = *input - num_shares_inverse;
            r_power *= joint_randomness;
        }
        // If the chunk of the input is smaller than chunk_length, use zeros instead of measurement
        // inputs for the remaining calls.
        for args in padded_chunk[chunk.len() * 2..].chunks_exact_mut(2) {
            args[0] = F::zero();
            args[1] = -num_shares_inverse;
            // Skip updating r_power. This inner loop is only used during the last iteration of the
            // outer loop, if the last input chunk is a partial chunk. Thus, r_power won't be
            // accessed again before returning.
        }

        output += gadget.call(&padded_chunk)?;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{
        random_vector, Field64 as TestField, FieldElement, FieldElementWithInteger,
    };
    use crate::flp::gadgets::ParallelSum;
    #[cfg(feature = "multithreaded")]
    use crate::flp::gadgets::ParallelSumMultithreaded;
    use crate::flp::test_utils::FlpTest;
    use std::cmp;

    #[test]
    fn test_count() {
        let count: Count<TestField> = Count::new();
        let zero = TestField::zero();
        let one = TestField::one();

        // Round trip
        assert_eq!(
            count
                .decode_result(
                    &count
                        .truncate(count.encode_measurement(&true).unwrap())
                        .unwrap(),
                    1
                )
                .unwrap(),
            1,
        );

        // Test FLP on valid input.
        FlpTest::expect_valid::<3>(&count, &count.encode_measurement(&true).unwrap(), &[one]);
        FlpTest::expect_valid::<3>(&count, &count.encode_measurement(&false).unwrap(), &[zero]);

        // Test FLP on invalid input.
        FlpTest::expect_invalid::<3>(&count, &[TestField::from(1337)]);

        // Try running the validity circuit on an input that's too short.
        count.valid(&mut count.gadget(), &[], &[], 1).unwrap_err();
        count
            .valid(&mut count.gadget(), &[1.into(), 2.into()], &[], 1)
            .unwrap_err();
    }

    #[test]
    fn test_sum() {
        let sum = Sum::new(11).unwrap();
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        // Round trip
        assert_eq!(
            sum.decode_result(
                &sum.truncate(sum.encode_measurement(&27).unwrap()).unwrap(),
                1
            )
            .unwrap(),
            27,
        );

        // Test FLP on valid input.
        FlpTest::expect_valid::<3>(
            &sum,
            &sum.encode_measurement(&1337).unwrap(),
            &[TestField::from(1337)],
        );
        FlpTest::expect_valid::<3>(&Sum::new(0).unwrap(), &[], &[zero]);
        FlpTest::expect_valid::<3>(&Sum::new(2).unwrap(), &[one, zero], &[one]);
        FlpTest::expect_valid::<3>(
            &Sum::new(9).unwrap(),
            &[one, zero, one, one, zero, one, one, one, zero],
            &[TestField::from(237)],
        );

        // Test FLP on invalid input.
        FlpTest::expect_invalid::<3>(&Sum::new(3).unwrap(), &[one, nine, zero]);
        FlpTest::expect_invalid::<3>(&Sum::new(5).unwrap(), &[zero, zero, zero, zero, nine]);
    }

    #[test]
    fn test_average() {
        let average = Average::new(11).unwrap();
        let zero = TestField::zero();
        let one = TestField::one();
        let ten = TestField::from(10);

        // Testing that average correctly quotients the sum of the measurements
        // by the number of measurements.
        assert_eq!(average.decode_result(&[zero], 1).unwrap(), 0.0);
        assert_eq!(average.decode_result(&[one], 1).unwrap(), 1.0);
        assert_eq!(average.decode_result(&[one], 2).unwrap(), 0.5);
        assert_eq!(average.decode_result(&[one], 4).unwrap(), 0.25);
        assert_eq!(average.decode_result(&[ten], 8).unwrap(), 1.25);

        // round trip of 12 with `num_measurements`=1
        assert_eq!(
            average
                .decode_result(
                    &average
                        .truncate(average.encode_measurement(&12).unwrap())
                        .unwrap(),
                    1
                )
                .unwrap(),
            12.0
        );

        // round trip of 12 with `num_measurements`=24
        assert_eq!(
            average
                .decode_result(
                    &average
                        .truncate(average.encode_measurement(&12).unwrap())
                        .unwrap(),
                    24
                )
                .unwrap(),
            0.5
        );
    }

    fn test_histogram<F, S>(f: F)
    where
        F: Fn(usize, usize) -> Result<Histogram<TestField, S>, FlpError>,
        S: ParallelSumGadget<TestField, Mul<TestField>> + Eq + 'static,
    {
        let hist = f(3, 2).unwrap();
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        assert_eq!(&hist.encode_measurement(&0).unwrap(), &[one, zero, zero]);
        assert_eq!(&hist.encode_measurement(&1).unwrap(), &[zero, one, zero]);
        assert_eq!(&hist.encode_measurement(&2).unwrap(), &[zero, zero, one]);

        // Round trip
        assert_eq!(
            hist.decode_result(
                &hist.truncate(hist.encode_measurement(&2).unwrap()).unwrap(),
                1
            )
            .unwrap(),
            [0, 0, 1]
        );

        // Test valid inputs.
        FlpTest::expect_valid::<3>(
            &hist,
            &hist.encode_measurement(&0).unwrap(),
            &[one, zero, zero],
        );

        FlpTest::expect_valid::<3>(
            &hist,
            &hist.encode_measurement(&1).unwrap(),
            &[zero, one, zero],
        );

        FlpTest::expect_valid::<3>(
            &hist,
            &hist.encode_measurement(&2).unwrap(),
            &[zero, zero, one],
        );

        // Test invalid inputs.
        FlpTest::expect_invalid::<3>(&hist, &[zero, zero, nine]);
        FlpTest::expect_invalid::<3>(&hist, &[zero, one, one]);
        FlpTest::expect_invalid::<3>(&hist, &[one, one, one]);
        FlpTest::expect_invalid::<3>(&hist, &[zero, zero, zero]);
    }

    #[test]
    fn test_histogram_serial() {
        test_histogram(Histogram::<TestField, ParallelSum<TestField, Mul<TestField>>>::new);
    }

    #[test]
    #[cfg(feature = "multithreaded")]
    fn test_histogram_parallel() {
        test_histogram(
            Histogram::<TestField, ParallelSumMultithreaded<TestField, Mul<TestField>>>::new,
        );
    }

    fn test_multihot<F, S>(constructor: F)
    where
        F: Fn(usize, usize, usize) -> Result<MultihotCountVec<TestField, S>, FlpError>,
        S: ParallelSumGadget<TestField, Mul<TestField>> + Eq + 'static,
    {
        const NUM_SHARES: usize = 3;

        // Chunk size for our range check gadget
        let chunk_size = 2;

        // Our test is on multihot vecs of length 3, with max weight 2
        let num_buckets = 3;
        let max_weight = 2;

        let multihot_instance = constructor(num_buckets, max_weight, chunk_size).unwrap();
        let zero = TestField::zero();
        let one = TestField::one();
        let nine = TestField::from(9);

        let encoded_weight_plus_offset = |weight| {
            let bits_for_weight = max_weight.ilog2() as usize + 1;
            let offset = (1 << bits_for_weight) - 1 - max_weight;
            TestField::encode_as_bitvector(
                <TestField as FieldElementWithInteger>::Integer::try_from(weight + offset).unwrap(),
                bits_for_weight,
            )
            .unwrap()
            .collect::<Vec<TestField>>()
        };

        assert_eq!(
            multihot_instance
                .encode_measurement(&vec![true, true, false])
                .unwrap(),
            [&[one, one, zero], &*encoded_weight_plus_offset(2)].concat(),
        );
        assert_eq!(
            multihot_instance
                .encode_measurement(&vec![false, true, true])
                .unwrap(),
            [&[zero, one, one], &*encoded_weight_plus_offset(2)].concat(),
        );

        // Round trip
        assert_eq!(
            multihot_instance
                .decode_result(
                    &multihot_instance
                        .truncate(
                            multihot_instance
                                .encode_measurement(&vec![false, true, true])
                                .unwrap()
                        )
                        .unwrap(),
                    1
                )
                .unwrap(),
            [0, 1, 1]
        );

        // Test valid inputs with weights 0, 1, and 2
        FlpTest::expect_valid::<NUM_SHARES>(
            &multihot_instance,
            &multihot_instance
                .encode_measurement(&vec![true, false, false])
                .unwrap(),
            &[one, zero, zero],
        );

        FlpTest::expect_valid::<NUM_SHARES>(
            &multihot_instance,
            &multihot_instance
                .encode_measurement(&vec![false, true, true])
                .unwrap(),
            &[zero, one, one],
        );

        FlpTest::expect_valid::<NUM_SHARES>(
            &multihot_instance,
            &multihot_instance
                .encode_measurement(&vec![false, false, false])
                .unwrap(),
            &[zero, zero, zero],
        );

        // Test invalid inputs.

        // Not binary
        FlpTest::expect_invalid::<NUM_SHARES>(
            &multihot_instance,
            &[&[zero, zero, nine], &*encoded_weight_plus_offset(1)].concat(),
        );
        // Wrong weight
        FlpTest::expect_invalid::<NUM_SHARES>(
            &multihot_instance,
            &[&[zero, zero, one], &*encoded_weight_plus_offset(2)].concat(),
        );
        // We cannot test the case where the weight is higher than max_weight. This is because
        // weight + offset cannot fit into a bitvector of the correct length. In other words, being
        // out-of-range requires the prover to lie about their weight, which is tested above
    }

    #[test]
    fn test_multihot_serial() {
        test_multihot(MultihotCountVec::<TestField, ParallelSum<TestField, Mul<TestField>>>::new);
    }

    fn test_sum_vec<F, S>(f: F)
    where
        F: Fn(usize, usize, usize) -> Result<SumVec<TestField, S>, FlpError>,
        S: 'static + ParallelSumGadget<TestField, Mul<TestField>> + Eq,
    {
        let one = TestField::one();
        let nine = TestField::from(9);

        // Test on valid inputs.
        for len in 1..10 {
            let chunk_length = cmp::max((len as f64).sqrt() as usize, 1);
            let sum_vec = f(1, len, chunk_length).unwrap();
            FlpTest::expect_valid_no_output::<3>(
                &sum_vec,
                &sum_vec.encode_measurement(&vec![1; len]).unwrap(),
            );
        }

        let len = 100;
        let sum_vec = f(1, len, 10).unwrap();
        FlpTest::expect_valid::<3>(
            &sum_vec,
            &sum_vec.encode_measurement(&vec![1; len]).unwrap(),
            &vec![one; len],
        );

        let len = 23;
        let sum_vec = f(4, len, 4).unwrap();
        FlpTest::expect_valid::<3>(
            &sum_vec,
            &sum_vec.encode_measurement(&vec![9; len]).unwrap(),
            &vec![nine; len],
        );

        // Test on invalid inputs.
        for len in 1..10 {
            let chunk_length = cmp::max((len as f64).sqrt() as usize, 1);
            let sum_vec = f(1, len, chunk_length).unwrap();
            FlpTest::expect_invalid::<3>(&sum_vec, &vec![nine; len]);
        }

        let len = 23;
        let sum_vec = f(2, len, 4).unwrap();
        FlpTest::expect_invalid::<3>(&sum_vec, &vec![nine; 2 * len]);

        // Round trip
        let want = vec![1; len];
        assert_eq!(
            sum_vec
                .decode_result(
                    &sum_vec
                        .truncate(sum_vec.encode_measurement(&want).unwrap())
                        .unwrap(),
                    1
                )
                .unwrap(),
            want
        );
    }

    #[test]
    fn test_sum_vec_serial() {
        test_sum_vec(SumVec::<TestField, ParallelSum<TestField, Mul<TestField>>>::new)
    }

    #[test]
    #[cfg(feature = "multithreaded")]
    fn test_sum_vec_parallel() {
        test_sum_vec(SumVec::<TestField, ParallelSumMultithreaded<TestField, Mul<TestField>>>::new)
    }

    #[test]
    fn sum_vec_serial_long() {
        let typ: SumVec<TestField, ParallelSum<TestField, _>> = SumVec::new(1, 1000, 31).unwrap();
        let input = typ.encode_measurement(&vec![0; 1000]).unwrap();
        assert_eq!(input.len(), typ.input_len());
        let joint_rand = random_vector(typ.joint_rand_len()).unwrap();
        let prove_rand = random_vector(typ.prove_rand_len()).unwrap();
        let query_rand = random_vector(typ.query_rand_len()).unwrap();
        let proof = typ.prove(&input, &prove_rand, &joint_rand).unwrap();
        let verifier = typ
            .query(&input, &proof, &query_rand, &joint_rand, 1)
            .unwrap();
        assert_eq!(verifier.len(), typ.verifier_len());
        assert!(typ.decide(&verifier).unwrap());
    }

    #[test]
    #[cfg(feature = "multithreaded")]
    fn sum_vec_parallel_long() {
        let typ: SumVec<TestField, ParallelSumMultithreaded<TestField, _>> =
            SumVec::new(1, 1000, 31).unwrap();
        let input = typ.encode_measurement(&vec![0; 1000]).unwrap();
        assert_eq!(input.len(), typ.input_len());
        let joint_rand = random_vector(typ.joint_rand_len()).unwrap();
        let prove_rand = random_vector(typ.prove_rand_len()).unwrap();
        let query_rand = random_vector(typ.query_rand_len()).unwrap();
        let proof = typ.prove(&input, &prove_rand, &joint_rand).unwrap();
        let verifier = typ
            .query(&input, &proof, &query_rand, &joint_rand, 1)
            .unwrap();
        assert_eq!(verifier.len(), typ.verifier_len());
        assert!(typ.decide(&verifier).unwrap());
    }
}

#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod fixedpoint_l2;
