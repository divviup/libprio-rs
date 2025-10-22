// SPDX-License-Identifier: MPL-2.0

//! A collection of [`Type`] implementations.

use crate::field::{FieldElementWithIntegerExt, Integer, NttFriendlyFieldElement};
use crate::flp::gadgets::{Mul, ParallelSumGadget, PolyEval};
use crate::flp::{Flp, FlpError, Gadget, Type};
use crate::polynomial::poly_range_check;
use std::convert::TryInto;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::slice;
use subtle::Choice;

#[cfg(feature = "experimental")]
mod dp;
#[cfg(feature = "experimental")]
mod l1boundsum;

#[cfg(feature = "experimental")]
pub use l1boundsum::L1BoundSum;

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

impl<F: NttFriendlyFieldElement> Count<F> {
    /// Return a new [`Count`] type instance.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: NttFriendlyFieldElement> Default for Count<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: NttFriendlyFieldElement> Flp for Count<F> {
    type Field = F;

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(Mul::new(1))]
    }

    fn num_gadgets(&self) -> usize {
        1
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        _num_shares: usize,
    ) -> Result<Vec<F>, FlpError> {
        self.valid_call_check(input, joint_rand)?;
        let out = g[0].call(&[input[0], input[0]])? - input[0];
        Ok(vec![out])
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

    fn joint_rand_len(&self) -> usize {
        0
    }

    fn eval_output_len(&self) -> usize {
        1
    }

    fn prove_rand_len(&self) -> usize {
        2
    }
}

impl<F: NttFriendlyFieldElement> Type for Count<F> {
    type Measurement = bool;

    type AggregateResult = F::Integer;

    fn encode_measurement(&self, value: &bool) -> Result<Vec<F>, FlpError> {
        Ok(vec![F::conditional_select(
            &F::zero(),
            &F::one(),
            Choice::from(u8::from(*value)),
        )])
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        Ok(input)
    }

    fn decode_result(&self, data: &[F], _num_measurements: usize) -> Result<F::Integer, FlpError> {
        decode_result(data)
    }

    fn output_len(&self) -> usize {
        self.input_len()
    }
}

/// The sum type. Each measurement is a integer in `[0, max_measurement]` and the aggregate is the
/// sum of the measurements.
///
/// The validity circuit is based on the SIMD circuit construction of [[BBCG+19], Theorem 5.3].
///
/// [BBCG+19]: https://ia.cr/2019/188
#[derive(Clone, PartialEq, Eq)]
pub struct Sum<F: NttFriendlyFieldElement> {
    max_measurement: F::Integer,

    // Computed from max_measurement
    offset: F::Integer,
    bits: usize,
    // Constant
    bit_range_checker: Vec<F>,
}

impl<F: NttFriendlyFieldElement> Debug for Sum<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sum")
            .field("max_measurement", &self.max_measurement)
            .field("bits", &self.bits)
            .finish()
    }
}

impl<F: NttFriendlyFieldElement> Sum<F> {
    /// Return a new [`Sum`] type parameter. Each value of this type is an integer in range `[0,
    /// max_measurement]` where `max_measurement > 0`. Errors if `max_measurement == 0`.
    pub fn new(max_measurement: F::Integer) -> Result<Self, FlpError> {
        if max_measurement == F::Integer::zero() {
            return Err(FlpError::InvalidParameter(
                "max measurement cannot be zero".to_string(),
            ));
        }

        // Number of bits needed to represent x is ⌊log₂(x)⌋ + 1
        let bits = max_measurement.checked_ilog2().unwrap() as usize + 1;

        // The offset we add to the summand for range-checking purposes
        let one = F::Integer::try_from(1).unwrap();
        let offset = (one << bits) - one - max_measurement;

        // Construct a range checker to ensure encoded bits are in the range [0, 2)
        let bit_range_checker = poly_range_check(0, 2);

        Ok(Self {
            bits,
            max_measurement,
            offset,
            bit_range_checker,
        })
    }
}

impl<F: NttFriendlyFieldElement> Flp for Sum<F> {
    type Field = F;

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(PolyEval::new(
            self.bit_range_checker.clone(),
            2 * self.bits,
        ))]
    }

    fn num_gadgets(&self) -> usize {
        1
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<Vec<F>, FlpError> {
        self.valid_call_check(input, joint_rand)?;
        let gadget = &mut g[0];
        let mut output = vec![F::zero(); input.len() + 1];
        for (bit, output_elem) in input.iter().zip(output[..input.len()].iter_mut()) {
            *output_elem = gadget.call(slice::from_ref(bit))?;
        }

        let range_check = {
            let offset = F::from(self.offset);
            let shares_inv = F::from(F::valid_integer_try_from(num_shares)?).inv();
            let sum = F::decode_bitvector(&input[..self.bits])?;
            let sum_plus_offset = F::decode_bitvector(&input[self.bits..])?;
            offset * shares_inv + sum - sum_plus_offset
        };
        output[input.len()] = range_check;

        Ok(output)
    }

    fn input_len(&self) -> usize {
        2 * self.bits
    }

    fn proof_len(&self) -> usize {
        2 * ((1 + 2 * self.bits).next_power_of_two() - 1) + 2
    }

    fn verifier_len(&self) -> usize {
        3
    }

    fn joint_rand_len(&self) -> usize {
        0
    }

    fn eval_output_len(&self) -> usize {
        2 * self.bits + 1
    }

    fn prove_rand_len(&self) -> usize {
        1
    }
}

impl<F: NttFriendlyFieldElement> Type for Sum<F> {
    type Measurement = F::Integer;
    type AggregateResult = F::Integer;

    fn encode_measurement(&self, summand: &F::Integer) -> Result<Vec<F>, FlpError> {
        if summand > &self.max_measurement {
            return Err(FlpError::Encode(format!(
                "unexpected measurement: got {:?}; want ≤{:?}",
                summand, self.max_measurement
            )));
        }

        let enc_summand = F::encode_as_bitvector(*summand, self.bits)?;
        let enc_summand_plus_offset = F::encode_as_bitvector(self.offset + *summand, self.bits)?;

        Ok(enc_summand.chain(enc_summand_plus_offset).collect())
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        let res = F::decode_bitvector(&input[..self.bits])?;
        Ok(vec![res])
    }

    fn decode_result(&self, data: &[F], _num_measurements: usize) -> Result<F::Integer, FlpError> {
        decode_result(data)
    }

    fn output_len(&self) -> usize {
        1
    }
}

/// The average type. Each measurement is an integer in `[0, max_measurement]` and the aggregate is
/// the arithmetic average of the measurements.
// This is just a `Sum` object under the hood. The only difference is that the aggregate result is
// an f64, which we get by dividing by `num_measurements`
#[derive(Clone, PartialEq, Eq)]
pub struct Average<F: NttFriendlyFieldElement> {
    summer: Sum<F>,
}

impl<F: NttFriendlyFieldElement> Debug for Average<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Average")
            .field("max_measurement", &self.summer.max_measurement)
            .field("bits", &self.summer.bits)
            .finish()
    }
}

impl<F: NttFriendlyFieldElement> Average<F> {
    /// Return a new [`Average`] type parameter. Each value of this type is an integer in range `[0,
    /// max_measurement]` where `max_measurement > 0`. Errors if `max_measurement == 0`.
    pub fn new(max_measurement: F::Integer) -> Result<Self, FlpError> {
        let summer = Sum::new(max_measurement)?;
        Ok(Average { summer })
    }
}

impl<F: NttFriendlyFieldElement> Flp for Average<F> {
    type Field = F;

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        self.summer.gadget()
    }

    fn num_gadgets(&self) -> usize {
        self.summer.num_gadgets()
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<Vec<F>, FlpError> {
        self.summer.valid(g, input, joint_rand, num_shares)
    }

    fn input_len(&self) -> usize {
        self.summer.input_len()
    }

    fn proof_len(&self) -> usize {
        self.summer.proof_len()
    }

    fn verifier_len(&self) -> usize {
        self.summer.verifier_len()
    }

    fn joint_rand_len(&self) -> usize {
        self.summer.joint_rand_len()
    }

    fn eval_output_len(&self) -> usize {
        self.summer.eval_output_len()
    }

    fn prove_rand_len(&self) -> usize {
        self.summer.prove_rand_len()
    }
}

impl<F: NttFriendlyFieldElement> Type for Average<F> {
    type Measurement = F::Integer;
    type AggregateResult = f64;

    fn encode_measurement(&self, summand: &F::Integer) -> Result<Vec<F>, FlpError> {
        self.summer.encode_measurement(summand)
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.summer.truncate(input)
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

    fn output_len(&self) -> usize {
        self.summer.output_len()
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

impl<F: NttFriendlyFieldElement, S> Debug for Histogram<F, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Histogram")
            .field("length", &self.length)
            .field("chunk_length", &self.chunk_length)
            .finish()
    }
}

impl<F: NttFriendlyFieldElement, S: ParallelSumGadget<F, Mul<F>>> Histogram<F, S> {
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

impl<F, S> Flp for Histogram<F, S>
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
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<Vec<F>, FlpError> {
        self.valid_call_check(input, joint_rand)?;

        // Check that each element of `input` is a 0 or 1.
        let range_check =
            parallel_sum_range_checks(&mut g[0], input, joint_rand, self.chunk_length, num_shares)?;

        // Check that the elements of `input` sum to 1.
        let mut sum_check = -F::from(F::valid_integer_try_from(num_shares)?).inv();
        for val in input.iter() {
            sum_check += *val;
        }

        Ok(vec![range_check, sum_check])
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

impl<F, S> Type for Histogram<F, S>
where
    F: NttFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
{
    type Measurement = usize;
    type AggregateResult = Vec<F::Integer>;

    fn encode_measurement(&self, measurement: &usize) -> Result<Vec<F>, FlpError> {
        let mut data = vec![F::zero(); self.length];

        data[*measurement] = F::one();
        Ok(data)
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        Ok(input)
    }

    fn decode_result(
        &self,
        data: &[F],
        _num_measurements: usize,
    ) -> Result<Vec<F::Integer>, FlpError> {
        decode_result_vec(data, self.length)
    }

    fn output_len(&self) -> usize {
        self.input_len()
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

impl<F: NttFriendlyFieldElement, S> Debug for MultihotCountVec<F, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultihotCountVec")
            .field("length", &self.length)
            .field("max_weight", &self.max_weight)
            .field("chunk_length", &self.chunk_length)
            .finish()
    }
}

impl<F: NttFriendlyFieldElement, S: ParallelSumGadget<F, Mul<F>>> MultihotCountVec<F, S> {
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
        let gadget_calls = meas_length.div_ceil(chunk_length);
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

impl<F, S> Flp for MultihotCountVec<F, S>
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
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<Vec<F>, FlpError> {
        self.valid_call_check(input, joint_rand)?;

        // Check that each element of `input` is a 0 or 1.
        let range_check =
            parallel_sum_range_checks(&mut g[0], input, joint_rand, self.chunk_length, num_shares)?;

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

        Ok(vec![range_check, weight_check])
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

    // The number of random values needed in the validity checks
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

impl<F, S> Type for MultihotCountVec<F, S>
where
    F: NttFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
{
    type Measurement = Vec<bool>;
    type AggregateResult = Vec<F::Integer>;

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
        let multihot_vec = measurement
            .iter()
            // We can unwrap because any Integer type can cast from bool
            .map(|bit| F::from(F::valid_integer_try_from(*bit as usize).unwrap()));

        // Encode the measurement weight in binary (actually, the weight plus some offset)
        let offset_weight_bits = {
            let offset_weight_reported = F::valid_integer_try_from(self.offset + weight_reported)?;
            F::encode_as_bitvector(offset_weight_reported, self.bits_for_weight)?
        };

        // Report the concat of the two
        Ok(multihot_vec.chain(offset_weight_bits).collect())
    }

    fn decode_result(
        &self,
        data: &[Self::Field],
        _num_measurements: usize,
    ) -> Result<Self::AggregateResult, FlpError> {
        // The aggregate is the same as the decoded result. Just convert to integers
        decode_result_vec(data, self.length)
    }

    // Truncates the measurement, removing extra data that was necessary for validity (here, the
    // encoded weight), but not important for aggregation
    fn truncate(&self, input: Vec<Self::Field>) -> Result<Vec<Self::Field>, FlpError> {
        self.truncate_call_check(&input)?;
        // Cut off the encoded weight
        Ok(input[..self.length].to_vec())
    }

    // The length of the truncated output (i.e., the output of [`Type::truncate`]).
    fn output_len(&self) -> usize {
        self.length
    }
}

/// A sequence of integers in range `[0, 2^bits)`. This type uses a neat trick from [[BBCG+19],
/// Corollary 4.9] to reduce the proof size to roughly the square root of the input size.
///
/// [BBCG+19]: https://eprint.iacr.org/2019/188
#[derive(PartialEq, Eq)]
pub struct SumVec<F: NttFriendlyFieldElement, S> {
    len: usize,
    bits: usize,
    flattened_len: usize,
    max: F::Integer,
    chunk_length: usize,
    gadget_calls: usize,
    phantom: PhantomData<S>,
}

impl<F: NttFriendlyFieldElement, S> Debug for SumVec<F, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SumVec")
            .field("len", &self.len)
            .field("bits", &self.bits)
            .field("chunk_length", &self.chunk_length)
            .finish()
    }
}

impl<F: NttFriendlyFieldElement, S: ParallelSumGadget<F, Mul<F>>> SumVec<F, S> {
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
                "bit width exceeds limit of {limit}"
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

impl<F: NttFriendlyFieldElement, S> Clone for SumVec<F, S> {
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

impl<F, S> Flp for SumVec<F, S>
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
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<Vec<F>, FlpError> {
        self.valid_call_check(input, joint_rand)?;

        parallel_sum_range_checks(&mut g[0], input, joint_rand, self.chunk_length, num_shares)
            .map(|out| vec![out])
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

    fn joint_rand_len(&self) -> usize {
        self.gadget_calls
    }

    fn eval_output_len(&self) -> usize {
        1
    }

    fn prove_rand_len(&self) -> usize {
        self.chunk_length * 2
    }
}

impl<F, S> Type for SumVec<F, S>
where
    F: NttFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
{
    type Measurement = Vec<F::Integer>;
    type AggregateResult = Vec<F::Integer>;

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

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        let mut unflattened = Vec::with_capacity(self.len);
        for chunk in input.chunks(self.bits) {
            unflattened.push(F::decode_bitvector(chunk)?);
        }
        Ok(unflattened)
    }

    fn decode_result(
        &self,
        data: &[F],
        _num_measurements: usize,
    ) -> Result<Vec<F::Integer>, FlpError> {
        decode_result_vec(data, self.len)
    }

    fn output_len(&self) -> usize {
        self.len
    }
}

/// Given a vector `data` of field elements which should contain exactly one entry, return the
/// integer representation of that entry.
pub(crate) fn decode_result<F: NttFriendlyFieldElement>(
    data: &[F],
) -> Result<F::Integer, FlpError> {
    if data.len() != 1 {
        return Err(FlpError::Decode("unexpected input length".into()));
    }
    Ok(F::Integer::from(data[0]))
}

/// Given a vector `data` of field elements, return a vector containing the corresponding integer
/// representations, if the number of entries matches `expected_len`.
pub(crate) fn decode_result_vec<F: NttFriendlyFieldElement>(
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
pub(crate) fn parallel_sum_range_checks<F: NttFriendlyFieldElement>(
    gadget: &mut Box<dyn Gadget<F>>,
    input: &[F],
    joint_randomness: &[F],
    chunk_length: usize,
    num_shares: usize,
) -> Result<F, FlpError> {
    let f_num_shares = F::from(F::valid_integer_try_from(num_shares)?);
    let num_shares_inverse = f_num_shares.inv();

    let mut output = F::zero();
    let mut padded_chunk = vec![F::zero(); 2 * chunk_length];

    for (chunk, &r) in input.chunks(chunk_length).zip(joint_randomness) {
        let mut r_power = r;

        // Construct arguments for the Mul subcircuits.
        for (input, args) in chunk.iter().zip(padded_chunk.chunks_exact_mut(2)) {
            args[0] = r_power * *input;
            args[1] = *input - num_shares_inverse;
            r_power *= r;
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
    use crate::field::{Field64 as TestField, FieldElement, FieldElementWithInteger};
    use crate::flp::gadgets::ParallelSum;
    #[cfg(feature = "multithreaded")]
    use crate::flp::gadgets::ParallelSumMultithreaded;
    use crate::flp::test_utils::TypeTest;
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
        TypeTest::expect_valid::<3>(&count, &count.encode_measurement(&true).unwrap(), &[one]);
        TypeTest::expect_valid::<3>(&count, &count.encode_measurement(&false).unwrap(), &[zero]);

        // Test FLP on invalid input.
        TypeTest::expect_invalid::<3>(&count, &[TestField::from(1337)]);

        // Try running the validity circuit on an input that's too short.
        count.valid(&mut count.gadget(), &[], &[], 1).unwrap_err();
        count
            .valid(&mut count.gadget(), &[1.into(), 2.into()], &[], 1)
            .unwrap_err();
    }

    #[test]
    fn test_sum() {
        let max_measurement = 1458;

        let sum = Sum::new(max_measurement).unwrap();
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
        TypeTest::expect_valid::<3>(
            &sum,
            &sum.encode_measurement(&1337).unwrap(),
            &[TestField::from(1337)],
        );

        {
            let sum = Sum::new(3).unwrap();
            let meas = 1;
            TypeTest::expect_valid::<3>(
                &sum,
                &sum.encode_measurement(&meas).unwrap(),
                &[TestField::from(meas)],
            );
        }

        {
            let sum = Sum::new(400).unwrap();
            let meas = 237;
            TypeTest::expect_valid::<3>(
                &sum,
                &sum.encode_measurement(&meas).unwrap(),
                &[TestField::from(meas)],
            );
        }

        // Test FLP on invalid input, specifically on field elements outside of {0,1}
        {
            let sum = Sum::new((1 << 3) - 1).unwrap();
            // The sum+offset value can be whatever. The binariness test should fail first
            let sum_plus_offset = vec![zero; 3];
            TypeTest::expect_invalid::<3>(
                &sum,
                &[&[one, nine, zero], sum_plus_offset.as_slice()].concat(),
            );
        }
        {
            let sum = Sum::new((1 << 5) - 1).unwrap();
            let sum_plus_offset = vec![zero; 5];
            TypeTest::expect_invalid::<3>(
                &sum,
                &[&[zero, zero, zero, zero, nine], sum_plus_offset.as_slice()].concat(),
            );
        }
    }

    #[test]
    fn test_average() {
        let max_measurement = (1 << 11) - 13;

        let average = Average::new(max_measurement).unwrap();
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
        TypeTest::expect_valid::<3>(
            &hist,
            &hist.encode_measurement(&0).unwrap(),
            &[one, zero, zero],
        );

        TypeTest::expect_valid::<3>(
            &hist,
            &hist.encode_measurement(&1).unwrap(),
            &[zero, one, zero],
        );

        TypeTest::expect_valid::<3>(
            &hist,
            &hist.encode_measurement(&2).unwrap(),
            &[zero, zero, one],
        );

        // Test invalid inputs.
        TypeTest::expect_invalid::<3>(&hist, &[zero, zero, nine]);
        TypeTest::expect_invalid::<3>(&hist, &[zero, one, one]);
        TypeTest::expect_invalid::<3>(&hist, &[one, one, one]);
        TypeTest::expect_invalid::<3>(&hist, &[zero, zero, zero]);
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
        TypeTest::expect_valid::<NUM_SHARES>(
            &multihot_instance,
            &multihot_instance
                .encode_measurement(&vec![true, false, false])
                .unwrap(),
            &[one, zero, zero],
        );

        TypeTest::expect_valid::<NUM_SHARES>(
            &multihot_instance,
            &multihot_instance
                .encode_measurement(&vec![false, true, true])
                .unwrap(),
            &[zero, one, one],
        );

        TypeTest::expect_valid::<NUM_SHARES>(
            &multihot_instance,
            &multihot_instance
                .encode_measurement(&vec![false, false, false])
                .unwrap(),
            &[zero, zero, zero],
        );

        // Test invalid inputs.

        // Not binary
        TypeTest::expect_invalid::<NUM_SHARES>(
            &multihot_instance,
            &[&[zero, zero, nine], &*encoded_weight_plus_offset(1)].concat(),
        );
        // Wrong weight
        TypeTest::expect_invalid::<NUM_SHARES>(
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
            TypeTest::expect_valid_no_output::<3>(
                &sum_vec,
                &sum_vec.encode_measurement(&vec![1; len]).unwrap(),
            );
        }

        let len = 100;
        let sum_vec = f(1, len, 10).unwrap();
        TypeTest::expect_valid::<3>(
            &sum_vec,
            &sum_vec.encode_measurement(&vec![1; len]).unwrap(),
            &vec![one; len],
        );

        let len = 23;
        let sum_vec = f(4, len, 4).unwrap();
        TypeTest::expect_valid::<3>(
            &sum_vec,
            &sum_vec.encode_measurement(&vec![9; len]).unwrap(),
            &vec![nine; len],
        );

        // Test on invalid inputs.
        for len in 1..10 {
            let chunk_length = cmp::max((len as f64).sqrt() as usize, 1);
            let sum_vec = f(1, len, chunk_length).unwrap();
            TypeTest::expect_invalid::<3>(&sum_vec, &vec![nine; len]);
        }

        let len = 23;
        let sum_vec = f(2, len, 4).unwrap();
        TypeTest::expect_invalid::<3>(&sum_vec, &vec![nine; 2 * len]);

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
        let joint_rand = TestField::random_vector(typ.joint_rand_len());
        let prove_rand = TestField::random_vector(typ.prove_rand_len());
        let query_rand = TestField::random_vector(typ.query_rand_len());
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
        let joint_rand = TestField::random_vector(typ.joint_rand_len());
        let prove_rand = TestField::random_vector(typ.prove_rand_len());
        let query_rand = TestField::random_vector(typ.query_rand_len());
        let proof = typ.prove(&input, &prove_rand, &joint_rand).unwrap();
        let verifier = typ
            .query(&input, &proof, &query_rand, &joint_rand, 1)
            .unwrap();
        assert_eq!(verifier.len(), typ.verifier_len());
        assert!(typ.decide(&verifier).unwrap());
    }

    /// Validity circuit using a degree 3 gadget, for test purposes.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct HigherDegree;

    impl HigherDegree {
        fn new() -> Self {
            Self
        }
    }

    impl Flp for HigherDegree {
        type Field = TestField;

        fn gadget(&self) -> Vec<Box<dyn Gadget<Self::Field>>> {
            vec![Box::new(PolyEval::new(
                vec![
                    TestField::from(0),
                    TestField::from(2),
                    -TestField::from(3),
                    TestField::from(1),
                ],
                1,
            ))]
        }

        fn num_gadgets(&self) -> usize {
            1
        }

        fn valid(
            &self,
            gadgets: &mut Vec<Box<dyn Gadget<Self::Field>>>,
            input: &[Self::Field],
            joint_rand: &[Self::Field],
            _num_shares: usize,
        ) -> Result<Vec<Self::Field>, FlpError> {
            self.valid_call_check(input, joint_rand)?;
            let check = gadgets[0].call(input)?;
            Ok(vec![check])
        }

        fn input_len(&self) -> usize {
            1
        }

        fn proof_len(&self) -> usize {
            let gadget_arity = 1;
            let gadget_degree = 3;
            let gadget_calls = 1usize;
            let p = (gadget_calls + 1).next_power_of_two();
            gadget_arity + gadget_degree * (p - 1) + 1
        }

        fn verifier_len(&self) -> usize {
            3
        }

        fn joint_rand_len(&self) -> usize {
            0
        }

        fn eval_output_len(&self) -> usize {
            1
        }

        fn prove_rand_len(&self) -> usize {
            1
        }
    }

    impl Type for HigherDegree {
        type Measurement = u64;

        type AggregateResult = u64;

        fn encode_measurement(
            &self,
            measurement: &Self::Measurement,
        ) -> Result<Vec<Self::Field>, FlpError> {
            Ok(vec![TestField::from(*measurement)])
        }

        fn truncate(&self, input: Vec<Self::Field>) -> Result<Vec<Self::Field>, FlpError> {
            self.truncate_call_check(&input)?;
            Ok(input)
        }

        fn decode_result(
            &self,
            data: &[Self::Field],
            _num_measurements: usize,
        ) -> Result<Self::AggregateResult, FlpError> {
            decode_result(data)
        }

        fn output_len(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_degree_3() {
        let typ = HigherDegree::new();
        TypeTest::expect_valid::<2>(&typ, &[TestField::from(0)], &[TestField::from(0)]);
        TypeTest::expect_valid::<2>(&typ, &[TestField::from(1)], &[TestField::from(1)]);
        TypeTest::expect_valid::<2>(&typ, &[TestField::from(2)], &[TestField::from(2)]);
        TypeTest::expect_invalid::<2>(&typ, &[TestField::from(3)]);
    }
}

#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod fixedpoint_l2;
