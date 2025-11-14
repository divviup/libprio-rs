// SPDX-License-Identifier: MPL-2.0

//! Implementation of the Prio3 VDAF [[draft-irtf-cfrg-vdaf-08]].
//!
//! **WARNING:** This code has not undergone significant security analysis. Use at your own risk.
//!
//! Prio3 is based on the Prio system desigend by Dan Boneh and Henry Corrigan-Gibbs and presented
//! at NSDI 2017 [[CGB17]]. However, it incorporates a few techniques from Boneh et al., CRYPTO
//! 2019 [[BBCG+19]], that lead to substantial improvements in terms of run time and communication
//! cost. The security of the construction was analyzed in [[DPRS23]].
//!
//! Prio3 is a transformation of a Fully Linear Proof (FLP) system [[draft-irtf-cfrg-vdaf-08]] into
//! a VDAF. The base type, [`Prio3`], supports a wide variety of aggregation functions, some of
//! which are instantiated here:
//!
//! - [`Prio3Count`] for aggregating a counter (*)
//! - [`Prio3Sum`] for copmputing the sum of integers (*)
//! - [`Prio3SumVec`] for aggregating a vector of integers
//! - [`Prio3Histogram`] for estimating a distribution via a histogram (*)
//!
//! Additional types can be constructed from [`Prio3`] as needed.
//!
//! (*) denotes that the type is specified in [[draft-irtf-cfrg-vdaf-08]].
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [CGB17]: https://crypto.stanford.edu/prio/
//! [DPRS23]: https://ia.cr/2023/130
//! [draft-irtf-cfrg-vdaf-08]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/08/

use super::xof::XofTurboShake128;
#[cfg(feature = "experimental")]
use super::AggregatorWithNoise;
use crate::codec::{encode_fixlen_items, CodecError, Decode, Encode, ParameterizedDecode};
#[cfg(feature = "experimental")]
use crate::dp::DifferentialPrivacyStrategy;
use crate::field::{
    add_assign_vector, decode_fieldvec, sub_assign_vector, FieldElement, FieldElementWithInteger,
    NttFriendlyFieldElement,
};
use crate::field::{Field128, Field64};
#[cfg(feature = "multithreaded")]
use crate::flp::gadgets::ParallelSumMultithreaded;
#[cfg(feature = "experimental")]
use crate::flp::gadgets::PolyEval;
use crate::flp::gadgets::{Mul, ParallelSum};
#[cfg(feature = "experimental")]
use crate::flp::types::fixedpoint_l2::{
    compatible_float::CompatibleFloat, FixedPointBoundedL2VecSum,
};
#[cfg(feature = "experimental")]
use crate::flp::TypeWithNoise;
use crate::flp::{
    types::{Average, Count, Histogram, MultihotCountVec, Sum, SumVec},
    Type,
};
use crate::prng::Prng;
use crate::vdaf::xof::{IntoFieldVec, Seed, Xof};
use crate::vdaf::{
    Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare, PrepareTransition,
    Share, ShareDecodingParameter, Vdaf, VdafError, VERSION,
};
#[cfg(feature = "experimental")]
use fixed::traits::Fixed;
use rand::{rng, Rng};
use std::borrow::Cow;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::io::Cursor;
use std::iter::{self, IntoIterator};
use std::marker::PhantomData;
use subtle::{Choice, ConstantTimeEq};

#[cfg(feature = "experimental")]
mod l1boundsum;

#[cfg(feature = "experimental")]
pub use l1boundsum::Prio3L1BoundSum;

const DST_MEASUREMENT_SHARE: u16 = 1;
const DST_PROOF_SHARE: u16 = 2;
const DST_JOINT_RANDOMNESS: u16 = 3;
const DST_PROVE_RANDOMNESS: u16 = 4;
const DST_QUERY_RANDOMNESS: u16 = 5;
const DST_JOINT_RAND_SEED: u16 = 6;
const DST_JOINT_RAND_PART: u16 = 7;

/// The count type. Each measurement is an integer in `[0,2)` and the aggregate result is the sum.
pub type Prio3Count = Prio3<Count<Field64>, XofTurboShake128, 32>;

impl Prio3Count {
    /// Construct an instance of Prio3Count with the given number of aggregators.
    pub fn new_count(num_aggregators: u8) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, 1, 0x00000001, Count::new())
    }
}

/// The count-vector type. Each measurement is a vector of integers in `[0,2^bits)` and the
/// aggregate is the element-wise sum.
pub type Prio3SumVec =
    Prio3<SumVec<Field128, ParallelSum<Field128, Mul<Field128>>>, XofTurboShake128, 32>;

impl Prio3SumVec {
    /// Construct an instance of Prio3SumVec with the given number of aggregators. `bits` defines
    /// the bit width of each summand of the measurement; `len` defines the length of the
    /// measurement vector.
    pub fn new_sum_vec(
        num_aggregators: u8,
        bits: usize,
        len: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(
            num_aggregators,
            1,
            0x00000003,
            SumVec::new(bits, len, chunk_length)?,
        )
    }
}

/// Like [`Prio3SumVec`] except this type uses multithreading to improve sharding
/// time. Note that the improvement is only noticeable for very large input lengths.
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3SumVecMultithreaded = Prio3<
    SumVec<Field128, ParallelSumMultithreaded<Field128, Mul<Field128>>>,
    XofTurboShake128,
    32,
>;

#[cfg(feature = "multithreaded")]
impl Prio3SumVecMultithreaded {
    /// Construct an instance of Prio3SumVecMultithreaded with the given number of
    /// aggregators. `bits` defines the bit width of each summand of the measurement; `len` defines
    /// the length of the measurement vector.
    pub fn new_sum_vec_multithreaded(
        num_aggregators: u8,
        bits: usize,
        len: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(
            num_aggregators,
            1,
            0x00000003,
            SumVec::new(bits, len, chunk_length)?,
        )
    }
}

/// The sum type. Each measurement is an integer in `[0, max_measurement]` and the
/// aggregate is the sum.
pub type Prio3Sum = Prio3<Sum<Field64>, XofTurboShake128, 32>;

impl Prio3Sum {
    /// Construct an instance of `Prio3Sum` with the given number of aggregators, where each summand
    /// must be in the range `[0, max_measurement]`. Errors if `max_measurement == 0`.
    pub fn new_sum(
        num_aggregators: u8,
        max_measurement: <Field64 as FieldElementWithInteger>::Integer,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, 1, 0x00000002, Sum::new(max_measurement)?)
    }
}

/// The fixed point vector sum type. Each measurement is a vector of fixed point numbers
/// and the aggregate is the sum represented as 64-bit floats. The preparation phase
/// ensures the L2 norm of the input vector is < 1.
///
/// This is useful for aggregating gradients in a federated version of
/// [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) with
/// [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy),
/// useful, e.g., for [differentially private deep learning](https://arxiv.org/pdf/1607.00133.pdf).
/// The bound on input norms is required for differential privacy. The fixed point representation
/// allows an easy conversion to the integer type used in internal computation, while leaving
/// conversion to the client. The model itself will have floating point parameters, so the output
/// sum has that type as well.
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub type Prio3FixedPointBoundedL2VecSum<Fx> = Prio3<
    FixedPointBoundedL2VecSum<
        Fx,
        ParallelSum<Field128, PolyEval<Field128>>,
        ParallelSum<Field128, Mul<Field128>>,
    >,
    XofTurboShake128,
    32,
>;

#[cfg(feature = "experimental")]
impl<Fx: Fixed + CompatibleFloat> Prio3FixedPointBoundedL2VecSum<Fx> {
    /// Construct an instance of this VDAF with the given number of aggregators and number of
    /// vector entries.
    pub fn new_fixedpoint_boundedl2_vec_sum(
        num_aggregators: u8,
        entries: usize,
    ) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;
        Prio3::new(
            num_aggregators,
            1,
            0xFFFF0000,
            FixedPointBoundedL2VecSum::new(entries)?,
        )
    }
}

/// The fixed point vector sum type. Each measurement is a vector of fixed point numbers
/// and the aggregate is the sum represented as 64-bit floats. The verification function
/// ensures the L2 norm of the input vector is < 1.
#[cfg(all(feature = "experimental", feature = "multithreaded"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "experimental", feature = "multithreaded")))
)]
pub type Prio3FixedPointBoundedL2VecSumMultithreaded<Fx> = Prio3<
    FixedPointBoundedL2VecSum<
        Fx,
        ParallelSumMultithreaded<Field128, PolyEval<Field128>>,
        ParallelSumMultithreaded<Field128, Mul<Field128>>,
    >,
    XofTurboShake128,
    32,
>;

#[cfg(all(feature = "experimental", feature = "multithreaded"))]
impl<Fx: Fixed + CompatibleFloat> Prio3FixedPointBoundedL2VecSumMultithreaded<Fx> {
    /// Construct an instance of this VDAF with the given number of aggregators and number of
    /// vector entries.
    pub fn new_fixedpoint_boundedl2_vec_sum_multithreaded(
        num_aggregators: u8,
        entries: usize,
    ) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;
        Prio3::new(
            num_aggregators,
            1,
            0xFFFF0000,
            FixedPointBoundedL2VecSum::new(entries)?,
        )
    }
}

/// The histogram type. Each measurement is an integer in `[0, length)` and the result is a
/// histogram counting the number of occurrences of each measurement.
pub type Prio3Histogram =
    Prio3<Histogram<Field128, ParallelSum<Field128, Mul<Field128>>>, XofTurboShake128, 32>;

impl Prio3Histogram {
    /// Constructs an instance of Prio3Histogram with the given number of aggregators,
    /// number of buckets, and parallel sum gadget chunk length.
    pub fn new_histogram(
        num_aggregators: u8,
        length: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(
            num_aggregators,
            1,
            0x00000004,
            Histogram::new(length, chunk_length)?,
        )
    }
}

/// Like [`Prio3Histogram`] except this type uses multithreading to improve sharding
/// time. Note that this improvement is only noticeable for very large input lengths.
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3HistogramMultithreaded = Prio3<
    Histogram<Field128, ParallelSumMultithreaded<Field128, Mul<Field128>>>,
    XofTurboShake128,
    32,
>;

#[cfg(feature = "multithreaded")]
impl Prio3HistogramMultithreaded {
    /// Construct an instance of Prio3HistogramMultithreaded with the given number of aggregators,
    /// number of buckets, and parallel sum gadget chunk length.
    pub fn new_histogram_multithreaded(
        num_aggregators: u8,
        length: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(
            num_aggregators,
            1,
            0x00000004,
            Histogram::new(length, chunk_length)?,
        )
    }
}

/// The multihot counter data type. Each measurement is a list of booleans of length `length`, with
/// at most `max_weight` true values, and the aggregate is a histogram counting the number of true
/// values at each position across all measurements.
pub type Prio3MultihotCountVec =
    Prio3<MultihotCountVec<Field128, ParallelSum<Field128, Mul<Field128>>>, XofTurboShake128, 32>;

impl Prio3MultihotCountVec {
    /// Constructs an instance of Prio3MultihotCountVec with the given number of aggregators, number
    /// of buckets, max weight, and parallel sum gadget chunk length.
    pub fn new_multihot_count_vec(
        num_aggregators: u8,
        num_buckets: usize,
        max_weight: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(
            num_aggregators,
            1,
            0x00000005,
            MultihotCountVec::new(num_buckets, max_weight, chunk_length)?,
        )
    }
}

/// Like [`Prio3MultihotCountVec`] except this type uses multithreading to improve sharding
/// time. Note that this improvement is only noticeable for very large input lengths.
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3MultihotCountVecMultithreaded = Prio3<
    MultihotCountVec<Field128, ParallelSumMultithreaded<Field128, Mul<Field128>>>,
    XofTurboShake128,
    32,
>;

#[cfg(feature = "multithreaded")]
impl Prio3MultihotCountVecMultithreaded {
    /// Constructs an instance of Prio3MultihotCountVecMultithreaded with the given number of
    /// aggregators, number of buckets, max weight, and parallel sum gadget chunk length.
    pub fn new_multihot_count_vec_multithreaded(
        num_aggregators: u8,
        num_buckets: usize,
        max_weight: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(
            num_aggregators,
            1,
            0x00000005,
            MultihotCountVec::new(num_buckets, max_weight, chunk_length)?,
        )
    }
}

/// The average type. Each measurement is an integer in `[0, max_measurement]` and
/// the aggregate is the arithmetic average.
pub type Prio3Average = Prio3<Average<Field128>, XofTurboShake128, 32>;

impl Prio3Average {
    /// Construct an instance of `Prio3Average` with the given number of aggregators, where each
    /// summand must be in the range `[0, max_measurement]`. Errors if `max_measurement == 0`.
    pub fn new_average(
        num_aggregators: u8,
        max_measurement: <Field128 as FieldElementWithInteger>::Integer,
    ) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        Ok(Prio3 {
            num_aggregators,
            num_proofs: 1,
            algorithm_id: 0xFFFF0000,
            typ: Average::new(max_measurement)?,
            phantom: PhantomData,
        })
    }
}

/// The base type for Prio3.
///
/// An instance of Prio3 is determined by:
///
/// - a [`Type`] that defines the set of valid input measurements; and
/// - a [`Xof`] for deriving vectors of field elements from seeds.
///
/// New instances can be defined by aliasing the base type. For example, [`Prio3Count`] is an alias
/// for `Prio3<Count<Field64>, XofTurboShake128, 16>`.
///
/// ```
/// use prio::vdaf::{
///     Aggregator, Client, Collector, PrepareTransition,
///     prio3::Prio3,
/// };
/// use rand::{rng, Rng, RngCore};
///
/// let num_shares = 2;
/// let ctx = b"my context str";
/// let vdaf = Prio3::new_count(num_shares).unwrap();
///
/// let mut out_shares = vec![vec![]; num_shares.into()];
/// let mut rng = rng();
/// let verify_key = rng.random();
/// let measurements = [false, true, true, true, false];
/// for measurement in measurements {
///     // Shard
///     let nonce = rng.random::<[u8; 16]>();
///     let (public_share, input_shares) = vdaf.shard(ctx, &measurement, &nonce).unwrap();
///
///     // Prepare
///     let mut prep_states = vec![];
///     let mut prep_shares = vec![];
///     for (agg_id, input_share) in input_shares.iter().enumerate() {
///         let (state, share) = vdaf.prepare_init(
///             &verify_key,
///             ctx,
///             agg_id,
///             &(),
///             &nonce,
///             &public_share,
///             input_share
///         ).unwrap();
///         prep_states.push(state);
///         prep_shares.push(share);
///     }
///     let prep_msg = vdaf.prepare_shares_to_prepare_message(ctx, &(), prep_shares).unwrap();
///
///     for (agg_id, state) in prep_states.into_iter().enumerate() {
///         let out_share = match vdaf.prepare_next(ctx, state, prep_msg.clone()).unwrap() {
///             PrepareTransition::Finish(out_share) => out_share,
///             _ => panic!("unexpected transition"),
///         };
///         out_shares[agg_id].push(out_share);
///     }
/// }
///
/// // Aggregate
/// let agg_shares = out_shares.into_iter()
///     .map(|o| vdaf.aggregate(&(), o).unwrap());
///
/// // Unshard
/// let agg_res = vdaf.unshard(&(), agg_shares, measurements.len()).unwrap();
/// assert_eq!(agg_res, 3);
/// ```
#[derive(Clone, Debug)]
pub struct Prio3<T, P, const SEED_SIZE: usize>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    num_aggregators: u8,
    num_proofs: u8,
    algorithm_id: u32,
    typ: T,
    phantom: PhantomData<P>,
}

impl<T, P, const SEED_SIZE: usize> Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// Construct an instance of this Prio3 VDAF with the given number of aggregators, number of
    /// proofs to generate and verify, the algorithm ID, and the underlying type.
    pub fn new(
        num_aggregators: u8,
        num_proofs: u8,
        algorithm_id: u32,
        typ: T,
    ) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;
        if num_proofs == 0 {
            return Err(VdafError::Uncategorized(
                "num_proofs must be at least 1".to_string(),
            ));
        }

        Ok(Self {
            num_aggregators,
            num_proofs,
            algorithm_id,
            typ,
            phantom: PhantomData,
        })
    }

    /// The output length of the underlying FLP.
    pub fn output_len(&self) -> usize {
        self.typ.output_len()
    }

    /// The verifier length of the underlying FLP.
    pub fn verifier_len(&self) -> usize {
        self.typ.verifier_len()
    }

    #[inline]
    fn num_proofs(&self) -> usize {
        self.num_proofs.into()
    }

    fn derive_prove_rands(&self, ctx: &[u8], prove_rand_seed: &Seed<SEED_SIZE>) -> Vec<T::Field> {
        P::seed_stream(
            prove_rand_seed.as_ref(),
            &[&self.domain_separation_tag(DST_PROVE_RANDOMNESS), ctx],
            &[&[self.num_proofs]],
        )
        .into_field_vec(self.typ.prove_rand_len() * self.num_proofs())
    }

    fn derive_joint_rand_seed<'a>(
        &self,
        ctx: &[u8],
        joint_rand_parts: impl Iterator<Item = &'a Seed<SEED_SIZE>>,
    ) -> Seed<SEED_SIZE> {
        let mut xof = P::init(
            &[0; SEED_SIZE],
            &[&self.domain_separation_tag(DST_JOINT_RAND_SEED), ctx],
        );
        for part in joint_rand_parts {
            xof.update(part.as_ref());
        }
        xof.into_seed()
    }

    fn derive_joint_rands<'a>(
        &self,
        ctx: &[u8],
        joint_rand_parts: impl Iterator<Item = &'a Seed<SEED_SIZE>>,
    ) -> (Seed<SEED_SIZE>, Vec<T::Field>) {
        let joint_rand_seed = self.derive_joint_rand_seed(ctx, joint_rand_parts);
        let joint_rands = P::seed_stream(
            joint_rand_seed.as_ref(),
            &[&self.domain_separation_tag(DST_JOINT_RANDOMNESS), ctx],
            &[&[self.num_proofs]],
        )
        .into_field_vec(self.typ.joint_rand_len() * self.num_proofs());

        (joint_rand_seed, joint_rands)
    }

    fn derive_helper_proofs_share(
        &self,
        ctx: &[u8],
        proofs_share_seed: &Seed<SEED_SIZE>,
        agg_id: u8,
    ) -> Prng<T::Field, P::SeedStream> {
        Prng::from_seed_stream(P::seed_stream(
            proofs_share_seed.as_ref(),
            &[&self.domain_separation_tag(DST_PROOF_SHARE), ctx],
            &[&[self.num_proofs, agg_id]],
        ))
    }

    fn derive_query_rands(
        &self,
        verify_key: &[u8; SEED_SIZE],
        ctx: &[u8],
        nonce: &[u8; 16],
    ) -> Vec<T::Field> {
        let mut xof = P::init(
            verify_key,
            &[&self.domain_separation_tag(DST_QUERY_RANDOMNESS), ctx],
        );
        xof.update(&[self.num_proofs]);
        xof.update(nonce);
        xof.into_seed_stream()
            .into_field_vec(self.typ.query_rand_len() * self.num_proofs())
    }

    /// Generate the domain separation tag for this VDAF. The output is used for domain separation
    /// by the XOF.
    fn domain_separation_tag(&self, usage: u16) -> [u8; 8] {
        // Prefix is 8 bytes and defined by the spec. Copy these values in
        let mut dst = [0; 8];
        dst[0] = VERSION;
        dst[1] = 0; // algorithm class
        dst[2..6].clone_from_slice(self.algorithm_id().to_be_bytes().as_slice());
        dst[6..8].clone_from_slice(usage.to_be_bytes().as_slice());
        dst
    }

    fn random_size(&self) -> usize {
        if self.typ.joint_rand_len() == 0 {
            // One seed per helper (share, proof) pair, plus one seed for proving randomness
            usize::from(self.num_aggregators) * SEED_SIZE
        } else {
            // One seed per helper (share, proof) pair, plus one seed for proving randomness, plus
            // one seed per aggregator for joint randomness blinds
            2 * usize::from(self.num_aggregators) * SEED_SIZE
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn shard_with_random<const N: usize>(
        &self,
        ctx: &[u8],
        measurement: &T::Measurement,
        nonce: &[u8; N],
        random: &[u8],
    ) -> Result<
        (
            Prio3PublicShare<SEED_SIZE>,
            Vec<Prio3InputShare<T::Field, SEED_SIZE>>,
        ),
        VdafError,
    > {
        if random.len() != self.random_size() {
            return Err(VdafError::Uncategorized(
                "incorrect random input length".to_string(),
            ));
        }
        let mut random_seeds = random.chunks_exact(SEED_SIZE);
        let num_aggregators = self.num_aggregators;
        let encoded_measurement = self.typ.encode_measurement(measurement)?;

        // Generate the measurement shares and compute the joint randomness.
        let mut shares_out = Vec::with_capacity(num_aggregators as usize);
        // The first share in the list is the leader's. We'll compute it later. Make a placeholder
        // value
        shares_out.push(Prio3InputShare::Leader {
            measurement_share: Vec::new(),
            proofs_share: Vec::new(),
            joint_rand_blind: None,
        });
        let mut helper_joint_rand_parts = if self.typ.joint_rand_len() > 0 {
            Some(Vec::with_capacity(num_aggregators as usize - 1))
        } else {
            None
        };
        let mut leader_measurement_share = encoded_measurement.clone();
        for agg_id in 1..num_aggregators {
            // The Option from the ChunksExact iterator is okay to unwrap because we checked that
            // the randomness slice is long enough for this VDAF. The slice-to-array conversion
            // Result is okay to unwrap because the ChunksExact iterator always returns slices of
            // the correct length.
            // This seed is used for both the helper measurement share and the helper proof share
            let meas_and_proof_share_seed = random_seeds.next().unwrap().try_into().unwrap();
            let measurement_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &meas_and_proof_share_seed,
                &[&self.domain_separation_tag(DST_MEASUREMENT_SHARE), ctx],
                &[&[agg_id]],
            ));
            let joint_rand_blind = if let Some(helper_joint_rand_parts) =
                helper_joint_rand_parts.as_mut()
            {
                let joint_rand_blind = random_seeds.next().unwrap().try_into().unwrap();
                let mut joint_rand_part_xof = P::init(
                    &joint_rand_blind,
                    &[&self.domain_separation_tag(DST_JOINT_RAND_PART), ctx],
                );
                joint_rand_part_xof.update(&[agg_id]); // Aggregator ID
                joint_rand_part_xof.update(nonce);

                let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                for (x, y) in leader_measurement_share
                    .iter_mut()
                    .zip(measurement_share_prng)
                {
                    *x -= y;
                    y.encode(&mut encoding_buffer).map_err(|_| {
                        VdafError::Uncategorized("failed to encode measurement share".to_string())
                    })?;
                    joint_rand_part_xof.update(&encoding_buffer);
                    encoding_buffer.clear();
                }

                helper_joint_rand_parts.push(joint_rand_part_xof.into_seed());

                Some(joint_rand_blind)
            } else {
                sub_assign_vector(&mut leader_measurement_share, measurement_share_prng);
                None
            };
            shares_out.push(Prio3InputShare::Helper {
                meas_and_proofs_share: Seed::from_bytes(meas_and_proof_share_seed),
                joint_rand_blind: joint_rand_blind.map(Seed::from_bytes),
            });
        }

        let mut leader_blind_opt = None;
        let public_share = Prio3PublicShare {
            joint_rand_parts: helper_joint_rand_parts
                .as_ref()
                .map(
                    |helper_joint_rand_parts| -> Result<Vec<Seed<SEED_SIZE>>, VdafError> {
                        let leader_blind_bytes = random_seeds.next().unwrap().try_into().unwrap();
                        let leader_blind = Seed::from_bytes(leader_blind_bytes);

                        let mut joint_rand_part_xof = P::init(
                            leader_blind.as_ref(),
                            &[&self.domain_separation_tag(DST_JOINT_RAND_PART), ctx],
                        );
                        joint_rand_part_xof.update(&[0]); // Aggregator ID
                        joint_rand_part_xof.update(nonce);
                        let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                        for x in leader_measurement_share.iter() {
                            x.encode(&mut encoding_buffer).map_err(|_| {
                                VdafError::Uncategorized(
                                    "failed to encode measurement share".to_string(),
                                )
                            })?;
                            joint_rand_part_xof.update(&encoding_buffer);
                            encoding_buffer.clear();
                        }
                        leader_blind_opt = Some(leader_blind);

                        let leader_joint_rand_seed_part = joint_rand_part_xof.into_seed();

                        let mut vec = Vec::with_capacity(self.num_aggregators());
                        vec.push(leader_joint_rand_seed_part);
                        vec.extend(helper_joint_rand_parts.iter().cloned());
                        Ok(vec)
                    },
                )
                .transpose()?,
        };

        // Compute the joint randomness.
        let joint_rands = public_share
            .joint_rand_parts
            .as_ref()
            .map(|joint_rand_parts| self.derive_joint_rands(ctx, joint_rand_parts.iter()).1)
            .unwrap_or_default();

        // Generate the proofs.
        let prove_rands = self.derive_prove_rands(
            ctx,
            &Seed::from_bytes(random_seeds.next().unwrap().try_into().unwrap()),
        );
        let mut leader_proofs_share = Vec::with_capacity(self.typ.proof_len() * self.num_proofs());
        for p in 0..self.num_proofs() {
            let prove_rand =
                &prove_rands[p * self.typ.prove_rand_len()..(p + 1) * self.typ.prove_rand_len()];
            let joint_rand =
                &joint_rands[p * self.typ.joint_rand_len()..(p + 1) * self.typ.joint_rand_len()];

            leader_proofs_share.append(&mut self.typ.prove(
                &encoded_measurement,
                prove_rand,
                joint_rand,
            )?);
        }

        // Generate the proof shares and distribute the joint randomness seed hints.
        // Skip the first element, which is reserved for leader share
        for (j, helper) in shares_out.iter_mut().skip(1).enumerate() {
            let prng = self.derive_helper_proofs_share(
                ctx,
                helper.meas_and_proofs_share().unwrap(),
                u8::try_from(j).unwrap() + 1,
            );

            sub_assign_vector(
                &mut leader_proofs_share,
                prng.take(self.typ.proof_len() * self.num_proofs()),
            );
        }

        // Overwrite the placeholder first element with the leader share
        shares_out[0] = Prio3InputShare::Leader {
            measurement_share: leader_measurement_share,
            proofs_share: leader_proofs_share,
            joint_rand_blind: leader_blind_opt,
        };

        Ok((public_share, shares_out))
    }

    fn role_try_from(&self, agg_id: usize) -> Result<u8, VdafError> {
        if agg_id >= self.num_aggregators as usize {
            return Err(VdafError::Uncategorized("unexpected aggregator id".into()));
        }
        Ok(u8::try_from(agg_id).unwrap())
    }
}

impl<T, P, const SEED_SIZE: usize> Vdaf for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    type Measurement = T::Measurement;
    type AggregateResult = T::AggregateResult;
    type AggregationParam = ();
    type PublicShare = Prio3PublicShare<SEED_SIZE>;
    type InputShare = Prio3InputShare<T::Field, SEED_SIZE>;
    type OutputShare = OutputShare<T::Field>;
    type AggregateShare = AggregateShare<T::Field>;

    fn algorithm_id(&self) -> u32 {
        self.algorithm_id
    }

    fn num_aggregators(&self) -> usize {
        self.num_aggregators as usize
    }
}

/// Message broadcast by the [`Client`] to every [`Aggregator`] during the Sharding phase.
#[derive(Clone, Debug)]
pub struct Prio3PublicShare<const SEED_SIZE: usize> {
    /// Contributions to the joint randomness from every aggregator's share.
    joint_rand_parts: Option<Vec<Seed<SEED_SIZE>>>,
}

impl<const SEED_SIZE: usize> Encode for Prio3PublicShare<SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        if let Some(joint_rand_parts) = self.joint_rand_parts.as_ref() {
            for part in joint_rand_parts.iter() {
                part.encode(bytes)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        if let Some(joint_rand_parts) = self.joint_rand_parts.as_ref() {
            // Each seed has the same size.
            Some(SEED_SIZE * joint_rand_parts.len())
        } else {
            Some(0)
        }
    }
}

impl<const SEED_SIZE: usize> PartialEq for Prio3PublicShare<SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<const SEED_SIZE: usize> Eq for Prio3PublicShare<SEED_SIZE> {}

impl<const SEED_SIZE: usize> ConstantTimeEq for Prio3PublicShare<SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_parts.
        option_ct_eq(
            self.joint_rand_parts.as_deref(),
            other.joint_rand_parts.as_deref(),
        )
    }
}

impl<T, P, const SEED_SIZE: usize> ParameterizedDecode<Prio3<T, P, SEED_SIZE>>
    for Prio3PublicShare<SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        decoding_parameter: &Prio3<T, P, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if decoding_parameter.typ.joint_rand_len() > 0 {
            let joint_rand_parts = iter::repeat_with(|| Seed::<SEED_SIZE>::decode(bytes))
                .take(decoding_parameter.num_aggregators.into())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self {
                joint_rand_parts: Some(joint_rand_parts),
            })
        } else {
            Ok(Self {
                joint_rand_parts: None,
            })
        }
    }
}

/// Message sent by the [`Client`] to each [`Aggregator`] during the Sharding phase.
#[derive(Clone, Debug)]
pub enum Prio3InputShare<F, const SEED_SIZE: usize> {
    /// The leader share. Includes the measurement share, the proof share, and the blinding seed
    Leader {
        /// The share for measurement
        measurement_share: Vec<F>,

        /// The proof share.
        proofs_share: Vec<F>,

        /// Blinding seed used by the Aggregator to compute the joint randomness. This field is
        /// optional because not every [`Type`] requires joint randomness.
        joint_rand_blind: Option<Seed<SEED_SIZE>>,
    },

    /// The helper share. Includes the seed for measurement/proofs share and the blinding seed
    Helper {
        /// The share for measurement and proof
        meas_and_proofs_share: Seed<SEED_SIZE>,

        /// Blinding seed used by the Aggregator to compute the joint randomness. This field is
        /// optional because not every [`Type`] requires joint randomness.
        joint_rand_blind: Option<Seed<SEED_SIZE>>,
    },
}

impl<F: FieldElement, const SEED_SIZE: usize> Prio3InputShare<F, SEED_SIZE> {
    /// Returns the measurement/proofs seed if this share a helper share, otherwise `None`
    fn meas_and_proofs_share(&self) -> Option<&Seed<SEED_SIZE>> {
        match self {
            Prio3InputShare::Leader { .. } => None,
            Prio3InputShare::Helper {
                meas_and_proofs_share,
                ..
            } => Some(meas_and_proofs_share),
        }
    }

    /// Returns the joint randomness blinding seed if there is one, otherwise `None`
    fn joint_rand_blind(&self) -> Option<&Seed<SEED_SIZE>> {
        match self {
            Prio3InputShare::Leader {
                joint_rand_blind, ..
            } => joint_rand_blind.as_ref(),
            Prio3InputShare::Helper {
                joint_rand_blind, ..
            } => joint_rand_blind.as_ref(),
        }
    }

    /// Returns the measurement share
    fn measurement_share(&self) -> Share<F, SEED_SIZE> {
        match self {
            Prio3InputShare::Leader {
                measurement_share, ..
            } => Share::Leader(measurement_share.to_vec()),
            Prio3InputShare::Helper {
                meas_and_proofs_share,
                ..
            } => Share::Helper(meas_and_proofs_share.clone()),
        }
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Prio3InputShare<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Prio3InputShare<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Prio3InputShare<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        match (self, other) {
            (
                Prio3InputShare::Leader {
                    measurement_share,
                    proofs_share,
                    joint_rand_blind,
                },
                Prio3InputShare::Leader {
                    measurement_share: other_measurement_share,
                    proofs_share: other_proofs_share,
                    joint_rand_blind: other_joint_rand_blind,
                },
            ) => {
                // We allow short-circuiting on the presence or absence of the joint_rand_blind.
                option_ct_eq(joint_rand_blind.as_ref(), other_joint_rand_blind.as_ref())
                    & measurement_share.ct_eq(other_measurement_share)
                    & proofs_share.ct_eq(other_proofs_share)
            }
            (
                Prio3InputShare::Helper {
                    meas_and_proofs_share,
                    joint_rand_blind,
                },
                Prio3InputShare::Helper {
                    meas_and_proofs_share: other_meas_and_proofs_share,
                    joint_rand_blind: other_joint_rand_blind,
                },
            ) => {
                option_ct_eq(joint_rand_blind.as_ref(), other_joint_rand_blind.as_ref())
                    & meas_and_proofs_share.ct_eq(other_meas_and_proofs_share)
            }
            // Different share types are never equal
            _ => Choice::from(0),
        }
    }
}

impl<F: NttFriendlyFieldElement, const SEED_SIZE: usize> Encode for Prio3InputShare<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match self {
            Prio3InputShare::Leader {
                measurement_share,
                proofs_share,
                joint_rand_blind,
            } => {
                encode_fixlen_items(bytes, measurement_share)?;
                encode_fixlen_items(bytes, proofs_share)?;
                if let Some(ref blind) = joint_rand_blind {
                    blind.encode(bytes)?;
                }
            }
            Prio3InputShare::Helper {
                meas_and_proofs_share,
                joint_rand_blind,
            } => {
                meas_and_proofs_share.encode(bytes)?;
                if let Some(ref blind) = joint_rand_blind {
                    blind.encode(bytes)?;
                }
            }
        }

        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            Prio3InputShare::Leader {
                measurement_share,
                proofs_share,
                joint_rand_blind,
            } => {
                let measurement_share_encoded_len = measurement_share.len() * F::ENCODED_SIZE;
                let proofs_share_encoded_len = proofs_share.len() * F::ENCODED_SIZE;
                let blind_len = if let Some(ref blind) = joint_rand_blind {
                    blind.encoded_len()?
                } else {
                    0
                };

                Some(measurement_share_encoded_len + proofs_share_encoded_len + blind_len)
            }
            Prio3InputShare::Helper {
                meas_and_proofs_share,
                joint_rand_blind,
            } => {
                let mut len = meas_and_proofs_share.encoded_len()?;
                if let Some(ref blind) = joint_rand_blind {
                    len += blind.encoded_len()?;
                }
                Some(len)
            }
        }
    }
}

impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, usize)>
    for Prio3InputShare<T::Field, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let agg_id = prio3
            .role_try_from(*agg_id)
            .map_err(|e| CodecError::Other(Box::new(e)))?;

        if agg_id == 0 {
            let measurement_share = decode_fieldvec(prio3.typ.input_len(), bytes)?;
            let proofs_share = decode_fieldvec(prio3.typ.proof_len() * prio3.num_proofs(), bytes)?;

            let joint_rand_blind = if prio3.typ.joint_rand_len() > 0 {
                let blind = Seed::decode(bytes)?;
                Some(blind)
            } else {
                None
            };

            Ok(Prio3InputShare::Leader {
                measurement_share,
                proofs_share,
                joint_rand_blind,
            })
        } else {
            let meas_and_proofs_share = Seed::decode(bytes)?;
            let joint_rand_blind = if prio3.typ.joint_rand_len() > 0 {
                let blind = Seed::decode(bytes)?;
                Some(blind)
            } else {
                None
            };

            Ok(Prio3InputShare::Helper {
                meas_and_proofs_share,
                joint_rand_blind,
            })
        }
    }
}

#[derive(Clone, Debug)]
/// Message broadcast by each [`Aggregator`] in each round of the Preparation phase.
pub struct Prio3PrepareShare<F, const SEED_SIZE: usize> {
    /// A share of the FLP verifier message. (See [`Type`].)
    verifiers: Vec<F>,

    /// A part of the joint randomness seed.
    joint_rand_part: Option<Seed<SEED_SIZE>>,
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Prio3PrepareShare<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Prio3PrepareShare<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Prio3PrepareShare<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_part.
        option_ct_eq(
            self.joint_rand_part.as_ref(),
            other.joint_rand_part.as_ref(),
        ) & self.verifiers.ct_eq(&other.verifiers)
    }
}

impl<F: NttFriendlyFieldElement, const SEED_SIZE: usize> Encode
    for Prio3PrepareShare<F, SEED_SIZE>
{
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        for x in &self.verifiers {
            x.encode(bytes)?;
        }
        if let Some(ref seed) = self.joint_rand_part {
            seed.encode(bytes)?;
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        // Each element of the verifier has the same size.
        let mut len = F::ENCODED_SIZE * self.verifiers.len();
        if let Some(ref seed) = self.joint_rand_part {
            len += seed.encoded_len()?;
        }
        Some(len)
    }
}

impl<F: NttFriendlyFieldElement, const SEED_SIZE: usize>
    ParameterizedDecode<Prio3PrepareState<F, SEED_SIZE>> for Prio3PrepareShare<F, SEED_SIZE>
{
    fn decode_with_param(
        decoding_parameter: &Prio3PrepareState<F, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let mut verifiers = Vec::with_capacity(decoding_parameter.verifiers_len);
        for _ in 0..decoding_parameter.verifiers_len {
            verifiers.push(F::decode(bytes)?);
        }

        let joint_rand_part = if decoding_parameter.joint_rand_seed.is_some() {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Prio3PrepareShare {
            verifiers,
            joint_rand_part,
        })
    }
}

#[derive(Clone, Debug)]
/// Result of combining a round of [`Prio3PrepareShare`] messages.
pub struct Prio3PrepareMessage<const SEED_SIZE: usize> {
    /// The joint randomness seed computed by the Aggregators.
    joint_rand_seed: Option<Seed<SEED_SIZE>>,
}

impl<const SEED_SIZE: usize> PartialEq for Prio3PrepareMessage<SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<const SEED_SIZE: usize> Eq for Prio3PrepareMessage<SEED_SIZE> {}

impl<const SEED_SIZE: usize> ConstantTimeEq for Prio3PrepareMessage<SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presnce or absence of the joint_rand_seed.
        option_ct_eq(
            self.joint_rand_seed.as_ref(),
            other.joint_rand_seed.as_ref(),
        )
    }
}

impl<const SEED_SIZE: usize> Encode for Prio3PrepareMessage<SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encode(bytes)?;
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encoded_len()
        } else {
            Some(0)
        }
    }
}

impl<F: NttFriendlyFieldElement, const SEED_SIZE: usize>
    ParameterizedDecode<Prio3PrepareState<F, SEED_SIZE>> for Prio3PrepareMessage<SEED_SIZE>
{
    fn decode_with_param(
        decoding_parameter: &Prio3PrepareState<F, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let joint_rand_seed = if decoding_parameter.joint_rand_seed.is_some() {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Prio3PrepareMessage { joint_rand_seed })
    }
}

impl<T, P, const SEED_SIZE: usize> Client<16> for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    #[allow(clippy::type_complexity)]
    fn shard(
        &self,
        ctx: &[u8],
        measurement: &T::Measurement,
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Prio3InputShare<T::Field, SEED_SIZE>>), VdafError> {
        let mut random = vec![0u8; self.random_size()];
        rng().fill(&mut random[..]);
        self.shard_with_random(ctx, measurement, nonce, &random)
    }
}

/// State of each [`Aggregator`] during the Preparation phase.
#[derive(Clone)]
pub struct Prio3PrepareState<F, const SEED_SIZE: usize> {
    /// An uncompressed output share, or a compressed measurement share.
    ///
    /// Note that helpers will need to pass the measurement share through the FLP's `truncate()`
    /// procedure after expanding it from their seed.
    share: Share<F, SEED_SIZE>,
    joint_rand_seed: Option<Seed<SEED_SIZE>>,
    agg_id: u8,
    verifiers_len: usize,
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Prio3PrepareState<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Prio3PrepareState<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Prio3PrepareState<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_seed, as well as
        // the aggregator ID & verifier length parameters.
        if self.agg_id != other.agg_id || self.verifiers_len != other.verifiers_len {
            return Choice::from(0);
        }

        option_ct_eq(
            self.joint_rand_seed.as_ref(),
            other.joint_rand_seed.as_ref(),
        ) & self.share.ct_eq(&other.share)
    }
}

impl<F, const SEED_SIZE: usize> Debug for Prio3PrepareState<F, SEED_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prio3PrepareState")
            .field("share", &"[redacted]")
            .field(
                "joint_rand_seed",
                match self.joint_rand_seed {
                    Some(_) => &"Some([redacted])",
                    None => &"None",
                },
            )
            .field("agg_id", &self.agg_id)
            .field("verifiers_len", &self.verifiers_len)
            .finish()
    }
}

impl<F: NttFriendlyFieldElement, const SEED_SIZE: usize> Encode
    for Prio3PrepareState<F, SEED_SIZE>
{
    /// Append the encoded form of this object to the end of `bytes`, growing the vector as needed.
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.share.encode(bytes)?;
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encode(bytes)?;
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut len = self.share.encoded_len()?;
        if let Some(ref seed) = self.joint_rand_seed {
            len += seed.encoded_len()?;
        }
        Some(len)
    }
}

impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, usize)>
    for Prio3PrepareState<T::Field, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let agg_id = prio3
            .role_try_from(*agg_id)
            .map_err(|e| CodecError::Other(Box::new(e)))?;

        let share_decoder = if agg_id == 0 {
            ShareDecodingParameter::Leader(prio3.typ.output_len())
        } else {
            ShareDecodingParameter::Helper
        };
        let share = Share::decode_with_param(&share_decoder, bytes)?;

        let joint_rand_seed = if prio3.typ.joint_rand_len() > 0 {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Self {
            share,
            joint_rand_seed,
            agg_id,
            verifiers_len: prio3.typ.verifier_len() * prio3.num_proofs(),
        })
    }
}

impl<T, P, const SEED_SIZE: usize> Aggregator<SEED_SIZE, 16> for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    type PrepareState = Prio3PrepareState<T::Field, SEED_SIZE>;
    type PrepareShare = Prio3PrepareShare<T::Field, SEED_SIZE>;
    type PrepareMessage = Prio3PrepareMessage<SEED_SIZE>;

    /// Begins the Prep process with the other aggregators. The result of this process is
    /// the aggregator's output share.
    #[allow(clippy::type_complexity)]
    fn prepare_init(
        &self,
        verify_key: &[u8; SEED_SIZE],
        ctx: &[u8],
        agg_id: usize,
        _agg_param: &Self::AggregationParam,
        nonce: &[u8; 16],
        public_share: &Self::PublicShare,
        msg: &Prio3InputShare<T::Field, SEED_SIZE>,
    ) -> Result<
        (
            Prio3PrepareState<T::Field, SEED_SIZE>,
            Prio3PrepareShare<T::Field, SEED_SIZE>,
        ),
        VdafError,
    > {
        let agg_id = self.role_try_from(agg_id)?;

        let (measurement_share, proofs_share) = match msg {
            Prio3InputShare::Leader {
                measurement_share,
                proofs_share,
                ..
            } => {
                let measurement_share = Cow::Borrowed(measurement_share);
                let proof_share = Cow::Borrowed(proofs_share);
                (measurement_share, proof_share)
            }
            Prio3InputShare::Helper {
                meas_and_proofs_share,
                ..
            } => {
                let measurement_share = Cow::Owned(
                    P::seed_stream(
                        meas_and_proofs_share.as_ref(),
                        &[&self.domain_separation_tag(DST_MEASUREMENT_SHARE), ctx],
                        &[&[agg_id]],
                    )
                    .into_field_vec(self.typ.input_len()),
                );
                let proof_share = Cow::Owned(
                    self.derive_helper_proofs_share(ctx, meas_and_proofs_share, agg_id)
                        .take(self.typ.proof_len() * self.num_proofs())
                        .collect::<Vec<_>>(),
                );
                (measurement_share, proof_share)
            }
        };

        // Compute the joint randomness.
        let (joint_rand_seed, joint_rand_part, joint_rands) = if self.typ.joint_rand_len() > 0 {
            let mut joint_rand_part_xof = P::init(
                msg.joint_rand_blind().as_ref().unwrap().as_ref(),
                &[&self.domain_separation_tag(DST_JOINT_RAND_PART), ctx],
            );
            joint_rand_part_xof.update(&[agg_id]);
            joint_rand_part_xof.update(nonce);
            let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
            for x in measurement_share.iter() {
                x.encode(&mut encoding_buffer).map_err(|_| {
                    VdafError::Uncategorized("failed to encode measurement share".to_string())
                })?;
                joint_rand_part_xof.update(&encoding_buffer);
                encoding_buffer.clear();
            }
            let own_joint_rand_part = joint_rand_part_xof.into_seed();

            // Make an iterator over the joint randomness parts, but use this aggregator's
            // contribution, computed from the input share, in lieu of the the corresponding part
            // from the public share.
            //
            // The locally computed part should match the part from the public share for honestly
            // generated reports. If they do not match, the joint randomness seed check during the
            // next round of preparation should fail.
            let corrected_joint_rand_parts = public_share
                .joint_rand_parts
                .iter()
                .flatten()
                .take(agg_id as usize)
                .chain(iter::once(&own_joint_rand_part))
                .chain(
                    public_share
                        .joint_rand_parts
                        .iter()
                        .flatten()
                        .skip(agg_id as usize + 1),
                );

            let (joint_rand_seed, joint_rands) =
                self.derive_joint_rands(ctx, corrected_joint_rand_parts);

            (
                Some(joint_rand_seed),
                Some(own_joint_rand_part),
                joint_rands,
            )
        } else {
            (None, None, Vec::new())
        };

        // Run the query-generation algorithm.
        let query_rands = self.derive_query_rands(verify_key, ctx, nonce);
        let mut verifiers_share = Vec::with_capacity(self.typ.verifier_len() * self.num_proofs());
        for p in 0..self.num_proofs() {
            let query_rand =
                &query_rands[p * self.typ.query_rand_len()..(p + 1) * self.typ.query_rand_len()];
            let joint_rand =
                &joint_rands[p * self.typ.joint_rand_len()..(p + 1) * self.typ.joint_rand_len()];
            let proof_share =
                &proofs_share[p * self.typ.proof_len()..(p + 1) * self.typ.proof_len()];

            verifiers_share.append(&mut self.typ.query(
                measurement_share.as_ref(),
                proof_share,
                query_rand,
                joint_rand,
                self.num_aggregators as usize,
            )?);
        }

        let state_share = match msg.measurement_share() {
            Share::Leader(measurement_share) => {
                Share::Leader(self.typ.truncate(measurement_share)?)
            }
            Share::Helper(seed) => Share::Helper(seed),
        };

        Ok((
            Prio3PrepareState {
                share: state_share,
                joint_rand_seed,
                agg_id,
                verifiers_len: verifiers_share.len(),
            },
            Prio3PrepareShare {
                verifiers: verifiers_share,
                joint_rand_part,
            },
        ))
    }

    fn prepare_shares_to_prepare_message<
        M: IntoIterator<Item = Prio3PrepareShare<T::Field, SEED_SIZE>>,
    >(
        &self,
        ctx: &[u8],
        _: &Self::AggregationParam,
        inputs: M,
    ) -> Result<Prio3PrepareMessage<SEED_SIZE>, VdafError> {
        let mut verifiers = vec![T::Field::zero(); self.typ.verifier_len() * self.num_proofs()];
        let mut joint_rand_parts = Vec::with_capacity(self.num_aggregators());
        let mut count = 0;
        for share in inputs.into_iter() {
            count += 1;

            if share.verifiers.len() != verifiers.len() {
                return Err(VdafError::Uncategorized(format!(
                    "unexpected verifier share length: got {}; want {}",
                    share.verifiers.len(),
                    verifiers.len(),
                )));
            }

            if self.typ.joint_rand_len() > 0 {
                let joint_rand_seed_part = share.joint_rand_part.unwrap();
                joint_rand_parts.push(joint_rand_seed_part);
            }

            add_assign_vector(&mut verifiers, share.verifiers.iter().copied());
        }

        if count != self.num_aggregators {
            return Err(VdafError::Uncategorized(format!(
                "unexpected message count: got {}; want {}",
                count, self.num_aggregators,
            )));
        }

        // Check the proof verifiers.
        for verifier in verifiers.chunks(self.typ.verifier_len()) {
            if !self.typ.decide(verifier)? {
                return Err(VdafError::Uncategorized(
                    "proof verifier check failed".into(),
                ));
            }
        }

        let joint_rand_seed = if self.typ.joint_rand_len() > 0 {
            Some(self.derive_joint_rand_seed(ctx, joint_rand_parts.iter()))
        } else {
            None
        };

        Ok(Prio3PrepareMessage { joint_rand_seed })
    }

    fn prepare_next(
        &self,
        ctx: &[u8],
        step: Prio3PrepareState<T::Field, SEED_SIZE>,
        msg: Prio3PrepareMessage<SEED_SIZE>,
    ) -> Result<PrepareTransition<Self, SEED_SIZE, 16>, VdafError> {
        if self.typ.joint_rand_len() > 0 {
            // Check that the joint randomness was correct.
            if step
                .joint_rand_seed
                .as_ref()
                .unwrap()
                .ct_ne(msg.joint_rand_seed.as_ref().unwrap())
                .into()
            {
                return Err(VdafError::Uncategorized(
                    "joint randomness mismatch".to_string(),
                ));
            }
        }

        // Compute the output share.
        let output_share = match step.share {
            Share::Leader(data) => data,
            Share::Helper(seed) => {
                let measurement_share = P::seed_stream(
                    seed.as_ref(),
                    &[&self.domain_separation_tag(DST_MEASUREMENT_SHARE), ctx],
                    &[&[step.agg_id]],
                )
                .into_field_vec(self.typ.input_len());
                self.typ.truncate(measurement_share)?
            }
        };

        Ok(PrepareTransition::Finish(OutputShare(output_share)))
    }

    fn aggregate_init(&self, _agg_param: &Self::AggregationParam) -> Self::AggregateShare {
        AggregateShare(vec![T::Field::zero(); self.typ.output_len()])
    }

    /// Returns `true` iff `prev.is_empty()`
    fn is_agg_param_valid(_cur: &Self::AggregationParam, prev: &[Self::AggregationParam]) -> bool {
        prev.is_empty()
    }
}

#[cfg(feature = "experimental")]
impl<T, P, S, const SEED_SIZE: usize> AggregatorWithNoise<SEED_SIZE, 16, S>
    for Prio3<T, P, SEED_SIZE>
where
    T: TypeWithNoise<S>,
    P: Xof<SEED_SIZE>,
    S: DifferentialPrivacyStrategy,
{
    fn add_noise_to_agg_share(
        &self,
        dp_strategy: &S,
        _agg_param: &Self::AggregationParam,
        agg_share: &mut Self::AggregateShare,
        num_measurements: usize,
    ) -> Result<(), VdafError> {
        self.typ
            .add_noise_to_agg_share(dp_strategy, &mut agg_share.0, num_measurements)?;
        Ok(())
    }
}

impl<T, P, const SEED_SIZE: usize> Collector for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// Combines aggregate shares into the aggregate result.
    fn unshard<It: IntoIterator<Item = AggregateShare<T::Field>>>(
        &self,
        _agg_param: &Self::AggregationParam,
        agg_shares: It,
        num_measurements: usize,
    ) -> Result<T::AggregateResult, VdafError> {
        let mut agg = AggregateShare(vec![T::Field::zero(); self.typ.output_len()]);
        for agg_share in agg_shares.into_iter() {
            agg.merge(&agg_share)?;
        }

        Ok(self.typ.decode_result(&agg.0, num_measurements)?)
    }
}

fn check_num_aggregators(num_aggregators: u8) -> Result<(), VdafError> {
    if num_aggregators == 0 {
        return Err(VdafError::Uncategorized(format!(
            "at least one aggregator is required; got {num_aggregators}"
        )));
    } else if num_aggregators > 254 {
        return Err(VdafError::Uncategorized(format!(
            "number of aggregators must not exceed 254; got {num_aggregators}"
        )));
    }

    Ok(())
}

impl<'a, F, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, &'a ())>
    for OutputShare<F>
where
    F: FieldElement,
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (vdaf, _): &(&'a Prio3<T, P, SEED_SIZE>, &'a ()),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        decode_fieldvec(vdaf.output_len(), bytes).map(Self)
    }
}

impl<'a, F, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, &'a ())>
    for AggregateShare<F>
where
    F: FieldElement,
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (vdaf, _): &(&'a Prio3<T, P, SEED_SIZE>, &'a ()),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        decode_fieldvec(vdaf.output_len(), bytes).map(Self)
    }
}

/// This function determines equality between two optional, constant-time comparable values. It
/// short-circuits on the existence (but not contents) of the values -- a timing side-channel may
/// reveal whether the values match on Some or None.
#[inline]
fn option_ct_eq<T>(left: Option<&T>, right: Option<&T>) -> Choice
where
    T: ConstantTimeEq + ?Sized,
{
    match (left, right) {
        (Some(left), Some(right)) => left.ct_eq(right),
        (None, None) => Choice::from(1),
        _ => Choice::from(0),
    }
}

/// Finds the optimal choice of chunk length for [`Prio3Histogram`] or [`Prio3SumVec`], given its
/// encoded measurement length. For [`Prio3Histogram`], the measurement length is equal to the
/// length parameter. For [`Prio3SumVec`], the measurement length is equal to the product of the
/// length and bits parameters.
pub fn optimal_chunk_length(measurement_length: usize) -> usize {
    if measurement_length <= 1 {
        return 1;
    }

    /// Candidate set of parameter choices for the parallel sum optimization.
    struct Candidate {
        gadget_calls: usize,
        chunk_length: usize,
    }

    let max_log2 = (measurement_length + 1).ilog2();
    let best_opt = (1..=max_log2)
        .rev()
        .map(|log2| {
            let gadget_calls = (1 << log2) - 1;
            let chunk_length = measurement_length.div_ceil(gadget_calls);
            Candidate {
                gadget_calls,
                chunk_length,
            }
        })
        .min_by_key(|candidate| {
            // Compute the proof length, in field elements, for either Prio3Histogram or Prio3SumVec
            (candidate.chunk_length * 2)
                + 2 * ((1 + candidate.gadget_calls).next_power_of_two() - 1)
        });
    // Unwrap safety: max_log2 must be at least 1, because smaller measurement_length inputs are
    // dealt with separately. Thus, the range iterator that the search is over will be nonempty,
    // and min_by_key() will always return Some.
    best_opt.unwrap().chunk_length
}

/// Utilities for testing Prio3.
#[cfg(feature = "test-util")]
pub mod test_utils {
    use crate::{
        field::Field64,
        flp::types::test_utils::HigherDegree,
        vdaf::{prio3::Prio3, xof::XofTurboShake128, VdafError},
    };

    /// A VDAF for testing purposes.
    ///
    /// This uses a gadget with a configurable degree and number of calls.
    pub type Prio3HigherDegree = Prio3<HigherDegree<Field64>, XofTurboShake128, 32>;

    impl Prio3HigherDegree {
        /// Construct an instance of `Prio3HigherDegree` with the given parameters.
        pub fn new_higher_degree(
            num_aggregators: u8,
            degree: usize,
            gadget_calls: usize,
        ) -> Result<Self, VdafError> {
            Prio3::new(
                num_aggregators,
                1,
                0xFFFFFFFF,
                HigherDegree::new(degree, gadget_calls)?,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "experimental")]
    use crate::flp::gadgets::ParallelSumGadget;
    use crate::{
        flp::Flp,
        vdaf::{
            equality_comparison_test, fieldvec_roundtrip_test,
            test_utils::{run_vdaf, run_vdaf_prepare},
        },
    };
    use assert_matches::assert_matches;
    #[cfg(feature = "experimental")]
    use fixed::{
        types::{
            extra::{U15, U31, U63},
            I1F15, I1F31, I1F63,
        },
        FixedI16, FixedI32, FixedI64,
    };

    const CTX_STR: &[u8] = b"prio3 ctx";

    impl<F: FieldElement, const SEED_SIZE: usize> Prio3InputShare<F, SEED_SIZE> {
        fn proofs_share(&self) -> Share<F, SEED_SIZE> {
            match self {
                Prio3InputShare::Leader { proofs_share, .. } => {
                    Share::Leader(proofs_share.to_vec())
                }
                Prio3InputShare::Helper {
                    meas_and_proofs_share,
                    ..
                } => Share::Helper(meas_and_proofs_share.clone()),
            }
        }

        // Needed for some feature-gated tests
        #[cfg(feature = "experimental")]
        fn joint_rand_blind_mut(&mut self) -> Option<&mut Seed<SEED_SIZE>> {
            match self {
                Prio3InputShare::Leader {
                    ref mut joint_rand_blind,
                    ..
                } => joint_rand_blind.as_mut(),
                Prio3InputShare::Helper {
                    ref mut joint_rand_blind,
                    ..
                } => joint_rand_blind.as_mut(),
            }
        }
    }

    #[test]
    fn test_prio3_count() {
        let prio3 = Prio3::new_count(2).unwrap();

        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [true, false, false, true, true]).unwrap(),
            3
        );

        let mut nonce = [0; 16];
        let mut verify_key = [0; 32];
        rng().fill(&mut verify_key[..]);
        rng().fill(&mut nonce[..]);

        let (public_share, input_shares) = prio3.shard(CTX_STR, &false, &nonce).unwrap();
        run_vdaf_prepare(
            &prio3,
            &verify_key,
            CTX_STR,
            &(),
            &nonce,
            public_share,
            input_shares,
        )
        .unwrap();

        let (public_share, input_shares) = prio3.shard(CTX_STR, &true, &nonce).unwrap();
        run_vdaf_prepare(
            &prio3,
            &verify_key,
            CTX_STR,
            &(),
            &nonce,
            public_share,
            input_shares,
        )
        .unwrap();

        test_serialization(&prio3, &true, &nonce).unwrap();

        let prio3_extra_helper = Prio3::new_count(3).unwrap();
        assert_eq!(
            run_vdaf(
                CTX_STR,
                &prio3_extra_helper,
                &(),
                [true, false, false, true, true]
            )
            .unwrap(),
            3,
        );
    }

    #[test]
    fn test_prio3_sum() {
        let max_measurement = 35_891;

        let prio3 = Prio3::new_sum(3, max_measurement).unwrap();

        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [0, max_measurement, 0, 1, 1]).unwrap(),
            max_measurement + 2,
        );

        let mut verify_key = [0; 32];
        rng().fill(&mut verify_key[..]);
        let nonce = [0; 16];

        let (public_share, mut input_shares) = prio3.shard(CTX_STR, &1, &nonce).unwrap();
        assert_matches!(
            &mut input_shares[0],
            Prio3InputShare::Leader { ref mut measurement_share, ..} => {
                measurement_share[0] += Field64::one();
            }
        );
        let result = run_vdaf_prepare(
            &prio3,
            &verify_key,
            CTX_STR,
            &(),
            &nonce,
            public_share,
            input_shares,
        );
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let (public_share, mut input_shares) = prio3.shard(CTX_STR, &1, &nonce).unwrap();
        assert_matches!(
            &mut input_shares[0],
            Prio3InputShare::Leader { ref mut proofs_share, ..} => {
                proofs_share[0] += Field64::one();
            }
        );
        let result = run_vdaf_prepare(
            &prio3,
            &verify_key,
            CTX_STR,
            &(),
            &nonce,
            public_share,
            input_shares,
        );
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        test_serialization(&prio3, &1, &nonce).unwrap();
    }

    #[test]
    fn test_prio3_sum_vec() {
        let prio3 = Prio3::new_sum_vec(2, 2, 20, 4).unwrap();
        assert_eq!(
            run_vdaf(
                CTX_STR,
                &prio3,
                &(),
                [
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    vec![0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0],
                    vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                ]
            )
            .unwrap(),
            vec![1, 3, 1, 0, 3, 1, 0, 1, 2, 2, 3, 3, 1, 5, 1, 2, 1, 3, 0, 2],
        );
    }

    #[test]
    fn test_prio3_sum_vec_multiproof() {
        let prio3 = Prio3::<
            SumVec<Field128, ParallelSum<Field128, Mul<Field128>>>,
            XofTurboShake128,
            32,
        >::new(2, 2, 0xFFFF0000, SumVec::new(2, 20, 4).unwrap())
        .unwrap();

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &prio3,
                &(),
                [
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    vec![0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0],
                    vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                ]
            )
            .unwrap(),
            vec![1, 3, 1, 0, 3, 1, 0, 1, 2, 2, 3, 3, 1, 5, 1, 2, 1, 3, 0, 2],
        );
    }

    #[test]
    #[cfg(feature = "multithreaded")]
    fn test_prio3_sum_vec_multithreaded() {
        let prio3 = Prio3::new_sum_vec_multithreaded(2, 2, 20, 4).unwrap();
        assert_eq!(
            run_vdaf(
                CTX_STR,
                &prio3,
                &(),
                [
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    vec![0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0],
                    vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                ]
            )
            .unwrap(),
            vec![1, 3, 1, 0, 3, 1, 0, 1, 2, 2, 3, 3, 1, 5, 1, 2, 1, 3, 0, 2],
        );
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn test_prio3_bounded_fpvec_sum_unaligned() {
        type P<Fx> = Prio3FixedPointBoundedL2VecSum<Fx>;
        #[cfg(feature = "multithreaded")]
        type PM<Fx> = Prio3FixedPointBoundedL2VecSumMultithreaded<Fx>;
        let ctor_32 = P::<FixedI32<U31>>::new_fixedpoint_boundedl2_vec_sum;
        #[cfg(feature = "multithreaded")]
        let ctor_mt_32 = PM::<FixedI32<U31>>::new_fixedpoint_boundedl2_vec_sum_multithreaded;

        {
            const SIZE: usize = 5;
            const FP32_0: I1F31 = I1F31::lit("0");

            // 32 bit fixedpoint, non-power-of-2 vector, single-threaded
            {
                let prio3_32 = ctor_32(2, SIZE).unwrap();
                test_fixed_vec::<_, _, _, SIZE>(FP32_0, prio3_32);
            }

            // 32 bit fixedpoint, non-power-of-2 vector, multi-threaded
            #[cfg(feature = "multithreaded")]
            {
                let prio3_mt_32 = ctor_mt_32(2, SIZE).unwrap();
                test_fixed_vec::<_, _, _, SIZE>(FP32_0, prio3_mt_32);
            }
        }

        fn test_fixed_vec<Fx, PE, M, const SIZE: usize>(
            fp_0: Fx,
            prio3: Prio3<FixedPointBoundedL2VecSum<Fx, PE, M>, XofTurboShake128, 32>,
        ) where
            Fx: Fixed + CompatibleFloat + std::ops::Neg<Output = Fx>,
            PE: Eq + ParallelSumGadget<Field128, PolyEval<Field128>> + Clone + 'static,
            M: Eq + ParallelSumGadget<Field128, Mul<Field128>> + Clone + 'static,
        {
            let fp_vec = vec![fp_0; SIZE];

            let measurements = [fp_vec.clone(), fp_vec];
            assert_eq!(
                run_vdaf(CTX_STR, &prio3, &(), measurements).unwrap(),
                vec![0.0; SIZE]
            );
        }
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn test_prio3_bounded_fpvec_sum() {
        const FP16_4_INV: I1F15 = I1F15::lit("0.25");
        const FP16_8_INV: I1F15 = I1F15::lit("0.125");
        const FP16_16_INV: I1F15 = I1F15::lit("0.0625");
        const FP32_4_INV: I1F31 = I1F31::lit("0.25");
        const FP32_8_INV: I1F31 = I1F31::lit("0.125");
        const FP32_16_INV: I1F31 = I1F31::lit("0.0625");
        const FP64_4_INV: I1F63 = I1F63::lit("0.25");
        const FP64_8_INV: I1F63 = I1F63::lit("0.125");
        const FP64_16_INV: I1F63 = I1F63::lit("0.0625");

        type P<Fx> = Prio3FixedPointBoundedL2VecSum<Fx>;
        let ctor_16 = P::<FixedI16<U15>>::new_fixedpoint_boundedl2_vec_sum;
        let ctor_32 = P::<FixedI32<U31>>::new_fixedpoint_boundedl2_vec_sum;
        let ctor_64 = P::<FixedI64<U63>>::new_fixedpoint_boundedl2_vec_sum;

        #[cfg(feature = "multithreaded")]
        type PM<Fx> = Prio3FixedPointBoundedL2VecSumMultithreaded<Fx>;
        #[cfg(feature = "multithreaded")]
        let ctor_mt_16 = PM::<FixedI16<U15>>::new_fixedpoint_boundedl2_vec_sum_multithreaded;
        #[cfg(feature = "multithreaded")]
        let ctor_mt_32 = PM::<FixedI32<U31>>::new_fixedpoint_boundedl2_vec_sum_multithreaded;
        #[cfg(feature = "multithreaded")]
        let ctor_mt_64 = PM::<FixedI64<U63>>::new_fixedpoint_boundedl2_vec_sum_multithreaded;

        {
            // 16 bit fixedpoint

            // two aggregators, three entries per vector.
            {
                let prio3_16 = ctor_16(2, 3).unwrap();
                test_fixed(FP16_4_INV, FP16_8_INV, FP16_16_INV, prio3_16);
            }

            #[cfg(feature = "multithreaded")]
            {
                let prio3_16_mt = ctor_mt_16(2, 3).unwrap();
                test_fixed(FP16_4_INV, FP16_8_INV, FP16_16_INV, prio3_16_mt);
            }
        }

        {
            // 32 bit fixedpoint

            {
                let prio3_32 = ctor_32(2, 3).unwrap();
                test_fixed(FP32_4_INV, FP32_8_INV, FP32_16_INV, prio3_32);
            }

            #[cfg(feature = "multithreaded")]
            {
                let prio3_32_mt = ctor_mt_32(2, 3).unwrap();
                test_fixed(FP32_4_INV, FP32_8_INV, FP32_16_INV, prio3_32_mt);
            }
        }

        {
            // 64 bit fixedpoint

            {
                let prio3_64 = ctor_64(2, 3).unwrap();
                test_fixed(FP64_4_INV, FP64_8_INV, FP64_16_INV, prio3_64);
            }

            #[cfg(feature = "multithreaded")]
            {
                let prio3_64_mt = ctor_mt_64(2, 3).unwrap();
                test_fixed(FP64_4_INV, FP64_8_INV, FP64_16_INV, prio3_64_mt);
            }
        }

        fn test_fixed<Fx, PE, M>(
            fp_4_inv: Fx,
            fp_8_inv: Fx,
            fp_16_inv: Fx,
            prio3: Prio3<FixedPointBoundedL2VecSum<Fx, PE, M>, XofTurboShake128, 32>,
        ) where
            Fx: Fixed + CompatibleFloat + std::ops::Neg<Output = Fx>,
            PE: Eq + ParallelSumGadget<Field128, PolyEval<Field128>> + Clone + 'static,
            M: Eq + ParallelSumGadget<Field128, Mul<Field128>> + Clone + 'static,
        {
            let fp_vec1 = vec![fp_4_inv, fp_8_inv, fp_16_inv];
            let fp_vec2 = vec![fp_4_inv, fp_8_inv, fp_16_inv];

            let fp_vec3 = vec![-fp_4_inv, -fp_8_inv, -fp_16_inv];
            let fp_vec4 = vec![-fp_4_inv, -fp_8_inv, -fp_16_inv];

            let fp_vec5 = vec![fp_4_inv, -fp_8_inv, -fp_16_inv];
            let fp_vec6 = vec![fp_4_inv, fp_8_inv, fp_16_inv];

            // positive entries
            let fp_list = [fp_vec1, fp_vec2];
            assert_eq!(
                run_vdaf(CTX_STR, &prio3, &(), fp_list).unwrap(),
                vec!(0.5, 0.25, 0.125),
            );

            // negative entries
            let fp_list2 = [fp_vec3, fp_vec4];
            assert_eq!(
                run_vdaf(CTX_STR, &prio3, &(), fp_list2).unwrap(),
                vec!(-0.5, -0.25, -0.125),
            );

            // both
            let fp_list3 = [fp_vec5, fp_vec6];
            assert_eq!(
                run_vdaf(CTX_STR, &prio3, &(), fp_list3).unwrap(),
                vec!(0.5, 0.0, 0.0),
            );

            let mut verify_key = [0; 32];
            let mut nonce = [0; 16];
            rng().fill(&mut verify_key);
            rng().fill(&mut nonce);

            let (public_share, mut input_shares) = prio3
                .shard(CTX_STR, &vec![fp_4_inv, fp_8_inv, fp_16_inv], &nonce)
                .unwrap();
            input_shares[0].joint_rand_blind_mut().unwrap().0[0] ^= 255;
            let result = run_vdaf_prepare(
                &prio3,
                &verify_key,
                CTX_STR,
                &(),
                &nonce,
                public_share,
                input_shares,
            );
            assert_matches!(result, Err(VdafError::Uncategorized(_)));

            let (public_share, mut input_shares) = prio3
                .shard(CTX_STR, &vec![fp_4_inv, fp_8_inv, fp_16_inv], &nonce)
                .unwrap();
            assert_matches!(
                &mut input_shares[0],
                Prio3InputShare::Leader { ref mut measurement_share, ..} => {
                    measurement_share[0] += Field128::one();
                }
            );
            let result = run_vdaf_prepare(
                &prio3,
                &verify_key,
                CTX_STR,
                &(),
                &nonce,
                public_share,
                input_shares,
            );
            assert_matches!(result, Err(VdafError::Uncategorized(_)));

            let (public_share, mut input_shares) = prio3
                .shard(CTX_STR, &vec![fp_4_inv, fp_8_inv, fp_16_inv], &nonce)
                .unwrap();
            assert_matches!(
                &mut input_shares[0],
                Prio3InputShare::Leader { ref mut proofs_share, ..} => {
                    proofs_share[0] += Field128::one();
                }
            );
            let result = run_vdaf_prepare(
                &prio3,
                &verify_key,
                CTX_STR,
                &(),
                &nonce,
                public_share,
                input_shares,
            );
            assert_matches!(result, Err(VdafError::Uncategorized(_)));

            test_serialization(&prio3, &vec![fp_4_inv, fp_8_inv, fp_16_inv], &nonce).unwrap();
        }
    }

    #[test]
    fn test_prio3_histogram() {
        let prio3 = Prio3::new_histogram(2, 4, 2).unwrap();

        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [0, 1, 2, 3]).unwrap(),
            vec![1, 1, 1, 1]
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [0]).unwrap(),
            vec![1, 0, 0, 0]
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [1]).unwrap(),
            vec![0, 1, 0, 0]
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [2]).unwrap(),
            vec![0, 0, 1, 0]
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [3]).unwrap(),
            vec![0, 0, 0, 1]
        );
        test_serialization(&prio3, &3, &[0; 16]).unwrap();
    }

    #[test]
    #[cfg(feature = "multithreaded")]
    fn test_prio3_histogram_multithreaded() {
        let prio3 = Prio3::new_histogram_multithreaded(2, 4, 2).unwrap();

        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [0, 1, 2, 3]).unwrap(),
            vec![1, 1, 1, 1]
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [0]).unwrap(),
            vec![1, 0, 0, 0]
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [1]).unwrap(),
            vec![0, 1, 0, 0]
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [2]).unwrap(),
            vec![0, 0, 1, 0]
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [3]).unwrap(),
            vec![0, 0, 0, 1]
        );
        test_serialization(&prio3, &3, &[0; 16]).unwrap();
    }

    #[test]
    fn test_prio3_average() {
        let max_measurement = 43_208;
        let prio3 = Prio3::new_average(2, max_measurement).unwrap();

        assert_eq!(run_vdaf(CTX_STR, &prio3, &(), [17, 8]).unwrap(), 12.5f64);
        assert_eq!(run_vdaf(CTX_STR, &prio3, &(), [1, 1, 1, 1]).unwrap(), 1f64);
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [0, 0, 0, 1]).unwrap(),
            0.25f64
        );
        assert_eq!(
            run_vdaf(CTX_STR, &prio3, &(), [1, 11, 111, 1111, 3, 8]).unwrap(),
            207.5f64
        );
    }

    #[test]
    fn test_prio3_input_share() {
        let max_measurement = 1;
        let prio3 = Prio3::new_sum(5, max_measurement).unwrap();
        let (_public_share, input_shares) = prio3.shard(CTX_STR, &1, &[0; 16]).unwrap();

        // Check that seed shares are distinct.
        for (i, x) in input_shares.iter().enumerate() {
            for (j, y) in input_shares.iter().enumerate() {
                if i != j {
                    if let (Share::Helper(left), Share::Helper(right)) =
                        (&x.measurement_share(), &y.measurement_share())
                    {
                        assert_ne!(left, right);
                    }

                    if let (Share::Helper(left), Share::Helper(right)) =
                        (&x.proofs_share(), &y.proofs_share())
                    {
                        assert_ne!(left, right);
                    }
                }
            }
        }
    }

    fn test_serialization<T, P, const SEED_SIZE: usize>(
        prio3: &Prio3<T, P, SEED_SIZE>,
        measurement: &T::Measurement,
        nonce: &[u8; 16],
    ) -> Result<(), VdafError>
    where
        T: Type,
        P: Xof<SEED_SIZE>,
    {
        let mut verify_key = [0; SEED_SIZE];
        rng().fill(&mut verify_key[..]);
        let (public_share, input_shares) = prio3.shard(CTX_STR, measurement, nonce)?;

        let encoded_public_share = public_share.get_encoded().unwrap();
        let decoded_public_share =
            Prio3PublicShare::get_decoded_with_param(prio3, &encoded_public_share)
                .expect("failed to decode public share");
        assert_eq!(decoded_public_share, public_share);
        assert_eq!(
            public_share.encoded_len().unwrap(),
            encoded_public_share.len()
        );

        for (agg_id, input_share) in input_shares.iter().enumerate() {
            let encoded_input_share = input_share.get_encoded().unwrap();
            let decoded_input_share =
                Prio3InputShare::get_decoded_with_param(&(prio3, agg_id), &encoded_input_share)
                    .expect("failed to decode input share");
            assert_eq!(&decoded_input_share, input_share);
            assert_eq!(
                input_share.encoded_len().unwrap(),
                encoded_input_share.len()
            );
        }

        let mut prepare_shares = Vec::new();
        let mut last_prepare_state = None;
        for (agg_id, input_share) in input_shares.iter().enumerate() {
            let (prepare_state, prepare_share) = prio3.prepare_init(
                &verify_key,
                CTX_STR,
                agg_id,
                &(),
                nonce,
                &public_share,
                input_share,
            )?;

            let encoded_prepare_state = prepare_state.get_encoded().unwrap();
            let decoded_prepare_state =
                Prio3PrepareState::get_decoded_with_param(&(prio3, agg_id), &encoded_prepare_state)
                    .expect("failed to decode prepare state");
            assert_eq!(decoded_prepare_state, prepare_state);
            assert_eq!(
                prepare_state.encoded_len().unwrap(),
                encoded_prepare_state.len()
            );

            let encoded_prepare_share = prepare_share.get_encoded().unwrap();
            let decoded_prepare_share =
                Prio3PrepareShare::get_decoded_with_param(&prepare_state, &encoded_prepare_share)
                    .expect("failed to decode prepare share");
            assert_eq!(decoded_prepare_share, prepare_share);
            assert_eq!(
                prepare_share.encoded_len().unwrap(),
                encoded_prepare_share.len()
            );

            prepare_shares.push(prepare_share);
            last_prepare_state = Some(prepare_state);
        }

        let prepare_message = prio3
            .prepare_shares_to_prepare_message(CTX_STR, &(), prepare_shares)
            .unwrap();

        let encoded_prepare_message = prepare_message.get_encoded().unwrap();
        let decoded_prepare_message = Prio3PrepareMessage::get_decoded_with_param(
            &last_prepare_state.unwrap(),
            &encoded_prepare_message,
        )
        .expect("failed to decode prepare message");
        assert_eq!(decoded_prepare_message, prepare_message);
        assert_eq!(
            prepare_message.encoded_len().unwrap(),
            encoded_prepare_message.len()
        );

        Ok(())
    }

    #[test]
    fn roundtrip_output_share() {
        let vdaf = Prio3::new_count(2).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Count, OutputShare<Field64>>(&vdaf, &(), 1);

        let max_measurement = 13;
        let vdaf = Prio3::new_sum(2, max_measurement).unwrap();
        fieldvec_roundtrip_test::<Field128, Prio3Sum, OutputShare<Field128>>(&vdaf, &(), 1);

        let vdaf = Prio3::new_histogram(2, 12, 3).unwrap();
        fieldvec_roundtrip_test::<Field128, Prio3Histogram, OutputShare<Field128>>(&vdaf, &(), 12);
    }

    #[test]
    fn roundtrip_aggregate_share() {
        let vdaf = Prio3::new_count(2).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Count, AggregateShare<Field64>>(&vdaf, &(), 1);

        let max_measurement = 13;
        let vdaf = Prio3::new_sum(2, max_measurement).unwrap();
        fieldvec_roundtrip_test::<Field128, Prio3Sum, AggregateShare<Field128>>(&vdaf, &(), 1);

        let vdaf = Prio3::new_histogram(2, 12, 3).unwrap();
        fieldvec_roundtrip_test::<Field128, Prio3Histogram, AggregateShare<Field128>>(
            &vdaf,
            &(),
            12,
        );
    }

    #[test]
    fn public_share_equality_test() {
        equality_comparison_test(&[
            Prio3PublicShare {
                joint_rand_parts: Some(Vec::from([Seed([0])])),
            },
            Prio3PublicShare {
                joint_rand_parts: Some(Vec::from([Seed([1])])),
            },
            Prio3PublicShare {
                joint_rand_parts: None,
            },
        ])
    }

    #[test]
    fn input_share_equality_test() {
        equality_comparison_test(&[
            // Default.
            Prio3InputShare::Leader {
                measurement_share: Vec::from([0]),
                proofs_share: Vec::from([1]),
                joint_rand_blind: Some(Seed([2])),
            },
            // Modified measurement share.
            Prio3InputShare::Leader {
                measurement_share: Vec::from([100]),
                proofs_share: Vec::from([1]),
                joint_rand_blind: Some(Seed([2])),
            },
            // Modified proof share.
            Prio3InputShare::Leader {
                measurement_share: Vec::from([0]),
                proofs_share: Vec::from([101]),
                joint_rand_blind: Some(Seed([2])),
            },
            // Modified joint_rand_blind.
            Prio3InputShare::Leader {
                measurement_share: Vec::from([0]),
                proofs_share: Vec::from([1]),
                joint_rand_blind: Some(Seed([102])),
            },
            // Missing joint_rand_blind.
            Prio3InputShare::Leader {
                measurement_share: Vec::from([0]),
                proofs_share: Vec::from([1]),
                joint_rand_blind: None,
            },
        ])
    }

    #[test]
    fn prepare_share_equality_test() {
        equality_comparison_test(&[
            // Default.
            Prio3PrepareShare {
                verifiers: Vec::from([0]),
                joint_rand_part: Some(Seed([1])),
            },
            // Modified verifier.
            Prio3PrepareShare {
                verifiers: Vec::from([100]),
                joint_rand_part: Some(Seed([1])),
            },
            // Modified joint_rand_part.
            Prio3PrepareShare {
                verifiers: Vec::from([0]),
                joint_rand_part: Some(Seed([101])),
            },
            // Missing joint_rand_part.
            Prio3PrepareShare {
                verifiers: Vec::from([0]),
                joint_rand_part: None,
            },
        ])
    }

    #[test]
    fn prepare_message_equality_test() {
        equality_comparison_test(&[
            // Default.
            Prio3PrepareMessage {
                joint_rand_seed: Some(Seed([0])),
            },
            // Modified joint_rand_seed.
            Prio3PrepareMessage {
                joint_rand_seed: Some(Seed([100])),
            },
            // Missing joint_rand_seed.
            Prio3PrepareMessage {
                joint_rand_seed: None,
            },
        ])
    }

    #[test]
    fn prepare_state_equality_test() {
        equality_comparison_test(&[
            // Default.
            Prio3PrepareState {
                share: Share::Leader(Vec::from([0])),
                joint_rand_seed: Some(Seed([1])),
                agg_id: 2,
                verifiers_len: 3,
            },
            // Modified measurement share.
            Prio3PrepareState {
                share: Share::Leader(Vec::from([100])),
                joint_rand_seed: Some(Seed([1])),
                agg_id: 2,
                verifiers_len: 3,
            },
            // Modified joint_rand_seed.
            Prio3PrepareState {
                share: Share::Leader(Vec::from([0])),
                joint_rand_seed: Some(Seed([101])),
                agg_id: 2,
                verifiers_len: 3,
            },
            // Missing joint_rand_seed.
            Prio3PrepareState {
                share: Share::Leader(Vec::from([0])),
                joint_rand_seed: None,
                agg_id: 2,
                verifiers_len: 3,
            },
            // Modified agg_id.
            Prio3PrepareState {
                share: Share::Leader(Vec::from([0])),
                joint_rand_seed: Some(Seed([1])),
                agg_id: 102,
                verifiers_len: 3,
            },
            // Modified verifier_len.
            Prio3PrepareState {
                share: Share::Leader(Vec::from([0])),
                joint_rand_seed: Some(Seed([1])),
                agg_id: 2,
                verifiers_len: 103,
            },
        ])
    }

    #[test]
    fn test_optimal_chunk_length() {
        // nonsense argument, but make sure it doesn't panic.
        optimal_chunk_length(0);

        // edge cases on either side of power-of-two jumps
        assert_eq!(optimal_chunk_length(1), 1);
        assert_eq!(optimal_chunk_length(2), 2);
        assert_eq!(optimal_chunk_length(3), 1);
        assert_eq!(optimal_chunk_length(18), 6);
        assert_eq!(optimal_chunk_length(19), 3);

        // additional arbitrary test cases
        assert_eq!(optimal_chunk_length(40), 6);
        assert_eq!(optimal_chunk_length(10_000), 79);
        assert_eq!(optimal_chunk_length(100_000), 393);

        // confirm that the chunk lengths are truly optimal
        for measurement_length in [2, 3, 4, 5, 18, 19, 40] {
            let optimal_chunk_length = optimal_chunk_length(measurement_length);
            let optimal_proof_length = Histogram::<Field128, ParallelSum<_, _>>::new(
                measurement_length,
                optimal_chunk_length,
            )
            .unwrap()
            .proof_len();
            for chunk_length in 1..=measurement_length {
                let proof_length =
                    Histogram::<Field128, ParallelSum<_, _>>::new(measurement_length, chunk_length)
                        .unwrap()
                        .proof_len();
                assert!(proof_length >= optimal_proof_length);
            }
        }
    }
}
