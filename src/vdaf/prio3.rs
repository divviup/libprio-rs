// SPDX-License-Identifier: MPL-2.0

//! Implementation of the Prio3 VDAF [[draft-irtf-cfrg-vdaf-03]].
//!
//! **WARNING:** Neither this code nor the cryptographic construction it implements has undergone
//! significant security analysis. Use at your own risk.
//!
//! Prio3 is based on the Prio system desigend by Dan Boneh and Henry Corrigan-Gibbs and presented
//! at NSDI 2017 [[CGB17]]. However, it incorporates a few techniques from Boneh et al., CRYPTO
//! 2019 [[BBCG+19]], that lead to substantial improvements in terms of run time and communication
//! cost.
//!
//! Prio3 is a transformation of a Fully Linear Proof (FLP) system [[draft-irtf-cfrg-vdaf-03]] into
//! a VDAF. The base type, [`Prio3`], supports a wide variety of aggregation functions, some of
//! which are instantiated here:
//!
//! - [`Prio3Aes128Count`] for aggregating a counter (*)
//! - [`Prio3Aes128Sum`] for copmputing the sum of integers (*)
//! - [`Prio3Aes128SumVec`] for aggregating a vector of integers
//! - [`Prio3Aes128Histogram`] for estimating a distribution via a histogram (*)
//!
//! Additional types can be constructed from [`Prio3`] as needed.
//!
//! (*) denotes that the type is specified in [[draft-irtf-cfrg-vdaf-03]].
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [CGB17]: https://crypto.stanford.edu/prio/
//! [draft-irtf-cfrg-vdaf-03]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/

#[cfg(feature = "crypto-dependencies")]
use super::prg::PrgAes128;
use crate::codec::{CodecError, Decode, Encode, ParameterizedDecode};
use crate::field::{decode_fieldvec, FftFriendlyFieldElement, FieldElement};
#[cfg(feature = "crypto-dependencies")]
use crate::field::{Field128, Field64};
#[cfg(feature = "multithreaded")]
use crate::flp::gadgets::ParallelSumMultithreaded;
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use crate::flp::gadgets::PolyEval;
#[cfg(feature = "crypto-dependencies")]
use crate::flp::gadgets::{BlindPolyEval, ParallelSum};
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use crate::flp::types::fixedpoint_l2::{
    compatible_float::CompatibleFloat, FixedPointBoundedL2VecSum,
};
#[cfg(feature = "crypto-dependencies")]
use crate::flp::types::{Average, Count, Histogram, Sum, SumVec};
use crate::flp::Type;
use crate::prng::Prng;
use crate::vdaf::prg::{Prg, RandSource, Seed};
use crate::vdaf::{
    Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare, PrepareTransition,
    Share, ShareDecodingParameter, Vdaf, VdafError,
};
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use fixed::traits::Fixed;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::io::Cursor;
use std::iter::{self, IntoIterator};
use std::marker::PhantomData;

const DST_MEASUREMENT_SHARE: u16 = 1;
const DST_PROOF_SHARE: u16 = 2;
const DST_JOINT_RANDOMNESS: u16 = 3;
const DST_PROVE_RANDOMNESS: u16 = 4;
const DST_QUERY_RANDOMNESS: u16 = 5;
const DST_JOINT_RAND_SEED: u16 = 6;
const DST_JOINT_RAND_PART: u16 = 7;

/// The count type. Each measurement is an integer in `[0,2)` and the aggregate result is the sum.
#[cfg(feature = "crypto-dependencies")]
pub type Prio3Aes128Count = Prio3<Count<Field64>, PrgAes128, 16>;

#[cfg(feature = "crypto-dependencies")]
impl Prio3Aes128Count {
    /// Construct an instance of Prio3Aes128Count with the given number of aggregators.
    pub fn new_aes128_count(num_aggregators: u8) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, Count::new())
    }
}

/// The count-vector type. Each measurement is a vector of integers in `[0,2)` and the aggregate is
/// the element-wise sum.
#[cfg(feature = "crypto-dependencies")]
pub type Prio3Aes128SumVec =
    Prio3<SumVec<Field128, ParallelSum<Field128, BlindPolyEval<Field128>>>, PrgAes128, 16>;

#[cfg(feature = "crypto-dependencies")]
impl Prio3Aes128SumVec {
    /// Construct an instance of Prio3Aes1238SumVec with the given number of aggregators. `bits`
    /// defines the bit width of each summand of the measurement; `len` defines the length of the
    /// measurement vector.
    pub fn new_aes128_sum_vec(
        num_aggregators: u8,
        bits: usize,
        len: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, SumVec::new(bits, len)?)
    }
}

/// Like [`Prio3Aes128SumVec`] except this type uses multithreading to improve sharding and
/// preparation time. Note that the improvement is only noticeable for very large input lengths,
/// e.g., 201 and up. (Your system's mileage may vary.)
#[cfg(feature = "multithreaded")]
#[cfg(feature = "crypto-dependencies")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3Aes128SumVecMultithreaded = Prio3<
    SumVec<Field128, ParallelSumMultithreaded<Field128, BlindPolyEval<Field128>>>,
    PrgAes128,
    16,
>;

#[cfg(feature = "multithreaded")]
#[cfg(feature = "crypto-dependencies")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
impl Prio3Aes128SumVecMultithreaded {
    /// Construct an instance of Prio3Aes1238SumVecMultithreaded with the given number of
    /// aggregators. `bits` defines the bit width of each summand of the measurement; `len` defines
    /// the length of the measurement vector.
    pub fn new_aes128_sum_vec_multithreaded(
        num_aggregators: u8,
        bits: usize,
        len: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, SumVec::new(bits, len)?)
    }
}

/// The sum type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and the
/// aggregate is the sum.
#[cfg(feature = "crypto-dependencies")]
pub type Prio3Aes128Sum = Prio3<Sum<Field128>, PrgAes128, 16>;

#[cfg(feature = "crypto-dependencies")]
impl Prio3Aes128Sum {
    /// Construct an instance of Prio3Aes128Sum with the given number of aggregators and required
    /// bit length. The bit length must not exceed 64.
    pub fn new_aes128_sum(num_aggregators: u8, bits: u32) -> Result<Self, VdafError> {
        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({bits}) exceeds limit for aggregate type (64)"
            )));
        }

        Prio3::new(num_aggregators, Sum::new(bits as usize)?)
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
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub type Prio3Aes128FixedPointBoundedL2VecSum<Fx> = Prio3<
    FixedPointBoundedL2VecSum<
        Fx,
        Field128,
        ParallelSum<Field128, PolyEval<Field128>>,
        ParallelSum<Field128, BlindPolyEval<Field128>>,
    >,
    PrgAes128,
    16,
>;

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
impl<Fx: Fixed + CompatibleFloat<Field128>> Prio3Aes128FixedPointBoundedL2VecSum<Fx> {
    /// Construct an instance of this VDAF with the given number of aggregators and number of
    /// vector entries.
    pub fn new_aes128_fixedpoint_boundedl2_vec_sum(
        num_aggregators: u8,
        entries: usize,
    ) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;
        Prio3::new(num_aggregators, FixedPointBoundedL2VecSum::new(entries)?)
    }
}

/// The fixed point vector sum type. Each measurement is a vector of fixed point numbers
/// and the aggregate is the sum represented as 64-bit floats. The verification function
/// ensures the L2 norm of the input vector is < 1.
#[cfg(all(
    feature = "crypto-dependencies",
    feature = "experimental",
    feature = "multithreaded"
))]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3Aes128FixedPointBoundedL2VecSumMultithreaded<Fx> = Prio3<
    FixedPointBoundedL2VecSum<
        Fx,
        Field128,
        ParallelSumMultithreaded<Field128, PolyEval<Field128>>,
        ParallelSumMultithreaded<Field128, BlindPolyEval<Field128>>,
    >,
    PrgAes128,
    16,
>;

#[cfg(all(
    feature = "crypto-dependencies",
    feature = "experimental",
    feature = "multithreaded"
))]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
impl<Fx: Fixed + CompatibleFloat<Field128>> Prio3Aes128FixedPointBoundedL2VecSumMultithreaded<Fx> {
    /// Construct an instance of this VDAF with the given number of aggregators and number of
    /// vector entries.
    pub fn new_aes128_fixedpoint_boundedl2_vec_sum_multithreaded(
        num_aggregators: u8,
        entries: usize,
    ) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;
        Prio3::new(num_aggregators, FixedPointBoundedL2VecSum::new(entries)?)
    }
}

/// The histogram type. Each measurement is an unsigned integer and the result is a histogram
/// representation of the distribution. The bucket boundaries are fixed in advance.
#[cfg(feature = "crypto-dependencies")]
pub type Prio3Aes128Histogram = Prio3<Histogram<Field128>, PrgAes128, 16>;

#[cfg(feature = "crypto-dependencies")]
impl Prio3Aes128Histogram {
    /// Constructs an instance of Prio3Aes128Histogram with the given number of aggregators and
    /// desired histogram bucket boundaries.
    pub fn new_aes128_histogram(num_aggregators: u8, buckets: &[u64]) -> Result<Self, VdafError> {
        let buckets = buckets.iter().map(|bucket| *bucket as u128).collect();

        Prio3::new(num_aggregators, Histogram::new(buckets)?)
    }
}

/// The average type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and
/// the aggregate is the arithmetic average.
#[cfg(feature = "crypto-dependencies")]
pub type Prio3Aes128Average = Prio3<Average<Field128>, PrgAes128, 16>;

#[cfg(feature = "crypto-dependencies")]
impl Prio3Aes128Average {
    /// Construct an instance of Prio3Aes128Average with the given number of aggregators and
    /// required bit length. The bit length must not exceed 64.
    pub fn new_aes128_average(num_aggregators: u8, bits: u32) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({bits}) exceeds limit for aggregate type (64)"
            )));
        }

        Ok(Prio3 {
            num_aggregators,
            typ: Average::new(bits as usize)?,
            phantom: PhantomData,
        })
    }
}

/// The base type for Prio3.
///
/// An instance of Prio3 is determined by:
///
/// - a [`Type`](crate::flp::Type) that defines the set of valid input measurements; and
/// - a [`Prg`](crate::vdaf::prg::Prg) for deriving vectors of field elements from seeds.
///
/// New instances can be defined by aliasing the base type. For example, [`Prio3Aes128Count`] is an
/// alias for `Prio3<Count<Field64>, PrgAes128, 16>`.
///
/// ```
/// use prio::vdaf::{
///     Aggregator, Client, Collector, PrepareTransition,
///     prio3::Prio3,
/// };
/// use rand::prelude::*;
///
/// let num_shares = 2;
/// let vdaf = Prio3::new_aes128_count(num_shares).unwrap();
///
/// let mut out_shares = vec![vec![]; num_shares.into()];
/// let mut rng = thread_rng();
/// let verify_key = rng.gen();
/// let measurements = [0, 1, 1, 1, 0];
/// for measurement in measurements {
///     // Shard
///     let nonce = rng.gen::<[u8; 16]>();
///     let (public_share, input_shares) = vdaf.shard(&measurement, &nonce).unwrap();
///
///     // Prepare
///     let mut prep_states = vec![];
///     let mut prep_shares = vec![];
///     for (agg_id, input_share) in input_shares.iter().enumerate() {
///         let (state, share) = vdaf.prepare_init(
///             &verify_key,
///             agg_id,
///             &(),
///             &nonce,
///             &public_share,
///             input_share
///         ).unwrap();
///         prep_states.push(state);
///         prep_shares.push(share);
///     }
///     let prep_msg = vdaf.prepare_preprocess(prep_shares).unwrap();
///
///     for (agg_id, state) in prep_states.into_iter().enumerate() {
///         let out_share = match vdaf.prepare_step(state, prep_msg.clone()).unwrap() {
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
///
/// [draft-irtf-cfrg-vdaf-03]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/
#[derive(Clone, Debug)]
pub struct Prio3<T, P, const L: usize>
where
    T: Type,
    P: Prg<L>,
{
    num_aggregators: u8,
    typ: T,
    phantom: PhantomData<P>,
}

impl<T, P, const L: usize> Prio3<T, P, L>
where
    T: Type,
    P: Prg<L>,
{
    /// Construct an instance of this Prio3 VDAF with the given number of aggregators and the
    /// underlying type.
    pub fn new(num_aggregators: u8, typ: T) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;
        Ok(Self {
            num_aggregators,
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

    fn derive_joint_rand_seed<'a>(parts: impl Iterator<Item = &'a Seed<L>>) -> Seed<L> {
        let mut prg = P::init(&[0; L], &Self::custom(DST_JOINT_RAND_SEED));
        for part in parts {
            prg.update(part.as_ref());
        }
        prg.into_seed()
    }

    #[allow(clippy::type_complexity)]
    fn shard_with_rand_source<const N: usize>(
        &self,
        measurement: &T::Measurement,
        nonce: &[u8; N],
        rand_source: RandSource,
    ) -> Result<(Prio3PublicShare<L>, Vec<Prio3InputShare<T::Field, L>>), VdafError> {
        let num_aggregators = self.num_aggregators;
        let encoded_measurement = self.typ.encode_measurement(measurement)?;

        // Generate the measurement shares and compute the joint randomness.
        let mut helper_shares = Vec::with_capacity(num_aggregators as usize - 1);
        let mut helper_joint_rand_parts = if self.typ.joint_rand_len() > 0 {
            Some(Vec::with_capacity(num_aggregators as usize - 1))
        } else {
            None
        };
        let mut leader_measurement_share = encoded_measurement.clone();
        for agg_id in 1..num_aggregators {
            let helper = HelperShare::from_rand_source(rand_source)?;
            let mut joint_rand_part_prg = P::init(
                helper.joint_rand_blind.as_ref(),
                &Self::custom(DST_JOINT_RAND_PART),
            );
            joint_rand_part_prg.update(&[agg_id]); // Aggregator ID
            joint_rand_part_prg.update(nonce);

            let measurement_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &helper.measurement_share,
                &Self::custom(DST_MEASUREMENT_SHARE),
                &[agg_id],
            ));
            for (x, y) in leader_measurement_share
                .iter_mut()
                .zip(measurement_share_prng)
                .take(self.typ.input_len())
            {
                *x -= y;
                joint_rand_part_prg.update(&y.into());
            }

            if let Some(helper_joint_rand_parts) = helper_joint_rand_parts.as_mut() {
                helper_joint_rand_parts.push(joint_rand_part_prg.into_seed());
            }
            helper_shares.push(helper);
        }

        let leader_blind = Seed::from_rand_source(rand_source)?;

        let mut joint_rand_part_prg =
            P::init(leader_blind.as_ref(), &Self::custom(DST_JOINT_RAND_PART));
        joint_rand_part_prg.update(&[0]); // Aggregator ID
        joint_rand_part_prg.update(nonce);
        for x in leader_measurement_share.iter() {
            joint_rand_part_prg.update(&(*x).into());
        }

        let leader_joint_rand_seed_part = joint_rand_part_prg.into_seed();

        // Compute the joint randomness seed.
        let joint_rand_seed = helper_joint_rand_parts.as_ref().map(|parts| {
            Self::derive_joint_rand_seed(
                std::iter::once(&leader_joint_rand_seed_part).chain(parts.iter()),
            )
        });

        let public_share = Prio3PublicShare {
            joint_rand_parts: helper_joint_rand_parts
                .as_ref()
                .map(|helper_joint_rand_parts| {
                    let mut vec = Vec::with_capacity(self.num_aggregators());
                    vec.push(leader_joint_rand_seed_part.clone());
                    vec.extend(helper_joint_rand_parts.iter().cloned());
                    vec
                }),
        };

        // Run the proof-generation algorithm.
        let joint_rand: Vec<T::Field> = joint_rand_seed
            .map(|joint_rand_seed| {
                let prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                    &joint_rand_seed,
                    &Self::custom(DST_JOINT_RANDOMNESS),
                    &[],
                ));
                prng.take(self.typ.joint_rand_len()).collect()
            })
            .unwrap_or_default();
        let prove_rand_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
            &Seed::from_rand_source(rand_source)?,
            &Self::custom(DST_PROVE_RANDOMNESS),
            &[],
        ));
        let prove_rand: Vec<T::Field> = prove_rand_prng.take(self.typ.prove_rand_len()).collect();
        let mut leader_proof_share =
            self.typ
                .prove(&encoded_measurement, &prove_rand, &joint_rand)?;

        // Generate the proof shares and distribute the joint randomness seed hints.
        for (j, helper) in helper_shares.iter_mut().enumerate() {
            let proof_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &helper.proof_share,
                &Self::custom(DST_PROOF_SHARE),
                &[j as u8 + 1],
            ));
            for (x, y) in leader_proof_share
                .iter_mut()
                .zip(proof_share_prng)
                .take(self.typ.proof_len())
            {
                *x -= y;
            }

            if let Some(helper_joint_rand_parts) = helper_joint_rand_parts.as_ref() {
                let mut hint = Vec::with_capacity(num_aggregators as usize - 1);
                hint.push(leader_joint_rand_seed_part.clone());
                hint.extend(helper_joint_rand_parts[..j].iter().cloned());
                hint.extend(helper_joint_rand_parts[j + 1..].iter().cloned());
            }
        }

        let leader_joint_rand_blind = if self.typ.joint_rand_len() > 0 {
            Some(leader_blind)
        } else {
            None
        };

        // Prep the output messages.
        let mut out = Vec::with_capacity(num_aggregators as usize);
        out.push(Prio3InputShare {
            measurement_share: Share::Leader(leader_measurement_share),
            proof_share: Share::Leader(leader_proof_share),
            joint_rand_blind: leader_joint_rand_blind,
        });

        for helper in helper_shares.into_iter() {
            let helper_joint_rand_blind = if self.typ.joint_rand_len() > 0 {
                Some(helper.joint_rand_blind)
            } else {
                None
            };

            out.push(Prio3InputShare {
                measurement_share: Share::Helper(helper.measurement_share),
                proof_share: Share::Helper(helper.proof_share),
                joint_rand_blind: helper_joint_rand_blind,
            });
        }

        Ok((public_share, out))
    }

    /// Shard measurement with constant randomness of repeated bytes.
    /// This method is not secure. It is used for running test vectors for Prio3.
    #[cfg(feature = "test-util")]
    #[allow(clippy::type_complexity)]
    pub fn test_vec_shard<const N: usize>(
        &self,
        measurement: &T::Measurement,
        nonce: &[u8; N],
    ) -> Result<(Prio3PublicShare<L>, Vec<Prio3InputShare<T::Field, L>>), VdafError> {
        self.shard_with_rand_source(measurement, nonce, |buf| {
            buf.fill(1);
            Ok(())
        })
    }

    fn role_try_from(&self, agg_id: usize) -> Result<u8, VdafError> {
        if agg_id >= self.num_aggregators as usize {
            return Err(VdafError::Uncategorized("unexpected aggregator id".into()));
        }
        Ok(u8::try_from(agg_id).unwrap())
    }
}

impl<T, P, const L: usize> Vdaf for Prio3<T, P, L>
where
    T: Type,
    P: Prg<L>,
{
    const ID: u32 = T::ID;
    type Measurement = T::Measurement;
    type AggregateResult = T::AggregateResult;
    type AggregationParam = ();
    type PublicShare = Prio3PublicShare<L>;
    type InputShare = Prio3InputShare<T::Field, L>;
    type OutputShare = OutputShare<T::Field>;
    type AggregateShare = AggregateShare<T::Field>;

    fn num_aggregators(&self) -> usize {
        self.num_aggregators as usize
    }
}

/// Message broadcast by the [`Client`] to every [`Aggregator`] during the Sharding phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prio3PublicShare<const L: usize> {
    /// Contributions to the joint randomness from every aggregator's share.
    joint_rand_parts: Option<Vec<Seed<L>>>,
}

impl<const L: usize> Encode for Prio3PublicShare<L> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        if let Some(joint_rand_parts) = self.joint_rand_parts.as_ref() {
            for part in joint_rand_parts.iter() {
                part.encode(bytes);
            }
        }
    }
}

impl<T, P, const L: usize> ParameterizedDecode<Prio3<T, P, L>> for Prio3PublicShare<L>
where
    T: Type,
    P: Prg<L>,
{
    fn decode_with_param(
        decoding_parameter: &Prio3<T, P, L>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if decoding_parameter.typ.joint_rand_len() > 0 {
            let joint_rand_parts = iter::repeat_with(|| Seed::<L>::decode(bytes))
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Prio3InputShare<F, const L: usize> {
    /// The measurement share.
    measurement_share: Share<F, L>,

    /// The proof share.
    proof_share: Share<F, L>,

    /// Blinding seed used by the Aggregator to compute the joint randomness. This field is optional
    /// because not every [`Type`] requires joint randomness.
    joint_rand_blind: Option<Seed<L>>,
}

impl<F: FftFriendlyFieldElement, const L: usize> Encode for Prio3InputShare<F, L> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        if matches!(
            (&self.measurement_share, &self.proof_share),
            (Share::Leader(_), Share::Helper(_)) | (Share::Helper(_), Share::Leader(_))
        ) {
            panic!("tried to encode input share with ambiguous encoding")
        }

        self.measurement_share.encode(bytes);
        self.proof_share.encode(bytes);
        if let Some(ref blind) = self.joint_rand_blind {
            blind.encode(bytes);
        }
    }
}

impl<'a, T, P, const L: usize> ParameterizedDecode<(&'a Prio3<T, P, L>, usize)>
    for Prio3InputShare<T::Field, L>
where
    T: Type,
    P: Prg<L>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, P, L>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let agg_id = prio3
            .role_try_from(*agg_id)
            .map_err(|e| CodecError::Other(Box::new(e)))?;
        let (input_decoder, proof_decoder) = if agg_id == 0 {
            (
                ShareDecodingParameter::Leader(prio3.typ.input_len()),
                ShareDecodingParameter::Leader(prio3.typ.proof_len()),
            )
        } else {
            (
                ShareDecodingParameter::Helper,
                ShareDecodingParameter::Helper,
            )
        };

        let measurement_share = Share::decode_with_param(&input_decoder, bytes)?;
        let proof_share = Share::decode_with_param(&proof_decoder, bytes)?;
        let joint_rand_blind = if prio3.typ.joint_rand_len() > 0 {
            let blind = Seed::decode(bytes)?;
            Some(blind)
        } else {
            None
        };

        Ok(Prio3InputShare {
            measurement_share,
            proof_share,
            joint_rand_blind,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Message broadcast by each [`Aggregator`](crate::vdaf::Aggregator) in each round of the
/// Preparation phase.
pub struct Prio3PrepareShare<F, const L: usize> {
    /// A share of the FLP verifier message. (See [`Type`](crate::flp::Type).)
    verifier: Vec<F>,

    /// A part of the joint randomness seed.
    joint_rand_part: Option<Seed<L>>,
}

impl<F: FftFriendlyFieldElement, const L: usize> Encode for Prio3PrepareShare<F, L> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        for x in &self.verifier {
            x.encode(bytes);
        }
        if let Some(ref seed) = self.joint_rand_part {
            seed.encode(bytes);
        }
    }
}

impl<F: FftFriendlyFieldElement, const L: usize> ParameterizedDecode<Prio3PrepareState<F, L>>
    for Prio3PrepareShare<F, L>
{
    fn decode_with_param(
        decoding_parameter: &Prio3PrepareState<F, L>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let mut verifier = Vec::with_capacity(decoding_parameter.verifier_len);
        for _ in 0..decoding_parameter.verifier_len {
            verifier.push(F::decode(bytes)?);
        }

        let joint_rand_part = if decoding_parameter.joint_rand_seed.is_some() {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Prio3PrepareShare {
            verifier,
            joint_rand_part,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Result of combining a round of [`Prio3PrepareShare`] messages.
pub struct Prio3PrepareMessage<const L: usize> {
    /// The joint randomness seed computed by the Aggregators.
    joint_rand_seed: Option<Seed<L>>,
}

impl<const L: usize> Encode for Prio3PrepareMessage<L> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encode(bytes);
        }
    }
}

impl<F: FftFriendlyFieldElement, const L: usize> ParameterizedDecode<Prio3PrepareState<F, L>>
    for Prio3PrepareMessage<L>
{
    fn decode_with_param(
        decoding_parameter: &Prio3PrepareState<F, L>,
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

impl<T, P, const L: usize> Client<16> for Prio3<T, P, L>
where
    T: Type,
    P: Prg<L>,
{
    #[allow(clippy::type_complexity)]
    fn shard(
        &self,
        measurement: &T::Measurement,
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Prio3InputShare<T::Field, L>>), VdafError> {
        self.shard_with_rand_source(measurement, nonce, getrandom::getrandom)
    }
}

/// State of each [`Aggregator`](crate::vdaf::Aggregator) during the Preparation phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Prio3PrepareState<F, const L: usize> {
    measurement_share: Share<F, L>,
    joint_rand_seed: Option<Seed<L>>,
    agg_id: u8,
    verifier_len: usize,
}

impl<F: FftFriendlyFieldElement, const L: usize> Encode for Prio3PrepareState<F, L> {
    /// Append the encoded form of this object to the end of `bytes`, growing the vector as needed.
    fn encode(&self, bytes: &mut Vec<u8>) {
        self.measurement_share.encode(bytes);
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encode(bytes);
        }
    }
}

impl<'a, T, P, const L: usize> ParameterizedDecode<(&'a Prio3<T, P, L>, usize)>
    for Prio3PrepareState<T::Field, L>
where
    T: Type,
    P: Prg<L>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, P, L>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let agg_id = prio3
            .role_try_from(*agg_id)
            .map_err(|e| CodecError::Other(Box::new(e)))?;

        let share_decoder = if agg_id == 0 {
            ShareDecodingParameter::Leader(prio3.typ.input_len())
        } else {
            ShareDecodingParameter::Helper
        };
        let measurement_share = Share::decode_with_param(&share_decoder, bytes)?;

        let joint_rand_seed = if prio3.typ.joint_rand_len() > 0 {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Self {
            measurement_share,
            joint_rand_seed,
            agg_id,
            verifier_len: prio3.typ.verifier_len(),
        })
    }
}

impl<T, P, const L: usize> Aggregator<L, 16> for Prio3<T, P, L>
where
    T: Type,
    P: Prg<L>,
{
    type PrepareState = Prio3PrepareState<T::Field, L>;
    type PrepareShare = Prio3PrepareShare<T::Field, L>;
    type PrepareMessage = Prio3PrepareMessage<L>;

    /// Begins the Prep process with the other aggregators. The result of this process is
    /// the aggregator's output share.
    #[allow(clippy::type_complexity)]
    fn prepare_init(
        &self,
        verify_key: &[u8; L],
        agg_id: usize,
        _agg_param: &Self::AggregationParam,
        nonce: &[u8; 16],
        public_share: &Self::PublicShare,
        msg: &Prio3InputShare<T::Field, L>,
    ) -> Result<
        (
            Prio3PrepareState<T::Field, L>,
            Prio3PrepareShare<T::Field, L>,
        ),
        VdafError,
    > {
        let agg_id = self.role_try_from(agg_id)?;
        let mut query_rand_prg = P::init(verify_key, &Self::custom(DST_QUERY_RANDOMNESS));
        query_rand_prg.update(nonce);
        let query_rand_prng = Prng::from_seed_stream(query_rand_prg.into_seed_stream());

        // Create a reference to the (expanded) measurement share.
        let expanded_measurement_share: Option<Vec<T::Field>> = match msg.measurement_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => {
                let measurement_share_prng = Prng::from_seed_stream(P::seed_stream(
                    seed,
                    &Self::custom(DST_MEASUREMENT_SHARE),
                    &[agg_id],
                ));
                Some(measurement_share_prng.take(self.typ.input_len()).collect())
            }
        };
        let measurement_share = match msg.measurement_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_measurement_share.as_ref().unwrap(),
        };

        // Create a reference to the (expanded) proof share.
        let expanded_proof_share: Option<Vec<T::Field>> = match msg.proof_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => {
                let prng = Prng::from_seed_stream(P::seed_stream(
                    seed,
                    &Self::custom(DST_PROOF_SHARE),
                    &[agg_id],
                ));
                Some(prng.take(self.typ.proof_len()).collect())
            }
        };
        let proof_share = match msg.proof_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_proof_share.as_ref().unwrap(),
        };

        // Compute the joint randomness.
        let (joint_rand_seed, joint_rand_part, joint_rand) = if self.typ.joint_rand_len() > 0 {
            let mut joint_rand_part_prg = P::init(
                msg.joint_rand_blind.as_ref().unwrap().as_ref(),
                &Self::custom(DST_JOINT_RAND_PART),
            );
            joint_rand_part_prg.update(&[agg_id]);
            joint_rand_part_prg.update(nonce);
            for x in measurement_share {
                joint_rand_part_prg.update(&(*x).into());
            }
            let own_joint_rand_part = joint_rand_part_prg.into_seed();

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

            let joint_rand_seed = Self::derive_joint_rand_seed(corrected_joint_rand_parts);

            let joint_rand_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &joint_rand_seed,
                &Self::custom(DST_JOINT_RANDOMNESS),
                &[],
            ));
            (
                Some(joint_rand_seed),
                Some(own_joint_rand_part),
                joint_rand_prng.take(self.typ.joint_rand_len()).collect(),
            )
        } else {
            (None, None, Vec::new())
        };

        // Compute the query randomness.
        let query_rand: Vec<T::Field> = query_rand_prng.take(self.typ.query_rand_len()).collect();

        // Run the query-generation algorithm.
        let verifier_share = self.typ.query(
            measurement_share,
            proof_share,
            &query_rand,
            &joint_rand,
            self.num_aggregators as usize,
        )?;

        Ok((
            Prio3PrepareState {
                measurement_share: msg.measurement_share.clone(),
                joint_rand_seed,
                agg_id,
                verifier_len: verifier_share.len(),
            },
            Prio3PrepareShare {
                verifier: verifier_share,
                joint_rand_part,
            },
        ))
    }

    fn prepare_preprocess<M: IntoIterator<Item = Prio3PrepareShare<T::Field, L>>>(
        &self,
        inputs: M,
    ) -> Result<Prio3PrepareMessage<L>, VdafError> {
        let mut verifier = vec![T::Field::zero(); self.typ.verifier_len()];
        let mut joint_rand_parts = Vec::with_capacity(self.num_aggregators());
        let mut count = 0;
        for share in inputs.into_iter() {
            count += 1;

            if share.verifier.len() != verifier.len() {
                return Err(VdafError::Uncategorized(format!(
                    "unexpected verifier share length: got {}; want {}",
                    share.verifier.len(),
                    verifier.len(),
                )));
            }

            if self.typ.joint_rand_len() > 0 {
                let joint_rand_seed_part = share.joint_rand_part.unwrap();
                joint_rand_parts.push(joint_rand_seed_part);
            }

            for (x, y) in verifier.iter_mut().zip(share.verifier) {
                *x += y;
            }
        }

        if count != self.num_aggregators {
            return Err(VdafError::Uncategorized(format!(
                "unexpected message count: got {}; want {}",
                count, self.num_aggregators,
            )));
        }

        // Check the proof verifier.
        match self.typ.decide(&verifier) {
            Ok(true) => (),
            Ok(false) => {
                return Err(VdafError::Uncategorized(
                    "proof verifier check failed".into(),
                ))
            }
            Err(err) => return Err(VdafError::from(err)),
        };

        let joint_rand_seed = if self.typ.joint_rand_len() > 0 {
            Some(Self::derive_joint_rand_seed(joint_rand_parts.iter()))
        } else {
            None
        };

        Ok(Prio3PrepareMessage { joint_rand_seed })
    }

    fn prepare_step(
        &self,
        step: Prio3PrepareState<T::Field, L>,
        msg: Prio3PrepareMessage<L>,
    ) -> Result<PrepareTransition<Self, L, 16>, VdafError> {
        if self.typ.joint_rand_len() > 0 {
            // Check that the joint randomness was correct.
            if step.joint_rand_seed.as_ref().unwrap() != msg.joint_rand_seed.as_ref().unwrap() {
                return Err(VdafError::Uncategorized(
                    "joint randomness mismatch".to_string(),
                ));
            }
        }

        // Compute the output share.
        let measurement_share = match step.measurement_share {
            Share::Leader(data) => data,
            Share::Helper(seed) => {
                let custom = Self::custom(DST_MEASUREMENT_SHARE);
                let prng = Prng::from_seed_stream(P::seed_stream(&seed, &custom, &[step.agg_id]));
                prng.take(self.typ.input_len()).collect()
            }
        };

        let output_share = match self.typ.truncate(measurement_share) {
            Ok(data) => OutputShare(data),
            Err(err) => {
                return Err(VdafError::from(err));
            }
        };

        Ok(PrepareTransition::Finish(output_share))
    }

    /// Aggregates a sequence of output shares into an aggregate share.
    fn aggregate<It: IntoIterator<Item = OutputShare<T::Field>>>(
        &self,
        _agg_param: &(),
        output_shares: It,
    ) -> Result<AggregateShare<T::Field>, VdafError> {
        let mut agg_share = AggregateShare(vec![T::Field::zero(); self.typ.output_len()]);
        for output_share in output_shares.into_iter() {
            agg_share.accumulate(&output_share)?;
        }

        Ok(agg_share)
    }
}

impl<T, P, const L: usize> Collector for Prio3<T, P, L>
where
    T: Type,
    P: Prg<L>,
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

#[derive(Clone)]
struct HelperShare<const L: usize> {
    measurement_share: Seed<L>,
    proof_share: Seed<L>,
    joint_rand_blind: Seed<L>,
}

impl<const L: usize> HelperShare<L> {
    fn from_rand_source(rand_source: RandSource) -> Result<Self, VdafError> {
        Ok(HelperShare {
            measurement_share: Seed::from_rand_source(rand_source)?,
            proof_share: Seed::from_rand_source(rand_source)?,
            joint_rand_blind: Seed::from_rand_source(rand_source)?,
        })
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

impl<'a, F, T, P, const L: usize> ParameterizedDecode<(&'a Prio3<T, P, L>, &'a ())>
    for OutputShare<F>
where
    F: FieldElement,
    T: Type,
    P: Prg<L>,
{
    fn decode_with_param(
        (vdaf, _): &(&'a Prio3<T, P, L>, &'a ()),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        decode_fieldvec(vdaf.output_len(), bytes).map(Self)
    }
}

impl<'a, F, T, P, const L: usize> ParameterizedDecode<(&'a Prio3<T, P, L>, &'a ())>
    for AggregateShare<F>
where
    F: FieldElement,
    T: Type,
    P: Prg<L>,
{
    fn decode_with_param(
        (vdaf, _): &(&'a Prio3<T, P, L>, &'a ()),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        decode_fieldvec(vdaf.output_len(), bytes).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "experimental")]
    use crate::flp::gadgets::ParallelSumGadget;
    use crate::vdaf::{fieldvec_roundtrip_test, run_vdaf, run_vdaf_prepare};
    use assert_matches::assert_matches;
    #[cfg(feature = "experimental")]
    use fixed::{
        types::extra::{U15, U31, U63},
        FixedI16, FixedI32, FixedI64,
    };
    #[cfg(feature = "experimental")]
    use fixed_macro::fixed;
    use rand::prelude::*;

    #[test]
    fn test_prio3_count() {
        let prio3 = Prio3::new_aes128_count(2).unwrap();

        assert_eq!(run_vdaf(&prio3, &(), [1, 0, 0, 1, 1]).unwrap(), 3);

        let mut nonce = [0; 16];
        let mut verify_key = [0; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let (public_share, input_shares) = prio3.shard(&0, &nonce).unwrap();
        run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares).unwrap();

        let (public_share, input_shares) = prio3.shard(&1, &nonce).unwrap();
        run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares).unwrap();

        test_prepare_state_serialization(&prio3, &1, &nonce).unwrap();

        let prio3_extra_helper = Prio3::new_aes128_count(3).unwrap();
        assert_eq!(
            run_vdaf(&prio3_extra_helper, &(), [1, 0, 0, 1, 1]).unwrap(),
            3,
        );
    }

    #[test]
    fn test_prio3_sum() {
        let prio3 = Prio3::new_aes128_sum(3, 16).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [0, (1 << 16) - 1, 0, 1, 1]).unwrap(),
            (1 << 16) + 1
        );

        let mut verify_key = [0; 16];
        thread_rng().fill(&mut verify_key[..]);
        let nonce = [0; 16];

        let (public_share, mut input_shares) = prio3.shard(&1, &nonce).unwrap();
        input_shares[0].joint_rand_blind.as_mut().unwrap().0[0] ^= 255;
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let (public_share, mut input_shares) = prio3.shard(&1, &nonce).unwrap();
        assert_matches!(input_shares[0].measurement_share, Share::Leader(ref mut data) => {
            data[0] += Field128::one();
        });
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let (public_share, mut input_shares) = prio3.shard(&1, &nonce).unwrap();
        assert_matches!(input_shares[0].proof_share, Share::Leader(ref mut data) => {
                data[0] += Field128::one();
        });
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        test_prepare_state_serialization(&prio3, &1, &nonce).unwrap();
    }

    #[test]
    fn test_prio3_sum_vec() {
        let prio3 = Prio3::new_aes128_sum_vec(2, 2, 20).unwrap();
        assert_eq!(
            run_vdaf(
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
        let prio3 = Prio3::new_aes128_sum_vec_multithreaded(2, 2, 20).unwrap();
        assert_eq!(
            run_vdaf(
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
        type P<Fx> = Prio3Aes128FixedPointBoundedL2VecSum<Fx>;
        #[cfg(feature = "multithreaded")]
        type PM<Fx> = Prio3Aes128FixedPointBoundedL2VecSumMultithreaded<Fx>;
        let ctor_32 = P::<FixedI32<U31>>::new_aes128_fixedpoint_boundedl2_vec_sum;
        #[cfg(feature = "multithreaded")]
        let ctor_mt_32 = PM::<FixedI32<U31>>::new_aes128_fixedpoint_boundedl2_vec_sum_multithreaded;

        {
            const SIZE: usize = 5;
            let fp32_0 = fixed!(0: I1F31);

            // 32 bit fixedpoint, non-power-of-2 vector, single-threaded
            {
                let prio3_32 = ctor_32(2, SIZE).unwrap();
                test_fixed_vec::<_, _, _, SIZE>(fp32_0, prio3_32);
            }

            // 32 bit fixedpoint, non-power-of-2 vector, multi-threaded
            #[cfg(feature = "multithreaded")]
            {
                let prio3_mt_32 = ctor_mt_32(2, SIZE).unwrap();
                test_fixed_vec::<_, _, _, SIZE>(fp32_0, prio3_mt_32);
            }
        }

        fn test_fixed_vec<Fx, PE, BPE, const SIZE: usize>(
            fp_0: Fx,
            prio3: Prio3<FixedPointBoundedL2VecSum<Fx, Field128, PE, BPE>, PrgAes128, 16>,
        ) where
            Fx: Fixed + CompatibleFloat<Field128> + std::ops::Neg<Output = Fx>,
            PE: Eq + ParallelSumGadget<Field128, PolyEval<Field128>> + Clone + 'static,
            BPE: Eq + ParallelSumGadget<Field128, BlindPolyEval<Field128>> + Clone + 'static,
        {
            let fp_vec = vec![fp_0; SIZE];

            let measurements = [fp_vec.clone(), fp_vec];
            assert_eq!(
                run_vdaf(&prio3, &(), measurements).unwrap(),
                vec![0.0; SIZE]
            );
        }
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn test_prio3_bounded_fpvec_sum() {
        type P<Fx> = Prio3Aes128FixedPointBoundedL2VecSum<Fx>;
        let ctor_16 = P::<FixedI16<U15>>::new_aes128_fixedpoint_boundedl2_vec_sum;
        let ctor_32 = P::<FixedI32<U31>>::new_aes128_fixedpoint_boundedl2_vec_sum;
        let ctor_64 = P::<FixedI64<U63>>::new_aes128_fixedpoint_boundedl2_vec_sum;

        #[cfg(feature = "multithreaded")]
        type PM<Fx> = Prio3Aes128FixedPointBoundedL2VecSumMultithreaded<Fx>;
        #[cfg(feature = "multithreaded")]
        let ctor_mt_16 = PM::<FixedI16<U15>>::new_aes128_fixedpoint_boundedl2_vec_sum_multithreaded;
        #[cfg(feature = "multithreaded")]
        let ctor_mt_32 = PM::<FixedI32<U31>>::new_aes128_fixedpoint_boundedl2_vec_sum_multithreaded;
        #[cfg(feature = "multithreaded")]
        let ctor_mt_64 = PM::<FixedI64<U63>>::new_aes128_fixedpoint_boundedl2_vec_sum_multithreaded;

        {
            // 16 bit fixedpoint
            let fp16_4_inv = fixed!(0.25: I1F15);
            let fp16_8_inv = fixed!(0.125: I1F15);
            let fp16_16_inv = fixed!(0.0625: I1F15);

            // two aggregators, three entries per vector.
            {
                let prio3_16 = ctor_16(2, 3).unwrap();
                test_fixed(fp16_4_inv, fp16_8_inv, fp16_16_inv, prio3_16);
            }

            #[cfg(feature = "multithreaded")]
            {
                let prio3_16_mt = ctor_mt_16(2, 3).unwrap();
                test_fixed(fp16_4_inv, fp16_8_inv, fp16_16_inv, prio3_16_mt);
            }
        }

        {
            // 32 bit fixedpoint
            let fp32_4_inv = fixed!(0.25: I1F31);
            let fp32_8_inv = fixed!(0.125: I1F31);
            let fp32_16_inv = fixed!(0.0625: I1F31);

            {
                let prio3_32 = ctor_32(2, 3).unwrap();
                test_fixed(fp32_4_inv, fp32_8_inv, fp32_16_inv, prio3_32);
            }

            #[cfg(feature = "multithreaded")]
            {
                let prio3_32_mt = ctor_mt_32(2, 3).unwrap();
                test_fixed(fp32_4_inv, fp32_8_inv, fp32_16_inv, prio3_32_mt);
            }
        }

        {
            // 64 bit fixedpoint
            let fp64_4_inv = fixed!(0.25: I1F63);
            let fp64_8_inv = fixed!(0.125: I1F63);
            let fp64_16_inv = fixed!(0.0625: I1F63);

            {
                let prio3_64 = ctor_64(2, 3).unwrap();
                test_fixed(fp64_4_inv, fp64_8_inv, fp64_16_inv, prio3_64);
            }

            #[cfg(feature = "multithreaded")]
            {
                let prio3_64_mt = ctor_mt_64(2, 3).unwrap();
                test_fixed(fp64_4_inv, fp64_8_inv, fp64_16_inv, prio3_64_mt);
            }
        }

        fn test_fixed<Fx, PE, BPE>(
            fp_4_inv: Fx,
            fp_8_inv: Fx,
            fp_16_inv: Fx,
            prio3: Prio3<FixedPointBoundedL2VecSum<Fx, Field128, PE, BPE>, PrgAes128, 16>,
        ) where
            Fx: Fixed + CompatibleFloat<Field128> + std::ops::Neg<Output = Fx>,
            PE: Eq + ParallelSumGadget<Field128, PolyEval<Field128>> + Clone + 'static,
            BPE: Eq + ParallelSumGadget<Field128, BlindPolyEval<Field128>> + Clone + 'static,
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
                run_vdaf(&prio3, &(), fp_list).unwrap(),
                vec!(0.5, 0.25, 0.125),
            );

            // negative entries
            let fp_list2 = [fp_vec3, fp_vec4];
            assert_eq!(
                run_vdaf(&prio3, &(), fp_list2).unwrap(),
                vec!(-0.5, -0.25, -0.125),
            );

            // both
            let fp_list3 = [fp_vec5, fp_vec6];
            assert_eq!(
                run_vdaf(&prio3, &(), fp_list3).unwrap(),
                vec!(0.5, 0.0, 0.0),
            );

            let mut verify_key = [0; 16];
            let mut nonce = [0; 16];
            thread_rng().fill(&mut verify_key);
            thread_rng().fill(&mut nonce);

            let (public_share, mut input_shares) = prio3
                .shard(&vec![fp_4_inv, fp_8_inv, fp_16_inv], &nonce)
                .unwrap();
            input_shares[0].joint_rand_blind.as_mut().unwrap().0[0] ^= 255;
            let result =
                run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
            assert_matches!(result, Err(VdafError::Uncategorized(_)));

            let (public_share, mut input_shares) = prio3
                .shard(&vec![fp_4_inv, fp_8_inv, fp_16_inv], &nonce)
                .unwrap();
            assert_matches!(input_shares[0].measurement_share, Share::Leader(ref mut data) => {
                data[0] += Field128::one();
            });
            let result =
                run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
            assert_matches!(result, Err(VdafError::Uncategorized(_)));

            let (public_share, mut input_shares) = prio3
                .shard(&vec![fp_4_inv, fp_8_inv, fp_16_inv], &nonce)
                .unwrap();
            assert_matches!(input_shares[0].proof_share, Share::Leader(ref mut data) => {
                    data[0] += Field128::one();
            });
            let result =
                run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
            assert_matches!(result, Err(VdafError::Uncategorized(_)));

            test_prepare_state_serialization(&prio3, &vec![fp_4_inv, fp_8_inv, fp_16_inv], &nonce)
                .unwrap();
        }
    }

    #[test]
    fn test_prio3_histogram() {
        let prio3 = Prio3::new_aes128_histogram(2, &[0, 10, 20]).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [0, 10, 20, 9999]).unwrap(),
            vec![1, 1, 1, 1]
        );
        assert_eq!(run_vdaf(&prio3, &(), [0]).unwrap(), vec![1, 0, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [5]).unwrap(), vec![0, 1, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [10]).unwrap(), vec![0, 1, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [15]).unwrap(), vec![0, 0, 1, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [20]).unwrap(), vec![0, 0, 1, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [25]).unwrap(), vec![0, 0, 0, 1]);
        test_prepare_state_serialization(&prio3, &23, &[0; 16]).unwrap();
    }

    #[test]
    fn test_prio3_average() {
        let prio3 = Prio3::new_aes128_average(2, 64).unwrap();

        assert_eq!(run_vdaf(&prio3, &(), [17, 8]).unwrap(), 12.5f64);
        assert_eq!(run_vdaf(&prio3, &(), [1, 1, 1, 1]).unwrap(), 1f64);
        assert_eq!(run_vdaf(&prio3, &(), [0, 0, 0, 1]).unwrap(), 0.25f64);
        assert_eq!(
            run_vdaf(&prio3, &(), [1, 11, 111, 1111, 3, 8]).unwrap(),
            207.5f64
        );
    }

    #[test]
    fn test_prio3_input_share() {
        let prio3 = Prio3::new_aes128_sum(5, 16).unwrap();
        let (_public_share, input_shares) = prio3.shard(&1, &[0; 16]).unwrap();

        // Check that seed shares are distinct.
        for (i, x) in input_shares.iter().enumerate() {
            for (j, y) in input_shares.iter().enumerate() {
                if i != j {
                    if let (Share::Helper(left), Share::Helper(right)) =
                        (&x.measurement_share, &y.measurement_share)
                    {
                        assert_ne!(left, right);
                    }

                    if let (Share::Helper(left), Share::Helper(right)) =
                        (&x.proof_share, &y.proof_share)
                    {
                        assert_ne!(left, right);
                    }

                    assert_ne!(x.joint_rand_blind, y.joint_rand_blind);
                }
            }
        }
    }

    fn test_prepare_state_serialization<T, P, const L: usize>(
        prio3: &Prio3<T, P, L>,
        measurement: &T::Measurement,
        nonce: &[u8; 16],
    ) -> Result<(), VdafError>
    where
        T: Type,
        P: Prg<L>,
    {
        let mut verify_key = [0; L];
        thread_rng().fill(&mut verify_key[..]);
        let (public_share, input_shares) = prio3.shard(measurement, nonce)?;
        for (agg_id, input_share) in input_shares.iter().enumerate() {
            let (want, _msg) =
                prio3.prepare_init(&verify_key, agg_id, &(), nonce, &public_share, input_share)?;
            let got =
                Prio3PrepareState::get_decoded_with_param(&(prio3, agg_id), &want.get_encoded())
                    .expect("failed to decode prepare step");
            assert_eq!(got, want);
        }
        Ok(())
    }

    #[test]
    fn roundtrip_output_share() {
        let vdaf = Prio3::new_aes128_count(2).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Aes128Count, OutputShare<Field64>>(&vdaf, &(), 1);

        let vdaf = Prio3::new_aes128_sum(2, 17).unwrap();
        fieldvec_roundtrip_test::<Field128, Prio3Aes128Sum, OutputShare<Field128>>(&vdaf, &(), 1);

        let vdaf = Prio3::new_aes128_histogram(2, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unwrap();
        fieldvec_roundtrip_test::<Field128, Prio3Aes128Histogram, OutputShare<Field128>>(
            &vdaf,
            &(),
            12,
        );
    }

    #[test]
    fn roundtrip_aggregate_share() {
        let vdaf = Prio3::new_aes128_count(2).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Aes128Count, AggregateShare<Field64>>(
            &vdaf,
            &(),
            1,
        );

        let vdaf = Prio3::new_aes128_sum(2, 17).unwrap();
        fieldvec_roundtrip_test::<Field128, Prio3Aes128Sum, AggregateShare<Field128>>(
            &vdaf,
            &(),
            1,
        );

        let vdaf = Prio3::new_aes128_histogram(2, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unwrap();
        fieldvec_roundtrip_test::<Field128, Prio3Aes128Histogram, AggregateShare<Field128>>(
            &vdaf,
            &(),
            12,
        );
    }
}
