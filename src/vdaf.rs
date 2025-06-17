// SPDX-License-Identifier: MPL-2.0

//! Verifiable Distributed Aggregation Functions (VDAFs) as described in
//! [[draft-irtf-cfrg-vdaf-08]].
//!
//! [draft-irtf-cfrg-vdaf-08]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/08/

#[cfg(feature = "experimental")]
use crate::dp::DifferentialPrivacyStrategy;
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use crate::idpf::IdpfError;
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use crate::vdaf::mastic::szk::SzkError;
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use crate::vidpf::VidpfError;
use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{encode_fieldvec, merge_vector, FieldElement, FieldError},
    flp::FlpError,
    vdaf::xof::Seed,
};
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt::Debug, io::Cursor};
use subtle::{Choice, ConstantTimeEq};

/// A component of the domain-separation tag, used to bind the VDAF operations to the document
/// version. This will be revised with each draft with breaking changes.
pub(crate) const VERSION: u8 = 12;

/// Errors emitted by this module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VdafError {
    /// An error occurred.
    #[error("vdaf error: {0}")]
    Uncategorized(String),

    /// Field error.
    #[error("field error: {0}")]
    Field(#[from] FieldError),

    /// An error occured while parsing a message.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// FLP error.
    #[error("flp error: {0}")]
    Flp(#[from] FlpError),

    /// SZK error.
    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    #[error("Szk error: {0}")]
    Szk(#[from] SzkError),

    /// IDPF error.
    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    #[error("idpf error: {0}")]
    Idpf(#[from] IdpfError),

    /// VIDPF error.
    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    #[error("vidpf error: {0}")]
    Vidpf(#[from] VidpfError),

    /// Errors from other VDAFs.
    #[error(transparent)]
    Other(Box<dyn Error + 'static + Send + Sync>),
}

/// An additive share of a vector of field elements.
#[derive(Clone, Debug)]
pub enum Share<F, const SEED_SIZE: usize> {
    /// An uncompressed share, typically sent to the leader.
    Leader(Vec<F>),

    /// A compressed share, typically sent to the helper.
    Helper(Seed<SEED_SIZE>),
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Share<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Share<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Share<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> subtle::Choice {
        // We allow short-circuiting on the type (Leader vs Helper) of the value, but not the types'
        // contents.
        match (self, other) {
            (Share::Leader(self_val), Share::Leader(other_val)) => self_val.ct_eq(other_val),
            (Share::Helper(self_val), Share::Helper(other_val)) => self_val.ct_eq(other_val),
            _ => Choice::from(0),
        }
    }
}

/// Parameters needed to decode a [`Share`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShareDecodingParameter<const SEED_SIZE: usize> {
    Leader(usize),
    Helper,
}

impl<F: FieldElement, const SEED_SIZE: usize> ParameterizedDecode<ShareDecodingParameter<SEED_SIZE>>
    for Share<F, SEED_SIZE>
{
    fn decode_with_param(
        decoding_parameter: &ShareDecodingParameter<SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        match decoding_parameter {
            ShareDecodingParameter::Leader(share_length) => {
                let mut data = Vec::with_capacity(*share_length);
                for _ in 0..*share_length {
                    data.push(F::decode(bytes)?)
                }
                Ok(Self::Leader(data))
            }
            ShareDecodingParameter::Helper => {
                let seed = Seed::decode(bytes)?;
                Ok(Self::Helper(seed))
            }
        }
    }
}

impl<F: FieldElement, const SEED_SIZE: usize> Encode for Share<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match self {
            Share::Leader(share_data) => {
                for x in share_data {
                    x.encode(bytes)?;
                }
                Ok(())
            }
            Share::Helper(share_seed) => share_seed.encode(bytes),
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            Share::Leader(share_data) => {
                // Each element of the data vector has the same size.
                Some(share_data.len() * F::ENCODED_SIZE)
            }
            Share::Helper(share_seed) => share_seed.encoded_len(),
        }
    }
}

/// The base trait for VDAF schemes. This trait is inherited by traits [`Client`], [`Aggregator`],
/// and [`Collector`], which define the roles of the various parties involved in the execution of
/// the VDAF.
pub trait Vdaf: Clone + Debug {
    /// The type of Client measurement to be aggregated.
    type Measurement: Clone + Debug;

    /// The aggregate result of the VDAF execution.
    type AggregateResult: Clone + Debug;

    /// The aggregation parameter, used by the Aggregators to map their input shares to output
    /// shares.
    type AggregationParam: Clone + Debug + Decode + Encode;

    /// A public share sent by a Client.
    type PublicShare: Clone + Debug + ParameterizedDecode<Self> + Encode;

    /// An input share sent by a Client.
    type InputShare: Clone + Debug + for<'a> ParameterizedDecode<(&'a Self, usize)> + Encode;

    /// An output share recovered from an input share by an Aggregator.
    type OutputShare: Clone
        + Debug
        + for<'a> ParameterizedDecode<(&'a Self, &'a Self::AggregationParam)>
        + Encode;

    /// An Aggregator's share of the aggregate result.
    type AggregateShare: Aggregatable<OutputShare = Self::OutputShare>
        + for<'a> ParameterizedDecode<(&'a Self, &'a Self::AggregationParam)>
        + Encode;

    /// Return the VDAF's algorithm ID.
    fn algorithm_id(&self) -> u32;

    /// The number of Aggregators. The Client generates as many input shares as there are
    /// Aggregators.
    fn num_aggregators(&self) -> usize;
}

/// The Client's role in the execution of a VDAF.
pub trait Client<const NONCE_SIZE: usize>: Vdaf {
    /// Shards a measurement into a public share and a sequence of input shares, one for each
    /// Aggregator.
    ///
    /// Implements `Vdaf::shard` from [VDAF].
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-13#section-5.1
    fn shard(
        &self,
        ctx: &[u8],
        measurement: &Self::Measurement,
        nonce: &[u8; NONCE_SIZE],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError>;
}

/// The Aggregator's role in the execution of a VDAF.
pub trait Aggregator<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>: Vdaf {
    /// State of the Aggregator during the Prepare process.
    type PrepareState: Clone + Debug + PartialEq + Eq;

    /// The type of messages sent by each aggregator at each round of the Prepare Process.
    ///
    /// Decoding takes a [`Self::PrepareState`] as a parameter; this [`Self::PrepareState`] may be
    /// associated with any aggregator involved in the execution of the VDAF.
    type PrepareShare: Clone + Debug + ParameterizedDecode<Self::PrepareState> + Encode;

    /// Result of preprocessing a round of preparation shares. This is used by all aggregators as an
    /// input to the next round of the Prepare Process.
    ///
    /// Decoding takes a [`Self::PrepareState`] as a parameter; this [`Self::PrepareState`] may be
    /// associated with any aggregator involved in the execution of the VDAF.
    type PrepareMessage: Clone
        + Debug
        + PartialEq
        + Eq
        + ParameterizedDecode<Self::PrepareState>
        + Encode;

    /// Begins the Prepare process with the other Aggregators. The [`Self::PrepareState`] returned
    /// is passed to [`Self::prepare_next`] to get this aggregator's first-round prepare message.
    ///
    /// Implements `Vdaf.prep_init` from [VDAF].
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.2
    #[allow(clippy::too_many_arguments)]
    fn prepare_init(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        agg_id: usize,
        agg_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::PrepareState, Self::PrepareShare), VdafError>;

    /// Preprocess a round of preparation shares into a single input to [`Self::prepare_next`].
    ///
    /// Implements `Vdaf.prep_shares_to_prep` from [VDAF].
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.2
    fn prepare_shares_to_prepare_message<M: IntoIterator<Item = Self::PrepareShare>>(
        &self,
        ctx: &[u8],
        agg_param: &Self::AggregationParam,
        inputs: M,
    ) -> Result<Self::PrepareMessage, VdafError>;

    /// Compute the next state transition from the current state and the previous round of input
    /// messages. If this returns [`PrepareTransition::Continue`], then the returned
    /// [`Self::PrepareShare`] should be combined with the other Aggregators' `PrepareShare`s from
    /// this round and passed into another call to this method. This continues until this method
    /// returns [`PrepareTransition::Finish`], at which point the returned output share may be
    /// aggregated. If the method returns an error, the aggregator should consider its input share
    /// invalid and not attempt to process it any further.
    ///
    /// Implements `Vdaf.prep_next` from [VDAF].
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.2
    fn prepare_next(
        &self,
        ctx: &[u8],
        state: Self::PrepareState,
        input: Self::PrepareMessage,
    ) -> Result<PrepareTransition<Self, VERIFY_KEY_SIZE, NONCE_SIZE>, VdafError>;

    /// Aggregates a sequence of output shares into an aggregate share.
    fn aggregate<M: IntoIterator<Item = Self::OutputShare>>(
        &self,
        agg_param: &Self::AggregationParam,
        output_shares: M,
    ) -> Result<Self::AggregateShare, VdafError> {
        let mut share = self.aggregate_init(agg_param);
        for output_share in output_shares {
            share.accumulate(&output_share)?;
        }
        Ok(share)
    }

    /// Create an empty aggregate share.
    fn aggregate_init(&self, agg_param: &Self::AggregationParam) -> Self::AggregateShare;

    /// Validates an aggregation parameter with respect to all previous aggregaiton parameters used
    /// for the same input share. `prev` MUST be sorted from least to most recently used.
    #[must_use]
    fn is_agg_param_valid(cur: &Self::AggregationParam, prev: &[Self::AggregationParam]) -> bool;
}

/// Aggregator that implements differential privacy with Aggregator-side noise addition.
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub trait AggregatorWithNoise<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    DPStrategy: DifferentialPrivacyStrategy,
>: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    /// Adds noise to an aggregate share such that the aggregate result is differentially private
    /// as long as one Aggregator is honest.
    fn add_noise_to_agg_share(
        &self,
        dp_strategy: &DPStrategy,
        agg_param: &Self::AggregationParam,
        agg_share: &mut Self::AggregateShare,
        num_measurements: usize,
    ) -> Result<(), VdafError>;
}

/// The Collector's role in the execution of a VDAF.
pub trait Collector: Vdaf {
    /// Combines aggregate shares into the aggregate result.
    fn unshard<M: IntoIterator<Item = Self::AggregateShare>>(
        &self,
        agg_param: &Self::AggregationParam,
        agg_shares: M,
        num_measurements: usize,
    ) -> Result<Self::AggregateResult, VdafError>;
}

/// A state transition of an Aggregator during the Prepare process.
#[derive(Clone, Debug)]
pub enum PrepareTransition<
    V: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
> {
    /// Continue processing.
    Continue(V::PrepareState, V::PrepareShare),

    /// Finish processing and return the output share.
    Finish(V::OutputShare),
}

/// An aggregate share resulting from aggregating output shares together that
/// can merged with aggregate shares of the same type.
pub trait Aggregatable: Clone + Debug + From<Self::OutputShare> {
    /// Type of output shares that can be accumulated into an aggregate share.
    type OutputShare;

    /// Update an aggregate share by merging it with another (`agg_share`).
    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError>;

    /// Update an aggregate share by adding `output_share`.
    fn accumulate(&mut self, output_share: &Self::OutputShare) -> Result<(), VdafError>;
}

/// An output share comprised of a vector of field elements.
#[derive(Clone)]
pub struct OutputShare<F>(Vec<F>);

impl<F: ConstantTimeEq> PartialEq for OutputShare<F> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq> Eq for OutputShare<F> {}

impl<F: ConstantTimeEq> ConstantTimeEq for OutputShare<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl<F> AsRef<[F]> for OutputShare<F> {
    fn as_ref(&self) -> &[F] {
        &self.0
    }
}

impl<F> From<Vec<F>> for OutputShare<F> {
    fn from(other: Vec<F>) -> Self {
        Self(other)
    }
}

impl<F: FieldElement> Encode for OutputShare<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        encode_fieldvec(&self.0, bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(F::ENCODED_SIZE * self.0.len())
    }
}

impl<F> Debug for OutputShare<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OutputShare").finish()
    }
}

/// An aggregate share comprised of a vector of field elements.
///
/// This is suitable for VDAFs where both output shares and aggregate shares are vectors of field
/// elements, and output shares need no special transformation to be merged into an aggregate share.
#[derive(Clone, Debug, Serialize, Deserialize)]

pub struct AggregateShare<F>(Vec<F>);

impl<F> From<Vec<F>> for AggregateShare<F> {
    fn from(other: Vec<F>) -> Self {
        Self(other)
    }
}

impl<F: ConstantTimeEq> PartialEq for AggregateShare<F> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq> Eq for AggregateShare<F> {}

impl<F: ConstantTimeEq> ConstantTimeEq for AggregateShare<F> {
    fn ct_eq(&self, other: &Self) -> subtle::Choice {
        self.0.ct_eq(&other.0)
    }
}

impl<F: FieldElement> AsRef<[F]> for AggregateShare<F> {
    fn as_ref(&self) -> &[F] {
        &self.0
    }
}

impl<F> From<OutputShare<F>> for AggregateShare<F> {
    fn from(other: OutputShare<F>) -> Self {
        Self(other.0)
    }
}

impl<F: FieldElement> Aggregatable for AggregateShare<F> {
    type OutputShare = OutputShare<F>;

    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError> {
        self.sum(agg_share.as_ref())
    }

    fn accumulate(&mut self, output_share: &Self::OutputShare) -> Result<(), VdafError> {
        // For Poplar1, Prio2, and Prio3, no conversion is needed between output shares and
        // aggregate shares.
        self.sum(output_share.as_ref())
    }
}

impl<F: FieldElement> AggregateShare<F> {
    fn sum(&mut self, other: &[F]) -> Result<(), VdafError> {
        merge_vector(&mut self.0, other).map_err(Into::into)
    }
}

impl<F: FieldElement> Encode for AggregateShare<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        encode_fieldvec(&self.0, bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(F::ENCODED_SIZE * self.0.len())
    }
}

/// Utilities for testing VDAFs.
#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod test_utils {
    use std::collections::HashMap;

    use super::{Aggregatable, Aggregator, Client, Collector, PrepareTransition, VdafError};
    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    use crate::vdaf::poplar1::Poplar1;
    use crate::{
        codec::{Decode, Encode, ParameterizedDecode},
        flp::Type,
        vdaf::{prio3::Prio3, xof::Xof, Vdaf},
    };
    use rand::{random, rng, Rng};
    use serde::Deserialize;
    use serde_json::Value;

    /// Execute the VDAF end-to-end and return the aggregate result.
    pub fn run_vdaf<V, M, const SEED_SIZE: usize>(
        ctx: &[u8],
        vdaf: &V,
        agg_param: &V::AggregationParam,
        measurements: M,
    ) -> Result<V::AggregateResult, VdafError>
    where
        V: Client<16> + Aggregator<SEED_SIZE, 16> + Collector,
        M: IntoIterator<Item = V::Measurement>,
    {
        let mut sharded_measurements = Vec::new();
        for measurement in measurements.into_iter() {
            let nonce = random();
            let (public_share, input_shares) = vdaf.shard(ctx, &measurement, &nonce)?;

            sharded_measurements.push((public_share, nonce, input_shares));
        }

        run_vdaf_sharded(ctx, vdaf, agg_param, sharded_measurements)
    }

    /// Execute the VDAF on sharded measurements and return the aggregate result.
    pub fn run_vdaf_sharded<V, M, I, const SEED_SIZE: usize>(
        ctx: &[u8],
        vdaf: &V,
        agg_param: &V::AggregationParam,
        sharded_measurements: M,
    ) -> Result<V::AggregateResult, VdafError>
    where
        V: Client<16> + Aggregator<SEED_SIZE, 16> + Collector,
        M: IntoIterator<Item = (V::PublicShare, [u8; 16], I)>,
        I: IntoIterator<Item = V::InputShare>,
    {
        let mut rng = rng();
        let mut verify_key = [0; SEED_SIZE];
        rng.fill(&mut verify_key[..]);

        let mut agg_shares: Vec<Option<V::AggregateShare>> = vec![None; vdaf.num_aggregators()];
        let mut num_measurements: usize = 0;
        for (public_share, nonce, input_shares) in sharded_measurements.into_iter() {
            num_measurements += 1;
            let out_shares = run_vdaf_prepare(
                vdaf,
                &verify_key,
                ctx,
                agg_param,
                &nonce,
                public_share,
                input_shares,
            )?;
            for (out_share, agg_share) in out_shares.into_iter().zip(agg_shares.iter_mut()) {
                // Check serialization of output shares
                let encoded_out_share = out_share.get_encoded().unwrap();
                let round_trip_out_share =
                    V::OutputShare::get_decoded_with_param(&(vdaf, agg_param), &encoded_out_share)
                        .unwrap();
                assert_eq!(
                    round_trip_out_share.get_encoded().unwrap(),
                    encoded_out_share
                );

                let this_agg_share = V::AggregateShare::from(out_share);
                if let Some(ref mut inner) = agg_share {
                    inner.merge(&this_agg_share)?;
                } else {
                    *agg_share = Some(this_agg_share);
                }
            }
        }

        for agg_share in agg_shares.iter() {
            // Check serialization of aggregate shares
            let encoded_agg_share = agg_share.as_ref().unwrap().get_encoded().unwrap();
            let round_trip_agg_share =
                V::AggregateShare::get_decoded_with_param(&(vdaf, agg_param), &encoded_agg_share)
                    .unwrap();
            assert_eq!(
                round_trip_agg_share.get_encoded().unwrap(),
                encoded_agg_share
            );
        }

        let res = vdaf.unshard(
            agg_param,
            agg_shares.into_iter().map(|option| option.unwrap()),
            num_measurements,
        )?;
        Ok(res)
    }

    /// Execute VDAF preparation for a single report and return the recovered output shares.
    pub fn run_vdaf_prepare<V, M, const SEED_SIZE: usize>(
        vdaf: &V,
        verify_key: &[u8; SEED_SIZE],
        ctx: &[u8],
        agg_param: &V::AggregationParam,
        nonce: &[u8; 16],
        public_share: V::PublicShare,
        input_shares: M,
    ) -> Result<Vec<V::OutputShare>, VdafError>
    where
        V: Client<16> + Aggregator<SEED_SIZE, 16> + Collector,
        M: IntoIterator<Item = V::InputShare>,
    {
        let public_share =
            V::PublicShare::get_decoded_with_param(vdaf, &public_share.get_encoded().unwrap())
                .unwrap();
        let input_shares = input_shares
            .into_iter()
            .map(|input_share| input_share.get_encoded().unwrap());

        let mut states = Vec::new();
        let mut outbound = Vec::new();
        for (agg_id, input_share) in input_shares.enumerate() {
            let (state, msg) = vdaf.prepare_init(
                verify_key,
                ctx,
                agg_id,
                agg_param,
                nonce,
                &public_share,
                &V::InputShare::get_decoded_with_param(&(vdaf, agg_id), &input_share)
                    .expect("failed to decode input share"),
            )?;
            states.push(state);
            outbound.push(msg.get_encoded().unwrap());
        }

        let mut inbound = vdaf
            .prepare_shares_to_prepare_message(
                ctx,
                agg_param,
                outbound.iter().map(|encoded| {
                    V::PrepareShare::get_decoded_with_param(&states[0], encoded)
                        .expect("failed to decode prep share")
                }),
            )?
            .get_encoded()
            .unwrap();

        let mut out_shares = Vec::new();
        loop {
            let mut outbound = Vec::new();
            for state in states.iter_mut() {
                match vdaf.prepare_next(
                    ctx,
                    state.clone(),
                    V::PrepareMessage::get_decoded_with_param(state, &inbound)
                        .expect("failed to decode prep message"),
                )? {
                    PrepareTransition::Continue(new_state, msg) => {
                        outbound.push(msg.get_encoded().unwrap());
                        *state = new_state
                    }
                    PrepareTransition::Finish(out_share) => {
                        out_shares.push(out_share);
                    }
                }
            }

            if outbound.len() == vdaf.num_aggregators() {
                // Another round is required before output shares are computed.
                inbound = vdaf
                    .prepare_shares_to_prepare_message(
                        ctx,
                        agg_param,
                        outbound.iter().map(|encoded| {
                            V::PrepareShare::get_decoded_with_param(&states[0], encoded)
                                .expect("failed to decode prep share")
                        }),
                    )?
                    .get_encoded()
                    .unwrap();
            } else if outbound.is_empty() {
                // Each Aggregator recovered an output share.
                break;
            } else {
                panic!("Aggregators did not finish the prepare phase at the same time");
            }
        }

        Ok(out_shares)
    }

    /// VDAF sharding with a fixed randomness value.
    pub trait TestVectorClient<const NONCE_SIZE: usize>: Client<NONCE_SIZE> {
        /// Shards a measurement using a fixed randomness value.
        fn shard_with_random(
            &self,
            ctx: &[u8],
            measurement: &Self::Measurement,
            nonce: &[u8; NONCE_SIZE],
            random: &[u8],
        ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError>;
    }

    impl<T, P, const SEED_SIZE: usize> TestVectorClient<16> for Prio3<T, P, SEED_SIZE>
    where
        T: Type,
        P: Xof<SEED_SIZE>,
    {
        fn shard_with_random(
            &self,
            ctx: &[u8],
            measurement: &Self::Measurement,
            nonce: &[u8; 16],
            random: &[u8],
        ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError> {
            Prio3::shard_with_random(self, ctx, measurement, nonce, random)
        }
    }

    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    impl<P, const SEED_SIZE: usize> TestVectorClient<16> for Poplar1<P, SEED_SIZE>
    where
        P: Xof<SEED_SIZE>,
    {
        fn shard_with_random(
            &self,
            ctx: &[u8],
            measurement: &Self::Measurement,
            nonce: &[u8; 16],
            random: &[u8],
        ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError> {
            let idpf_random = [
                random[..16].try_into().unwrap(),
                random[16..32].try_into().unwrap(),
            ];
            let poplar_random = [
                random[32..32 + SEED_SIZE].try_into().unwrap(),
                random[32 + SEED_SIZE..32 + SEED_SIZE * 2]
                    .try_into()
                    .unwrap(),
                random[32 + SEED_SIZE * 2..].try_into().unwrap(),
            ];
            Poplar1::shard_with_random(self, ctx, measurement, nonce, &idpf_random, &poplar_random)
        }
    }

    /// Trait for per-VDAF deserialization of test vector fields.
    pub trait TestVectorVdaf: Vdaf {
        /// Deserialize VDAF parameters and construct a VDAF instance from them.
        fn new(shares: u8, parameters: &HashMap<String, Value>) -> Self;
        /// Deserialize a measurement.
        fn deserialize_measurement(measurement: &Value) -> Self::Measurement;
        /// Deserialize an aggregate result.
        fn deserialize_aggregate_result(aggregate_result: &Value) -> Self::AggregateResult;
    }

    /// Test vector for a VDAF.
    #[derive(Debug, Deserialize)]
    pub struct TestVector {
        /// List of operations to perform.
        pub operations: Vec<TestVectorOperation>,
        /// Number of report shares, i.e. number of aggregators.
        pub shares: u8,
        /// Context string.
        pub ctx: HexEncoded,
        /// VDAF verification key.
        pub verify_key: HexEncoded,
        /// Aggregation parameter.
        pub agg_param: HexEncoded,
        /// Per-report preparation information.
        pub prep: Vec<PreparationTestVector>,
        /// Aggregate shares.
        pub agg_shares: Vec<HexEncoded>,
        /// Aggregate result.
        pub agg_result: Value,
        /// VDAF parameters.
        #[serde(flatten)]
        pub other_params: HashMap<String, Value>,
    }

    /// Wrapper struct for hex-encoded byte strings in test vectors.
    #[derive(Debug, Deserialize)]
    pub struct HexEncoded(#[serde(with = "hex")] pub Vec<u8>);

    impl AsRef<[u8]> for HexEncoded {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }

    /// Describes an operation represented within a test vector.
    #[derive(Debug, Deserialize)]
    pub struct TestVectorOperation {
        /// The type of the operation.
        pub operation: OperationKind,
        /// The round number associated with the operation, if applicable.
        pub round: Option<usize>,
        /// The aggregator that performs the operation, if applicable.
        pub aggregator_id: Option<usize>,
        /// The index of the report associated with the operation, if applicable.
        pub report_index: Option<usize>,
        /// Whether the operation completes successfully.
        pub success: bool,
    }

    /// The different kinds of operations represented within test vectors.
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum OperationKind {
        /// Shard a report.
        Shard,
        /// Initial step of preparation.
        PrepInit,
        /// Combine prepare shares into a prepare message.
        PrepSharesToPrep,
        /// Subsequent steps of preparation.
        PrepNext,
        /// Aggregate output shares into an aggregate share.
        Aggregate,
        /// Unshard aggregate shares into an aggregate result.
        Unshard,
    }

    /// Per-report preparation information in a test vector.
    #[derive(Debug, Deserialize)]
    pub struct PreparationTestVector {
        /// The original measurement.
        pub measurement: Value,
        /// The nonce associated with this report.
        pub nonce: HexEncoded,
        /// Sharding randomness.
        pub rand: HexEncoded,
        /// Public share.
        pub public_share: HexEncoded,
        /// Input shares.
        ///
        /// This is indexed by aggregator ID.
        pub input_shares: Vec<HexEncoded>,
        /// Prepare shares.
        ///
        /// This is indexed first by round, then by aggregator ID.
        pub prep_shares: Vec<Vec<HexEncoded>>,
        /// Prepare messages.
        ///
        /// This is indexed by round.
        pub prep_messages: Vec<HexEncoded>,
        /// Output shares.
        ///
        /// This is indexed by aggregator ID.
        pub out_shares: Vec<HexEncoded>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct StateKey {
        round: usize,
        aggregator_id: usize,
        report_index: usize,
    }

    /// Check a VDAF implementation against a deserialized test vector file.
    pub fn check_test_vector<V, const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>(
        test_vector: &TestVector,
    ) where
        V: TestVectorClient<16>
            + TestVectorVdaf
            + Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>
            + Collector,
        V::PublicShare: PartialEq,
        V::AggregateResult: PartialEq,
        V::InputShare: PartialEq,
        V::PrepareShare: PartialEq,
        V::OutputShare: PartialEq,
        V::AggregateShare: PartialEq,
    {
        check_test_vector_custom_constructor(test_vector, V::new);
    }

    /// Check a VDAF implementation against a deserialized test vector file.
    ///
    /// This version allows overriding the constructor used for the VDAF.
    pub fn check_test_vector_custom_constructor<
        V,
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
    >(
        test_vector: &TestVector,
        constructor: impl Fn(u8, &HashMap<String, Value>) -> V,
    ) where
        V: TestVectorClient<16>
            + TestVectorVdaf
            + Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>
            + Collector,
        V::PublicShare: PartialEq,
        V::AggregateResult: PartialEq,
        V::InputShare: PartialEq,
        V::PrepareShare: PartialEq,
        V::OutputShare: PartialEq,
        V::AggregateShare: PartialEq,
    {
        let vdaf = constructor(test_vector.shares, &test_vector.other_params);
        let agg_param = V::AggregationParam::get_decoded(test_vector.agg_param.as_ref()).unwrap();
        let mut prepare_states = HashMap::new();
        for operation in test_vector.operations.iter() {
            match operation.operation {
                OperationKind::Shard => {
                    assert!(operation.success);
                    let report_index = operation.report_index.unwrap();
                    let preparation_test_vector = &test_vector.prep[report_index];

                    let input_shares = preparation_test_vector
                        .input_shares
                        .iter()
                        .map(|encoded| encoded.as_ref())
                        .collect::<Vec<_>>();

                    check_shard_operation(
                        &vdaf,
                        test_vector.ctx.as_ref(),
                        &preparation_test_vector.measurement,
                        preparation_test_vector.nonce.as_ref(),
                        preparation_test_vector.rand.as_ref(),
                        preparation_test_vector.public_share.as_ref(),
                        &input_shares,
                    );
                }
                OperationKind::PrepInit => {
                    assert!(operation.success);
                    let aggregator_id = operation.aggregator_id.unwrap();
                    let report_index = operation.report_index.unwrap();
                    let preparation_test_vector = &test_vector.prep[report_index];

                    let state = check_prep_init_operation(
                        &vdaf,
                        test_vector.verify_key.as_ref(),
                        test_vector.ctx.as_ref(),
                        aggregator_id,
                        &agg_param,
                        preparation_test_vector.nonce.as_ref(),
                        preparation_test_vector.public_share.as_ref(),
                        preparation_test_vector.input_shares[aggregator_id].as_ref(),
                        preparation_test_vector.prep_shares[0][aggregator_id].as_ref(),
                    );
                    prepare_states.insert(
                        StateKey {
                            round: 0,
                            aggregator_id,
                            report_index,
                        },
                        state,
                    );
                }
                OperationKind::PrepSharesToPrep => {
                    let round = operation.round.unwrap();
                    let report_index = operation.report_index.unwrap();
                    let preparation_test_vector = &test_vector.prep[report_index];

                    let prep_shares = preparation_test_vector.prep_shares[round]
                        .iter()
                        .map(|share| share.as_ref())
                        .collect::<Vec<_>>();
                    for aggregator_id in 0..vdaf.num_aggregators() {
                        let prep_state = &prepare_states[&StateKey {
                            round,
                            aggregator_id,
                            report_index,
                        }];

                        if operation.success {
                            check_prep_shares_to_prep_operation_success(
                                &vdaf,
                                test_vector.ctx.as_ref(),
                                &agg_param,
                                prep_state,
                                &prep_shares,
                                preparation_test_vector.prep_messages[round].as_ref(),
                            );
                        } else {
                            check_prep_shares_to_prep_operation_failure(
                                &vdaf,
                                test_vector.ctx.as_ref(),
                                &agg_param,
                                prep_state,
                                &prep_shares,
                            );
                        }
                    }
                }
                OperationKind::PrepNext => {
                    let round = operation.round.unwrap();
                    let aggregator_id = operation.aggregator_id.unwrap();
                    let report_index = operation.report_index.unwrap();
                    let preparation_test_vector = &test_vector.prep[report_index];

                    let prep_state = &prepare_states[&StateKey {
                        round: round - 1,
                        aggregator_id,
                        report_index,
                    }];

                    if operation.success {
                        let prep_share_opt = preparation_test_vector
                            .prep_shares
                            .get(round)
                            .map(|list| list[aggregator_id].as_ref());
                        let out_share_opt = prep_share_opt
                            .is_none()
                            .then(|| preparation_test_vector.out_shares[aggregator_id].as_ref());

                        let state_opt = check_prep_next_operation_success(
                            &vdaf,
                            test_vector.ctx.as_ref(),
                            &agg_param,
                            prep_state,
                            preparation_test_vector.prep_messages[round - 1].as_ref(),
                            prep_share_opt,
                            out_share_opt,
                        );

                        if let Some(state) = state_opt {
                            prepare_states.insert(
                                StateKey {
                                    round,
                                    aggregator_id,
                                    report_index,
                                },
                                state,
                            );
                        }
                    } else {
                        check_prep_next_operation_failure(
                            &vdaf,
                            test_vector.ctx.as_ref(),
                            prep_state,
                            preparation_test_vector.prep_messages[round - 1].as_ref(),
                        );
                    }
                }
                OperationKind::Aggregate => {
                    assert!(operation.success);
                    let aggregator_id = operation.aggregator_id.unwrap();

                    let output_shares = test_vector
                        .prep
                        .iter()
                        .map(|preparation_test_vector| {
                            preparation_test_vector.out_shares[aggregator_id].as_ref()
                        })
                        .collect::<Vec<_>>();

                    check_aggregate_operation(
                        &vdaf,
                        &agg_param,
                        &output_shares,
                        test_vector.agg_shares[aggregator_id].as_ref(),
                    );
                }
                OperationKind::Unshard => {
                    assert!(operation.success);

                    let agg_shares = test_vector
                        .agg_shares
                        .iter()
                        .map(|share| share.as_ref())
                        .collect::<Vec<_>>();

                    check_unshard_operation(
                        &vdaf,
                        &agg_param,
                        &agg_shares,
                        test_vector.prep.len(),
                        &test_vector.agg_result,
                    );
                }
            }
        }
    }

    fn check_shard_operation<V>(
        vdaf: &V,
        ctx: &[u8],
        measurement: &Value,
        nonce: &[u8],
        random: &[u8],
        expected_public_share: &[u8],
        expected_input_shares: &[&[u8]],
    ) where
        V: TestVectorClient<16> + TestVectorVdaf,
        V::PublicShare: PartialEq,
        V::InputShare: PartialEq,
    {
        let measurement = V::deserialize_measurement(measurement);
        let (public_share, input_shares) = vdaf
            .shard_with_random(ctx, &measurement, nonce.try_into().unwrap(), random)
            .unwrap();

        assert_eq!(public_share.get_encoded().unwrap(), expected_public_share);
        assert_eq!(
            public_share,
            V::PublicShare::get_decoded_with_param(vdaf, expected_public_share).unwrap()
        );

        assert_eq!(input_shares.len(), expected_input_shares.len());
        for (agg_id, (input_share, expected_input_share)) in input_shares
            .iter()
            .zip(expected_input_shares.iter())
            .enumerate()
        {
            assert_eq!(input_share.get_encoded().unwrap(), *expected_input_share);
            assert_eq!(
                input_share,
                &V::InputShare::get_decoded_with_param(&(vdaf, agg_id), expected_input_share)
                    .unwrap()
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_prep_init_operation<V, const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>(
        vdaf: &V,
        verify_key: &[u8],
        ctx: &[u8],
        agg_id: usize,
        agg_param: &V::AggregationParam,
        nonce: &[u8],
        public_share: &[u8],
        input_share: &[u8],
        expected_prep_share: &[u8],
    ) -> V::PrepareState
    where
        V: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
        V::PrepareShare: PartialEq,
    {
        let verify_key = verify_key.try_into().unwrap();
        let public_share = V::PublicShare::get_decoded_with_param(vdaf, public_share).unwrap();
        let input_share =
            V::InputShare::get_decoded_with_param(&(vdaf, agg_id), input_share).unwrap();
        let (prepare_state, prepare_share) = vdaf
            .prepare_init(
                verify_key,
                ctx,
                agg_id,
                agg_param,
                &nonce.try_into().unwrap(),
                &public_share,
                &input_share,
            )
            .unwrap();

        assert_eq!(prepare_share.get_encoded().unwrap(), expected_prep_share);
        assert_eq!(
            prepare_share,
            V::PrepareShare::get_decoded_with_param(&prepare_state, expected_prep_share).unwrap()
        );

        prepare_state
    }

    fn check_prep_shares_to_prep_operation_success<
        V,
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
    >(
        vdaf: &V,
        ctx: &[u8],
        agg_param: &V::AggregationParam,
        prep_state: &V::PrepareState,
        prep_shares: &[&[u8]],
        expected_prep_message: &[u8],
    ) where
        V: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    {
        let prep_shares = prep_shares
            .iter()
            .map(|bytes| V::PrepareShare::get_decoded_with_param(prep_state, bytes).unwrap())
            .collect::<Vec<_>>();
        let prep_message = vdaf
            .prepare_shares_to_prepare_message(ctx, agg_param, prep_shares)
            .unwrap();

        assert_eq!(prep_message.get_encoded().unwrap(), expected_prep_message);
        assert_eq!(
            prep_message,
            V::PrepareMessage::get_decoded_with_param(prep_state, expected_prep_message).unwrap()
        );
    }

    fn check_prep_shares_to_prep_operation_failure<
        V,
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
    >(
        vdaf: &V,
        ctx: &[u8],
        agg_param: &V::AggregationParam,
        prep_state: &V::PrepareState,
        prep_shares: &[&[u8]],
    ) where
        V: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    {
        let prep_shares = prep_shares
            .iter()
            .map(|bytes| V::PrepareShare::get_decoded_with_param(prep_state, bytes).unwrap())
            .collect::<Vec<_>>();
        vdaf.prepare_shares_to_prepare_message(ctx, agg_param, prep_shares)
            .unwrap_err();
    }

    fn check_prep_next_operation_success<V, const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>(
        vdaf: &V,
        ctx: &[u8],
        agg_param: &V::AggregationParam,
        prep_state: &V::PrepareState,
        prep_message: &[u8],
        expected_prep_share: Option<&[u8]>,
        expected_out_share: Option<&[u8]>,
    ) -> Option<V::PrepareState>
    where
        V: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
        V::PrepareShare: PartialEq,
        V::OutputShare: PartialEq,
    {
        let prep_message =
            V::PrepareMessage::get_decoded_with_param(prep_state, prep_message).unwrap();
        let transition = vdaf
            .prepare_next(ctx, prep_state.clone(), prep_message)
            .unwrap();
        match transition {
            PrepareTransition::Continue(prep_state, prep_share) => {
                assert_eq!(
                    prep_share.get_encoded().unwrap(),
                    expected_prep_share.unwrap()
                );
                assert_eq!(
                    prep_share,
                    V::PrepareShare::get_decoded_with_param(
                        &prep_state,
                        expected_prep_share.unwrap()
                    )
                    .unwrap()
                );

                Some(prep_state)
            }
            PrepareTransition::Finish(out_share) => {
                assert_eq!(
                    out_share.get_encoded().unwrap(),
                    expected_out_share.unwrap()
                );
                assert_eq!(
                    out_share,
                    V::OutputShare::get_decoded_with_param(
                        &(vdaf, agg_param),
                        expected_out_share.unwrap()
                    )
                    .unwrap()
                );

                None
            }
        }
    }

    fn check_prep_next_operation_failure<V, const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>(
        vdaf: &V,
        ctx: &[u8],
        prep_state: &V::PrepareState,
        prep_message: &[u8],
    ) where
        V: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    {
        let prep_message =
            V::PrepareMessage::get_decoded_with_param(prep_state, prep_message).unwrap();

        vdaf.prepare_next(ctx, prep_state.clone(), prep_message)
            .unwrap_err();
    }

    fn check_aggregate_operation<V, const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>(
        vdaf: &V,
        agg_param: &V::AggregationParam,
        output_shares: &[&[u8]],
        expected_agg_share: &[u8],
    ) where
        V: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
        V::AggregateShare: PartialEq,
    {
        let output_shares = output_shares
            .iter()
            .map(|share| V::OutputShare::get_decoded_with_param(&(vdaf, agg_param), share).unwrap())
            .collect::<Vec<_>>();
        let agg_share = vdaf.aggregate(agg_param, output_shares).unwrap();

        assert_eq!(agg_share.get_encoded().unwrap(), expected_agg_share);
        assert_eq!(
            agg_share,
            V::AggregateShare::get_decoded_with_param(&(vdaf, agg_param), expected_agg_share)
                .unwrap()
        );
    }

    fn check_unshard_operation<V>(
        vdaf: &V,
        agg_param: &V::AggregationParam,
        agg_shares: &[&[u8]],
        num_measurements: usize,
        expected_agg_result: &Value,
    ) where
        V: Collector + TestVectorVdaf,
        V::AggregateResult: PartialEq,
    {
        let agg_shares = agg_shares
            .iter()
            .map(|share| {
                V::AggregateShare::get_decoded_with_param(&(vdaf, agg_param), share).unwrap()
            })
            .collect::<Vec<_>>();
        let expected_agg_result = V::deserialize_aggregate_result(expected_agg_result);

        let agg_result = vdaf
            .unshard(agg_param, agg_shares, num_measurements)
            .unwrap();

        assert_eq!(agg_result, expected_agg_result);
    }
}

#[cfg(test)]
fn fieldvec_roundtrip_test<F, V, T>(vdaf: &V, agg_param: &V::AggregationParam, length: usize)
where
    F: FieldElement,
    V: Vdaf,
    T: Encode,
    for<'a> T: ParameterizedDecode<(&'a V, &'a V::AggregationParam)>,
{
    // Generate an arbitrary vector of field elements.
    let vec = F::random_vector(length);

    // Serialize the field element vector into a vector of bytes.
    let mut bytes = Vec::with_capacity(vec.len() * F::ENCODED_SIZE);
    encode_fieldvec(&vec, &mut bytes).unwrap();

    // Deserialize the type of interest from those bytes.
    let value = T::get_decoded_with_param(&(vdaf, agg_param), &bytes).unwrap();

    // Round-trip the value back to a vector of bytes.
    let encoded = value.get_encoded().unwrap();

    assert_eq!(encoded, bytes);
}

#[cfg(test)]
fn equality_comparison_test<T>(values: &[T])
where
    T: Debug + PartialEq,
{
    use std::ptr;

    // This function expects that every value passed in `values` is distinct, i.e. should not
    // compare as equal to any other element. We test both (i, j) and (j, i) to gain confidence that
    // equality implementations are symmetric.
    for (i, i_val) in values.iter().enumerate() {
        for (j, j_val) in values.iter().enumerate() {
            if i == j {
                assert!(ptr::eq(i_val, j_val)); // sanity
                assert_eq!(
                    i_val, j_val,
                    "Expected element at index {i} to be equal to itself, but it was not"
                );
            } else {
                assert_ne!(
                    i_val, j_val,
                    "Expected elements at indices {i} & {j} to not be equal, but they were"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vdaf::{equality_comparison_test, xof::Seed, AggregateShare, OutputShare, Share};

    #[test]
    fn share_equality_test() {
        equality_comparison_test(&[
            Share::Leader(Vec::from([1, 2, 3])),
            Share::Leader(Vec::from([3, 2, 1])),
            Share::Helper(Seed([1, 2, 3])),
            Share::Helper(Seed([3, 2, 1])),
        ])
    }

    #[test]
    fn output_share_equality_test() {
        equality_comparison_test(&[
            OutputShare(Vec::from([1, 2, 3])),
            OutputShare(Vec::from([3, 2, 1])),
        ])
    }

    #[test]
    fn aggregate_share_equality_test() {
        equality_comparison_test(&[
            AggregateShare(Vec::from([1, 2, 3])),
            AggregateShare(Vec::from([3, 2, 1])),
        ])
    }
}

#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod dummy;
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
pub mod mastic;
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "crypto-dependencies", feature = "experimental")))
)]
pub mod poplar1;
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "crypto-dependencies", feature = "experimental")))
)]
pub mod prio2;
pub mod prio3;
#[cfg(any(test, feature = "test-util"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod prio3_test;
pub mod xof;
