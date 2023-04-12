// SPDX-License-Identifier: MPL-2.0

//! Verifiable Distributed Aggregation Functions (VDAFs) as described in
//! [[draft-irtf-cfrg-vdaf-05]].
//!
//! [draft-irtf-cfrg-vdaf-05]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/05/

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use crate::idpf::IdpfError;
use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{encode_fieldvec, merge_vector, FieldElement, FieldError},
    flp::FlpError,
    prng::PrngError,
    vdaf::prg::Seed,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, io::Cursor};

/// A component of the domain-separation tag, used to bind the VDAF operations to the document
/// version. This will be revised with each draft with breaking changes.
pub(crate) const VERSION: u8 = 5;

/// Errors emitted by this module.
#[derive(Debug, thiserror::Error)]
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

    /// PRNG error.
    #[error("prng error: {0}")]
    Prng(#[from] PrngError),

    /// Failure when calling getrandom().
    #[error("getrandom: {0}")]
    GetRandom(#[from] getrandom::Error),

    /// IDPF error.
    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    #[error("idpf error: {0}")]
    Idpf(#[from] IdpfError),
}

/// An additive share of a vector of field elements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Share<F, const SEED_SIZE: usize> {
    /// An uncompressed share, typically sent to the leader.
    Leader(Vec<F>),

    /// A compressed share, typically sent to the helper.
    Helper(Seed<SEED_SIZE>),
}

impl<F: Clone, const SEED_SIZE: usize> Share<F, SEED_SIZE> {
    /// Truncate the Leader's share to the given length. If this is the Helper's share, then this
    /// method clones the input without modifying it.
    #[cfg(feature = "prio2")]
    pub(crate) fn truncated(&self, len: usize) -> Self {
        match self {
            Self::Leader(ref data) => Self::Leader(data[..len].to_vec()),
            Self::Helper(ref seed) => Self::Helper(seed.clone()),
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
    fn encode(&self, bytes: &mut Vec<u8>) {
        match self {
            Share::Leader(share_data) => {
                for x in share_data {
                    x.encode(bytes);
                }
            }
            Share::Helper(share_seed) => {
                share_seed.encode(bytes);
            }
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
    /// Algorithm identifier for this VDAF.
    const ID: u32;

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

    /// The number of Aggregators. The Client generates as many input shares as there are
    /// Aggregators.
    fn num_aggregators(&self) -> usize;

    /// Generate the customization string for this VDAF. The output is used for domain separation
    /// by the PRG.
    fn custom(usage: u16) -> [u8; 8] {
        let mut custom = [0_u8; 8];
        custom[0] = VERSION;
        custom[1] = 0; // algorithm class
        custom[2..6].copy_from_slice(&(Self::ID).to_be_bytes());
        custom[6..8].copy_from_slice(&usage.to_be_bytes());
        custom
    }
}

/// The Client's role in the execution of a VDAF.
pub trait Client<const NONCE_SIZE: usize>: Vdaf {
    /// Shards a measurement into a public share and a sequence of input shares, one for each
    /// Aggregator.
    fn shard(
        &self,
        measurement: &Self::Measurement,
        nonce: &[u8; NONCE_SIZE],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError>;
}

/// The Aggregator's role in the execution of a VDAF.
pub trait Aggregator<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>: Vdaf {
    /// State of the Aggregator during the Prepare process.
    type PrepareState: Clone + Debug;

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
    type PrepareMessage: Clone + Debug + ParameterizedDecode<Self::PrepareState> + Encode;

    /// Begins the Prepare process with the other Aggregators. The [`Self::PrepareState`] returned
    /// is passed to [`Aggregator::prepare_step`] to get this aggregator's first-round prepare
    /// message.
    fn prepare_init(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        agg_id: usize,
        agg_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::PrepareState, Self::PrepareShare), VdafError>;

    /// Preprocess a round of preparation shares into a single input to [`Aggregator::prepare_step`].
    fn prepare_preprocess<M: IntoIterator<Item = Self::PrepareShare>>(
        &self,
        inputs: M,
    ) -> Result<Self::PrepareMessage, VdafError>;

    /// Compute the next state transition from the current state and the previous round of input
    /// messages. If this returns [`PrepareTransition::Continue`], then the returned
    /// [`Self::PrepareShare`] should be combined with the other Aggregators' `PrepareShare`s from
    /// this round and passed into another call to this method. This continues until this method
    /// returns [`PrepareTransition::Finish`], at which point the returned output share may be
    /// aggregated. If the method returns an error, the aggregator should consider its input share
    /// invalid and not attempt to process it any further.
    fn prepare_step(
        &self,
        state: Self::PrepareState,
        input: Self::PrepareMessage,
    ) -> Result<PrepareTransition<Self, VERIFY_KEY_SIZE, NONCE_SIZE>, VdafError>;

    /// Aggregates a sequence of output shares into an aggregate share.
    fn aggregate<M: IntoIterator<Item = Self::OutputShare>>(
        &self,
        agg_param: &Self::AggregationParam,
        output_shares: M,
    ) -> Result<Self::AggregateShare, VdafError>;
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputShare<F>(Vec<F>);

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
    fn encode(&self, bytes: &mut Vec<u8>) {
        encode_fieldvec(&self.0, bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(F::ENCODED_SIZE * self.0.len())
    }
}

/// An aggregate share comprised of a vector of field elements.
///
/// This is suitable for VDAFs where both output shares and aggregate shares are vectors of field
/// elements, and output shares need no special transformation to be merged into an aggregate share.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateShare<F>(Vec<F>);

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
    fn encode(&self, bytes: &mut Vec<u8>) {
        encode_fieldvec(&self.0, bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(F::ENCODED_SIZE * self.0.len())
    }
}

#[cfg(test)]
pub(crate) fn run_vdaf<V, M, const SEED_SIZE: usize>(
    vdaf: &V,
    agg_param: &V::AggregationParam,
    measurements: M,
) -> Result<V::AggregateResult, VdafError>
where
    V: Client<16> + Aggregator<SEED_SIZE, 16> + Collector,
    M: IntoIterator<Item = V::Measurement>,
{
    use rand::prelude::*;
    let mut rng = thread_rng();
    let mut verify_key = [0; SEED_SIZE];
    rng.fill(&mut verify_key[..]);

    let mut agg_shares: Vec<Option<V::AggregateShare>> = vec![None; vdaf.num_aggregators()];
    let mut num_measurements: usize = 0;
    for measurement in measurements.into_iter() {
        num_measurements += 1;
        let nonce = rng.gen();
        let (public_share, input_shares) = vdaf.shard(&measurement, &nonce)?;
        let out_shares = run_vdaf_prepare(
            vdaf,
            &verify_key,
            agg_param,
            &nonce,
            public_share,
            input_shares,
        )?;
        for (out_share, agg_share) in out_shares.into_iter().zip(agg_shares.iter_mut()) {
            // Check serialization of output shares
            let encoded_out_share = out_share.get_encoded();
            let round_trip_out_share =
                V::OutputShare::get_decoded_with_param(&(vdaf, agg_param), &encoded_out_share)
                    .unwrap();
            assert_eq!(round_trip_out_share.get_encoded(), encoded_out_share);

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
        let encoded_agg_share = agg_share.as_ref().unwrap().get_encoded();
        let round_trip_agg_share =
            V::AggregateShare::get_decoded_with_param(&(vdaf, agg_param), &encoded_agg_share)
                .unwrap();
        assert_eq!(round_trip_agg_share.get_encoded(), encoded_agg_share);
    }

    let res = vdaf.unshard(
        agg_param,
        agg_shares.into_iter().map(|option| option.unwrap()),
        num_measurements,
    )?;
    Ok(res)
}

#[cfg(test)]
pub(crate) fn run_vdaf_prepare<V, M, const SEED_SIZE: usize>(
    vdaf: &V,
    verify_key: &[u8; SEED_SIZE],
    agg_param: &V::AggregationParam,
    nonce: &[u8; 16],
    public_share: V::PublicShare,
    input_shares: M,
) -> Result<Vec<V::OutputShare>, VdafError>
where
    V: Client<16> + Aggregator<SEED_SIZE, 16> + Collector,
    M: IntoIterator<Item = V::InputShare>,
{
    let input_shares = input_shares
        .into_iter()
        .map(|input_share| input_share.get_encoded());

    let mut states = Vec::new();
    let mut outbound = Vec::new();
    for (agg_id, input_share) in input_shares.enumerate() {
        let (state, msg) = vdaf.prepare_init(
            verify_key,
            agg_id,
            agg_param,
            nonce,
            &public_share,
            &V::InputShare::get_decoded_with_param(&(vdaf, agg_id), &input_share)
                .expect("failed to decode input share"),
        )?;
        states.push(state);
        outbound.push(msg.get_encoded());
    }

    let mut inbound = vdaf
        .prepare_preprocess(outbound.iter().map(|encoded| {
            V::PrepareShare::get_decoded_with_param(&states[0], encoded)
                .expect("failed to decode prep share")
        }))?
        .get_encoded();

    let mut out_shares = Vec::new();
    loop {
        let mut outbound = Vec::new();
        for state in states.iter_mut() {
            match vdaf.prepare_step(
                state.clone(),
                V::PrepareMessage::get_decoded_with_param(state, &inbound)
                    .expect("failed to decode prep message"),
            )? {
                PrepareTransition::Continue(new_state, msg) => {
                    outbound.push(msg.get_encoded());
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
                .prepare_preprocess(outbound.iter().map(|encoded| {
                    V::PrepareShare::get_decoded_with_param(&states[0], encoded)
                        .expect("failed to decode prep share")
                }))?
                .get_encoded();
        } else if outbound.is_empty() {
            // Each Aggregator recovered an output share.
            break;
        } else {
            panic!("Aggregators did not finish the prepare phase at the same time");
        }
    }

    Ok(out_shares)
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
    let g = F::one() + F::one();
    let vec: Vec<F> = itertools::iterate(F::one(), |&v| g * v)
        .take(length)
        .collect();

    // Serialize the field element vector into a vector of bytes.
    let mut bytes = Vec::with_capacity(vec.len() * F::ENCODED_SIZE);
    encode_fieldvec(&vec, &mut bytes);

    // Deserialize the type of interest from those bytes.
    let value = T::get_decoded_with_param(&(vdaf, agg_param), &bytes).unwrap();

    // Round-trip the value back to a vector of bytes.
    let encoded = value.get_encoded();

    assert_eq!(encoded, bytes);
}

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "crypto-dependencies", feature = "experimental")))
)]
pub mod poplar1;
pub mod prg;
#[cfg(feature = "prio2")]
#[cfg_attr(docsrs, doc(cfg(feature = "prio2")))]
pub mod prio2;
pub mod prio3;
#[cfg(test)]
mod prio3_test;
