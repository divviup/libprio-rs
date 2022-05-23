// SPDX-License-Identifier: MPL-2.0

//! This module defines an API for Verifiable Distributed Aggregation Functions (VDAFs) and
//! implements two constructions described in [[draft-irtf-cfrg-vdaf-00]].
//!
//! [draft-irtf-cfrg-vdaf-00]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/00/

use crate::codec::{CodecError, Decode, Encode, ParameterizedDecode};
use crate::field::{FieldElement, FieldError};
use crate::flp::FlpError;
use crate::prng::PrngError;
use crate::vdaf::prg::Seed;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::io::Cursor;

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

    /// failure when calling getrandom().
    #[error("getrandom: {0}")]
    GetRandom(#[from] getrandom::Error),
}

/// An additive share of a vector of field elements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Share<F, const L: usize> {
    /// An uncompressed share, typically sent to the leader.
    Leader(Vec<F>),

    /// A compressed share, typically sent to the helper.
    Helper(Seed<L>),
}

/// Parameters needed to decode a [`Share`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShareDecodingParameter<const L: usize> {
    Leader(usize),
    Helper,
}

impl<F: FieldElement, const L: usize> ParameterizedDecode<ShareDecodingParameter<L>>
    for Share<F, L>
{
    fn decode_with_param(
        decoding_parameter: &ShareDecodingParameter<L>,
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

impl<F: FieldElement, const L: usize> Encode for Share<F, L> {
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
}

/// The base trait for VDAF schemes. This trait is inherited by traits [`Client`], [`Aggregator`],
/// and [`Collector`], which define the roles of the various parties involved in the execution of
/// the VDAF.
// TODO(brandon): once GATs are stabilized [https://github.com/rust-lang/rust/issues/44265],
// state the "&AggregateShare must implement Into<Vec<u8>>" constraint in terms of a where clause
// on the associated type instead of a where clause on the trait.
pub trait Vdaf: Clone + Debug
where
    for<'a> &'a Self::AggregateShare: Into<Vec<u8>>,
{
    /// The type of Client measurement to be aggregated.
    type Measurement: Clone + Debug;

    /// The aggregate result of the VDAF execution.
    type AggregateResult: Clone + Debug;

    /// The aggregation parameter, used by the Aggregators to map their input shares to output
    /// shares.
    type AggregationParam: Clone + Debug + Decode + Encode;

    /// The public parameter used by Clients to shard their measurement into input shares.
    type PublicParam: Clone + Debug;

    /// A verification parameter, used by an Aggregator in the Prepare process to ensure that the
    /// Aggregators have recovered valid output shares.
    type VerifyParam: Clone + Debug;

    /// An input share sent by a Client.
    type InputShare: Clone + Debug + ParameterizedDecode<Self::VerifyParam> + Encode;

    /// An output share recovered from an input share by an Aggregator.
    type OutputShare: Clone + Debug;

    /// An Aggregator's share of the aggregate result.
    type AggregateShare: Aggregatable<OutputShare = Self::OutputShare> + for<'a> TryFrom<&'a [u8]>;

    /// Generates the long-lived parameters used by the Clients and Aggregators.
    fn setup(&self) -> Result<(Self::PublicParam, Vec<Self::VerifyParam>), VdafError>;

    /// The number of Aggregators. The Client generates as many input shares as there are
    /// Aggregators.
    fn num_aggregators(&self) -> usize;
}

/// The Client's role in the execution of a VDAF.
pub trait Client: Vdaf
where
    for<'a> &'a Self::AggregateShare: Into<Vec<u8>>,
{
    /// Shards a measurement into a sequence of input shares, one for each Aggregator.
    fn shard(
        &self,
        public_param: &Self::PublicParam,
        measurement: &Self::Measurement,
    ) -> Result<Vec<Self::InputShare>, VdafError>;
}

/// The Aggregator's role in the execution of a VDAF.
pub trait Aggregator: Vdaf
where
    for<'a> &'a Self::AggregateShare: Into<Vec<u8>>,
{
    /// State of the Aggregator during the Prepare process.
    type PrepareState: Clone + Debug;

    /// The type of messages exchanged among the Aggregators during the Prepare process.
    type PrepareMessage: Clone + Debug + ParameterizedDecode<Self::PrepareState> + Encode;

    /// Begins the Prepare process with the other Aggregators. The [`Self::PrepareState`] returned
    /// is passed to [`Aggregator::prepare_step`] to get this aggregator's first-round prepare
    /// message.
    fn prepare_init(
        &self,
        verify_param: &Self::VerifyParam,
        agg_param: &Self::AggregationParam,
        nonce: &[u8],
        input_share: &Self::InputShare,
    ) -> Result<Self::PrepareState, VdafError>;

    /// Preprocess a round of prepare messages into a single input to [`Aggregator::prepare_step`].
    fn prepare_preprocess<M: IntoIterator<Item = Self::PrepareMessage>>(
        &self,
        inputs: M,
    ) -> Result<Self::PrepareMessage, VdafError>;

    /// Compute the next state transition from the current state and the previous round of input
    /// messages. If this returns [`PrepareTransition::Continue`], then the returned
    /// [`Self::PrepareMessage`] should be combined with the other aggregator's `PrepareMessage`s from
    /// this round and passed into another call to this method. This continues until this method
    /// returns [`PrepareTransition::Finish`], at which point the returned output share may be
    /// aggregated. If the method returns [`PrepareTransition::Fail`], the aggregator should
    /// consider its input share invalid and not attempt to process it any further.
    fn prepare_step(
        &self,
        state: Self::PrepareState,
        input: Option<Self::PrepareMessage>,
    ) -> PrepareTransition<Self>;

    /// Aggregates a sequence of output shares into an aggregate share.
    fn aggregate<M: IntoIterator<Item = Self::OutputShare>>(
        &self,
        agg_param: &Self::AggregationParam,
        output_shares: M,
    ) -> Result<Self::AggregateShare, VdafError>;
}

/// The Collector's role in the execution of a VDAF.
pub trait Collector: Vdaf
where
    for<'a> &'a Self::AggregateShare: Into<Vec<u8>>,
{
    /// Combines aggregate shares into the aggregate result.
    fn unshard<M: IntoIterator<Item = Self::AggregateShare>>(
        &self,
        agg_param: &Self::AggregationParam,
        agg_shares: M,
    ) -> Result<Self::AggregateResult, VdafError>;
}

/// A state transition of an Aggregator during the Prepare process.
#[derive(Debug)]
pub enum PrepareTransition<V: Aggregator>
where
    for<'a> &'a V::AggregateShare: Into<Vec<u8>>,
{
    /// Continue processing.
    Continue(V::PrepareState, V::PrepareMessage),

    /// Finish processing and return the output share.
    Finish(V::OutputShare),

    /// Fail and return an error.
    Fail(VdafError),
}

/// An aggregate share resulting from aggregating output shares together that
/// can merged with aggregate shares of the same type.
pub trait Aggregatable: Clone + Debug + From<Self::OutputShare> {
    /// Type of output shares that can be accumulated into an aggregate share.
    type OutputShare;

    /// Update an aggregate share by merging it with another (`agg_share`).
    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError>;

    /// Update an aggregate share by adding `output share`
    fn accumulate(&mut self, output_share: &Self::OutputShare) -> Result<(), VdafError>;
}

/// An output share comprised of a vector of `F` elements.
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

impl<F: FieldElement> TryFrom<&[u8]> for OutputShare<F> {
    type Error = FieldError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        fieldvec_try_from_bytes(bytes)
    }
}

impl<F: FieldElement> From<&OutputShare<F>> for Vec<u8> {
    fn from(output_share: &OutputShare<F>) -> Self {
        fieldvec_to_vec(&output_share.0)
    }
}

/// An aggregate share suitable for VDAFs whose output shares and aggregate
/// shares are vectors of `F` elements, and an output share needs no special
/// transformation to be merged into an aggregate share.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateShare<F>(Vec<F>);

impl<F> AsRef<[F]> for AggregateShare<F> {
    fn as_ref(&self) -> &[F] {
        &self.0
    }
}

impl<F> From<OutputShare<F>> for AggregateShare<F> {
    fn from(other: OutputShare<F>) -> Self {
        Self(other.0)
    }
}

impl<F> From<Vec<F>> for AggregateShare<F> {
    fn from(other: Vec<F>) -> Self {
        Self(other)
    }
}

impl<F: FieldElement> Aggregatable for AggregateShare<F> {
    type OutputShare = OutputShare<F>;

    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError> {
        self.sum(agg_share.as_ref())
    }

    fn accumulate(&mut self, output_share: &Self::OutputShare) -> Result<(), VdafError> {
        // For prio3 and poplar1, no conversion is needed between output shares and aggregation
        // shares.
        self.sum(output_share.as_ref())
    }
}

impl<F: FieldElement> AggregateShare<F> {
    fn sum(&mut self, other: &[F]) -> Result<(), VdafError> {
        if self.0.len() != other.len() {
            return Err(VdafError::Uncategorized(format!(
                "cannot sum shares of different lengths (left = {}, right = {}",
                self.0.len(),
                other.len()
            )));
        }

        for (x, y) in self.0.iter_mut().zip(other) {
            *x += *y;
        }

        Ok(())
    }
}

impl<F: FieldElement> TryFrom<&[u8]> for AggregateShare<F> {
    type Error = FieldError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        fieldvec_try_from_bytes(bytes)
    }
}

impl<F: FieldElement> From<&AggregateShare<F>> for Vec<u8> {
    fn from(aggregate_share: &AggregateShare<F>) -> Self {
        fieldvec_to_vec(&aggregate_share.0)
    }
}

/// fieldvec_try_from_bytes converts a slice of bytes to a type that is equivalent to a vector of
/// field elements.
#[inline(always)]
fn fieldvec_try_from_bytes<F: FieldElement, T: From<Vec<F>>>(
    bytes: &[u8],
) -> Result<T, FieldError> {
    F::byte_slice_into_vec(bytes).map(T::from)
}

/// fieldvec_to_vec converts a type that is equivalent to a vector of field elements into a vector
/// of bytes.
#[inline(always)]
fn fieldvec_to_vec<F: FieldElement, T: AsRef<[F]>>(val: T) -> Vec<u8> {
    F::slice_into_byte_vec(val.as_ref())
}

#[cfg(test)]
pub(crate) fn run_vdaf<V, M>(
    vdaf: &V,
    agg_param: &V::AggregationParam,
    measurements: M,
) -> Result<V::AggregateResult, VdafError>
where
    V: Client + Aggregator + Collector,
    for<'a> &'a V::AggregateShare: Into<Vec<u8>>,
    M: IntoIterator<Item = V::Measurement>,
{
    // NOTE Here we use the same nonce for each measurement for testing purposes. However, this is
    // not secure. In use, the Aggregators MUST ensure that nonces are unique for each measurement.
    let nonce = b"this is a nonce";

    let (public_param, verify_params) = vdaf.setup()?;

    let mut agg_shares: Vec<Option<V::AggregateShare>> = vec![None; vdaf.num_aggregators()];
    for measurement in measurements.into_iter() {
        let input_shares = vdaf.shard(&public_param, &measurement)?;
        let out_shares = run_vdaf_prepare(vdaf, &verify_params, agg_param, nonce, input_shares)?;
        for (out_share, agg_share) in out_shares.into_iter().zip(agg_shares.iter_mut()) {
            if let Some(ref mut inner) = agg_share {
                inner.merge(&out_share.into())?;
            } else {
                *agg_share = Some(out_share.into());
            }
        }
    }

    let res = vdaf.unshard(
        agg_param,
        agg_shares.into_iter().map(|option| option.unwrap()),
    )?;
    Ok(res)
}

#[cfg(test)]
pub(crate) fn run_vdaf_prepare<V, M>(
    vdaf: &V,
    verify_params: &[V::VerifyParam],
    agg_param: &V::AggregationParam,
    nonce: &[u8],
    input_shares: M,
) -> Result<Vec<V::OutputShare>, VdafError>
where
    V: Client + Aggregator + Collector,
    for<'a> &'a V::AggregateShare: Into<Vec<u8>>,
    M: IntoIterator<Item = V::InputShare>,
{
    let input_shares = input_shares
        .into_iter()
        .map(|input_share| input_share.get_encoded());

    let mut states = Vec::new();
    for (verify_param, input_share) in verify_params.iter().zip(input_shares) {
        let state = vdaf.prepare_init(
            verify_param,
            agg_param,
            nonce,
            &V::InputShare::get_decoded_with_param(verify_param, &input_share)
                .expect("failed to decode input share"),
        )?;
        states.push(state);
    }

    let mut inbound = None;
    let mut out_shares = Vec::new();
    loop {
        let mut outbound = Vec::new();
        for state in states.iter_mut() {
            match vdaf.prepare_step(state.clone(), inbound.clone()) {
                PrepareTransition::Continue(new_state, msg) => {
                    outbound.push(msg.get_encoded());
                    *state = new_state
                }
                PrepareTransition::Finish(out_share) => {
                    out_shares.push(out_share);
                }
                PrepareTransition::Fail(err) => {
                    return Err(err);
                }
            }
        }

        if outbound.len() == vdaf.num_aggregators() {
            // Another round is required before output shares are computed.
            inbound = Some(vdaf.prepare_preprocess(outbound.iter().map(|encoded| {
                V::PrepareMessage::get_decoded_with_param(&states[0], encoded)
                    .expect("failed to decode papare message")
            }))?);
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
mod tests {
    use super::{AggregateShare, OutputShare};
    use crate::field::{Field128, Field64, FieldElement};
    use itertools::iterate;
    use std::convert::TryFrom;
    use std::fmt::Debug;

    fn fieldvec_roundtrip_test<F, T>()
    where
        F: FieldElement,
        for<'a> T: Debug + PartialEq + From<Vec<F>> + TryFrom<&'a [u8]>,
        for<'a> <T as TryFrom<&'a [u8]>>::Error: Debug,
        for<'a> Vec<u8>: From<&'a T>,
    {
        // Generate a value based on an arbitrary vector of field elements.
        let g = F::generator();
        let want_value = T::from(iterate(F::one(), |&v| g * v).take(10).collect());

        // Round-trip the value through a byte-vector.
        let buf: Vec<u8> = (&want_value).into();
        let got_value = T::try_from(&buf).unwrap();

        assert_eq!(want_value, got_value);
    }

    #[test]
    fn roundtrip_output_share() {
        fieldvec_roundtrip_test::<Field64, OutputShare<Field64>>();
        fieldvec_roundtrip_test::<Field128, OutputShare<Field128>>();
    }

    #[test]
    fn roundtrip_aggregate_share() {
        fieldvec_roundtrip_test::<Field64, AggregateShare<Field64>>();
        fieldvec_roundtrip_test::<Field128, AggregateShare<Field128>>();
    }
}

pub mod poplar1;
pub mod prg;
pub mod prio3;
#[cfg(test)]
mod prio3_test;
