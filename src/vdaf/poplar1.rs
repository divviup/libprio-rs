// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module
//! partially implements the core component of the Poplar protocol [[BBCG+21]]. Named for the
//! Poplar1 section of [[draft-irtf-cfrg-vdaf-03]], the specification of this VDAF is under active
//! development. Thus this code should be regarded as experimental and not compliant with any
//! existing speciication.
//!
//! TODO Make the input shares stateful so that applications can efficiently evaluate the IDPF over
//! multiple rounds. Question: Will this require API changes to [`crate::vdaf::Vdaf`]?
//!
//! [BBCG+21]: https://eprint.iacr.org/2021/017
//! [draft-irtf-cfrg-vdaf-03]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/

use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{Field255, Field64},
    vdaf::{
        prg::{Prg, Seed},
        Aggregatable, Aggregator, Client, Collector, PrepareTransition, Vdaf, VdafError,
    },
};
use std::{fmt::Debug, io::Cursor, marker::PhantomData};

/// An input for an IDPF.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IdpfInput {
    index: usize,
    level: usize,
}

/// The poplar1 VDAF.
#[derive(Debug)]
pub struct Poplar1<P, const L: usize> {
    input_length: usize,
    phantom: PhantomData<P>,
}

impl<P, const L: usize> Poplar1<P, L> {
    /// Create an instance of the Poplar1 VDAF. The caller provides a cipher suite used for deriving
    /// pseudorandom sequences of field elements, and an input length in bits, corresponding to
    /// `BITS` as defined in the [VDAF specification][1].
    ///
    /// [1]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/
    pub fn new(bits: usize) -> Self {
        Self {
            input_length: bits,
            phantom: PhantomData,
        }
    }
}

impl<P, const L: usize> Clone for Poplar1<P, L> {
    fn clone(&self) -> Self {
        Self::new(self.input_length)
    }
}
impl<P, const L: usize> Vdaf for Poplar1<P, L>
where
    P: Prg<L>,
{
    // TODO: This currently uses a codepoint reserved for testing purposes. Replace it with
    // 0x00001000 once the implementation is updated to match draft-irtf-cfrg-vdaf-03.
    const ID: u32 = 0xFFFF0000;
    type Measurement = IdpfInput;
    type AggregateResult = Poplar1AggregateResult;
    type AggregationParam = Poplar1AggregationParam;
    type PublicShare = (); // TODO: Replace this when the IDPF from [BBCGGI21] is implemented.
    type InputShare = Poplar1InputShare<L>;
    type OutputShare = Poplar1OutputShare;
    type AggregateShare = Poplar1AggregateShare;

    fn num_aggregators(&self) -> usize {
        2
    }
}

impl<P, const L: usize> Client for Poplar1<P, L>
where
    P: Prg<L>,
{
    fn shard(
        &self,
        _input: &IdpfInput,
    ) -> Result<(Self::PublicShare, Vec<Poplar1InputShare<L>>), VdafError> {
        unimplemented!()
    }
}

impl<P, const L: usize> Aggregator<L> for Poplar1<P, L>
where
    P: Prg<L>,
{
    type PrepareState = Poplar1PrepareState;
    type PrepareShare = Poplar1PrepareMessage;
    type PrepareMessage = Poplar1PrepareMessage;

    #[allow(clippy::type_complexity)]
    fn prepare_init(
        &self,
        _verify_key: &[u8; L],
        _agg_id: usize,
        _agg_param: &Poplar1AggregationParam,
        _nonce: &[u8],
        _public_share: &Self::PublicShare,
        _input_share: &Self::InputShare,
    ) -> Result<(Poplar1PrepareState, Poplar1PrepareMessage), VdafError> {
        unimplemented!()
    }

    fn prepare_preprocess<M: IntoIterator<Item = Poplar1PrepareMessage>>(
        &self,
        _inputs: M,
    ) -> Result<Poplar1PrepareMessage, VdafError> {
        unimplemented!()
    }

    fn prepare_step(
        &self,
        _state: Poplar1PrepareState,
        _msg: Poplar1PrepareMessage,
    ) -> Result<PrepareTransition<Self, L>, VdafError> {
        unimplemented!()
    }

    fn aggregate<M: IntoIterator<Item = Poplar1OutputShare>>(
        &self,
        _agg_param: &Poplar1AggregationParam,
        _output_shares: M,
    ) -> Result<Poplar1AggregateShare, VdafError> {
        unimplemented!()
    }
}

impl<P, const L: usize> Collector for Poplar1<P, L>
where
    P: Prg<L>,
{
    fn unshard<M: IntoIterator<Item = Poplar1AggregateShare>>(
        &self,
        _agg_param: &Poplar1AggregationParam,
        _agg_shares: M,
        _num_measurements: usize,
    ) -> Result<Poplar1AggregateResult, VdafError> {
        unimplemented!()
    }
}

/// An input share for the Poplar1 VDAF.
#[derive(Debug, Clone)]
pub struct Poplar1InputShare<const L: usize> {
    /// IDPF share of input.
    _idpf_key: Seed<L>,

    _correlation_seed: Seed<L>,

    _correlation_inner: Vec<[Field64; 2]>,

    _correlation_leaf: [Field255; 2],
}

impl<const L: usize> Encode for Poplar1InputShare<L> {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        // TODO: see encode_input_shares() in VDAF-03.
        unimplemented!()
    }
}

impl<'a, P, const L: usize> ParameterizedDecode<(&'a Poplar1<P, L>, usize)>
    for Poplar1InputShare<L>
{
    fn decode_with_param(
        (_poplar1, _agg_id): &(&Poplar1<P, L>, usize),
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        // TODO: see decode_input_share() in VDAF-03.
        unimplemented!()
    }
}

/// A public share for the Poplar1 VDAF.
#[derive(Debug, Clone)]
pub struct Poplar1PublicShare<const L: usize> {}

impl<const L: usize> Encode for Poplar1PublicShare<L> {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        unimplemented!()
    }
}

impl<P, const L: usize> ParameterizedDecode<(&Poplar1<P, L>, usize)> for Poplar1PublicShare<L> {
    fn decode_with_param(
        (_poplar1, _agg_id): &(&Poplar1<P, L>, usize),
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        unimplemented!()
    }
}

/// The aggregation parameter for the Poplar1 VDAF, a collection of prefixes at some IDPF
/// level.
#[derive(Debug, Clone)]
pub struct Poplar1AggregationParam {}

impl Encode for Poplar1AggregationParam {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        // TODO: see encode_agg_param() in VDAF-03.
        unimplemented!()
    }
}

impl Decode for Poplar1AggregationParam {
    fn decode(_bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        // TODO: inverse of encode_agg_param() in VDAF-03.
        unimplemented!()
    }
}

/// An aggregate result from the `poplar1` VDAF, a mapping from input prefixes to numbers, such
/// that all input prefixes are at the same IDPF tree level.
#[derive(Debug, Clone)]
pub struct Poplar1AggregateResult {}

/// A prepare message exchanged between Poplar1 aggregators.
#[derive(Clone, Debug)]
pub struct Poplar1PrepareMessage {}

impl Encode for Poplar1PrepareMessage {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        // TODO: encode a vector of field elements
        unimplemented!()
    }
}

impl ParameterizedDecode<Poplar1PrepareState> for Poplar1PrepareMessage {
    fn decode_with_param(
        _decoding_parameter: &Poplar1PrepareState,
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        // TODO: Determine the current level's field based on the decoding parameter, and decode
        // a vector of field elements.
        unimplemented!()
    }
}

/// The state of each Aggregator during the Prepare process.
#[derive(Clone, Debug)]
pub struct Poplar1PrepareState {
    /// State of the secure sketching protocol.
    _sketch: SketchState,

    /// The output share.
    _output_share: Poplar1OutputShare,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum SketchState {
    RoundOne,
    RoundTwo,
}

/// An output share for the Poplar1 VDAF.
#[derive(Debug, Clone)]
pub struct Poplar1OutputShare {}

impl Encode for Poplar1OutputShare {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        // TODO: Serialize the vector of field elements.
        unimplemented!()
    }
}

impl<P, const L: usize> ParameterizedDecode<(&Poplar1<P, L>, &Poplar1AggregationParam)>
    for Poplar1OutputShare
{
    fn decode_with_param(
        _decoding_parameter: &(&Poplar1<P, L>, &Poplar1AggregationParam),
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        unimplemented!()
    }
}

/// An aggregate share for the Poplar1 VDAF.
#[derive(Debug, Clone)]
pub struct Poplar1AggregateShare {}

impl From<Poplar1OutputShare> for Poplar1AggregateShare {
    fn from(_: Poplar1OutputShare) -> Self {
        // TODO: Both types are vectors of field elements, so this is a trivial conversion.
        unimplemented!()
    }
}

impl Aggregatable for Poplar1AggregateShare {
    type OutputShare = Poplar1OutputShare;

    fn merge(&mut self, _agg_share: &Self) -> Result<(), VdafError> {
        // TODO: vector addition
        unimplemented!()
    }

    fn accumulate(&mut self, _output_share: &Self::OutputShare) -> Result<(), VdafError> {
        // TODO: vector addition
        unimplemented!()
    }
}

impl Encode for Poplar1AggregateShare {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        // TODO: Serialize the vector of field elements.
        unimplemented!()
    }
}

impl<P, const L: usize> ParameterizedDecode<(&Poplar1<P, L>, &Poplar1AggregationParam)>
    for Poplar1AggregateShare
{
    fn decode_with_param(
        _decoding_parameter: &(&Poplar1<P, L>, &Poplar1AggregationParam),
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {}
