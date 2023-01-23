// SPDX-License-Identifier: MPL-2.0

//! Work-in-progress implementation of Poplar1 as specified in [[draft-irtf-cfrg-vdaf-03]].
//!
//! [draft-irtf-cfrg-vdaf-03]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/

use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{Field255, Field64},
    idpf::{IdpfInput, IdpfPublicShare},
    vdaf::{
        prg::{Prg, PrgAes128},
        Aggregatable, Aggregator, Client, Collector, PrepareTransition, Vdaf, VdafError,
    },
};
use std::{fmt::Debug, io::Cursor, marker::PhantomData};

/// Poplar1 with [`PrgAes128`].
pub type Poplar1Aes128 = Poplar1<PrgAes128, 16>;

impl Poplar1Aes128 {
    /// Create an instance of [`Poplar1Aes128`]. The caller provides the bit length of each
    /// measurement (`BITS` as defined in the [[draft-irtf-cfrg-vdaf-03]]).
    ///
    /// [draft-irtf-cfrg-vdaf-03]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/
    pub fn new_aes128(bits: usize) -> Self {
        Self {
            bits,
            phantom: PhantomData,
        }
    }
}

/// The poplar1 VDAF.
#[derive(Debug)]
pub struct Poplar1<P, const L: usize> {
    bits: usize,
    phantom: PhantomData<P>,
}

impl<P, const L: usize> Clone for Poplar1<P, L> {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits,
            phantom: PhantomData,
        }
    }
}

/// Poplar1 public share. This is comprised of the correction words generated for the IDPF.
pub type Poplar1PublicShare<const L: usize> = IdpfPublicShare<Field64, Field255, L, 2>;

impl<P, const L: usize> ParameterizedDecode<Poplar1<P, L>> for Poplar1PublicShare<L> {
    fn decode_with_param(
        _poplar1: &Poplar1<P, L>,
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        todo!()
    }
}

/// Poplar1 input share. Comprised of an IDPF key share and the correlated randomness used to
/// compute the sketch during preparation.
#[derive(Debug, Clone)]
pub struct Poplar1InputShare<const L: usize> {}

impl<const L: usize> Encode for Poplar1InputShare<L> {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        todo!()
    }
}

impl<'a, P, const L: usize> ParameterizedDecode<(&'a Poplar1<P, L>, usize)>
    for Poplar1InputShare<L>
{
    fn decode_with_param(
        (_poplar1, _agg_id): &(&'a Poplar1<P, L>, usize),
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        todo!()
    }
}

/// Poplar1 preparation state.
#[derive(Clone, Debug)]
pub struct Poplar1PrepareState();

/// Poplar1 preparation message.
#[derive(Clone, Debug)]
pub struct Poplar1PrepareMessage();

impl Encode for Poplar1PrepareMessage {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        todo!()
    }
}

impl ParameterizedDecode<Poplar1PrepareState> for Poplar1PrepareMessage {
    fn decode_with_param(
        _state: &Poplar1PrepareState,
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        todo!()
    }
}

/// A vector of field elements transmitted whilst evaluating Poplar1.
#[derive(Clone, Debug)]
pub enum Poplar1FieldVec {
    /// Field type for inner nodes of the IDPF tree.
    Inner(Vec<Field64>),

    /// Field type for leaf nodes of the IDPF tree.
    Leaf(Vec<Field255>),
}

impl Encode for Poplar1FieldVec {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        todo!()
    }
}

impl<'a, P: Prg<L>, const L: usize>
    ParameterizedDecode<(&'a Poplar1<P, L>, &'a Poplar1AggregationParam)> for Poplar1FieldVec
{
    fn decode_with_param(
        (_poplar1, _agg_param): &(&'a Poplar1<P, L>, &'a Poplar1AggregationParam),
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        todo!()
    }
}

impl ParameterizedDecode<Poplar1PrepareState> for Poplar1FieldVec {
    fn decode_with_param(
        _state: &Poplar1PrepareState,
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        todo!()
    }
}

impl Aggregatable for Poplar1FieldVec {
    type OutputShare = Self;

    fn merge(&mut self, _agg_share: &Self) -> Result<(), VdafError> {
        todo!()
    }

    fn accumulate(&mut self, _output_share: &Self) -> Result<(), VdafError> {
        todo!()
    }
}

/// Poplar1 aggregation parameter. This includes an indication of what level of the IDPF tree is
/// being evaluated and the set of prefixes to evaluate at that level.
//
// TODO(cjpatton) spec: Make sure repeated prefixes are disallowed. To make this check easier,
// consider requring the prefixes to be in lexicographic order.
#[derive(Clone, Debug)]
pub struct Poplar1AggregationParam {}

impl Encode for Poplar1AggregationParam {
    fn encode(&self, _bytes: &mut Vec<u8>) {
        todo!()
    }
}

impl Decode for Poplar1AggregationParam {
    fn decode(_bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        todo!()
    }
}

impl<P: Prg<L>, const L: usize> Vdaf for Poplar1<P, L> {
    const ID: u32 = 0x00001000;
    type Measurement = IdpfInput;
    type AggregateResult = Vec<u64>;
    type AggregationParam = Poplar1AggregationParam;
    type PublicShare = Poplar1PublicShare<L>;
    type InputShare = Poplar1InputShare<L>;
    type OutputShare = Poplar1FieldVec;
    type AggregateShare = Poplar1FieldVec;

    fn num_aggregators(&self) -> usize {
        2
    }
}

impl<P: Prg<L>, const L: usize> Client for Poplar1<P, L> {
    fn shard(
        &self,
        _input: &IdpfInput,
    ) -> Result<(Self::PublicShare, Vec<Poplar1InputShare<L>>), VdafError> {
        todo!()
    }
}

impl<P: Prg<L>, const L: usize> Aggregator<L> for Poplar1<P, L> {
    type PrepareState = Poplar1PrepareState;
    type PrepareShare = Poplar1FieldVec;
    type PrepareMessage = Poplar1PrepareMessage;

    #[allow(clippy::type_complexity)]
    fn prepare_init(
        &self,
        _verify_key: &[u8; L],
        _agg_id: usize,
        _agg_param: &Poplar1AggregationParam,
        _nonce: &[u8],
        _public_share: &Poplar1PublicShare<L>,
        _input_share: &Poplar1InputShare<L>,
    ) -> Result<(Poplar1PrepareState, Poplar1FieldVec), VdafError> {
        todo!()
    }

    fn prepare_preprocess<M: IntoIterator<Item = Poplar1FieldVec>>(
        &self,
        _inputs: M,
    ) -> Result<Poplar1PrepareMessage, VdafError> {
        todo!()
    }

    fn prepare_step(
        &self,
        _state: Poplar1PrepareState,
        _msg: Poplar1PrepareMessage,
    ) -> Result<PrepareTransition<Self, L>, VdafError> {
        todo!()
    }

    fn aggregate<M: IntoIterator<Item = Poplar1FieldVec>>(
        &self,
        _agg_param: &Poplar1AggregationParam,
        _output_shares: M,
    ) -> Result<Poplar1FieldVec, VdafError> {
        todo!()
    }
}

impl<P: Prg<L>, const L: usize> Collector for Poplar1<P, L> {
    fn unshard<M: IntoIterator<Item = Poplar1FieldVec>>(
        &self,
        _agg_param: &Poplar1AggregationParam,
        _agg_shares: M,
        _num_measurements: usize,
    ) -> Result<Vec<u64>, VdafError> {
        todo!()
    }
}
