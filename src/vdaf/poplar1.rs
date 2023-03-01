// SPDX-License-Identifier: MPL-2.0

//! Work-in-progress implementation of Poplar1 as specified in [[draft-irtf-cfrg-vdaf-04]].
//!
//! [draft-irtf-cfrg-vdaf-04]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/04/

use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{merge_vector, Field255, Field64, FieldElement},
    idpf::{self, IdpfInput, IdpfOutputShare, IdpfPublicShare, IdpfValue, RingBufferCache},
    prng::Prng,
    vdaf::{
        prg::{CoinToss, Prg, PrgSha3, Seed, SeedStream},
        Aggregatable, Aggregator, Client, Collector, PrepareTransition, Vdaf, VdafError,
    },
};
use std::{
    convert::TryFrom,
    fmt::Debug,
    io::Cursor,
    marker::PhantomData,
    ops::{Add, AddAssign, Sub},
};
use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable, ConstantTimeEq};

const DST_SHARD_RANDOMNESS: u16 = 1;
const DST_CORR_INNER: u16 = 2;
const DST_CORR_LEAF: u16 = 3;
const DST_VERIFY_RANDOMNESS: u16 = 4;

impl<P, const L: usize> Poplar1<P, L> {
    /// Create an instance of [`Poplar1`]. The caller provides the bit length of each
    /// measurement (`BITS` as defined in the [[draft-irtf-cfrg-vdaf-04]]).
    ///
    /// [draft-irtf-cfrg-vdaf-04]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/04/
    pub fn new(bits: usize) -> Self {
        Self {
            bits,
            phantom: PhantomData,
        }
    }
}

impl Poplar1<PrgSha3, 16> {
    /// Create an instance of [`Poplar1`] using [`PrgSha3`]. The caller provides the bit length of
    /// each measurement (`BITS` as defined in the [[draft-irtf-cfrg-vdaf-04]]).
    ///
    /// [draft-irtf-cfrg-vdaf-04]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/04/
    pub fn new_sha3(bits: usize) -> Self {
        Poplar1::new(bits)
    }
}

/// The Poplar1 VDAF.
#[derive(Debug)]
pub struct Poplar1<P, const L: usize> {
    bits: usize,
    phantom: PhantomData<P>,
}

impl<P: Prg<L>, const L: usize> Poplar1<P, L> {
    /// Construct a `Prng` with the given seed and info-string suffix.
    fn init_prng<I, B, F>(seed: &[u8; L], usage: u16, binder_chunks: I) -> Prng<F, P::SeedStream>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
        P: Prg<L>,
        F: FieldElement,
    {
        let mut prg = P::init(seed, &Self::custom(usage));
        for binder_chunk in binder_chunks.into_iter() {
            prg.update(binder_chunk.as_ref());
        }
        Prng::from_seed_stream(prg.into_seed_stream())
    }
}

impl<P, const L: usize> Clone for Poplar1<P, L> {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits,
            phantom: PhantomData,
        }
    }
}

/// Poplar1 public share.
///
/// This is comprised of the correction words generated for the IDPF.
pub type Poplar1PublicShare<const L: usize> =
    IdpfPublicShare<Poplar1IdpfValue<Field64>, Poplar1IdpfValue<Field255>, L>;

impl<P, const L: usize> ParameterizedDecode<Poplar1<P, L>> for Poplar1PublicShare<L> {
    fn decode_with_param(
        _poplar1: &Poplar1<P, L>,
        _bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        todo!()
    }
}

/// Poplar1 input share.
///
/// This is comprised of an IDPF key share and the correlated randomness used to compute the sketch
/// during preparation.
#[derive(Debug, Clone)]
pub struct Poplar1InputShare<const L: usize> {
    /// IDPF key share.
    idpf_key: Seed<L>,

    /// Seed used to generate the Aggregator's share of the correlated randomness used in the first
    /// part of the sketch.
    corr_seed: Seed<L>,

    /// Aggregator's share of the correlated randomness used in the second part of the sketch. Used
    /// for inner nodes of the IDPF tree.
    corr_inner: Vec<[Field64; 2]>,

    /// Aggregator's share of the correlated randomness used in the second part of the sketch. Used
    /// for leaf nodes of the IDPF tree.
    corr_leaf: [Field255; 2],
}

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
pub struct Poplar1PrepareState(PrepareStateVariant);

#[derive(Clone, Debug)]
enum PrepareStateVariant {
    Inner(PrepareState<Field64>),
    Leaf(PrepareState<Field255>),
}

#[derive(Clone, Debug)]
struct PrepareState<F> {
    sketch: SketchState<F>,
    output_share: Vec<F>,
}

#[derive(Clone, Debug)]
enum SketchState<F> {
    #[allow(non_snake_case)]
    RoundOne {
        A_share: F,
        B_share: F,
        is_leader: bool,
    },
    RoundTwo,
}

/// Poplar1 preparation message.
#[derive(Clone, Debug)]
pub struct Poplar1PrepareMessage(PrepareMessageVariant);

#[derive(Clone, Debug)]
enum PrepareMessageVariant {
    SketchInner([Field64; 3]),
    SketchLeaf([Field255; 3]),
    Done,
}

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

/// A vector of field elements transmitted while evaluating Poplar1.
#[derive(Clone, Debug)]
pub enum Poplar1FieldVec {
    /// Field type for inner nodes of the IDPF tree.
    Inner(Vec<Field64>),

    /// Field type for leaf nodes of the IDPF tree.
    Leaf(Vec<Field255>),
}

impl Poplar1FieldVec {
    fn zero(is_leaf: bool, len: usize) -> Self {
        if is_leaf {
            Self::Leaf(vec![Field255::zero(); len])
        } else {
            Self::Inner(vec![Field64::zero(); len])
        }
    }
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

    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError> {
        match (self, agg_share) {
            (Self::Inner(ref mut left), Self::Inner(right)) => Ok(merge_vector(left, right)?),
            (Self::Leaf(ref mut left), Self::Leaf(right)) => Ok(merge_vector(left, right)?),
            _ => Err(VdafError::Uncategorized(
                "cannot merge leaf nodes wiith inner nodes".into(),
            )),
        }
    }

    fn accumulate(&mut self, output_share: &Self) -> Result<(), VdafError> {
        match (self, output_share) {
            (Self::Inner(ref mut left), Self::Inner(right)) => Ok(merge_vector(left, right)?),
            (Self::Leaf(ref mut left), Self::Leaf(right)) => Ok(merge_vector(left, right)?),
            _ => Err(VdafError::Uncategorized(
                "cannot accumulate leaf nodes with inner nodes".into(),
            )),
        }
    }
}

/// Poplar1 aggregation parameter.
///
/// This includes an indication of what level of the IDPF tree is being evaluated and the set of
/// prefixes to evaluate at that level.
//
// TODO(cjpatton) spec: Make sure repeated prefixes are disallowed. To make this check easier,
// consider requring the prefixes to be in lexicographic order.
#[derive(Clone, Debug)]
pub struct Poplar1AggregationParam {
    level: usize,
    prefixes: Vec<IdpfInput>,
}

impl Poplar1AggregationParam {
    /// Construct an aggregation parameter from a set of candidate prefixes.
    ///
    /// # Errors
    ///
    /// * The list of prefixes is empty.
    /// * The prefixes have different lengths (they must all be the same).
    //
    // TODO spec: Ensure that prefixes don't repeat. To make this check easier, consider requiring
    // them to appear in alphabetical order.
    // https://github.com/cfrg/draft-irtf-cfrg-vdaf/issues/134
    pub fn try_from_prefixes(prefixes: Vec<IdpfInput>) -> Result<Self, VdafError> {
        if prefixes.is_empty() {
            return Err(VdafError::Uncategorized(
                "at least one prefix is required".into(),
            ));
        }

        let len = prefixes[0].len();
        for prefix in prefixes.iter() {
            if prefix.len() != len {
                return Err(VdafError::Uncategorized(
                    "all prefixes must have the same length".into(),
                ));
            }
        }

        Ok(Self {
            level: len - 1,
            prefixes,
        })
    }

    /// Return the level of the IDPF tree.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Return the prefixes.
    pub fn prefixes(&self) -> &[IdpfInput] {
        self.prefixes.as_ref()
    }
}

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

impl<P: Prg<L>, const L: usize> Client<16> for Poplar1<P, L> {
    fn shard(
        &self,
        input: &IdpfInput,
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Poplar1InputShare<L>>), VdafError> {
        if input.len() != self.bits {
            return Err(VdafError::Uncategorized(format!(
                "unexpected input length ({})",
                input.len()
            )));
        }

        // Generate the authenticator for each inner level of the IDPF tree.
        let mut prng = Self::init_prng::<_, _, Field64>(
            Seed::generate()?.as_ref(),
            DST_SHARD_RANDOMNESS,
            [&[]],
        );
        let auth_inner: Vec<Field64> = (0..self.bits - 1).map(|_| prng.get()).collect();

        // Generate the authenticator for the last level of the IDPF tree (i.e., the leaves).
        //
        // TODO(cjpatton) spec: Consider using a different PRG for the leaf and inner nodes.
        // "Switching" the PRG between field types is awkward.
        let mut prng = prng.into_new_field::<Field255>();
        let auth_leaf = prng.get();

        // Generate the IDPF shares.
        let (public_share, [idpf_key_0, idpf_key_1]) = idpf::gen::<_, _, _, P, L>(
            input,
            auth_inner
                .iter()
                .map(|auth| Poplar1IdpfValue([Field64::one(), *auth])),
            Poplar1IdpfValue([Field255::one(), auth_leaf]),
        )?;

        // Generate the correlated randomness for the inner nodes. This includes additive shares of
        // the random offsets `a, b, c` and additive shares of `A := -2*a + auth` and `B := a^2 + b
        // - a*auth + c`, where `auth` is the authenticator for the level of the tree. These values
        // are used, respectively, to compute and verify the sketch during the preparation phase.
        // (See Section 4.2 of [BBCG+21].)
        let corr_seed_0 = Seed::generate()?;
        let corr_seed_1 = Seed::generate()?;
        let mut prng = prng.into_new_field::<Field64>();
        let mut corr_prng_0 = Self::init_prng::<_, _, Field64>(
            corr_seed_0.as_ref(),
            DST_CORR_INNER,
            [[0].as_slice(), nonce.as_slice()],
        );
        let mut corr_prng_1 = Self::init_prng::<_, _, Field64>(
            corr_seed_1.as_ref(),
            DST_CORR_INNER,
            [[1].as_slice(), nonce.as_slice()],
        );
        let mut corr_inner_0 = Vec::with_capacity(self.bits - 1);
        let mut corr_inner_1 = Vec::with_capacity(self.bits - 1);
        for auth in auth_inner.into_iter() {
            let (next_corr_inner_0, next_corr_inner_1) =
                compute_next_corr_shares(&mut prng, &mut corr_prng_0, &mut corr_prng_1, auth);
            corr_inner_0.push(next_corr_inner_0);
            corr_inner_1.push(next_corr_inner_1);
        }

        // Generate the correlated randomness for the leaf nodes.
        let mut prng = prng.into_new_field::<Field255>();
        let mut corr_prng_0 = Self::init_prng::<_, _, Field255>(
            corr_seed_0.as_ref(),
            DST_CORR_LEAF,
            [[0].as_slice(), nonce.as_slice()],
        );
        let mut corr_prng_1 = Self::init_prng::<_, _, Field255>(
            corr_seed_1.as_ref(),
            DST_CORR_LEAF,
            [[1].as_slice(), nonce.as_slice()],
        );
        let (corr_leaf_0, corr_leaf_1) =
            compute_next_corr_shares(&mut prng, &mut corr_prng_0, &mut corr_prng_1, auth_leaf);

        Ok((
            public_share,
            vec![
                Poplar1InputShare {
                    idpf_key: idpf_key_0,
                    corr_seed: corr_seed_0,
                    corr_inner: corr_inner_0,
                    corr_leaf: corr_leaf_0,
                },
                Poplar1InputShare {
                    idpf_key: idpf_key_1,
                    corr_seed: corr_seed_1,
                    corr_inner: corr_inner_1,
                    corr_leaf: corr_leaf_1,
                },
            ],
        ))
    }
}

impl<P: Prg<L>, const L: usize> Aggregator<L, 16> for Poplar1<P, L> {
    type PrepareState = Poplar1PrepareState;
    type PrepareShare = Poplar1FieldVec;
    type PrepareMessage = Poplar1PrepareMessage;

    #[allow(clippy::type_complexity)]
    fn prepare_init(
        &self,
        verify_key: &[u8; L],
        agg_id: usize,
        agg_param: &Poplar1AggregationParam,
        nonce: &[u8; 16],
        public_share: &Poplar1PublicShare<L>,
        input_share: &Poplar1InputShare<L>,
    ) -> Result<(Poplar1PrepareState, Poplar1FieldVec), VdafError> {
        let is_leader = match agg_id {
            0 => true,
            1 => false,
            _ => {
                return Err(VdafError::Uncategorized(format!(
                    "invalid aggregator ID ({agg_id})"
                )))
            }
        };

        if agg_param.level < self.bits - 1 {
            let mut corr_prng = Self::init_prng::<_, _, Field64>(
                input_share.corr_seed.as_ref(),
                DST_CORR_INNER,
                [[agg_id as u8].as_slice(), nonce.as_slice()],
            );
            // Fast-forward the correlated randomness PRG to the level of the tree that we are
            // aggregating.
            for _ in 0..3 * agg_param.level {
                corr_prng.get();
            }

            let (output_share, sketch_share) = eval_and_sketch::<P, Field64, L>(
                verify_key,
                agg_id,
                nonce,
                agg_param,
                public_share,
                &input_share.idpf_key,
                &mut corr_prng,
            )?;

            Ok((
                Poplar1PrepareState(PrepareStateVariant::Inner(PrepareState {
                    sketch: SketchState::RoundOne {
                        A_share: input_share.corr_inner[agg_param.level][0],
                        B_share: input_share.corr_inner[agg_param.level][1],
                        is_leader,
                    },
                    output_share,
                })),
                Poplar1FieldVec::Inner(sketch_share),
            ))
        } else {
            let corr_prng = Self::init_prng::<_, _, Field255>(
                input_share.corr_seed.as_ref(),
                DST_CORR_LEAF,
                [[agg_id as u8].as_slice(), nonce.as_slice()],
            );

            let (output_share, sketch_share) = eval_and_sketch::<P, Field255, L>(
                verify_key,
                agg_id,
                nonce,
                agg_param,
                public_share,
                &input_share.idpf_key,
                &mut corr_prng.into_new_field(),
            )?;

            Ok((
                Poplar1PrepareState(PrepareStateVariant::Leaf(PrepareState {
                    sketch: SketchState::RoundOne {
                        A_share: input_share.corr_leaf[0],
                        B_share: input_share.corr_leaf[1],
                        is_leader,
                    },
                    output_share,
                })),
                Poplar1FieldVec::Leaf(sketch_share),
            ))
        }
    }

    fn prepare_preprocess<M: IntoIterator<Item = Poplar1FieldVec>>(
        &self,
        inputs: M,
    ) -> Result<Poplar1PrepareMessage, VdafError> {
        let mut inputs = inputs.into_iter();
        let prep_share_0 = inputs
            .next()
            .ok_or_else(|| VdafError::Uncategorized("insufficient number of prep shares".into()))?;
        let prep_share_1 = inputs
            .next()
            .ok_or_else(|| VdafError::Uncategorized("insufficient number of prep shares".into()))?;
        if inputs.next().is_some() {
            return Err(VdafError::Uncategorized(
                "more prep shares than expected".into(),
            ));
        }

        match (prep_share_0, prep_share_1) {
            (Poplar1FieldVec::Inner(share_0), Poplar1FieldVec::Inner(share_1)) => {
                Ok(Poplar1PrepareMessage(
                    next_message(share_0, share_1)?.map_or(PrepareMessageVariant::Done, |sketch| {
                        PrepareMessageVariant::SketchInner(sketch)
                    }),
                ))
            }
            (Poplar1FieldVec::Leaf(share_0), Poplar1FieldVec::Leaf(share_1)) => {
                Ok(Poplar1PrepareMessage(
                    next_message(share_0, share_1)?.map_or(PrepareMessageVariant::Done, |sketch| {
                        PrepareMessageVariant::SketchLeaf(sketch)
                    }),
                ))
            }
            _ => Err(VdafError::Uncategorized(
                "received prep shares with mismatched field types".into(),
            )),
        }
    }

    fn prepare_step(
        &self,
        state: Poplar1PrepareState,
        msg: Poplar1PrepareMessage,
    ) -> Result<PrepareTransition<Self, L, 16>, VdafError> {
        match (state.0, msg.0) {
            // Round one
            (
                PrepareStateVariant::Inner(PrepareState {
                    sketch:
                        SketchState::RoundOne {
                            A_share,
                            B_share,
                            is_leader,
                        },
                    output_share,
                }),
                PrepareMessageVariant::SketchInner(sketch),
            ) => Ok(PrepareTransition::Continue(
                Poplar1PrepareState(PrepareStateVariant::Inner(PrepareState {
                    sketch: SketchState::RoundTwo,
                    output_share,
                })),
                Poplar1FieldVec::Inner(finish_sketch(sketch, A_share, B_share, is_leader)),
            )),
            (
                PrepareStateVariant::Leaf(PrepareState {
                    sketch:
                        SketchState::RoundOne {
                            A_share,
                            B_share,
                            is_leader,
                        },
                    output_share,
                }),
                PrepareMessageVariant::SketchLeaf(sketch),
            ) => Ok(PrepareTransition::Continue(
                Poplar1PrepareState(PrepareStateVariant::Leaf(PrepareState {
                    sketch: SketchState::RoundTwo,
                    output_share,
                })),
                Poplar1FieldVec::Leaf(finish_sketch(sketch, A_share, B_share, is_leader)),
            )),

            // Round two
            (
                PrepareStateVariant::Inner(PrepareState {
                    sketch: SketchState::RoundTwo,
                    output_share,
                }),
                PrepareMessageVariant::Done,
            ) => Ok(PrepareTransition::Finish(Poplar1FieldVec::Inner(
                output_share,
            ))),
            (
                PrepareStateVariant::Leaf(PrepareState {
                    sketch: SketchState::RoundTwo,
                    output_share,
                }),
                PrepareMessageVariant::Done,
            ) => Ok(PrepareTransition::Finish(Poplar1FieldVec::Leaf(
                output_share,
            ))),

            _ => Err(VdafError::Uncategorized(
                "prep message field type does not match state".into(),
            )),
        }
    }

    fn aggregate<M: IntoIterator<Item = Poplar1FieldVec>>(
        &self,
        agg_param: &Poplar1AggregationParam,
        output_shares: M,
    ) -> Result<Poplar1FieldVec, VdafError> {
        aggregate(
            agg_param.level == self.bits - 1,
            agg_param.prefixes.len(),
            output_shares,
        )
    }
}

impl<P: Prg<L>, const L: usize> Collector for Poplar1<P, L> {
    fn unshard<M: IntoIterator<Item = Poplar1FieldVec>>(
        &self,
        agg_param: &Poplar1AggregationParam,
        agg_shares: M,
        _num_measurements: usize,
    ) -> Result<Vec<u64>, VdafError> {
        let result = aggregate(
            agg_param.level == self.bits - 1,
            agg_param.prefixes.len(),
            agg_shares,
        )?;

        match result {
            Poplar1FieldVec::Inner(vec) => Ok(vec.into_iter().map(u64::from).collect()),
            Poplar1FieldVec::Leaf(vec) => Ok(vec
                .into_iter()
                .map(u64::try_from)
                .collect::<Result<Vec<_>, _>>()?),
        }
    }
}

impl From<IdpfOutputShare<Poplar1IdpfValue<Field64>, Poplar1IdpfValue<Field255>>>
    for Poplar1IdpfValue<Field64>
{
    fn from(
        out_share: IdpfOutputShare<Poplar1IdpfValue<Field64>, Poplar1IdpfValue<Field255>>,
    ) -> Poplar1IdpfValue<Field64> {
        match out_share {
            IdpfOutputShare::Inner(array) => array,
            IdpfOutputShare::Leaf(..) => panic!("tried to convert leaf share into inner field"),
        }
    }
}

impl From<IdpfOutputShare<Poplar1IdpfValue<Field64>, Poplar1IdpfValue<Field255>>>
    for Poplar1IdpfValue<Field255>
{
    fn from(
        out_share: IdpfOutputShare<Poplar1IdpfValue<Field64>, Poplar1IdpfValue<Field255>>,
    ) -> Poplar1IdpfValue<Field255> {
        match out_share {
            IdpfOutputShare::Inner(..) => panic!("tried to convert inner share into leaf field"),
            IdpfOutputShare::Leaf(array) => array,
        }
    }
}

/// Derive shares of the correlated randomness for the next level of the IDPF tree.
//
// TODO(cjpatton) spec: Consider deriving the shares of a, b, c for each level directly from the
// seed, rather than iteratively, as we do in Doplar. This would be more efficient for the
// Aggregators. As long as the Client isn't significantly slower, this should be a win.
#[allow(non_snake_case)]
fn compute_next_corr_shares<F: FieldElement + From<u64>, S: SeedStream>(
    prng: &mut Prng<F, S>,
    corr_prng_0: &mut Prng<F, S>,
    corr_prng_1: &mut Prng<F, S>,
    auth: F,
) -> ([F; 2], [F; 2]) {
    let two = F::from(2);
    let a = corr_prng_0.get() + corr_prng_1.get();
    let b = corr_prng_0.get() + corr_prng_1.get();
    let c = corr_prng_0.get() + corr_prng_1.get();
    let A = -two * a + auth;
    let B = a * a + b - a * auth + c;
    let corr_1 = [prng.get(), prng.get()];
    let corr_0 = [A - corr_1[0], B - corr_1[1]];
    (corr_0, corr_1)
}

/// Evaluate the IDPF at the given prefixes and compute the Aggregator's share of the sketch.
fn eval_and_sketch<P, F, const L: usize>(
    verify_key: &[u8; L],
    agg_id: usize,
    nonce: &[u8; 16],
    agg_param: &Poplar1AggregationParam,
    public_share: &Poplar1PublicShare<L>,
    idpf_key: &Seed<L>,
    corr_prng: &mut Prng<F, P::SeedStream>,
) -> Result<(Vec<F>, Vec<F>), VdafError>
where
    P: Prg<L>,
    F: FieldElement,
    Poplar1IdpfValue<F>:
        From<IdpfOutputShare<Poplar1IdpfValue<Field64>, Poplar1IdpfValue<Field255>>>,
{
    let level = u16::try_from(agg_param.level)
        .map_err(|_| VdafError::Uncategorized(format!("level too deep ({})", agg_param.level)))?
        .to_be_bytes();

    // TODO(cjpatton) spec: Consider not encoding the prefixes here.
    let mut verify_prng = Poplar1::<P, L>::init_prng(
        verify_key,
        DST_VERIFY_RANDOMNESS,
        [nonce.as_slice(), level.as_slice()],
    );

    let mut out_share = Vec::with_capacity(agg_param.prefixes.len());
    let mut sketch_share = vec![
        corr_prng.get(), // a_share
        corr_prng.get(), // b_share
        corr_prng.get(), // c_share
    ];

    let mut idpf_eval_cache = RingBufferCache::new(agg_param.prefixes.len());
    for prefix in agg_param.prefixes.iter() {
        let share = Poplar1IdpfValue::<F>::from(idpf::eval::<
            Poplar1IdpfValue<Field64>,
            Poplar1IdpfValue<Field255>,
            P,
            L,
        >(
            agg_id,
            public_share,
            idpf_key,
            prefix,
            &mut idpf_eval_cache,
        )?);

        let r = verify_prng.get();
        let checked_data_share = share.0[0] * r;
        sketch_share[0] += checked_data_share;
        sketch_share[1] += checked_data_share * r;
        sketch_share[2] += share.0[1] * r;
        out_share.push(share.0[0]);
    }

    Ok((out_share, sketch_share))
}

/// Compute the Aggregator's share of the sketch verifier. The shares should sum to zero.
#[allow(non_snake_case)]
fn finish_sketch<F: FieldElement>(
    sketch: [F; 3],
    A_share: F,
    B_share: F,
    is_leader: bool,
) -> Vec<F> {
    let mut next_sketch_share = A_share * sketch[0] + B_share;
    if !is_leader {
        next_sketch_share += sketch[0] * sketch[0] - sketch[1] - sketch[2];
    }
    vec![next_sketch_share]
}

fn next_message<F: FieldElement>(
    mut share_0: Vec<F>,
    share_1: Vec<F>,
) -> Result<Option<[F; 3]>, VdafError> {
    merge_vector(&mut share_0, &share_1)?;

    if share_0.len() == 1 {
        if share_0[0] != F::zero() {
            Err(VdafError::Uncategorized(
                "sketch verification failed".into(),
            )) // Invalid sketch
        } else {
            Ok(None) // Sketch verification succeeded
        }
    } else if share_0.len() == 3 {
        Ok(Some([share_0[0], share_0[1], share_0[2]])) // Sketch verification continues
    } else {
        Err(VdafError::Uncategorized(format!(
            "unexpected sketch length ({})",
            share_0.len()
        )))
    }
}

fn aggregate<M: IntoIterator<Item = Poplar1FieldVec>>(
    is_leaf: bool,
    len: usize,
    shares: M,
) -> Result<Poplar1FieldVec, VdafError> {
    let mut result = Poplar1FieldVec::zero(is_leaf, len);
    for share in shares.into_iter() {
        result.accumulate(&share)?;
    }
    Ok(result)
}

/// A vector of two field elements.
///
/// This represents the values that Poplar1 programs into IDPFs while sharding.
#[derive(Debug, Clone, Copy)]
pub struct Poplar1IdpfValue<F>([F; 2]);

impl<F> Poplar1IdpfValue<F> {
    /// Create a new value from a pair of field elements.
    pub fn new(array: [F; 2]) -> Self {
        Self(array)
    }
}

impl<F> IdpfValue for Poplar1IdpfValue<F>
where
    F: FieldElement,
{
    fn zero() -> Self {
        Self([F::zero(); 2])
    }
}

impl<F> Add for Poplar1IdpfValue<F>
where
    F: Copy + Add<Output = F>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl<F> AddAssign for Poplar1IdpfValue<F>
where
    F: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
    }
}

impl<F> Sub for Poplar1IdpfValue<F>
where
    F: Copy + Sub<Output = F>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl<F> PartialEq for Poplar1IdpfValue<F>
where
    F: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<F> ConstantTimeEq for Poplar1IdpfValue<F>
where
    F: ConstantTimeEq,
{
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl<F> Encode for Poplar1IdpfValue<F>
where
    F: Encode,
{
    fn encode(&self, bytes: &mut Vec<u8>) {
        self.0[0].encode(bytes);
        self.0[1].encode(bytes);
    }
}

impl<F> Decode for Poplar1IdpfValue<F>
where
    F: Decode,
{
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self([F::decode(bytes)?, F::decode(bytes)?]))
    }
}

impl<F> CoinToss for Poplar1IdpfValue<F>
where
    F: CoinToss,
{
    fn sample<S>(seed_stream: &mut S) -> Self
    where
        S: SeedStream,
    {
        Self([F::sample(seed_stream), F::sample(seed_stream)])
    }
}

impl<F> ConditionallySelectable for Poplar1IdpfValue<F>
where
    F: ConditionallySelectable,
{
    fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
        Self([
            F::conditional_select(&a.0[0], &b.0[0], choice),
            F::conditional_select(&a.0[1], &b.0[1], choice),
        ])
    }
}

impl<F> ConditionallyNegatable for Poplar1IdpfValue<F>
where
    F: ConditionallyNegatable,
{
    fn conditional_negate(&mut self, choice: subtle::Choice) {
        F::conditional_negate(&mut self.0[0], choice);
        F::conditional_negate(&mut self.0[1], choice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::collections::HashSet;

    // Run preparation on the given report (i.e., public share and input shares). This is similar
    // to `vdaf::run_vdaf_prepare`, but does not exercise serialization.
    fn run_prepare<P: Prg<L>, const L: usize>(
        vdaf: &Poplar1<P, L>,
        verify_key: &[u8; L],
        nonce: &[u8; 16],
        public_share: &Poplar1PublicShare<L>,
        input_shares: &[Poplar1InputShare<L>],
        agg_param: &Poplar1AggregationParam,
    ) -> (Poplar1FieldVec, Poplar1FieldVec) {
        let (prep_state_0, prep_share_0) = vdaf
            .prepare_init(
                verify_key,
                0,
                agg_param,
                nonce,
                public_share,
                &input_shares[0],
            )
            .unwrap();

        let (prep_state_1, prep_share_1) = vdaf
            .prepare_init(
                verify_key,
                1,
                agg_param,
                nonce,
                public_share,
                &input_shares[1],
            )
            .unwrap();

        // Round one
        let prep_msg_1 = vdaf
            .prepare_preprocess([prep_share_0, prep_share_1])
            .unwrap();

        let (prep_state_0, prep_share_0) =
            match vdaf.prepare_step(prep_state_0, prep_msg_1.clone()).unwrap() {
                PrepareTransition::Continue(state, share) => (state, share),
                _ => panic!("expected continue"),
            };

        let (prep_state_1, prep_share_1) =
            match vdaf.prepare_step(prep_state_1, prep_msg_1).unwrap() {
                PrepareTransition::Continue(state, share) => (state, share),
                _ => panic!("expected continue"),
            };

        // Round two
        let prep_msg_2 = vdaf
            .prepare_preprocess([prep_share_0, prep_share_1])
            .unwrap();

        let out_share_0 = match vdaf.prepare_step(prep_state_0, prep_msg_2.clone()).unwrap() {
            PrepareTransition::Finish(share) => share,
            _ => panic!("expected finish"),
        };

        let out_share_1 = match vdaf.prepare_step(prep_state_1, prep_msg_2).unwrap() {
            PrepareTransition::Finish(share) => share,
            _ => panic!("expected finish"),
        };

        (out_share_0, out_share_1)
    }

    fn test_prepare<P: Prg<L>, const L: usize>(
        vdaf: &Poplar1<P, L>,
        verify_key: &[u8; L],
        nonce: &[u8; 16],
        public_share: &Poplar1PublicShare<L>,
        input_shares: &[Poplar1InputShare<L>],
        agg_param: &Poplar1AggregationParam,
        expected_result: Vec<u64>,
    ) {
        let (out_share_0, out_share_1) = run_prepare(
            vdaf,
            verify_key,
            nonce,
            public_share,
            input_shares,
            agg_param,
        );

        // Convert aggregate shares and unshard.
        let agg_share_0 = vdaf.aggregate(agg_param, [out_share_0]).unwrap();
        let agg_share_1 = vdaf.aggregate(agg_param, [out_share_1]).unwrap();
        let result = vdaf
            .unshard(agg_param, [agg_share_0, agg_share_1], 1)
            .unwrap();
        assert_eq!(
            result, expected_result,
            "unexpected result (level={})",
            agg_param.level
        );
    }

    fn run_heavy_hitters<B: AsRef<[u8]>, P: Prg<L>, const L: usize>(
        vdaf: &Poplar1<P, L>,
        verify_key: &[u8; L],
        threshold: usize,
        measurements: impl IntoIterator<Item = B>,
        expected_result: impl IntoIterator<Item = B>,
    ) {
        let mut rng = thread_rng();

        // Sharding step
        let reports: Vec<([u8; 16], Poplar1PublicShare<L>, Vec<Poplar1InputShare<L>>)> =
            measurements
                .into_iter()
                .map(|measurement| {
                    let nonce = rng.gen();
                    let (public_share, input_shares) = vdaf
                        .shard(&IdpfInput::from_bytes(measurement.as_ref()), &nonce)
                        .unwrap();
                    (nonce, public_share, input_shares)
                })
                .collect();

        let mut agg_param = Poplar1AggregationParam {
            level: 0,
            prefixes: vec![
                IdpfInput::from_bools(&[false]),
                IdpfInput::from_bools(&[true]),
            ],
        };

        let mut agg_result = Vec::new();
        for level in 0..vdaf.bits {
            let mut out_shares_0 = Vec::with_capacity(reports.len());
            let mut out_shares_1 = Vec::with_capacity(reports.len());

            // Preparation step
            for (nonce, public_share, input_shares) in reports.iter() {
                let (out_share_0, out_share_1) = run_prepare(
                    vdaf,
                    verify_key,
                    nonce,
                    public_share,
                    input_shares,
                    &agg_param,
                );
                out_shares_0.push(out_share_0);
                out_shares_1.push(out_share_1);
            }

            // Aggregation step
            let agg_share_0 = vdaf.aggregate(&agg_param, out_shares_0).unwrap();
            let agg_share_1 = vdaf.aggregate(&agg_param, out_shares_1).unwrap();

            // Unsharding step
            agg_result = vdaf
                .unshard(&agg_param, [agg_share_0, agg_share_1], reports.len())
                .unwrap();

            agg_param.level += 1;

            // Unless this is the last level of the tree, construct the next set of candidate
            // prefixes.
            if level < vdaf.bits - 1 {
                let mut next_prefixes = Vec::new();
                for (prefix, count) in agg_param.prefixes.into_iter().zip(agg_result.iter()) {
                    if *count >= threshold as u64 {
                        next_prefixes.push(prefix.clone_with_suffix(&[false]));
                        next_prefixes.push(prefix.clone_with_suffix(&[true]));
                    }
                }

                agg_param.prefixes = next_prefixes;
            }
        }

        let got: HashSet<IdpfInput> = agg_param
            .prefixes
            .into_iter()
            .zip(agg_result.iter())
            .filter(|(_prefix, count)| **count >= threshold as u64)
            .map(|(prefix, _count)| prefix)
            .collect();

        let want: HashSet<IdpfInput> = expected_result
            .into_iter()
            .map(|bytes| IdpfInput::from_bytes(bytes.as_ref()))
            .collect();

        assert_eq!(got, want);
    }

    #[test]
    fn shard_prepare() {
        let mut rng = thread_rng();
        let verify_key = rng.gen();
        let nonce = rng.gen::<[u8; 16]>();
        let vdaf = Poplar1::new_sha3(64);

        let input = IdpfInput::from_bytes(b"12341324");
        let (public_share, input_shares) = vdaf.shard(&input, &nonce).unwrap();

        test_prepare(
            &vdaf,
            &verify_key,
            &nonce,
            &public_share,
            &input_shares,
            &Poplar1AggregationParam {
                level: 7,
                prefixes: vec![
                    IdpfInput::from_bytes(b"0"),
                    IdpfInput::from_bytes(b"1"),
                    IdpfInput::from_bytes(b"2"),
                    IdpfInput::from_bytes(b"f"),
                ],
            },
            vec![0, 1, 0, 0],
        );

        for level in 0..vdaf.bits {
            test_prepare(
                &vdaf,
                &verify_key,
                &nonce,
                &public_share,
                &input_shares,
                &Poplar1AggregationParam {
                    level,
                    prefixes: vec![input.prefix(level)],
                },
                vec![1],
            );
        }
    }

    #[test]
    fn heavy_hitters() {
        let mut rng = thread_rng();
        let verify_key = rng.gen();
        let vdaf = Poplar1::new_sha3(8);

        run_heavy_hitters(
            &vdaf,
            &verify_key,
            2, // threshold
            [
                "a", "b", "c", "d", "e", "f", "g", "g", "h", "i", "i", "i", "j", "j", "k", "l",
            ], // measurements
            ["g", "i", "j"], // heavy hitters
        );
    }
}
