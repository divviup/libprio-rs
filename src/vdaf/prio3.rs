// SPDX-License-Identifier: MPL-2.0

//! This modulde implements the Prio3 VDAF as specified in [[draft-irtf-cfrg-vdaf-01]]. The
//! construction is based on a transform of a Fully Linear Proof (FLP) system (i.e., a concrete
//! [`Type`](crate::flp::Type) into a zero-knowledge proof system on distributed data as described
//! in [[BBCG+19], Section 6].
//!
//! Implementations of `Prio3Aes128Count`, `Prio3Aes128Sum`, and `Prio3Aes128Histogram` as
//! specified in [[draft-irtf-cfrg-vdaf-01], Section 6.4] are provided. This module also provides
//! additional instantiations of Prio3 that are not in the draft.
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [draft-irtf-cfrg-vdaf-01]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/01/

use crate::codec::{CodecError, Decode, Encode, ParameterizedDecode};
use crate::field::{Field128, Field64, FieldElement};
#[cfg(feature = "multithreaded")]
use crate::flp::gadgets::ParallelSumMultithreaded;
use crate::flp::gadgets::{BlindPolyEval, ParallelSum, ParallelSumGadget};
use crate::flp::types::{Count, CountVec, Histogram, Sum};
use crate::flp::Type;
use crate::prng::Prng;
use crate::vdaf::prg::{Prg, PrgAes128, RandSource, Seed};
use crate::vdaf::{
    Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare, PrepareTransition,
    Share, ShareDecodingParameter, Vdaf, VdafError,
};
use std::convert::{TryFrom, TryInto};
use std::fmt::Debug;
use std::io::Cursor;
use std::iter::IntoIterator;
use std::marker::PhantomData;

// Domain-separation tag used to bind the VDAF operations to the document version. This will be
// reved with each new draft.
//
// NOTE The CFRG has not yet adopted this spec. Version "vdaf-01" will match
// draft-irtf-cfrg-vdaf-01.
const VERS_PRIO3: &[u8] = b"vdaf-01 prio3";

/// The count type. Each measurement is an integer in `[0,2)` and the aggregate is the sum.
pub type Prio3Aes128Count = Prio3<Count<Field64>, Prio3Result<u64>, PrgAes128, 16>;

impl Prio3Aes128Count {
    /// Construct an instance of this VDAF with the given suite and the given number of aggregators.
    pub fn new(num_aggregators: u8) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        Ok(Prio3 {
            num_aggregators,
            typ: Count::new(),
            phantom: PhantomData,
        })
    }
}

/// The count-vector type. Each measurement is a vector of integers in `[0,2)` and the aggregate is
/// the element-wise sum.
pub type Prio3Aes128CountVec = Prio3<
    CountVec<Field128, ParallelSum<Field128, BlindPolyEval<Field128>>>,
    Prio3ResultVec<u64>,
    PrgAes128,
    16,
>;

/// Like [`Prio3CountVec`] except this type uses multithreading to improve sharding and
/// preparation time. Note that the improvement is only noticeable for very large input lengths,
/// e.g., 201 and up. (Your system's mileage may vary.)
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3Aes128CountVecMultithreaded = Prio3<
    CountVec<Field128, ParallelSumMultithreaded<Field128, BlindPolyEval<Field128>>>,
    Prio3ResultVec<u64>,
    PrgAes128,
    16,
>;

impl<S, P, const L: usize> Prio3<CountVec<Field128, S>, Prio3ResultVec<u64>, P, L>
where
    S: 'static + ParallelSumGadget<Field128, BlindPolyEval<Field128>> + Eq,
    P: Prg<L>,
{
    /// Construct an instance of this VDAF with the given suite and the given number of
    /// aggregators. `len` defines the length of each measurement.
    pub fn new(num_aggregators: u8, len: usize) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        Ok(Prio3 {
            num_aggregators,
            typ: CountVec::new(len),
            phantom: PhantomData,
        })
    }
}

/// The sum type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and the
/// aggregate is the sum.
pub type Prio3Aes128Sum = Prio3<Sum<Field128>, Prio3Result<u64>, PrgAes128, 16>;

impl Prio3Aes128Sum {
    /// Construct an instance of this VDAF with the given suite, number of aggregators and required
    /// bit length. The bit length must not exceed 64.
    pub fn new(num_aggregators: u8, bits: u32) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({}) exceeds limit for aggregate type (64)",
                bits
            )));
        }

        Ok(Prio3 {
            num_aggregators,
            typ: Sum::new(bits as usize)?,
            phantom: PhantomData,
        })
    }
}

/// the histogram type. Each measurement is an unsigned, 64-bit integer and the result is a
/// histogram representation of the measurement.
pub type Prio3Aes128Histogram = Prio3<Histogram<Field128>, Prio3ResultVec<u64>, PrgAes128, 16>;

impl Prio3Aes128Histogram {
    /// Constructs an instance of this VDAF with the given suite, number of aggregators, and
    /// desired histogram bucket boundaries.
    pub fn new(num_aggregators: u8, buckets: &[u64]) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        let buckets = buckets.iter().map(|bucket| *bucket as u128).collect();

        Ok(Prio3 {
            num_aggregators,
            typ: Histogram::<Field128>::new(buckets)?,
            phantom: PhantomData,
        })
    }
}

/// Aggregate result for singleton data types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prio3Result<T: Eq>(pub T);

impl<F: FieldElement> TryFrom<AggregateShare<F>> for Prio3Result<u64> {
    type Error = VdafError;

    fn try_from(data: AggregateShare<F>) -> Result<Self, VdafError> {
        if data.0.len() != 1 {
            return Err(VdafError::Uncategorized(format!(
                "unexpected aggregate length for count type: got {}; want 1",
                data.0.len()
            )));
        }

        let out: u64 = F::Integer::from(data.0[0]).try_into().map_err(|err| {
            VdafError::Uncategorized(format!("result too large for output type: {:?}", err))
        })?;

        Ok(Prio3Result(out))
    }
}

/// Aggregate result for vector data types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prio3ResultVec<T: Eq>(pub Vec<T>);

impl<F: FieldElement> TryFrom<AggregateShare<F>> for Prio3ResultVec<u64> {
    type Error = VdafError;

    fn try_from(data: AggregateShare<F>) -> Result<Self, VdafError> {
        let mut out = Vec::with_capacity(data.0.len());
        for elem in data.0.into_iter() {
            out.push(F::Integer::from(elem).try_into().map_err(|err| {
                VdafError::Uncategorized(format!("result too large for output type: {:?}", err))
            })?);
        }

        Ok(Prio3ResultVec(out))
    }
}

fn check_num_aggregators(num_aggregators: u8) -> Result<(), VdafError> {
    if num_aggregators == 0 {
        return Err(VdafError::Uncategorized(format!(
            "at least one aggregator is required; got {}",
            num_aggregators
        )));
    } else if num_aggregators > 254 {
        return Err(VdafError::Uncategorized(format!(
            "number of aggregators must not exceed 254; got {}",
            num_aggregators
        )));
    }

    Ok(())
}

/// The base type for prio3.
#[derive(Clone, Debug)]
pub struct Prio3<T, A, P, const L: usize>
where
    T: Type,
    A: Clone + Debug,
    P: Prg<L>,
{
    num_aggregators: u8,
    typ: T,
    phantom: PhantomData<(A, P)>,
}

impl<T, A, P, const L: usize> Prio3<T, A, P, L>
where
    T: Type,
    A: Clone + Debug,
    P: Prg<L>,
{
    /// The output length of the underlying FLP.
    pub fn output_len(&self) -> usize {
        self.typ.output_len()
    }

    /// The verifier length of the underlying FLP.
    pub fn verifier_len(&self) -> usize {
        self.typ.verifier_len()
    }

    fn shard_with_rand_source(
        &self,
        measurement: &T::Measurement,
        rand_source: RandSource,
    ) -> Result<Vec<Prio3InputShare<T::Field, L>>, VdafError> {
        let mut info = [0; VERS_PRIO3.len() + 1];
        info[..VERS_PRIO3.len()].clone_from_slice(VERS_PRIO3);

        let num_aggregators = self.num_aggregators;
        let input = self.typ.encode(measurement)?;

        // Generate the input shares and compute the joint randomness.
        let mut helper_shares = Vec::with_capacity(num_aggregators as usize - 1);
        let mut leader_input_share = input.clone();
        let mut joint_rand_seed = Seed::uninitialized();
        for agg_id in 1..num_aggregators {
            let mut helper = HelperShare::from_rand_source(rand_source)?;

            let mut deriver = P::init(helper.joint_rand_param.blind.as_ref());
            deriver.update(&[agg_id]);
            info[VERS_PRIO3.len()] = agg_id;
            let prng: Prng<T::Field, _> =
                Prng::from_seed_stream(P::seed_stream(&helper.input_share, &info));
            for (x, y) in leader_input_share
                .iter_mut()
                .zip(prng)
                .take(self.typ.input_len())
            {
                *x -= y;
                deriver.update(&y.into());
            }

            helper.joint_rand_param.seed_hint = deriver.into_seed();
            joint_rand_seed.xor_accumulate(&helper.joint_rand_param.seed_hint);

            helper_shares.push(helper);
        }

        let leader_blind = Seed::from_rand_source(rand_source)?;

        let mut deriver = P::init(leader_blind.as_ref());
        deriver.update(&[0]); // ID of the leader
        for x in leader_input_share.iter() {
            deriver.update(&(*x).into());
        }

        let mut leader_joint_rand_seed_hint = deriver.into_seed();
        joint_rand_seed.xor_accumulate(&leader_joint_rand_seed_hint);

        // Run the proof-generation algorithm.
        let prng: Prng<T::Field, _> =
            Prng::from_seed_stream(P::seed_stream(&joint_rand_seed, VERS_PRIO3));
        let joint_rand: Vec<T::Field> = prng.take(self.typ.joint_rand_len()).collect();
        let prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
            &Seed::from_rand_source(rand_source)?,
            VERS_PRIO3,
        ));
        let prove_rand: Vec<T::Field> = prng.take(self.typ.prove_rand_len()).collect();
        let mut leader_proof_share = self.typ.prove(&input, &prove_rand, &joint_rand)?;

        // Generate the proof shares and finalize the joint randomness seed hints.
        for (j, helper) in helper_shares.iter_mut().enumerate() {
            info[VERS_PRIO3.len()] = j as u8 + 1;
            let prng: Prng<T::Field, _> =
                Prng::from_seed_stream(P::seed_stream(&helper.proof_share, &info));
            for (x, y) in leader_proof_share
                .iter_mut()
                .zip(prng)
                .take(self.typ.proof_len())
            {
                *x -= y;
            }

            helper
                .joint_rand_param
                .seed_hint
                .xor_accumulate(&joint_rand_seed);
        }

        leader_joint_rand_seed_hint.xor_accumulate(&joint_rand_seed);

        let leader_joint_rand_param = if self.typ.joint_rand_len() > 0 {
            Some(JointRandParam {
                seed_hint: leader_joint_rand_seed_hint,
                blind: leader_blind,
            })
        } else {
            None
        };

        // Prep the output messages.
        let mut out = Vec::with_capacity(num_aggregators as usize);
        out.push(Prio3InputShare {
            input_share: Share::Leader(leader_input_share),
            proof_share: Share::Leader(leader_proof_share),
            joint_rand_param: leader_joint_rand_param,
        });

        for helper in helper_shares.into_iter() {
            let helper_joint_rand_param = if self.typ.joint_rand_len() > 0 {
                Some(helper.joint_rand_param)
            } else {
                None
            };

            out.push(Prio3InputShare {
                input_share: Share::Helper(helper.input_share),
                proof_share: Share::Helper(helper.proof_share),
                joint_rand_param: helper_joint_rand_param,
            });
        }

        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn test_vec_shard(
        &self,
        measurement: &T::Measurement,
    ) -> Result<Vec<Prio3InputShare<T::Field, L>>, VdafError> {
        self.shard_with_rand_source(measurement, |buf| {
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

impl<T, A, P, const L: usize> Vdaf for Prio3<T, A, P, L>
where
    T: Type,
    A: Clone + Debug + Sync + Send,
    P: Prg<L>,
{
    type Measurement = T::Measurement;
    type AggregateResult = A;
    type AggregationParam = ();
    type InputShare = Prio3InputShare<T::Field, L>;
    type OutputShare = OutputShare<T::Field>;
    type AggregateShare = AggregateShare<T::Field>;

    fn num_aggregators(&self) -> usize {
        self.num_aggregators as usize
    }
}

/// The message sent by the client to each aggregator. This includes the client's input share and
/// the initial message of the input-validation protocol.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Prio3InputShare<F, const L: usize> {
    /// The input share.
    input_share: Share<F, L>,

    /// The proof share.
    proof_share: Share<F, L>,

    /// Parameters used by the Aggregator to compute the joint randomness. This field is optional
    /// because not every [`Type`](`crate::flp::Type`) requires joint randomness.
    joint_rand_param: Option<JointRandParam<L>>,
}

impl<F: FieldElement, const L: usize> Encode for Prio3InputShare<F, L> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        if matches!(
            (&self.input_share, &self.proof_share),
            (Share::Leader(_), Share::Helper(_)) | (Share::Helper(_), Share::Leader(_))
        ) {
            panic!("tried to encode input share with ambiguous encoding")
        }

        self.input_share.encode(bytes);
        self.proof_share.encode(bytes);
        if let Some(ref param) = self.joint_rand_param {
            param.blind.encode(bytes);
            param.seed_hint.encode(bytes);
        }
    }
}

impl<'a, T, A, P, const L: usize> ParameterizedDecode<(&'a Prio3<T, A, P, L>, usize)>
    for Prio3InputShare<T::Field, L>
where
    T: Type,
    A: Clone + Debug,
    P: Prg<L>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, A, P, L>, usize),
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

        let input_share = Share::decode_with_param(&input_decoder, bytes)?;
        let proof_share = Share::decode_with_param(&proof_decoder, bytes)?;
        let joint_rand_param = if prio3.typ.joint_rand_len() > 0 {
            Some(JointRandParam {
                blind: Seed::decode(bytes)?,
                seed_hint: Seed::decode(bytes)?,
            })
        } else {
            None
        };

        Ok(Prio3InputShare {
            input_share,
            proof_share,
            joint_rand_param,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Prepare broadcast message for Prio3.
pub struct Prio3PrepareShare<F, const L: usize> {
    /// A share of the FLP verifier message. (See [`Type`](crate::flp::Type).)
    verifier: Vec<F>,

    /// A share of the joint randomness seed.
    joint_rand_seed: Option<Seed<L>>,
}

impl<F: FieldElement, const L: usize> Encode for Prio3PrepareShare<F, L> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        for x in &self.verifier {
            x.encode(bytes);
        }
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encode(bytes);
        }
    }
}

impl<F: FieldElement, const L: usize> ParameterizedDecode<Prio3PrepareState<F, L>>
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

        let joint_rand_seed = if decoding_parameter.joint_rand_seed.is_some() {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Prio3PrepareShare {
            verifier,
            joint_rand_seed,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Prepare message for Prio3.
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

impl<F: FieldElement, const L: usize> ParameterizedDecode<Prio3PrepareState<F, L>>
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

impl<T, A, P, const L: usize> Client for Prio3<T, A, P, L>
where
    T: Type,
    A: Clone + Debug + Sync + Send,
    P: Prg<L>,
{
    fn shard(
        &self,
        measurement: &T::Measurement,
    ) -> Result<Vec<Prio3InputShare<T::Field, L>>, VdafError> {
        self.shard_with_rand_source(measurement, getrandom::getrandom)
    }
}

/// State of each aggregator during the preparation phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Prio3PrepareState<F, const L: usize> {
    input_share: Share<F, L>,
    joint_rand_seed: Option<Seed<L>>,
    agg_id: u8,
    verifier_len: usize,
}

impl<F: FieldElement, const L: usize> Encode for Prio3PrepareState<F, L> {
    /// Append the encoded form of this object to the end of `bytes`, growing the vector as needed.
    fn encode(&self, bytes: &mut Vec<u8>) {
        self.input_share.encode(bytes);
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encode(bytes);
        }
    }
}

impl<'a, T, A, P, const L: usize> ParameterizedDecode<(&'a Prio3<T, A, P, L>, usize)>
    for Prio3PrepareState<T::Field, L>
where
    T: Type,
    A: Clone + Debug,
    P: Prg<L>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, A, P, L>, usize),
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
        let input_share = Share::decode_with_param(&share_decoder, bytes)?;

        let joint_rand_seed = if prio3.typ.joint_rand_len() > 0 {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Self {
            input_share,
            joint_rand_seed,
            agg_id,
            verifier_len: prio3.typ.verifier_len(),
        })
    }
}

impl<T, A, P, const L: usize> Aggregator<L> for Prio3<T, A, P, L>
where
    T: Type,
    A: Clone + Debug + Sync + Send,
    P: Prg<L>,
{
    type PrepareState = Prio3PrepareState<T::Field, L>;
    type PrepareShare = Prio3PrepareShare<T::Field, L>;
    type PrepareMessage = Prio3PrepareMessage<L>;

    /// Begins the Prep process with the other aggregators. The result of this process is
    /// the aggregator's output share.
    fn prepare_init(
        &self,
        verify_key: &[u8; L],
        agg_id: usize,
        _agg_param: &(),
        nonce: &[u8],
        msg: &Prio3InputShare<T::Field, L>,
    ) -> Result<
        (
            Prio3PrepareState<T::Field, L>,
            Prio3PrepareShare<T::Field, L>,
        ),
        VdafError,
    > {
        let agg_id = self.role_try_from(agg_id)?;
        let mut info = [0; VERS_PRIO3.len() + 1];
        info[..VERS_PRIO3.len()].clone_from_slice(VERS_PRIO3);
        info[VERS_PRIO3.len()] = agg_id;

        let mut deriver = P::init(verify_key);
        deriver.update(&[255]);
        deriver.update(nonce);
        let query_rand_seed = deriver.into_seed();

        // Create a reference to the (expanded) input share.
        let expanded_input_share: Option<Vec<T::Field>> = match msg.input_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => {
                let prng = Prng::from_seed_stream(P::seed_stream(seed, &info));
                Some(prng.take(self.typ.input_len()).collect())
            }
        };
        let input_share = match msg.input_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_input_share.as_ref().unwrap(),
        };

        // Create a reference to the (expanded) proof share.
        let expanded_proof_share: Option<Vec<T::Field>> = match msg.proof_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => {
                let prng = Prng::from_seed_stream(P::seed_stream(seed, &info));
                Some(prng.take(self.typ.proof_len()).collect())
            }
        };
        let proof_share = match msg.proof_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_proof_share.as_ref().unwrap(),
        };

        // Compute the joint randomness.
        let (joint_rand_seed, joint_rand_seed_share, joint_rand) = if self.typ.joint_rand_len() > 0
        {
            let mut deriver = P::init(msg.joint_rand_param.as_ref().unwrap().blind.as_ref());
            deriver.update(&[agg_id]);
            for x in input_share {
                deriver.update(&(*x).into());
            }
            let joint_rand_seed_share = deriver.into_seed();

            let mut joint_rand_seed = Seed::uninitialized();
            joint_rand_seed.xor(
                &msg.joint_rand_param.as_ref().unwrap().seed_hint,
                &joint_rand_seed_share,
            );

            let prng: Prng<T::Field, _> =
                Prng::from_seed_stream(P::seed_stream(&joint_rand_seed, VERS_PRIO3));
            (
                Some(joint_rand_seed),
                Some(joint_rand_seed_share),
                prng.take(self.typ.joint_rand_len()).collect(),
            )
        } else {
            (None, None, Vec::new())
        };

        // Compute the query randomness.
        let prng: Prng<T::Field, _> =
            Prng::from_seed_stream(P::seed_stream(&query_rand_seed, VERS_PRIO3));
        let query_rand: Vec<T::Field> = prng.take(self.typ.query_rand_len()).collect();

        // Run the query-generation algorithm.
        let verifier_share = self.typ.query(
            input_share,
            proof_share,
            &query_rand,
            &joint_rand,
            self.num_aggregators as usize,
        )?;

        Ok((
            Prio3PrepareState {
                input_share: msg.input_share.clone(),
                joint_rand_seed,
                agg_id,
                verifier_len: verifier_share.len(),
            },
            Prio3PrepareShare {
                verifier: verifier_share,
                joint_rand_seed: joint_rand_seed_share,
            },
        ))
    }

    fn prepare_preprocess<M: IntoIterator<Item = Prio3PrepareShare<T::Field, L>>>(
        &self,
        inputs: M,
    ) -> Result<Prio3PrepareMessage<L>, VdafError> {
        let mut verifier = vec![T::Field::zero(); self.typ.verifier_len()];
        let mut joint_rand_seed = Seed::uninitialized();
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
                let joint_rand_seed_share = share.joint_rand_seed.unwrap();
                joint_rand_seed.xor_accumulate(&joint_rand_seed_share);
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
            Some(joint_rand_seed)
        } else {
            None
        };

        Ok(Prio3PrepareMessage { joint_rand_seed })
    }

    fn prepare_step(
        &self,
        step: Prio3PrepareState<T::Field, L>,
        msg: Prio3PrepareMessage<L>,
    ) -> Result<PrepareTransition<Self, L>, VdafError> {
        if self.typ.joint_rand_len() > 0 {
            // Check that the joint randomness was correct.
            if step.joint_rand_seed.as_ref().unwrap() != msg.joint_rand_seed.as_ref().unwrap() {
                return Err(VdafError::Uncategorized(
                    "joint randomness mismatch".to_string(),
                ));
            }
        }

        // Compute the output share.
        let input_share = match step.input_share {
            Share::Leader(data) => data,
            Share::Helper(seed) => {
                let mut info = [0; VERS_PRIO3.len() + 1];
                info[..VERS_PRIO3.len()].clone_from_slice(VERS_PRIO3);
                info[VERS_PRIO3.len()] = step.agg_id;
                let prng = Prng::from_seed_stream(P::seed_stream(&seed, &info));
                prng.take(self.typ.input_len()).collect()
            }
        };

        let output_share = match self.typ.truncate(input_share) {
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

impl<T, A, P, const L: usize> Collector for Prio3<T, A, P, L>
where
    T: Type,
    A: Clone + Debug + Sync + Send + TryFrom<AggregateShare<T::Field>, Error = VdafError> + Eq,
    P: Prg<L>,
{
    /// Combines aggregate shares into the aggregate result.
    fn unshard<It: IntoIterator<Item = AggregateShare<T::Field>>>(
        &self,
        _agg_param: &(),
        agg_shares: It,
    ) -> Result<A, VdafError> {
        let mut agg = AggregateShare(vec![T::Field::zero(); self.typ.output_len()]);
        for agg_share in agg_shares.into_iter() {
            agg.merge(&agg_share)?;
        }

        A::try_from(agg)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct JointRandParam<const L: usize> {
    /// The sum of the joint randomness seed shares sent to the other Aggregators.
    seed_hint: Seed<L>,

    /// The blinding factor, used to derive the aggregator's joint randomness seed share.
    blind: Seed<L>,
}

#[derive(Clone)]
struct HelperShare<const L: usize> {
    input_share: Seed<L>,
    proof_share: Seed<L>,
    joint_rand_param: JointRandParam<L>,
}

impl<const L: usize> HelperShare<L> {
    fn from_rand_source(rand_source: RandSource) -> Result<Self, VdafError> {
        Ok(HelperShare {
            input_share: Seed::from_rand_source(rand_source)?,
            proof_share: Seed::from_rand_source(rand_source)?,
            joint_rand_param: JointRandParam {
                seed_hint: Seed::uninitialized(),
                blind: Seed::from_rand_source(rand_source)?,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vdaf::{run_vdaf, run_vdaf_prepare};
    use assert_matches::assert_matches;
    use rand::prelude::*;

    #[test]
    fn test_prio3_count() {
        let prio3 = Prio3Aes128Count::new(2).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [1, 0, 0, 1, 1]).unwrap(),
            Prio3Result(3)
        );

        let mut verify_key = [0; 16];
        thread_rng().fill(&mut verify_key[..]);
        let nonce = b"This is a good nonce.";

        let input_shares = prio3.shard(&0).unwrap();
        run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares).unwrap();

        let input_shares = prio3.shard(&1).unwrap();
        run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares).unwrap();

        test_prepare_state_serialization(&prio3, &1).unwrap();
    }

    #[test]
    fn test_prio3_sum() {
        let prio3 = Prio3Aes128Sum::new(3, 16).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [0, (1 << 16) - 1, 0, 1, 1]).unwrap(),
            Prio3Result((1 << 16) + 1)
        );

        let mut verify_key = [0; 16];
        thread_rng().fill(&mut verify_key[..]);
        let nonce = b"This is a good nonce.";

        let mut input_shares = prio3.shard(&1).unwrap();
        input_shares[0].joint_rand_param.as_mut().unwrap().blind.0[0] ^= 255;
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let mut input_shares = prio3.shard(&1).unwrap();
        input_shares[0]
            .joint_rand_param
            .as_mut()
            .unwrap()
            .seed_hint
            .0[0] ^= 255;
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let mut input_shares = prio3.shard(&1).unwrap();
        assert_matches!(input_shares[0].input_share, Share::Leader(ref mut data) => {
            data[0] += Field128::one();
        });
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let mut input_shares = prio3.shard(&1).unwrap();
        assert_matches!(input_shares[0].proof_share, Share::Leader(ref mut data) => {
                data[0] += Field128::one();
        });
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        test_prepare_state_serialization(&prio3, &1).unwrap();
    }

    #[test]
    fn test_prio3_histogram() {
        let prio3 = Prio3Aes128Histogram::new(2, &[0, 10, 20]).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [0, 10, 20, 9999]).unwrap(),
            Prio3ResultVec(vec![1, 1, 1, 1])
        );

        assert_eq!(
            run_vdaf(&prio3, &(), [0]).unwrap(),
            Prio3ResultVec(vec![1, 0, 0, 0])
        );

        assert_eq!(
            run_vdaf(&prio3, &(), [5]).unwrap(),
            Prio3ResultVec(vec![0, 1, 0, 0])
        );

        assert_eq!(
            run_vdaf(&prio3, &(), [10]).unwrap(),
            Prio3ResultVec(vec![0, 1, 0, 0])
        );

        assert_eq!(
            run_vdaf(&prio3, &(), [15]).unwrap(),
            Prio3ResultVec(vec![0, 0, 1, 0])
        );

        assert_eq!(
            run_vdaf(&prio3, &(), [20]).unwrap(),
            Prio3ResultVec(vec![0, 0, 1, 0])
        );

        assert_eq!(
            run_vdaf(&prio3, &(), [25]).unwrap(),
            Prio3ResultVec(vec![0, 0, 0, 1])
        );

        test_prepare_state_serialization(&prio3, &23).unwrap();
    }

    #[test]
    fn test_prio3_input_share() {
        let prio3 = Prio3Aes128Sum::new(5, 16).unwrap();
        let input_shares = prio3.shard(&1).unwrap();

        // Check that seed shares are distinct.
        for (i, x) in input_shares.iter().enumerate() {
            for (j, y) in input_shares.iter().enumerate() {
                if i != j {
                    if let (Share::Helper(left), Share::Helper(right)) =
                        (&x.input_share, &y.input_share)
                    {
                        assert_ne!(left, right);
                    }

                    if let (Share::Helper(left), Share::Helper(right)) =
                        (&x.proof_share, &y.proof_share)
                    {
                        assert_ne!(left, right);
                    }

                    assert_ne!(x.joint_rand_param, y.joint_rand_param);
                }
            }
        }
    }

    fn test_prepare_state_serialization<T, A, P, const L: usize>(
        prio3: &Prio3<T, A, P, L>,
        measurement: &T::Measurement,
    ) -> Result<(), VdafError>
    where
        T: Type,
        A: Clone + Debug + Sync + Send,
        P: Prg<L>,
    {
        let mut verify_key = [0; L];
        thread_rng().fill(&mut verify_key[..]);
        let input_shares = prio3.shard(measurement)?;
        for (agg_id, input_share) in input_shares.iter().enumerate() {
            let (want, _msg) = prio3.prepare_init(&verify_key, agg_id, &(), &[], &input_share)?;
            let got =
                Prio3PrepareState::get_decoded_with_param(&(prio3, agg_id), &want.get_encoded())
                    .expect("failed to decode prepare step");
            assert_eq!(got, want);
        }
        Ok(())
    }
}
