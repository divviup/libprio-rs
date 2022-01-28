// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This modulde
//! implements the prio3 [VDAF]. The construction is based on a transform of a Fully Linear Proof
//! (FLP) system (i.e., a concrete [`Type`](crate::pcp::Type) into a zero-knowledge proof system on
//! distributed data as described in [[BBCG+19], Section 6].
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [BBCG+21]: https://ia.cr/2021/017
//! [VDAF]: https://datatracker.ietf.org/doc/draft-patton-cfrg-vdaf/

use crate::field::{Field64, Field96, FieldElement};
#[cfg(feature = "multithreaded")]
use crate::pcp::gadgets::ParallelSumMultithreaded;
use crate::pcp::gadgets::{BlindPolyEval, ParallelSum, ParallelSumGadget};
use crate::pcp::types::{Count, CountVec, Histogram, Sum};
use crate::pcp::Type;
use crate::prng::Prng;
use crate::vdaf::suite::{Key, KeyDeriver, KeyStream, Suite};
use crate::vdaf::{
    Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare, PrepareTransition,
    Share, Vdaf, VdafError,
};
use std::convert::{TryFrom, TryInto};
use std::fmt::Debug;
use std::io::{Read, Write};
use std::iter::IntoIterator;
use std::marker::PhantomData;

/// The count type. Each measurement is an integer in `[0,2)` and the aggregate is the sum.
pub type Prio3Count64 = Prio3<Count<Field64>, Prio3Result<u64>>;

impl Prio3Count64 {
    /// Construct an instance of this VDAF with the given suite and the given number of aggregators.
    pub fn new(suite: Suite, num_aggregators: u8) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        Ok(Prio3 {
            num_aggregators,
            suite,
            typ: Count::new(),
            phantom: PhantomData,
        })
    }
}

/// The count-vector type. Each measurement is a vector of integers in `[0,2)` and the aggregate is
/// the element-wise sum.
pub type Prio3CountVec64 =
    Prio3<CountVec<Field96, ParallelSum<Field96, BlindPolyEval<Field96>>>, Prio3ResultVec<u64>>;

/// Like [`Prio3CountVec64`] except this type uses multithreading to improve sharding and
/// preparation time. Note that the improvement is only noticeable for very large input lengths,
/// e.g., 200 and up. (Your system's mileage may vary.)
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3CountVec64Multithreaded = Prio3<
    CountVec<Field96, ParallelSumMultithreaded<Field96, BlindPolyEval<Field96>>>,
    Prio3ResultVec<u64>,
>;

impl<S> Prio3<CountVec<Field96, S>, Prio3ResultVec<u64>>
where
    S: 'static + ParallelSumGadget<Field96, BlindPolyEval<Field96>> + Eq,
{
    /// Construct an instance of this VDAF with the given suite and the given number of
    /// aggregators. `len` defines the length of each measurement.
    pub fn new(suite: Suite, num_aggregators: u8, len: usize) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        Ok(Prio3 {
            num_aggregators,
            suite,
            typ: CountVec::new(len),
            phantom: PhantomData,
        })
    }
}

/// The sum type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and the
/// aggregate is the sum.
pub type Prio3Sum64 = Prio3<Sum<Field96>, Prio3Result<u64>>;

impl Prio3Sum64 {
    /// Construct an instance of this VDAF with the given suite, number of aggregators and required
    /// bit length. The bit length must not exceed 64.
    pub fn new(suite: Suite, num_aggregators: u8, bits: u32) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({}) exceeds limit for aggregate type (64)",
                bits
            )));
        }

        Ok(Prio3 {
            num_aggregators,
            suite,
            typ: Sum::new(bits as usize)?,
            phantom: PhantomData,
        })
    }
}

/// the histogram type. Each measurement is an unsigned, 64-bit integer and the result is a
/// histogram representation of the measurement.
pub type Prio3Histogram64 = Prio3<Histogram<Field96>, Prio3ResultVec<u64>>;

impl Prio3Histogram64 {
    /// Constructs an instance of this VDAF with the given suite, number of aggregators, and
    /// desired histogram bucket boundaries.
    pub fn new(suite: Suite, num_aggregators: u8, buckets: &[u64]) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        let buckets = buckets.iter().map(|bucket| *bucket as u128).collect();

        Ok(Prio3 {
            num_aggregators,
            suite,
            typ: Histogram::<Field96>::new(buckets)?,
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
#[derive(Debug)]
pub struct Prio3<T: Type, A: Debug> {
    num_aggregators: u8,
    suite: Suite,
    typ: T,
    phantom: PhantomData<A>,
}

impl<T: Type, A: Debug> Clone for Prio3<T, A> {
    fn clone(&self) -> Self {
        Self {
            num_aggregators: self.num_aggregators,
            suite: self.suite,
            typ: self.typ.clone(),
            phantom: PhantomData,
        }
    }
}

impl<T: Type, A: Debug> Vdaf for Prio3<T, A> {
    type Measurement = T::Measurement;
    type AggregateResult = A;
    type AggregationParam = ();
    type PublicParam = ();
    type VerifyParam = Prio3VerifyParam;
    type OutputShare = OutputShare<T::Field>;
    type AggregateShare = AggregateShare<T::Field>;

    fn setup(&self) -> Result<((), Vec<Prio3VerifyParam>), VdafError> {
        let query_rand_init = Key::generate(self.suite)?;
        Ok((
            (),
            (0..self.num_aggregators)
                .map(|aggregator_id| Prio3VerifyParam {
                    query_rand_init: query_rand_init.clone(),
                    aggregator_id,
                })
                .collect(),
        ))
    }

    fn num_aggregators(&self) -> usize {
        self.num_aggregators as usize
    }
}

/// The verification parameter used by each aggregator to evaluate the VDAF.
#[derive(Clone, Debug)]
pub struct Prio3VerifyParam {
    /// Key used to derive the query randomness from the nonce.
    pub query_rand_init: Key,

    /// The identity of the aggregator.
    pub aggregator_id: u8,
}

/// The message sent by the client to each aggregator. This includes the client's input share and
/// the initial message of the input-validation protocol.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Prio3InputShare<F> {
    /// The input share.
    pub input_share: Share<F>,

    /// The proof share.
    pub proof_share: Share<F>,

    /// The sum of the joint randomness seed shares sent to the other aggregators.
    //
    // TODO(cjpatton) If `input.joint_rand_len() == 0`, then we don't need to bother with the joint
    // randomness seed at all. See https://github.com/cjpatton/vdaf/issues/15.
    pub joint_rand_seed_hint: Key,

    /// The blinding factor, used to derive the aggregator's joint randomness seed share.
    pub blind: Key,
}

impl<F: FieldElement> Prio3InputShare<F> {
    fn append<W: Write>(&self, writer: &mut W) -> Result<(), std::io::Error> {
        match (&self.input_share, &self.proof_share) {
            (Share::Leader(input_share_data), Share::Leader(proof_share_data)) => {
                for x in input_share_data {
                    writer.write_all(&(*x).into())?;
                }
                for x in proof_share_data {
                    writer.write_all(&(*x).into())?;
                }
            }
            (Share::Helper(input_share_seed), Share::Helper(proof_share_seed)) => {
                writer.write_all(input_share_seed.as_slice())?;
                writer.write_all(proof_share_seed.as_slice())?;
            }
            _ => panic!("tried to encode input share with ambiguous encoding"),
        };
        writer.write_all(self.joint_rand_seed_hint.as_slice())?;
        writer.write_all(self.blind.as_slice())?;
        Ok(())
    }

    fn read<R, A, T>(is_leader: bool, vdaf: &Prio3<T, A>, reader: &mut R) -> Result<Self, VdafError>
    where
        R: Read,
        A: Debug,
        T: Type<Field = F>,
    {
        let (input_share, proof_share) = if is_leader {
            (
                Share::read_leader(vdaf.typ.input_len(), reader)?,
                Share::read_leader(vdaf.typ.proof_len(), reader)?,
            )
        } else {
            (
                Share::read_helper(vdaf.suite, reader)?,
                Share::read_helper(vdaf.suite, reader)?,
            )
        };

        let mut joint_rand_seed_hint = Key::uninitialized(vdaf.suite);
        let mut blind = Key::uninitialized(vdaf.suite);
        reader.read_exact(joint_rand_seed_hint.as_mut_slice())?;
        reader.read_exact(blind.as_mut_slice())?;

        Ok(Prio3InputShare {
            input_share,
            proof_share,
            joint_rand_seed_hint,
            blind,
        })
    }
}

#[derive(Clone, Debug)]
/// The verification message emitted by each aggregator during the Prepare process.
struct Prio3PrepareMessage<F> {
    /// (A share of) the FLP verifier message. (See [`Type`](crate::pcp::Type).)
    pub verifier: Vec<F>,

    /// (A share of) the joint randomness seed.
    //
    // TODO(cjpatton) If `input.joint_rand_len() == 0`, then we don't need to bother with the joint
    // randomness seed at all. See https://github.com/cjpatton/vdaf/issues/15.
    pub joint_rand_seed: Key,
}

impl<F: FieldElement> Prio3PrepareMessage<F> {
    fn append<W: Write>(&self, writer: &mut W) -> Result<(), std::io::Error> {
        for x in self.verifier.iter() {
            writer.write_all(&(*x).into())?;
        }
        writer.write_all(self.joint_rand_seed.as_slice())?;
        Ok(())
    }

    fn read<R, A, T>(vdaf: &Prio3<T, A>, reader: &mut R) -> Result<Self, VdafError>
    where
        R: Read,
        A: Debug,
        T: Type<Field = F>,
    {
        let verifier_len = vdaf.typ.verifier_len();
        let mut verifier = Vec::with_capacity(verifier_len);
        for _ in 0..verifier_len {
            verifier.push(T::Field::try_from_reader(reader)?);
        }

        let mut joint_rand_seed = Key::uninitialized(vdaf.suite);
        reader.read_exact(joint_rand_seed.as_mut_slice())?;

        Ok(Prio3PrepareMessage {
            verifier,
            joint_rand_seed,
        })
    }
}

impl<T: Type, A: Debug> Client for Prio3<T, A> {
    fn shard(
        &self,
        _public_param: &(),
        measurement: &T::Measurement,
    ) -> Result<Vec<Vec<u8>>, VdafError> {
        let num_aggregators = self.num_aggregators;
        let input = self.typ.encode(measurement)?;

        // Generate the input shares and compute the joint randomness.
        let mut helper_shares = Vec::with_capacity(num_aggregators as usize - 1);
        let mut leader_input_share = input.clone();
        let mut joint_rand_seed = Key::uninitialized(self.suite);
        for aggregator_id in 1..num_aggregators {
            let mut helper = HelperShare::new(self.suite)?;

            let mut deriver = KeyDeriver::from_key(&helper.blind);
            deriver.update(&[aggregator_id]);
            let prng: Prng<T::Field> =
                Prng::from_key_stream(KeyStream::from_key(&helper.input_share));
            for (x, y) in leader_input_share
                .iter_mut()
                .zip(prng)
                .take(self.typ.input_len())
            {
                *x -= y;
                deriver.update(&y.into());
            }

            helper.joint_rand_seed_hint = deriver.finish();
            for (x, y) in joint_rand_seed
                .as_mut_slice()
                .iter_mut()
                .zip(helper.joint_rand_seed_hint.as_slice().iter())
            {
                *x ^= y;
            }

            helper_shares.push(helper);
        }

        let leader_blind = Key::generate(self.suite)?;

        let mut deriver = KeyDeriver::from_key(&leader_blind);
        deriver.update(&[0]); // ID of the leader
        for x in leader_input_share.iter() {
            deriver.update(&(*x).into());
        }

        let mut leader_joint_rand_seed_hint = deriver.finish();
        for (x, y) in joint_rand_seed
            .as_mut_slice()
            .iter_mut()
            .zip(leader_joint_rand_seed_hint.as_slice().iter())
        {
            *x ^= y;
        }

        // Run the proof-generation algorithm.
        let prng: Prng<T::Field> = Prng::from_key_stream(KeyStream::from_key(&joint_rand_seed));
        let joint_rand: Vec<T::Field> = prng.take(self.typ.joint_rand_len()).collect();
        let prng: Prng<T::Field> = Prng::generate(self.suite)?;
        let prove_rand: Vec<T::Field> = prng.take(self.typ.prove_rand_len()).collect();
        let mut leader_proof_share = self.typ.prove(&input, &prove_rand, &joint_rand)?;

        // Generate the proof shares and finalize the joint randomness seed hints.
        for helper in helper_shares.iter_mut() {
            let prng: Prng<T::Field> =
                Prng::from_key_stream(KeyStream::from_key(&helper.proof_share));
            for (x, y) in leader_proof_share
                .iter_mut()
                .zip(prng)
                .take(self.typ.proof_len())
            {
                *x -= y;
            }

            for (x, y) in helper
                .joint_rand_seed_hint
                .as_mut_slice()
                .iter_mut()
                .zip(joint_rand_seed.as_slice().iter())
            {
                *x ^= y;
            }
        }

        for (x, y) in leader_joint_rand_seed_hint
            .as_mut_slice()
            .iter_mut()
            .zip(joint_rand_seed.as_slice().iter())
        {
            *x ^= y;
        }

        let mut input_shares = Vec::with_capacity(num_aggregators as usize);
        let base_len = leader_joint_rand_seed_hint.size() + leader_blind.size();

        let mut encoded = Vec::with_capacity(
            base_len + T::Field::ENCODED_SIZE * (self.typ.input_len() + self.typ.proof_len()),
        );
        Prio3InputShare {
            input_share: Share::Leader(leader_input_share),
            proof_share: Share::Leader(leader_proof_share),
            joint_rand_seed_hint: leader_joint_rand_seed_hint,
            blind: leader_blind,
        }
        .append(&mut encoded)?;
        input_shares.push(encoded);

        for helper in helper_shares.into_iter() {
            let mut encoded = Vec::with_capacity(
                base_len + helper.input_share.size() + helper.proof_share.size(),
            );
            Prio3InputShare::<T::Field> {
                input_share: Share::Helper(helper.input_share),
                proof_share: Share::Helper(helper.proof_share),
                joint_rand_seed_hint: helper.joint_rand_seed_hint,
                blind: helper.blind,
            }
            .append(&mut encoded)?;
            input_shares.push(encoded);
        }

        Ok(input_shares)
    }
}

/// State of each aggregator during the Prepare process.
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub enum Prio3PrepareStep<F> {
    /// Ready to send the verifier message.
    Ready {
        input_share: Share<F>,
        joint_rand_seed: Key,
        verifier_msg: Vec<u8>,
    },
    /// Waiting for the set of verifier messages.
    Waiting {
        input_share: Share<F>,
        joint_rand_seed: Key,
    },
}

impl<T: Type, A: Debug> Aggregator for Prio3<T, A> {
    type PrepareStep = Prio3PrepareStep<T::Field>;

    /// Begins the Prep process with the other aggregators. The result of this process is
    /// the aggregator's output share.
    //
    // TODO(cjpatton) Check that the input share matches `self.suite`.
    fn prepare_init(
        &self,
        verify_param: &Prio3VerifyParam,
        _agg_param: &(),
        nonce: &[u8],
        input_share: &[u8],
    ) -> Result<Prio3PrepareStep<T::Field>, VdafError> {
        let mut deriver = KeyDeriver::from_key(&verify_param.query_rand_init);
        deriver.update(&[255]);
        deriver.update(nonce);
        let query_rand_seed = deriver.finish();

        let mut reader = std::io::Cursor::new(input_share);
        let msg = Prio3InputShare::read(verify_param.aggregator_id == 0, self, &mut reader)?;
        if reader.position() as usize != input_share.len() {
            return Err(VdafError::Uncategorized(
                "unparsed data in input buffer".to_string(),
            ));
        }

        // Create a reference to the (expanded) input share.
        let expanded_input_share: Option<Vec<T::Field>> = match msg.input_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => {
                let prng = Prng::from_key_stream(KeyStream::from_key(seed));
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
                let prng = Prng::from_key_stream(KeyStream::from_key(seed));
                Some(prng.take(self.typ.proof_len()).collect())
            }
        };
        let proof_share = match msg.proof_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_proof_share.as_ref().unwrap(),
        };

        // Compute the joint randomness.
        let mut deriver = KeyDeriver::from_key(&msg.blind);
        deriver.update(&[verify_param.aggregator_id]);
        for x in input_share {
            deriver.update(&(*x).into());
        }
        let joint_rand_seed_share = deriver.finish();

        let mut joint_rand_seed = Key::uninitialized(query_rand_seed.suite());
        for (j, x) in joint_rand_seed.as_mut_slice().iter_mut().enumerate() {
            *x = msg.joint_rand_seed_hint.as_slice()[j] ^ joint_rand_seed_share.as_slice()[j];
        }

        let prng: Prng<T::Field> = Prng::from_key_stream(KeyStream::from_key(&joint_rand_seed));
        let joint_rand: Vec<T::Field> = prng.take(self.typ.joint_rand_len()).collect();

        // Compute the query randomness.
        let prng: Prng<T::Field> = Prng::from_key_stream(KeyStream::from_key(&query_rand_seed));
        let query_rand: Vec<T::Field> = prng.take(self.typ.query_rand_len()).collect();

        // Run the query-generation algorithm.
        let verifier_share = self.typ.query(
            input_share,
            proof_share,
            &query_rand,
            &joint_rand,
            self.num_aggregators as usize,
        )?;

        let mut verifier_msg = Vec::with_capacity(
            T::Field::ENCODED_SIZE * self.typ.verifier_len() + joint_rand_seed.size(),
        );
        Prio3PrepareMessage {
            verifier: verifier_share,
            joint_rand_seed: joint_rand_seed_share,
        }
        .append(&mut verifier_msg)?;

        Ok(Prio3PrepareStep::Ready {
            input_share: msg.input_share.clone(),
            joint_rand_seed,
            verifier_msg,
        })
    }

    fn prepare_preprocess<M: AsRef<[u8]>, I: IntoIterator<Item = M>>(
        &self,
        inputs: I,
    ) -> Result<Vec<u8>, VdafError> {
        let mut verifier = vec![T::Field::zero(); self.typ.verifier_len()];
        let mut joint_rand_seed = Key::uninitialized(self.suite);
        let mut count = 0;
        for input in inputs.into_iter() {
            count += 1;

            let mut reader = std::io::Cursor::new(&input);
            let share = Prio3PrepareMessage::read(self, &mut reader)?;
            if reader.position() as usize != input.as_ref().len() {
                return Err(VdafError::Uncategorized(
                    "unparsed data in buffer".to_string(),
                ));
            }

            if share.verifier.len() != verifier.len() {
                return Err(VdafError::Uncategorized(format!(
                    "unexpected verifier share length: got {}; want {}",
                    share.verifier.len(),
                    verifier.len(),
                )));
            }

            if share.joint_rand_seed.suite() != self.suite {
                return Err(VdafError::Uncategorized(format!(
                    "unexpected suite for joint randomness seed share: got {:?}; want {:?}",
                    share.joint_rand_seed.suite(),
                    self.suite,
                )));
            }

            for (x, y) in verifier.iter_mut().zip(share.verifier) {
                *x += y;
            }

            for (x, y) in joint_rand_seed
                .as_mut_slice()
                .iter_mut()
                .zip(share.joint_rand_seed.as_slice())
            {
                *x ^= y;
            }
        }

        if count != self.num_aggregators {
            return Err(VdafError::Uncategorized(format!(
                "unexpected message count: got {}; want {}",
                count, self.num_aggregators,
            )));
        }

        let mut output = Vec::with_capacity(
            T::Field::ENCODED_SIZE * self.typ.verifier_len() + joint_rand_seed.size(),
        );
        Prio3PrepareMessage {
            verifier,
            joint_rand_seed,
        }
        .append(&mut output)?;

        Ok(output)
    }

    // TODO Fix this clippy warning instead of bypassing it.
    #[allow(clippy::type_complexity)]
    fn prepare_step<M: AsRef<[u8]>>(
        &self,
        state: Prio3PrepareStep<T::Field>,
        input: Option<M>,
    ) -> PrepareTransition<Prio3PrepareStep<T::Field>, OutputShare<T::Field>> {
        match (state, input) {
            (
                Prio3PrepareStep::Ready {
                    input_share,
                    joint_rand_seed,
                    verifier_msg,
                },
                None,
            ) => PrepareTransition::Continue(
                Prio3PrepareStep::Waiting {
                    input_share,
                    joint_rand_seed,
                },
                verifier_msg,
            ),
            (
                Prio3PrepareStep::Waiting {
                    input_share,
                    joint_rand_seed,
                },
                Some(encoded),
            ) => {
                let mut reader = std::io::Cursor::new(&encoded);
                let msg = Prio3PrepareMessage::read(self, &mut reader);
                if msg.is_err() {
                    return PrepareTransition::Fail(msg.unwrap_err());
                }
                if reader.position() as usize != encoded.as_ref().len() {
                    return PrepareTransition::Fail(VdafError::Uncategorized(
                        "unparsed data in input buffer".to_string(),
                    ));
                }
                let msg = msg.unwrap();

                // Check that the joint randomness was correct.
                if joint_rand_seed != msg.joint_rand_seed {
                    return PrepareTransition::Fail(VdafError::Uncategorized(
                        "joint randomness mismatch".to_string(),
                    ));
                }

                // Check the proof.
                let res = match self.typ.decide(&msg.verifier) {
                    Ok(res) => res,
                    Err(err) => {
                        return PrepareTransition::Fail(VdafError::from(err));
                    }
                };

                if !res {
                    return PrepareTransition::Fail(VdafError::Uncategorized(
                        "proof check failed".to_string(),
                    ));
                }

                // Compute the output share.
                let input_share = match input_share {
                    Share::Leader(data) => data,
                    Share::Helper(seed) => {
                        let prng = Prng::from_key_stream(KeyStream::from_key(&seed));
                        prng.take(self.typ.input_len()).collect()
                    }
                };

                let output_share = match self.typ.truncate(input_share) {
                    Ok(data) => OutputShare(data),
                    Err(err) => {
                        return PrepareTransition::Fail(VdafError::from(err));
                    }
                };

                PrepareTransition::Finish(output_share)
            }
            _ => PrepareTransition::Fail(VdafError::Uncategorized(
                "invalid state transition".to_string(),
            )),
        }
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

impl<T, A> Collector for Prio3<T, A>
where
    T: Type,
    A: TryFrom<AggregateShare<T::Field>, Error = VdafError> + Eq + Debug,
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

#[derive(Clone)]
struct HelperShare {
    input_share: Key,
    proof_share: Key,
    joint_rand_seed_hint: Key,
    blind: Key,
}

impl HelperShare {
    fn new(suite: Suite) -> Result<Self, VdafError> {
        Ok(HelperShare {
            input_share: Key::generate(suite)?,
            proof_share: Key::generate(suite)?,
            joint_rand_seed_hint: Key::uninitialized(suite),
            blind: Key::generate(suite)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prio3_count64() {
        let prio3 = Prio3Count64::new(Suite::Blake3, 23).unwrap();
        let (_, verify_params) = prio3.setup().unwrap();
        let nonce = b"This is a good nonce.";
        let input_shares = prio3.shard(&(), &1).unwrap();
        eval_vdaf(
            &prio3,
            &input_shares,
            &verify_params,
            nonce,
            Some(Prio3Result(1)),
        )
        .unwrap();

        // TODO(cjpatton) Add failure test cases.
    }

    #[test]
    fn test_prio3_input_share() {
        let prio3 = Prio3Count64::new(Suite::Blake3, 5).unwrap();
        let input_shares: Vec<_> = prio3
            .shard(&(), &1)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, encoded)| {
                let mut reader = std::io::Cursor::new(encoded);
                Prio3InputShare::read(i == 0, &prio3, &mut reader).unwrap()
            })
            .collect();

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

                    assert_ne!(x.joint_rand_seed_hint, y.joint_rand_seed_hint);
                    assert_ne!(x.blind, y.blind);
                }
            }
        }
    }

    // Execute the VDAF end-to-end on a single user measurement.
    fn eval_vdaf<T, A>(
        prio3: &Prio3<T, A>,
        input_shares: &Vec<Vec<u8>>,
        verify_params: &[Prio3VerifyParam],
        nonce: &[u8],
        expected_agg: Option<A>,
    ) -> Result<(), VdafError>
    where
        T: Type,
        A: TryFrom<AggregateShare<T::Field>, Error = VdafError> + Eq + Debug,
    {
        let mut state0: Vec<Prio3PrepareStep<T::Field>> =
            Vec::with_capacity(prio3.num_aggregators());
        for (verify_param, input_share) in verify_params.iter().zip(input_shares.iter()) {
            let state = prio3.prepare_init(verify_param, &(), nonce, input_share)?;
            state0.push(state);
        }

        let mut round1 = Vec::with_capacity(prio3.num_aggregators());
        let mut state1: Vec<Prio3PrepareStep<T::Field>> =
            Vec::with_capacity(prio3.num_aggregators());
        for state in state0.into_iter() {
            let (state, msg) = prio3.prepare_start(state)?;
            state1.push(state);
            round1.push(msg);
        }

        // Preprocess the prepare messages.
        let round1_preprocessed = prio3.prepare_preprocess(round1)?;

        let mut agg_shares: Vec<AggregateShare<T::Field>> =
            Vec::with_capacity(prio3.num_aggregators());
        for state in state1.into_iter() {
            let output_share = prio3.prepare_finish(state, &round1_preprocessed)?;
            agg_shares.push(prio3.aggregate(&(), [output_share])?);
        }

        if let Some(want) = expected_agg {
            let got = prio3.unshard(&(), agg_shares)?;
            if got != want {
                return Err(VdafError::Uncategorized(
                    "unexpected output for test case".to_string(),
                ));
            }
        } else {
            return Err(VdafError::Uncategorized(
                "test case expected no output, got some".to_string(),
            ));
        }

        Ok(())
    }
}
