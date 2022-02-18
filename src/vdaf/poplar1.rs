// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module
//! implemets the poplar1 [VDAF]. This is the core component of the Poplar protocol for
//! privacy-preserving heavy hitters [[BBCG+21]].
//!
//! TODO Make the input shares stateful so that applications can efficiently evaluate the IDPF over
//! multiple rounds. Question: Will this require API changes to [`crate::vdaf::Vdaf`]?
//!
//! TODO Update trait [`Idpf`] so thtat the IPDF can have a different field type at the leaves than
//! at the inner nodes.
//!
//! TODO Implement the efficient IDPF of [[BBCG+21]]. [`ToyIdpf`] is not space efficient and is
//! merely intended as a proof-of-concept.
//!
//! [VDAF]: https://cjpatton.github.io/vdaf/draft-patton-cfrg-vdaf.html
//! [BBCG+21]: https://eprint.iacr.org/2021/017

use std::array::IntoIter;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::{TryFrom, TryInto};
use std::fmt::Debug;
use std::io::Cursor;
use std::iter::FromIterator;
use std::marker::PhantomData;

use crate::codec::{
    decode_u16_items, decode_u24_items, encode_u16_items, encode_u24_items, CodecError, Decode,
    Encode,
};
use crate::field::{split_vector, FieldElement};
use crate::fp::log2;
use crate::prng::Prng;
use crate::vdaf::suite::{Key, KeyDeriver, KeyStream, Suite};
use crate::vdaf::{
    Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare, PrepareTransition,
    Share, ShareDecodingParameter, Vdaf, VdafError,
};

/// An input for an IDPF ([`Idpf`]).
///
/// TODO Make this an associated type of `Ipdf`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IdpfInput {
    index: usize,
    level: usize,
}

impl IdpfInput {
    /// Constructs an IDPF input using the first `level` bits of `data`.
    pub fn new(data: &[u8], level: usize) -> Result<Self, VdafError> {
        if level > data.len() << 3 {
            return Err(VdafError::Uncategorized(format!(
                "desired bit length ({} bits) exceeds data length ({} bytes)",
                level,
                data.len()
            )));
        }

        let mut index = 0;
        let mut i = 0;
        for byte in data {
            for j in 0..8 {
                let bit = (byte >> j) & 1;
                if i < level {
                    index |= (bit as usize) << i;
                }
                i += 1;
            }
        }

        Ok(Self { index, level })
    }

    /// Construct a new input that is a prefix of `self`. Bounds checking is performed by the
    /// caller.
    fn prefix(&self, level: usize) -> Self {
        let index = self.index & ((1 << level) - 1);
        Self { index, level }
    }

    /// Return the position of `self` in the look-up table of `ToyIdpf`.
    fn data_index(&self) -> usize {
        self.index | (1 << self.level)
    }
}

impl Ord for IdpfInput {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.level.cmp(&other.level) {
            Ordering::Equal => self.index.cmp(&other.index),
            ord => ord,
        }
    }
}

impl PartialOrd for IdpfInput {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Encode for IdpfInput {
    fn encode(&self, bytes: &mut Vec<u8>) {
        (self.index as u64).encode(bytes);
        (self.level as u64).encode(bytes);
    }
}

impl Decode<()> for IdpfInput {
    fn decode(_decoding_parameter: &(), bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let index = u64::decode(&(), bytes)? as usize;
        let level = u64::decode(&(), bytes)? as usize;

        Ok(Self { index, level })
    }
}

/// An Incremental Distributed Point Function (IDPF), as defined by [[BBCG+21]].
///
/// [BBCG+21]: https://eprint.iacr.org/2021/017
//
// NOTE(cjpatton) The real IDPF API probably needs to be stateful.
pub trait Idpf<const KEY_LEN: usize, const OUT_LEN: usize>:
    Sized + Clone + Debug + Encode + Decode<()>
{
    /// The finite field over which the IDPF is defined.
    //
    // NOTE(cjpatton) The IDPF of [BBCG+21] might use different fields for different levels of the
    // prefix tree.
    type Field: FieldElement;

    /// Generate and return a sequence of IDPF shares for `input`. Parameter `output` is an
    /// iterator that is invoked to get the output value for each successive level of the prefix
    /// tree.
    fn gen<M: IntoIterator<Item = [Self::Field; OUT_LEN]>>(
        input: &IdpfInput,
        values: M,
    ) -> Result<[Self; KEY_LEN], VdafError>;

    /// Evaluate an IDPF share on `prefix`.
    fn eval(&self, prefix: &IdpfInput) -> Result<[Self::Field; OUT_LEN], VdafError>;
}

/// A "toy" IPDF used for demonstration purposes. The space consumed by each share is `O(2^n)`,
/// where `n` is the length of the input. The size of each share is restricted to 1MB, so this IDPF
/// is only suitable for very short inputs.
//
// NOTE(cjpatton) It would be straight-forward to generalize this construction to any `KEY_LEN` and
// `OUT_LEN`.
#[derive(Debug, Clone)]
pub struct ToyIdpf<F> {
    data0: Vec<F>,
    data1: Vec<F>,
    level: usize,
}

impl<F: FieldElement> Idpf<2, 2> for ToyIdpf<F> {
    type Field = F;

    fn gen<M: IntoIterator<Item = [Self::Field; 2]>>(
        input: &IdpfInput,
        values: M,
    ) -> Result<[Self; 2], VdafError> {
        const MAX_DATA_BYTES: usize = 1024 * 1024; // 1MB

        let max_input_len =
            usize::try_from(log2((MAX_DATA_BYTES / F::ENCODED_SIZE) as u128)).unwrap();
        if input.level > max_input_len {
            return Err(VdafError::Uncategorized(format!(
                "input length ({}) exceeds maximum of ({})",
                input.level, max_input_len
            )));
        }

        let data_len = 1 << (input.level + 1);
        let mut data0 = vec![F::zero(); data_len];
        let mut data1 = vec![F::zero(); data_len];
        let mut values = values.into_iter();
        for level in 0..input.level + 1 {
            let value = values.next().unwrap();
            let index = input.prefix(level).data_index();
            data0[index] = value[0];
            data1[index] = value[1];
        }

        let mut data0 = split_vector(&data0, 2)?.into_iter();
        let mut data1 = split_vector(&data1, 2)?.into_iter();
        Ok([
            ToyIdpf {
                data0: data0.next().unwrap(),
                data1: data1.next().unwrap(),
                level: input.level,
            },
            ToyIdpf {
                data0: data0.next().unwrap(),
                data1: data1.next().unwrap(),
                level: input.level,
            },
        ])
    }

    fn eval(&self, prefix: &IdpfInput) -> Result<[F; 2], VdafError> {
        if prefix.level > self.level {
            return Err(VdafError::Uncategorized(format!(
                "prefix length ({}) exceeds input length ({})",
                prefix.level, self.level
            )));
        }

        let index = prefix.data_index();
        Ok([self.data0[index], self.data1[index]])
    }
}

impl<F: FieldElement> Encode for ToyIdpf<F> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        encode_u24_items(bytes, &self.data0);
        encode_u24_items(bytes, &self.data1);
        (self.level as u64).encode(bytes);
    }
}

impl<F: FieldElement> Decode<()> for ToyIdpf<F> {
    fn decode(_decoding_parameter: &(), bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let data0 = decode_u24_items(&(), bytes)?;
        let data1 = decode_u24_items(&(), bytes)?;
        let level = u64::decode(&(), bytes)? as usize;

        Ok(Self {
            data0,
            data1,
            level,
        })
    }
}

impl Encode for BTreeSet<IdpfInput> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        // Encodes the aggregation parameter as a variable length vector of
        // [`IdpfInput`], because the size of the aggregation parameter is not
        // determined by the VDAF.
        let items: Vec<IdpfInput> = self.iter().map(IdpfInput::clone).collect();
        encode_u24_items(bytes, &items);
    }
}

impl Decode<()> for BTreeSet<IdpfInput> {
    fn decode(_decoding_parameter: &(), bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let inputs = decode_u24_items(&(), bytes)?;
        Ok(Self::from_iter(inputs.into_iter()))
    }
}

/// An input share for the `poplar1` VDAF.
#[derive(Debug, Clone)]
pub struct Poplar1InputShare<I: Idpf<2, 2>> {
    /// IDPF share of input
    pub idpf: I,

    /// PRNG seed used to generate the aggregator's share of the randomness used in the first part
    /// of the sketching protocol.
    pub sketch_start_seed: Key,

    /// Aggregator's share of the randomness used in the second part of the sketching protocol.
    pub sketch_next: Share<I::Field>,
}

impl<I: Idpf<2, 2>> Encode for Poplar1InputShare<I> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        self.idpf.encode(bytes);
        self.sketch_start_seed.encode(bytes);
        self.sketch_next.encode(bytes);
    }
}

impl<I: Idpf<2, 2>> Decode<Poplar1VerifyParam> for Poplar1InputShare<I> {
    fn decode(
        decoding_parameter: &Poplar1VerifyParam,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let idpf = I::decode(&(), bytes)?;
        let sketch_start_seed = Key::decode(&decoding_parameter.rand_init.suite(), bytes)?;

        let share_decoding_parameter = if decoding_parameter.is_leader {
            // TODO: I think this is correct for `Idpf<2, 2>`
            ShareDecodingParameter::Leader(2)
        } else {
            ShareDecodingParameter::Helper(decoding_parameter.rand_init.suite())
        };

        let sketch_next = <Share<I::Field>>::decode(&share_decoding_parameter, bytes)?;

        Ok(Self {
            idpf,
            sketch_start_seed,
            sketch_next,
        })
    }
}

/// The poplar1 VDAF.
#[derive(Debug)]
pub struct Poplar1<I> {
    suite: Suite,
    phantom: PhantomData<I>,
}

impl<I> Poplar1<I> {
    /// Create an instance of the poplar1 VDAF. The caller provides a cipher suite `suite` used for
    /// deriving pseudorandom sequences of field elements.
    pub fn new(suite: Suite) -> Self {
        Self {
            suite,
            phantom: PhantomData,
        }
    }
}

impl<I> Clone for Poplar1<I> {
    fn clone(&self) -> Self {
        Self::new(self.suite)
    }
}

impl<I: Idpf<2, 2>> Vdaf for Poplar1<I> {
    type Measurement = IdpfInput;
    type AggregateResult = BTreeMap<IdpfInput, u64>;
    type AggregationParam = BTreeSet<IdpfInput>;
    type PublicParam = ();
    type VerifyParam = Poplar1VerifyParam;
    type InputShare = Poplar1InputShare<I>;
    type OutputShare = OutputShare<I::Field>;
    type AggregateShare = AggregateShare<I::Field>;

    fn setup(&self) -> Result<((), Vec<Poplar1VerifyParam>), VdafError> {
        let verify_rand_init = Key::generate(self.suite)?;
        Ok((
            (),
            vec![
                Poplar1VerifyParam::new(&verify_rand_init, true),
                Poplar1VerifyParam::new(&verify_rand_init, false),
            ],
        ))
    }

    fn num_aggregators(&self) -> usize {
        2
    }
}

impl<I: Idpf<2, 2>> Client for Poplar1<I> {
    #[allow(clippy::many_single_char_names)]
    fn shard(
        &self,
        _public_param: &(),
        input: &IdpfInput,
    ) -> Result<Vec<Poplar1InputShare<I>>, VdafError> {
        let idpf_values: Vec<[I::Field; 2]> = Prng::generate(self.suite)?
            .take(input.level + 1)
            .map(|k| [I::Field::one(), k])
            .collect();

        // For each level of the prefix tree, generate correlated randomness that the aggregators use
        // to validate the output. See [BBCG+21, Appendix C.4].
        let leader_sketch_start_seed = Key::generate(self.suite)?;
        let helper_sketch_start_seed = Key::generate(self.suite)?;
        let helper_sketch_next_seed = Key::generate(self.suite)?;
        let mut leader_sketch_start_prng: Prng<I::Field> =
            Prng::from_key_stream(KeyStream::from_key(&leader_sketch_start_seed));
        let mut helper_sketch_start_prng: Prng<I::Field> =
            Prng::from_key_stream(KeyStream::from_key(&helper_sketch_start_seed));
        let mut helper_sketch_next_prng: Prng<I::Field> =
            Prng::from_key_stream(KeyStream::from_key(&helper_sketch_next_seed));
        let mut leader_sketch_next: Vec<I::Field> = Vec::with_capacity(2 * input.level);
        for value in idpf_values.iter() {
            let k = value[1];

            // [BBCG+21, Appendix C.4]
            //
            // $(a, b, c)$
            let a = leader_sketch_start_prng.get() + helper_sketch_start_prng.get();
            let b = leader_sketch_start_prng.get() + helper_sketch_start_prng.get();
            let c = leader_sketch_start_prng.get() + helper_sketch_start_prng.get();

            // $A = -2a + k$
            // $B = a^2 + b + -ak + c$
            let d = k - (a + a);
            let e = (a * a) + b - (a * k) + c;
            leader_sketch_next.push(d - helper_sketch_next_prng.get());
            leader_sketch_next.push(e - helper_sketch_next_prng.get());
        }

        // Generate IDPF shares of the data and authentication vectors.
        let mut idpf_shares = IntoIter::new(I::gen(input, idpf_values)?);

        Ok(vec![
            Poplar1InputShare {
                idpf: idpf_shares.next().unwrap(),
                sketch_start_seed: leader_sketch_start_seed,
                sketch_next: Share::Leader(leader_sketch_next),
            },
            Poplar1InputShare {
                idpf: idpf_shares.next().unwrap(),
                sketch_start_seed: helper_sketch_start_seed,
                sketch_next: Share::Helper(helper_sketch_next_seed),
            },
        ])
    }
}

/// The verification parameter used by the aggregators to evaluate the VDAF on a distributed input.
#[derive(Clone, Debug)]
pub struct Poplar1VerifyParam {
    /// Key used to derive the verification randomness from the nonce.
    rand_init: Key,

    /// Indicates whether this Aggregator is the leader.
    is_leader: bool,
}

impl Poplar1VerifyParam {
    /// Construct a new verification parameter.
    pub fn new(key: &Key, is_leader: bool) -> Self {
        Self {
            rand_init: key.clone(),
            is_leader,
        }
    }
}

fn get_level(agg_param: &BTreeSet<IdpfInput>) -> Result<usize, VdafError> {
    let mut level = None;
    for prefix in agg_param {
        if let Some(l) = level {
            if prefix.level != l {
                return Err(VdafError::Uncategorized(
                    "prefixes must all have the same length".to_string(),
                ));
            }
        } else {
            level = Some(prefix.level);
        }
    }

    match level {
        Some(level) => Ok(level),
        None => Err(VdafError::Uncategorized("prefix set is empty".to_string())),
    }
}

impl<I: Idpf<2, 2>> Aggregator for Poplar1<I> {
    type PrepareStep = Poplar1PrepareStep<I::Field>;
    type PrepareMessage = Poplar1PrepareMessage<I::Field>;

    fn prepare_init(
        &self,
        verify_param: &Poplar1VerifyParam,
        agg_param: &BTreeSet<IdpfInput>,
        nonce: &[u8],
        input_share: &Self::InputShare,
    ) -> Result<Poplar1PrepareStep<I::Field>, VdafError> {
        let level = get_level(agg_param)?;

        // Derive the verification randomness.
        let mut deriver = KeyDeriver::from_key(&verify_param.rand_init);
        deriver.update(nonce);
        let verify_rand_seed = deriver.finish();
        let mut verify_rand_prng: Prng<I::Field> =
            Prng::from_key_stream(KeyStream::from_key(&verify_rand_seed));

        // Evaluate the IDPF shares and compute the polynomial coefficients.
        let mut z = [I::Field::zero(); 3];
        let mut output_share = Vec::with_capacity(agg_param.len());
        for prefix in agg_param.iter() {
            let value = input_share.idpf.eval(prefix)?;
            let (v, k) = (value[0], value[1]);
            let r = verify_rand_prng.get();

            // [BBCG+21, Appendix C.4]
            //
            // $(z_\sigma, z^*_\sigma, z^{**}_\sigma)$
            let tmp = r * v;
            z[0] += tmp;
            z[1] += r * tmp;
            z[2] += r * k;
            output_share.push(v);
        }

        // [BBCG+21, Appendix C.4]
        //
        // Add blind shares $(a_\sigma b_\sigma, c_\sigma)$
        //
        // NOTE(cjpatton) We can make this faster by a factor of 3 by using three seed shares instead
        // of one. On the other hand, if the input shares are made stateful, then we could store
        // the PRNG state theire and avoid fast-forwarding.
        let mut prng =
            Prng::<I::Field>::from_key_stream(KeyStream::from_key(&input_share.sketch_start_seed))
                .skip(3 * level);
        z[0] += prng.next().unwrap();
        z[1] += prng.next().unwrap();
        z[2] += prng.next().unwrap();

        let (d, e) = match &input_share.sketch_next {
            Share::Leader(data) => (data[2 * level], data[2 * level + 1]),
            Share::Helper(seed) => {
                let mut prng =
                    Prng::<I::Field>::from_key_stream(KeyStream::from_key(seed)).skip(2 * level);
                (prng.next().unwrap(), prng.next().unwrap())
            }
        };

        let x = if verify_param.is_leader {
            I::Field::one()
        } else {
            I::Field::zero()
        };

        Ok(Poplar1PrepareStep {
            sketch: SketchState::Ready,
            output_share: OutputShare(output_share),
            z,
            d,
            e,
            x,
        })
    }

    fn prepare_preprocess<M: IntoIterator<Item = Poplar1PrepareMessage<I::Field>>>(
        &self,
        inputs: M,
    ) -> Result<Poplar1PrepareMessage<I::Field>, VdafError> {
        let mut output: Option<Vec<I::Field>> = None;
        let mut count = 0;
        for data_share in inputs.into_iter() {
            count += 1;
            if let Some(ref mut data) = output {
                if data_share.0.len() != data.len() {
                    return Err(VdafError::Uncategorized(format!(
                        "unexpected message length: got {}; want {}",
                        data_share.0.len(),
                        data.len(),
                    )));
                }

                for (x, y) in data.iter_mut().zip(data_share.0.iter()) {
                    *x += *y;
                }
            } else {
                output = Some(data_share.0);
            }
        }

        if count != 2 {
            return Err(VdafError::Uncategorized(format!(
                "unexpected message count: got {}; want 2",
                count,
            )));
        }

        Ok(Poplar1PrepareMessage(output.unwrap()))
    }

    // TODO Fix this clippy warning instead of bypassing it.
    #[allow(clippy::type_complexity)]
    fn prepare_step(
        &self,
        mut state: Poplar1PrepareStep<I::Field>,
        input: Option<Poplar1PrepareMessage<I::Field>>,
    ) -> PrepareTransition<
        Poplar1PrepareStep<I::Field>,
        Poplar1PrepareMessage<I::Field>,
        OutputShare<I::Field>,
    > {
        match (&state.sketch, input) {
            (SketchState::Ready, None) => {
                let z_share = state.z.to_vec();
                state.sketch = SketchState::RoundOne;
                PrepareTransition::Continue(state, Poplar1PrepareMessage(z_share))
            }

            (SketchState::RoundOne, Some(msg)) => {
                if msg.0.len() != 3 {
                    return PrepareTransition::Fail(VdafError::Uncategorized(format!(
                        "unexpected message length ({:?}): got {}; want 3",
                        state.sketch,
                        msg.0.len(),
                    )));
                }

                // Compute polynomial coefficients.
                let z: [I::Field; 3] = msg.0.try_into().unwrap();
                let y_share =
                    vec![(state.d * z[0]) + state.e + state.x * ((z[0] * z[0]) - z[1] - z[2])];

                state.sketch = SketchState::RoundTwo;
                PrepareTransition::Continue(state, Poplar1PrepareMessage(y_share))
            }

            (SketchState::RoundTwo, Some(msg)) => {
                if msg.0.len() != 1 {
                    return PrepareTransition::Fail(VdafError::Uncategorized(format!(
                        "unexpected message length ({:?}): got {}; want 1",
                        state.sketch,
                        msg.0.len(),
                    )));
                }

                let y = msg.0[0];
                if y != I::Field::zero() {
                    return PrepareTransition::Fail(VdafError::Uncategorized(format!(
                        "output is invalid: polynomial evaluated to {}; want {}",
                        y,
                        I::Field::zero(),
                    )));
                }

                PrepareTransition::Finish(state.output_share)
            }
            _ => PrepareTransition::Fail(VdafError::Uncategorized(
                "invalid state transition".to_string(),
            )),
        }
    }

    fn aggregate<M: IntoIterator<Item = OutputShare<I::Field>>>(
        &self,
        agg_param: &BTreeSet<IdpfInput>,
        output_shares: M,
    ) -> Result<AggregateShare<I::Field>, VdafError> {
        let mut agg_share = AggregateShare(vec![I::Field::zero(); agg_param.len()]);
        for output_share in output_shares.into_iter() {
            agg_share.accumulate(&output_share)?;
        }

        Ok(agg_share)
    }
}

/// A prepare message sent exchanged between Poplar1 aggregators
#[derive(Clone, Debug)]
pub struct Poplar1PrepareMessage<F>(Vec<F>);

impl<F> AsRef<[F]> for Poplar1PrepareMessage<F> {
    fn as_ref(&self) -> &[F] {
        &self.0
    }
}

impl<F: FieldElement> Encode for Poplar1PrepareMessage<F> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        // TODO: This is encoded as a variable length vector of F, but we may
        // be able to make this a fixed-length vector for specific Poplar1
        // instantations
        encode_u16_items(bytes, &self.0);
    }
}

impl<F: FieldElement> Decode<Poplar1PrepareStep<F>> for Poplar1PrepareMessage<F> {
    fn decode(
        _decoding_parameter: &Poplar1PrepareStep<F>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        // TODO: This is decoded as a variable length vector of F, but we may be
        // able to make this a fixed-length vector for specific Poplar1
        // instantiations.
        let items = decode_u16_items(&(), bytes)?;

        Ok(Self(items))
    }
}

/// The state of each Aggregator during the Prepare process.
#[derive(Clone, Debug)]
pub struct Poplar1PrepareStep<F> {
    /// State of the secure sketching protocol.
    sketch: SketchState,

    /// The output share.
    output_share: OutputShare<F>,

    /// Shares of the blinded polynomial coefficients. See [BBCG+21, Appendix C.4] for details.
    z: [F; 3],

    /// Aggregator's share of $A = -2a + k$.
    d: F,

    /// Aggregator's share of $B = a^2 + b -ak + c$.
    e: F,

    /// Equal to 1 if this Aggregator is the "leader" and 0 otherwise.
    x: F,
}

#[derive(Clone, Debug)]
enum SketchState {
    Ready,
    RoundOne,
    RoundTwo,
}

impl<I: Idpf<2, 2>> Collector for Poplar1<I> {
    fn unshard<M: IntoIterator<Item = AggregateShare<I::Field>>>(
        &self,
        agg_param: &BTreeSet<IdpfInput>,
        agg_shares: M,
    ) -> Result<BTreeMap<IdpfInput, u64>, VdafError> {
        let mut agg_data = AggregateShare(vec![I::Field::zero(); agg_param.len()]);
        for agg_share in agg_shares.into_iter() {
            agg_data.merge(&agg_share)?;
        }

        let mut agg = BTreeMap::new();
        for (prefix, count) in agg_param.iter().zip(agg_data.as_ref()) {
            let count = <I::Field as FieldElement>::Integer::from(*count);
            let count: u64 = count
                .try_into()
                .map_err(|_| VdafError::Uncategorized("aggregate overflow".to_string()))?;
            agg.insert(*prefix, count);
        }
        Ok(agg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::Field128;

    #[test]
    fn test_ipdf() {
        // IDPF input equality tests.
        assert_eq!(
            IdpfInput::new(b"hello", 40).unwrap(),
            IdpfInput::new(b"hello", 40).unwrap()
        );
        assert_eq!(
            IdpfInput::new(b"hi", 9).unwrap(),
            IdpfInput::new(b"ha", 9).unwrap(),
        );
        assert_eq!(
            IdpfInput::new(b"hello", 25).unwrap(),
            IdpfInput::new(b"help", 25).unwrap()
        );
        assert_ne!(
            IdpfInput::new(b"hello", 40).unwrap(),
            IdpfInput::new(b"hello", 39).unwrap()
        );
        assert_ne!(
            IdpfInput::new(b"hello", 40).unwrap(),
            IdpfInput::new(b"hell-", 40).unwrap()
        );

        // IDPF uniqueness tests
        let mut unique = BTreeSet::new();
        assert!(unique.insert(IdpfInput::new(b"hello", 40).unwrap()));
        assert!(!unique.insert(IdpfInput::new(b"hello", 40).unwrap()));
        assert!(unique.insert(IdpfInput::new(b"hello", 39).unwrap()));
        assert!(unique.insert(IdpfInput::new(b"bye", 20).unwrap()));

        // Generate IPDF keys.
        let input = IdpfInput::new(b"hi", 16).unwrap();
        let keys = ToyIdpf::<Field128>::gen(
            &input,
            std::iter::repeat([Field128::one(), Field128::one()]),
        )
        .unwrap();

        // Try evaluating the IPDF keys on all prefixes.
        for prefix_len in 0..input.level + 1 {
            let res = eval_idpf(
                &keys,
                &input.prefix(prefix_len),
                &[Field128::one(), Field128::one()],
            );
            assert!(res.is_ok(), "prefix_len={} error: {:?}", prefix_len, res);
        }

        // Try evaluating the IPDF keys on incorrect prefixes.
        eval_idpf(
            &keys,
            &IdpfInput::new(&[2], 2).unwrap(),
            &[Field128::zero(), Field128::zero()],
        )
        .unwrap();

        eval_idpf(
            &keys,
            &IdpfInput::new(&[23, 1], 12).unwrap(),
            &[Field128::zero(), Field128::zero()],
        )
        .unwrap();
    }

    fn eval_idpf<I, const KEY_LEN: usize, const OUT_LEN: usize>(
        keys: &[I; KEY_LEN],
        input: &IdpfInput,
        expected_output: &[I::Field; OUT_LEN],
    ) -> Result<(), VdafError>
    where
        I: Idpf<KEY_LEN, OUT_LEN>,
    {
        let mut output = [I::Field::zero(); OUT_LEN];
        for key in keys {
            let output_share = key.eval(input)?;
            for (x, y) in output.iter_mut().zip(output_share) {
                *x += y;
            }
        }

        if expected_output != &output {
            return Err(VdafError::Uncategorized(format!(
                "eval_idpf(): unexpected output: got {:?}; want {:?}",
                output, expected_output
            )));
        }

        Ok(())
    }

    #[test]
    fn test_poplar1() {
        let vdaf: Poplar1<ToyIdpf<Field128>> = Poplar1::new(Suite::Blake3);
        assert_eq!(vdaf.num_aggregators(), 2);

        let (public_param, verify_params) = vdaf.setup().unwrap();

        // Run the VDAF input-distribution algorithm.
        let input = IdpfInput::new(b"hi", 16).unwrap();
        let input_shares = vdaf.shard(&public_param, &input).unwrap();
        let nonce = b"This is a nonce";

        let mut agg_param = BTreeSet::new();
        agg_param.insert(input);
        let res = eval_vdaf(
            &vdaf,
            &input_shares,
            &verify_params,
            &nonce[..],
            &agg_param,
            Some(&[1]),
        );
        assert!(res.is_ok(), "error: {:?}", res);

        // Try evaluating the VDAF on each prefix of the input.
        for prefix_len in 0..input.level + 1 {
            let mut agg_param = BTreeSet::new();
            agg_param.insert(input.prefix(prefix_len));
            let res = eval_vdaf(
                &vdaf,
                &input_shares,
                &verify_params,
                &nonce[..],
                &agg_param,
                Some(&[1]),
            );
            assert!(res.is_ok(), "prefix_len={} error: {:?}", prefix_len, res);
        }

        // Try various prefixes.
        let prefix_len = 9;
        let mut agg_param = BTreeSet::new();
        agg_param.insert(IdpfInput::new(&[0, 0], prefix_len).unwrap());
        agg_param.insert(IdpfInput::new(&[0, 1], prefix_len).unwrap());
        agg_param.insert(IdpfInput::new(&[13, 1], prefix_len).unwrap());
        agg_param.insert(IdpfInput::new(&[16, 1], prefix_len).unwrap());
        agg_param.insert(IdpfInput::new(&[23, 1], prefix_len).unwrap());
        agg_param.insert(IdpfInput::new(b"aa", prefix_len).unwrap());
        agg_param.insert(IdpfInput::new(b"hi", prefix_len).unwrap());
        agg_param.insert(IdpfInput::new(b"kk", prefix_len).unwrap());
        agg_param.insert(IdpfInput::new(b"xy", prefix_len).unwrap());
        let res = eval_vdaf(
            &vdaf,
            &input_shares,
            &verify_params,
            &nonce[..],
            &agg_param,
            Some(&[0, 0, 0, 0, 0, 0, 1, 0, 0]),
        );
        assert!(res.is_ok(), "error: {:?}", res);

        // Try evaluating the VDAF with an invalid aggregation parameter. (It's an error to have a
        // mixture of prefix lengths.)
        let mut agg_param = BTreeSet::new();
        agg_param.insert(IdpfInput::new(b"xx", 12).unwrap());
        agg_param.insert(IdpfInput::new(b"hi", 13).unwrap());
        eval_vdaf(
            &vdaf,
            &input_shares,
            &verify_params,
            &nonce[..],
            &agg_param,
            None,
        )
        .unwrap_err();

        // Try evaluating the VDAF with malformed inputs.
        //
        // This IDPF key pair evaluates to 1 everywhere, which is illegal.
        let mut input_shares = vdaf.shard(&public_param, &input).unwrap();
        for (i, x) in input_shares[0].idpf.data0.iter_mut().enumerate() {
            if i != input.index {
                *x += Field128::one();
            }
        }
        let mut agg_param = BTreeSet::new();
        agg_param.insert(IdpfInput::new(b"xx", 16).unwrap());
        eval_vdaf(
            &vdaf,
            &input_shares,
            &verify_params,
            &nonce[..],
            &agg_param,
            None,
        )
        .unwrap_err();

        // This IDPF key pair has a garbled authentication vector.
        let mut input_shares = vdaf.shard(&public_param, &input).unwrap();
        for x in input_shares[0].idpf.data1.iter_mut() {
            *x = Field128::zero();
        }
        let mut agg_param = BTreeSet::new();
        agg_param.insert(IdpfInput::new(b"xx", 16).unwrap());
        eval_vdaf(
            &vdaf,
            &input_shares,
            &verify_params,
            &nonce[..],
            &agg_param,
            None,
        )
        .unwrap_err();
    }

    // Execute the VDAF end-to-end on a single user measurement.
    fn eval_vdaf<I: Idpf<2, 2>>(
        vdaf: &Poplar1<I>,
        input_shares: &[Poplar1InputShare<I>],
        verify_params: &[Poplar1VerifyParam],
        nonce: &[u8],
        agg_param: &BTreeSet<IdpfInput>,
        expected_counts: Option<&[u64]>,
    ) -> Result<(), VdafError> {
        let mut state0: Vec<Poplar1PrepareStep<I::Field>> = Vec::with_capacity(2);
        for (verify_param, input_share) in verify_params.iter().zip(input_shares.iter()) {
            let state = vdaf.prepare_init(verify_param, agg_param, nonce, input_share)?;
            state0.push(state);
        }

        let mut round1: Vec<Poplar1PrepareMessage<I::Field>> = Vec::with_capacity(2);
        let mut state1: Vec<Poplar1PrepareStep<I::Field>> = Vec::with_capacity(2);
        for state in state0.into_iter() {
            let (state, msg) = vdaf.prepare_start(state)?;
            state1.push(state);
            round1.push(msg);
        }

        let mut round2: Vec<Poplar1PrepareMessage<I::Field>> = Vec::with_capacity(2);
        let mut state2: Vec<Poplar1PrepareStep<I::Field>> = Vec::with_capacity(2);
        for state in state1.into_iter() {
            let (state, msg) =
                vdaf.prepare_next(state, vdaf.prepare_preprocess(round1.clone())?)?;
            state2.push(state);
            round2.push(msg);
        }

        let mut agg_shares: Vec<AggregateShare<I::Field>> = Vec::with_capacity(2);
        for state in state2.into_iter() {
            let output_share =
                vdaf.prepare_finish(state, vdaf.prepare_preprocess(round2.clone())?)?;
            agg_shares.push(vdaf.aggregate(agg_param, [output_share])?);
        }

        if let Some(counts) = expected_counts {
            let agg = vdaf.unshard(agg_param, agg_shares)?;
            for (got, want) in agg.values().zip(counts.iter()) {
                if got != want {
                    return Err(VdafError::Uncategorized(
                        "unexpected output for test case".to_string(),
                    ));
                }
            }
        } else {
            return Err(VdafError::Uncategorized(
                "test case expected no output, got some".to_string(),
            ));
        }

        Ok(())
    }
}
