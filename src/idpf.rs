//! This module implements the incremental distributed point function (IDPF) described in
//! [[draft-irtf-cfrg-vdaf-03]].
//!
//! [draft-irtf-cfrg-vdaf-03]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/

use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{Field255, Field64, FieldElement},
    prng::Prng,
    vdaf::{
        prg::{Prg, RandSource, Seed, SeedStream},
        VdafError, VERSION,
    },
};
use bitvec::{
    bitvec,
    boxed::BitBox,
    prelude::{BitOrder, Lsb0},
    slice::BitSlice,
    store::BitStore,
    vec::BitVec,
};
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    io::{Cursor, Read},
    ops::Index,
};
use subtle::{Choice, ConstantTimeEq};

/// IDPF-related errors.
#[derive(Debug, thiserror::Error)]
pub enum IdpfError {
    /// Error from incompatible shares at different levels.
    #[error("tried to merge shares from incompatible levels")]
    MismatchedLevel,

    /// Invalid parameter.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

/// An index used as the input to an IDPF evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdpfInput {
    /// The index as a boxed bit slice.
    index: BitBox,
}

impl IdpfInput {
    /// Convert a slice of bytes into an IDPF input, where the bits of each byte are processed in
    /// LSB-to-MSB order. (and subsequent bytes are processed in their natural order)
    pub fn from_bytes(bytes: &[u8]) -> IdpfInput {
        let bit_box_u8_storage = BitBox::<u8, Lsb0>::from_boxed_slice(Box::from(bytes));
        bit_box_u8_storage.as_bitslice().into()
    }

    /// Convert a slice of booleans into an IDPF input.
    pub fn from_bools(bools: &[bool]) -> IdpfInput {
        let bits = bools.iter().collect::<BitVec>();
        IdpfInput {
            index: bits.into_boxed_bitslice(),
        }
    }

    /// Get the length of the input in bits.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if the input is empty, i.e. it does not contain any bits.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get an iterator over the bits that make up this input.
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        self.index.iter().by_vals()
    }
}

impl<T, O> From<&'_ BitSlice<T, O>> for IdpfInput
where
    T: BitStore,
    O: BitOrder,
{
    fn from(bit_slice: &'_ BitSlice<T, O>) -> Self {
        let mut bit_vec = bitvec![0; bit_slice.len()];
        bit_vec.clone_from_bitslice(bit_slice);
        IdpfInput {
            index: bit_vec.into_boxed_bitslice(),
        }
    }
}

impl<I> Index<I> for IdpfInput
where
    BitSlice: Index<I>,
{
    type Output = <BitSlice as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.index[index]
    }
}

/// An output from evaluation of an IDPF at some level and index.
#[derive(Debug, PartialEq, Eq)]
pub enum IdpfOutputShare<const OUT_LEN: usize, FI, FL> {
    /// An IDPF output share corresponding to an inner tree node.
    Inner([FI; OUT_LEN]),
    /// An IDPF output share corresponding to a leaf tree node.
    Leaf([FL; OUT_LEN]),
}

impl<const OUT_LEN: usize, FI, FL> IdpfOutputShare<OUT_LEN, FI, FL>
where
    FI: FieldElement,
    FL: FieldElement,
{
    /// Combine two output share vectors into one.
    pub fn merge(&self, other: &Self) -> Result<IdpfOutputShare<OUT_LEN, FI, FL>, IdpfError> {
        match self {
            IdpfOutputShare::Inner(self_array) => match other {
                IdpfOutputShare::Inner(other_array) => {
                    let mut array = *self_array;
                    for (dest, src) in array.iter_mut().zip(other_array.iter()) {
                        *dest += *src;
                    }
                    Ok(IdpfOutputShare::Inner(array))
                }
                IdpfOutputShare::Leaf(_) => Err(IdpfError::MismatchedLevel),
            },
            IdpfOutputShare::Leaf(self_array) => match other {
                IdpfOutputShare::Inner(_) => Err(IdpfError::MismatchedLevel),
                IdpfOutputShare::Leaf(other_array) => {
                    let mut array = *self_array;
                    for (dest, src) in array.iter_mut().zip(other_array.iter()) {
                        *dest += *src;
                    }
                    Ok(IdpfOutputShare::Leaf(array))
                }
            },
        }
    }
}

/// An auxiliary function that acts as a pseudorandom generator, extending one seed into two seeds
/// plus some additional bits.
fn extend<P, const L: usize>(seed: &[u8; L]) -> ([[u8; L]; 2], [Choice; 2])
where
    P: Prg<L>,
{
    let mut prg = P::init(seed);
    prg.update(VERSION);
    prg.update(b" idpf poplar extend");
    let mut seed_stream = prg.into_seed_stream();

    let mut seeds = [[0u8; L], [0u8; L]];
    seed_stream.fill(&mut seeds[0]);
    seed_stream.fill(&mut seeds[1]);

    let mut byte = [0u8];
    seed_stream.fill(&mut byte);
    let control_bits = [(byte[0] & 1).into(), ((byte[0] >> 1) & 1).into()];

    (seeds, control_bits)
}

/// An auxiliary function that acts as a pseudorandom generator, returning field elements.
fn convert<F, P, const L: usize, const OUT_LEN: usize>(seed: &[u8; L]) -> ([u8; L], [F; OUT_LEN])
where
    F: FieldElement,
    P: Prg<L>,
{
    let mut prg = P::init(seed);
    prg.update(VERSION);
    prg.update(b" idpf poplar convert");
    let mut seed_stream = prg.into_seed_stream();

    let mut next_seed = [0u8; L];
    seed_stream.fill(&mut next_seed);

    let prng = Prng::from_seed_stream(seed_stream);
    let mut w = [F::zero(); OUT_LEN];
    for (w_i, output) in w.iter_mut().zip(prng) {
        *w_i = output;
    }

    (next_seed, w)
}

/// Helper method to update seeds, update control bits, and output correction words for one level of
/// the IDPF key generation process.
fn generate_correction_words<F, P, const L: usize, const OUT_LEN: usize>(
    input_bit: bool,
    value: [F; OUT_LEN],
    seeds: &mut [[u8; L]; 2],
    control_bits: &mut [Choice; 2],
) -> IdpfPoplarCorrectionWords<F, L, OUT_LEN>
where
    F: FieldElement + From<u64>,
    P: Prg<L>,
{
    // expand both seeds into two more seeds, two more control bits each
    let (seed_0, control_bits_0) = extend::<P, L>(&seeds[0]);
    let (seed_1, control_bits_1) = extend::<P, L>(&seeds[1]);

    let (keep, lose) = (input_bit as usize, !input_bit as usize);

    let seed_correction_word = xor_seeds(&seed_0[lose], &seed_1[lose]);
    let control_bit_correction_words = [
        control_bits_0[0] ^ control_bits_1[0] ^ (input_bit as u8).into() ^ Choice::from(1),
        control_bits_0[1] ^ control_bits_1[1] ^ (input_bit as u8).into(),
    ];

    let previous_control_bits = *control_bits;
    control_bits[0] =
        control_bits_0[keep] ^ (previous_control_bits[0] & control_bit_correction_words[keep]);
    control_bits[1] =
        control_bits_1[keep] ^ (previous_control_bits[1] & control_bit_correction_words[keep]);

    let seeds_corrected = [
        xor_seeds(
            &seed_0[keep],
            &and_seeds(
                &seed_correction_word,
                &control_bit_to_seed_mask(previous_control_bits[0]),
            ),
        ),
        xor_seeds(
            &seed_1[keep],
            &and_seeds(
                &seed_correction_word,
                &control_bit_to_seed_mask(previous_control_bits[1]),
            ),
        ),
    ];

    let (new_seed_0, elements_0) = convert::<F, P, L, OUT_LEN>(&seeds_corrected[0]);
    let (new_seed_1, elements_1) = convert::<F, P, L, OUT_LEN>(&seeds_corrected[1]);

    seeds[0] = new_seed_0;
    seeds[1] = new_seed_1;

    let mut field_element_correction_words = [F::zero(); OUT_LEN];
    for (((out, element_0), element_1), value) in field_element_correction_words
        .iter_mut()
        .zip(elements_0.iter())
        .zip(elements_1.iter())
        .zip(value.iter())
    {
        let bit_converted = F::from(control_bits[1].unwrap_u8() as u64);
        let sign = F::one() - bit_converted - bit_converted;
        *out = (*value - *element_0 + *element_1) * sign;
    }

    IdpfPoplarCorrectionWords {
        seed_correction_word,
        control_bit_correction_words,
        field_element_correction_words,
    }
}

/// Helper function to evaluate one level of an IDPF. This updates the seed and control bit
/// arguments that are passed in.
fn eval_next<F, P, const L: usize, const OUT_LEN: usize>(
    which_aggregator: bool,
    seed: &mut [u8; L],
    control_bit: &mut Choice,
    correction_words: &IdpfPoplarCorrectionWords<F, L, OUT_LEN>,
    input_bit: bool,
) -> [F; OUT_LEN]
where
    F: FieldElement + From<u64>,
    P: Prg<L>,
{
    let (mut seeds, mut control_bits) = extend::<P, L>(seed);

    seeds[0] = xor_seeds(
        &seeds[0],
        &and_seeds(
            &correction_words.seed_correction_word,
            &control_bit_to_seed_mask(*control_bit),
        ),
    );
    control_bits[0] ^= correction_words.control_bit_correction_words[0] & *control_bit;
    seeds[1] = xor_seeds(
        &seeds[1],
        &and_seeds(
            &correction_words.seed_correction_word,
            &control_bit_to_seed_mask(*control_bit),
        ),
    );
    control_bits[1] ^= correction_words.control_bit_correction_words[1] & *control_bit;

    let seed_corrected = &seeds[input_bit as usize];
    *control_bit = control_bits[input_bit as usize];

    let (new_seed, elements) = convert::<F, P, L, OUT_LEN>(seed_corrected);
    *seed = new_seed;

    let mut elements_out = [F::zero(); OUT_LEN];
    for ((out, input), correction) in elements_out
        .iter_mut()
        .zip(elements.iter())
        .zip(correction_words.field_element_correction_words.iter())
    {
        *out = *input + *correction * F::from(control_bit.unwrap_u8() as u64);
        if which_aggregator {
            *out = -*out;
        }
    }
    elements_out
}

fn gen_with_rand_source<
    FI,
    FL,
    M: IntoIterator<Item = [FI; OUT_LEN]>,
    P,
    const L: usize,
    const OUT_LEN: usize,
>(
    bits: usize,
    input: &IdpfInput,
    inner_values: M,
    leaf_value: [FL; OUT_LEN],
    rand_source: RandSource,
) -> Result<(IdpfPoplarPublicShare<FI, FL, L, OUT_LEN>, [Seed<L>; 2]), VdafError>
where
    FI: FieldElement + From<u64>,
    FL: FieldElement + From<u64>,
    P: Prg<L>,
{
    if input.len() != bits {
        return Err(IdpfError::InvalidParameter(format!(
            "input length ({}) did not match configured number of bits ({})",
            input.len(),
            bits,
        ))
        .into());
    }

    let initial_seeds: [Seed<L>; 2] = [
        Seed::from_rand_source(rand_source)?,
        Seed::from_rand_source(rand_source)?,
    ];

    let mut seeds = [initial_seeds[0].0, initial_seeds[1].0];
    let mut control_bits = [Choice::from(0u8), Choice::from(1u8)];
    let mut inner_correction_words = Vec::with_capacity(bits - 1);

    for (level, value) in inner_values.into_iter().enumerate() {
        if level >= bits - 1 {
            return Err(
                IdpfError::InvalidParameter("too many values were supplied".to_string()).into(),
            );
        }
        inner_correction_words.push(generate_correction_words::<FI, P, L, OUT_LEN>(
            input[level],
            value,
            &mut seeds,
            &mut control_bits,
        ));
    }
    if inner_correction_words.len() != bits - 1 {
        return Err(IdpfError::InvalidParameter("too few values were supplied".to_string()).into());
    }
    let leaf_correction_words = generate_correction_words::<FL, P, L, OUT_LEN>(
        input[bits - 1],
        leaf_value,
        &mut seeds,
        &mut control_bits,
    );
    let public_share = IdpfPoplarPublicShare {
        inner_correction_words,
        leaf_correction_words,
    };

    Ok((public_share, initial_seeds))
}

/// The IdpfPoplar key generation algorithm.
///
/// Generate and return a sequence of IDPF shares for `input`. The parameters `inner_values`
/// and `leaf_value` provide the output values for each successive level of the prefix tree.
pub fn gen<M, P, const L: usize, const OUT_LEN: usize>(
    bits: usize,
    input: &IdpfInput,
    inner_values: M,
    leaf_value: [Field255; OUT_LEN],
) -> Result<
    (
        IdpfPoplarPublicShare<Field64, Field255, L, OUT_LEN>,
        [Seed<L>; 2],
    ),
    VdafError,
>
where
    M: IntoIterator<Item = [Field64; OUT_LEN]>,
    P: Prg<L>,
{
    if bits == 0 {
        return Err(IdpfError::InvalidParameter("invalid number of bits: 0".to_string()).into());
    }
    gen_with_rand_source::<_, _, _, P, L, OUT_LEN>(
        bits,
        input,
        inner_values,
        leaf_value,
        getrandom::getrandom,
    )
}

/// Evaluate an IDPF share on `prefix`, starting from a particular tree level with known
/// intermediate values.
#[allow(clippy::too_many_arguments)]
fn eval_inner<P, const L: usize, const OUT_LEN: usize>(
    bits: usize,
    which_aggregator: bool,
    public_share: &IdpfPoplarPublicShare<Field64, Field255, L, OUT_LEN>,
    start_level: usize,
    mut seed: [u8; L],
    mut control_bit: Choice,
    prefix: &IdpfInput,
    cache: &mut dyn IdpfCache<L>,
) -> Result<IdpfOutputShare<OUT_LEN, Field64, Field255>, IdpfError>
where
    P: Prg<L>,
{
    let mut last_inner_output = None;
    for ((correction_words, input_bit), level) in public_share.inner_correction_words[start_level..]
        .iter()
        .zip(prefix[start_level..].iter())
        .zip(start_level..)
    {
        last_inner_output = Some(eval_next::<_, P, L, OUT_LEN>(
            which_aggregator,
            &mut seed,
            &mut control_bit,
            correction_words,
            *input_bit,
        ));
        let cache_key = prefix[..=level].into();
        cache.insert(&cache_key, &(seed, control_bit.unwrap_u8()));
    }

    if prefix.len() == bits {
        let leaf_output = eval_next::<_, P, L, OUT_LEN>(
            which_aggregator,
            &mut seed,
            &mut control_bit,
            &public_share.leaf_correction_words,
            prefix[bits - 1],
        );
        // Note: there's no point caching this seed, because we will always run the eval_next()
        // call for the leaf level.
        Ok(IdpfOutputShare::Leaf(leaf_output))
    } else {
        Ok(IdpfOutputShare::Inner(last_inner_output.unwrap()))
    }
}

/// The IdpfPoplar key evaluation algorithm.
///
/// Evaluate an IDPF share on `prefix`.
pub fn eval<P, const L: usize, const OUT_LEN: usize>(
    bits: usize,
    agg_id: usize,
    public_share: &IdpfPoplarPublicShare<Field64, Field255, L, OUT_LEN>,
    key: &Seed<L>,
    prefix: &IdpfInput,
    cache: &mut dyn IdpfCache<L>,
) -> Result<IdpfOutputShare<OUT_LEN, Field64, Field255>, IdpfError>
where
    P: Prg<L>,
{
    if bits == 0 {
        return Err(IdpfError::InvalidParameter(
            "invalid number of bits: 0".to_string(),
        ));
    }
    if prefix.is_empty() {
        return Err(IdpfError::InvalidParameter("empty prefix".to_string()));
    }
    if prefix.len() > bits {
        return Err(IdpfError::InvalidParameter(format!(
            "prefix length ({}) exceeds configured number of bits ({})",
            prefix.len(),
            bits,
        )));
    }

    if public_share.inner_correction_words.len() != bits - 1 {
        return Err(IdpfError::InvalidParameter(format!(
            "public share had wrong number of correction words ({}, expected {})",
            public_share.inner_correction_words.len() + 1,
            bits,
        )));
    }

    let which_aggregator = agg_id != 0;

    // Check for cached seeds first.
    if prefix.len() > 1 {
        // Skip checking for `prefix` in the cache, because we don't store field element
        // values along with seeds and control bits. Instead, start looking one node higher
        // up, so we can recompute everything for the last level of `prefix`.
        let mut cache_key: IdpfInput = prefix[..prefix.len() - 1].into();
        while !cache_key.is_empty() {
            if let Some((seed, control_bit)) = cache.get(&cache_key) {
                // Evaluate the IDPF starting from the cached data at a previously-computed
                // node.
                return eval_inner::<P, L, OUT_LEN>(
                    bits,
                    which_aggregator,
                    public_share,
                    cache_key.len(),
                    seed,
                    Choice::from(control_bit),
                    prefix,
                    cache,
                );
            }
            cache_key = cache_key[..cache_key.len() - 1].into();
        }
    }
    // Evaluate starting from the root node.
    eval_inner::<P, L, OUT_LEN>(
        bits,
        which_aggregator,
        public_share,
        0,
        key.0,
        Choice::from(which_aggregator as u8),
        prefix,
        cache,
    )
}

/// IDPF public share used by Poplar1. This contains the lists of correction words used by all
/// parties when evaluating the IDPF.
#[derive(Debug, Clone)]
pub struct IdpfPoplarPublicShare<FI, FL, const L: usize, const OUT_LEN: usize> {
    /// Correction words for each inner node level.
    inner_correction_words: Vec<IdpfPoplarCorrectionWords<FI, L, OUT_LEN>>,
    /// Correction words for the leaf node level.
    leaf_correction_words: IdpfPoplarCorrectionWords<FL, L, OUT_LEN>,
}

impl<FI, FL, const L: usize, const OUT_LEN: usize> ConstantTimeEq
    for IdpfPoplarPublicShare<FI, FL, L, OUT_LEN>
where
    FI: ConstantTimeEq,
    FL: ConstantTimeEq,
{
    fn ct_eq(&self, other: &Self) -> Choice {
        self.inner_correction_words
            .ct_eq(&other.inner_correction_words)
            & self
                .leaf_correction_words
                .ct_eq(&other.leaf_correction_words)
    }
}

impl<FI, FL, const L: usize, const OUT_LEN: usize> PartialEq
    for IdpfPoplarPublicShare<FI, FL, L, OUT_LEN>
where
    FI: ConstantTimeEq,
    FL: ConstantTimeEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<FI, FL, const L: usize, const OUT_LEN: usize> Eq for IdpfPoplarPublicShare<FI, FL, L, OUT_LEN>
where
    FI: ConstantTimeEq,
    FL: ConstantTimeEq,
{
}

impl<FI, FL, const L: usize, const OUT_LEN: usize> Encode
    for IdpfPoplarPublicShare<FI, FL, L, OUT_LEN>
where
    FI: Encode,
    FL: Encode,
{
    fn encode(&self, bytes: &mut Vec<u8>) {
        let mut control_bits: BitVec<u8, Lsb0> =
            BitVec::with_capacity(self.inner_correction_words.len() * 2 + 2);
        for cws in self.inner_correction_words.iter() {
            control_bits.extend(
                cws.control_bit_correction_words
                    .iter()
                    .map(|x| bool::from(*x)),
            );
        }
        control_bits.extend(
            self.leaf_correction_words
                .control_bit_correction_words
                .iter()
                .map(|x| bool::from(*x)),
        );
        control_bits.set_uninitialized(false);
        let mut packed_control = control_bits.into_vec();
        packed_control.reverse();
        bytes.append(&mut packed_control);

        for cws in self.inner_correction_words.iter() {
            Seed(cws.seed_correction_word).encode(bytes);
            for elem in cws.field_element_correction_words.iter() {
                elem.encode(bytes);
            }
        }
        Seed(self.leaf_correction_words.seed_correction_word).encode(bytes);
        for elem in self
            .leaf_correction_words
            .field_element_correction_words
            .iter()
        {
            elem.encode(bytes);
        }
    }
}

impl<FI, FL, const L: usize, const OUT_LEN: usize> ParameterizedDecode<usize>
    for IdpfPoplarPublicShare<FI, FL, L, OUT_LEN>
where
    FI: Decode + Default + Copy,
    FL: Decode + Default + Copy,
{
    fn decode_with_param(bits: &usize, bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let packed_control_len = (bits + 3) / 4;
        let mut packed = vec![0u8; packed_control_len];
        bytes.read_exact(&mut packed)?;
        packed.reverse();
        let control_bits: BitVec<u8, Lsb0> = BitVec::from_vec(packed);

        let mut inner_correction_words = Vec::with_capacity(bits - 1);
        for chunk in control_bits[0..(bits - 1) * 2].chunks(2) {
            let control_bit_correction_words = [(chunk[0] as u8).into(), (chunk[1] as u8).into()];
            let seed_correction_word = Seed::<L>::decode(bytes)?.0;
            let mut field_element_correction_words = [FI::default(); OUT_LEN];
            for out in field_element_correction_words.iter_mut() {
                *out = FI::decode(bytes)?;
            }
            inner_correction_words.push(IdpfPoplarCorrectionWords {
                seed_correction_word,
                control_bit_correction_words,
                field_element_correction_words,
            })
        }

        let control_bit_correction_words = [
            (control_bits[(bits - 1) * 2] as u8).into(),
            (control_bits[bits * 2 - 1] as u8).into(),
        ];
        let seed_correction_word = Seed::<L>::decode(bytes)?.0;
        let mut field_element_correction_words = [FL::default(); OUT_LEN];
        for out in field_element_correction_words.iter_mut() {
            *out = FL::decode(bytes)?;
        }
        let leaf_correction_words = IdpfPoplarCorrectionWords {
            seed_correction_word,
            control_bit_correction_words,
            field_element_correction_words,
        };

        // check that unused packed bits are zero
        if control_bits[bits * 2..].any() {
            return Err(CodecError::UnexpectedValue);
        }

        Ok(IdpfPoplarPublicShare {
            inner_correction_words,
            leaf_correction_words,
        })
    }
}

#[derive(Debug, Clone)]
struct IdpfPoplarCorrectionWords<F, const L: usize, const OUT_LEN: usize> {
    seed_correction_word: [u8; L],
    control_bit_correction_words: [Choice; 2],
    field_element_correction_words: [F; OUT_LEN],
}

impl<F, const L: usize, const OUT_LEN: usize> ConstantTimeEq
    for IdpfPoplarCorrectionWords<F, L, OUT_LEN>
where
    F: ConstantTimeEq,
{
    fn ct_eq(&self, other: &Self) -> Choice {
        self.seed_correction_word.ct_eq(&other.seed_correction_word)
            & self
                .control_bit_correction_words
                .ct_eq(&other.control_bit_correction_words)
            & self
                .field_element_correction_words
                .ct_eq(&other.field_element_correction_words)
    }
}

impl<F, const L: usize, const OUT_LEN: usize> PartialEq for IdpfPoplarCorrectionWords<F, L, OUT_LEN>
where
    F: ConstantTimeEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F, const L: usize, const OUT_LEN: usize> Eq for IdpfPoplarCorrectionWords<F, L, OUT_LEN> where
    F: ConstantTimeEq
{
}

fn xor_seeds<const L: usize>(left: &[u8; L], right: &[u8; L]) -> [u8; L] {
    let mut seed = [0u8; L];
    for (a, (b, c)) in left.iter().zip(right.iter().zip(seed.iter_mut())) {
        *c = a ^ b;
    }
    seed
}

fn and_seeds<const L: usize>(left: &[u8; L], right: &[u8; L]) -> [u8; L] {
    let mut seed = [0u8; L];
    for (a, (b, c)) in left.iter().zip(right.iter().zip(seed.iter_mut())) {
        *c = a & b;
    }
    seed
}

/// Take a control bit, and fan it out into a byte array that can be used as a mask for PRG seeds,
/// without branching. If the control bit input is 0, all bytes will be equal to 0, and if the
/// control bit input is 1, all bytes will be equal to 255.
fn control_bit_to_seed_mask<const L: usize>(control: Choice) -> [u8; L] {
    let mask = -(control.unwrap_u8() as i8) as u8;
    [mask; L]
}

/// An interface that provides memoization of IDPF computations.
///
/// Each instance of a type implementing `IdpfCache` should only be used with one IDPF key and
/// public share.
///
/// In typical use, IDPFs will be evaluated repeatedly on inputs of increasing length, as part of a
/// protocol executed by multiple participants. Each IDPF evaluation computes seeds and control
/// bits corresponding to tree nodes along a path determined by the input to the IDPF. Thus, the
/// values from nodes further up in the tree may be cached and reused in evaluations of subsequent
/// longer inputs. If one IDPF input is a prefix of another input, then the first input's path down
/// the tree is a prefix of the other input's path.
pub trait IdpfCache<const L: usize> {
    /// Fetch cached values for the node identified by the IDPF input.
    fn get(&self, input: &IdpfInput) -> Option<([u8; L], u8)>;

    /// Store values corresponding to the node identified by the IDPF input.
    fn insert(&mut self, input: &IdpfInput, values: &([u8; L], u8));
}

/// A no-op [`IdpfCache`] implementation that always reports a cache miss.
#[derive(Default)]
pub struct NoCache {}

impl NoCache {
    /// Construct a `NoCache` object.
    pub fn new() -> NoCache {
        NoCache::default()
    }
}

impl<const L: usize> IdpfCache<L> for NoCache {
    fn get(&self, _: &IdpfInput) -> Option<([u8; L], u8)> {
        None
    }

    fn insert(&mut self, _: &IdpfInput, _: &([u8; L], u8)) {}
}

/// A simple [`IdpfCache`] implementation that caches intermediate results in an in-memory hash map,
/// with no eviction.
#[derive(Default)]
pub struct HashMapCache<const L: usize> {
    map: HashMap<IdpfInput, ([u8; L], u8)>,
}

impl<const L: usize> HashMapCache<L> {
    /// Create a new unpopulated `HashMapCache`.
    pub fn new() -> HashMapCache<L> {
        HashMapCache::default()
    }

    /// Create a new unpopulated `HashMapCache`, with a set pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> HashMapCache<L> {
        Self {
            map: HashMap::with_capacity(capacity),
        }
    }
}

impl<const L: usize> IdpfCache<L> for HashMapCache<L> {
    fn get(&self, input: &IdpfInput) -> Option<([u8; L], u8)> {
        self.map.get(input).cloned()
    }

    fn insert(&mut self, input: &IdpfInput, values: &([u8; L], u8)) {
        self.map.insert(input.clone(), *values);
    }
}

/// A simple [`IdpfCache`] implementation that caches intermediate results in memory, with
/// least-recently-used eviction, and lookups via linear probing.
pub struct RingBufferCache<const L: usize> {
    ring: VecDeque<(IdpfInput, [u8; L], u8)>,
}

impl<const L: usize> RingBufferCache<L> {
    /// Create a new unpopulated `RingBufferCache`.
    pub fn new(capacity: usize) -> RingBufferCache<L> {
        Self {
            ring: VecDeque::with_capacity(std::cmp::max(capacity, 1)),
        }
    }
}

impl<const L: usize> IdpfCache<L> for RingBufferCache<L> {
    fn get(&self, input: &IdpfInput) -> Option<([u8; L], u8)> {
        // iterate back-to-front, so that we check the most recently pushed entry first.
        for entry in self.ring.iter().rev() {
            if input == &entry.0 {
                return Some((entry.1, entry.2));
            }
        }
        None
    }

    fn insert(&mut self, input: &IdpfInput, values: &([u8; L], u8)) {
        // evict first (to avoid growing the storage)
        if self.ring.len() == self.ring.capacity() {
            self.ring.pop_front();
        }
        self.ring.push_back((input.clone(), values.0, values.1));
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        convert::{TryFrom, TryInto},
        sync::Mutex,
    };

    use bitvec::{
        bitbox, bits,
        prelude::{BitBox, Lsb0},
    };
    use rand::random;
    use subtle::Choice;

    use super::{
        HashMapCache, IdpfCache, IdpfInput, IdpfOutputShare, IdpfPoplarCorrectionWords,
        IdpfPoplarPublicShare, NoCache, RingBufferCache,
    };
    use crate::{
        codec::{Decode, Encode, ParameterizedDecode},
        field::{Field255, Field64, FieldElement},
        idpf,
        prng::Prng,
        vdaf::prg::{Prg, PrgAes128, Seed},
    };

    #[test]
    fn idpf_input_conversion() {
        let input_1 = IdpfInput::from_bools(&[
            true, false, false, false, false, false, true, false, false, true, false, false, false,
            false, true, false,
        ]);
        let input_2 = IdpfInput::from_bytes(b"AB");
        assert_eq!(input_1, input_2);
        let bits = bits![1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0];
        assert_eq!(input_1[..], bits);
    }

    /// A lossy IDPF cache, for testing purposes, that randomly returns cache misses.
    #[derive(Default)]
    struct LossyCache<const L: usize> {
        map: HashMap<IdpfInput, ([u8; L], u8)>,
    }

    impl<const L: usize> LossyCache<L> {
        /// Create a new unpopulated `LossyCache`.
        fn new() -> LossyCache<L> {
            LossyCache::default()
        }
    }

    impl<const L: usize> IdpfCache<L> for LossyCache<L> {
        fn get(&self, input: &IdpfInput) -> Option<([u8; L], u8)> {
            if random() {
                self.map.get(input).cloned()
            } else {
                None
            }
        }

        fn insert(&mut self, input: &IdpfInput, values: &([u8; L], u8)) {
            self.map.insert(input.clone(), *values);
        }
    }

    /// A wrapper [`IdpfCache`] implementation that records `get()` calls, for testing purposes.
    struct SnoopingCache<T, const L: usize> {
        inner: T,
        get_calls: Mutex<Vec<IdpfInput>>,
        insert_calls: Mutex<Vec<(IdpfInput, [u8; L], u8)>>,
    }

    impl<T, const L: usize> SnoopingCache<T, L> {
        fn new(inner: T) -> SnoopingCache<T, L> {
            SnoopingCache {
                inner,
                get_calls: Mutex::new(Vec::new()),
                insert_calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl<T, const L: usize> IdpfCache<L> for SnoopingCache<T, L>
    where
        T: IdpfCache<L>,
    {
        fn get(&self, input: &IdpfInput) -> Option<([u8; L], u8)> {
            self.get_calls.lock().unwrap().push(input.clone());
            self.inner.get(input)
        }

        fn insert(&mut self, input: &IdpfInput, values: &([u8; L], u8)) {
            self.insert_calls
                .lock()
                .unwrap()
                .push((input.clone(), values.0, values.1));
            self.inner.insert(input, values)
        }
    }

    #[test]
    fn test_idpf_poplar() {
        let bits = 5;
        let input = bitbox![0, 1, 1, 0, 1].as_bitslice().into();
        let (public_share, keys) = idpf::gen::<_, PrgAes128, 16, 2>(
            bits,
            &input,
            Vec::from([[Field64::one(), Field64::one()]; 4]),
            [Field255::one(), Field255::one()],
        )
        .unwrap();

        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![0].as_bitslice().into(),
            &IdpfOutputShare::Inner([Field64::one(), Field64::one()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![1].as_bitslice().into(),
            &IdpfOutputShare::Inner([Field64::zero(), Field64::zero()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![0, 1].as_bitslice().into(),
            &IdpfOutputShare::Inner([Field64::one(), Field64::one()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![0, 0].as_bitslice().into(),
            &IdpfOutputShare::Inner([Field64::zero(), Field64::zero()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![1, 0].as_bitslice().into(),
            &IdpfOutputShare::Inner([Field64::zero(), Field64::zero()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![1, 1].as_bitslice().into(),
            &IdpfOutputShare::Inner([Field64::zero(), Field64::zero()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![0, 1, 1].as_bitslice().into(),
            &IdpfOutputShare::Inner([Field64::one(), Field64::one()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![0, 1, 1, 0].as_bitslice().into(),
            &IdpfOutputShare::Inner([Field64::one(), Field64::one()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![0, 1, 1, 0, 1].as_bitslice().into(),
            &IdpfOutputShare::Leaf([Field255::one(), Field255::one()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![0, 1, 1, 0, 0].as_bitslice().into(),
            &IdpfOutputShare::Leaf([Field255::zero(), Field255::zero()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits,
            &public_share,
            &keys,
            &bitbox![1, 0, 1, 0, 0].as_bitslice().into(),
            &IdpfOutputShare::Leaf([Field255::zero(), Field255::zero()]),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
    }

    fn check_idpf_poplar_evaluation<P, const L: usize, const OUT_LEN: usize>(
        bits: usize,
        public_share: &IdpfPoplarPublicShare<Field64, Field255, L, OUT_LEN>,
        keys: &[Seed<L>; 2],
        prefix: &IdpfInput,
        expected_output: &IdpfOutputShare<OUT_LEN, Field64, Field255>,
        cache_0: &mut dyn IdpfCache<L>,
        cache_1: &mut dyn IdpfCache<L>,
    ) where
        P: Prg<L>,
    {
        let share_0 =
            idpf::eval::<P, L, OUT_LEN>(bits, 0, public_share, &keys[0], prefix, cache_0).unwrap();
        let share_1 =
            idpf::eval::<P, L, OUT_LEN>(bits, 1, public_share, &keys[1], prefix, cache_1).unwrap();
        let output = share_0.merge(&share_1).unwrap();
        assert_eq!(&output, expected_output);
    }

    #[test]
    fn test_idpf_poplar_medium() {
        // This test on 40 byte inputs takes about a second in debug mode. (and ten milliseconds in
        // release mode)
        const INPUT_LEN: usize = 320;
        let mut bits = bitbox![0; INPUT_LEN];
        for mut bit in bits.iter_mut() {
            bit.set(random());
        }

        let mut inner_values = [[Field64::one(), Field64::zero()]; INPUT_LEN - 1];
        let mut prng = Prng::new().unwrap();
        for level in inner_values.iter_mut() {
            level[1] = prng.next().unwrap();
        }
        let leaf_values = [Field255::one(), Prng::new().unwrap().next().unwrap()];

        let (public_share, keys) = idpf::gen::<_, PrgAes128, 16, 2>(
            INPUT_LEN,
            &bits.as_bitslice().into(),
            Vec::from(inner_values),
            leaf_values,
        )
        .unwrap();
        let mut cache_0 = RingBufferCache::new(3);
        let mut cache_1 = RingBufferCache::new(3);

        for (level, values) in inner_values.iter().enumerate() {
            let mut prefix = BitBox::from_bitslice(&bits[..=level]);
            check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
                INPUT_LEN,
                &public_share,
                &keys,
                &prefix.as_bitslice().into(),
                &IdpfOutputShare::Inner(*values),
                &mut cache_0,
                &mut cache_1,
            );
            let flipped_bit = !prefix[level];
            prefix.set(level, flipped_bit);
            check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
                INPUT_LEN,
                &public_share,
                &keys,
                &prefix.as_bitslice().into(),
                &IdpfOutputShare::Inner([Field64::zero(), Field64::zero()]),
                &mut cache_0,
                &mut cache_1,
            );
        }
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            INPUT_LEN,
            &public_share,
            &keys,
            &bits.as_bitslice().into(),
            &IdpfOutputShare::Leaf(leaf_values),
            &mut cache_0,
            &mut cache_1,
        );
        let mut modified_bits = bits.clone();
        modified_bits.set(INPUT_LEN - 1, !bits[INPUT_LEN - 1]);
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            INPUT_LEN,
            &public_share,
            &keys,
            &modified_bits.as_bitslice().into(),
            &IdpfOutputShare::Leaf([Field255::zero(), Field255::zero()]),
            &mut cache_0,
            &mut cache_1,
        );
    }

    #[test]
    fn idpf_poplar_cache_behavior() {
        let bits = bitbox![0, 1, 1, 1, 0, 1, 0, 0];
        let input = bits.as_bitslice().into();

        let mut inner_values = [[Field64::one(), Field64::zero()]; 7];
        let mut prng = Prng::new().unwrap();
        for level in inner_values.iter_mut() {
            level[1] = prng.next().unwrap();
        }
        let leaf_values = [Field255::one(), Prng::new().unwrap().next().unwrap()];

        let (public_share, keys) = idpf::gen::<_, PrgAes128, 16, 2>(
            bits.len(),
            &input,
            Vec::from(inner_values),
            leaf_values,
        )
        .unwrap();
        let mut cache_0 = SnoopingCache::new(HashMapCache::new());
        let mut cache_1 = HashMapCache::new();

        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits.len(),
            &public_share,
            &keys,
            &bits![1, 1, 0, 0].into(),
            &IdpfOutputShare::Inner([Field64::zero(), Field64::zero()]),
            &mut cache_0,
            &mut cache_1,
        );
        assert_eq!(
            cache_0
                .get_calls
                .lock()
                .unwrap()
                .drain(..)
                .collect::<Vec<_>>(),
            vec![bits![1, 1, 0].into(), bits![1, 1].into(), bits![1].into()],
        );
        assert_eq!(
            cache_0
                .insert_calls
                .lock()
                .unwrap()
                .drain(..)
                .map(|(input, _, _)| input)
                .collect::<Vec<_>>(),
            vec![
                bits![1].into(),
                bits![1, 1].into(),
                bits![1, 1, 0].into(),
                bits![1, 1, 0, 0].into()
            ],
        );

        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits.len(),
            &public_share,
            &keys,
            &bits![0].into(),
            &IdpfOutputShare::Inner(inner_values[0]),
            &mut cache_0,
            &mut cache_1,
        );
        assert_eq!(
            cache_0
                .get_calls
                .lock()
                .unwrap()
                .drain(..)
                .collect::<Vec<_>>(),
            vec![],
        );
        assert_eq!(
            cache_0
                .insert_calls
                .lock()
                .unwrap()
                .drain(..)
                .map(|(input, _, _)| input)
                .collect::<Vec<_>>(),
            vec![bits![0].into()],
        );

        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits.len(),
            &public_share,
            &keys,
            &bits![0, 1].into(),
            &IdpfOutputShare::Inner(inner_values[1]),
            &mut cache_0,
            &mut cache_1,
        );
        assert_eq!(
            cache_0
                .get_calls
                .lock()
                .unwrap()
                .drain(..)
                .collect::<Vec<_>>(),
            vec![bits![0].into()],
        );
        assert_eq!(
            cache_0
                .insert_calls
                .lock()
                .unwrap()
                .drain(..)
                .map(|(input, _, _)| input)
                .collect::<Vec<_>>(),
            vec![bits![0, 1].into()],
        );

        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits.len(),
            &public_share,
            &keys,
            &input,
            &IdpfOutputShare::Leaf(leaf_values),
            &mut cache_0,
            &mut cache_1,
        );
        assert_eq!(
            cache_0
                .get_calls
                .lock()
                .unwrap()
                .drain(..)
                .collect::<Vec<_>>(),
            vec![
                bits![0, 1, 1, 1, 0, 1, 0].into(),
                bits![0, 1, 1, 1, 0, 1].into(),
                bits![0, 1, 1, 1, 0].into(),
                bits![0, 1, 1, 1].into(),
                bits![0, 1, 1].into(),
                bits![0, 1].into(),
            ],
        );
        assert_eq!(
            cache_0
                .insert_calls
                .lock()
                .unwrap()
                .drain(..)
                .map(|(input, _, _)| input)
                .collect::<Vec<_>>(),
            vec![
                bits![0, 1, 1].into(),
                bits![0, 1, 1, 1].into(),
                bits![0, 1, 1, 1, 0].into(),
                bits![0, 1, 1, 1, 0, 1].into(),
                bits![0, 1, 1, 1, 0, 1, 0].into(),
            ],
        );

        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits.len(),
            &public_share,
            &keys,
            &input,
            &IdpfOutputShare::Leaf(leaf_values),
            &mut cache_0,
            &mut cache_1,
        );
        assert_eq!(
            cache_0
                .get_calls
                .lock()
                .unwrap()
                .drain(..)
                .collect::<Vec<_>>(),
            vec![bits![0, 1, 1, 1, 0, 1, 0].into()],
        );
        assert!(cache_0.insert_calls.lock().unwrap().is_empty());
    }

    #[test]
    fn idpf_poplar_lossy_cache() {
        let bits = bitbox![1, 0, 0, 1, 1, 0, 1, 0];
        let input = bits.as_bitslice().into();

        let mut inner_values = [[Field64::one(), Field64::zero()]; 7];
        let mut prng = Prng::new().unwrap();
        for level in inner_values.iter_mut() {
            level[1] = prng.next().unwrap();
        }
        let leaf_values = [Field255::one(), Prng::new().unwrap().next().unwrap()];

        let (public_share, keys) = idpf::gen::<_, PrgAes128, 16, 2>(
            bits.len(),
            &input,
            Vec::from(inner_values),
            leaf_values,
        )
        .unwrap();
        let mut cache_0 = LossyCache::new();
        let mut cache_1 = LossyCache::new();

        for (level, values) in inner_values.iter().enumerate() {
            check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
                bits.len(),
                &public_share,
                &keys,
                &bits[..=level].into(),
                &IdpfOutputShare::Inner(*values),
                &mut cache_0,
                &mut cache_1,
            );
        }
        check_idpf_poplar_evaluation::<PrgAes128, 16, 2>(
            bits.len(),
            &public_share,
            &keys,
            &input,
            &IdpfOutputShare::Leaf(leaf_values),
            &mut cache_0,
            &mut cache_1,
        );
    }

    #[test]
    fn test_idpf_poplar_error_cases() {
        // Zero bits does not make sense.
        idpf::gen::<_, PrgAes128, 16, 2>(
            0,
            &bitbox![].as_bitslice().into(),
            Vec::new(),
            [Field255::zero(); 2],
        )
        .unwrap_err();

        let bits = 10;
        let (public_share, keys) = idpf::gen::<_, PrgAes128, 16, 2>(
            bits,
            &bitbox![0; 10].as_bitslice().into(),
            Vec::from([[Field64::zero(); 2]; 9]),
            [Field255::zero(); 2],
        )
        .unwrap();

        // Wrong number of bits in the input.
        idpf::gen::<_, PrgAes128, 16, 2>(
            bits,
            &bitbox![].as_bitslice().into(),
            Vec::from([[Field64::zero(); 2]; 9]),
            [Field255::zero(); 2],
        )
        .unwrap_err();
        idpf::gen::<_, PrgAes128, 16, 2>(
            bits,
            &bitbox![0; 9].as_bitslice().into(),
            Vec::from([[Field64::zero(); 2]; 9]),
            [Field255::zero(); 2],
        )
        .unwrap_err();
        idpf::gen::<_, PrgAes128, 16, 2>(
            bits,
            &bitbox![0; 11].as_bitslice().into(),
            Vec::from([[Field64::zero(); 2]; 9]),
            [Field255::zero(); 2],
        )
        .unwrap_err();

        // Wrong number of values.
        idpf::gen::<_, PrgAes128, 16, 2>(
            bits,
            &bitbox![0; 10].as_bitslice().into(),
            Vec::from([[Field64::zero(); 2]; 8]),
            [Field255::zero(); 2],
        )
        .unwrap_err();
        idpf::gen::<_, PrgAes128, 16, 2>(
            bits,
            &bitbox![0; 10].as_bitslice().into(),
            Vec::from([[Field64::zero(); 2]; 10]),
            [Field255::zero(); 2],
        )
        .unwrap_err();

        // Evaluating with empty prefix.
        assert!(idpf::eval::<PrgAes128, 16, 2>(
            bits,
            0,
            &public_share,
            &keys[0],
            &bitbox![].as_bitslice().into(),
            &mut NoCache::new(),
        )
        .is_err());
        // Evaluating with too-long prefix.
        assert!(idpf::eval::<PrgAes128, 16, 2>(
            bits,
            0,
            &public_share,
            &keys[0],
            &bitbox![0; 11].as_bitslice().into(),
            &mut NoCache::new(),
        )
        .is_err());

        // Public share was generated with a different number of bits configured.
        let wrong_bits_low = 9;
        assert!(idpf::eval::<PrgAes128, 16, 2>(
            wrong_bits_low,
            0,
            &public_share,
            &keys[0],
            &bitbox![1].as_bitslice().into(),
            &mut NoCache::new(),
        )
        .is_err());
        let wrong_bits_high = 11;
        assert!(idpf::eval::<PrgAes128, 16, 2>(
            wrong_bits_high,
            0,
            &public_share,
            &keys[0],
            &bitbox![1].as_bitslice().into(),
            &mut NoCache::new(),
        )
        .is_err());
    }

    #[test]
    fn idpf_poplar_public_share_round_trip() {
        let public_share = IdpfPoplarPublicShare {
            inner_correction_words: Vec::from([
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0xab; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(0)],
                    field_element_correction_words: [
                        Field64::try_from(83261u64).unwrap(),
                        Field64::try_from(125159u64).unwrap(),
                    ],
                },
                IdpfPoplarCorrectionWords{
                    seed_correction_word: [0xcd;16],
                    control_bit_correction_words: [Choice::from(0), Choice::from(1)],
                    field_element_correction_words: [
                        Field64::try_from(17614120u64).unwrap(),
                        Field64::try_from(20674u64).unwrap(),
                    ],
                },
            ]),
            leaf_correction_words: IdpfPoplarCorrectionWords {
                seed_correction_word: [0xff; 16],
                control_bit_correction_words: [Choice::from(1), Choice::from(1)],
                field_element_correction_words: [
                    Field255::one(),
                    Field255::get_decoded(
                        b"\x12\x34\x56\x78\x9a\xbc\xde\xf0\x12\x34\x56\x78\x9a\xbc\xde\xf0\x12\x34\x56\x78\x9a\xbc\xde\xf0\x12\x34\x56\x78\x9a\xbc\xde\xf0",
                    ).unwrap(),
                ],
            },
        };
        let message = hex::decode(concat!(
            "39",                               // packed control bit correction words (0b00111001)
            "abababababababababababababababab", // seed correction word, first level
            "000000000001453d",                 // field element correction words
            "000000000001e8e7",                 // field element correction words, continued
            "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd", // seed correction word, second level
            "00000000010cc528",                 // field element correction words
            "00000000000050c2",                 // field element correction words, continued
            "ffffffffffffffffffffffffffffffff", // seed correction word, third level
            "0000000000000000000000000000000000000000000000000000000000000001", // field element correction words, leaf field
            "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0", // field element correction words, continued
        ))
        .unwrap();
        let encoded = public_share.get_encoded();
        let decoded = IdpfPoplarPublicShare::get_decoded_with_param(&3, &message).unwrap();
        assert_eq!(public_share, decoded);
        assert_eq!(message, encoded);

        // check serialization of packed control bits when they span multiple bytes:
        let public_share = IdpfPoplarPublicShare {
            inner_correction_words: Vec::from([
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(1)],
                    field_element_correction_words: [Field64::zero(), Field64::zero()],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(1)],
                    field_element_correction_words: [Field64::zero(), Field64::zero()],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(0)],
                    field_element_correction_words: [Field64::zero(), Field64::zero()],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(1)],
                    field_element_correction_words: [Field64::zero(), Field64::zero()],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(1)],
                    field_element_correction_words: [Field64::zero(), Field64::zero()],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(0), Choice::from(1)],
                    field_element_correction_words: [Field64::zero(), Field64::zero()],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(1)],
                    field_element_correction_words: [Field64::zero(), Field64::zero()],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(1)],
                    field_element_correction_words: [Field64::zero(), Field64::zero()],
                },
            ]),
            leaf_correction_words: IdpfPoplarCorrectionWords {
                seed_correction_word: [0; 16],
                control_bit_correction_words: [Choice::from(0), Choice::from(1)],
                field_element_correction_words: [Field255::zero(), Field255::zero()],
            },
        };
        let message = hex::decode(concat!(
            "02fbdf", // packed control bit CWs, 0b10_11111011_11011111, encoded big-endian
            "00000000000000000000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00000000000000000000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00000000000000000000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00000000000000000000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00000000000000000000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00000000000000000000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00000000000000000000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00000000000000000000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00000000000000000000000000000000",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "0000000000000000000000000000000000000000000000000000000000000000",
        ))
        .unwrap();
        let encoded = public_share.get_encoded();
        let decoded = IdpfPoplarPublicShare::get_decoded_with_param(&9, &message).unwrap();
        assert_eq!(public_share, decoded);
        assert_eq!(message, encoded);
    }

    /// Stores certain values from a Poplar1 test vector that are relevant to testing IdpfPoplar.
    struct IdpfPoplarTestVector {
        bits: usize,
        public_share: Vec<u8>,
    }

    /// Load the number of bits and serialized public share from the test vector.
    fn load_test_vector() -> IdpfPoplarTestVector {
        let test_vec: serde_json::Value =
            serde_json::from_str(include_str!("vdaf/test_vec/03/Poplar1Aes128_0.json")).unwrap();
        let test_vec_obj = test_vec.as_object().unwrap();
        let bits = test_vec_obj.get("bits").unwrap().as_u64().unwrap();
        let prep = test_vec_obj.get("prep").unwrap().as_array().unwrap();
        let prep_first = prep.get(0).unwrap().as_object().unwrap();
        let public_share_hex = prep_first.get("public_share").unwrap().as_str().unwrap();
        IdpfPoplarTestVector {
            bits: bits.try_into().unwrap(),
            public_share: hex::decode(public_share_hex).unwrap(),
        }
    }

    #[test]
    fn idpf_poplar_public_share_deserialize_test_vector() {
        let test_vector = load_test_vector();

        let public_share =
            IdpfPoplarPublicShare::<Field64, Field255, 16, 2>::get_decoded_with_param(
                &test_vector.bits,
                &test_vector.public_share,
            )
            .unwrap();

        let expected_public_share = IdpfPoplarPublicShare {
            inner_correction_words: Vec::from([
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(0), Choice::from(1)],
                    field_element_correction_words: [
                        Field64::from(1u64),
                        Field64::from(16949890756552313413u64),
                    ],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(0), Choice::from(1)],
                    field_element_correction_words: [
                        Field64::from(18446744069414584320u64),
                        Field64::from(2473087798058630316u64),
                    ],
                },
                IdpfPoplarCorrectionWords {
                    seed_correction_word: [0; 16],
                    control_bit_correction_words: [Choice::from(1), Choice::from(0)],
                    field_element_correction_words: [
                        Field64::from(18446744069414584320u64),
                        Field64::from(7634761277030804329u64),
                    ],
                },
            ]),
            leaf_correction_words: IdpfPoplarCorrectionWords {
                seed_correction_word: [0; 16],
                control_bit_correction_words: [Choice::from(0), Choice::from(1)],
                field_element_correction_words: [
                    Field255::one(),
                    Field255::try_from(
                        [
                            125u8, 31, 214, 223, 148, 40, 1, 69, 160, 220, 201, 51, 206, 183, 6,
                            233, 33, 157, 80, 231, 196, 249, 47, 216, 202, 154, 15, 251, 125, 129,
                            150, 70,
                        ]
                        .as_slice(),
                    )
                    .unwrap(),
                ],
            },
        };

        assert_eq!(public_share, expected_public_share);
    }

    #[test]
    fn idpf_poplar_generate_test_vector() {
        let test_vector = load_test_vector();

        // Values expected to be passed to IdpfPoplar.gen(). These values were observed during
        // generation of the test vector in the Sage proof-of-concept implementation.
        let alpha = bitbox![1, 1, 0, 1].as_bitslice().into();
        let beta_inner = Vec::from([
            [Field64::one(), Field64::from(16949890756552313413u64)],
            [Field64::one(), Field64::from(15973656271355954005u64)],
            [Field64::one(), Field64::from(10811982792383779992u64)],
        ]);
        let beta_leaf = [
            Field255::one(),
            Field255::try_from(
                [
                    0x7d, 0x1f, 0xd6, 0xdf, 0x94, 0x28, 0x1, 0x45, 0xa0, 0xdc, 0xc9, 0x33, 0xce,
                    0xb7, 0x6, 0xe9, 0x21, 0x9d, 0x50, 0xe7, 0xc4, 0xf9, 0x2f, 0xd8, 0xca, 0x9a,
                    0xf, 0xfb, 0x7d, 0x81, 0x96, 0x46,
                ]
                .as_slice(),
            )
            .unwrap(),
        ];

        let (public_share, keys) = idpf::gen_with_rand_source::<_, _, _, PrgAes128, 16, 2>(
            test_vector.bits,
            &alpha,
            beta_inner,
            beta_leaf,
            |buf| {
                buf.fill(1);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(keys, [Seed([0x01; 16]), Seed([0x01; 16])]);
        let expected_public_share = IdpfPoplarPublicShare::get_decoded_with_param(
            &test_vector.bits,
            &test_vector.public_share,
        )
        .unwrap();
        for (level, (cws, expected_cws)) in public_share
            .inner_correction_words
            .iter()
            .zip(expected_public_share.inner_correction_words.iter())
            .enumerate()
        {
            assert_eq!(
                cws, expected_cws,
                "layer {} did not match\n{:#x?}\n{:#x?}",
                level, cws, expected_cws,
            );
        }
        assert_eq!(
            public_share.leaf_correction_words,
            expected_public_share.leaf_correction_words
        );

        assert_eq!(
            public_share, expected_public_share,
            "public share did not match\n{:#x?}\n{:#x?}",
            &public_share, &expected_public_share,
        );
        let encoded_public_share = public_share.get_encoded();
        assert_eq!(encoded_public_share, test_vector.public_share);
    }
}
