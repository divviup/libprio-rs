//! This module implements the incremental distributed point function (IDPF) described in
//! [[draft-irtf-cfrg-vdaf-03]].
//!
//! [draft-irtf-cfrg-vdaf-03]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/

use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    vdaf::{
        prg::{CoinToss, Prg, RandSource, Seed, SeedStream},
        VdafError, VERSION,
    },
};
use bitvec::{bitvec, boxed::BitBox, prelude::Lsb0, slice::BitSlice, vec::BitVec, view::BitView};
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    io::{Cursor, Read},
    ops::{Add, AddAssign, Index, Sub},
};
use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable, ConstantTimeEq};

/// IDPF-related errors.
#[derive(Debug, thiserror::Error)]
pub enum IdpfError {
    /// Error from incompatible shares at different levels.
    #[error("tried to merge shares from incompatible levels")]
    MismatchedLevel,

    /// Invalid parameter, indicates an invalid input to either [`gen`] or [`eval`].
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
    /// LSB-to-MSB order. (Subsequent bytes are processed in their natural order.)
    pub fn from_bytes(bytes: &[u8]) -> IdpfInput {
        let bit_slice_u8_storage = bytes.view_bits::<Lsb0>();
        let mut bit_vec_usize_storage = bitvec![0; bit_slice_u8_storage.len()];
        bit_vec_usize_storage.clone_from_bitslice(bit_slice_u8_storage);
        IdpfInput {
            index: bit_vec_usize_storage.into_boxed_bitslice(),
        }
    }

    /// Convert a slice of booleans into an IDPF input.
    pub fn from_bools(bools: &[bool]) -> IdpfInput {
        let bits = bools.iter().collect::<BitVec>();
        IdpfInput {
            index: bits.into_boxed_bitslice(),
        }
    }

    /// Create a new IDPF input by appending to this input.
    pub fn clone_with_suffix(&self, suffix: &[bool]) -> IdpfInput {
        let mut vec = BitVec::with_capacity(self.index.len() + suffix.len());
        vec.extend_from_bitslice(&self.index);
        vec.extend(suffix);
        IdpfInput {
            index: vec.into_boxed_bitslice(),
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

    /// Convert the IDPF into a byte slice. If the length of the underlying bit vector is not a
    /// multiple of `8`, then the last byte is `0`-padded.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut vec = BitVec::<u8, Lsb0>::with_capacity(self.index.len());
        vec.extend_from_bitslice(&self.index);
        vec.set_uninitialized(false);
        vec.into_vec()
    }

    /// Return the `level`-bit prefix of this IDPF input.
    pub fn prefix(&self, level: usize) -> Self {
        Self {
            index: self.index[..=level].to_owned().into(),
        }
    }
}

impl From<BitVec<usize, Lsb0>> for IdpfInput {
    fn from(bit_vec: BitVec<usize, Lsb0>) -> Self {
        IdpfInput {
            index: bit_vec.into_boxed_bitslice(),
        }
    }
}

impl From<BitBox<usize, Lsb0>> for IdpfInput {
    fn from(bit_box: BitBox<usize, Lsb0>) -> Self {
        IdpfInput { index: bit_box }
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

/// Trait for values to be programmed into an IDPF.
///
/// Values must form an Abelian group, so that they can be secret-shared, and the group operation
/// must be represented by [`Add`]. An implementation of [`CoinToss`] must be provided to randomly
/// select a value using PRG output. Values must be encodable and decodable, without need for a
/// decoding parameter.
pub trait IdpfValue:
    Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + ConditionallySelectable
    + ConditionallyNegatable
    + CoinToss
    + Encode
    + Decode
    + Sized
{
    /// Returns the additive identity.
    fn zero() -> Self;
}

/// An output from evaluation of an IDPF at some level and index.
#[derive(Debug, PartialEq, Eq)]
pub enum IdpfOutputShare<VI, VL> {
    /// An IDPF output share corresponding to an inner tree node.
    Inner(VI),
    /// An IDPF output share corresponding to a leaf tree node.
    Leaf(VL),
}

impl<VI, VL> IdpfOutputShare<VI, VL>
where
    VI: IdpfValue,
    VL: IdpfValue,
{
    /// Combine two output share values into one.
    pub fn merge(self, other: Self) -> Result<IdpfOutputShare<VI, VL>, IdpfError> {
        match (self, other) {
            (IdpfOutputShare::Inner(mut self_value), IdpfOutputShare::Inner(other_value)) => {
                self_value += other_value;
                Ok(IdpfOutputShare::Inner(self_value))
            }
            (IdpfOutputShare::Leaf(mut self_value), IdpfOutputShare::Leaf(other_value)) => {
                self_value += other_value;
                Ok(IdpfOutputShare::Leaf(self_value))
            }
            (_, _) => Err(IdpfError::MismatchedLevel),
        }
    }
}

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

fn convert<V, P, const L: usize>(seed: &[u8; L]) -> ([u8; L], V)
where
    V: IdpfValue,
    P: Prg<L>,
{
    let mut prg = P::init(seed);
    prg.update(VERSION);
    prg.update(b" idpf poplar convert");
    let mut seed_stream = prg.into_seed_stream();

    let mut next_seed = [0u8; L];
    seed_stream.fill(&mut next_seed);

    (next_seed, V::sample(&mut seed_stream))
}

/// Helper method to update seeds, update control bits, and output the correction word for one level
/// of the IDPF key generation process.
fn generate_correction_word<V, P, const L: usize>(
    input_bit: Choice,
    value: V,
    keys: &mut [[u8; L]; 2],
    control_bits: &mut [Choice; 2],
) -> IdpfCorrectionWord<V, L>
where
    V: IdpfValue,
    P: Prg<L>,
{
    // Expand both keys into two seeds and two control bits each.
    let (seed_0, control_bits_0) = extend::<P, L>(&keys[0]);
    let (seed_1, control_bits_1) = extend::<P, L>(&keys[1]);

    let (keep, lose) = (input_bit, !input_bit);

    let cw_seed = xor_seeds(
        &conditional_select_seed(lose, &seed_0),
        &conditional_select_seed(lose, &seed_1),
    );
    let cw_control_bits = [
        control_bits_0[0] ^ control_bits_1[0] ^ input_bit ^ Choice::from(1),
        control_bits_0[1] ^ control_bits_1[1] ^ input_bit,
    ];
    let cw_control_bits_keep =
        Choice::conditional_select(&cw_control_bits[0], &cw_control_bits[1], keep);

    let previous_control_bits = *control_bits;
    let control_bits_0_keep =
        Choice::conditional_select(&control_bits_0[0], &control_bits_0[1], keep);
    let control_bits_1_keep =
        Choice::conditional_select(&control_bits_1[0], &control_bits_1[1], keep);
    control_bits[0] = control_bits_0_keep ^ (cw_control_bits_keep & previous_control_bits[0]);
    control_bits[1] = control_bits_1_keep ^ (cw_control_bits_keep & previous_control_bits[1]);

    let seed_0_keep = conditional_select_seed(keep, &seed_0);
    let seed_1_keep = conditional_select_seed(keep, &seed_1);
    let seeds_corrected = [
        conditional_xor_seeds(&seed_0_keep, &cw_seed, previous_control_bits[0]),
        conditional_xor_seeds(&seed_1_keep, &cw_seed, previous_control_bits[1]),
    ];

    let (new_key_0, elements_0) = convert::<V, P, L>(&seeds_corrected[0]);
    let (new_key_1, elements_1) = convert::<V, P, L>(&seeds_corrected[1]);

    keys[0] = new_key_0;
    keys[1] = new_key_1;

    let mut cw_value = value - elements_0 + elements_1;
    cw_value.conditional_negate(control_bits[1]);

    IdpfCorrectionWord {
        seed: cw_seed,
        control_bits: cw_control_bits,
        value: cw_value,
    }
}

/// Helper function to evaluate one level of an IDPF. This updates the seed and control bit
/// arguments that are passed in.
fn eval_next<V, P, const L: usize>(
    is_leader: bool,
    key: &mut [u8; L],
    control_bit: &mut Choice,
    correction_word: &IdpfCorrectionWord<V, L>,
    input_bit: Choice,
) -> V
where
    V: IdpfValue,
    P: Prg<L>,
{
    let (mut seeds, mut control_bits) = extend::<P, L>(key);

    seeds[0] = conditional_xor_seeds(&seeds[0], &correction_word.seed, *control_bit);
    control_bits[0] ^= correction_word.control_bits[0] & *control_bit;
    seeds[1] = conditional_xor_seeds(&seeds[1], &correction_word.seed, *control_bit);
    control_bits[1] ^= correction_word.control_bits[1] & *control_bit;

    let seed_corrected = conditional_select_seed(input_bit, &seeds);
    *control_bit = Choice::conditional_select(&control_bits[0], &control_bits[1], input_bit);

    let (new_key, elements) = convert::<V, P, L>(&seed_corrected);
    *key = new_key;

    let mut out =
        elements + V::conditional_select(&V::zero(), &correction_word.value, *control_bit);
    out.conditional_negate(Choice::from((!is_leader) as u8));
    out
}

fn gen_with_rand_source<VI, VL, M: IntoIterator<Item = VI>, P, const L: usize>(
    input: &IdpfInput,
    inner_values: M,
    leaf_value: VL,
    rand_source: RandSource,
) -> Result<(IdpfPublicShare<VI, VL, L>, [Seed<L>; 2]), VdafError>
where
    VI: IdpfValue,
    VL: IdpfValue,
    P: Prg<L>,
{
    let bits = input.len();

    let initial_keys: [Seed<L>; 2] = [
        Seed::from_rand_source(rand_source)?,
        Seed::from_rand_source(rand_source)?,
    ];

    let mut keys = [initial_keys[0].0, initial_keys[1].0];
    let mut control_bits = [Choice::from(0u8), Choice::from(1u8)];
    let mut inner_correction_words = Vec::with_capacity(bits - 1);

    for (level, value) in inner_values.into_iter().enumerate() {
        if level >= bits - 1 {
            return Err(
                IdpfError::InvalidParameter("too many values were supplied".to_string()).into(),
            );
        }
        inner_correction_words.push(generate_correction_word::<VI, P, L>(
            Choice::from(input[level] as u8),
            value,
            &mut keys,
            &mut control_bits,
        ));
    }
    if inner_correction_words.len() != bits - 1 {
        return Err(IdpfError::InvalidParameter("too few values were supplied".to_string()).into());
    }
    let leaf_correction_word = generate_correction_word::<VL, P, L>(
        Choice::from(input[bits - 1] as u8),
        leaf_value,
        &mut keys,
        &mut control_bits,
    );
    let public_share = IdpfPublicShare {
        inner_correction_words,
        leaf_correction_word,
    };

    Ok((public_share, initial_keys))
}

/// The IDPF key generation algorithm.
///
/// Generate and return a sequence of IDPF shares for `input`. The parameters `inner_values`
/// and `leaf_value` provide the output values for each successive level of the prefix tree.
pub fn gen<VI, VL, M, P, const L: usize>(
    input: &IdpfInput,
    inner_values: M,
    leaf_value: VL,
) -> Result<(IdpfPublicShare<VI, VL, L>, [Seed<L>; 2]), VdafError>
where
    VI: IdpfValue,
    VL: IdpfValue,
    M: IntoIterator<Item = VI>,
    P: Prg<L>,
{
    if input.is_empty() {
        return Err(IdpfError::InvalidParameter("invalid number of bits: 0".to_string()).into());
    }
    gen_with_rand_source::<_, _, _, P, L>(input, inner_values, leaf_value, getrandom::getrandom)
}

/// Evaluate an IDPF share on `prefix`, starting from a particular tree level with known
/// intermediate values.
#[allow(clippy::too_many_arguments)]
fn eval_from_node<VI, VL, P, const L: usize>(
    is_leader: bool,
    public_share: &IdpfPublicShare<VI, VL, L>,
    start_level: usize,
    mut key: [u8; L],
    mut control_bit: Choice,
    prefix: &IdpfInput,
    cache: &mut dyn IdpfCache<L>,
) -> Result<IdpfOutputShare<VI, VL>, IdpfError>
where
    VI: IdpfValue,
    VL: IdpfValue,
    P: Prg<L>,
{
    let bits = public_share.inner_correction_words.len() + 1;
    let mut last_inner_output = None;
    for ((correction_word, input_bit), level) in public_share.inner_correction_words[start_level..]
        .iter()
        .zip(prefix[start_level..].iter())
        .zip(start_level..)
    {
        last_inner_output = Some(eval_next::<_, P, L>(
            is_leader,
            &mut key,
            &mut control_bit,
            correction_word,
            Choice::from(*input_bit as u8),
        ));
        let cache_key = &prefix[..=level];
        cache.insert(cache_key, &(key, control_bit.unwrap_u8()));
    }

    if prefix.len() == bits {
        let leaf_output = eval_next::<_, P, L>(
            is_leader,
            &mut key,
            &mut control_bit,
            &public_share.leaf_correction_word,
            Choice::from(prefix[bits - 1] as u8),
        );
        // Note: there's no point caching this node's key, because we will always run the
        // eval_next() call for the leaf level.
        Ok(IdpfOutputShare::Leaf(leaf_output))
    } else {
        Ok(IdpfOutputShare::Inner(last_inner_output.unwrap()))
    }
}

/// The IDPF key evaluation algorithm.
///
/// Evaluate an IDPF share on `prefix`.
pub fn eval<VI, VL, P, const L: usize>(
    agg_id: usize,
    public_share: &IdpfPublicShare<VI, VL, L>,
    key: &Seed<L>,
    prefix: &IdpfInput,
    cache: &mut dyn IdpfCache<L>,
) -> Result<IdpfOutputShare<VI, VL>, IdpfError>
where
    VI: IdpfValue,
    VL: IdpfValue,
    P: Prg<L>,
{
    let bits = public_share.inner_correction_words.len() + 1;
    if agg_id > 1 {
        return Err(IdpfError::InvalidParameter(format!(
            "invalid aggregator ID {agg_id}"
        )));
    }
    let is_leader = agg_id == 0;
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

    // Check for cached keys first, starting from the end of our desired path down the tree, and
    // walking back up. If we get a hit, stop there and evaluate the remainder of the tree path
    // going forward.
    if prefix.len() > 1 {
        // Skip checking for `prefix` in the cache, because we don't store field element
        // values along with keys and control bits. Instead, start looking one node higher
        // up, so we can recompute everything for the last level of `prefix`.
        let mut cache_key = &prefix[..prefix.len() - 1];
        while !cache_key.is_empty() {
            if let Some((key, control_bit)) = cache.get(cache_key) {
                // Evaluate the IDPF starting from the cached data at a previously-computed
                // node, and return the result.
                return eval_from_node::<VI, VL, P, L>(
                    is_leader,
                    public_share,
                    /* start_level */ cache_key.len(),
                    key,
                    Choice::from(control_bit),
                    prefix,
                    cache,
                );
            }
            cache_key = &cache_key[..cache_key.len() - 1];
        }
    }
    // Evaluate starting from the root node.
    eval_from_node::<VI, VL, P, L>(
        is_leader,
        public_share,
        /* start_level */ 0,
        key.0,
        /* control_bit */ Choice::from((!is_leader) as u8),
        prefix,
        cache,
    )
}

/// An IDPF public share. This contains the list of correction words used by all parties when
/// evaluating the IDPF.
#[derive(Debug, Clone)]
pub struct IdpfPublicShare<VI, VL, const L: usize> {
    /// Correction words for each inner node level.
    inner_correction_words: Vec<IdpfCorrectionWord<VI, L>>,
    /// Correction word for the leaf node level.
    leaf_correction_word: IdpfCorrectionWord<VL, L>,
}

impl<VI, VL, const L: usize> ConstantTimeEq for IdpfPublicShare<VI, VL, L>
where
    VI: ConstantTimeEq,
    VL: ConstantTimeEq,
{
    fn ct_eq(&self, other: &Self) -> Choice {
        self.inner_correction_words
            .ct_eq(&other.inner_correction_words)
            & self.leaf_correction_word.ct_eq(&other.leaf_correction_word)
    }
}

impl<VI, VL, const L: usize> PartialEq for IdpfPublicShare<VI, VL, L>
where
    VI: ConstantTimeEq,
    VL: ConstantTimeEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<VI, VL, const L: usize> Eq for IdpfPublicShare<VI, VL, L>
where
    VI: ConstantTimeEq,
    VL: ConstantTimeEq,
{
}

impl<VI, VL, const L: usize> Encode for IdpfPublicShare<VI, VL, L>
where
    VI: Encode,
    VL: Encode,
{
    fn encode(&self, bytes: &mut Vec<u8>) {
        // Control bits need to be written within each byte in LSB-to-MSB order, and assigned into
        // bytes in big-endian order. Thus, the first four levels will have their control bits
        // encoded in the last byte, and the last levels will have their control bits encoded in the
        // first byte.
        let mut control_bits: BitVec<u8, Lsb0> =
            BitVec::with_capacity(self.inner_correction_words.len() * 2 + 2);
        for correction_words in self.inner_correction_words.iter() {
            control_bits.extend(correction_words.control_bits.iter().map(|x| bool::from(*x)));
        }
        control_bits.extend(
            self.leaf_correction_word
                .control_bits
                .iter()
                .map(|x| bool::from(*x)),
        );
        control_bits.set_uninitialized(false);
        let mut packed_control = control_bits.into_vec();
        // Flip the byte order from `bitvec`'s native order to the required big-endian order.
        packed_control.reverse();
        bytes.append(&mut packed_control);

        for correction_words in self.inner_correction_words.iter() {
            Seed(correction_words.seed).encode(bytes);
            correction_words.value.encode(bytes);
        }
        Seed(self.leaf_correction_word.seed).encode(bytes);
        self.leaf_correction_word.value.encode(bytes);
    }
}

impl<VI, VL, const L: usize> ParameterizedDecode<usize> for IdpfPublicShare<VI, VL, L>
where
    VI: Decode + Copy,
    VL: Decode + Copy,
{
    fn decode_with_param(bits: &usize, bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let packed_control_len = (bits + 3) / 4;
        let mut packed = vec![0u8; packed_control_len];
        bytes.read_exact(&mut packed)?;
        packed.reverse();
        let unpacked_control_bits: BitVec<u8, Lsb0> = BitVec::from_vec(packed);

        let mut inner_correction_words = Vec::with_capacity(bits - 1);
        for chunk in unpacked_control_bits[0..(bits - 1) * 2].chunks(2) {
            let control_bits = [(chunk[0] as u8).into(), (chunk[1] as u8).into()];
            let seed = Seed::<L>::decode(bytes)?.0;
            let value = VI::decode(bytes)?;
            inner_correction_words.push(IdpfCorrectionWord {
                seed,
                control_bits,
                value,
            })
        }

        let control_bits = [
            (unpacked_control_bits[(bits - 1) * 2] as u8).into(),
            (unpacked_control_bits[bits * 2 - 1] as u8).into(),
        ];
        let seed = Seed::<L>::decode(bytes)?.0;
        let value = VL::decode(bytes)?;
        let leaf_correction_word = IdpfCorrectionWord {
            seed,
            control_bits,
            value,
        };

        // Check that unused packed bits are zero.
        if unpacked_control_bits[bits * 2..].any() {
            return Err(CodecError::UnexpectedValue);
        }

        Ok(IdpfPublicShare {
            inner_correction_words,
            leaf_correction_word,
        })
    }
}

#[derive(Debug, Clone)]
struct IdpfCorrectionWord<V, const L: usize> {
    seed: [u8; L],
    control_bits: [Choice; 2],
    value: V,
}

impl<V, const L: usize> ConstantTimeEq for IdpfCorrectionWord<V, L>
where
    V: ConstantTimeEq,
{
    fn ct_eq(&self, other: &Self) -> Choice {
        self.seed.ct_eq(&other.seed)
            & self.control_bits.ct_eq(&other.control_bits)
            & self.value.ct_eq(&other.value)
    }
}

impl<V, const L: usize> PartialEq for IdpfCorrectionWord<V, L>
where
    V: ConstantTimeEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<V, const L: usize> Eq for IdpfCorrectionWord<V, L> where V: ConstantTimeEq {}

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

fn or_seeds<const L: usize>(left: &[u8; L], right: &[u8; L]) -> [u8; L] {
    let mut seed = [0u8; L];
    for (a, (b, c)) in left.iter().zip(right.iter().zip(seed.iter_mut())) {
        *c = a | b;
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

/// Take two seeds and a control bit, and return the first seed if the control bit is zero, or the
/// XOR of the two seeds if the control bit is one. This does not branch on the control bit.
fn conditional_xor_seeds<const L: usize>(
    normal_input: &[u8; L],
    switched_input: &[u8; L],
    control: Choice,
) -> [u8; L] {
    xor_seeds(
        normal_input,
        &and_seeds(switched_input, &control_bit_to_seed_mask(control)),
    )
}

/// Returns one of two seeds, depending on the value of a selector bit. Does not branch on the
/// selector input or make selector-dependent memory accesses.
fn conditional_select_seed<const L: usize>(select: Choice, seeds: &[[u8; L]; 2]) -> [u8; L] {
    or_seeds(
        &and_seeds(&control_bit_to_seed_mask(!select), &seeds[0]),
        &and_seeds(&control_bit_to_seed_mask(select), &seeds[1]),
    )
}

/// An interface that provides memoization of IDPF computations.
///
/// Each instance of a type implementing `IdpfCache` should only be used with one IDPF key and
/// public share.
///
/// In typical use, IDPFs will be evaluated repeatedly on inputs of increasing length, as part of a
/// protocol executed by multiple participants. Each IDPF evaluation computes keys and control
/// bits corresponding to tree nodes along a path determined by the input to the IDPF. Thus, the
/// values from nodes further up in the tree may be cached and reused in evaluations of subsequent
/// longer inputs. If one IDPF input is a prefix of another input, then the first input's path down
/// the tree is a prefix of the other input's path.
pub trait IdpfCache<const L: usize> {
    /// Fetch cached values for the node identified by the IDPF input.
    fn get(&self, input: &BitSlice) -> Option<([u8; L], u8)>;

    /// Store values corresponding to the node identified by the IDPF input.
    fn insert(&mut self, input: &BitSlice, values: &([u8; L], u8));
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
    fn get(&self, _: &BitSlice) -> Option<([u8; L], u8)> {
        None
    }

    fn insert(&mut self, _: &BitSlice, _: &([u8; L], u8)) {}
}

/// A simple [`IdpfCache`] implementation that caches intermediate results in an in-memory hash map,
/// with no eviction.
#[derive(Default)]
pub struct HashMapCache<const L: usize> {
    map: HashMap<BitBox, ([u8; L], u8)>,
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
    fn get(&self, input: &BitSlice) -> Option<([u8; L], u8)> {
        self.map.get(input).cloned()
    }

    fn insert(&mut self, input: &BitSlice, values: &([u8; L], u8)) {
        if !self.map.contains_key(input) {
            self.map
                .insert(input.to_owned().into_boxed_bitslice(), *values);
        }
    }
}

/// A simple [`IdpfCache`] implementation that caches intermediate results in memory, with
/// least-recently-used eviction, and lookups via linear probing.
pub struct RingBufferCache<const L: usize> {
    ring: VecDeque<(BitBox, [u8; L], u8)>,
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
    fn get(&self, input: &BitSlice) -> Option<([u8; L], u8)> {
        // iterate back-to-front, so that we check the most recently pushed entry first.
        for entry in self.ring.iter().rev() {
            if input == entry.0 {
                return Some((entry.1, entry.2));
            }
        }
        None
    }

    fn insert(&mut self, input: &BitSlice, values: &([u8; L], u8)) {
        // evict first (to avoid growing the storage)
        if self.ring.len() == self.ring.capacity() {
            self.ring.pop_front();
        }
        self.ring
            .push_back((input.to_owned().into_boxed_bitslice(), values.0, values.1));
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        convert::{TryFrom, TryInto},
        io::Cursor,
        str::FromStr,
        sync::Mutex,
    };

    use assert_matches::assert_matches;
    use bitvec::{
        bitbox,
        prelude::{BitBox, Lsb0},
        slice::BitSlice,
        vec::BitVec,
    };
    use num_bigint::BigUint;
    use rand::random;
    use subtle::Choice;

    use super::{
        HashMapCache, IdpfCache, IdpfCorrectionWord, IdpfInput, IdpfOutputShare, IdpfPublicShare,
        NoCache, RingBufferCache,
    };
    use crate::{
        codec::{CodecError, Decode, Encode, ParameterizedDecode},
        field::{Field255, Field64, FieldElement},
        idpf,
        prng::Prng,
        vdaf::{
            poplar1::Poplar1IdpfValue,
            prg::{Prg, PrgAes128, Seed},
        },
    };

    #[test]
    fn idpf_input_conversion() {
        let input_1 = IdpfInput::from_bools(&[
            true, false, false, false, false, false, true, false, false, true, false, false, false,
            false, true, false,
        ]);
        let input_2 = IdpfInput::from_bytes(b"AB");
        assert_eq!(input_1, input_2);
        let bits = bitbox![1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0];
        assert_eq!(input_1[..], bits);
    }

    /// A lossy IDPF cache, for testing purposes, that randomly returns cache misses.
    #[derive(Default)]
    struct LossyCache<const L: usize> {
        map: HashMap<BitBox, ([u8; L], u8)>,
    }

    impl<const L: usize> LossyCache<L> {
        /// Create a new unpopulated `LossyCache`.
        fn new() -> LossyCache<L> {
            LossyCache::default()
        }
    }

    impl<const L: usize> IdpfCache<L> for LossyCache<L> {
        fn get(&self, input: &BitSlice) -> Option<([u8; L], u8)> {
            if random() {
                self.map.get(input).cloned()
            } else {
                None
            }
        }

        fn insert(&mut self, input: &BitSlice, values: &([u8; L], u8)) {
            if !self.map.contains_key(input) {
                self.map
                    .insert(input.to_owned().into_boxed_bitslice(), *values);
            }
        }
    }

    /// A wrapper [`IdpfCache`] implementation that records `get()` calls, for testing purposes.
    struct SnoopingCache<T, const L: usize> {
        inner: T,
        get_calls: Mutex<Vec<BitBox>>,
        insert_calls: Mutex<Vec<(BitBox, [u8; L], u8)>>,
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
        fn get(&self, input: &BitSlice) -> Option<([u8; L], u8)> {
            self.get_calls
                .lock()
                .unwrap()
                .push(input.to_owned().into_boxed_bitslice());
            self.inner.get(input)
        }

        fn insert(&mut self, input: &BitSlice, values: &([u8; L], u8)) {
            self.insert_calls.lock().unwrap().push((
                input.to_owned().into_boxed_bitslice(),
                values.0,
                values.1,
            ));
            self.inner.insert(input, values)
        }
    }

    #[test]
    fn test_idpf_poplar() {
        let input = bitbox![0, 1, 1, 0, 1].into();
        let (public_share, keys) = idpf::gen::<_, _, _, PrgAes128, 16>(
            &input,
            Vec::from([Poplar1IdpfValue::new([Field64::one(), Field64::one()]); 4]),
            Poplar1IdpfValue::new([Field255::one(), Field255::one()]),
        )
        .unwrap();

        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::one(), Field64::one()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![1].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::zero(), Field64::zero()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0, 1].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::one(), Field64::one()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0, 0].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::zero(), Field64::zero()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![1, 0].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::zero(), Field64::zero()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![1, 1].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::zero(), Field64::zero()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0, 1, 1].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::one(), Field64::one()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0, 1, 1, 0].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::one(), Field64::one()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0, 1, 1, 0, 1].into(),
            &IdpfOutputShare::Leaf(Poplar1IdpfValue::new([Field255::one(), Field255::one()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0, 1, 1, 0, 0].into(),
            &IdpfOutputShare::Leaf(Poplar1IdpfValue::new([Field255::zero(), Field255::zero()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![1, 0, 1, 0, 0].into(),
            &IdpfOutputShare::Leaf(Poplar1IdpfValue::new([Field255::zero(), Field255::zero()])),
            &mut NoCache::new(),
            &mut NoCache::new(),
        );
    }

    fn check_idpf_poplar_evaluation<P, const L: usize>(
        public_share: &IdpfPublicShare<Poplar1IdpfValue<Field64>, Poplar1IdpfValue<Field255>, L>,
        keys: &[Seed<L>; 2],
        prefix: &IdpfInput,
        expected_output: &IdpfOutputShare<Poplar1IdpfValue<Field64>, Poplar1IdpfValue<Field255>>,
        cache_0: &mut dyn IdpfCache<L>,
        cache_1: &mut dyn IdpfCache<L>,
    ) where
        P: Prg<L>,
    {
        let share_0 = idpf::eval::<_, _, P, L>(0, public_share, &keys[0], prefix, cache_0).unwrap();
        let share_1 = idpf::eval::<_, _, P, L>(1, public_share, &keys[1], prefix, cache_1).unwrap();
        let output = share_0.merge(share_1).unwrap();
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
        let input = bits.clone().into();

        let mut inner_values = Vec::with_capacity(INPUT_LEN - 1);
        let mut prng = Prng::new().unwrap();
        for _ in 0..INPUT_LEN - 1 {
            inner_values.push(Poplar1IdpfValue::new([
                Field64::one(),
                prng.next().unwrap(),
            ]));
        }
        let leaf_values =
            Poplar1IdpfValue::new([Field255::one(), Prng::new().unwrap().next().unwrap()]);

        let (public_share, keys) =
            idpf::gen::<_, _, _, PrgAes128, 16>(&input, inner_values.clone(), leaf_values).unwrap();
        let mut cache_0 = RingBufferCache::new(3);
        let mut cache_1 = RingBufferCache::new(3);

        for (level, values) in inner_values.iter().enumerate() {
            let mut prefix = BitBox::from_bitslice(&bits[..=level]).into();
            check_idpf_poplar_evaluation::<PrgAes128, 16>(
                &public_share,
                &keys,
                &prefix,
                &IdpfOutputShare::Inner(*values),
                &mut cache_0,
                &mut cache_1,
            );
            let flipped_bit = !prefix[level];
            prefix.index.set(level, flipped_bit);
            check_idpf_poplar_evaluation::<PrgAes128, 16>(
                &public_share,
                &keys,
                &prefix,
                &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::zero(), Field64::zero()])),
                &mut cache_0,
                &mut cache_1,
            );
        }
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &input,
            &IdpfOutputShare::Leaf(leaf_values),
            &mut cache_0,
            &mut cache_1,
        );
        let mut modified_bits = bits.clone();
        modified_bits.set(INPUT_LEN - 1, !bits[INPUT_LEN - 1]);
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &modified_bits.into(),
            &IdpfOutputShare::Leaf(Poplar1IdpfValue::new([Field255::zero(), Field255::zero()])),
            &mut cache_0,
            &mut cache_1,
        );
    }

    #[test]
    fn idpf_poplar_cache_behavior() {
        let bits = bitbox![0, 1, 1, 1, 0, 1, 0, 0];
        let input = bits.into();

        let mut inner_values = Vec::with_capacity(7);
        let mut prng = Prng::new().unwrap();
        for _ in 0..7 {
            inner_values.push(Poplar1IdpfValue::new([
                Field64::one(),
                prng.next().unwrap(),
            ]));
        }
        let leaf_values =
            Poplar1IdpfValue::new([Field255::one(), Prng::new().unwrap().next().unwrap()]);

        let (public_share, keys) =
            idpf::gen::<_, _, _, PrgAes128, 16>(&input, inner_values.clone(), leaf_values).unwrap();
        let mut cache_0 = SnoopingCache::new(HashMapCache::new());
        let mut cache_1 = HashMapCache::new();

        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![1, 1, 0, 0].into(),
            &IdpfOutputShare::Inner(Poplar1IdpfValue::new([Field64::zero(), Field64::zero()])),
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
            vec![bitbox![1, 1, 0], bitbox![1, 1], bitbox![1]],
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
                bitbox![1],
                bitbox![1, 1],
                bitbox![1, 1, 0],
                bitbox![1, 1, 0, 0]
            ],
        );

        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0].into(),
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
                .collect::<Vec<BitBox>>(),
            Vec::<BitBox>::new(),
        );
        assert_eq!(
            cache_0
                .insert_calls
                .lock()
                .unwrap()
                .drain(..)
                .map(|(input, _, _)| input)
                .collect::<Vec<_>>(),
            vec![bitbox![0]],
        );

        check_idpf_poplar_evaluation::<PrgAes128, 16>(
            &public_share,
            &keys,
            &bitbox![0, 1].into(),
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
            vec![bitbox![0]],
        );
        assert_eq!(
            cache_0
                .insert_calls
                .lock()
                .unwrap()
                .drain(..)
                .map(|(input, _, _)| input)
                .collect::<Vec<_>>(),
            vec![bitbox![0, 1]],
        );

        check_idpf_poplar_evaluation::<PrgAes128, 16>(
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
                bitbox![0, 1, 1, 1, 0, 1, 0],
                bitbox![0, 1, 1, 1, 0, 1],
                bitbox![0, 1, 1, 1, 0],
                bitbox![0, 1, 1, 1],
                bitbox![0, 1, 1],
                bitbox![0, 1],
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
                bitbox![0, 1, 1],
                bitbox![0, 1, 1, 1],
                bitbox![0, 1, 1, 1, 0],
                bitbox![0, 1, 1, 1, 0, 1],
                bitbox![0, 1, 1, 1, 0, 1, 0],
            ],
        );

        check_idpf_poplar_evaluation::<PrgAes128, 16>(
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
            vec![bitbox![0, 1, 1, 1, 0, 1, 0]],
        );
        assert!(cache_0.insert_calls.lock().unwrap().is_empty());
    }

    #[test]
    fn idpf_poplar_lossy_cache() {
        let bits = bitbox![1, 0, 0, 1, 1, 0, 1, 0];
        let input = bits.into();

        let mut inner_values = Vec::with_capacity(7);
        let mut prng = Prng::new().unwrap();
        for _ in 0..7 {
            inner_values.push(Poplar1IdpfValue::new([
                Field64::one(),
                prng.next().unwrap(),
            ]));
        }
        let leaf_values =
            Poplar1IdpfValue::new([Field255::one(), Prng::new().unwrap().next().unwrap()]);

        let (public_share, keys) =
            idpf::gen::<_, _, _, PrgAes128, 16>(&input, inner_values.clone(), leaf_values).unwrap();
        let mut cache_0 = LossyCache::new();
        let mut cache_1 = LossyCache::new();

        for (level, values) in inner_values.iter().enumerate() {
            check_idpf_poplar_evaluation::<PrgAes128, 16>(
                &public_share,
                &keys,
                &input[..=level].to_owned().into(),
                &IdpfOutputShare::Inner(*values),
                &mut cache_0,
                &mut cache_1,
            );
        }
        check_idpf_poplar_evaluation::<PrgAes128, 16>(
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
        idpf::gen::<_, _, _, PrgAes128, 16>(
            &bitbox![].into(),
            Vec::<Poplar1IdpfValue<Field64>>::new(),
            Poplar1IdpfValue::new([Field255::zero(); 2]),
        )
        .unwrap_err();

        let (public_share, keys) = idpf::gen::<_, _, _, PrgAes128, 16>(
            &bitbox![0;10].into(),
            Vec::from([Poplar1IdpfValue::new([Field64::zero(); 2]); 9]),
            Poplar1IdpfValue::new([Field255::zero(); 2]),
        )
        .unwrap();

        // Wrong number of values.
        idpf::gen::<_, _, _, PrgAes128, 16>(
            &bitbox![0; 10].into(),
            Vec::from([Poplar1IdpfValue::new([Field64::zero(); 2]); 8]),
            Poplar1IdpfValue::new([Field255::zero(); 2]),
        )
        .unwrap_err();
        idpf::gen::<_, _, _, PrgAes128, 16>(
            &bitbox![0; 10].into(),
            Vec::from([Poplar1IdpfValue::new([Field64::zero(); 2]); 10]),
            Poplar1IdpfValue::new([Field255::zero(); 2]),
        )
        .unwrap_err();

        // Evaluating with empty prefix.
        assert!(idpf::eval::<_, _, PrgAes128, 16>(
            0,
            &public_share,
            &keys[0],
            &bitbox![].into(),
            &mut NoCache::new(),
        )
        .is_err());
        // Evaluating with too-long prefix.
        assert!(idpf::eval::<_, _, PrgAes128, 16>(
            0,
            &public_share,
            &keys[0],
            &bitbox![0; 11].into(),
            &mut NoCache::new(),
        )
        .is_err());
    }

    #[test]
    fn idpf_poplar_public_share_round_trip() {
        let public_share = IdpfPublicShare {
            inner_correction_words: Vec::from([
                IdpfCorrectionWord {
                    seed: [0xab; 16],
                    control_bits: [Choice::from(1), Choice::from(0)],
                    value: Poplar1IdpfValue::new([
                        Field64::try_from(83261u64).unwrap(),
                        Field64::try_from(125159u64).unwrap(),
                    ]),
                },
                IdpfCorrectionWord{
                    seed: [0xcd;16],
                    control_bits: [Choice::from(0), Choice::from(1)],
                    value: Poplar1IdpfValue::new([
                        Field64::try_from(17614120u64).unwrap(),
                        Field64::try_from(20674u64).unwrap(),
                    ]),
                },
            ]),
            leaf_correction_word: IdpfCorrectionWord {
                seed: [0xff; 16],
                control_bits: [Choice::from(1), Choice::from(1)],
                value: Poplar1IdpfValue::new([
                    Field255::one(),
                    Field255::get_decoded(
                        b"\x12\x34\x56\x78\x9a\xbc\xde\xf0\x12\x34\x56\x78\x9a\xbc\xde\xf0\x12\x34\x56\x78\x9a\xbc\xde\xf0\x12\x34\x56\x78\x9a\xbc\xde\xf0",
                    ).unwrap(),
                ]),
            },
        };
        let message = hex::decode(concat!(
            "39",                               // packed control bit correction words (0b00111001)
            "abababababababababababababababab", // seed correction word, first level
            "000000000001453d",                 // field element correction word
            "000000000001e8e7",                 // field element correction word, continued
            "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd", // seed correction word, second level
            "00000000010cc528",                 // field element correction word
            "00000000000050c2",                 // field element correction word, continued
            "ffffffffffffffffffffffffffffffff", // seed correction word, third level
            "0000000000000000000000000000000000000000000000000000000000000001", // field element correction word, leaf field
            "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0", // field element correction word, continued
        ))
        .unwrap();
        let encoded = public_share.get_encoded();
        let decoded = IdpfPublicShare::get_decoded_with_param(&3, &message).unwrap();
        assert_eq!(public_share, decoded);
        assert_eq!(message, encoded);

        // check serialization of packed control bits when they span multiple bytes:
        let public_share = IdpfPublicShare {
            inner_correction_words: Vec::from([
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(1), Choice::from(1)],
                    value: Poplar1IdpfValue::new([Field64::zero(), Field64::zero()]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(1), Choice::from(1)],
                    value: Poplar1IdpfValue::new([Field64::zero(), Field64::zero()]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(1), Choice::from(0)],
                    value: Poplar1IdpfValue::new([Field64::zero(), Field64::zero()]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(1), Choice::from(1)],
                    value: Poplar1IdpfValue::new([Field64::zero(), Field64::zero()]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(1), Choice::from(1)],
                    value: Poplar1IdpfValue::new([Field64::zero(), Field64::zero()]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(0), Choice::from(1)],
                    value: Poplar1IdpfValue::new([Field64::zero(), Field64::zero()]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(1), Choice::from(1)],
                    value: Poplar1IdpfValue::new([Field64::zero(), Field64::zero()]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(1), Choice::from(1)],
                    value: Poplar1IdpfValue::new([Field64::zero(), Field64::zero()]),
                },
            ]),
            leaf_correction_word: IdpfCorrectionWord {
                seed: [0; 16],
                control_bits: [Choice::from(0), Choice::from(1)],
                value: Poplar1IdpfValue::new([Field255::zero(), Field255::zero()]),
            },
        };
        let message = hex::decode(concat!(
            "02fbdf", // packed correction word control bits, 0b10_11111011_11011111, encoded big-endian
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
        let decoded = IdpfPublicShare::get_decoded_with_param(&9, &message).unwrap();
        assert_eq!(public_share, decoded);
        assert_eq!(message, encoded);
    }

    #[test]
    fn idpf_poplar_public_share_control_bit_codec() {
        let test_cases = [
            (&[false, true][..], &[0b10][..]),
            (
                &[false, false, true, false, false, true][..],
                &[0b10_0100u8][..],
            ),
            (
                &[
                    true, true, false, true, false, false, false, false, true, true,
                ][..],
                &[0b11, 0b0000_1011][..],
            ),
            (
                &[
                    true, true, false, true, false, true, true, true, false, true, false, true,
                    false, false, true, false,
                ][..],
                &[0b0100_1010, 0b1110_1011][..],
            ),
            (
                &[
                    true, true, true, true, true, false, true, true, false, true, true, true,
                    false, true, false, true, false, false, true, false, true, true,
                ][..],
                &[0b11_0100, 0b1010_1110, 0b1101_1111][..],
            ),
        ];

        for (control_bits, serialized_control_bits) in test_cases {
            let public_share = IdpfPublicShare::<
                Poplar1IdpfValue<Field64>,
                Poplar1IdpfValue<Field255>,
                16,
            > {
                inner_correction_words: control_bits[..control_bits.len() - 2]
                    .chunks(2)
                    .map(|chunk| IdpfCorrectionWord {
                        seed: [0; 16],
                        control_bits: [Choice::from(chunk[0] as u8), Choice::from(chunk[1] as u8)],
                        value: Poplar1IdpfValue::new([Field64::zero(); 2]),
                    })
                    .collect(),
                leaf_correction_word: IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [
                        Choice::from(control_bits[control_bits.len() - 2] as u8),
                        Choice::from(control_bits[control_bits.len() - 1] as u8),
                    ],
                    value: Poplar1IdpfValue::new([Field255::zero(); 2]),
                },
            };

            let mut serialized_public_share = serialized_control_bits.to_owned();
            let idpf_bits = control_bits.len() / 2;
            let size_seeds = 16 * idpf_bits;
            let size_field_vecs =
                Field64::ENCODED_SIZE * 2 * (idpf_bits - 1) + Field255::ENCODED_SIZE * 2;
            serialized_public_share.resize(
                serialized_control_bits.len() + size_seeds + size_field_vecs,
                0,
            );

            assert_eq!(public_share.get_encoded(), serialized_public_share);
            assert_eq!(
                IdpfPublicShare::get_decoded_with_param(&idpf_bits, &serialized_public_share)
                    .unwrap(),
                public_share
            );
        }
    }

    #[test]
    fn idpf_poplar_public_share_unused_bits() {
        let mut buf = vec![0u8; 4096];

        buf[0] = 1 << 2;
        let err =
            IdpfPublicShare::<Field64, Field255, 16>::decode_with_param(&1, &mut Cursor::new(&buf))
                .unwrap_err();
        assert_matches!(err, CodecError::UnexpectedValue);
        let err =
            IdpfPublicShare::<Field64, Field255, 16>::decode_with_param(&5, &mut Cursor::new(&buf))
                .unwrap_err();
        assert_matches!(err, CodecError::UnexpectedValue);

        buf[0] = 1 << 4;
        let err =
            IdpfPublicShare::<Field64, Field255, 16>::decode_with_param(&2, &mut Cursor::new(&buf))
                .unwrap_err();
        assert_matches!(err, CodecError::UnexpectedValue);

        buf[0] = 1 << 6;
        let err =
            IdpfPublicShare::<Field64, Field255, 16>::decode_with_param(&3, &mut Cursor::new(&buf))
                .unwrap_err();
        assert_matches!(err, CodecError::UnexpectedValue);
    }

    /// Stores a test vector for the IDPF key generation algorithm.
    struct IdpfTestVector {
        /// The number of bits in IDPF inputs.
        bits: usize,
        /// The IDPF input provided to the key generation algorithm.
        alpha: IdpfInput,
        /// The IDPF output values, at each inner level, provided to the key generation algorithm.
        beta_inner: Vec<Poplar1IdpfValue<Field64>>,
        /// The IDPF output values for the leaf level, provided to the key generation algorithm.
        beta_leaf: Poplar1IdpfValue<Field255>,
        /// The two keys returned by the key generation algorithm.
        keys: [[u8; 16]; 2],
        /// The public share returned by the key generation algorithm.
        public_share: Vec<u8>,
    }

    /// Load a test vector for Idpf key generation.
    fn load_idpfpoplar_test_vector() -> IdpfTestVector {
        let test_vec: serde_json::Value =
            serde_json::from_str(include_str!("vdaf/test_vec/03/IdpfPoplarAes128_0.json")).unwrap();
        let test_vec_obj = test_vec.as_object().unwrap();

        let bits = test_vec_obj
            .get("bits")
            .unwrap()
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap();

        let alpha_str = test_vec_obj.get("alpha").unwrap().as_str().unwrap();
        let alpha_bignum = BigUint::from_str(alpha_str).unwrap();
        let zero_bignum = BigUint::from(0u8);
        let one_bignum = BigUint::from(1u8);
        let alpha_bits = (0..bits)
            .map(|level| (&alpha_bignum >> (bits - level - 1)) & &one_bignum != zero_bignum)
            .collect::<BitVec>();
        let alpha = alpha_bits.into();

        let beta_inner_level_array = test_vec_obj.get("beta_inner").unwrap().as_array().unwrap();
        let beta_inner = beta_inner_level_array
            .iter()
            .map(|array| {
                Poplar1IdpfValue::new([
                    Field64::from(array[0].as_str().unwrap().parse::<u64>().unwrap()),
                    Field64::from(array[1].as_str().unwrap().parse::<u64>().unwrap()),
                ])
            })
            .collect::<Vec<_>>();

        let beta_leaf_array = test_vec_obj.get("beta_leaf").unwrap().as_array().unwrap();
        let beta_leaf = Poplar1IdpfValue::new([
            Field255::from(
                beta_leaf_array[0]
                    .as_str()
                    .unwrap()
                    .parse::<BigUint>()
                    .unwrap(),
            ),
            Field255::from(
                beta_leaf_array[1]
                    .as_str()
                    .unwrap()
                    .parse::<BigUint>()
                    .unwrap(),
            ),
        ]);

        let keys_array = test_vec_obj.get("keys").unwrap().as_array().unwrap();
        let keys = [
            hex::decode(keys_array[0].as_str().unwrap())
                .unwrap()
                .try_into()
                .unwrap(),
            hex::decode(keys_array[1].as_str().unwrap())
                .unwrap()
                .try_into()
                .unwrap(),
        ];

        let public_share_hex = test_vec_obj.get("public_share").unwrap();
        let public_share = hex::decode(public_share_hex.as_str().unwrap()).unwrap();

        IdpfTestVector {
            bits,
            alpha,
            beta_inner,
            beta_leaf,
            keys,
            public_share,
        }
    }

    #[test]
    fn idpf_poplar_public_share_deserialize() {
        // This encoded public share, and the expected struct below, are taken from the
        // 1Aes128 test vector.
        let data = hex::decode(concat!(
            "9a",
            "0000000000000000000000000000000000000000000000",
            "01eb3a1bd6b5fa4a4500000000000000000000000000000000",
            "ffffffff0000000022522c3fd5a33cac00000000000000000000000000000000",
            "ffffffff0000000069f41eee46542b6900000000000000000000000000000000",
            "00000000000000000000000000000000000000000000000000000000000000",
            "017d1fd6df94280145a0dcc933ceb706e9219d50e7c4f92fd8ca9a0ffb7d819646",
        ))
        .unwrap();
        let bits = 4;
        let public_share = IdpfPublicShare::<
            Poplar1IdpfValue<Field64>,
            Poplar1IdpfValue<Field255>,
            16,
        >::get_decoded_with_param(&bits, &data)
        .unwrap();

        let expected_public_share = IdpfPublicShare {
            inner_correction_words: Vec::from([
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(0), Choice::from(1)],
                    value: Poplar1IdpfValue::new([
                        Field64::from(1u64),
                        Field64::from(16949890756552313413u64),
                    ]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(0), Choice::from(1)],
                    value: Poplar1IdpfValue::new([
                        Field64::from(18446744069414584320u64),
                        Field64::from(2473087798058630316u64),
                    ]),
                },
                IdpfCorrectionWord {
                    seed: [0; 16],
                    control_bits: [Choice::from(1), Choice::from(0)],
                    value: Poplar1IdpfValue::new([
                        Field64::from(18446744069414584320u64),
                        Field64::from(7634761277030804329u64),
                    ]),
                },
            ]),
            leaf_correction_word: IdpfCorrectionWord {
                seed: [0; 16],
                control_bits: [Choice::from(0), Choice::from(1)],
                value: Poplar1IdpfValue::new([
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
                ]),
            },
        };

        assert_eq!(public_share, expected_public_share);
    }

    #[test]
    fn idpf_poplar_generate_test_vector() {
        let test_vector = load_idpfpoplar_test_vector();
        let (public_share, keys) = idpf::gen_with_rand_source::<_, _, _, PrgAes128, 16>(
            &test_vector.alpha,
            test_vector.beta_inner,
            test_vector.beta_leaf,
            |buf| {
                buf.fill(1);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(keys, [Seed([0x01; 16]), Seed([0x01; 16])]);
        assert_eq!(keys[0].0, test_vector.keys[0]);
        assert_eq!(keys[1].0, test_vector.keys[1]);

        let expected_public_share =
            IdpfPublicShare::get_decoded_with_param(&test_vector.bits, &test_vector.public_share)
                .unwrap();
        for (level, (correction_words, expected_correction_words)) in public_share
            .inner_correction_words
            .iter()
            .zip(expected_public_share.inner_correction_words.iter())
            .enumerate()
        {
            assert_eq!(
                correction_words, expected_correction_words,
                "layer {level} did not match\n{correction_words:#x?}\n{expected_correction_words:#x?}"
            );
        }
        assert_eq!(
            public_share.leaf_correction_word,
            expected_public_share.leaf_correction_word
        );

        assert_eq!(
            public_share, expected_public_share,
            "public share did not match\n{public_share:#x?}\n{expected_public_share:#x?}"
        );
        let encoded_public_share = public_share.get_encoded();
        assert_eq!(encoded_public_share, test_vector.public_share);
    }

    #[test]
    fn idpf_input_from_bytes_to_bytes() {
        let test_cases: &[&[u8]] = &[b"hello", b"banana", &[1], &[127], &[1, 2, 3, 4], &[]];
        for test_case in test_cases {
            assert_eq!(&IdpfInput::from_bytes(test_case).to_bytes(), test_case);
        }
    }

    #[test]
    fn idpf_input_from_bools_to_bytes() {
        let input = IdpfInput::from_bools(&[true; 7]);
        assert_eq!(input.to_bytes(), &[127]);
        let input = IdpfInput::from_bools(&[true; 9]);
        assert_eq!(input.to_bytes(), &[255, 1]);
    }
}
