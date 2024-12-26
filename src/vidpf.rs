// SPDX-License-Identifier: MPL-2.0

//! Verifiable Incremental Distributed Point Function (VIDPF).
//!
//! The VIDPF construction is specified in [[draft-mouris-cfrg-mastic]] and builds
//! on techniques from [[MST23]] and [[CP22]] to lift an IDPF to a VIDPF.
//!
//! [CP22]: https://eprint.iacr.org/2021/580
//! [MST23]: https://eprint.iacr.org/2023/080
//! [draft-mouris-cfrg-mastic]: https://datatracker.ietf.org/doc/draft-mouris-cfrg-mastic/02/

use core::{
    iter::zip,
    ops::{Add, AddAssign, BitXor, BitXorAssign, Index, Sub},
};

use bitvec::field::BitField;
use bitvec::prelude::{BitVec, Lsb0};
use rand_core::RngCore;
use std::fmt::Debug;
use std::io::{Cursor, Read};
use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable, ConstantTimeEq};

use crate::{
    bt::{BinaryTree, Node},
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::FieldElement,
    idpf::{conditional_swap_seed, conditional_xor_seeds, xor_seeds, IdpfInput, IdpfValue},
    vdaf::xof::{Seed, Xof, XofFixedKeyAes128, XofTurboShake128},
};

/// VIDPF errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VidpfError {
    /// Error when level does not fit in a 32-bit number.
    #[error("level is not representable as a 32-bit integer")]
    LevelTooBig,

    /// Error during VIDPF evaluation: tried to access a level index out of bounds.
    #[error("level index out of bounds")]
    IndexLevel,

    /// Error when input attribute has too few or many bits to be a path in an initialized
    /// VIDPF tree.
    #[error("invalid attribute length")]
    InvalidAttributeLength,

    /// Error when weight's length mismatches the length in weight's parameter.
    #[error("invalid weight length")]
    InvalidWeightLength,

    /// Failure when calling getrandom().
    #[error("getrandom: {0}")]
    GetRandom(#[from] getrandom::Error),
}

/// Represents the domain of an incremental point function.
pub type VidpfInput = IdpfInput;

/// Represents the codomain of an incremental point function.
pub trait VidpfValue: IdpfValue + Clone + Debug + PartialEq + ConstantTimeEq {}

#[derive(Clone, Debug)]
/// An instance of the VIDPF.
pub struct Vidpf<W: VidpfValue> {
    /// Any parameters required to instantiate a weight value.
    pub(crate) weight_parameter: W::ValueParameter,
}

impl<W: VidpfValue> Vidpf<W> {
    /// Creates a VIDPF instance.
    ///
    /// # Arguments
    ///
    /// * `weight_parameter`, any parameters required to instantiate a weight value.
    pub const fn new(weight_parameter: W::ValueParameter) -> Self {
        Self { weight_parameter }
    }

    /// Splits an incremental point function `F` into two private keys
    /// used by the aggregation servers, and a common public share.
    ///
    /// The incremental point function is defined as `F`: [`VidpfInput`] --> [`VidpfValue`]
    /// such that:
    ///
    /// ```txt
    /// F(x) = weight, if x is a prefix of the input.
    /// F(x) = 0,      if x is not a prefix of the input.
    /// ```
    ///
    /// # Arguments
    ///
    /// * `input`, determines the input of the function.
    /// * `weight`, determines the input's weight of the function.
    /// * `nonce`, a nonce, typically the same value provided to the
    ///   [`Client`](crate::vdaf::Client) and [`Aggregator`](crate::vdaf::Aggregator).
    ///   APIs.
    pub fn gen(
        &self,
        input: &VidpfInput,
        weight: &W,
        nonce: &[u8],
    ) -> Result<(VidpfPublicShare<W>, [VidpfKey; 2]), VidpfError> {
        let keys = [VidpfKey::generate()?, VidpfKey::generate()?];
        let public = self.gen_with_keys(&keys, input, weight, nonce)?;
        Ok((public, keys))
    }

    /// Produce the public share for the given keys, input, and weight.
    pub(crate) fn gen_with_keys(
        &self,
        keys: &[VidpfKey; 2],
        input: &VidpfInput,
        weight: &W,
        nonce: &[u8],
    ) -> Result<VidpfPublicShare<W>, VidpfError> {
        let mut seed = [keys[0].0, keys[1].0];
        let mut ctrl = [
            Choice::from(VidpfServerId::S0),
            Choice::from(VidpfServerId::S1),
        ];

        let n = input.len();
        let mut cw = Vec::with_capacity(n);

        for level in 0..n {
            let bit = Choice::from(u8::from(input.get(level).ok_or(VidpfError::IndexLevel)?));

            // Extend.
            let e = [Self::extend(&seed[0], nonce), Self::extend(&seed[1], nonce)];

            // Select the seed and control bit.
            let (seed_keep_0, seed_lose_0) = &mut (e[0].seed_right, e[0].seed_left);
            conditional_swap_seed(seed_keep_0, seed_lose_0, !bit);
            let (seed_keep_1, seed_lose_1) = &mut (e[1].seed_right, e[1].seed_left);
            conditional_swap_seed(seed_keep_1, seed_lose_1, !bit);
            let ctrl_keep_0 = Choice::conditional_select(&e[0].ctrl_left, &e[0].ctrl_right, bit);
            let ctrl_keep_1 = Choice::conditional_select(&e[1].ctrl_left, &e[1].ctrl_right, bit);

            // Compute the correction word seed and control bit.
            let cw_seed = xor_seeds(seed_lose_0, seed_lose_1);
            let cw_ctrl_left = e[0].ctrl_left ^ e[1].ctrl_left ^ bit ^ Choice::from(1);
            let cw_ctrl_right = e[0].ctrl_right ^ e[1].ctrl_right ^ bit;

            // Correct the seed and control bit.
            let seed_keep_0 = conditional_xor_seeds(seed_keep_0, &cw_seed, ctrl[0]);
            let seed_keep_1 = conditional_xor_seeds(seed_keep_1, &cw_seed, ctrl[1]);
            let cw_ctrl_keep = Choice::conditional_select(&cw_ctrl_left, &cw_ctrl_right, bit);
            let ctrl_keep_0 = ctrl_keep_0 ^ (ctrl[0] & cw_ctrl_keep);
            let ctrl_keep_1 = ctrl_keep_1 ^ (ctrl[1] & cw_ctrl_keep);

            // Convert.
            let weight_0;
            let weight_1;
            (seed[0], weight_0) = self.convert(seed_keep_0, nonce);
            (seed[1], weight_1) = self.convert(seed_keep_1, nonce);
            ctrl[0] = ctrl_keep_0;
            ctrl[1] = ctrl_keep_1;

            // Compute the correction word payload.
            let mut cw_weight = weight_1 - weight_0 + weight.clone();
            cw_weight.conditional_negate(ctrl[1]);

            // Compute the correction word node proof.
            let cw_proof = xor_proof(
                Self::node_proof(input, level, &seed[0])?,
                &Self::node_proof(input, level, &seed[1])?,
            );

            cw.push(VidpfCorrectionWord {
                seed: cw_seed,
                ctrl_left: cw_ctrl_left,
                ctrl_right: cw_ctrl_right,
                weight: cw_weight,
                proof: cw_proof,
            });
        }

        Ok(VidpfPublicShare { cw })
    }

    /// Evaluate a given VIDPF (comprised of the key and public share) at a given input.
    pub fn eval(
        &self,
        id: VidpfServerId,
        key: &VidpfKey,
        public: &VidpfPublicShare<W>,
        input: &VidpfInput,
        nonce: &[u8],
    ) -> Result<VidpfValueShare<W>, VidpfError> {
        let mut state = VidpfEvalState::init_from_key(id, key);
        let mut share = W::zero(&self.weight_parameter);

        let n = input.len();
        if n > public.cw.len() {
            return Err(VidpfError::InvalidAttributeLength);
        }
        for level in 0..n {
            (state, share) = self.eval_next(id, public, input, level, &state, nonce)?;
        }

        Ok(VidpfValueShare {
            share,
            proof: state.proof,
        })
    }

    /// Evaluates the entire `input` and produces a share of the
    /// input's weight. It reuses computation from previous levels available in the
    /// cache.
    pub(crate) fn eval_with_cache(
        &self,
        id: VidpfServerId,
        key: &VidpfKey,
        public: &VidpfPublicShare<W>,
        input: &VidpfInput,
        cache_tree: &mut BinaryTree<VidpfEvalCache<W>>,
        nonce: &[u8],
    ) -> Result<VidpfValueShare<W>, VidpfError> {
        let n = input.len();
        if n > public.cw.len() {
            return Err(VidpfError::InvalidAttributeLength);
        }

        let mut sub_tree = cache_tree.root.get_or_insert_with(|| {
            Box::new(Node::new(VidpfEvalCache {
                state: VidpfEvalState::init_from_key(id, key),
                share: W::zero(&self.weight_parameter), // not used
            }))
        });

        for (level, bit) in input.iter().enumerate() {
            sub_tree = if !bit {
                if sub_tree.left.is_none() {
                    let (new_state, new_share) =
                        self.eval_next(id, public, input, level, &sub_tree.value.state, nonce)?;
                    sub_tree.left = Some(Box::new(Node::new(VidpfEvalCache {
                        state: new_state,
                        share: new_share,
                    })));
                }
                sub_tree.left.as_mut().unwrap()
            } else {
                if sub_tree.right.is_none() {
                    let (new_state, new_share) =
                        self.eval_next(id, public, input, level, &sub_tree.value.state, nonce)?;
                    sub_tree.right = Some(Box::new(Node::new(VidpfEvalCache {
                        state: new_state,
                        share: new_share,
                    })));
                }
                sub_tree.right.as_mut().unwrap()
            }
        }
        Ok(sub_tree.value.to_share())
    }

    /// Evaluates the `input` at the given level using the provided initial
    /// state, and returns a new state and a share of the input's weight at that level.
    fn eval_next(
        &self,
        id: VidpfServerId,
        public: &VidpfPublicShare<W>,
        input: &VidpfInput,
        level: usize,
        state: &VidpfEvalState,
        nonce: &[u8],
    ) -> Result<(VidpfEvalState, W), VidpfError> {
        let bit = Choice::from(u8::from(input.get(level).ok_or(VidpfError::IndexLevel)?));
        let cw = public.cw.get(level).ok_or(VidpfError::IndexLevel)?;

        // Extend.
        let e = Self::extend(&state.seed, nonce);

        // Select the seed and control bit.
        let (seed_keep, seed_lose) = &mut (e.seed_right, e.seed_left);
        conditional_swap_seed(seed_keep, seed_lose, !bit);
        let ctrl_keep = Choice::conditional_select(&e.ctrl_left, &e.ctrl_right, bit);

        // Correct the seed and control bit.
        let seed_keep = conditional_xor_seeds(seed_keep, &cw.seed, state.control_bit);
        let cw_ctrl_keep = Choice::conditional_select(&cw.ctrl_left, &cw.ctrl_right, bit);
        let next_ctrl = ctrl_keep ^ (state.control_bit & cw_ctrl_keep);

        // Convert and correct the payload.
        let (next_seed, w) = self.convert(seed_keep, nonce);
        let mut weight = <W as IdpfValue>::conditional_select(
            &<W as IdpfValue>::zero(&self.weight_parameter),
            &cw.weight,
            next_ctrl,
        );
        weight += w;
        weight.conditional_negate(Choice::from(id));

        // Compute and correct the node proof.
        let node_proof = conditional_xor_proof(
            Self::node_proof(input, level, &next_seed)?,
            &cw.proof,
            next_ctrl,
        );

        // Compute the onehot proof.
        let onehot_proof = xor_proof(
            state.proof,
            &Self::hash_proof(xor_proof(state.proof, &node_proof)),
        );

        let next_state = VidpfEvalState {
            seed: next_seed,
            control_bit: next_ctrl,
            proof: onehot_proof,
        };

        Ok((next_state, weight))
    }

    pub(crate) fn get_root_weight_share(
        &self,
        id: VidpfServerId,
        key: &VidpfKey,
        public_share: &VidpfPublicShare<W>,
        cache_tree: &mut BinaryTree<VidpfEvalCache<W>>,
        nonce: &[u8],
    ) -> Result<W, VidpfError> {
        Ok(self
            .eval_with_cache(
                id,
                key,
                public_share,
                &VidpfInput::from_bools(&[false]),
                cache_tree,
                nonce,
            )?
            .share
            + self
                .eval_with_cache(
                    id,
                    key,
                    public_share,
                    &VidpfInput::from_bools(&[true]),
                    cache_tree,
                    nonce,
                )?
                .share)
    }

    pub(crate) fn eval_root(
        &self,
        id: VidpfServerId,
        key: &VidpfKey,
        public_share: &VidpfPublicShare<W>,
        nonce: &[u8],
    ) -> Result<W, VidpfError> {
        Ok(self
            .eval(
                id,
                key,
                public_share,
                &VidpfInput::from_bools(&[false]),
                nonce,
            )?
            .share
            + self
                .eval(
                    id,
                    key,
                    public_share,
                    &VidpfInput::from_bools(&[true]),
                    nonce,
                )?
                .share)
    }

    fn extend(seed: &VidpfSeed, nonce: &[u8]) -> ExtendedSeed {
        let mut rng = XofFixedKeyAes128::seed_stream(&Seed(*seed), VidpfDomainSepTag::PRG, nonce);

        let mut seed_left = VidpfSeed::default();
        let mut seed_right = VidpfSeed::default();
        rng.fill_bytes(&mut seed_left);
        rng.fill_bytes(&mut seed_right);
        // Use the LSB of seeds as control bits, and clears the bit,
        // i.e., seeds produced by `prg` always have their LSB = 0.
        // This ensures `prg` costs two AES calls only.
        let ctrl_left = Choice::from(seed_left[0] & 0x01);
        let ctrl_right = Choice::from(seed_right[0] & 0x01);
        seed_left[0] &= 0xFE;
        seed_right[0] &= 0xFE;

        ExtendedSeed {
            seed_left,
            ctrl_left,
            seed_right,
            ctrl_right,
        }
    }

    fn convert(&self, seed: VidpfSeed, nonce: &[u8]) -> (VidpfSeed, W) {
        let mut rng =
            XofFixedKeyAes128::seed_stream(&Seed(seed), VidpfDomainSepTag::CONVERT, nonce);

        let mut out_seed = VidpfSeed::default();
        rng.fill_bytes(&mut out_seed);
        let value = <W as IdpfValue>::generate(&mut rng, &self.weight_parameter);

        (out_seed, value)
    }

    fn node_proof(
        input: &VidpfInput,
        level: usize,
        seed: &VidpfSeed,
    ) -> Result<VidpfProof, VidpfError> {
        let mut shake = XofTurboShake128::from_seed_slice(&seed[..], VidpfDomainSepTag::NODE_PROOF);
        for chunk128 in input
            .index(..=level)
            .chunks(128)
            .map(BitField::load_le::<u128>)
            .map(u128::to_le_bytes)
        {
            shake.update(&chunk128);
        }
        shake.update(
            &u16::try_from(level)
                .map_err(|_e| VidpfError::LevelTooBig)?
                .to_le_bytes(),
        );
        let mut rng = shake.into_seed_stream();

        let mut proof = VidpfProof::default();
        rng.fill_bytes(&mut proof);

        Ok(proof)
    }

    fn hash_proof(mut proof: VidpfProof) -> VidpfProof {
        let mut rng = XofTurboShake128::seed_stream(
            &Seed(Default::default()),
            VidpfDomainSepTag::NODE_PROOF_ADJUST,
            &proof,
        );
        rng.fill_bytes(&mut proof);

        proof
    }
}

/// VIDPF domain separation tag.
///
/// Contains the domain separation tags for invoking different oracles.
struct VidpfDomainSepTag;
impl VidpfDomainSepTag {
    const PRG: &'static [u8] = b"Prg";
    const CONVERT: &'static [u8] = b"Convert";
    const NODE_PROOF: &'static [u8] = b"NodeProof";
    const NODE_PROOF_ADJUST: &'static [u8] = b"NodeProofAdjust";
}

/// VIDPF key.
///
/// Private key of an aggregation server.
pub type VidpfKey = Seed<VIDPF_SEED_SIZE>;

/// VIDPF server ID.
///
/// Identifies the two aggregation servers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VidpfServerId {
    /// S0 is the first server.
    S0,
    /// S1 is the second server.
    S1,
}

impl From<VidpfServerId> for Choice {
    fn from(value: VidpfServerId) -> Self {
        match value {
            VidpfServerId::S0 => Self::from(0),
            VidpfServerId::S1 => Self::from(1),
        }
    }
}

/// VIDPF correction word.
#[derive(Clone, Debug)]
struct VidpfCorrectionWord<W: VidpfValue> {
    seed: VidpfSeed,
    ctrl_left: Choice,
    ctrl_right: Choice,
    weight: W,
    proof: VidpfProof,
}

impl<W: VidpfValue> ConstantTimeEq for VidpfCorrectionWord<W> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.seed.ct_eq(&other.seed)
            & self.ctrl_left.ct_eq(&other.ctrl_left)
            & self.ctrl_right.ct_eq(&other.ctrl_right)
            & self.weight.ct_eq(&other.weight)
            & self.proof.ct_eq(&other.proof)
    }
}

impl<W: VidpfValue> PartialEq for VidpfCorrectionWord<W>
where
    W: ConstantTimeEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

/// VIDPF public share.
#[derive(Clone, Debug, PartialEq)]
pub struct VidpfPublicShare<W: VidpfValue> {
    cw: Vec<VidpfCorrectionWord<W>>,
}

impl<W: VidpfValue> Encode for VidpfPublicShare<W> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        // Control bits need to be written within each byte in LSB-to-MSB order, and assigned into
        // bytes in big-endian order. Thus, the first four levels will have their control bits
        // encoded in the last byte, and the last levels will have their control bits encoded in the
        // first byte.
        let mut control_bits: BitVec<u8, Lsb0> = BitVec::with_capacity(self.cw.len() * 2);
        for correction_words in self.cw.iter() {
            control_bits.extend(
                [
                    bool::from(correction_words.ctrl_left),
                    bool::from(correction_words.ctrl_right),
                ]
                .iter(),
            );
        }
        control_bits.set_uninitialized(false);
        let mut packed_control = control_bits.into_vec();
        bytes.append(&mut packed_control);

        for VidpfCorrectionWord {
            seed,
            ctrl_left: _,
            ctrl_right: _,
            weight,
            proof,
        } in self.cw.iter()
        {
            bytes.extend_from_slice(seed);
            weight.encode(bytes)?;
            bytes.extend_from_slice(proof);
        }

        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        let control_bits_count = (self.cw.len()) * 2;
        let mut len = 0;
        len += (control_bits_count + 7) / 8; // control bits
        let cw_encoded_len = VIDPF_SEED_SIZE
            + VIDPF_PROOF_SIZE
            + self.cw.first().and_then(|cw| cw.weight.encoded_len())?;
        len += self.cw.len() * cw_encoded_len;
        Some(len)
    }
}

impl<W: VidpfValue> ParameterizedDecode<(usize, W::ValueParameter)> for VidpfPublicShare<W> {
    fn decode_with_param(
        (bits, weight_parameter): &(usize, W::ValueParameter),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let packed_control_len = (bits + 3) / 4;
        let mut packed = vec![0u8; packed_control_len];
        bytes.read_exact(&mut packed)?;
        let unpacked_control_bits: BitVec<u8, Lsb0> = BitVec::from_vec(packed);

        let mut cw = Vec::<VidpfCorrectionWord<W>>::with_capacity(*bits);
        for chunk in unpacked_control_bits[0..(bits) * 2].chunks(2) {
            let ctrl_left = (chunk[0] as u8).into();
            let ctrl_right = (chunk[1] as u8).into();
            let seed = Seed::decode(bytes)?.0;
            let weight = W::decode_with_param(weight_parameter, bytes)?;
            let mut proof = [0u8; VIDPF_PROOF_SIZE];
            bytes.read_exact(&mut proof)?;
            cw.push(VidpfCorrectionWord {
                seed,
                ctrl_left,
                ctrl_right,
                weight,
                proof,
            })
        }
        Ok(Self { cw })
    }
}

/// VIDPF evaluation state.
#[derive(Debug)]
pub(crate) struct VidpfEvalState {
    seed: VidpfSeed,
    control_bit: Choice,
    proof: VidpfProof,
}

impl VidpfEvalState {
    fn init_from_key(id: VidpfServerId, key: &VidpfKey) -> Self {
        Self {
            seed: key.0,
            control_bit: Choice::from(id),
            proof: VidpfProof::default(),
        }
    }
}

/// VIDPF evaluation cache.
#[derive(Debug)]
pub(crate) struct VidpfEvalCache<W: VidpfValue> {
    state: VidpfEvalState,
    share: W,
}

impl<W: VidpfValue> VidpfEvalCache<W> {
    fn to_share(&self) -> VidpfValueShare<W> {
        VidpfValueShare::<W> {
            share: self.share.clone(),
            proof: self.state.proof,
        }
    }
}

/// VIDPF value share, the output of [`eval()`](Vidpf::eval).
pub struct VidpfValueShare<W: VidpfValue> {
    /// Secret share of the input's weight.
    pub share: W,
    /// Proof used to verify the share.
    pub proof: VidpfProof,
}

const VIDPF_PROOF_SIZE: usize = 32;
const VIDPF_SEED_SIZE: usize = 16;

/// Allows to validate user input and shares after evaluation.
type VidpfProof = [u8; VIDPF_PROOF_SIZE];

fn xor_proof(mut lhs: VidpfProof, rhs: &VidpfProof) -> VidpfProof {
    zip(&mut lhs, rhs).for_each(|(a, b)| a.bitxor_assign(b));
    lhs
}

fn conditional_xor_proof(mut lhs: VidpfProof, rhs: &VidpfProof, choice: Choice) -> VidpfProof {
    zip(&mut lhs, rhs).for_each(|(a, b)| a.conditional_assign(&a.bitxor(b), choice));
    lhs
}

/// Feeds a pseudorandom generator during evaluation.
type VidpfSeed = [u8; VIDPF_SEED_SIZE];

/// Output of [`extend()`](Vidpf::extend).
struct ExtendedSeed {
    seed_left: VidpfSeed,
    ctrl_left: Choice,
    seed_right: VidpfSeed,
    ctrl_right: Choice,
}

/// Represents an array of field elements that implements the [`VidpfValue`] trait.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct VidpfWeight<F: FieldElement>(pub(crate) Vec<F>);

impl<F: FieldElement> From<Vec<F>> for VidpfWeight<F> {
    fn from(value: Vec<F>) -> Self {
        Self(value)
    }
}

impl<F: FieldElement> AsRef<[F]> for VidpfWeight<F> {
    fn as_ref(&self) -> &[F] {
        &self.0
    }
}

impl<F: FieldElement> VidpfValue for VidpfWeight<F> {}

impl<F: FieldElement> IdpfValue for VidpfWeight<F> {
    /// The parameter determines the number of field elements in the vector.
    type ValueParameter = usize;

    fn generate<S: RngCore>(seed_stream: &mut S, length: &Self::ValueParameter) -> Self {
        Self(
            (0..*length)
                .map(|_| <F as IdpfValue>::generate(seed_stream, &()))
                .collect(),
        )
    }

    fn zero(length: &Self::ValueParameter) -> Self {
        Self((0..*length).map(|_| <F as IdpfValue>::zero(&())).collect())
    }

    /// Panics if weight lengths are different.
    fn conditional_select(lhs: &Self, rhs: &Self, choice: Choice) -> Self {
        assert_eq!(
            lhs.0.len(),
            rhs.0.len(),
            "{}",
            VidpfError::InvalidWeightLength
        );

        Self(
            zip(&lhs.0, &rhs.0)
                .map(|(a, b)| <F as IdpfValue>::conditional_select(a, b, choice))
                .collect(),
        )
    }
}

impl<F: FieldElement> ConditionallyNegatable for VidpfWeight<F> {
    fn conditional_negate(&mut self, choice: Choice) {
        self.0.iter_mut().for_each(|a| a.conditional_negate(choice));
    }
}

impl<F: FieldElement> Add for VidpfWeight<F> {
    type Output = Self;

    /// Panics if weight lengths are different.
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.0.len(),
            rhs.0.len(),
            "{}",
            VidpfError::InvalidWeightLength
        );

        Self(zip(self.0, rhs.0).map(|(a, b)| a.add(b)).collect())
    }
}

impl<F: FieldElement> AddAssign for VidpfWeight<F> {
    /// Panics if weight lengths are different.
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(
            self.0.len(),
            rhs.0.len(),
            "{}",
            VidpfError::InvalidWeightLength
        );

        zip(&mut self.0, rhs.0).for_each(|(a, b)| a.add_assign(b));
    }
}

impl<F: FieldElement> Sub for VidpfWeight<F> {
    type Output = Self;

    /// Panics if weight lengths are different.
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.0.len(),
            rhs.0.len(),
            "{}",
            VidpfError::InvalidWeightLength
        );

        Self(zip(self.0, rhs.0).map(|(a, b)| a.sub(b)).collect())
    }
}

impl<F: FieldElement> ConstantTimeEq for VidpfWeight<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0[..].ct_eq(&other.0[..])
    }
}

impl<F: FieldElement> Encode for VidpfWeight<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        for e in &self.0 {
            F::encode(e, bytes)?;
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.0.len() * F::ENCODED_SIZE)
    }
}

impl<F: FieldElement> ParameterizedDecode<<Self as IdpfValue>::ValueParameter> for VidpfWeight<F> {
    fn decode_with_param(
        decoding_parameter: &<Self as IdpfValue>::ValueParameter,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let mut v = Vec::with_capacity(*decoding_parameter);
        for _ in 0..*decoding_parameter {
            v.push(F::decode_with_param(&(), bytes)?);
        }

        Ok(Self(v))
    }
}

#[cfg(test)]
mod tests {

    use crate::field::Field128;

    use super::VidpfWeight;

    type TestWeight = VidpfWeight<Field128>;
    const TEST_WEIGHT_LEN: usize = 3;
    const TEST_NONCE_SIZE: usize = 16;
    const TEST_NONCE: &[u8; TEST_NONCE_SIZE] = b"Test Nonce VIDPF";

    mod vidpf {
        use crate::{
            bt::BinaryTree,
            codec::{Encode, ParameterizedDecode},
            idpf::IdpfValue,
            vidpf::{
                Vidpf, VidpfEvalCache, VidpfEvalState, VidpfInput, VidpfKey, VidpfPublicShare,
                VidpfServerId,
            },
        };

        use super::{TestWeight, TEST_NONCE, TEST_NONCE_SIZE, TEST_WEIGHT_LEN};

        #[test]
        fn roundtrip_codec() {
            let input = VidpfInput::from_bytes(&[0xFF]);
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);
            let (_, public, _, _) = vidpf_gen_setup(&input, &weight);

            let bytes = public.get_encoded().unwrap();
            assert_eq!(public.encoded_len().unwrap(), bytes.len());

            let decoded = VidpfPublicShare::<TestWeight>::get_decoded_with_param(
                &(8, TEST_WEIGHT_LEN),
                &bytes,
            )
            .unwrap();
            assert_eq!(public, decoded);
        }

        fn vidpf_gen_setup(
            input: &VidpfInput,
            weight: &TestWeight,
        ) -> (
            Vidpf<TestWeight>,
            VidpfPublicShare<TestWeight>,
            [VidpfKey; 2],
            [u8; TEST_NONCE_SIZE],
        ) {
            let vidpf = Vidpf::new(TEST_WEIGHT_LEN);
            let (public, keys) = vidpf.gen(input, weight, TEST_NONCE).unwrap();
            (vidpf, public, keys, *TEST_NONCE)
        }

        #[test]
        fn correctness_at_last_level() {
            let input = VidpfInput::from_bytes(&[0xFF]);
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);
            let (vidpf, public, [key_0, key_1], nonce) = vidpf_gen_setup(&input, &weight);

            let value_share_0 = vidpf
                .eval(VidpfServerId::S0, &key_0, &public, &input, &nonce)
                .unwrap();
            let value_share_1 = vidpf
                .eval(VidpfServerId::S1, &key_1, &public, &input, &nonce)
                .unwrap();

            assert_eq!(
                value_share_0.share + value_share_1.share,
                weight,
                "shares must add up to the expected weight",
            );

            assert_eq!(
                value_share_0.proof, value_share_1.proof,
                "proofs must be equal"
            );

            let bad_input = VidpfInput::from_bytes(&[0x00]);
            let zero = TestWeight::zero(&TEST_WEIGHT_LEN);
            let value_share_0 = vidpf
                .eval(VidpfServerId::S0, &key_0, &public, &bad_input, &nonce)
                .unwrap();
            let value_share_1 = vidpf
                .eval(VidpfServerId::S1, &key_1, &public, &bad_input, &nonce)
                .unwrap();

            assert_eq!(
                value_share_0.share + value_share_1.share,
                zero,
                "shares must add up to zero",
            );

            assert_eq!(
                value_share_0.proof, value_share_1.proof,
                "proofs must be equal"
            );
        }

        #[test]
        fn correctness_at_each_level() {
            let input = VidpfInput::from_bytes(&[0xFF]);
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);
            let (vidpf, public, keys, nonce) = vidpf_gen_setup(&input, &weight);

            assert_eval_at_each_level(&vidpf, &keys, &public, &input, &weight, &nonce);

            let bad_input = VidpfInput::from_bytes(&[0x00]);
            let zero = TestWeight::zero(&TEST_WEIGHT_LEN);

            assert_eval_at_each_level(&vidpf, &keys, &public, &bad_input, &zero, &nonce);
        }

        fn assert_eval_at_each_level(
            vidpf: &Vidpf<TestWeight>,
            [key_0, key_1]: &[VidpfKey; 2],
            public: &VidpfPublicShare<TestWeight>,
            input: &VidpfInput,
            weight: &TestWeight,
            nonce: &[u8],
        ) {
            let mut state_0 = VidpfEvalState::init_from_key(VidpfServerId::S0, key_0);
            let mut state_1 = VidpfEvalState::init_from_key(VidpfServerId::S1, key_1);

            let n = input.len();
            for level in 0..n {
                let share_0;
                let share_1;
                (state_0, share_0) = vidpf
                    .eval_next(VidpfServerId::S0, public, input, level, &state_0, nonce)
                    .unwrap();
                (state_1, share_1) = vidpf
                    .eval_next(VidpfServerId::S1, public, input, level, &state_1, nonce)
                    .unwrap();

                assert_eq!(
                    share_0 + share_1,
                    *weight,
                    "shares must add up to the expected weight at the current level: {:?}",
                    level
                );

                assert_eq!(
                    state_0.proof, state_1.proof,
                    "proofs must be equal at the current level: {:?}",
                    level
                );
            }
        }

        #[test]
        fn caching_at_each_level() {
            let input = VidpfInput::from_bytes(&[0xFF]);
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);
            let (vidpf, public, keys, nonce) = vidpf_gen_setup(&input, &weight);

            test_equivalence_of_eval_with_caching(&vidpf, &keys, &public, &input, &nonce);
        }

        /// Ensures that VIDPF outputs match regardless of whether the path to
        /// each node is recomputed or cached during evaluation.
        fn test_equivalence_of_eval_with_caching(
            vidpf: &Vidpf<TestWeight>,
            [key_0, key_1]: &[VidpfKey; 2],
            public: &VidpfPublicShare<TestWeight>,
            input: &VidpfInput,
            nonce: &[u8],
        ) {
            let mut cache_tree_0 = BinaryTree::<VidpfEvalCache<TestWeight>>::default();
            let mut cache_tree_1 = BinaryTree::<VidpfEvalCache<TestWeight>>::default();

            let n = input.len();
            for level in 0..n {
                let val_share_0 = vidpf
                    .eval(
                        VidpfServerId::S0,
                        key_0,
                        public,
                        &input.prefix(level),
                        nonce,
                    )
                    .unwrap();
                let val_share_1 = vidpf
                    .eval(
                        VidpfServerId::S1,
                        key_1,
                        public,
                        &input.prefix(level),
                        nonce,
                    )
                    .unwrap();
                let val_share_0_cached = vidpf
                    .eval_with_cache(
                        VidpfServerId::S0,
                        key_0,
                        public,
                        &input.prefix(level),
                        &mut cache_tree_0,
                        nonce,
                    )
                    .unwrap();
                let val_share_1_cached = vidpf
                    .eval_with_cache(
                        VidpfServerId::S1,
                        key_1,
                        public,
                        &input.prefix(level),
                        &mut cache_tree_1,
                        nonce,
                    )
                    .unwrap();

                assert_eq!(
                    val_share_0.share, val_share_0_cached.share,
                    "shares must be computed equally with or without caching: {:?}",
                    level
                );

                assert_eq!(
                    val_share_1.share, val_share_1_cached.share,
                    "shares must be computed equally with or without caching: {:?}",
                    level
                );

                assert_eq!(
                    val_share_0.proof, val_share_0_cached.proof,
                    "proofs must be equal with or without caching: {:?}",
                    level
                );

                assert_eq!(
                    val_share_1.proof, val_share_1_cached.proof,
                    "proofs must be equal with or without caching: {:?}",
                    level
                );
            }
        }
    }

    mod weight {
        use std::io::Cursor;
        use subtle::{Choice, ConditionallyNegatable};

        use crate::{
            codec::{Encode, ParameterizedDecode},
            idpf::IdpfValue,
            vdaf::xof::{Seed, Xof, XofTurboShake128},
        };

        use super::{TestWeight, TEST_WEIGHT_LEN};

        #[test]
        fn roundtrip_codec() {
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);

            let mut bytes = vec![];
            weight.encode(&mut bytes).unwrap();

            let expected_bytes = [
                [vec![21], vec![0u8; 15]].concat(),
                [vec![22], vec![0u8; 15]].concat(),
                [vec![23], vec![0u8; 15]].concat(),
            ]
            .concat();

            assert_eq!(weight.encoded_len().unwrap(), expected_bytes.len());
            // Check endianness of encoding
            assert_eq!(bytes, expected_bytes);

            let decoded =
                TestWeight::decode_with_param(&TEST_WEIGHT_LEN, &mut Cursor::new(&bytes)).unwrap();
            assert_eq!(weight, decoded);
        }

        #[test]
        fn add_sub() {
            let [a, b] = compatible_weights();
            let mut c = a.clone();
            c += a.clone();

            assert_eq!(
                (a.clone() + b.clone()) + (a.clone() - b.clone()),
                c,
                "a: {:?} b:{:?}",
                a,
                b
            );
        }

        #[test]
        fn conditional_negate() {
            let [a, _] = compatible_weights();
            let mut c = a.clone();
            c.conditional_negate(Choice::from(0));
            let mut d = a.clone();
            d.conditional_negate(Choice::from(1));
            let zero = TestWeight::zero(&TEST_WEIGHT_LEN);

            assert_eq!(c + d, zero, "a: {:?}", a);
        }

        #[test]
        #[should_panic = "invalid weight length"]
        fn add_panics() {
            let [w0, w1] = incompatible_weights();
            let _ = w0 + w1;
        }

        #[test]
        #[should_panic = "invalid weight length"]
        fn add_assign_panics() {
            let [mut w0, w1] = incompatible_weights();
            w0 += w1;
        }

        #[test]
        #[should_panic = "invalid weight length"]
        fn sub_panics() {
            let [w0, w1] = incompatible_weights();
            let _ = w0 - w1;
        }

        #[test]
        #[should_panic = "invalid weight length"]
        fn conditional_select_panics() {
            let [w0, w1] = incompatible_weights();
            TestWeight::conditional_select(&w0, &w1, Choice::from(0));
        }

        fn compatible_weights() -> [TestWeight; 2] {
            let mut xof = XofTurboShake128::seed_stream(&Seed(Default::default()), &[], &[]);
            [
                TestWeight::generate(&mut xof, &TEST_WEIGHT_LEN),
                TestWeight::generate(&mut xof, &TEST_WEIGHT_LEN),
            ]
        }

        fn incompatible_weights() -> [TestWeight; 2] {
            let mut xof = XofTurboShake128::seed_stream(&Seed(Default::default()), &[], &[]);
            [
                TestWeight::generate(&mut xof, &TEST_WEIGHT_LEN),
                TestWeight::generate(&mut xof, &(2 * TEST_WEIGHT_LEN)),
            ]
        }
    }
}
