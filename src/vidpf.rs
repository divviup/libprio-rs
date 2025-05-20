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

use bitvec::prelude::{BitVec, Lsb0};
use rand::prelude::*;
use std::fmt::Debug;
use std::io::{Cursor, Read};
use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable, ConstantTimeEq};

use crate::{
    bt::{BinaryTree, Node},
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::FieldElement,
    idpf::{conditional_swap_seed, conditional_xor_seeds, xor_seeds, IdpfInput, IdpfValue},
    vdaf::{
        mastic,
        xof::{Seed, Xof, XofFixedKeyAes128, XofTurboShake128},
    },
};

/// VIDPF errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VidpfError {
    /// Input is too long to be represented.
    #[error("bit length too long")]
    BitLengthTooLong,

    /// Error when an input has an unexpected bit length.
    #[error("invalid input length")]
    InvalidInputLength,

    /// Error when a weight has an unexpected length.
    #[error("invalid weight length")]
    InvalidWeightLength,
}

/// Represents the domain of an incremental point function.
pub type VidpfInput = IdpfInput;

/// Represents the codomain of an incremental point function.
pub trait VidpfValue: IdpfValue + Clone + Debug + PartialEq + ConstantTimeEq {}

#[derive(Clone, Debug)]
/// An instance of the VIDPF.
pub struct Vidpf<W: VidpfValue> {
    pub(crate) bits: u16,
    pub(crate) weight_len: W::ValueParameter,
}

impl<W: VidpfValue> Vidpf<W> {
    /// Creates a VIDPF instance.
    ///
    /// # Arguments
    ///
    /// * `bits`, the length of the input in bits.
    /// * `weight_len`, the length of the weight in number of field elements.
    pub fn new(bits: usize, weight_len: W::ValueParameter) -> Result<Self, VidpfError> {
        let bits = u16::try_from(bits).map_err(|_| VidpfError::BitLengthTooLong)?;
        Ok(Self { bits, weight_len })
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
        ctx: &[u8],
        input: &VidpfInput,
        weight: &W,
        nonce: &[u8],
    ) -> Result<(VidpfPublicShare<W>, [VidpfKey; 2]), VidpfError> {
        let mut rng = thread_rng();
        let keys = rng.gen();
        let public = self.gen_with_keys(ctx, &keys, input, weight, nonce)?;
        Ok((public, keys))
    }

    /// Produce the public share for the given keys, input, and weight.
    pub(crate) fn gen_with_keys(
        &self,
        ctx: &[u8],
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

        let mut cw = Vec::with_capacity(input.len());
        for idx in self.index_iter(input)? {
            let bit = idx.bit;

            // Extend.
            let e = [
                Self::extend(seed[0], ctx, nonce),
                Self::extend(seed[1], ctx, nonce),
            ];

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
            (seed[0], weight_0) = self.convert(seed_keep_0, ctx, nonce);
            (seed[1], weight_1) = self.convert(seed_keep_1, ctx, nonce);
            ctrl[0] = ctrl_keep_0;
            ctrl[1] = ctrl_keep_1;

            // Compute the correction word payload.
            let mut cw_weight = weight_1 - weight_0 + weight.clone();
            cw_weight.conditional_negate(ctrl[1]);

            // Compute the correction word node proof.
            let cw_proof = xor_proof(
                idx.node_proof(&seed[0], ctx),
                &idx.node_proof(&seed[1], ctx),
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

    /// Evaluate a given VIDPF (comprised of the key and public share) at a given prefix. Return
    /// the weight for that prefix along with a hash of the node proofs along the path from the
    /// root to the prefix.
    pub fn eval(
        &self,
        ctx: &[u8],
        id: VidpfServerId,
        key: &VidpfKey,
        public: &VidpfPublicShare<W>,
        input: &VidpfInput,
        nonce: &[u8],
    ) -> Result<(W, VidpfProof), VidpfError> {
        use sha3::{Digest, Sha3_256};

        let mut r = VidpfEvalResult {
            state: VidpfEvalState::init_from_key(id, key),
            share: W::zero(&self.weight_len), // not used
        };

        if input.len() > public.cw.len() {
            return Err(VidpfError::InvalidInputLength);
        }

        let mut hash = Sha3_256::new();
        for (idx, cw) in self.index_iter(input)?.zip(public.cw.iter()) {
            r = self.eval_next(ctx, cw, idx, &r.state, nonce);
            hash.update(r.state.node_proof);
        }

        let mut weight = r.share;
        weight.conditional_negate(Choice::from(id));
        Ok((weight, hash.finalize().into()))
    }

    /// Evaluates the `input` at the given level using the provided initial
    /// state, and returns a new state and a share of the input's weight at that level.
    fn eval_next(
        &self,
        ctx: &[u8],
        cw: &VidpfCorrectionWord<W>,
        idx: VidpfEvalIndex<'_>,
        state: &VidpfEvalState,
        nonce: &[u8],
    ) -> VidpfEvalResult<W> {
        let bit = idx.bit;

        // Extend.
        let e = Self::extend(state.seed, ctx, nonce);

        // Select the seed and control bit.
        let (seed_keep, seed_lose) = &mut (e.seed_right, e.seed_left);
        conditional_swap_seed(seed_keep, seed_lose, !bit);
        let ctrl_keep = Choice::conditional_select(&e.ctrl_left, &e.ctrl_right, bit);

        // Correct the seed and control bit.
        let seed_keep = conditional_xor_seeds(seed_keep, &cw.seed, state.control_bit);
        let cw_ctrl_keep = Choice::conditional_select(&cw.ctrl_left, &cw.ctrl_right, bit);
        let next_ctrl = ctrl_keep ^ (state.control_bit & cw_ctrl_keep);

        // Convert and correct the payload.
        let (next_seed, w) = self.convert(seed_keep, ctx, nonce);
        let mut weight = <W as IdpfValue>::conditional_select(
            &<W as IdpfValue>::zero(&self.weight_len),
            &cw.weight,
            next_ctrl,
        );
        weight += w;

        // Compute and correct the node proof.
        let node_proof =
            conditional_xor_proof(idx.node_proof(&next_seed, ctx), &cw.proof, next_ctrl);

        let next_state = VidpfEvalState {
            seed: next_seed,
            control_bit: next_ctrl,
            node_proof,
        };

        VidpfEvalResult {
            state: next_state,
            share: weight,
        }
    }

    pub(crate) fn get_beta_share(
        &self,
        ctx: &[u8],
        id: VidpfServerId,
        public: &VidpfPublicShare<W>,
        key: &VidpfKey,
        nonce: &[u8],
    ) -> Result<W, VidpfError> {
        let cw = public.cw.first().ok_or(VidpfError::InvalidInputLength)?;

        let state = VidpfEvalState::init_from_key(id, key);
        let input_left = VidpfInput::from_bools(&[false]);
        let idx_left = self.index(&input_left)?;

        let VidpfEvalResult {
            state: _,
            share: weight_share_left,
        } = self.eval_next(ctx, cw, idx_left, &state, nonce);

        let VidpfEvalResult {
            state: _,
            share: weight_share_right,
        } = self.eval_next(ctx, cw, idx_left.right_sibling(), &state, nonce);

        let mut beta_share = weight_share_left + weight_share_right;
        beta_share.conditional_negate(Choice::from(id));
        Ok(beta_share)
    }

    /// Ensure `prefix_tree` contains the prefix tree for `prefixes`, as well as the sibling of
    /// each node in the prefix tree. The return value is the weights for the prefixes
    /// concatenated together.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_prefix_tree_with_siblings(
        &self,
        ctx: &[u8],
        id: VidpfServerId,
        public: &VidpfPublicShare<W>,
        key: &VidpfKey,
        nonce: &[u8],
        prefixes: &[VidpfInput],
        prefix_tree: &mut BinaryTree<VidpfEvalResult<W>>,
    ) -> Result<Vec<W>, VidpfError> {
        let mut out_shares = Vec::with_capacity(prefixes.len());

        for prefix in prefixes {
            if prefix.len() > public.cw.len() {
                return Err(VidpfError::InvalidInputLength);
            }

            let mut sub_tree = prefix_tree.root.get_or_insert_with(|| {
                Box::new(Node::new(VidpfEvalResult {
                    state: VidpfEvalState::init_from_key(id, key),
                    share: W::zero(&self.weight_len), // not used
                }))
            });

            for (idx, cw) in self.index_iter(prefix)?.zip(public.cw.iter()) {
                let left = sub_tree.left.get_or_insert_with(|| {
                    Box::new(Node::new(self.eval_next(
                        ctx,
                        cw,
                        idx.left_sibling(),
                        &sub_tree.value.state,
                        nonce,
                    )))
                });
                let right = sub_tree.right.get_or_insert_with(|| {
                    Box::new(Node::new(self.eval_next(
                        ctx,
                        cw,
                        idx.right_sibling(),
                        &sub_tree.value.state,
                        nonce,
                    )))
                });

                sub_tree = if idx.bit.unwrap_u8() == 0 {
                    left
                } else {
                    right
                };
            }

            out_shares.push(sub_tree.value.share.clone());
        }

        for out_share in out_shares.iter_mut() {
            out_share.conditional_negate(Choice::from(id));
        }
        Ok(out_shares)
    }

    fn extend(seed: VidpfSeed, ctx: &[u8], nonce: &[u8]) -> ExtendedSeed {
        let mut rng = XofFixedKeyAes128::seed_stream(
            &seed,
            &[&mastic::dst_usage(mastic::USAGE_EXTEND), ctx],
            &[nonce],
        );

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

    fn convert(&self, seed: VidpfSeed, ctx: &[u8], nonce: &[u8]) -> (VidpfSeed, W) {
        let mut seed_stream = XofFixedKeyAes128::seed_stream(
            &seed,
            &[&mastic::dst_usage(mastic::USAGE_CONVERT), ctx],
            &[nonce],
        );

        let mut next_seed = VidpfSeed::default();
        seed_stream.fill_bytes(&mut next_seed);
        let weight = W::generate(&mut seed_stream, &self.weight_len);
        (next_seed, weight)
    }

    fn index_iter<'a>(
        &'a self,
        input: &'a VidpfInput,
    ) -> Result<impl Iterator<Item = VidpfEvalIndex<'a>>, VidpfError> {
        let n = u16::try_from(input.len()).map_err(|_| VidpfError::InvalidInputLength)?;
        if n > self.bits {
            return Err(VidpfError::InvalidInputLength);
        }
        Ok(Box::new((0..n).zip(input.iter()).map(
            move |(level, bit)| VidpfEvalIndex {
                bit: Choice::from(u8::from(bit)),
                input,
                level,
                bits: self.bits,
            },
        )))
    }

    fn index<'a>(&self, input: &'a VidpfInput) -> Result<VidpfEvalIndex<'a>, VidpfError> {
        let level = u16::try_from(input.len()).map_err(|_| VidpfError::InvalidInputLength)? - 1;
        if level >= self.bits {
            return Err(VidpfError::InvalidInputLength);
        }
        let bit = Choice::from(u8::from(input.get(usize::from(level)).unwrap()));
        Ok(VidpfEvalIndex {
            bit,
            input,
            level,
            bits: self.bits,
        })
    }
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
        // Control bits
        let mut control_bits: BitVec<u8, Lsb0> = BitVec::with_capacity(self.cw.len() * 2);
        for cw in self.cw.iter() {
            control_bits.push(bool::from(cw.ctrl_left));
            control_bits.push(bool::from(cw.ctrl_right));
        }
        control_bits.set_uninitialized(false);
        let mut packed_control = control_bits.into_vec();
        bytes.append(&mut packed_control);

        // Seeds
        for cw in self.cw.iter() {
            bytes.extend_from_slice(&cw.seed);
        }

        // Weights
        for cw in self.cw.iter() {
            cw.weight.encode(bytes)?;
        }

        // Node proofs
        for cw in self.cw.iter() {
            bytes.extend_from_slice(&cw.proof);
        }

        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        // We assume the weight has the same length at each level of the tree.
        let weight_len = self
            .cw
            .first()
            .map_or(Some(0), |cw| cw.weight.encoded_len())?;

        let mut len = 0;
        len += (2 * self.cw.len()).div_ceil(8); // packed control bits
        len += self.cw.len() * VIDPF_SEED_SIZE; // seeds
        len += self.cw.len() * weight_len; // weights
        len += self.cw.len() * VIDPF_PROOF_SIZE; // nod proofs
        Some(len)
    }
}

impl<W: VidpfValue> ParameterizedDecode<Vidpf<W>> for VidpfPublicShare<W> {
    fn decode_with_param(vidpf: &Vidpf<W>, bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let bits = usize::from(vidpf.bits);
        let packed_control_len = bits.div_ceil(4);
        let mut packed_control_bits = vec![0u8; packed_control_len];
        bytes.read_exact(&mut packed_control_bits)?;
        let unpacked_control_bits: BitVec<u8, Lsb0> = BitVec::from_vec(packed_control_bits);

        // Control bits
        let mut control_bits = Vec::with_capacity(bits);
        for chunk in unpacked_control_bits[0..bits * 2].chunks(2) {
            control_bits.push([(chunk[0] as u8).into(), (chunk[1] as u8).into()]);
        }

        // Check that unused packed bits are zero.
        if unpacked_control_bits[bits * 2..].any() {
            return Err(CodecError::UnexpectedValue);
        }

        // Seeds
        let seeds = std::iter::repeat_with(|| Seed::decode(bytes).map(|seed| seed.0))
            .take(bits)
            .collect::<Result<Vec<_>, _>>()?;

        // Weights
        let weights = std::iter::repeat_with(|| W::decode_with_param(&vidpf.weight_len, bytes))
            .take(bits)
            .collect::<Result<Vec<_>, _>>()?;

        let proofs = std::iter::repeat_with(|| {
            let mut proof = [0; VIDPF_PROOF_SIZE];
            bytes.read_exact(&mut proof)?;
            Ok::<_, CodecError>(proof)
        })
        .take(bits)
        .collect::<Result<Vec<_>, _>>()?;

        let cw = seeds
            .into_iter()
            .zip(
                control_bits
                    .into_iter()
                    .zip(weights.into_iter().zip(proofs)),
            )
            .map(
                |(seed, ([ctrl_left, ctrl_right], (weight, proof)))| VidpfCorrectionWord {
                    seed,
                    ctrl_left,
                    ctrl_right,
                    weight,
                    proof,
                },
            )
            .collect::<Vec<_>>();

        Ok(Self { cw })
    }
}

/// VIDPF evaluation state.
#[derive(Debug)]
pub(crate) struct VidpfEvalState {
    seed: VidpfSeed,
    control_bit: Choice,
    pub(crate) node_proof: VidpfProof,
}

impl VidpfEvalState {
    fn init_from_key(id: VidpfServerId, key: &VidpfKey) -> Self {
        Self {
            seed: key.0,
            control_bit: Choice::from(id),
            node_proof: VidpfProof::default(), // not used
        }
    }
}

/// Result of VIDPF evaluation.
#[derive(Debug)]
pub(crate) struct VidpfEvalResult<W: VidpfValue> {
    pub(crate) state: VidpfEvalState,
    pub(crate) share: W,
}

pub(crate) const VIDPF_PROOF_SIZE: usize = 32;
const VIDPF_SEED_SIZE: usize = 16;

/// Allows to validate user input and shares after evaluation.
pub(crate) type VidpfProof = [u8; VIDPF_PROOF_SIZE];

pub(crate) fn xor_proof(mut lhs: VidpfProof, rhs: &VidpfProof) -> VidpfProof {
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

#[derive(Copy, Clone)]
struct VidpfEvalIndex<'a> {
    bit: Choice,
    input: &'a VidpfInput,
    level: u16,
    bits: u16,
}

impl VidpfEvalIndex<'_> {
    fn left_sibling(&self) -> Self {
        Self {
            bit: Choice::from(0),
            input: self.input,
            level: self.level,
            bits: self.bits,
        }
    }

    fn right_sibling(&self) -> Self {
        Self {
            bit: Choice::from(1),
            input: self.input,
            level: self.level,
            bits: self.bits,
        }
    }

    fn node_proof(&self, seed: &VidpfSeed, ctx: &[u8]) -> VidpfProof {
        let mut xof = XofTurboShake128::from_seed_slice(
            &seed[..],
            &[&mastic::dst_usage(mastic::USAGE_NODE_PROOF), ctx],
        );
        xof.update(&self.bits.to_le_bytes());
        xof.update(&self.level.to_le_bytes());

        for byte in self
            .input
            .index(..=usize::from(self.level))
            .chunks(8)
            .enumerate()
            .map(|(byte_index, chunk)| {
                let mut byte = 0;
                for (bit_index, bit) in chunk.iter().enumerate() {
                    byte |= u8::from(*bit) << (7 - bit_index);
                }

                // Typically `input[level] == bit` , but `bit` may be overwritten by either
                // `left_sibling()` or `right_sibling()`. Adjust its value accordingly.
                if byte_index == usize::from(self.level) / 8 {
                    let bit_index = self.level % 8;
                    let m = 1 << (7 - bit_index);
                    byte = u8::conditional_select(&(byte & !m), &(byte | m), self.bit);
                }
                byte
            })
        {
            xof.update(&[byte]);
        }
        xof.into_seed().0
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
            codec::{Encode, ParameterizedDecode},
            idpf::IdpfValue,
            vidpf::{
                Vidpf, VidpfCorrectionWord, VidpfEvalState, VidpfInput, VidpfKey, VidpfPublicShare,
                VidpfServerId,
            },
        };

        use super::{TestWeight, TEST_NONCE, TEST_NONCE_SIZE, TEST_WEIGHT_LEN};

        #[test]
        fn roundtrip_codec() {
            let ctx = b"appliction context";
            let input = VidpfInput::from_bytes(&[0xFF]);
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);
            let (vidpf, public, _, _) = vidpf_gen_setup(ctx, &input, &weight);

            let bytes = public.get_encoded().unwrap();
            assert_eq!(public.encoded_len().unwrap(), bytes.len());

            let decoded = VidpfPublicShare::get_decoded_with_param(&vidpf, &bytes).unwrap();
            assert_eq!(public, decoded);
        }

        fn vidpf_gen_setup(
            ctx: &[u8],
            input: &VidpfInput,
            weight: &TestWeight,
        ) -> (
            Vidpf<TestWeight>,
            VidpfPublicShare<TestWeight>,
            [VidpfKey; 2],
            [u8; TEST_NONCE_SIZE],
        ) {
            let vidpf = Vidpf::new(input.len(), TEST_WEIGHT_LEN).unwrap();
            let (public, keys) = vidpf.gen(ctx, input, weight, TEST_NONCE).unwrap();
            (vidpf, public, keys, *TEST_NONCE)
        }

        #[test]
        fn correctness_at_last_level() {
            let ctx = b"some application";
            let input = VidpfInput::from_bytes(&[0xFF]);
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);
            let (vidpf, public, [key_0, key_1], nonce) = vidpf_gen_setup(ctx, &input, &weight);

            let (value_share_0, onehot_proof_0) = vidpf
                .eval(ctx, VidpfServerId::S0, &key_0, &public, &input, &nonce)
                .unwrap();
            let (value_share_1, onehot_proof_1) = vidpf
                .eval(ctx, VidpfServerId::S1, &key_1, &public, &input, &nonce)
                .unwrap();

            assert_eq!(
                value_share_0 + value_share_1,
                weight,
                "shares must add up to the expected weight",
            );

            assert_eq!(onehot_proof_0, onehot_proof_1, "proofs must be equal");

            let bad_input = VidpfInput::from_bytes(&[0x00]);
            let zero = TestWeight::zero(&TEST_WEIGHT_LEN);
            let (value_share_0, onehot_proof_0) = vidpf
                .eval(ctx, VidpfServerId::S0, &key_0, &public, &bad_input, &nonce)
                .unwrap();
            let (value_share_1, onehot_proof_1) = vidpf
                .eval(ctx, VidpfServerId::S1, &key_1, &public, &bad_input, &nonce)
                .unwrap();

            assert_eq!(
                value_share_0 + value_share_1,
                zero,
                "shares must add up to zero",
            );

            assert_eq!(onehot_proof_0, onehot_proof_1, "proofs must be equal");
        }

        #[test]
        fn correctness_at_each_level() {
            let ctx = b"application context";
            let input = VidpfInput::from_bytes(&[0xFF]);
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);
            let (vidpf, public, keys, nonce) = vidpf_gen_setup(ctx, &input, &weight);

            assert_eval_at_each_level(&vidpf, ctx, &keys, &public, &input, &weight, &nonce);

            let bad_input = VidpfInput::from_bytes(&[0x00]);
            let zero = TestWeight::zero(&TEST_WEIGHT_LEN);

            assert_eval_at_each_level(&vidpf, ctx, &keys, &public, &bad_input, &zero, &nonce);
        }

        fn assert_eval_at_each_level(
            vidpf: &Vidpf<TestWeight>,
            ctx: &[u8],
            [key_0, key_1]: &[VidpfKey; 2],
            public: &VidpfPublicShare<TestWeight>,
            input: &VidpfInput,
            weight: &TestWeight,
            nonce: &[u8],
        ) {
            let mut state_0 = VidpfEvalState::init_from_key(VidpfServerId::S0, key_0);
            let mut state_1 = VidpfEvalState::init_from_key(VidpfServerId::S1, key_1);

            for (idx, cw) in vidpf.index_iter(input).unwrap().zip(public.cw.iter()) {
                let r0 = vidpf.eval_next(ctx, cw, idx, &state_0, nonce);
                let r1 = vidpf.eval_next(ctx, cw, idx, &state_1, nonce);

                assert_eq!(
                    r0.share - r1.share,
                    *weight,
                    "shares must add up to the expected weight at the current level: {:?}",
                    idx.level
                );

                assert_eq!(
                    r0.state.node_proof, r1.state.node_proof,
                    "proofs must be equal at the current level: {:?}",
                    idx.level
                );

                state_0 = r0.state;
                state_1 = r1.state;
            }
        }

        // Assert that the length of the weight is the same at each level of the tree. This
        // assumption is made in `PublicShare::encoded_len()`.
        #[test]
        fn public_share_weight_len() {
            let input = VidpfInput::from_bools(&vec![false; 237]);
            let weight = TestWeight::from(vec![21.into(), 22.into(), 23.into()]);
            let (vidpf, public, _, _) = vidpf_gen_setup(b"some application", &input, &weight);
            for VidpfCorrectionWord { weight, .. } in public.cw {
                assert_eq!(weight.0.len(), vidpf.weight_len);
            }
        }
    }

    mod weight {
        use std::io::Cursor;
        use subtle::{Choice, ConditionallyNegatable};

        use crate::{
            codec::{Encode, ParameterizedDecode},
            idpf::IdpfValue,
            vdaf::xof::{Xof, XofTurboShake128},
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
            let mut xof = XofTurboShake128::seed_stream(&[0; 32], &[], &[]);
            [
                TestWeight::generate(&mut xof, &TEST_WEIGHT_LEN),
                TestWeight::generate(&mut xof, &TEST_WEIGHT_LEN),
            ]
        }

        fn incompatible_weights() -> [TestWeight; 2] {
            let mut xof = XofTurboShake128::seed_stream(&[0; 32], &[], &[]);
            [
                TestWeight::generate(&mut xof, &TEST_WEIGHT_LEN),
                TestWeight::generate(&mut xof, &(2 * TEST_WEIGHT_LEN)),
            ]
        }
    }
}
