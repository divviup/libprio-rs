// SPDX-License-Identifier: MPL-2.0

//! Implementation of Mastic as specified in [[draft-mouris-cfrg-mastic-01]].
//!
//! [draft-mouris-cfrg-mastic-01]: https://www.ietf.org/archive/id/draft-mouris-cfrg-mastic-01.html

use crate::{
    bt::BinaryTree,
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{decode_fieldvec, FieldElement, FieldElementWithInteger},
    flp::{
        szk::{Szk, SzkJointShare, SzkProofShare, SzkQueryShare, SzkQueryState},
        Type,
    },
    vdaf::{
        poplar1::{Poplar1, Poplar1AggregationParam},
        xof::{Seed, Xof},
        Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare,
        PrepareTransition, Vdaf, VdafError,
    },
    vidpf::{
        xor_proof, Vidpf, VidpfError, VidpfInput, VidpfKey, VidpfProof, VidpfPublicShare,
        VidpfServerId, VidpfWeight, VIDPF_PROOF_SIZE,
    },
};

use rand::prelude::*;
use std::io::{Cursor, Read};
use std::ops::BitAnd;
use std::slice::from_ref;
use std::{collections::VecDeque, fmt::Debug};
use subtle::{Choice, ConstantTimeEq};

use super::xof::XofTurboShake128;

const NONCE_SIZE: usize = 16;

// draft-jimouris-cfrg-mastic:
//
// ONEHOT_PROOF_INIT = XofTurboShake128(zeros(XofTurboShake128.SEED_SIZE),
//                                      dst(b'', USAGE_ONEHOT_PROOF_INIT),
//                                      b'').next(PROOF_SIZE)
pub(crate) const ONEHOT_PROOF_INIT: [u8; 32] = [
    253, 211, 45, 179, 139, 135, 183, 67, 202, 144, 13, 205, 241, 39, 165, 73, 232, 54, 57, 193,
    106, 154, 133, 22, 15, 194, 223, 162, 79, 108, 60, 133,
];

pub(crate) const USAGE_PROVE_RAND: u8 = 0;
pub(crate) const USAGE_PROOF_SHARE: u8 = 1;
pub(crate) const USAGE_QUERY_RAND: u8 = 2;
pub(crate) const USAGE_JOINT_RAND_SEED: u8 = 3;
pub(crate) const USAGE_JOINT_RAND_PART: u8 = 4;
pub(crate) const USAGE_JOINT_RAND: u8 = 5;
pub(crate) const USAGE_ONEHOT_PROOF_HASH: u8 = 7;
pub(crate) const USAGE_NODE_PROOF: u8 = 8;
pub(crate) const USAGE_EVAL_PROOF: u8 = 9;
pub(crate) const USAGE_EXTEND: u8 = 10;
pub(crate) const USAGE_CONVERT: u8 = 11;
pub(crate) const USAGE_PAYLOAD_CHECK: u8 = 12;

pub(crate) fn dst_usage(usage: u8) -> [u8; 8] {
    const VERSION: u8 = 0;
    [b'm', b'a', b's', b't', b'i', b'c', VERSION, usage]
}

/// The main struct implementing the Mastic VDAF.
/// Composed of a shared zero knowledge proof system and a verifiable incremental
/// distributed point function.
#[derive(Clone, Debug)]
pub struct Mastic<T: Type> {
    id: [u8; 4],
    pub(crate) szk: Szk<T>,
    pub(crate) vidpf: Vidpf<VidpfWeight<T::Field>>,
    /// The length of the private attribute associated with any input.
    pub(crate) bits: usize,
}

impl<T: Type> Mastic<T> {
    /// Creates a new instance of Mastic, with a specific attribute length and weight type.
    pub fn new(algorithm_id: u32, typ: T, bits: usize) -> Result<Self, VdafError> {
        let vidpf = Vidpf::new(bits, typ.input_len() + 1)?;
        let szk = Szk::new(typ, algorithm_id);
        Ok(Self {
            id: algorithm_id.to_le_bytes(),
            szk,
            vidpf,
            bits,
        })
    }

    fn agg_share_len(&self, agg_param: &MasticAggregationParam) -> usize {
        // The aggregate share consists of the counter and truncated weight for each candidate prefix.
        (1 + self.szk.typ.output_len()) * agg_param.level_and_prefixes.prefixes().len()
    }
}

/// Mastic aggregation parameter.
///
/// This includes the VIDPF tree level under evaluation and a set of prefixes to evaluate at that level.
#[derive(Clone, Debug, PartialEq)]
pub struct MasticAggregationParam {
    /// aggregation parameter inherited from [`Poplar1`]: contains the level (attribute length) and a vector of attribute prefixes (IdpfInputs)
    level_and_prefixes: Poplar1AggregationParam,
    /// Flag indicating whether the VIDPF weight needs to be validated using SZK.
    /// This flag must be set the first time any report is aggregated; however this may happen at any level of the tree.
    require_weight_check: bool,
}

#[cfg(test)]
impl MasticAggregationParam {
    fn new(prefixes: Vec<VidpfInput>, require_weight_check: bool) -> Result<Self, VdafError> {
        Ok(Self {
            level_and_prefixes: Poplar1AggregationParam::try_from_prefixes(prefixes)?,
            require_weight_check,
        })
    }
}

impl Encode for MasticAggregationParam {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.level_and_prefixes.encode(bytes)?;
        let require_weight_check = if self.require_weight_check { 1u8 } else { 0u8 };
        require_weight_check.encode(bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.level_and_prefixes.encoded_len()? + 1usize)
    }
}

impl Decode for MasticAggregationParam {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let level_and_prefixes = Poplar1AggregationParam::decode(bytes)?;
        let require_weight_check_u8 = u8::decode(bytes)?;
        let require_weight_check = require_weight_check_u8 != 0;
        Ok(Self {
            level_and_prefixes,
            require_weight_check,
        })
    }
}

/// Mastic public share.
///
/// Contains broadcast information shared between parties to support VIDPF correctness.
pub type MasticPublicShare<V> = VidpfPublicShare<V>;

impl<T: Type> ParameterizedDecode<Mastic<T>> for MasticPublicShare<VidpfWeight<T::Field>> {
    fn decode_with_param(
        mastic: &Mastic<T>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        VidpfPublicShare::decode_with_param(&(mastic.bits, mastic.vidpf.weight_len), bytes)
    }
}

/// Mastic input share.
///
/// Message sent by the [`Client`] to each Aggregator during the Sharding phase.
#[derive(Clone, Debug)]
pub struct MasticInputShare<F: FieldElement> {
    /// VIDPF key share.
    vidpf_key: VidpfKey,

    /// The proof share.
    proof_share: SzkProofShare<F>,
}

impl<F: FieldElement> Encode for MasticInputShare<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&self.vidpf_key.0[..]);
        self.proof_share.encode(bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(16 + self.proof_share.encoded_len()?)
    }
}

impl<'a, T: Type> ParameterizedDecode<(&'a Mastic<T>, usize)> for MasticInputShare<T::Field> {
    fn decode_with_param(
        (mastic, agg_id): &(&'a Mastic<T>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *agg_id > 1 {
            return Err(CodecError::UnexpectedValue);
        }
        let mut value = [0; 16];
        bytes.read_exact(&mut value)?;
        let vidpf_key = VidpfKey::from_bytes(value);
        let proof_share = SzkProofShare::decode_with_param(
            &(
                *agg_id == 0,
                mastic.szk.typ.proof_len(),
                mastic.szk.typ.joint_rand_len() != 0,
            ),
            bytes,
        )?;
        Ok(Self {
            vidpf_key,
            proof_share,
        })
    }
}

impl<F: FieldElement> PartialEq for MasticInputShare<F> {
    fn eq(&self, other: &MasticInputShare<F>) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: FieldElement> ConstantTimeEq for MasticInputShare<F> {
    fn ct_eq(&self, other: &MasticInputShare<F>) -> Choice {
        self.vidpf_key
            .ct_eq(&other.vidpf_key)
            .bitand(self.proof_share.ct_eq(&other.proof_share))
    }
}

/// Mastic output share.
///
/// Contains a flattened vector of VIDPF outputs: one for each prefix.
pub type MasticOutputShare<V> = OutputShare<V>;

/// Mastic aggregate share.
///
/// Contains a flattened vector of VIDPF outputs to be aggregated by Mastic aggregators
pub type MasticAggregateShare<V> = AggregateShare<V>;

impl<'a, T: Type> ParameterizedDecode<(&'a Mastic<T>, &'a MasticAggregationParam)>
    for MasticAggregateShare<T::Field>
{
    fn decode_with_param(
        (mastic, agg_param): &(&Mastic<T>, &MasticAggregationParam),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        decode_fieldvec(mastic.agg_share_len(agg_param), bytes).map(AggregateShare)
    }
}

impl<'a, T: Type> ParameterizedDecode<(&'a Mastic<T>, &'a MasticAggregationParam)>
    for MasticOutputShare<T::Field>
{
    fn decode_with_param(
        (mastic, agg_param): &(&Mastic<T>, &MasticAggregationParam),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        decode_fieldvec(mastic.agg_share_len(agg_param), bytes).map(OutputShare)
    }
}

impl<T: Type> Vdaf for Mastic<T> {
    type Measurement = (VidpfInput, T::Measurement);
    type AggregateResult = Vec<T::AggregateResult>;
    type AggregationParam = MasticAggregationParam;
    type PublicShare = MasticPublicShare<VidpfWeight<T::Field>>;
    type InputShare = MasticInputShare<T::Field>;
    type OutputShare = MasticOutputShare<T::Field>;
    type AggregateShare = MasticAggregateShare<T::Field>;

    fn algorithm_id(&self) -> u32 {
        u32::from_be_bytes(self.id)
    }

    fn num_aggregators(&self) -> usize {
        2
    }
}
impl<T: Type> Mastic<T> {
    fn shard_with_random(
        &self,
        ctx: &[u8],
        (alpha, weight): &(VidpfInput, T::Measurement),
        nonce: &[u8; 16],
        vidpf_keys: [VidpfKey; 2],
        szk_random: [Seed<32>; 2],
        joint_random_opt: Option<Seed<32>>,
    ) -> Result<(<Self as Vdaf>::PublicShare, Vec<<Self as Vdaf>::InputShare>), VdafError> {
        if alpha.len() != self.bits {
            return Err(VdafError::Vidpf(VidpfError::InvalidInputLength));
        }

        // The output with which we program the VIDPF is a counter and the encoded measurement.
        let mut beta = VidpfWeight(self.szk.typ.encode_measurement(weight)?);
        beta.0.insert(0, T::Field::one());

        // Compute the measurement shares for each aggregator by generating VIDPF
        // keys for the measurement and evaluating each of them.
        let public_share = self
            .vidpf
            .gen_with_keys(ctx, &vidpf_keys, alpha, &beta, nonce)?;

        let leader_beta_share = self.vidpf.get_beta_share(
            ctx,
            VidpfServerId::S0,
            &public_share,
            &vidpf_keys[0],
            nonce,
        )?;
        let helper_beta_share = self.vidpf.get_beta_share(
            ctx,
            VidpfServerId::S1,
            &public_share,
            &vidpf_keys[1],
            nonce,
        )?;

        let [leader_szk_proof_share, helper_szk_proof_share] = self.szk.prove(
            ctx,
            &leader_beta_share.as_ref()[1..],
            &helper_beta_share.as_ref()[1..],
            &beta.as_ref()[1..],
            szk_random,
            joint_random_opt,
            nonce,
        )?;
        let [leader_vidpf_key, helper_vidpf_key] = vidpf_keys;
        let leader_share = MasticInputShare {
            vidpf_key: leader_vidpf_key,
            proof_share: leader_szk_proof_share,
        };
        let helper_share = MasticInputShare {
            vidpf_key: helper_vidpf_key,
            proof_share: helper_szk_proof_share,
        };
        Ok((public_share, vec![leader_share, helper_share]))
    }

    fn hash_proof(&self, mut proof: VidpfProof, ctx: &[u8]) -> VidpfProof {
        let mut xof = XofTurboShake128::from_seed_slice(
            &[],
            &[&dst_usage(USAGE_ONEHOT_PROOF_HASH), &self.id, ctx],
        );
        xof.update(&proof);
        xof.into_seed_stream().fill_bytes(&mut proof);
        proof
    }
}

impl<T: Type> Client<16> for Mastic<T> {
    fn shard(
        &self,
        ctx: &[u8],
        measurement: &(VidpfInput, T::Measurement),
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError> {
        let vidpf_keys = [VidpfKey::generate()?, VidpfKey::generate()?];
        let joint_random_opt = if self.szk.requires_joint_rand() {
            Some(Seed::generate()?)
        } else {
            None
        };
        let szk_random = [Seed::generate()?, Seed::generate()?];

        self.shard_with_random(
            ctx,
            measurement,
            nonce,
            vidpf_keys,
            szk_random,
            joint_random_opt,
        )
    }
}

/// Mastic preparation state.
///
/// State held by an aggregator waiting for a message during Mastic preparation. Includes intermediate
/// state for [`Szk`] verification, the output shares currently being validated, and
/// parameters of Mastic used for encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MasticPrepareState<F: FieldElement> {
    /// The counter and truncated weight for each candidate prefix.
    output_shares: MasticOutputShare<F>,
    /// If [`Szk`]` verification is being performed, we also store the relevant state for that operation.
    szk_query_state: SzkQueryState,
    verifier_len: Option<usize>,
}

impl<F: FieldElement> Encode for MasticPrepareState<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.output_shares.encode(bytes)?;
        if let Some(joint_rand_seed) = &self.szk_query_state {
            joint_rand_seed.encode(bytes)?;
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.output_shares.as_ref().len() * F::ENCODED_SIZE
                + self.szk_query_state.as_ref().map_or(0, |_| 32),
        )
    }
}

impl<'a, T: Type> ParameterizedDecode<(&'a Mastic<T>, &'a MasticAggregationParam)>
    for MasticPrepareState<T::Field>
{
    fn decode_with_param(
        decoder @ (mastic, agg_param): &(&Mastic<T>, &MasticAggregationParam),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let output_shares = MasticOutputShare::decode_with_param(decoder, bytes)?;
        let szk_query_state = (mastic.szk.typ.joint_rand_len() > 0
            && agg_param.require_weight_check)
            .then(|| Seed::decode(bytes))
            .transpose()?;
        let verifier_len = agg_param
            .require_weight_check
            .then_some(mastic.szk.typ.verifier_len());

        Ok(Self {
            output_shares,
            szk_query_state,
            verifier_len,
        })
    }
}

/// Mastic preparation share.
///
/// Broadcast message from an aggregator preparing Mastic output shares. Includes the
/// [`Vidpf`] evaluation proof covering every prefix in the aggregation parameter, and optionally
/// the verification message for Szk.
#[derive(Clone, Debug, PartialEq)]
pub struct MasticPrepareShare<F: FieldElement> {
    ///  [`Vidpf`] evaluation proof, which guarantees one-hotness and payload consistency.
    eval_proof: [u8; VIDPF_PROOF_SIZE],

    /// If [`Szk`]` verification of the root weight is needed, a verification message.
    szk_query_share_opt: Option<SzkQueryShare<F>>,
}

impl<F: FieldElement> Encode for MasticPrepareShare<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&self.eval_proof);
        match &self.szk_query_share_opt {
            Some(query_share) => query_share.encode(bytes),
            None => Ok(()),
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            VIDPF_PROOF_SIZE
                + match &self.szk_query_share_opt {
                    Some(query_share) => query_share.encoded_len()?,
                    None => 0,
                },
        )
    }
}

impl<F: FieldElement> ParameterizedDecode<MasticPrepareState<F>> for MasticPrepareShare<F> {
    fn decode_with_param(
        prep_state: &MasticPrepareState<F>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let mut eval_proof = [0; VIDPF_PROOF_SIZE];
        bytes.read_exact(&mut eval_proof[..])?;
        let requires_joint_rand = prep_state.szk_query_state.is_some();
        let szk_query_share_opt = prep_state
            .verifier_len
            .map(|verifier_len| {
                SzkQueryShare::decode_with_param(&(requires_joint_rand, verifier_len), bytes)
            })
            .transpose()?;
        Ok(Self {
            eval_proof,
            szk_query_share_opt,
        })
    }
}

/// Mastic preparation message.
///
/// Result of preprocessing the broadcast messages of both aggregators during the
/// preparation phase.
pub type MasticPrepareMessage = SzkJointShare;

impl<F: FieldElement> ParameterizedDecode<MasticPrepareState<F>> for MasticPrepareMessage {
    fn decode_with_param(
        prep_state: &MasticPrepareState<F>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        match prep_state.szk_query_state {
            Some(_) => SzkJointShare::decode_with_param(&true, bytes),
            None => SzkJointShare::decode_with_param(&false, bytes),
        }
    }
}

impl<T: Type> Aggregator<32, NONCE_SIZE> for Mastic<T> {
    type PrepareState = MasticPrepareState<T::Field>;
    type PrepareShare = MasticPrepareShare<T::Field>;
    type PrepareMessage = MasticPrepareMessage;

    fn is_agg_param_valid(cur: &MasticAggregationParam, prev: &[MasticAggregationParam]) -> bool {
        // First agg param should be the only one that requires weight check.
        if cur.require_weight_check != prev.is_empty() {
            return false;
        };

        if prev.is_empty() {
            return true;
        }
        // Unpack this agg param and the last one in the list
        let cur_poplar_agg_param = &cur.level_and_prefixes;
        let prev_poplar_agg_param = from_ref(&prev.last().as_ref().unwrap().level_and_prefixes);
        Poplar1::<XofTurboShake128, 32>::is_agg_param_valid(
            cur_poplar_agg_param,
            prev_poplar_agg_param,
        )
    }

    fn prepare_init(
        &self,
        verify_key: &[u8; 32],
        ctx: &[u8],
        agg_id: usize,
        agg_param: &MasticAggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &MasticPublicShare<VidpfWeight<T::Field>>,
        input_share: &MasticInputShare<T::Field>,
    ) -> Result<(MasticPrepareState<T::Field>, MasticPrepareShare<T::Field>), VdafError> {
        let id = match agg_id {
            0 => Ok(VidpfServerId::S0),
            1 => Ok(VidpfServerId::S1),
            _ => Err(VdafError::Uncategorized(
                "Invalid aggregator ID".to_string(),
            )),
        }?;
        let prefixes = agg_param.level_and_prefixes.prefixes();

        let mut prefix_tree = BinaryTree::default();
        let out_shares = self.vidpf.eval_prefix_tree_with_siblings(
            ctx,
            id,
            public_share,
            &input_share.vidpf_key,
            nonce,
            prefixes,
            &mut prefix_tree,
        )?;

        let root = prefix_tree.root.as_ref().unwrap();

        // Onehot and payload checks
        let (payload_check, onehot_proof) = {
            let mut payload_check_xof =
                XofTurboShake128::init(&[0; 32], &[&dst_usage(USAGE_PAYLOAD_CHECK), &self.id, ctx]);
            let mut payload_check_buf = Vec::with_capacity(T::Field::ENCODED_SIZE);
            let mut onehot_proof = ONEHOT_PROOF_INIT;

            // Traverse the prefix tree breadth-first.
            let mut q = VecDeque::with_capacity(100);
            q.push_back(root.left.as_ref().unwrap());
            q.push_back(root.right.as_ref().unwrap());
            while let Some(node) = q.pop_front() {
                // Update onehot proof.
                onehot_proof = xor_proof(
                    onehot_proof,
                    &self.hash_proof(xor_proof(onehot_proof, &node.value.state.node_proof), ctx),
                );

                // Update payload check.
                if let (Some(left), Some(right)) = (node.left.as_ref(), node.right.as_ref()) {
                    for (w, (w_left, w_right)) in node
                        .value
                        .share
                        .0
                        .iter()
                        .zip(left.value.share.0.iter().zip(right.value.share.0.iter()))
                    {
                        (*w - (*w_left + *w_right))
                            .encode(&mut payload_check_buf)
                            .unwrap();
                        payload_check_xof.update(&payload_check_buf);
                        payload_check_buf.clear();
                    }

                    q.push_back(left);
                    q.push_back(right);
                }
            }

            let payload_check = payload_check_xof.into_seed().0;

            (payload_check, onehot_proof)
        };

        // Counter check.
        let counter_check = {
            let c_left = &root.left.as_ref().unwrap().value.share.0[0];
            let c_right = &root.right.as_ref().unwrap().value.share.0[0];
            let mut c = *c_left + *c_right;
            if id == VidpfServerId::S1 {
                c += T::Field::one();
            }
            c.get_encoded().unwrap()
        };

        let eval_proof = {
            let mut eval_proof_xof =
                XofTurboShake128::init(&[0; 32], &[&dst_usage(USAGE_EVAL_PROOF), &self.id, ctx]);
            eval_proof_xof.update(&onehot_proof);
            eval_proof_xof.update(&counter_check);
            eval_proof_xof.update(&payload_check);
            eval_proof_xof.into_seed().0
        };

        let mut truncated_out_shares =
            Vec::with_capacity(self.szk.typ.output_len() * prefixes.len());
        for VidpfWeight(mut out_share) in out_shares.into_iter() {
            let mut truncated_out_share = self.szk.typ.truncate(out_share.drain(1..).collect())?;
            truncated_out_shares.append(&mut out_share);
            truncated_out_shares.append(&mut truncated_out_share);
        }

        Ok(if agg_param.require_weight_check {
            // Range check.
            let VidpfWeight(beta_share) =
                self.vidpf
                    .get_beta_share(ctx, id, public_share, &input_share.vidpf_key, nonce)?;
            let (szk_query_share, szk_query_state) = self.szk.query(
                ctx,
                agg_param
                    .level_and_prefixes
                    .level()
                    .try_into()
                    .map_err(|_| VdafError::Vidpf(VidpfError::InvalidInputLength))?,
                &beta_share[1..],
                &input_share.proof_share,
                verify_key,
                nonce,
            )?;

            let verifier_len = szk_query_share.flp_verifier.len();
            (
                MasticPrepareState {
                    output_shares: MasticOutputShare::from(truncated_out_shares),
                    szk_query_state,
                    verifier_len: Some(verifier_len),
                },
                MasticPrepareShare {
                    eval_proof,
                    szk_query_share_opt: Some(szk_query_share),
                },
            )
        } else {
            (
                MasticPrepareState {
                    output_shares: MasticOutputShare::from(truncated_out_shares),
                    szk_query_state: None,
                    verifier_len: None,
                },
                MasticPrepareShare {
                    eval_proof,
                    szk_query_share_opt: None,
                },
            )
        })
    }

    fn prepare_shares_to_prepare_message<M: IntoIterator<Item = MasticPrepareShare<T::Field>>>(
        &self,
        ctx: &[u8],
        _agg_param: &MasticAggregationParam,
        inputs: M,
    ) -> Result<MasticPrepareMessage, VdafError> {
        let mut inputs_iter = inputs.into_iter();
        let leader_share = inputs_iter.next().ok_or(VdafError::Uncategorized(
            "No leader share received".to_string(),
        ))?;
        let helper_share = inputs_iter.next().ok_or(VdafError::Uncategorized(
            "No helper share received".to_string(),
        ))?;
        if inputs_iter.next().is_some() {
            return Err(VdafError::Uncategorized(
                "Received more than two prepare shares".to_string(),
            ));
        };
        if leader_share.eval_proof != helper_share.eval_proof {
            return Err(VdafError::Uncategorized(
                "Vidpf proof verification failed".to_string(),
            ));
        };
        match (
            leader_share.szk_query_share_opt,
            helper_share.szk_query_share_opt,
        ) {
            // The SZK is only used once, during the first round of aggregation.
            (Some(leader_query_share), Some(helper_query_share)) => Ok(self
                .szk
                .merge_query_shares(ctx, leader_query_share, helper_query_share)?),
            (None, None) => Ok(SzkJointShare::default()),
            (_, _) => Err(VdafError::Uncategorized(
                "Only one of leader and helper query shares is present".to_string(),
            )),
        }
    }

    fn prepare_next(
        &self,
        _ctx: &[u8],
        state: MasticPrepareState<T::Field>,
        input: MasticPrepareMessage,
    ) -> Result<PrepareTransition<Self, 32, NONCE_SIZE>, VdafError> {
        let MasticPrepareState {
            output_shares,
            szk_query_state,
            verifier_len: _,
        } = state;
        self.szk.decide(szk_query_state, input)?;
        Ok(PrepareTransition::Finish(output_shares))
    }

    fn aggregate_init(&self, agg_param: &Self::AggregationParam) -> Self::AggregateShare {
        MasticAggregateShare::<T::Field>::from(vec![
            T::Field::zero();
            (1 + self.szk.typ.output_len())
                * agg_param
                    .level_and_prefixes
                    .prefixes()
                    .len()
        ])
    }

    fn aggregate<M: IntoIterator<Item = MasticOutputShare<T::Field>>>(
        &self,
        agg_param: &MasticAggregationParam,
        output_shares: M,
    ) -> Result<MasticAggregateShare<T::Field>, VdafError> {
        let mut agg_share =
            MasticAggregateShare::from(vec![T::Field::zero(); self.agg_share_len(agg_param)]);
        for output_share in output_shares.into_iter() {
            agg_share.accumulate(&output_share)?;
        }
        Ok(agg_share)
    }
}

impl<T: Type> Collector for Mastic<T> {
    fn unshard<M: IntoIterator<Item = Self::AggregateShare>>(
        &self,
        agg_param: &MasticAggregationParam,
        agg_shares: M,
        _num_measurements: usize,
    ) -> Result<Self::AggregateResult, VdafError> {
        let num_prefixes = agg_param.level_and_prefixes.prefixes().len();

        let AggregateShare(agg) = agg_shares.into_iter().try_fold(
            AggregateShare(vec![T::Field::zero(); self.agg_share_len(agg_param)]),
            |mut agg, agg_share| {
                agg.merge(&agg_share)?;
                Result::<_, VdafError>::Ok(agg)
            },
        )?;

        let mut result = Vec::with_capacity(num_prefixes);
        for agg_for_prefix in agg.chunks(1 + self.szk.typ.output_len()) {
            let num_measurements = agg_for_prefix[0];
            let num_measurements =
                <T::Field as FieldElementWithInteger>::Integer::from(num_measurements);
            let num_measurements: u64 = num_measurements.try_into().map_err(|e| {
                VdafError::Uncategorized(format!("failed to convert num_measurements to u64: {e}"))
            })?;
            let num_measurements = usize::try_from(num_measurements).map_err(|e| {
                VdafError::Uncategorized(format!(
                    "failed to convert num_measurements to usize: {e}"
                ))
            })?;
            let encoded_agg_result = &agg_for_prefix[1..];
            result.push(
                self.szk
                    .typ
                    .decode_result(encoded_agg_result, num_measurements)?,
            );
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field128, Field64};
    use crate::flp::gadgets::{Mul, ParallelSum};
    use crate::flp::types::{Count, Histogram, Sum, SumVec};
    use crate::vdaf::test_utils::run_vdaf;
    use rand::{thread_rng, Rng};

    const CTX_STR: &[u8] = b"mastic ctx";

    #[test]
    fn test_mastic_sum() {
        let algorithm_id = 6;
        let max_measurement = 29;
        let sum_typ = Sum::<Field128>::new(max_measurement).unwrap();
        let mastic = Mastic::new(algorithm_id, sum_typ, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let inputs = [
            VidpfInput::from_bytes(&[240u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[112u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[48u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[32u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[0u8, 0u8, 1u8, 4u8][..]),
        ];
        let three_prefixes = vec![VidpfInput::from_bools(&[false, false, true])];
        let individual_prefixes = vec![
            VidpfInput::from_bools(&[false]),
            VidpfInput::from_bools(&[true]),
        ];

        let first_agg_param = MasticAggregationParam::new(three_prefixes.clone(), true).unwrap();
        let second_agg_param = MasticAggregationParam::new(individual_prefixes, true).unwrap();
        let third_agg_param = MasticAggregationParam::new(three_prefixes, false).unwrap();

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &first_agg_param,
                [
                    (inputs[0].clone(), 24),
                    (inputs[1].clone(), 0),
                    (inputs[2].clone(), 0),
                    (inputs[3].clone(), 3),
                    (inputs[4].clone(), 28)
                ]
            )
            .unwrap(),
            vec![3]
        );

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &second_agg_param,
                [
                    (inputs[0].clone(), 24),
                    (inputs[1].clone(), 0),
                    (inputs[2].clone(), 0),
                    (inputs[3].clone(), 3),
                    (inputs[4].clone(), 28)
                ]
            )
            .unwrap(),
            vec![31, 24]
        );

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &third_agg_param,
                [
                    (inputs[0].clone(), 24),
                    (inputs[1].clone(), 0),
                    (inputs[2].clone(), 0),
                    (inputs[3].clone(), 3),
                    (inputs[4].clone(), 28)
                ]
            )
            .unwrap(),
            vec![3]
        );
    }

    #[test]
    fn test_input_share_encode_sum() {
        let algorithm_id = 6;
        let max_measurement = 29;
        let sum_typ = Sum::<Field128>::new(max_measurement).unwrap();
        let mastic = Mastic::new(algorithm_id, sum_typ, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let (_, input_shares) = mastic
            .shard(CTX_STR, &(first_input, 26u128), &nonce)
            .unwrap();
        let [leader_input_share, helper_input_share] = [&input_shares[0], &input_shares[1]];

        assert_eq!(
            leader_input_share.encoded_len().unwrap(),
            leader_input_share.get_encoded().unwrap().len()
        );
        assert_eq!(
            helper_input_share.encoded_len().unwrap(),
            helper_input_share.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_agg_param_roundtrip() {
        let three_prefixes = vec![VidpfInput::from_bools(&[false, false, true])];
        let individual_prefixes = vec![
            VidpfInput::from_bools(&[false]),
            VidpfInput::from_bools(&[true]),
        ];
        let agg_params = [
            MasticAggregationParam::new(three_prefixes.clone(), true).unwrap(),
            MasticAggregationParam::new(individual_prefixes, true).unwrap(),
            MasticAggregationParam::new(three_prefixes, false).unwrap(),
        ];

        let encoded_agg_params = agg_params
            .iter()
            .map(|agg_param| agg_param.get_encoded().unwrap());
        let decoded_agg_params = encoded_agg_params
            .map(|encoded_ap| MasticAggregationParam::get_decoded(&encoded_ap).unwrap());
        agg_params
            .iter()
            .zip(decoded_agg_params)
            .for_each(|(agg_param, decoded_agg_param)| assert_eq!(*agg_param, decoded_agg_param));
    }

    #[test]
    fn test_public_share_roundtrip_sum() {
        let algorithm_id = 6;
        let max_measurement = 29;
        let sum_typ = Sum::<Field128>::new(max_measurement).unwrap();
        let mastic = Mastic::new(algorithm_id, sum_typ, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let (public, _) = mastic
            .shard(CTX_STR, &(first_input, 4u128), &nonce)
            .unwrap();

        let encoded_public = public.get_encoded().unwrap();
        let decoded_public =
            MasticPublicShare::get_decoded_with_param(&mastic, &encoded_public[..]).unwrap();
        assert_eq!(public, decoded_public);
    }

    #[test]
    fn test_mastic_count() {
        let algorithm_id = 6;
        let count = Count::<Field128>::new();
        let mastic = Mastic::new(algorithm_id, count, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let inputs = [
            VidpfInput::from_bytes(&[240u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[112u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[48u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[32u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[0u8, 0u8, 1u8, 4u8][..]),
        ];
        let three_prefixes = vec![VidpfInput::from_bools(&[false, false, true])];
        let individual_prefixes = vec![
            VidpfInput::from_bools(&[false]),
            VidpfInput::from_bools(&[true]),
        ];
        let first_agg_param = MasticAggregationParam::new(three_prefixes.clone(), true).unwrap();
        let second_agg_param = MasticAggregationParam::new(individual_prefixes, true).unwrap();
        let third_agg_param = MasticAggregationParam::new(three_prefixes, false).unwrap();

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &first_agg_param,
                [
                    (inputs[0].clone(), true),
                    (inputs[1].clone(), false),
                    (inputs[2].clone(), false),
                    (inputs[3].clone(), true),
                    (inputs[4].clone(), true)
                ]
            )
            .unwrap(),
            vec![1]
        );

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &second_agg_param,
                [
                    (inputs[0].clone(), true),
                    (inputs[1].clone(), false),
                    (inputs[2].clone(), false),
                    (inputs[3].clone(), true),
                    (inputs[4].clone(), true)
                ]
            )
            .unwrap(),
            vec![2, 1]
        );

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &third_agg_param,
                [
                    (inputs[0].clone(), true),
                    (inputs[1].clone(), false),
                    (inputs[2].clone(), false),
                    (inputs[3].clone(), true),
                    (inputs[4].clone(), true)
                ]
            )
            .unwrap(),
            vec![1]
        );
    }

    #[test]
    fn test_public_share_encoded_len() {
        let algorithm_id = 6;
        let count = Count::<Field64>::new();
        let mastic = Mastic::new(algorithm_id, count, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let (public, _) = mastic.shard(CTX_STR, &(first_input, true), &nonce).unwrap();

        assert_eq!(
            public.encoded_len().unwrap(),
            public.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_public_share_roundtrip_count() {
        let algorithm_id = 6;
        let count = Count::<Field64>::new();
        let mastic = Mastic::new(algorithm_id, count, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let (public, _) = mastic.shard(CTX_STR, &(first_input, true), &nonce).unwrap();

        let encoded_public = public.get_encoded().unwrap();
        let decoded_public =
            MasticPublicShare::get_decoded_with_param(&mastic, &encoded_public[..]).unwrap();
        assert_eq!(public, decoded_public);
    }

    #[test]
    fn test_mastic_sumvec() {
        let algorithm_id = 6;
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let mastic = Mastic::new(algorithm_id, sumvec, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let inputs = [
            VidpfInput::from_bytes(&[240u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[112u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[48u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[32u8, 0u8, 1u8, 4u8][..]),
            VidpfInput::from_bytes(&[0u8, 0u8, 1u8, 4u8][..]),
        ];

        let measurements = [
            vec![1u128, 16u128, 0u128],
            vec![0u128, 0u128, 0u128],
            vec![0u128, 0u128, 0u128],
            vec![1u128, 17u128, 31u128],
            vec![6u128, 4u128, 11u128],
        ];

        let three_prefixes = vec![VidpfInput::from_bools(&[false, false, true])];
        let individual_prefixes = vec![
            VidpfInput::from_bools(&[false]),
            VidpfInput::from_bools(&[true]),
        ];
        let first_agg_param = MasticAggregationParam::new(three_prefixes.clone(), true).unwrap();
        let second_agg_param = MasticAggregationParam::new(individual_prefixes, true).unwrap();
        let third_agg_param = MasticAggregationParam::new(three_prefixes, false).unwrap();

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &first_agg_param,
                [
                    (inputs[0].clone(), measurements[0].clone()),
                    (inputs[1].clone(), measurements[1].clone()),
                    (inputs[2].clone(), measurements[2].clone()),
                    (inputs[3].clone(), measurements[3].clone()),
                    (inputs[4].clone(), measurements[4].clone()),
                ]
            )
            .unwrap(),
            vec![vec![1, 17, 31]]
        );

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &second_agg_param,
                [
                    (inputs[0].clone(), measurements[0].clone()),
                    (inputs[1].clone(), measurements[1].clone()),
                    (inputs[2].clone(), measurements[2].clone()),
                    (inputs[3].clone(), measurements[3].clone()),
                    (inputs[4].clone(), measurements[4].clone()),
                ]
            )
            .unwrap(),
            vec![vec![7, 21, 42], vec![1, 16, 0]]
        );

        assert_eq!(
            run_vdaf(
                CTX_STR,
                &mastic,
                &third_agg_param,
                [
                    (inputs[0].clone(), measurements[0].clone()),
                    (inputs[1].clone(), measurements[1].clone()),
                    (inputs[2].clone(), measurements[2].clone()),
                    (inputs[3].clone(), measurements[3].clone()),
                    (inputs[4].clone(), measurements[4].clone()),
                ]
            )
            .unwrap(),
            vec![vec![1, 17, 31]]
        );
    }

    #[test]
    fn test_input_share_encode_sumvec() {
        let algorithm_id = 6;
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let measurement = vec![1, 16, 0];
        let mastic = Mastic::new(algorithm_id, sumvec, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let (_public, input_shares) = mastic
            .shard(CTX_STR, &(first_input, measurement), &nonce)
            .unwrap();
        let leader_input_share = &input_shares[0];
        let helper_input_share = &input_shares[1];

        assert_eq!(
            leader_input_share.encoded_len().unwrap(),
            leader_input_share.get_encoded().unwrap().len()
        );
        assert_eq!(
            helper_input_share.encoded_len().unwrap(),
            helper_input_share.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_input_share_roundtrip_sumvec() {
        let algorithm_id = 6;
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let measurement = vec![1, 16, 0];
        let mastic = Mastic::new(algorithm_id, sumvec, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let (_public, input_shares) = mastic
            .shard(CTX_STR, &(first_input, measurement), &nonce)
            .unwrap();
        let leader_input_share = &input_shares[0];
        let helper_input_share = &input_shares[1];

        let encoded_input_share = leader_input_share.get_encoded().unwrap();
        let decoded_leader_input_share =
            MasticInputShare::get_decoded_with_param(&(&mastic, 0), &encoded_input_share[..])
                .unwrap();
        assert_eq!(leader_input_share, &decoded_leader_input_share);
        let encoded_input_share = helper_input_share.get_encoded().unwrap();
        let decoded_helper_input_share =
            MasticInputShare::get_decoded_with_param(&(&mastic, 1), &encoded_input_share[..])
                .unwrap();
        assert_eq!(helper_input_share, &decoded_helper_input_share);
    }

    #[test]
    fn test_public_share_encode_sumvec() {
        let algorithm_id = 6;
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let measurement = vec![1, 16, 0];
        let mastic = Mastic::new(algorithm_id, sumvec, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let (public, _) = mastic
            .shard(CTX_STR, &(first_input, measurement), &nonce)
            .unwrap();

        assert_eq!(
            public.encoded_len().unwrap(),
            public.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_public_share_roundtrip_sumvec() {
        let algorithm_id = 6;
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let measurement = vec![1, 16, 0];
        let mastic = Mastic::new(algorithm_id, sumvec, 32).unwrap();

        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let (public, _) = mastic
            .shard(CTX_STR, &(first_input, measurement), &nonce)
            .unwrap();

        let encoded_public_share = public.get_encoded().unwrap();
        let decoded_public_share =
            MasticPublicShare::get_decoded_with_param(&mastic, &encoded_public_share[..]).unwrap();
        assert_eq!(public, decoded_public_share);
    }

    mod prep_state {
        use super::*;

        fn test_prep_state_roundtrip<T: Type>(
            typ: T,
            weight: T::Measurement,
            require_weight_check: bool,
        ) {
            let mastic = Mastic::new(0, typ, 256).unwrap();
            let ctx = b"some application";
            let verify_key = [0u8; 32];
            let nonce = [0u8; 16];
            let alpha = VidpfInput::from_bools(&[false; 256][..]);
            let (public_share, input_shares) =
                mastic.shard(ctx, &(alpha.clone(), weight), &nonce).unwrap();
            let agg_param = MasticAggregationParam::new(
                vec![alpha, VidpfInput::from_bools(&[true; 256][..])],
                require_weight_check,
            )
            .unwrap();

            // Test both aggregators.
            for agg_id in [0, 1] {
                let (prep_state, _prep_share) = mastic
                    .prepare_init(
                        &verify_key,
                        ctx,
                        agg_id,
                        &agg_param,
                        &nonce,
                        &public_share,
                        &input_shares[agg_id],
                    )
                    .unwrap();

                let encoded = prep_state.get_encoded().unwrap();
                assert_eq!(Some(encoded.len()), prep_state.encoded_len());
                assert_eq!(
                    MasticPrepareState::get_decoded_with_param(&(&mastic, &agg_param), &encoded)
                        .unwrap(),
                    prep_state
                );
            }
        }

        #[test]
        fn without_joint_rand() {
            // The Count type doesn't use joint randomness, which means the prep share won't carry the
            // aggregator's joint randomness part in the weight check.
            test_prep_state_roundtrip(Count::<Field64>::new(), true, true);
        }

        #[test]
        fn without_weight_check() {
            let histogram: Histogram<Field128, ParallelSum<_, Mul<_>>> =
                Histogram::new(10, 3).unwrap();
            // The agg param doesn't request a weight check, so the prep share won't include it.
            test_prep_state_roundtrip(histogram, 0, false);
        }

        #[test]
        fn with_weight_check_and_joint_rand() {
            let histogram: Histogram<Field128, ParallelSum<_, Mul<_>>> =
                Histogram::new(10, 3).unwrap();
            test_prep_state_roundtrip(histogram, 0, true);
        }
    }

    mod test_vec {
        use serde::Deserialize;
        use std::collections::HashMap;

        use super::*;
        use crate::{
            flp::{
                types::{Histogram, MultihotCountVec},
                Type,
            },
            idpf::IdpfInput,
        };

        #[derive(Debug, Deserialize)]
        struct HexEncoded(#[serde(with = "hex")] Vec<u8>);

        fn check_test_vec<T: Type>(
            algorithm_id: u32,
            new_typ: impl Fn(&HashMap<String, serde_json::Value>) -> T,
            test_vec_str: &str,
        ) where
            T::Measurement: for<'a> Deserialize<'a>,
            T::AggregateResult: for<'a> Deserialize<'a> + PartialEq,
        {
            #[derive(Debug, Deserialize)]
            struct TestVector<T>
            where
                T: Type,
                T::Measurement: for<'a> Deserialize<'a>,
                T::AggregateResult: for<'a> Deserialize<'a>,
            {
                agg_param: HexEncoded,
                agg_result: Vec<T::AggregateResult>,
                agg_shares: [HexEncoded; 2],
                vidpf_bits: usize,
                ctx: HexEncoded,
                prep: Vec<PrepTestVector<T::Measurement>>,
                #[serde(flatten)]
                type_params: HashMap<String, serde_json::Value>,
                shares: usize,
                verify_key: HexEncoded,
            }

            #[derive(Debug, Deserialize)]
            struct PrepTestVector<M> {
                input_shares: [HexEncoded; 2],
                measurement: (Vec<bool>, M),
                nonce: HexEncoded,
                out_shares: [Vec<HexEncoded>; 2],
                prep_messages: [HexEncoded; 1],
                prep_shares: [[HexEncoded; 2]; 1],
                public_share: HexEncoded,
                rand: HexEncoded,
            }

            let test_vec: TestVector<T> = serde_json::from_str(test_vec_str).unwrap();

            let mastic = Mastic::new(
                algorithm_id,
                new_typ(&test_vec.type_params),
                test_vec.vidpf_bits,
            )
            .unwrap();

            let agg_param = MasticAggregationParam::get_decoded(&test_vec.agg_param.0).unwrap();

            let verify_key = <[u8; 32]>::try_from(&test_vec.verify_key.0[..]).unwrap();

            let HexEncoded(ctx) = &test_vec.ctx;

            let mut out_shares_0 = Vec::new();
            let mut out_shares_1 = Vec::new();
            for prep in &test_vec.prep {
                let measurement = (
                    IdpfInput::from_bools(&prep.measurement.0),
                    prep.measurement.1.clone(),
                );
                let nonce = <[u8; NONCE_SIZE]>::try_from(&prep.nonce.0[..]).unwrap();
                let (vidpf_keys, szk_random, joint_random_opt) = {
                    let mut r = Cursor::new(prep.rand.0.as_ref());
                    let vidpf_keys = [Seed::decode(&mut r).unwrap(), Seed::decode(&mut r).unwrap()];
                    let szk_random = [Seed::decode(&mut r).unwrap(), Seed::decode(&mut r).unwrap()];

                    let joint_random_opt = mastic
                        .szk
                        .requires_joint_rand()
                        .then(|| Seed::decode(&mut r).unwrap());
                    (vidpf_keys, szk_random, joint_random_opt)
                };

                // Sharding.
                let (public_share, input_shares) = mastic
                    .shard_with_random(
                        ctx,
                        &measurement,
                        &nonce,
                        vidpf_keys,
                        szk_random,
                        joint_random_opt,
                    )
                    .unwrap();
                {
                    let expected_public_share =
                        MasticPublicShare::get_decoded_with_param(&mastic, &prep.public_share.0)
                            .unwrap();
                    assert_eq!(public_share, expected_public_share);
                    assert_eq!(public_share.get_encoded().unwrap(), prep.public_share.0);

                    let expected_input_shares = prep
                        .input_shares
                        .iter()
                        .enumerate()
                        .map(|(agg_id, HexEncoded(bytes))| {
                            MasticInputShare::get_decoded_with_param(&(&mastic, agg_id), bytes)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(input_shares, expected_input_shares);
                    assert_eq!(
                        input_shares[0].get_encoded().unwrap(),
                        prep.input_shares[0].0
                    );
                    assert_eq!(
                        input_shares[1].get_encoded().unwrap(),
                        prep.input_shares[1].0
                    );
                }

                // Preparation.
                let (prep_state_0, prep_share_0) = mastic
                    .prepare_init(
                        &verify_key,
                        ctx,
                        0,
                        &agg_param,
                        &nonce,
                        &public_share,
                        &input_shares[0],
                    )
                    .unwrap();
                let (prep_state_1, prep_share_1) = mastic
                    .prepare_init(
                        &verify_key,
                        ctx,
                        1,
                        &agg_param,
                        &nonce,
                        &public_share,
                        &input_shares[1],
                    )
                    .unwrap();

                {
                    let expected_prep_share_0 = MasticPrepareShare::get_decoded_with_param(
                        &prep_state_0,
                        &prep.prep_shares[0][0].0,
                    )
                    .unwrap();
                    assert_eq!(prep_share_0, expected_prep_share_0);
                    assert_eq!(
                        prep_share_0.get_encoded().unwrap(),
                        prep.prep_shares[0][0].0
                    );

                    let expected_prep_share_1 = MasticPrepareShare::get_decoded_with_param(
                        &prep_state_1,
                        &prep.prep_shares[0][1].0,
                    )
                    .unwrap();
                    assert_eq!(prep_share_1, expected_prep_share_1);
                    assert_eq!(
                        prep_share_1.get_encoded().unwrap(),
                        prep.prep_shares[0][1].0
                    );
                }

                let prep_msg = mastic
                    .prepare_shares_to_prepare_message(
                        ctx,
                        &agg_param,
                        [prep_share_0, prep_share_1],
                    )
                    .unwrap();
                {
                    let expected_prep_msg = MasticPrepareMessage::get_decoded_with_param(
                        &prep_state_0,
                        &prep.prep_messages[0].0,
                    )
                    .unwrap();
                    assert_eq!(prep_msg, expected_prep_msg);
                    assert_eq!(prep_msg.get_encoded().unwrap(), prep.prep_messages[0].0);
                }

                let PrepareTransition::Finish(out_share_0) = mastic
                    .prepare_next(ctx, prep_state_0, prep_msg.clone())
                    .unwrap()
                else {
                    panic!("unexpected transition");
                };
                let PrepareTransition::Finish(out_share_1) =
                    mastic.prepare_next(ctx, prep_state_1, prep_msg).unwrap()
                else {
                    panic!("unexpected transition");
                };
                {
                    let expected_out_shares = prep
                        .out_shares
                        .iter()
                        .map(|out_share| {
                            MasticOutputShare::from(
                                out_share
                                    .iter()
                                    .map(|HexEncoded(bytes)| T::Field::get_decoded(bytes).unwrap())
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(out_share_0, expected_out_shares[0]);
                    assert_eq!(out_share_1, expected_out_shares[1]);
                }

                out_shares_0.push(out_share_0);
                out_shares_1.push(out_share_1);
            }

            // Aggregation.
            let agg_share_0 = mastic.aggregate(&agg_param, out_shares_0).unwrap();
            let agg_share_1 = mastic.aggregate(&agg_param, out_shares_1).unwrap();
            {
                let expected_agg_shares = test_vec
                    .agg_shares
                    .iter()
                    .map(|HexEncoded(bytes)| {
                        MasticAggregateShare::get_decoded_with_param(&(&mastic, &agg_param), bytes)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(agg_share_0, expected_agg_shares[0]);
                assert_eq!(agg_share_1, expected_agg_shares[1]);
            }

            // Unsharding.
            let agg_result = mastic
                .unshard(&agg_param, [agg_share_0, agg_share_1], test_vec.prep.len())
                .unwrap();
            assert_eq!(agg_result, test_vec.agg_result);

            assert_eq!(test_vec.shares, 2);
        }

        #[test]
        fn count_0() {
            check_test_vec(
                0xFFFF0001,
                |_type_params| Count::<Field64>::new(),
                include_str!("test_vec/mastic/04/MasticCount_0.json"),
            );
        }

        #[test]
        fn count_1() {
            check_test_vec(
                0xFFFF0001,
                |_type_params| Count::<Field64>::new(),
                include_str!("test_vec/mastic/04/MasticCount_1.json"),
            );
        }

        #[test]
        fn count_2() {
            check_test_vec(
                0xFFFF0001,
                |_type_params| Count::<Field64>::new(),
                include_str!("test_vec/mastic/04/MasticCount_2.json"),
            );
        }

        #[test]
        fn count_3() {
            check_test_vec(
                0xFFFF0001,
                |_type_params| Count::<Field64>::new(),
                include_str!("test_vec/mastic/04/MasticCount_3.json"),
            );
        }

        #[test]
        fn sum_0() {
            check_test_vec(
                0xFFFF0002,
                |type_params| {
                    let max_measurement = type_params["max_measurement"].as_u64().unwrap();
                    Sum::<Field64>::new(max_measurement).unwrap()
                },
                include_str!("test_vec/mastic/04/MasticSum_0.json"),
            );
        }

        #[test]
        fn sum_1() {
            check_test_vec(
                0xFFFF0002,
                |type_params| {
                    let max_measurement = type_params["max_measurement"].as_u64().unwrap();
                    Sum::<Field64>::new(max_measurement).unwrap()
                },
                include_str!("test_vec/mastic/04/MasticSum_1.json"),
            );
        }

        #[test]
        fn sum_vec_0() {
            check_test_vec(
                0xFFFF0003,
                |type_params| {
                    let bits = type_params["bits"].as_u64().unwrap() as usize;
                    let length = type_params["length"].as_u64().unwrap() as usize;
                    let chunk_length = type_params["chunk_length"].as_u64().unwrap() as usize;
                    SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(
                        bits,
                        length,
                        chunk_length,
                    )
                    .unwrap()
                },
                include_str!("test_vec/mastic/04/MasticSumVec_0.json"),
            );
        }

        #[test]
        fn histogram_0() {
            check_test_vec(
                0xFFFF0004,
                |type_params| {
                    let length = type_params["length"].as_u64().unwrap() as usize;
                    let chunk_length = type_params["chunk_length"].as_u64().unwrap() as usize;
                    Histogram::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(
                        length,
                        chunk_length,
                    )
                    .unwrap()
                },
                include_str!("test_vec/mastic/04/MasticHistogram_0.json"),
            );
        }

        #[test]
        fn multihot_count_vec_0() {
            check_test_vec(
                0xFFFF0005,
                |type_params| {
                    let length = type_params["length"].as_u64().unwrap() as usize;
                    let max_weight = type_params["max_weight"].as_u64().unwrap() as usize;
                    let chunk_length = type_params["chunk_length"].as_u64().unwrap() as usize;
                    MultihotCountVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(
                        length,
                        max_weight,
                        chunk_length,
                    )
                    .unwrap()
                },
                include_str!("test_vec/mastic/04/MasticMultihotCountVec_0.json"),
            );
        }
    }
}
