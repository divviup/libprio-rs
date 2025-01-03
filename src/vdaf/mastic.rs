// SPDX-License-Identifier: MPL-2.0

//! Implementation of Mastic as specified in [[draft-mouris-cfrg-mastic-01]].
//!
//! [draft-mouris-cfrg-mastic-01]: https://www.ietf.org/archive/id/draft-mouris-cfrg-mastic-01.html

use crate::{
    bt::BinaryTree,
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{decode_fieldvec, FieldElement},
    flp::{
        szk::{Szk, SzkJointShare, SzkProofShare, SzkQueryShare, SzkQueryState},
        FlpError, Type,
    },
    vdaf::{
        poplar1::{Poplar1, Poplar1AggregationParam},
        xof::{Seed, Xof},
        Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare,
        PrepareTransition, Vdaf, VdafError,
    },
    vidpf::{
        Vidpf, VidpfError, VidpfEvalResult, VidpfInput, VidpfKey, VidpfPublicShare, VidpfServerId,
        VidpfWeight,
    },
};

use std::fmt::Debug;
use std::io::{Cursor, Read};
use std::ops::BitAnd;
use std::slice::from_ref;
use subtle::{Choice, ConstantTimeEq};

const DST_PATH_CHECK_BATCH: u16 = 6;
const NONCE_SIZE: usize = 16;

/// The main struct implementing the Mastic VDAF.
/// Composed of a shared zero knowledge proof system and a verifiable incremental
/// distributed point function.
#[derive(Clone, Debug)]
pub struct Mastic<T, P, const SEED_SIZE: usize>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    algorithm_id: u32,
    szk: Szk<T, P, SEED_SIZE>,
    pub(crate) vidpf: Vidpf<VidpfWeight<T::Field>>,
    /// The length of the private attribute associated with any input.
    pub(crate) bits: usize,
}

impl<T, P, const SEED_SIZE: usize> Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// Creates a new instance of Mastic, with a specific attribute length and weight type.
    pub fn new(
        algorithm_id: u32,
        szk: Szk<T, P, SEED_SIZE>,
        vidpf: Vidpf<VidpfWeight<T::Field>>,
        bits: usize,
    ) -> Self {
        Self {
            algorithm_id,
            szk,
            vidpf,
            bits,
        }
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

impl<T, P, const SEED_SIZE: usize> ParameterizedDecode<Mastic<T, P, SEED_SIZE>>
    for MasticPublicShare<VidpfWeight<T::Field>>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        mastic: &Mastic<T, P, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        MasticPublicShare::<VidpfWeight<T::Field>>::decode_with_param(
            &(mastic.bits, mastic.vidpf.weight_parameter),
            bytes,
        )
    }
}

/// Mastic input share.
///
/// Message sent by the [`Client`] to each Aggregator during the Sharding phase.
#[derive(Clone, Debug)]
pub struct MasticInputShare<F: FieldElement, const SEED_SIZE: usize> {
    /// VIDPF key share.
    vidpf_key: VidpfKey,

    /// The proof share.
    proof_share: SzkProofShare<F, SEED_SIZE>,
}

impl<F: FieldElement, const SEED_SIZE: usize> Encode for MasticInputShare<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&self.vidpf_key.0[..]);
        self.proof_share.encode(bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(16 + self.proof_share.encoded_len()?)
    }
}

impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Mastic<T, P, SEED_SIZE>, usize)>
    for MasticInputShare<T::Field, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (mastic, agg_id): &(&'a Mastic<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *agg_id > 1 {
            return Err(CodecError::UnexpectedValue);
        }
        let mut value = [0; 16];
        bytes.read_exact(&mut value)?;
        let vidpf_key = VidpfKey::from_bytes(value);
        let proof_share = SzkProofShare::<T::Field, SEED_SIZE>::decode_with_param(
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

impl<F: FieldElement, const SEED_SIZE: usize> PartialEq for MasticInputShare<F, SEED_SIZE> {
    fn eq(&self, other: &MasticInputShare<F, SEED_SIZE>) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: FieldElement, const SEED_SIZE: usize> ConstantTimeEq for MasticInputShare<F, SEED_SIZE> {
    fn ct_eq(&self, other: &MasticInputShare<F, SEED_SIZE>) -> Choice {
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

impl<'a, T, P, const SEED_SIZE: usize>
    ParameterizedDecode<(&'a Mastic<T, P, SEED_SIZE>, &'a MasticAggregationParam)>
    for MasticAggregateShare<T::Field>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        decoding_parameter: &(&Mastic<T, P, SEED_SIZE>, &MasticAggregationParam),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let (mastic, agg_param) = decoding_parameter;
        let l = mastic
            .vidpf
            .weight_parameter
            .checked_mul(agg_param.level_and_prefixes.prefixes().len())
            .ok_or_else(|| CodecError::Other("multiplication overflow".into()))?;
        let result = decode_fieldvec(l, bytes)?;
        Ok(AggregateShare(result))
    }
}

impl<'a, T, P, const SEED_SIZE: usize>
    ParameterizedDecode<(&'a Mastic<T, P, SEED_SIZE>, &'a MasticAggregationParam)>
    for MasticOutputShare<T::Field>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        decoding_parameter: &(&Mastic<T, P, SEED_SIZE>, &MasticAggregationParam),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let (mastic, agg_param) = decoding_parameter;
        let l = mastic
            .vidpf
            .weight_parameter
            .checked_mul(agg_param.level_and_prefixes.prefixes().len())
            .ok_or_else(|| CodecError::Other("multiplication overflow".into()))?;
        let result = decode_fieldvec(l, bytes)?;
        Ok(OutputShare(result))
    }
}

impl<T, P, const SEED_SIZE: usize> Vdaf for Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    type Measurement = (VidpfInput, T::Measurement);
    type AggregateResult = Vec<T::AggregateResult>;
    type AggregationParam = MasticAggregationParam;
    type PublicShare = MasticPublicShare<VidpfWeight<T::Field>>;
    type InputShare = MasticInputShare<T::Field, SEED_SIZE>;
    type OutputShare = MasticOutputShare<T::Field>;
    type AggregateShare = MasticAggregateShare<T::Field>;

    fn algorithm_id(&self) -> u32 {
        self.algorithm_id
    }

    fn num_aggregators(&self) -> usize {
        2
    }
}
impl<T, P, const SEED_SIZE: usize> Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn shard_with_random(
        &self,
        measurement_attribute: &VidpfInput,
        measurement_weight: &VidpfWeight<T::Field>,
        nonce: &[u8; 16],
        vidpf_keys: [VidpfKey; 2],
        szk_random: [Seed<SEED_SIZE>; 2],
        joint_random_opt: Option<Seed<SEED_SIZE>>,
    ) -> Result<(<Self as Vdaf>::PublicShare, Vec<<Self as Vdaf>::InputShare>), VdafError> {
        // Compute the measurement shares for each aggregator by generating VIDPF
        // keys for the measurement and evaluating each of them.
        let public_share = self.vidpf.gen_with_keys(
            &vidpf_keys,
            measurement_attribute,
            measurement_weight,
            nonce,
        )?;

        let leader_measurement_share = self.vidpf.eval_root(
            VidpfServerId::S0,
            &vidpf_keys[0],
            &public_share,
            &mut BinaryTree::default(),
            nonce,
        )?;
        let helper_measurement_share = self.vidpf.eval_root(
            VidpfServerId::S1,
            &vidpf_keys[1],
            &public_share,
            &mut BinaryTree::default(),
            nonce,
        )?;

        let [leader_szk_proof_share, helper_szk_proof_share] = self.szk.prove(
            leader_measurement_share.as_ref(),
            helper_measurement_share.as_ref(),
            measurement_weight.as_ref(),
            szk_random,
            joint_random_opt,
            nonce,
        )?;
        let [leader_vidpf_key, helper_vidpf_key] = vidpf_keys;
        let leader_share = MasticInputShare::<T::Field, SEED_SIZE> {
            vidpf_key: leader_vidpf_key,
            proof_share: leader_szk_proof_share,
        };
        let helper_share = MasticInputShare::<T::Field, SEED_SIZE> {
            vidpf_key: helper_vidpf_key,
            proof_share: helper_szk_proof_share,
        };
        Ok((public_share, vec![leader_share, helper_share]))
    }

    fn encode_measurement(
        &self,
        measurement: &T::Measurement,
    ) -> Result<VidpfWeight<T::Field>, VdafError> {
        Ok(VidpfWeight::<T::Field>::from(
            self.szk.typ.encode_measurement(measurement)?,
        ))
    }
}

impl<T, P, const SEED_SIZE: usize> Client<16> for Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn shard(
        &self,
        _ctx: &[u8],
        (attribute, weight): &(VidpfInput, T::Measurement),
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError> {
        if attribute.len() != self.bits {
            return Err(VdafError::Vidpf(VidpfError::InvalidAttributeLength));
        }

        let vidpf_keys = [VidpfKey::generate()?, VidpfKey::generate()?];
        let joint_random_opt = if self.szk.requires_joint_rand() {
            Some(Seed::<SEED_SIZE>::generate()?)
        } else {
            None
        };
        let szk_random = [Seed::generate()?, Seed::generate()?];

        let encoded_measurement = self.encode_measurement(weight)?;
        if encoded_measurement.as_ref().len() != self.vidpf.weight_parameter {
            return Err(VdafError::Uncategorized(
                "encoded_measurement is the wrong length".to_string(),
            ));
        }
        self.shard_with_random(
            attribute,
            &encoded_measurement,
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
pub struct MasticPrepareState<F: FieldElement, const SEED_SIZE: usize> {
    /// Includes output shares for eventual aggregation.
    output_shares: MasticOutputShare<F>,
    /// If [`Szk`]` verification is being performed, we also store the relevant state for that operation.
    szk_query_state: SzkQueryState<SEED_SIZE>,
    verifier_len: Option<usize>,
}

/// Mastic preparation share.
///
/// Broadcast message from an aggregator preparing Mastic output shares. Includes the
/// [`Vidpf`] evaluation proof covering every prefix in the aggregation parameter, and optionally
/// the verification message for Szk.
#[derive(Clone, Debug)]
pub struct MasticPrepareShare<F: FieldElement, const SEED_SIZE: usize> {
    ///  [`Vidpf`] evaluation proof, which guarantees one-hotness and payload consistency.
    vidpf_proof: Seed<SEED_SIZE>,

    /// If [`Szk`]` verification of the root weight is needed, a verification message.
    szk_query_share_opt: Option<SzkQueryShare<F, SEED_SIZE>>,
}

impl<F: FieldElement, const SEED_SIZE: usize> Encode for MasticPrepareShare<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.vidpf_proof.encode(bytes)?;
        match &self.szk_query_share_opt {
            Some(query_share) => query_share.encode(bytes),
            None => Ok(()),
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.vidpf_proof.encoded_len()?
                + match &self.szk_query_share_opt {
                    Some(query_share) => query_share.encoded_len()?,
                    None => 0,
                },
        )
    }
}

impl<F: FieldElement, const SEED_SIZE: usize> ParameterizedDecode<MasticPrepareState<F, SEED_SIZE>>
    for MasticPrepareShare<F, SEED_SIZE>
{
    fn decode_with_param(
        prep_state: &MasticPrepareState<F, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let vidpf_proof = Seed::decode(bytes)?;
        let requires_joint_rand = prep_state.szk_query_state.is_some();
        let szk_query_share_opt = prep_state
            .verifier_len
            .map(|verifier_len| {
                SzkQueryShare::<F, SEED_SIZE>::decode_with_param(
                    &(requires_joint_rand, verifier_len),
                    bytes,
                )
            })
            .transpose()?;
        Ok(Self {
            vidpf_proof,
            szk_query_share_opt,
        })
    }
}

/// Mastic preparation message.
///
/// Result of preprocessing the broadcast messages of both aggregators during the
/// preparation phase.
pub type MasticPrepareMessage<const SEED_SIZE: usize> = SzkJointShare<SEED_SIZE>;

impl<F: FieldElement, const SEED_SIZE: usize> ParameterizedDecode<MasticPrepareState<F, SEED_SIZE>>
    for MasticPrepareMessage<SEED_SIZE>
{
    fn decode_with_param(
        prep_state: &MasticPrepareState<F, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        match prep_state.szk_query_state {
            Some(_) => SzkJointShare::<SEED_SIZE>::decode_with_param(&true, bytes),
            None => SzkJointShare::<SEED_SIZE>::decode_with_param(&false, bytes),
        }
    }
}

impl<T, P, const SEED_SIZE: usize> Aggregator<SEED_SIZE, NONCE_SIZE> for Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    type PrepareState = MasticPrepareState<T::Field, SEED_SIZE>;
    type PrepareShare = MasticPrepareShare<T::Field, SEED_SIZE>;
    type PrepareMessage = MasticPrepareMessage<SEED_SIZE>;

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
        Poplar1::<P, SEED_SIZE>::is_agg_param_valid(cur_poplar_agg_param, prev_poplar_agg_param)
    }

    fn prepare_init(
        &self,
        verify_key: &[u8; SEED_SIZE],
        ctx: &[u8],
        agg_id: usize,
        agg_param: &MasticAggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &MasticPublicShare<VidpfWeight<T::Field>>,
        input_share: &MasticInputShare<T::Field, SEED_SIZE>,
    ) -> Result<
        (
            MasticPrepareState<T::Field, SEED_SIZE>,
            MasticPrepareShare<T::Field, SEED_SIZE>,
        ),
        VdafError,
    > {
        let id = match agg_id {
            0 => Ok(VidpfServerId::S0),
            1 => Ok(VidpfServerId::S1),
            _ => Err(VdafError::Uncategorized(
                "Invalid aggregator ID".to_string(),
            )),
        }?;
        let mut eval_proof = P::init(
            verify_key,
            &self.domain_separation_tag(DST_PATH_CHECK_BATCH, ctx),
        );
        let mut output_shares = Vec::<T::Field>::with_capacity(
            self.vidpf.weight_parameter * agg_param.level_and_prefixes.prefixes().len(),
        );
        let mut cache_tree = BinaryTree::<VidpfEvalResult<VidpfWeight<T::Field>>>::default();
        for prefix in agg_param.level_and_prefixes.prefixes() {
            let (VidpfWeight(mut weight_share), onehot_proof_for_prefix) =
                self.vidpf.eval_with_cache(
                    id,
                    &input_share.vidpf_key,
                    public_share,
                    prefix,
                    &mut cache_tree,
                    nonce,
                )?;
            eval_proof.update(&onehot_proof_for_prefix);
            output_shares.append(&mut weight_share);
        }

        Ok(if agg_param.require_weight_check {
            let MasticInputShare {
                vidpf_key,
                proof_share,
            } = input_share;
            let root_share =
                self.vidpf
                    .eval_root(id, vidpf_key, public_share, &mut cache_tree, nonce)?;
            let (szk_query_share, szk_query_state) =
                self.szk
                    .query(root_share.as_ref(), proof_share, verify_key, nonce)?;
            let verifier_len = szk_query_share.flp_verifier.len();
            (
                MasticPrepareState {
                    output_shares: MasticOutputShare::<T::Field>::from(output_shares),
                    szk_query_state,
                    verifier_len: Some(verifier_len),
                },
                MasticPrepareShare {
                    vidpf_proof: eval_proof.into_seed(),
                    szk_query_share_opt: Some(szk_query_share),
                },
            )
        } else {
            (
                MasticPrepareState {
                    output_shares: MasticOutputShare::<T::Field>::from(output_shares),
                    szk_query_state: None,
                    verifier_len: None,
                },
                MasticPrepareShare {
                    vidpf_proof: eval_proof.into_seed(),
                    szk_query_share_opt: None,
                },
            )
        })
    }

    fn prepare_shares_to_prepare_message<
        M: IntoIterator<Item = MasticPrepareShare<T::Field, SEED_SIZE>>,
    >(
        &self,
        _ctx: &[u8],
        _agg_param: &MasticAggregationParam,
        inputs: M,
    ) -> Result<MasticPrepareMessage<SEED_SIZE>, VdafError> {
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
        if leader_share.vidpf_proof != helper_share.vidpf_proof {
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
                .merge_query_shares(leader_query_share, helper_query_share)?),
            (None, None) => Ok(SzkJointShare::none()),
            (_, _) => Err(VdafError::Uncategorized(
                "Only one of leader and helper query shares is present".to_string(),
            )),
        }
    }

    fn prepare_next(
        &self,
        _ctx: &[u8],
        state: MasticPrepareState<T::Field, SEED_SIZE>,
        input: MasticPrepareMessage<SEED_SIZE>,
    ) -> Result<PrepareTransition<Self, SEED_SIZE, NONCE_SIZE>, VdafError> {
        let MasticPrepareState {
            output_shares,
            szk_query_state,
            verifier_len: _,
        } = state;
        self.szk.decide(szk_query_state, input)?;
        Ok(PrepareTransition::Finish(output_shares))
    }

    fn aggregate<M: IntoIterator<Item = MasticOutputShare<T::Field>>>(
        &self,
        agg_param: &MasticAggregationParam,
        output_shares: M,
    ) -> Result<MasticAggregateShare<T::Field>, VdafError> {
        let mut agg_share = MasticAggregateShare::<T::Field>::from(vec![
            T::Field::zero();
            self.vidpf.weight_parameter
                * agg_param
                    .level_and_prefixes
                    .prefixes()
                    .len()
        ]);
        for output_share in output_shares.into_iter() {
            agg_share.accumulate(&output_share)?;
        }
        Ok(agg_share)
    }
}

impl<T, P, const SEED_SIZE: usize> Collector for Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn unshard<M: IntoIterator<Item = Self::AggregateShare>>(
        &self,
        agg_param: &MasticAggregationParam,
        agg_shares: M,
        _num_measurements: usize,
    ) -> Result<Self::AggregateResult, VdafError> {
        let num_prefixes = agg_param.level_and_prefixes.prefixes().len();
        let mut agg_final = MasticAggregateShare::<T::Field>::from(vec![
            T::Field::zero();
            self.vidpf.weight_parameter
                * num_prefixes
        ]);
        for agg_share in agg_shares.into_iter() {
            agg_final.merge(&agg_share)?;
        }
        let mut iter = agg_final
            .0
            .chunks(self.vidpf.weight_parameter)
            .take(num_prefixes);
        let mut result = Vec::<T::AggregateResult>::with_capacity(num_prefixes);
        iter.try_for_each(|encoded_result| -> Result<(), FlpError> {
            result.push(
                self.szk
                    .typ
                    .decode_result(&self.szk.typ.truncate(encoded_result.to_vec())?[..], 1)?,
            );
            Ok(())
        })?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field128;
    use crate::flp::gadgets::{Mul, ParallelSum};
    use crate::flp::types::{Count, Sum, SumVec};
    use crate::vdaf::test_utils::run_vdaf;
    use rand::{thread_rng, Rng};

    const CTX_STR: &[u8] = b"mastic ctx";

    #[test]
    fn test_mastic_sum() {
        let algorithm_id = 6;
        let max_measurement = 29;
        let sum_typ = Sum::<Field128>::new(max_measurement).unwrap();
        let encoded_meas_len = sum_typ.input_len();

        let sum_szk = Szk::new_turboshake128(sum_typ, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(encoded_meas_len);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
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

        let mastic = Mastic::new(algorithm_id, sum_szk, sum_vidpf, 32);
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
        let encoded_meas_len = sum_typ.input_len();

        let sum_szk = Szk::new_turboshake128(sum_typ, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(encoded_meas_len);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, sum_szk, sum_vidpf, 32);
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
        let encoded_meas_len = sum_typ.input_len();
        let sum_szk = Szk::new_turboshake128(sum_typ, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(encoded_meas_len);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, sum_szk, sum_vidpf, 32);
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
        let szk = Szk::new_turboshake128(count, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(1);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
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
        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
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
        let count = Count::<Field128>::new();
        let szk = Szk::new_turboshake128(count, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(1);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);
        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
        let (public, _) = mastic.shard(CTX_STR, &(first_input, true), &nonce).unwrap();

        assert_eq!(
            public.encoded_len().unwrap(),
            public.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_public_share_roundtrip_count() {
        let algorithm_id = 6;
        let count = Count::<Field128>::new();
        let szk = Szk::new_turboshake128(count, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(1);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
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
        let szk = Szk::new_turboshake128(sumvec, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
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
        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
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
        let szk = Szk::new_turboshake128(sumvec, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
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
        let szk = Szk::new_turboshake128(sumvec, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
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
        let szk = Szk::new_turboshake128(sumvec, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
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
        let szk = Szk::new_turboshake128(sumvec, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
        let (public, _) = mastic
            .shard(CTX_STR, &(first_input, measurement), &nonce)
            .unwrap();

        let encoded_public_share = public.get_encoded().unwrap();
        let decoded_public_share =
            MasticPublicShare::get_decoded_with_param(&mastic, &encoded_public_share[..]).unwrap();
        assert_eq!(public, decoded_public_share);
    }
}
