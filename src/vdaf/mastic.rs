// SPDX-License-Identifier: MPL-2.0

//! Implementation of Mastic as specified in [[draft-mouris-cfrg-mastic-01]].
//!
//! [draft-mouris-cfrg-mastic-01]: https://www.ietf.org/archive/id/draft-mouris-cfrg-mastic-01.html

use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{decode_fieldvec, FieldElement},
    flp::{
        szk::{Szk, SzkProofShare},
        Type,
    },
    vdaf::{
        poplar1::Poplar1AggregationParam,
        xof::{Seed, Xof},
        AggregateShare, Client, OutputShare, Vdaf, VdafError,
    },
    vidpf::{
        Vidpf, VidpfError, VidpfInput, VidpfKey, VidpfPublicShare, VidpfServerId, VidpfWeight,
    },
};

use std::fmt::Debug;
use std::io::{Cursor, Read};
use std::ops::BitAnd;
use subtle::{Choice, ConstantTimeEq};

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
    pub(crate) vidpf: Vidpf<VidpfWeight<T::Field>, 16>,
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
        vidpf: Vidpf<VidpfWeight<T::Field>, 16>,
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
#[derive(Clone, Debug)]
pub struct MasticAggregationParam {
    /// aggregation parameter inherited from [`Poplar1`]: contains the level (attribute length) and a vector of attribute prefixes (IdpfInputs)
    level_and_prefixes: Poplar1AggregationParam,
    /// Flag indicating whether the VIDPF weight needs to be validated using SZK.
    /// This flag must be set the first time any report is aggregated; however this may happen at any level of the tree.
    require_check_flag: bool,
}

impl Encode for MasticAggregationParam {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.level_and_prefixes.encode(bytes)?;
        let require_check = if self.require_check_flag { 1u8 } else { 0u8 };
        require_check.encode(bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.level_and_prefixes.encoded_len()? + 1usize)
    }
}

impl Decode for MasticAggregationParam {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let level_and_prefixes = Poplar1AggregationParam::decode(bytes)?;
        let require_check = u8::decode(bytes)?;
        let require_check_flag = require_check != 0;
        Ok(Self {
            level_and_prefixes,
            require_check_flag,
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

/// Mastic input share
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
        bytes.extend_from_slice(&self.vidpf_key.value[..]);
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
        let vidpf_key = VidpfKey::new(
            if *agg_id == 0 {
                VidpfServerId::S0
            } else {
                VidpfServerId::S1
            },
            value,
        );

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

#[cfg(test)]
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
    type AggregateResult = T::AggregateResult;
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

        let leader_measurement_share =
            self.vidpf.eval_root(&vidpf_keys[0], &public_share, nonce)?;
        let helper_measurement_share =
            self.vidpf.eval_root(&vidpf_keys[1], &public_share, nonce)?;

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
        (attribute, weight): &(VidpfInput, T::Measurement),
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError> {
        if attribute.len() != self.bits {
            return Err(VdafError::Vidpf(VidpfError::InvalidAttributeLength));
        }

        let vidpf_keys = [
            VidpfKey::gen(VidpfServerId::S0)?,
            VidpfKey::gen(VidpfServerId::S1)?,
        ];
        let joint_random_opt = if self.szk.has_joint_rand() {
            Some(Seed::<SEED_SIZE>::generate()?)
        } else {
            None
        };
        let szk_random = [
            Seed::<SEED_SIZE>::generate()?,
            Seed::<SEED_SIZE>::generate()?,
        ];

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field128;
    use crate::flp::gadgets::{Mul, ParallelSum};
    use crate::flp::types::{Count, Sum, SumVec};
    use rand::{thread_rng, Rng};

    const TEST_NONCE_SIZE: usize = 16;

    #[test]
    fn test_mastic_shard_sum() {
        let algorithm_id = 6;
        let sum_typ = Sum::<Field128>::new(5).unwrap();
        let sum_szk = Szk::new_turboshake128(sum_typ, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(5);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, sum_szk, sum_vidpf, 32);
        let (_public, _input_shares) = mastic.shard(&(first_input, 24u128), &nonce).unwrap();
    }

    #[test]
    fn test_input_share_encode_sum() {
        let algorithm_id = 6;
        let sum_typ = Sum::<Field128>::new(5).unwrap();
        let sum_szk = Szk::new_turboshake128(sum_typ, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(5);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, sum_szk, sum_vidpf, 32);
        let (_, input_shares) = mastic.shard(&(first_input, 26u128), &nonce).unwrap();
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
    fn test_mastic_shard_count() {
        let algorithm_id = 6;
        let count = Count::<Field128>::new();
        let szk = Szk::new_turboshake128(count, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(1);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
        let (_public, _input_shares) = mastic.shard(&(first_input, true), &nonce).unwrap();
    }

    #[test]
    fn test_mastic_shard_sumvec() {
        let algorithm_id = 6;
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let measurement = vec![1, 16, 0];
        let szk = Szk::new_turboshake128(sumvec, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
        let (_public, _input_shares) = mastic.shard(&(first_input, measurement), &nonce).unwrap();
    }

    #[test]
    fn test_input_share_encode_sumvec() {
        let algorithm_id = 6;
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let measurement = vec![1, 16, 0];
        let szk = Szk::new_turboshake128(sumvec, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
        let (_public, input_shares) = mastic.shard(&(first_input, measurement), &nonce).unwrap();
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
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
        let (_public, input_shares) = mastic.shard(&(first_input, measurement), &nonce).unwrap();
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
}
