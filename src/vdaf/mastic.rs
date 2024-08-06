// SPDX-License-Identifier: MPL-2.0

//! Implementation of Mastic as specified in [[draft-mouris-cfrg-mastic-01]].
//!
//! [draft-mouris-cfrg-mastic-01]: https://www.ietf.org/archive/id/draft-mouris-cfrg-mastic-01.html

use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::FieldElement,
    flp::{
        szk::{Szk, SzkProofShare},
        Type,
    },
    vdaf::{
        poplar1::Poplar1AggregationParam,
        xof::{Seed, Xof},
        Aggregatable, Client, OutputShare, Vdaf, VdafError,
    },
    vidpf::{
        Vidpf, VidpfError, VidpfInput, VidpfKey, VidpfPublicShare, VidpfServerId, VidpfWeight,
    },
};
use std::ops::{BitAnd, Not};
use std::{fmt::Debug, io::Cursor};
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
    /// aggregation parameter inherited from [`Poplar1`]: contains the level (attribute length) and a vector of attributes (IdpfInputs)
    poplar_param: Poplar1AggregationParam,
    /// Flag indicating whether the VIDPF weight needs to be validated using SZK.
    root_check_flag: bool,
}

impl Encode for MasticAggregationParam {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.poplar_param.encode(bytes)?;
        let root_check = if self.root_check_flag { 1u8 } else { 0u8 };
        root_check.encode(bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.poplar_param.encoded_len()? + 1usize)
    }
}

impl Decode for MasticAggregationParam {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let poplar_param = Poplar1AggregationParam::decode(bytes)?;
        let root_check = u8::decode(bytes)?;
        let root_check_flag = root_check == 0;
        Ok(Self {
            poplar_param,
            root_check_flag,
        })
    }
}

/// Mastic public share
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
    proofs_share: SzkProofShare<F, SEED_SIZE>,
}

impl<F: FieldElement, const SEED_SIZE: usize> Encode for MasticInputShare<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.vidpf_key.encode(bytes)?;
        self.proofs_share.encode(bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.vidpf_key.encoded_len()? + self.proofs_share.encoded_len()?)
    }
}

impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Mastic<T, P, SEED_SIZE>, usize)>
    for MasticInputShare<T::Field, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (mastic, role): &(&'a Mastic<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *role > 1 {
            return Err(CodecError::UnexpectedValue);
        }
        let vidpf_key = VidpfKey::decode_with_param(&(*role == 0), bytes)?;
        let proofs_share = SzkProofShare::<T::Field, SEED_SIZE>::decode_with_param(
            &(
                *role == 0,
                mastic.szk.proof_len(),
                mastic.szk.has_joint_rand(),
            ),
            bytes,
        )?;
        Ok(Self {
            vidpf_key,
            proofs_share,
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
            .bitand(self.proofs_share.ct_eq(&other.proofs_share))
    }

    fn ct_ne(&self, other: &MasticInputShare<F, SEED_SIZE>) -> Choice {
        self.ct_eq(other).not()
    }
}

/// Mastic output share
///
/// Contains a vector of VIDPF outputs: one for each prefix.
pub type MasticOutputShare<V> = OutputShare<V>;

/// Vector of VidpfWeights to be aggregated by Mastic aggregators
pub type MasticAggregateShare<V> = MasticOutputShare<V>;

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
        let l = mastic.vidpf.weight_parameter * agg_param.poplar_param.prefixes().len();
        let mut result = Vec::<T::Field>::with_capacity(l);
        for _ in 0..l {
            result.append(&mut Vec::<T::Field>::decode_with_param(
                &mastic.vidpf.weight_parameter,
                bytes,
            )?);
        }
        Ok(OutputShare(result))
    }
}

impl<F: FieldElement> Aggregatable for MasticOutputShare<F> {
    type OutputShare = MasticOutputShare<F>;
    /// Update an aggregate share by merging it with another (`agg_share`).
    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError> {
        if self.0.len() != agg_share.0.len() {
            return Err(VdafError::Uncategorized(
                "Attempted to merge two output shares with different agg_params".to_string(),
            ));
        };
        for (a, o) in self.0.iter_mut().zip(agg_share.0.iter()) {
            *a += *o;
        }
        Ok(())
    }

    /// Update an aggregate share by adding `output_share`.
    fn accumulate(&mut self, output_share: &Self::OutputShare) -> Result<(), VdafError> {
        if self.0.len() != output_share.0.len() {
            return Err(VdafError::Uncategorized(
                "Attempted to accumulate two output shares with different agg_params".to_string(),
            ));
        };
        // Would love to get rid of the below clone if possible.
        for (a, o) in self.0.iter_mut().zip(output_share.0.iter()) {
            *a += *o;
        }
        Ok(())
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
    type AggregateShare = MasticOutputShare<T::Field>;

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
        opt_random: Option<Seed<SEED_SIZE>>,
    ) -> Result<(<Self as Vdaf>::PublicShare, Vec<<Self as Vdaf>::InputShare>), VdafError> {
        // Compute the measurement shares for each aggregator by generating VIDPF
        // keys for the measurement and evaluating each of them.
        let public_share = self.vidpf.gen_with_keys(
            &vidpf_keys,
            measurement_attribute,
            measurement_weight,
            nonce,
        )?;

        let leader_measurement_share = self
            .vidpf
            .eval(
                &vidpf_keys[0],
                &public_share,
                &VidpfInput::from_bools(&[false]),
                nonce,
            )?
            .share
            + self
                .vidpf
                .eval(
                    &vidpf_keys[0],
                    &public_share,
                    &VidpfInput::from_bools(&[true]),
                    nonce,
                )?
                .share;
        let helper_measurement_share = self
            .vidpf
            .eval(
                &vidpf_keys[1],
                &public_share,
                &VidpfInput::from_bools(&[false]),
                nonce,
            )?
            .share
            + self
                .vidpf
                .eval(
                    &vidpf_keys[1],
                    &public_share,
                    &VidpfInput::from_bools(&[true]),
                    nonce,
                )?
                .share;

        let szk_proof_shares = self.szk.prove(
            leader_measurement_share.as_ref(),
            helper_measurement_share.as_ref(),
            measurement_weight.as_ref(),
            szk_random,
            opt_random,
            nonce,
        )?;
        let leader_share = MasticInputShare::<T::Field, SEED_SIZE> {
            vidpf_key: vidpf_keys[0].clone(),
            proofs_share: szk_proof_shares[0].clone(),
        };
        let helper_share = MasticInputShare::<T::Field, SEED_SIZE> {
            vidpf_key: vidpf_keys[1].clone(),
            proofs_share: szk_proof_shares[1].clone(),
        };
        Ok((public_share, vec![leader_share, helper_share]))
    }

    fn encode_measurement(
        &self,
        measurement: &T::Measurement,
    ) -> Result<VidpfWeight<T::Field>, VdafError> {
        Ok(VidpfWeight::<T::Field>::from(
            self.szk.typ().encode_measurement(measurement)?,
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
        measurement: &(VidpfInput, T::Measurement),
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError> {
        if measurement.0.len() != self.bits {
            return Err(VdafError::Vidpf(VidpfError::InvalidAttributeLength));
        }

        let vidpf_keys = [
            VidpfKey::gen(VidpfServerId::S0)?,
            VidpfKey::gen(VidpfServerId::S1)?,
        ];
        let opt_random = match self.szk.has_joint_rand() {
            true => Some(Seed::<SEED_SIZE>::generate()?),
            false => None,
        };
        let szk_random = [
            Seed::<SEED_SIZE>::generate()?,
            Seed::<SEED_SIZE>::generate()?,
        ];

        let encoded_measurement = self.encode_measurement(&measurement.1)?;
        if encoded_measurement.as_ref().len() != self.vidpf.weight_parameter {
            return Err(VdafError::Uncategorized(
                "encoded_measurement is wrong length".to_string(),
            ));
        }
        self.shard_with_random(
            &measurement.0,
            &self.encode_measurement(&measurement.1)?,
            nonce,
            vidpf_keys,
            szk_random,
            opt_random,
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
        let (public, _) = mastic.shard(&(first_input, 26u128), &nonce).unwrap();

        assert_eq!(
            public.encoded_len().unwrap(),
            public.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_public_share_roundtrip_sum() {
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
        let (public, _) = mastic.shard(&(first_input, 25u128), &nonce).unwrap();

        let encoded_public = public.get_encoded().unwrap();
        let decoded_public =
            MasticPublicShare::get_decoded_with_param(&mastic, &encoded_public[..]).unwrap();
        assert_eq!(public, decoded_public);
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
    fn test_public_share_encoded_len() {
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
        let (public, _) = mastic.shard(&(first_input, true), &nonce).unwrap();

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
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(1);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
        let (public, _) = mastic.shard(&(first_input, true), &nonce).unwrap();

        let encoded_public = public.get_encoded().unwrap();
        let decoded_public =
            MasticPublicShare::get_decoded_with_param(&mastic, &encoded_public[..]).unwrap();
        assert_eq!(public, decoded_public);
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

    #[test]
    fn test_public_share_encode_sumvec() {
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
        let (public, _) = mastic.shard(&(first_input, measurement), &nonce).unwrap();

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
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(15);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8, 4u8][..]);

        let mastic = Mastic::new(algorithm_id, szk, sum_vidpf, 32);
        let (public, _) = mastic.shard(&(first_input, measurement), &nonce).unwrap();

        let encoded_public_share = public.get_encoded().unwrap();
        let decoded_public_share =
            MasticPublicShare::get_decoded_with_param(&mastic, &encoded_public_share[..]).unwrap();
        assert_eq!(public, decoded_public_share);
    }
}
