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
        Aggregatable, Client, Vdaf, VdafError,
    },
    vidpf::{
        Vidpf, VidpfInput, VidpfKey, VidpfPublicShare, VidpfServerId, VidpfValue, VidpfWeight,
    },
};
use std::{fmt::Debug, io::Cursor};
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
    vidpf: Vidpf<VidpfWeight<T::Field>, 16>,
    /// The maximum length of the private label associated with any input.
    bits: usize,
}

impl<T, P, const SEED_SIZE: usize> Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// Creates a new instance of Mastic, with a specific label length and validity type.
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

    pub(crate) fn bits(&self) -> usize {
        self.bits
    }

    pub(crate) fn vidpf(&self) -> &Vidpf<VidpfWeight<T::Field>, 16> {
        &self.vidpf
    }
}

/// Mastic aggregation parameter.
///
/// This includes the VIDPF tree level under evaluation and a set of prefixes to evaluate at that level.
pub type MasticAggregationParam = Poplar1AggregationParam;

/// Contains broadcast information shared between parties to support VIDPF correctness.
pub type MasticPublicShare<V> = VidpfPublicShare<V>;

impl<T, W, P, const SEED_SIZE: usize> ParameterizedDecode<Mastic<T, P, SEED_SIZE>>
    for MasticPublicShare<W>
where
    T: Type,
    W: VidpfValue,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        decoding_parameter: &Mastic<T, P, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        MasticPublicShare::<W>::decode_with_param(&(decoding_parameter.bits(), *decoding_parameter.vidpf().weight_parameter()), bytes)
    }
}

/// Message sent by the [`Client`] to each Aggregator during the Sharding phase.
#[derive(Clone, Debug)]
pub struct MasticInputShare<F: FieldElement, const SEED_SIZE: usize> {
    /// VIDPF key share.
    vidpf_key: VidpfKey,

    /// The proof share.
    proofs_share: SzkProofShare<F, SEED_SIZE>,
}
impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Mastic<T, P, SEED_SIZE>, usize)>
    for MasticInputShare<T::Field, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (mastic, _): &(&'a Mastic<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let role = <u8>::decode(bytes)?;
        let vidpf_key = VidpfKey::decode(bytes)?;
        let proofs_share =
            SzkProofShare::<T::Field, SEED_SIZE>::decode_with_param(&(role, mastic.szk.proof_len(), mastic.is_randomized()), bytes)?;
        Ok(Self {
            vidpf_key,
            proofs_share,
        })
    }
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

#[cfg(test)]
impl<F: FieldElement, const SEED_SIZE: usize> PartialEq for MasticInputShare<F, SEED_SIZE> {
    fn eq(&self, other: &MasticInputShare<F, SEED_SIZE>) -> bool {
        self.vidpf_key == other.vidpf_key && self.proofs_share == other.proofs_share
    }
}
/// Struct containing a vector of VIDPF outputs: one for each prefix. Likely needs to be changed in the next PR to include proofs as well.
#[derive(Clone, Debug)]
pub struct MasticOutputShare<V: VidpfValue> {
    result: Vec<V>,
}

impl<V: VidpfValue + Debug> Aggregatable for MasticOutputShare<V> {
    type OutputShare = MasticOutputShare<V>;
    /// Update an aggregate share by merging it with another (`agg_share`).
    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError> {
        if self.result.len() != agg_share.result.len() {
            return Err(VdafError::Uncategorized(
                "Attempted to merge two output shares with different agg_params".to_string(),
            ));
        };
        // Would love to get rid of the below clone if possible.
        for (a, o) in self.result.iter_mut().zip(agg_share.result.iter()) {
            *a += o.clone();
        }
        Ok(())
    }

    /// Update an aggregate share by adding `output_share`.
    fn accumulate(&mut self, output_share: &Self::OutputShare) -> Result<(), VdafError> {
        if self.result.len() != output_share.result.len() {
            return Err(VdafError::Uncategorized(
                "Attempted to accumulate two output shares with different agg_params".to_string(),
            ));
        };
        // Would love to get rid of the below clone if possible.
        for (a, o) in self.result.iter_mut().zip(output_share.result.iter()) {
            *a += o.clone();
        }
        Ok(())
    }
}

impl<'a, T, P, const SEED_SIZE: usize>
    ParameterizedDecode<(&'a Mastic<T, P, SEED_SIZE>, &'a MasticAggregationParam)>
    for MasticOutputShare<VidpfWeight<T::Field>>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        decoding_parameter: &(&Mastic<T, P, SEED_SIZE>, &MasticAggregationParam),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let (mastic, agg_param) = decoding_parameter;
        let l = agg_param.prefixes().len();
        let mut result = Vec::<VidpfWeight<T::Field>>::with_capacity(l);
        for _ in 0..l {
            result.push(VidpfWeight::<T::Field>::decode_with_param(&mastic.vidpf().weight_parameter(), bytes)?);
        }
        Ok(MasticOutputShare { result })
    }
}

impl<V: VidpfValue> Encode for MasticOutputShare<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        for elem in &self.result {
            elem.encode(bytes)?;
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut total = 0;
        for elem in &self.result {
            total += elem.encoded_len()?
        }
        Some(total)
    }
}

impl<V: VidpfValue> PartialEq for MasticOutputShare<V> {
    fn eq(&self, other: &MasticOutputShare<V>) -> bool {
        self.result == other.result
    }
}

impl<T, P, const SEED_SIZE: usize> Vdaf for Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    type Measurement = (VidpfInput, VidpfWeight<T::Field>);
    type AggregateResult = T::AggregateResult;
    type AggregationParam = MasticAggregationParam;
    type PublicShare = MasticPublicShare<VidpfWeight<T::Field>>;
    type InputShare = MasticInputShare<T::Field, SEED_SIZE>;
    type OutputShare = MasticOutputShare<VidpfWeight<T::Field>>;
    type AggregateShare = MasticOutputShare<VidpfWeight<T::Field>>;

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
        measurement_label: &VidpfInput,
        measurement_weight: &VidpfWeight<T::Field>,
        nonce: &[u8; 16],
        vidpf_keys: [VidpfKey; 2],
        szk_random: [Seed<SEED_SIZE>; 2],
        opt_random: Option<Seed<SEED_SIZE>>,
    ) -> Result<(<Self as Vdaf>::PublicShare, Vec<<Self as Vdaf>::InputShare>), VdafError> {
        // Compute the measurement shares for each aggregator by generating VIDPF
        // keys for the measurement and evaluating each of them.
        let public_share =
            self.vidpf()
                .gen_with_keys(&vidpf_keys, measurement_label, measurement_weight, nonce)?;

        let leader_measurement_share = self.vidpf().eval(
            &vidpf_keys[0],
            &public_share,
            &measurement_label.prefix(1),
            nonce,
        )?;
        let helper_measurement_share = self.vidpf().eval(
            &vidpf_keys[1],
            &public_share,
            &measurement_label.prefix(1),
            nonce,
        )?;

        let szk_proof_shares = self.szk.prove(
            leader_measurement_share.share.as_slice(),
            helper_measurement_share.share.as_slice(),
            measurement_weight.as_slice(),
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

    pub(crate) fn is_randomized(&self) -> bool {
        self.szk.has_joint_rand()
    }

    pub(crate) fn proof_len(&self) -> usize {
        self.szk.proof_len()
    }
}

impl<T, P, const SEED_SIZE: usize> Client<16> for Mastic<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn shard(
        &self,
        measurement: &(VidpfInput, VidpfWeight<T::Field>),
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError> {
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
        self.shard_with_random(
            &measurement.0,
            &measurement.1,
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
    use crate::flp::types::Sum;
    use rand::{thread_rng, Rng};

    const TEST_WEIGHT_LEN: usize = 5;
    const TEST_NONCE_SIZE: usize = 16;

    #[test]
    fn test_mastic_shard_sum() {
        let algorithm_id = 6;
        let sum_typ = Sum::<Field128>::new(5).unwrap();
        let sum_szk = Szk::new_turboshake128(sum_typ, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(TEST_WEIGHT_LEN);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8][..]);
        let first_weight = VidpfWeight::<Field128>::from(vec![
            24.into(),
            25.into(),
            26.into(),
            27.into(),
            28.into(),
        ]);

        let mastic = Mastic::new(algorithm_id, sum_szk, sum_vidpf, 16);
        let (_public, _input_shares) = mastic.shard(&(first_input, first_weight), &nonce).unwrap();
    }

    #[test]
    fn test_input_share_encode() {
        let algorithm_id = 6;
        let sum_typ = Sum::<Field128>::new(5).unwrap();
        let sum_szk = Szk::new_turboshake128(sum_typ, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(TEST_WEIGHT_LEN);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8][..]);
        let first_weight = VidpfWeight::<Field128>::from(vec![
            24.into(),
            25.into(),
            26.into(),
            27.into(),
            28.into(),
        ]);

        let mastic = Mastic::new(algorithm_id, sum_szk, sum_vidpf, 16);
        let (_public, input_shares) = mastic.shard(&(first_input, first_weight), &nonce).unwrap();
        let leader_input_share = &input_shares[0];
        let helper_input_share = &input_shares[1];

        assert_eq!(leader_input_share.encoded_len().unwrap(), leader_input_share.get_encoded().unwrap().len());
        assert_eq!(helper_input_share.encoded_len().unwrap(), helper_input_share.get_encoded().unwrap().len());
    }

    #[test]
    fn test_input_share_roundtrip() {
        let algorithm_id = 6;
        let sum_typ = Sum::<Field128>::new(5).unwrap();
        let sum_szk = Szk::new_turboshake128(sum_typ, algorithm_id);
        let sum_vidpf = Vidpf::<VidpfWeight<Field128>, TEST_NONCE_SIZE>::new(TEST_WEIGHT_LEN);

        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let first_input = VidpfInput::from_bytes(&[15u8, 0u8, 1u8][..]);
        let first_weight = VidpfWeight::<Field128>::from(vec![
            24.into(),
            25.into(),
            26.into(),
            27.into(),
            28.into(),
        ]);

        let mastic = Mastic::new(algorithm_id, sum_szk, sum_vidpf, 16);
        let (_public, input_shares) = mastic.shard(&(first_input, first_weight), &nonce).unwrap();
        let leader_input_share = &input_shares[0];
        let helper_input_share = &input_shares[1];

        let mut bytes = vec![];
        leader_input_share.encode(&mut bytes).unwrap();
        let decoded_leader_input_share = MasticInputShare::decode_with_param(&(&mastic, 0), &mut Cursor::new(&bytes)).unwrap();
        assert_eq!(leader_input_share, &decoded_leader_input_share);
        let mut bytes = vec![];
        helper_input_share.encode(&mut bytes).unwrap();
        let decoded_helper_input_share = MasticInputShare::decode_with_param(&(&mastic, 0), &mut Cursor::new(&bytes)).unwrap();
        assert_eq!(helper_input_share, &decoded_helper_input_share);


    }
}
