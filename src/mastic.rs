use crate::{

    codec::{CodecError, Decode, Encode},
    field::FieldElement,
    flp::{
        szk::{Szk, SzkProofShare}, Type
    },
    vdaf::{
        poplar1::Poplar1AggregationParam, xof::{Seed, Xof}, Aggregatable, Aggregator, Client, Collector, PrepareTransition, Vdaf, VdafError
    }, vidpf::{
        Vidpf,
        VidpfInput,
        VidpfKey,
        VidpfPublicShare,
        VidpfServerId,
        VidpfValue
    }
};
use std::{
    fmt::Debug,
    io::Cursor,
    marker::PhantomData,
};
/// The MASTIC VDAF.
#[derive(Clone, Debug)]
pub struct Mastic<T, P, V,  const SEED_SIZE: usize>
where
    T: Type<Measurement = V>,
    V: VidpfValue,
    P: Xof<SEED_SIZE>,
{
    algorithm_id: u32,
    szk: Szk<T, P>,
    vpf: V,
    bits: usize,
    phantom: PhantomData<P>,
}

impl<T, P, V, const SEED_SIZE: usize> Mastic<T, P, V, SEED_SIZE>
where
    T: Type<Measurement = V>,
    V: VidpfValue,
    P: Xof<SEED_SIZE>,
{
pub fn new(
    algorithm_id: u32,
    szk: Szk<T, P, SEED_SIZE>,
    vpf: Vidpf<V, SEED_SIZE>,
    bits: usize) -> Self {
    Self {
        algorithm_id,
        szk,
        vpf,
        bits,
        phantom: PhantomData,
    }
}
}
/// Mastic aggregation parameter.
///
/// This includes the VIDPF tree level under evaluation, a set of prefixes to evaluate at that level,
/// and, optionally, the aggregate results of prior levels.
pub type MasticAggregationParam = Poplar1AggregationParam;

pub type MasticPublicShare<V> = VidpfPublicShare<V>;

impl<V: VidpfValue> Decode for MasticPublicShare<V> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        todo!();
    }
}

impl<V: VidpfValue, const SEED_SIZE: usize> Encode for MasticPublicShare<V, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        todo!();
    }

    fn encoded_len(&self) -> Option<usize> {
        todo!();
    }
}


/// Add necessary traits for MasticPublicShare here

/// Message sent by the [`Client`] to each [`Aggregator`] during the Sharding phase.
#[derive(Clone, Debug)]
pub struct MasticInputShare<W: VidpfValue, const SEED_SIZE: usize> {
    /// VIDPF key share.
    vidpf_key: VidpfKey,

    /// The proof share.
    proofs_share: SzkProofShare<W, SEED_SIZE>,
}
impl<V: VidpfValue, const SEED_SIZE: usize> Decode for MasticInputShare<V, SEED_SIZE> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let vidpf_key = VidpfKey::decode(bytes)?;
        let proofs_share = SzkProofShare::<V, SEED_SIZE>::decode(bytes)?;
        Ok(Self { vidpf_key, proofs_share })
    }
}

impl<V: VidpfValue, const SEED_SIZE: usize> Encode for MasticInputShare<V, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.vidpf_key.encode(bytes)?;
        self.proofs_share.encode(bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        self.vidpf_key.encoded_len()? + self.proofs_share.encoded_len()?
    }
}

#[derive(Clone, Debug)]
pub struct MasticOutputShare<W: VidpfValue> {
    result: Vec<W>,
}
impl <V: VidpfValue + Debug> Aggregatable for MasticOutputShare<V> {
    type OutputShare = MasticOutputShare<V>;
    /// Update an aggregate share by merging it with another (`agg_share`).
    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError>{
        todo!();
    }

    /// Update an aggregate share by adding `output_share`.
    fn accumulate(&mut self, output_share: &Self::OutputShare) -> Result<(), VdafError>{
        todo!();
    }
}

impl<V: VidpfValue> Decode for MasticOutputShare<V> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let v = Vec::V::decode(bytes)?;
        Ok(Self { result: v })
    }
}

impl<V: VidpfValue> Encode for MasticOutputShare<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.result.encode(bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        self.result.encoded_len()
    }
}

impl<T, P, V, const SEED_SIZE: usize> Vdaf for Mastic<T, P, V, SEED_SIZE>
where
    T: Type<Measurement = V>,
    V: VidpfValue + Debug,
    P: Xof<SEED_SIZE>,
{
    type Measurement = T::Measurement;
    type AggregateResult = T::AggregateResult;
    type AggregationParam = MasticAggregationParam;
    type PublicShare = MasticPublicShare<V>;
    type InputShare = MasticInputShare<V, SEED_SIZE>;
    type OutputShare = MasticOutputShare<V>;
    type AggregateShare = MasticOutputShare<V>;

    fn algorithm_id(&self) -> u32 {
        self.algorithm_id
    }

    fn num_aggregators(&self) -> usize {
        2
    }
}

impl<T, P, V, const SEED_SIZE: usize> Mastic<T, P, V, SEED_SIZE>
where
    T: Type<Measurement = V>,
    P: Xof<SEED_SIZE>,
    V: VidpfValue + Debug {

    fn shard_with_random(
        &self,
        measurement_label: &VidpfInput,
        measurement_weight: V,
        nonce: &[u8; 16],
        vidpf_keys: &[VidpfKey; 2],
        szk_random: &Vec<Seed<SEED_SIZE>>,
    ) -> Result<(MasticPublicShare<V>, Vec<MasticInputShare<V, SEED_SIZE>>), VdafError> {

        if measurement_label.len() != self.bits {
            return Err(VdafError::Uncategorized(format!(
                "unexpected input length ({})",
                measurement_label.len()
            )));
        }
    // Compute the measurement shares for each aggregator by generating VIDPF
    // keys for the measurement and evaluating each of them.
    let (public_share, keys) = self.vpf.gen_with_keys(vidpf_keys, measurement_label, measurement_weight, nonce);
    let leader_measurement_share = self.vpf.eval(keys[0], public, measurement_label.prefix(1), nonce);
    let helper_measurement_share = self.vpf.eval(keys[1], public, measurement_label.prefix(1), nonce);
    match (self.szk.has_joint_rand(), szk_random.len()){
        (true, 3) => (),
        (false, 2) => (),
        (_, _) => return Err(VdafError::Uncategorized(format!(
            "incorrect Szk coins length ({})",
            szk_random.len(),
        )))
    }
    // Compute the Szk proof shares for each aggregator
    let leader_seed_opt = if self.szk.has_joint_rand() {
        Some(szk_random[2])
    } else {
        None
    };
    let szk_proof_shares = self.szk.prove(
        &self,
        leader_measurement_share,
        helper_measurement_share,
        szk_random[0..2],
        leader_seed_opt,
        nonce,
    )?;
    let leader_share = MasticInputShare::<V, SEED_SIZE> {
        vidpf_key: keys[0],
        proofs_share: szk_proof_shares[0],
    };
    let helper_share = MasticInputShare::<V, SEED_SIZE> {
        vidpf_key: keys[1],
        proofs_share: szk_proof_shares[1],
    };
    Ok((public_share, vec![leader_share, helper_share]))
    }
}

impl<T, P, V,  const SEED_SIZE: usize> Client<16> for Mastic<T, P, V, SEED_SIZE>
where
    T: Type<Measurement = V>,
    P: Xof<SEED_SIZE>,
    V: VidpfValue + Debug,
    {

    fn shard(
        &self,
        measurement: (&VidpfInput, &V),
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError>{
        let vidpf_keys = [
            VidpfKey::gen(VidpfServerId::S0)?,
            VidpfKey::gen(VidpfServerId::S1)?,
        ];
        self.shard_with_random(measurement.0,
            measurement.1,
            nonce,
            &vidpf_keys,
            )
    }

}