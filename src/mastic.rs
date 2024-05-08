use crate::{

    field::{FieldElement},
    idpf::{Idpf, IdpfInput, IdpfOutputShare, IdpfPublicShare, IdpfValue, RingBufferCache},
    prng::Prng,
    flp::Type,
    szk::{Szk, SzkProofShare},
    vidpf::{Vidpf, VidpfInput, VidpfValue},
    vdaf::{
        xof::{Seed, Xof, XofTurboShake128},
        Aggregatable, Aggregator, Client, Collector, PrepareTransition, Vdaf, VdafError,
    },
};

/// The MASTIC VDAF.
#[derive(Clone, Debug)]
pub struct Mastic<T, P, V,  const SEED_SIZE: usize>
where
    T: Type,
    T::Measurement: VidpfValue,
    P: Xof<SEED_SIZE>,
    V: Vidpf<T::Measurement, 16>
{
    algorithm_id: u32,
    szk: Szk<T, P>,
    vpf: V,
    bits: usize,
    phantom: PhantomData<P>,
}

impl<T, P, V, const SEED_SIZE:usize> Mastic<T, P, V, SEED_SIZE>
where
    T: Type,
    T::Measurement: VidpfValue,
    P: Xof<SEED_SIZE>,
    V: Vidpf<T::Measurement, 16>
{
pub fn new(
    algorithm_id: u32,
    szk: S,
    vpf: V,
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
#[derive(Clone, Debug)]
pub struct MasticAggregationParam<V: VidpfValue>  {
    level: u16,
    prefixes: Vec<VipfInput>,
    counts: Vec<Vec<V>>
}

/// Add necessary traits for MasticAggregationParam here.

pub struct MasticPublicShare<W: VidpfValue> {
    joint_rand_parts: Option<Vec<Seed<SEED_SIZE>>>,
    vidpf_public_share: VidpfPublicShare<W>,
}

/// Add necessary traits for MasticPublicShare here



/// Message sent by the [`Client`] to each [`Aggregator`] during the Sharding phase.
#[derive(Debug, Clone)]
pub struct MasticInputShare<W: VidpfValue, const SEED_SIZE: usize> {
    /// VIDPF key share.
    vidpf_key: VidpfKey,

    /// The proof share.
    proofs_share: SzkProofShare<W, SEED_SIZE>,

    /// Blinding seed used by the Aggregator to compute the joint randomness. This field is optional
    /// because not every [`Type`] requires joint randomness.
    joint_rand_blind: Option<Seed<SEED_SIZE>>,
}

pub struct MasticOutputShare<W: VidpfValue> {
    result: Vec<W>,
}


impl<S, P, V, const SEED_SIZE: usize> Vdaf for Mastic<S, P, V, SEED_SIZE>
where
    S: Szk,
    S::Type::Measurement: VidpfValue,
    P: Xof<SEED_SIZE>,
    V: Vidpf<T::Measurement, 16>
{
    type Measurement = S::Type::Measurement;
    type AggregateResult = S::Type::AggregateResult;
    type AggregationParam = MasticAggregationParam;
    type PublicShare = MasticPublicShare<SEED_SIZE>;
    type InputShare = MasticInputShare<S::Type::Field, SEED_SIZE>;
    type OutputShare = MasticOutputShare<S::Type::Measurement>;
    type AggregateShare = MasticOutputShare<S::Type::Measurement>;

    fn algorithm_id(&self) -> u32 {
        self.algorithm_id
    }

    fn num_aggregators(&self) -> usize {
        2
    }
}

impl<T, P, V, const SEED_SIZE: usize> Mastic<S::Type, P, V, SEED_SIZE>
where
    S: Szk,
    P: Xof<SEED_SIZE>,
    V: Vidpf<S::Type::Measurement, 16> {

    fn shard_with_random(
        &self,
        measurement_label: &VidpfInput,
        measurement_weight: &VidpfValue,
        nonce: &[u8; 16],
        vidpf_random: &[[u8; 16]; 2],
        szk_random: &[[u8; SEED_SIZE]],
    ) -> Result<(MasticPublicShare, Vec<MasticInputShare<SEED_SIZE>>), VdafError> {

        if input.len() != self.bits {
            return Err(VdafError::Uncategorized(format!(
                "unexpected input length ({})",
                input.len()
            )));
        }
    // Compute the measurement shares for each aggregator by generating VIDPF
    // keys for the measurement and evaluating each of them.
    let (public, keys) = self.vpf.gen(measurement_label, measurement_weight, nonce);
    let leader_measurement_share = self.vpf.eval(keys[0], public, input, nonce);
    let helper_measurement_share = self.vpf.eval(keys[1], public, input, nonce);
    let encoded_measurement = leader_measurement_share.clone();
    for (x, y) in encoded_measurement
            .iter_mut()
            .zip(helper_measurement_share)
        {
            *x -= y;
        }
    match (self.szk.has_joint_rand(), szk_random.len()){
        (true, 3) => (),
        (false, 2) => (),
        (_, _) => return Err(VdafError::Uncategorized(format!(
            "incorrect Szk coins length ({})",
            szk_random.len();,
        )))
    }
    // Compute the Szk proof shares for each aggregator
    let szk_coins = [Seed::SEED_SIZE::from_bytes(szk_random[0]), Seed::SEED_SIZE::from_bytes(szk_random[1])];
    let leader_seed_opt = if self.szk.has_joint_rand() {
        Some(Seed::SEED_SIZE::from_bytes(szk_random[2]))
    } else {
        None
    };
    let szk_proof_shares = prove(
        &self,
        leader_measurement_share,
        helper_measurement_share,
        encoded_measurement,
        szk_coins,
        leader_seed_opt,
        nonce,
    );
    // Compute the joint randomness.
    let mut helper_joint_rand_parts = if self.typ.joint_rand_len() > 0 {
        Some(Vec::with_capacity(num_aggregators as usize - 1))
    } else {
        None
    };
    let proof_share_seed = random_seeds.next().unwrap().try_into().unwrap();
    let joint_rand_blind = if let Some(helper_joint_rand_parts) =
        helper_joint_rand_parts.as_mut() {
            let joint_rand_blind = random_seeds.next().unwrap().try_into().unwrap();
            let mut joint_rand_part_xof = P::init(
                &joint_rand_blind,
                &self.domain_separation_tag(DST_JOINT_RAND_PART),
            );
            joint_rand_part_xof.update(&[agg_id]); // Aggregator ID
            joint_rand_part_xof.update(nonce);

            let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
            for (x, y) in leader_measurement_share
                .iter_mut()
                .zip(measurement_share_prng)
            {
                *x -= y;
                y.encode(&mut encoding_buffer).map_err(|_| {
                    VdafError::Uncategorized("failed to encode measurement share".to_string())
                })?;
                joint_rand_part_xof.update(&encoding_buffer);
                encoding_buffer.clear();
            }

            helper_joint_rand_parts.push(joint_rand_part_xof.into_seed());

            Some(joint_rand_blind)
        } else {
            for (x, y) in leader_measurement_share
                .iter_mut()
                .zip(measurement_share_prng)
            {
                *x -= y;
            }
            None
        };
    let helper =
            HelperShare::from_seeds(measurement_share_seed, proof_share_seed, joint_rand_blind);
    helper_shares.push(helper);

    let mut leader_blind_opt = None;
    let public_share = Prio3PublicShare {
        joint_rand_parts: helper_joint_rand_parts
            .as_ref()
            .map(
                |helper_joint_rand_parts| -> Result<Vec<Seed<SEED_SIZE>>, VdafError> {
                    let leader_blind_bytes = random_seeds.next().unwrap().try_into().unwrap();
                    let leader_blind = Seed::from_bytes(leader_blind_bytes);

                    let mut joint_rand_part_xof = P::init(
                        leader_blind.as_ref(),
                        &self.domain_separation_tag(DST_JOINT_RAND_PART),
                    );
                    joint_rand_part_xof.update(&[0]); // Aggregator ID
                    joint_rand_part_xof.update(nonce);
                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in leader_measurement_share.iter() {
                        x.encode(&mut encoding_buffer).map_err(|_| {
                            VdafError::Uncategorized(
                                "failed to encode measurement share".to_string(),
                            )
                        })?;
                        joint_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }
                    leader_blind_opt = Some(leader_blind);

                    let leader_joint_rand_seed_part = joint_rand_part_xof.into_seed();

                    let mut vec = Vec::with_capacity(self.num_aggregators());
                    vec.push(leader_joint_rand_seed_part);
                    vec.extend(helper_joint_rand_parts.iter().cloned());
                    Ok(vec)
                },
            )
            .transpose()?,
    };

    // Compute the joint randomness.
    let joint_rands = public_share
        .joint_rand_parts
        .as_ref()
        .map(|joint_rand_parts| self.derive_joint_rands(joint_rand_parts.iter()).1)
        .unwrap_or_default();

    // Generate the proofs.
    let prove_rands = self.derive_prove_rands(&Seed::from_bytes(
        random_seeds.next().unwrap().try_into().unwrap(),
    ));
    let mut leader_proofs_share = Vec::with_capacity(self.typ.proof_len() * self.num_proofs());
    for p in 0..self.num_proofs() {
        let prove_rand =
            &prove_rands[p * self.typ.prove_rand_len()..(p + 1) * self.typ.prove_rand_len()];
        let joint_rand =
            &joint_rands[p * self.typ.joint_rand_len()..(p + 1) * self.typ.joint_rand_len()];

        leader_proofs_share.append(&mut self.typ.prove(
            &encoded_measurement,
            prove_rand,
            joint_rand,
        )?);
    }

    // Generate the proof shares and distribute the joint randomness seed hints.
    for (j, helper) in helper_shares.iter_mut().enumerate() {
        for (x, y) in
            leader_proofs_share
                .iter_mut()
                .zip(self.derive_helper_proofs_share(
                    &helper.proofs_share,
                    u8::try_from(j).unwrap() + 1,
                ))
                .take(self.typ.proof_len() * self.num_proofs())
        {
            *x -= y;
        }
    }

    // Prep the output messages.
    let mut out = Vec::with_capacity(num_aggregators as usize);
    out.push(Prio3InputShare {
        measurement_share: Share::Leader(leader_measurement_share),
        proofs_share: Share::Leader(leader_proofs_share),
        joint_rand_blind: leader_blind_opt,
    });

    for helper in helper_shares.into_iter() {
        out.push(Prio3InputShare {
            measurement_share: Share::Helper(helper.measurement_share),
            proofs_share: Share::Helper(helper.proofs_share),
            joint_rand_blind: helper.joint_rand_blind,
        });
    }

    Ok((public_share, out))

}


impl<T, P, V,  const SEED_SIZE: usize> Client<16>
for Mastic<T, P, V, SEED_SIZE>
where
    T: Type,
    T::Measurement: VidpfValue,
    P: Xof<SEED_SIZE>,
    V: Vidpf<T::Measurement, 16>{
    fn shard(
        &self,
        measurement: &Self::Measurement,
        nonce: &[u8; NONCE_SIZE],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError>{
        let mut random = vec![0u8; self.random_size()];
        getrandom::getrandom(&mut random)?;
        self.shard_with_random(measurement, nonce, &random)
    }

}