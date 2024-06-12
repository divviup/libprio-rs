// SPDX-License-Identifier: MPL-2.0

//! A wrapper for the [FLP](crate::flp) proof system as it is used in Mastic.
//!
//! [`Szk`] is a wrapper struct for the FLP proof system (accessible through the [`Type`] trait)
//! that is designed to accept inputs that are already secret-shared between two aggregators
//! and returns proofs that are also shared between the two aggregators. The underlying FLP
//! accepts inputs that are vectors over a finite field, and its random coins and proofs are also
//! vectors over the same field. In contrast, only the leader's SZK proof share contains any finite
//! field elements, and the initial coins are bytestrings ([`Seed`]s). The wrapper struct defined
//! here uses an [`Xof`] (to be modeled as a random oracle) to sample coins and the helper's proof share,
//! following a strategy similar to [`Prio3`](crate::vdaf::prio3::Prio3).

use crate::{
    codec::{CodecError, Encode},
    field::{FftFriendlyFieldElement, FieldElement},
    flp::{FlpError, Type},
    prng::{Prng, PrngError},
    vdaf::xof::{IntoFieldVec, Seed, Xof, XofTurboShake128},
};
use std::{borrow::Cow, marker::PhantomData};

// Domain separation tags
const DST_PROVE_RANDOMNESS: u16 = 0;
const DST_PROOF_SHARE: u16 = 1;
const DST_QUERY_RANDOMNESS: u16 = 2;
const DST_JOINT_RAND_SEED: u16 = 3;
const DST_JOINT_RAND_PART: u16 = 4;
const DST_JOINT_RANDOMNESS: u16 = 5;

const MASTIC_VERSION: u8 = 0;

/// Errors propagated by methods in this module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SzkError {
    #[error("Szk decide error: {0}")]
    /// Returned for errors in Szk verification step
    Decide(String),

    #[error("Szk query error: {0}")]
    /// Returned for errors in query evaluation
    Query(String),

    /// Returned if an FLP operation encountered an error.
    #[error("Flp error: {0}")]
    Flp(#[from] FlpError),

    /// PRNG error.
    #[error("prng error: {0}")]
    Prng(#[from] PrngError),

    /// Codec error.
    #[error("codec error: {0}")]
    Codec(#[from] CodecError),
}

/// Contains an FLP proof share, and if joint randomness is needed, the blind
/// used to derive it and the other party's joint randomness part.
#[derive(Clone)]
pub enum SzkProofShare<F: FftFriendlyFieldElement, const SEED_SIZE: usize> {
    /// Leader's proof share is uncompressed. The first Seed is a blind, second
    /// is a joint randomness part.
    Leader {
        uncompressed_proof_share: Vec<F>,
        leader_blind_and_helper_joint_rand_part: Option<(Seed<SEED_SIZE>, Seed<SEED_SIZE>)>,
    },
    /// The Helper uses one seed for both its compressed proof share and as the blind for its joint
    /// randomness.
    Helper {
        proof_share_seed_and_blind: Seed<SEED_SIZE>,
        leader_joint_rand_part: Option<Seed<SEED_SIZE>>,
    },
}

/// A tuple containing the state and messages produced by an SZK query.
#[derive(Clone)]
pub(crate) struct SzkQueryShare<F, const SEED_SIZE: usize> {
    joint_rand_part: Option<Seed<SEED_SIZE>>,
    verifier: SzkVerifier<F>,
}

/// The state that needs to be stored by an Szk verifier between query() and decide()
pub(crate) struct SzkQueryState<const SEED_SIZE: usize> {
    joint_rand_seed: Option<Seed<SEED_SIZE>>,
}

/// Verifier type for the SZK proof.
pub type SzkVerifier<F> = Vec<F>;

/// Main struct encapsulating the shared zero-knowledge functionality. The type
/// T is the underlying FLP proof system. P is the XOF used to derive all random
/// coins (it should be indifferentiable from a random oracle for security.)
pub struct Szk<T, P, const SEED_SIZE: usize>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    typ: T,
    algorithm_id: u32,
    phantom: PhantomData<P>,
}

#[cfg(test)]
impl<T: Type> Szk<T, XofTurboShake128, 16> {
    /// Create an instance of [`Szk`] using [`XofTurboShake128`].
    pub fn new_turboshake128(typ: T, algorithm_id: u32) -> Self {
        Szk::new(typ, algorithm_id)
    }
}

impl<T, P, const SEED_SIZE: usize> Szk<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// Construct an instance of this sharedZK proof system with the underlying
    /// FLP.
    pub fn new(typ: T, algorithm_id: u32) -> Self {
        Self {
            typ,
            algorithm_id,
            phantom: PhantomData,
        }
    }

    fn domain_separation_tag(&self, usage: u16) -> [u8; 8] {
        let mut dst = [0u8; 8];
        dst[0] = MASTIC_VERSION;
        dst[1] = 0; // algorithm class
        dst[2..6].copy_from_slice(&(self.algorithm_id).to_be_bytes());
        dst[6..8].copy_from_slice(&usage.to_be_bytes());
        dst
    }

    /// Derive a vector of random field elements for consumption by the FLP
    /// prover.
    fn derive_prove_rand(&self, prove_rand_seed: &Seed<SEED_SIZE>) -> Vec<T::Field> {
        P::seed_stream(
            prove_rand_seed,
            &self.domain_separation_tag(DST_PROVE_RANDOMNESS),
            &[],
        )
        .into_field_vec(self.typ.prove_rand_len())
    }

    fn derive_joint_rand_part(
        &self,
        aggregator_blind: &Seed<SEED_SIZE>,
        measurement_share: &[T::Field],
        nonce: &[u8; 16],
    ) -> Result<Seed<SEED_SIZE>, SzkError> {
        let mut xof = P::init(
            aggregator_blind.as_ref(),
            &self.domain_separation_tag(DST_JOINT_RAND_PART),
        );
        xof.update(nonce);
        // Encode measurement_share (currently an array of field elements) into
        // bytes and include it in the XOF input.
        let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
        for field_elem in measurement_share {
            field_elem.encode(&mut encoding_buffer)?;
            xof.update(&encoding_buffer);
            encoding_buffer.clear();
        }
        Ok(xof.into_seed())
    }

    fn derive_joint_rand_seed(
        &self,
        leader_joint_rand_part: &Seed<SEED_SIZE>,
        helper_joint_rand_part: &Seed<SEED_SIZE>,
    ) -> Seed<SEED_SIZE> {
        let mut xof = P::init(
            &[0; SEED_SIZE],
            &self.domain_separation_tag(DST_JOINT_RAND_SEED),
        );
        xof.update(&leader_joint_rand_part.0);
        xof.update(&helper_joint_rand_part.0);
        xof.into_seed()
    }

    fn derive_joint_rand_and_seed(
        &self,
        leader_joint_rand_part: &Seed<SEED_SIZE>,
        helper_joint_rand_part: &Seed<SEED_SIZE>,
    ) -> (Seed<SEED_SIZE>, Vec<T::Field>) {
        let joint_rand_seed =
            self.derive_joint_rand_seed(leader_joint_rand_part, helper_joint_rand_part);
        let joint_rand = P::seed_stream(
            &joint_rand_seed,
            &self.domain_separation_tag(DST_JOINT_RANDOMNESS),
            &[],
        )
        .into_field_vec(self.typ.joint_rand_len());

        (joint_rand_seed, joint_rand)
    }

    fn derive_helper_proof_share(&self, proof_share_seed: &Seed<SEED_SIZE>) -> Vec<T::Field> {
        Prng::from_seed_stream(P::seed_stream(
            proof_share_seed,
            &self.domain_separation_tag(DST_PROOF_SHARE),
            &[],
        ))
        .take(self.typ.proof_len())
        .collect()
    }

    fn derive_query_rand(&self, verify_key: &[u8; SEED_SIZE], nonce: &[u8; 16]) -> Vec<T::Field> {
        let mut xof = P::init(
            verify_key,
            &self.domain_separation_tag(DST_QUERY_RANDOMNESS),
        );
        xof.update(nonce);
        xof.into_seed_stream()
            .into_field_vec(self.typ.query_rand_len())
    }

    pub(crate) fn has_joint_rand(&self) -> bool {
        self.typ.joint_rand_len() > 0
    }

    fn prove(
        &self,
        leader_input_share: &[T::Field],
        helper_input_share: &[T::Field],
        encoded_measurement: &[T::Field],
        rand_seeds: [Seed<SEED_SIZE>; 2],
        leader_seed_opt: Option<Seed<SEED_SIZE>>,
        nonce: &[u8; 16],
    ) -> Result<[SzkProofShare<T::Field, SEED_SIZE>; 2], SzkError> {
        let [prove_rand_seed, helper_seed] = rand_seeds;
        // If joint randomness is used, derive it from the two input shares,
        // the seeds used to blind the derivation, and the nonce. Pass the
        // leader its blinding seed and the helper's joint randomness part, and
        // pass the helper the leader's joint randomness part. (The seed used to
        // derive the helper's proof share is reused as the helper's blind.)
        let (leader_blind_and_helper_joint_rand_part, leader_joint_rand_part, joint_rand) =
            if let Some(leader_seed) = leader_seed_opt.clone() {
                let leader_joint_rand_part =
                    self.derive_joint_rand_part(&leader_seed, leader_input_share, nonce)?;
                let helper_joint_rand_part =
                    self.derive_joint_rand_part(&helper_seed, helper_input_share, nonce)?;
                let (_joint_rand_seed, joint_rand) = self
                    .derive_joint_rand_and_seed(&leader_joint_rand_part, &helper_joint_rand_part);
                (
                    Some((leader_seed, helper_joint_rand_part)),
                    Some(leader_joint_rand_part),
                    joint_rand,
                )
            } else {
                (None, None, Vec::new())
            };

        let prove_rand = self.derive_prove_rand(&prove_rand_seed);
        let mut leader_proof_share =
            self.typ
                .prove(encoded_measurement, &prove_rand, &joint_rand)?;

        // Generate the proof shares.
        for (x, y) in leader_proof_share
            .iter_mut()
            .zip(self.derive_helper_proof_share(&helper_seed))
        {
            *x -= y;
        }

        // Construct the output messages.
        let leader_proof_share = SzkProofShare::Leader {
            uncompressed_proof_share: leader_proof_share,
            leader_blind_and_helper_joint_rand_part,
        };
        let helper_proof_share = SzkProofShare::Helper {
            proof_share_seed_and_blind: helper_seed.clone(),
            leader_joint_rand_part,
        };
        Ok([leader_proof_share, helper_proof_share])
    }

    fn query(
        &self,
        input_share: &[T::Field],
        proof_share: SzkProofShare<T::Field, SEED_SIZE>,
        verify_key: &[u8; SEED_SIZE],
        nonce: &[u8; 16],
    ) -> Result<(SzkQueryShare<T::Field, SEED_SIZE>, SzkQueryState<SEED_SIZE>), SzkError> {
        let query_rand = self.derive_query_rand(verify_key, nonce);
        let flp_proof_share = match proof_share {
            SzkProofShare::Leader {
                ref uncompressed_proof_share,
                ..
            } => Cow::Borrowed(uncompressed_proof_share),
            SzkProofShare::Helper {
                ref proof_share_seed_and_blind,
                ..
            } => Cow::Owned(self.derive_helper_proof_share(proof_share_seed_and_blind)),
        };

        let (joint_rand, joint_rand_seed, joint_rand_part) = if self.has_joint_rand() {
            let ((joint_rand_seed, joint_rand), host_joint_rand_part) = match proof_share {
                SzkProofShare::Leader {
                    uncompressed_proof_share: _,
                    leader_blind_and_helper_joint_rand_part,
                } => match leader_blind_and_helper_joint_rand_part {
                    Some((seed, helper_joint_rand_part)) => {
                        match self.derive_joint_rand_part(&seed, input_share, nonce) {
                            Ok(leader_joint_rand_part) => (
                                self.derive_joint_rand_and_seed(
                                    &leader_joint_rand_part,
                                    &helper_joint_rand_part,
                                ),
                                leader_joint_rand_part,
                            ),
                            Err(e) => return Err(e),
                        }
                    }
                    None => {
                        return Err(SzkError::Query(
                            "leader_blind_and_helper_joint_rand_part should be set".to_string(),
                        ))
                    }
                },
                SzkProofShare::Helper {
                    proof_share_seed_and_blind,
                    leader_joint_rand_part,
                } => match leader_joint_rand_part {
                    Some(leader_joint_rand_part) => match self.derive_joint_rand_part(
                        &proof_share_seed_and_blind,
                        input_share,
                        nonce,
                    ) {
                        Ok(helper_joint_rand_part) => (
                            self.derive_joint_rand_and_seed(
                                &leader_joint_rand_part,
                                &helper_joint_rand_part,
                            ),
                            helper_joint_rand_part,
                        ),
                        Err(e) => return Err(e),
                    },
                    None => {
                        return Err(SzkError::Query(
                            "leader_joint_rand_part should be set".to_string(),
                        ))
                    }
                },
            };
            (
                joint_rand,
                Some(joint_rand_seed),
                Some(host_joint_rand_part),
            )
        } else {
            (Vec::new(), None, None)
        };
        let verifier_share = self.typ.query(
            input_share,
            flp_proof_share.as_ref(),
            &query_rand,
            &joint_rand,
            2,
        )?;
        Ok((
            SzkQueryShare {
                joint_rand_part,
                verifier: verifier_share,
            },
            SzkQueryState { joint_rand_seed },
        ))
    }

    /// Returns true if the verifier message indicates that the input from which
    /// it was generated is valid.
    fn decide(
        &self,
        verifier: &[T::Field],
        leader_joint_rand_part_opt: Option<Seed<SEED_SIZE>>,
        helper_joint_rand_part_opt: Option<Seed<SEED_SIZE>>,
        joint_rand_seed_opt: Option<Seed<SEED_SIZE>>,
    ) -> Result<bool, SzkError> {
        // Check if underlying FLP proof validates
        let check_flp_proof = self.typ.decide(verifier)?;
        if !check_flp_proof {
            return Ok(false);
        }
        // Check that joint randomness was properly derived from both
        // aggregators' parts
        match (
            joint_rand_seed_opt,
            leader_joint_rand_part_opt,
            helper_joint_rand_part_opt,
        ) {
            (Some(joint_rand_seed), Some(leader_joint_rand_part), Some(helper_joint_rand_part)) => {
                let expected_joint_rand_seed =
                    self.derive_joint_rand_seed(&leader_joint_rand_part, &helper_joint_rand_part);
                Ok(joint_rand_seed == expected_joint_rand_seed)
            }
            (None, None, None) => Ok(true),
            (_, _, _) => Err(SzkError::Decide(
                "at least one of the input seeds is missing".to_string(),
            )),
        }
    }
}

mod tests {
    use super::*;
    use crate::field::Field128 as TestField;
    use crate::field::{random_vector, FieldElementWithInteger};
    use crate::flp::types::{Count, Sum};
    use crate::flp::Type;
    use rand::{thread_rng, Rng};

    fn generic_szk_test<T: Type>(typ: T, encoded_measurement: &[T::Field], valid: bool) {
        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 16];
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(typ.clone(), algorithm_id);
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = if szk_typ.has_joint_rand() {
            Some(Seed::<16>::generate().unwrap())
        } else {
            None
        };
        let helper_input_share: Vec<T::Field> = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let proof_shares = szk_typ.prove(
            &leader_input_share,
            &helper_input_share,
            encoded_measurement,
            [prove_rand_seed, helper_seed],
            leader_seed_opt,
            &nonce,
        );

        let [l_proof_share, h_proof_share] = proof_shares.unwrap();
        let (l_query_share, l_query_state) = szk_typ
            .query(
                &leader_input_share,
                l_proof_share.clone(),
                &verify_key,
                &nonce,
            )
            .unwrap();
        let (h_query_share, h_query_state) = szk_typ
            .query(&helper_input_share, h_proof_share, &verify_key, &nonce)
            .unwrap();

        let mut verifier = l_query_share.clone().verifier;

        for (x, y) in verifier.iter_mut().zip(h_query_share.clone().verifier) {
            *x += y;
        }
        let h_jr_part = h_query_share.clone().joint_rand_part;
        let h_jr_seed = h_query_state.joint_rand_seed;
        let l_jr_part = l_query_share.joint_rand_part;
        let l_jr_seed = l_query_state.joint_rand_seed;
        if let Ok(leader_decision) = szk_typ.decide(
            &verifier,
            l_jr_part.clone(),
            h_jr_part.clone(),
            l_jr_seed.clone(),
        ) {
            assert_eq!(
                leader_decision, valid,
                "Leader incorrectly determined validity",
            );
        } else {
            panic!("Leader failed during decision");
        };
        if let Ok(helper_decision) = szk_typ.decide(
            &verifier,
            l_jr_part.clone(),
            h_jr_part.clone(),
            h_jr_seed.clone(),
        ) {
            assert_eq!(
                helper_decision, valid,
                "Helper incorrectly determined validity",
            );
        } else {
            panic!("Helper failed during decision");
        };

        //test mutated jr seed
        if szk_typ.has_joint_rand() {
            let joint_rand_seed_opt = Some(Seed::<16>::generate().unwrap());
            if let Ok(leader_decision) = szk_typ.decide(
                &verifier,
                l_jr_part.clone(),
                h_jr_part.clone(),
                joint_rand_seed_opt,
            ) {
                assert!(!leader_decision, "Leader accepted wrong jr seed");
            };
        };

        //test mutated verifier
        let mut verifier = l_query_share.verifier;

        for (x, y) in verifier.iter_mut().zip(h_query_share.clone().verifier) {
            *x += y + T::Field::from(
                <T::Field as FieldElementWithInteger>::Integer::try_from(7).unwrap(),
            );
        }

        let leader_decision = szk_typ
            .decide(
                &verifier,
                l_jr_part.clone(),
                h_jr_part.clone(),
                l_jr_seed.clone(),
            )
            .unwrap();
        assert!(!leader_decision, "Leader validated after proof mutation");

        // test mutated input share
        let mut mutated_input = leader_input_share.clone();
        mutated_input[0] *=
            T::Field::from(<T::Field as FieldElementWithInteger>::Integer::try_from(23).unwrap());
        let (mutated_query_share, mutated_query_state) = szk_typ
            .query(&mutated_input, l_proof_share.clone(), &verify_key, &nonce)
            .unwrap();
        let mut verifier = mutated_query_share.verifier;

        for (x, y) in verifier.iter_mut().zip(h_query_share.clone().verifier) {
            *x += y;
        }

        let mutated_jr_seed = mutated_query_state.joint_rand_seed;
        let mutated_jr_part = mutated_query_share.joint_rand_part;
        if let Ok(leader_decision) = szk_typ.decide(
            &verifier,
            mutated_jr_part.clone(),
            h_jr_part.clone(),
            mutated_jr_seed,
        ) {
            assert!(!leader_decision, "Leader validated after input mutation");
        };

        // test mutated proof share
        let (mut mutated_proof, leader_blind_and_helper_joint_rand_part) = match l_proof_share {
            SzkProofShare::Leader {
                uncompressed_proof_share,
                leader_blind_and_helper_joint_rand_part,
            } => (
                uncompressed_proof_share.clone(),
                leader_blind_and_helper_joint_rand_part,
            ),
            _ => (vec![], None),
        };
        mutated_proof[0] *=
            T::Field::from(<T::Field as FieldElementWithInteger>::Integer::try_from(23).unwrap());
        let mutated_proof_share = SzkProofShare::Leader {
            uncompressed_proof_share: mutated_proof,
            leader_blind_and_helper_joint_rand_part,
        };
        let (l_query_share, l_query_state) = szk_typ
            .query(
                &leader_input_share,
                mutated_proof_share,
                &verify_key,
                &nonce,
            )
            .unwrap();
        let mut verifier = l_query_share.verifier;

        for (x, y) in verifier.iter_mut().zip(h_query_share.clone().verifier) {
            *x += y;
        }

        let mutated_jr_seed = l_query_state.joint_rand_seed;
        let mutated_jr_part = l_query_share.joint_rand_part;
        if let Ok(leader_decision) = szk_typ.decide(
            &verifier,
            mutated_jr_part.clone(),
            h_jr_part.clone(),
            mutated_jr_seed,
        ) {
            assert!(!leader_decision, "Leader validated after proof mutation");
        };
    }

    #[test]
    fn test_sum() {
        let sum = Sum::<TestField>::new(5).unwrap();

        let five = TestField::from(5);
        let nine = sum.encode_measurement(&9).unwrap();
        let bad_encoding = &vec![five; sum.input_len()];
        generic_szk_test(sum.clone(), &nine, true);
        generic_szk_test(sum, bad_encoding, false);
    }

    #[test]
    fn test_count() {
        let count = Count::<TestField>::new();
        let encoded_true = count.encode_measurement(&true).unwrap();
        generic_szk_test(count, &encoded_true, true);
    }
}
