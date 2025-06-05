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
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{add_assign_vector, decode_fieldvec, encode_fieldvec, sub_assign_vector, FieldElement},
    flp::{FlpError, Type},
    vdaf::{
        mastic::{self, NONCE_SIZE, SEED_SIZE, USAGE_PROOF_SHARE},
        xof::{IntoFieldVec, Seed, Xof, XofTurboShake128},
    },
};
use std::borrow::Cow;
use std::io::Cursor;
use std::ops::BitAnd;
use subtle::{Choice, ConstantTimeEq};

// Domain separation tags

/// Errors propagated by methods in this module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SzkError {
    /// Returned for errors in Szk verification step
    #[error("Szk decide error: {0}")]
    Decide(String),

    /// Returned for errors in query evaluation
    #[error("Szk query error: {0}")]
    Query(String),

    /// Returned if an FLP operation encountered an error.
    #[error("Flp error: {0}")]
    Flp(#[from] FlpError),

    /// Codec error.
    #[error("codec error: {0}")]
    Codec(#[from] CodecError),
}

/// Contains an FLP proof share, and if joint randomness is needed, the blind
/// used to derive it and the other party's joint randomness part.
#[derive(Debug, Clone)]
pub enum SzkProofShare<F> {
    /// Leader's proof share is uncompressed.
    Leader {
        /// Share of an FLP proof, as a vector of Field elements.
        uncompressed_proof_share: Vec<F>,
        /// Set only if joint randomness is needed. The first Seed is a blind, second
        /// is the helper's joint randomness part.
        leader_blind_and_helper_joint_rand_part_opt: Option<(Seed<32>, Seed<32>)>,
    },
    /// The Helper uses one seed for both its compressed proof share and as the blind for its joint
    /// randomness.
    Helper {
        /// Used to derive the helper's input and proof shares and its blind for FLP joint
        /// randomness computation.
        proof_share_seed_and_blind: Seed<SEED_SIZE>,
        /// Set only if joint randomness is needed for the FLP.
        leader_joint_rand_part_opt: Option<Seed<SEED_SIZE>>,
    },
}

impl<F: FieldElement> PartialEq for SzkProofShare<F> {
    fn eq(&self, other: &SzkProofShare<F>) -> bool {
        bool::from(self.ct_eq(other))
    }
}

impl<F: FieldElement> ConstantTimeEq for SzkProofShare<F> {
    fn ct_eq(&self, other: &SzkProofShare<F>) -> Choice {
        match (self, other) {
            (
                SzkProofShare::Leader {
                    uncompressed_proof_share: s_proof,
                    leader_blind_and_helper_joint_rand_part_opt: s_blind,
                },
                SzkProofShare::Leader {
                    uncompressed_proof_share: o_proof,
                    leader_blind_and_helper_joint_rand_part_opt: o_blind,
                },
            ) => s_proof[..]
                .ct_eq(&o_proof[..])
                .bitand(option_tuple_ct_eq(s_blind, o_blind)),
            (
                SzkProofShare::Helper {
                    proof_share_seed_and_blind: s_seed,
                    leader_joint_rand_part_opt: s_rand,
                },
                SzkProofShare::Helper {
                    proof_share_seed_and_blind: o_seed,
                    leader_joint_rand_part_opt: o_rand,
                },
            ) => s_seed.ct_eq(o_seed).bitand(option_ct_eq(s_rand, o_rand)),
            _ => Choice::from(0),
        }
    }
}

impl<F: FieldElement> Encode for SzkProofShare<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match self {
            SzkProofShare::Leader {
                uncompressed_proof_share,
                leader_blind_and_helper_joint_rand_part_opt,
            } => (
                encode_fieldvec(uncompressed_proof_share, bytes)?,
                if let Some((blind, helper_joint_rand_part)) =
                    leader_blind_and_helper_joint_rand_part_opt
                {
                    blind.encode(bytes)?;
                    helper_joint_rand_part.encode(bytes)?;
                },
            ),
            SzkProofShare::Helper {
                proof_share_seed_and_blind,
                leader_joint_rand_part_opt,
            } => (
                proof_share_seed_and_blind.encode(bytes)?,
                if let Some(leader_joint_rand_part) = leader_joint_rand_part_opt {
                    leader_joint_rand_part.encode(bytes)?;
                },
            ),
        };
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            SzkProofShare::Leader {
                uncompressed_proof_share,
                leader_blind_and_helper_joint_rand_part_opt,
            } => Some(
                uncompressed_proof_share.len() * F::ENCODED_SIZE
                    + if let Some((blind, helper_joint_rand_part)) =
                        leader_blind_and_helper_joint_rand_part_opt
                    {
                        blind.encoded_len()? + helper_joint_rand_part.encoded_len()?
                    } else {
                        0
                    },
            ),
            SzkProofShare::Helper {
                proof_share_seed_and_blind,
                leader_joint_rand_part_opt,
            } => Some(
                proof_share_seed_and_blind.encoded_len()?
                    + if let Some(leader_joint_rand_part) = leader_joint_rand_part_opt {
                        leader_joint_rand_part.encoded_len()?
                    } else {
                        0
                    },
            ),
        }
    }
}

impl<F: FieldElement + Decode> ParameterizedDecode<(bool, usize, bool)> for SzkProofShare<F> {
    fn decode_with_param(
        (is_leader, proof_len, requires_joint_rand): &(bool, usize, bool),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *is_leader {
            Ok(SzkProofShare::Leader {
                uncompressed_proof_share: decode_fieldvec::<F>(*proof_len, bytes)?,
                leader_blind_and_helper_joint_rand_part_opt: if *requires_joint_rand {
                    Some((Seed::decode(bytes)?, Seed::decode(bytes)?))
                } else {
                    None
                },
            })
        } else {
            Ok(SzkProofShare::Helper {
                proof_share_seed_and_blind: Seed::decode(bytes)?,
                leader_joint_rand_part_opt: if *requires_joint_rand {
                    Some(Seed::decode(bytes)?)
                } else {
                    None
                },
            })
        }
    }
}

/// A tuple containing the state and messages produced by an SZK query.
#[derive(Clone, Debug, PartialEq)]
pub struct SzkQueryShare<F: FieldElement> {
    joint_rand_part_opt: Option<Seed<SEED_SIZE>>,
    pub(crate) flp_verifier: Vec<F>,
}

impl<F: FieldElement> Encode for SzkQueryShare<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        if let Some(ref part) = self.joint_rand_part_opt {
            part.encode(bytes)?;
        };

        encode_fieldvec(&self.flp_verifier, bytes)?;
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.flp_verifier.len() * F::ENCODED_SIZE
                + match self.joint_rand_part_opt {
                    Some(ref part) => part.encoded_len()?,
                    None => 0,
                },
        )
    }
}

impl<F: FieldElement + Decode> ParameterizedDecode<(bool, usize)> for SzkQueryShare<F> {
    fn decode_with_param(
        (requires_joint_rand, verifier_len): &(bool, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        Ok(SzkQueryShare {
            joint_rand_part_opt: (*requires_joint_rand)
                .then(|| Seed::decode(bytes))
                .transpose()?,
            flp_verifier: decode_fieldvec(*verifier_len, bytes)?,
        })
    }
}

/// Szk query state.
///
/// The state that needs to be stored by an Szk verifier between query() and decide().
pub(crate) type SzkQueryState = Option<Seed<SEED_SIZE>>;

/// Joint share type for the SZK proof.
///
/// This is produced as the result of combining two query shares.
/// It contains the re-computed joint randomness seed, if applicable. It is consumed by [`Szk::decide`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SzkJointShare(Option<Seed<SEED_SIZE>>);

impl Encode for SzkJointShare {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        if let Some(ref expected_seed) = self.0 {
            expected_seed.encode(bytes)?;
        };
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(match self.0 {
            Some(ref seed) => seed.encoded_len()?,
            None => 0,
        })
    }
}

impl ParameterizedDecode<bool> for SzkJointShare {
    fn decode_with_param(
        requires_joint_rand: &bool,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *requires_joint_rand {
            Ok(SzkJointShare(Some(Seed::decode(bytes)?)))
        } else {
            Ok(SzkJointShare(None))
        }
    }
}

/// Main struct encapsulating the shared zero-knowledge functionality. The type
/// T is the underlying FLP proof system. P is the XOF used to derive all random
/// coins (it should be indifferentiable from a random oracle for security.)
#[derive(Clone, Debug)]
pub struct Szk<T: Type> {
    /// The Type representing the specific FLP system used to prove validity of an input.
    pub(crate) typ: T,
    id: [u8; 4],
}

impl<T: Type> Szk<T> {
    /// Construct an instance of this sharedZK proof system with the underlying
    /// FLP.
    pub fn new(typ: T, algorithm_id: u32) -> Self {
        Self {
            typ,
            id: algorithm_id.to_be_bytes(),
        }
    }

    /// Derive a vector of random field elements for consumption by the FLP
    /// prover.
    fn derive_prove_rand(&self, prove_rand_seed: &Seed<SEED_SIZE>, ctx: &[u8]) -> Vec<T::Field> {
        XofTurboShake128::seed_stream(
            prove_rand_seed.as_ref(),
            &[&mastic::dst_usage(mastic::USAGE_PROVE_RAND), &self.id, ctx],
            &[],
        )
        .into_field_vec(self.typ.prove_rand_len())
    }

    fn derive_joint_rand_part(
        &self,
        aggregator_blind: &Seed<SEED_SIZE>,
        measurement_share: &[T::Field],
        nonce: &[u8; NONCE_SIZE],
        ctx: &[u8],
    ) -> Result<Seed<SEED_SIZE>, SzkError> {
        let mut xof = XofTurboShake128::init(
            aggregator_blind.as_ref(),
            &[
                &mastic::dst_usage(mastic::USAGE_JOINT_RAND_PART),
                &self.id,
                ctx,
            ],
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
        ctx: &[u8],
    ) -> Seed<SEED_SIZE> {
        let mut xof = XofTurboShake128::from_seed_slice(
            &[],
            &[
                &mastic::dst_usage(mastic::USAGE_JOINT_RAND_SEED),
                &self.id,
                ctx,
            ],
        );
        xof.update(&leader_joint_rand_part.0);
        xof.update(&helper_joint_rand_part.0);
        xof.into_seed()
    }

    fn derive_joint_rand_and_seed(
        &self,
        leader_joint_rand_part: &Seed<SEED_SIZE>,
        helper_joint_rand_part: &Seed<SEED_SIZE>,
        ctx: &[u8],
    ) -> (Seed<SEED_SIZE>, Vec<T::Field>) {
        let joint_rand_seed =
            self.derive_joint_rand_seed(leader_joint_rand_part, helper_joint_rand_part, ctx);
        let joint_rand = XofTurboShake128::seed_stream(
            joint_rand_seed.as_ref(),
            &[&mastic::dst_usage(mastic::USAGE_JOINT_RAND), &self.id, ctx],
            &[],
        )
        .into_field_vec(self.typ.joint_rand_len());

        (joint_rand_seed, joint_rand)
    }

    fn derive_helper_proof_share(
        &self,
        proof_share_seed: &Seed<SEED_SIZE>,
        ctx: &[u8],
    ) -> Vec<T::Field> {
        XofTurboShake128::seed_stream(
            proof_share_seed.as_ref(),
            &[&mastic::dst_usage(USAGE_PROOF_SHARE), &self.id, ctx],
            &[],
        )
        .into_field_vec(self.typ.proof_len())
    }

    fn derive_query_rand(
        &self,
        verify_key: &[u8; SEED_SIZE],
        nonce: &[u8; NONCE_SIZE],
        level: u16,
        ctx: &[u8],
    ) -> Vec<T::Field> {
        XofTurboShake128::seed_stream(
            verify_key,
            &[&mastic::dst_usage(mastic::USAGE_QUERY_RAND), &self.id, ctx],
            &[nonce, &level.to_le_bytes()],
        )
        .into_field_vec(self.typ.query_rand_len())
    }

    pub(crate) fn requires_joint_rand(&self) -> bool {
        self.typ.joint_rand_len() > 0
    }

    /// Used by a client to prove validity (according to an FLP system) of an input
    /// that is both shared between the leader and helper
    /// and encoded as a measurement. Has a precondition that leader_input_share
    /// \+ helper_input_share = encoded_measurement.
    /// leader_seed_opt should be set only if the underlying FLP system requires
    /// joint randomness.
    /// In this case, the helper uses the same seed to derive its proof share and
    /// joint randomness.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prove(
        &self,
        ctx: &[u8],
        leader_input_share: &[T::Field],
        helper_input_share: &[T::Field],
        encoded_measurement: &[T::Field],
        rand_seeds: [Seed<SEED_SIZE>; 2],
        leader_seed_opt: Option<Seed<SEED_SIZE>>,
        nonce: &[u8; NONCE_SIZE],
    ) -> Result<[SzkProofShare<T::Field>; 2], SzkError> {
        let [prove_rand_seed, helper_seed] = rand_seeds;
        // If joint randomness is used, derive it from the two input shares,
        // the seeds used to blind the derivation, and the nonce. Pass the
        // leader its blinding seed and the helper's joint randomness part, and
        // pass the helper the leader's joint randomness part. (The seed used to
        // derive the helper's proof share is reused as the helper's blind.)
        let (leader_blind_and_helper_joint_rand_part_opt, leader_joint_rand_part_opt, joint_rand) =
            if let Some(leader_seed) = leader_seed_opt {
                let leader_joint_rand_part =
                    self.derive_joint_rand_part(&leader_seed, leader_input_share, nonce, ctx)?;
                let helper_joint_rand_part =
                    self.derive_joint_rand_part(&helper_seed, helper_input_share, nonce, ctx)?;
                let (_joint_rand_seed, joint_rand) = self.derive_joint_rand_and_seed(
                    &leader_joint_rand_part,
                    &helper_joint_rand_part,
                    ctx,
                );
                (
                    Some((leader_seed, helper_joint_rand_part)),
                    Some(leader_joint_rand_part),
                    joint_rand,
                )
            } else {
                (None, None, Vec::new())
            };

        let prove_rand = self.derive_prove_rand(&prove_rand_seed, ctx);
        let mut leader_proof_share =
            self.typ
                .prove(encoded_measurement, &prove_rand, &joint_rand)?;

        // Generate the proof shares.
        sub_assign_vector(
            &mut leader_proof_share,
            self.derive_helper_proof_share(&helper_seed, ctx),
        );

        // Construct the output messages.
        let leader_proof_share = SzkProofShare::Leader {
            uncompressed_proof_share: leader_proof_share,
            leader_blind_and_helper_joint_rand_part_opt,
        };
        let helper_proof_share = SzkProofShare::Helper {
            proof_share_seed_and_blind: helper_seed,
            leader_joint_rand_part_opt,
        };
        Ok([leader_proof_share, helper_proof_share])
    }

    pub(crate) fn query(
        &self,
        ctx: &[u8],
        level: u16, // level of the prefix tree
        input_share: &[T::Field],
        proof_share: &SzkProofShare<T::Field>,
        verify_key: &[u8; SEED_SIZE],
        nonce: &[u8; NONCE_SIZE],
    ) -> Result<(SzkQueryShare<T::Field>, SzkQueryState), SzkError> {
        let query_rand = self.derive_query_rand(verify_key, nonce, level, ctx);
        let flp_proof_share = match proof_share {
            SzkProofShare::Leader {
                ref uncompressed_proof_share,
                ..
            } => Cow::Borrowed(uncompressed_proof_share),
            SzkProofShare::Helper {
                ref proof_share_seed_and_blind,
                ..
            } => Cow::Owned(self.derive_helper_proof_share(proof_share_seed_and_blind, ctx)),
        };

        let (joint_rand, joint_rand_seed, joint_rand_part) = if self.requires_joint_rand() {
            let ((joint_rand_seed, joint_rand), host_joint_rand_part) = match proof_share {
                SzkProofShare::Leader {
                    uncompressed_proof_share: _,
                    leader_blind_and_helper_joint_rand_part_opt,
                } => match leader_blind_and_helper_joint_rand_part_opt {
                    Some((seed, helper_joint_rand_part)) => {
                        match self.derive_joint_rand_part(seed, input_share, nonce, ctx) {
                            Ok(leader_joint_rand_part) => (
                                self.derive_joint_rand_and_seed(
                                    &leader_joint_rand_part,
                                    helper_joint_rand_part,
                                    ctx,
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
                    leader_joint_rand_part_opt,
                } => match leader_joint_rand_part_opt {
                    Some(leader_joint_rand_part) => match self.derive_joint_rand_part(
                        proof_share_seed_and_blind,
                        input_share,
                        nonce,
                        ctx,
                    ) {
                        Ok(helper_joint_rand_part) => (
                            self.derive_joint_rand_and_seed(
                                leader_joint_rand_part,
                                &helper_joint_rand_part,
                                ctx,
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
                joint_rand_part_opt: joint_rand_part,
                flp_verifier: verifier_share,
            },
            joint_rand_seed,
        ))
    }

    pub(crate) fn merge_query_shares(
        &self,
        ctx: &[u8],
        mut leader_share: SzkQueryShare<T::Field>,
        helper_share: SzkQueryShare<T::Field>,
    ) -> Result<SzkJointShare, SzkError> {
        add_assign_vector(
            &mut leader_share.flp_verifier,
            helper_share.flp_verifier.iter().copied(),
        );
        if self.typ.decide(&leader_share.flp_verifier)? {
            match (
                leader_share.joint_rand_part_opt,
                helper_share.joint_rand_part_opt,
            ) {
                (Some(ref leader_part), Some(ref helper_part)) => Ok(SzkJointShare(Some(
                    self.derive_joint_rand_seed(leader_part, helper_part, ctx),
                ))),
                (None, None) => Ok(SzkJointShare(None)),
                _ => Err(SzkError::Decide(
                    "at least one of the joint randomness parts is missing".to_string(),
                )),
            }
        } else {
            Err(SzkError::Decide("failed to verify FLP proof".to_string()))
        }
    }

    /// Returns true if the joint randomness seed used during the query phase
    /// was correctly computed from both aggregators' parts.
    pub fn decide(
        &self,
        query_state: SzkQueryState,
        joint_share: SzkJointShare,
    ) -> Result<(), SzkError> {
        // Check that joint randomness was properly derived from both
        // aggregators' parts
        match (query_state, joint_share) {
            (Some(joint_rand_seed), SzkJointShare(Some(expected_joint_rand_seed))) => {
                if joint_rand_seed == expected_joint_rand_seed {
                    Ok(())
                } else {
                    Err(SzkError::Decide(
                        "Aggregators failed to compute identical joint randomness seeds"
                            .to_string(),
                    ))
                }
            }

            (None, SzkJointShare(None)) => Ok(()),
            _ => Err(SzkError::Decide(
                "Either computed or stored joint randomness seed is missing".to_string(),
            )),
        }
    }
}

#[inline]
fn option_ct_eq<T>(left: &Option<T>, right: &Option<T>) -> Choice
where
    T: ConstantTimeEq + Sized,
{
    match (left, right) {
        (Some(left), Some(right)) => left.ct_eq(right),
        (None, None) => Choice::from(1),
        _ => Choice::from(0),
    }
}

// This function determines equality between two optional, constant-time comparable tuples. It
// short-circuits on the existence (but not contents) of the values -- a timing side-channel may
// reveal whether the values match on Some or None.
#[inline]
fn option_tuple_ct_eq<T>(left: &Option<(T, T)>, right: &Option<(T, T)>) -> Choice
where
    T: ConstantTimeEq + Sized,
{
    match (left, right) {
        (Some((left_0, left_1)), Some((right_0, right_1))) => {
            left_0.ct_eq(right_0).bitand(left_1.ct_eq(right_1))
        }
        (None, None) => Choice::from(1),
        _ => Choice::from(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        field::{sub_assign_vector, Field128, FieldElementWithInteger},
        flp::{
            gadgets::{Mul, ParallelSum},
            types::{Count, Sum, SumVec},
            Flp, Type,
        },
    };
    use rand::{rng, Rng};

    fn generic_szk_test<T: Type>(typ: T, encoded_measurement: &[T::Field], valid: bool) {
        let mut rng = rng();
        let ctx = b"some application context";
        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 32];
        let szk_typ = Szk::new(typ.clone(), 0);
        rng.fill(&mut verify_key[..]);
        rng.fill(&mut nonce[..]);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = szk_typ.requires_joint_rand().then(|| rng.random());
        let helper_input_share = T::Field::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let proof_shares = szk_typ.prove(
            ctx,
            &leader_input_share,
            &helper_input_share,
            encoded_measurement,
            [prove_rand_seed, helper_seed],
            leader_seed_opt,
            &nonce,
        );

        let [leader_proof_share, helper_proof_share] = proof_shares.unwrap();
        let (leader_query_share, leader_query_state) = szk_typ
            .query(
                ctx,
                0,
                &leader_input_share,
                &leader_proof_share,
                &verify_key,
                &nonce,
            )
            .unwrap();
        let (helper_query_share, helper_query_state) = szk_typ
            .query(
                ctx,
                0,
                &helper_input_share,
                &helper_proof_share,
                &verify_key,
                &nonce,
            )
            .unwrap();

        let joint_share_result =
            szk_typ.merge_query_shares(ctx, leader_query_share.clone(), helper_query_share.clone());
        let joint_share = match joint_share_result {
            Ok(joint_share) => {
                let leader_decision = szk_typ
                    .decide(leader_query_state.clone(), joint_share.clone())
                    .is_ok();
                assert_eq!(
                    leader_decision, valid,
                    "Leader incorrectly determined validity",
                );
                let helper_decision = szk_typ
                    .decide(helper_query_state.clone(), joint_share.clone())
                    .is_ok();
                assert_eq!(
                    helper_decision, valid,
                    "Helper incorrectly determined validity",
                );
                joint_share
            }
            Err(_) => {
                assert!(!valid, "Aggregator incorrectly determined validity");
                SzkJointShare(None)
            }
        };

        //test mutated jr seed
        if szk_typ.requires_joint_rand() {
            let joint_rand_seed_opt = Some(rng.random());
            if let Ok(()) = szk_typ.decide(joint_rand_seed_opt.clone(), joint_share) {
                panic!("Leader accepted wrong jr seed");
            };
        };

        // test mutated verifier
        let mut mutated_query_share = leader_query_share.clone();
        for x in mutated_query_share.flp_verifier.iter_mut() {
            *x += T::Field::from(
                <T::Field as FieldElementWithInteger>::Integer::try_from(7).unwrap(),
            );
        }

        let joint_share_res =
            szk_typ.merge_query_shares(ctx, mutated_query_share, helper_query_share.clone());
        let leader_decision = match joint_share_res {
            Ok(joint_share) => szk_typ
                .decide(leader_query_state.clone(), joint_share)
                .is_ok(),
            Err(_) => false,
        };
        assert!(!leader_decision, "Leader validated after proof mutation");

        // test mutated input share
        let mut mutated_input = leader_input_share.clone();
        mutated_input[0] *=
            T::Field::from(<T::Field as FieldElementWithInteger>::Integer::try_from(23).unwrap());
        let (mutated_query_share, mutated_query_state) = szk_typ
            .query(
                ctx,
                0,
                &mutated_input,
                &leader_proof_share,
                &verify_key,
                &nonce,
            )
            .unwrap();

        let joint_share_res =
            szk_typ.merge_query_shares(ctx, mutated_query_share, helper_query_share.clone());

        let leader_decision = match joint_share_res {
            Ok(joint_share) => szk_typ.decide(mutated_query_state, joint_share).is_ok(),
            Err(_) => false,
        };
        assert!(!leader_decision, "Leader validated after input mutation");

        // test mutated proof share
        let (mut mutated_proof, leader_blind_and_helper_joint_rand_part_opt) =
            match leader_proof_share {
                SzkProofShare::Leader {
                    uncompressed_proof_share,
                    leader_blind_and_helper_joint_rand_part_opt,
                } => (
                    uncompressed_proof_share,
                    leader_blind_and_helper_joint_rand_part_opt,
                ),
                _ => (vec![], None),
            };
        mutated_proof[0] *=
            T::Field::from(<T::Field as FieldElementWithInteger>::Integer::try_from(23).unwrap());
        let mutated_proof_share = SzkProofShare::Leader {
            uncompressed_proof_share: mutated_proof,
            leader_blind_and_helper_joint_rand_part_opt,
        };
        let (leader_query_share, leader_query_state) = szk_typ
            .query(
                ctx,
                0,
                &leader_input_share,
                &mutated_proof_share,
                &verify_key,
                &nonce,
            )
            .unwrap();
        let joint_share_res =
            szk_typ.merge_query_shares(ctx, leader_query_share, helper_query_share.clone());

        let leader_decision = match joint_share_res {
            Ok(joint_share) => szk_typ
                .decide(leader_query_state.clone(), joint_share)
                .is_ok(),
            Err(_) => false,
        };
        assert!(!leader_decision, "Leader validated after proof mutation");
    }

    #[test]
    fn test_sum_proof_share_encode() {
        let mut rng = rng();
        let mut nonce = [0u8; 16];
        let max_measurement = 13;
        rng.fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(max_measurement).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let szk_typ = Szk::new(sum, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = Some(rng.random());
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [leader_proof_share, _] = szk_typ
            .prove(
                b"some application",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        assert_eq!(
            leader_proof_share.encoded_len().unwrap(),
            leader_proof_share.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_sumvec_proof_share_encode() {
        let mut rng = rng();
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let szk_typ = Szk::new(sumvec, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = Some(rng.random());
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [l_proof_share, _] = szk_typ
            .prove(
                b"some application",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        assert_eq!(
            l_proof_share.encoded_len().unwrap(),
            l_proof_share.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_count_proof_share_encode() {
        let mut rng = rng();
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let szk_typ = Szk::new(count, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = Some(rng.random());
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [l_proof_share, _] = szk_typ
            .prove(
                b"some application",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        assert_eq!(
            l_proof_share.encoded_len().unwrap(),
            l_proof_share.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_sum_leader_proof_share_roundtrip() {
        let mut rng = rng();
        let max_measurement = 13;
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(max_measurement).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let szk_typ = Szk::new(sum, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = None;
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [l_proof_share, _] = szk_typ
            .prove(
                b"some application",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let decoding_parameter = (
            true,
            szk_typ.typ.proof_len(),
            szk_typ.typ.joint_rand_len() != 0,
        );
        let encoded_proof_share = l_proof_share.get_encoded().unwrap();
        let decoded_proof_share =
            SzkProofShare::get_decoded_with_param(&decoding_parameter, &encoded_proof_share[..])
                .unwrap();
        assert_eq!(l_proof_share, decoded_proof_share);
    }

    #[test]
    fn test_sum_helper_proof_share_roundtrip() {
        let mut rng = rng();
        let max_measurement = 13;
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(max_measurement).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let szk_typ = Szk::new(sum, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = None;
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [_, h_proof_share] = szk_typ
            .prove(
                b"some application",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let decoding_parameter = (
            false,
            szk_typ.typ.proof_len(),
            szk_typ.typ.joint_rand_len() != 0,
        );
        let encoded_proof_share = h_proof_share.get_encoded().unwrap();
        let decoded_proof_share =
            SzkProofShare::get_decoded_with_param(&decoding_parameter, &encoded_proof_share[..])
                .unwrap();
        assert_eq!(h_proof_share, decoded_proof_share);
    }

    #[test]
    fn test_count_leader_proof_share_roundtrip() {
        let mut rng = rng();
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let szk_typ = Szk::new(count, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = None;
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [l_proof_share, _] = szk_typ
            .prove(
                b"some application",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let decoding_parameter = (
            true,
            szk_typ.typ.proof_len(),
            szk_typ.typ.joint_rand_len() != 0,
        );
        let encoded_proof_share = l_proof_share.get_encoded().unwrap();
        let decoded_proof_share =
            SzkProofShare::get_decoded_with_param(&decoding_parameter, &encoded_proof_share[..])
                .unwrap();
        assert_eq!(l_proof_share, decoded_proof_share);
    }

    #[test]
    fn test_count_helper_proof_share_roundtrip() {
        let mut rng = rng();
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let szk_typ = Szk::new(count, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = None;
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [_, h_proof_share] = szk_typ
            .prove(
                b"some application",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let decoding_parameter = (
            false,
            szk_typ.typ.proof_len(),
            szk_typ.typ.joint_rand_len() != 0,
        );
        let encoded_proof_share = h_proof_share.get_encoded().unwrap();
        let decoded_proof_share =
            SzkProofShare::get_decoded_with_param(&decoding_parameter, &encoded_proof_share[..])
                .unwrap();
        assert_eq!(h_proof_share, decoded_proof_share);
    }

    #[test]
    fn test_sumvec_leader_proof_share_roundtrip() {
        let mut rng = rng();
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let szk_typ = Szk::new(sumvec, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = Some(rng.random());
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [l_proof_share, _] = szk_typ
            .prove(
                b"some application",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let decoding_parameter = (
            true,
            szk_typ.typ.proof_len(),
            szk_typ.typ.joint_rand_len() != 0,
        );
        let encoded_proof_share = l_proof_share.get_encoded().unwrap();
        let decoded_proof_share =
            SzkProofShare::get_decoded_with_param(&decoding_parameter, &encoded_proof_share[..])
                .unwrap();
        assert_eq!(l_proof_share, decoded_proof_share);
    }

    #[test]
    fn test_sumvec_helper_proof_share_roundtrip() {
        let mut rng = rng();
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let szk_typ = Szk::new(sumvec, 0);
        let prove_rand_seed = rng.random();
        let helper_seed = rng.random();
        let leader_seed_opt = Some(rng.random());
        let helper_input_share = Field128::random_vector(szk_typ.typ.input_len());
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        sub_assign_vector(&mut leader_input_share, helper_input_share.iter().copied());

        let [_, h_proof_share] = szk_typ
            .prove(
                b"some applicqation",
                &leader_input_share,
                &helper_input_share,
                &encoded_measurement[..],
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let decoding_parameter = (
            false,
            szk_typ.typ.proof_len(),
            szk_typ.typ.joint_rand_len() != 0,
        );
        let encoded_proof_share = h_proof_share.get_encoded().unwrap();
        let decoded_proof_share =
            SzkProofShare::get_decoded_with_param(&decoding_parameter, &encoded_proof_share[..])
                .unwrap();
        assert_eq!(h_proof_share, decoded_proof_share);
    }

    #[test]
    fn test_sum() {
        let max_measurement = 13;
        let sum = Sum::<Field128>::new(max_measurement).unwrap();

        let five = Field128::from(5);
        let nine = sum.encode_measurement(&9).unwrap();
        let bad_encoding = &vec![five; sum.input_len()];
        generic_szk_test(sum.clone(), &nine, true);
        generic_szk_test(sum, bad_encoding, false);
    }

    #[test]
    fn test_sumvec() {
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();

        let five = Field128::from(5);
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let bad_encoding = &vec![five; sumvec.input_len()];
        generic_szk_test(sumvec.clone(), &encoded_measurement, true);
        generic_szk_test(sumvec, bad_encoding, false);
    }

    #[test]
    fn test_count() {
        let count = Count::<Field128>::new();
        let encoded_true = count.encode_measurement(&true).unwrap();
        generic_szk_test(count, &encoded_true, true);
    }
}
