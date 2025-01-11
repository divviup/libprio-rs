// SPDX-License-Identifier: MPL-2.0

//! A wrapper for the [FLP](crate::flp) proof system as it is used in Mastic.
//!
//! [`Szk`] is a wrapper struct for the FLP proof system (accessible through the [`Type`] trait).
//! It consists of a method for splitting a measurement into shares and jointly verifying the
//! validating the measurment.
//!
//! The underlying FLP accepts inputs that are vectors over a finite field, and its random coins
//! and proofs are also vectors over the same field. In contrast, only the leader's SZK proof share
//! contains any finite field elements, and the initial coins are bytestrings ([`Seed`]s). The
//! wrapper struct defined here uses an [`Xof`] (to be modeled as a random oracle) to sample coins
//! and the helper's proof share, following a strategy similar to
//! [`Prio3`](crate::vdaf::prio3::Prio3).

use crate::{
    codec::{CodecError, Decode, Encode, ParameterizedDecode},
    field::{decode_fieldvec, encode_fieldvec, FieldElement},
    flp::{FlpError, Type},
    prng::PrngError,
    vdaf::{
        mastic::{self, USAGE_PROOF_SHARE, USAGE_WEIGHT_SHARE},
        xof::{IntoFieldVec, Seed, Xof, XofTurboShake128},
    },
};
use std::borrow::Cow;
use std::io::Cursor;
use std::ops::BitAnd;
use subtle::{Choice, ConstantTimeEq};

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

    /// PRNG error.
    #[error("prng error: {0}")]
    Prng(#[from] PrngError),

    /// Codec error.
    #[error("codec error: {0}")]
    Codec(#[from] CodecError),
}

/// An SZK input share.
#[derive(Debug, Clone)]
pub enum SzkInputShare<F> {
    /// The leader's input share.
    Leader {
        /// Share of the underlying measurement. The Helper's share is derived from its seed.
        uncompressed_meas_share: Vec<F>,

        /// Share of the validity proof. The Helper's share is derived from its seed.
        uncompressed_proof_share: Vec<F>,

        /// Set only if joint randomness is needed for the FLP. The first component is the leader's
        /// blind and the second is the helper's joint randomness part.
        leader_blind_and_helper_joint_rand_part_opt: Option<(Seed<32>, Seed<32>)>,
    },

    /// The helper's input share.
    Helper {
        /// Used to derive the helper's input and proof shares and its blind for FLP joint
        /// randomness computation.
        share_seed_and_blind: Seed<32>,
        /// Set only if joint randomness is needed for the FLP.
        leader_joint_rand_part_opt: Option<Seed<32>>,
    },
}

impl<F: FieldElement> PartialEq for SzkInputShare<F> {
    fn eq(&self, other: &SzkInputShare<F>) -> bool {
        bool::from(self.ct_eq(other))
    }
}

impl<F: FieldElement> ConstantTimeEq for SzkInputShare<F> {
    fn ct_eq(&self, other: &SzkInputShare<F>) -> Choice {
        match (self, other) {
            (
                SzkInputShare::Leader {
                    uncompressed_meas_share: s_meas,
                    uncompressed_proof_share: s_proof,
                    leader_blind_and_helper_joint_rand_part_opt: s_blind,
                },
                SzkInputShare::Leader {
                    uncompressed_meas_share: o_meas,
                    uncompressed_proof_share: o_proof,
                    leader_blind_and_helper_joint_rand_part_opt: o_blind,
                },
            ) => s_meas.ct_eq(o_meas).bitand(
                s_proof[..]
                    .ct_eq(&o_proof[..])
                    .bitand(option_tuple_ct_eq(s_blind, o_blind)),
            ),
            (
                SzkInputShare::Helper {
                    share_seed_and_blind: s_seed,
                    leader_joint_rand_part_opt: s_rand,
                },
                SzkInputShare::Helper {
                    share_seed_and_blind: o_seed,
                    leader_joint_rand_part_opt: o_rand,
                },
            ) => s_seed.ct_eq(o_seed).bitand(option_ct_eq(s_rand, o_rand)),
            _ => Choice::from(0),
        }
    }
}

impl<F: FieldElement> Encode for SzkInputShare<F> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match self {
            SzkInputShare::Leader {
                uncompressed_meas_share,
                uncompressed_proof_share,
                leader_blind_and_helper_joint_rand_part_opt,
            } => {
                if let Some((blind, helper_joint_rand_part)) =
                    leader_blind_and_helper_joint_rand_part_opt
                {
                    blind.encode(bytes)?;
                    helper_joint_rand_part.encode(bytes)?;
                }
                encode_fieldvec(uncompressed_meas_share, bytes)?;
                encode_fieldvec(uncompressed_proof_share, bytes)?;
            }
            SzkInputShare::Helper {
                share_seed_and_blind,
                leader_joint_rand_part_opt,
            } => {
                share_seed_and_blind.encode(bytes)?;
                if let Some(leader_joint_rand_part) = leader_joint_rand_part_opt {
                    leader_joint_rand_part.encode(bytes)?;
                }
            }
        };
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            SzkInputShare::Leader {
                uncompressed_meas_share,
                uncompressed_proof_share,
                leader_blind_and_helper_joint_rand_part_opt,
            } => Some(
                uncompressed_meas_share.len() * F::ENCODED_SIZE
                    + uncompressed_proof_share.len() * F::ENCODED_SIZE
                    + if let Some((blind, helper_joint_rand_part)) =
                        leader_blind_and_helper_joint_rand_part_opt
                    {
                        blind.encoded_len()? + helper_joint_rand_part.encoded_len()?
                    } else {
                        0
                    },
            ),
            SzkInputShare::Helper {
                share_seed_and_blind,
                leader_joint_rand_part_opt,
            } => Some(
                share_seed_and_blind.encoded_len()?
                    + if let Some(leader_joint_rand_part) = leader_joint_rand_part_opt {
                        leader_joint_rand_part.encoded_len()?
                    } else {
                        0
                    },
            ),
        }
    }
}

impl<T: Type> ParameterizedDecode<(&Szk<T>, bool)> for SzkInputShare<T::Field> {
    fn decode_with_param(
        (szk, is_leader): &(&Szk<T>, bool),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *is_leader {
            Ok(SzkInputShare::Leader {
                leader_blind_and_helper_joint_rand_part_opt: if szk.typ.joint_rand_len() > 0 {
                    Some((Seed::decode(bytes)?, Seed::decode(bytes)?))
                } else {
                    None
                },
                uncompressed_meas_share: decode_fieldvec(szk.typ.input_len(), bytes)?,
                uncompressed_proof_share: decode_fieldvec(szk.typ.proof_len(), bytes)?,
            })
        } else {
            Ok(SzkInputShare::Helper {
                share_seed_and_blind: Seed::decode(bytes)?,
                leader_joint_rand_part_opt: if szk.typ.joint_rand_len() > 0 {
                    Some(Seed::decode(bytes)?)
                } else {
                    None
                },
            })
        }
    }
}

/// SZK query share.
#[derive(Clone, Debug, PartialEq)]
pub struct SzkQueryShare<F: FieldElement> {
    joint_rand_part_opt: Option<Seed<32>>,
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

/// SZK query state.
///
/// The state that needs to be stored by an Szk verifier between query() and decide().
pub(crate) type SzkQueryState = Option<Seed<32>>;

/// Joint share type for the SZK proof.
///
/// This is produced as the result of combining two query shares.
/// It contains the re-computed joint randomness seed, if applicable. It is consumed by [`Szk::decide`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SzkJointShare(Option<Seed<32>>);

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

/// Main struct encapsulating the shared zero-knowledge functionality. The type `T` is the
/// underlying FLP proof system.
#[derive(Clone, Debug)]
pub struct Szk<T: Type> {
    /// The Type representing the specific FLP system used to prove validity of a measurement.
    pub(crate) typ: T,
    id: [u8; 4],
}

impl<T: Type> Szk<T> {
    /// Construct an instance of this sharedZK proof system with the underlying
    /// FLP.
    pub fn new(typ: T, algorithm_id: u32) -> Self {
        Self {
            typ,
            id: algorithm_id.to_le_bytes(),
        }
    }

    /// Derive a vector of random field elements for consumption by the FLP
    /// prover.
    fn derive_prove_rand(&self, prove_rand_seed: &Seed<32>, ctx: &[u8]) -> Vec<T::Field> {
        XofTurboShake128::seed_stream(
            prove_rand_seed,
            &[&mastic::dst_usage(mastic::USAGE_PROVE_RAND), &self.id, ctx],
            &[],
        )
        .into_field_vec(self.typ.prove_rand_len())
    }

    fn derive_joint_rand_part(
        &self,
        aggregator_blind: &Seed<32>,
        measurement_share: &[T::Field],
        nonce: &[u8; 16],
        ctx: &[u8],
    ) -> Result<Seed<32>, SzkError> {
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
        leader_joint_rand_part: &Seed<32>,
        helper_joint_rand_part: &Seed<32>,
        ctx: &[u8],
    ) -> Seed<32> {
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
        leader_joint_rand_part: &Seed<32>,
        helper_joint_rand_part: &Seed<32>,
        ctx: &[u8],
    ) -> (Seed<32>, Vec<T::Field>) {
        let joint_rand_seed =
            self.derive_joint_rand_seed(leader_joint_rand_part, helper_joint_rand_part, ctx);
        let joint_rand = XofTurboShake128::seed_stream(
            &joint_rand_seed,
            &[&mastic::dst_usage(mastic::USAGE_JOINT_RAND), &self.id, ctx],
            &[],
        )
        .into_field_vec(self.typ.joint_rand_len());

        (joint_rand_seed, joint_rand)
    }

    pub(crate) fn derive_helper_meas_share(
        &self,
        meas_share_seed: &Seed<32>,
        ctx: &[u8],
    ) -> Vec<T::Field> {
        XofTurboShake128::seed_stream(
            meas_share_seed,
            &[&mastic::dst_usage(USAGE_WEIGHT_SHARE), &self.id, ctx],
            &[],
        )
        .into_field_vec(self.typ.input_len())
    }

    fn derive_helper_proof_share(&self, proof_share_seed: &Seed<32>, ctx: &[u8]) -> Vec<T::Field> {
        XofTurboShake128::seed_stream(
            proof_share_seed,
            &[&mastic::dst_usage(USAGE_PROOF_SHARE), &self.id, ctx],
            &[],
        )
        .into_field_vec(self.typ.proof_len())
    }

    fn derive_query_rand(
        &self,
        verify_key: &[u8; 32],
        nonce: &[u8; 16],
        level: u16,
        ctx: &[u8],
    ) -> Vec<T::Field> {
        let mut xof = XofTurboShake128::init(
            verify_key,
            &[&mastic::dst_usage(mastic::USAGE_QUERY_RAND), &self.id, ctx],
        );
        xof.update(nonce);
        xof.update(&level.to_le_bytes());
        xof.into_seed_stream()
            .into_field_vec(self.typ.query_rand_len())
    }

    pub(crate) fn requires_joint_rand(&self) -> bool {
        self.typ.joint_rand_len() > 0
    }

    /// Produce SZK input shares for a given encoded measurement.
    ///
    /// `leader_seed_opt` should be set if and only if the underlying FLP system requires joint
    /// randomness. In this case, the helper uses the same seed to derive its proof share and joint
    /// randomness.
    pub(crate) fn prove(
        &self,
        ctx: &[u8],
        encoded_measurement: &[T::Field],
        rand_seeds: [Seed<32>; 2],
        leader_seed_opt: Option<Seed<32>>,
        nonce: &[u8; 16],
    ) -> Result<[SzkInputShare<T::Field>; 2], SzkError> {
        let [prove_rand_seed, helper_seed] = rand_seeds;

        let mut leader_meas_share = encoded_measurement.to_vec();
        let helper_meas_share = &self.derive_helper_meas_share(&helper_seed, ctx);
        for (x, y) in leader_meas_share.iter_mut().zip(helper_meas_share) {
            *x -= *y;
        }

        // If joint randomness is used, derive it from the two input shares,
        // the seeds used to blind the derivation, and the nonce. Pass the
        // leader its blinding seed and the helper's joint randomness part, and
        // pass the helper the leader's joint randomness part. (The seed used to
        // derive the helper's proof share is reused as the helper's blind.)
        let (leader_blind_and_helper_joint_rand_part_opt, leader_joint_rand_part_opt, joint_rand) =
            if let Some(leader_seed) = leader_seed_opt {
                let leader_joint_rand_part =
                    self.derive_joint_rand_part(&leader_seed, &leader_meas_share, nonce, ctx)?;
                let helper_joint_rand_part =
                    self.derive_joint_rand_part(&helper_seed, helper_meas_share, nonce, ctx)?;
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
        for (x, y) in leader_proof_share
            .iter_mut()
            .zip(self.derive_helper_proof_share(&helper_seed, ctx))
        {
            *x -= y;
        }

        // Construct the output messages.
        let leader_proof_share = SzkInputShare::Leader {
            uncompressed_meas_share: leader_meas_share,
            uncompressed_proof_share: leader_proof_share,
            leader_blind_and_helper_joint_rand_part_opt,
        };
        let helper_proof_share = SzkInputShare::Helper {
            share_seed_and_blind: helper_seed,
            leader_joint_rand_part_opt,
        };
        Ok([leader_proof_share, helper_proof_share])
    }

    pub(crate) fn query(
        &self,
        ctx: &[u8],
        level: u16, // level of the prefix tree
        szk_input_share: &SzkInputShare<T::Field>,
        verify_key: &[u8; 32],
        nonce: &[u8; 16],
    ) -> Result<(SzkQueryShare<T::Field>, SzkQueryState), SzkError> {
        let query_rand = self.derive_query_rand(verify_key, nonce, level, ctx);
        let (flp_input_share, flp_proof_share) = match szk_input_share {
            SzkInputShare::Leader {
                ref uncompressed_meas_share,
                ref uncompressed_proof_share,
                ..
            } => (
                Cow::Borrowed(uncompressed_meas_share),
                Cow::Borrowed(uncompressed_proof_share),
            ),
            SzkInputShare::Helper {
                ref share_seed_and_blind,
                ..
            } => (
                Cow::Owned(self.derive_helper_meas_share(share_seed_and_blind, ctx)),
                Cow::Owned(self.derive_helper_proof_share(share_seed_and_blind, ctx)),
            ),
        };

        let (joint_rand, joint_rand_seed, joint_rand_part) = if self.requires_joint_rand() {
            let ((joint_rand_seed, joint_rand), host_joint_rand_part) = match szk_input_share {
                SzkInputShare::Leader {
                    uncompressed_meas_share: _,
                    uncompressed_proof_share: _,
                    leader_blind_and_helper_joint_rand_part_opt,
                } => match leader_blind_and_helper_joint_rand_part_opt {
                    Some((seed, helper_joint_rand_part)) => {
                        match self.derive_joint_rand_part(
                            seed,
                            flp_input_share.as_ref(),
                            nonce,
                            ctx,
                        ) {
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
                SzkInputShare::Helper {
                    share_seed_and_blind,
                    leader_joint_rand_part_opt,
                } => match leader_joint_rand_part_opt {
                    Some(leader_joint_rand_part) => match self.derive_joint_rand_part(
                        share_seed_and_blind,
                        flp_input_share.as_ref(),
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
            flp_input_share.as_ref(),
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
        for (x, y) in leader_share
            .flp_verifier
            .iter_mut()
            .zip(helper_share.flp_verifier)
        {
            *x += y;
        }
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
        field::Field128,
        field::{random_vector, FieldElementWithInteger},
        flp::gadgets::{Mul, ParallelSum},
        flp::types::{Count, Sum, SumVec},
        flp::Type,
    };
    use assert_matches::assert_matches;
    use rand::{thread_rng, Rng};

    fn generic_szk_test<T: Type>(typ: T, encoded_measurement: &[T::Field], valid: bool) {
        let ctx = b"some application context";
        let mut nonce = [0u8; 16];
        let mut verify_key = [0u8; 32];
        let szk = Szk::new(typ.clone(), 0);
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = szk.requires_joint_rand().then(|| Seed::generate().unwrap());

        let [mut leader_input_share, helper_input_share] = szk
            .prove(
                ctx,
                encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let (leader_query_share, leader_query_state) = szk
            .query(ctx, 0, &leader_input_share, &verify_key, &nonce)
            .unwrap();
        let (helper_query_share, helper_query_state) = szk
            .query(ctx, 0, &helper_input_share, &verify_key, &nonce)
            .unwrap();

        let joint_share_result =
            szk.merge_query_shares(ctx, leader_query_share.clone(), helper_query_share.clone());
        let joint_share = match joint_share_result {
            Ok(joint_share) => {
                let leader_decision = szk
                    .decide(leader_query_state.clone(), joint_share.clone())
                    .is_ok();
                assert_eq!(
                    leader_decision, valid,
                    "Leader incorrectly determined validity",
                );
                let helper_decision = szk
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
        if szk.requires_joint_rand() {
            let joint_rand_seed_opt = Some(Seed::generate().unwrap());
            if let Ok(()) = szk.decide(joint_rand_seed_opt.clone(), joint_share) {
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
            szk.merge_query_shares(ctx, mutated_query_share, helper_query_share.clone());
        let leader_decision = match joint_share_res {
            Ok(joint_share) => szk.decide(leader_query_state.clone(), joint_share).is_ok(),
            Err(_) => false,
        };
        assert!(!leader_decision, "Leader validated after proof mutation");

        // test mutated input share
        assert_matches!(leader_input_share,
        SzkInputShare::Leader { ref mut uncompressed_meas_share, ..} => {
            uncompressed_meas_share[0] *= T::Field::from(<T::Field as FieldElementWithInteger>::Integer::try_from(23).unwrap())
        });
        let (mutated_query_share, mutated_query_state) = szk
            .query(ctx, 0, &leader_input_share, &verify_key, &nonce)
            .unwrap();

        let joint_share_res =
            szk.merge_query_shares(ctx, mutated_query_share, helper_query_share.clone());

        let leader_decision = match joint_share_res {
            Ok(joint_share) => szk.decide(mutated_query_state, joint_share).is_ok(),
            Err(_) => false,
        };
        assert!(!leader_decision, "Leader validated after input mutation");

        // test mutated proof share
        let (
            uncompressed_meas_share,
            mut mutated_proof,
            leader_blind_and_helper_joint_rand_part_opt,
        ) = match leader_input_share {
            SzkInputShare::Leader {
                uncompressed_meas_share,
                uncompressed_proof_share,
                leader_blind_and_helper_joint_rand_part_opt,
            } => (
                uncompressed_meas_share,
                uncompressed_proof_share,
                leader_blind_and_helper_joint_rand_part_opt,
            ),
            _ => (vec![], vec![], None),
        };
        mutated_proof[0] *=
            T::Field::from(<T::Field as FieldElementWithInteger>::Integer::try_from(23).unwrap());
        let mutated_input_share = SzkInputShare::Leader {
            uncompressed_meas_share,
            uncompressed_proof_share: mutated_proof,
            leader_blind_and_helper_joint_rand_part_opt,
        };
        let (leader_query_share, leader_query_state) = szk
            .query(ctx, 0, &mutated_input_share, &verify_key, &nonce)
            .unwrap();
        let joint_share_res =
            szk.merge_query_shares(ctx, leader_query_share, helper_query_share.clone());

        let leader_decision = match joint_share_res {
            Ok(joint_share) => szk.decide(leader_query_state.clone(), joint_share).is_ok(),
            Err(_) => false,
        };
        assert!(!leader_decision, "Leader validated after proof mutation");
    }

    #[test]
    fn test_sum_input_share_encode() {
        let mut nonce = [0u8; 16];
        let max_measurement = 13;
        thread_rng().fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(max_measurement).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let szk = Szk::new(sum, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = Some(Seed::generate().unwrap());
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [leader_input_share, _] = szk
            .prove(
                b"some application",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        assert_eq!(
            leader_input_share.encoded_len().unwrap(),
            leader_input_share.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_sumvec_input_share_encode() {
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let szk = Szk::new(sumvec, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = Some(Seed::generate().unwrap());
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_input_share, _] = szk
            .prove(
                b"some application",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        assert_eq!(
            l_input_share.encoded_len().unwrap(),
            l_input_share.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_count_input_share_encode() {
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let szk = Szk::new(count, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = Some(Seed::generate().unwrap());
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_input_share, _] = szk
            .prove(
                b"some application",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        assert_eq!(
            l_input_share.encoded_len().unwrap(),
            l_input_share.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn test_sum_leader_input_share_roundtrip() {
        let max_measurement = 13;
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(max_measurement).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let szk = Szk::new(sum, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = None;
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_input_share, _] = szk
            .prove(
                b"some application",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let encoded_input_share = l_input_share.get_encoded().unwrap();
        let decoded_input_share =
            SzkInputShare::get_decoded_with_param(&(&szk, true), &encoded_input_share[..]).unwrap();
        assert_eq!(l_input_share, decoded_input_share);
    }

    #[test]
    fn test_sum_helper_input_share_roundtrip() {
        let max_measurement = 13;
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(max_measurement).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let szk = Szk::new(sum, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = None;
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [_, h_input_share] = szk
            .prove(
                b"some application",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let encoded_input_share = h_input_share.get_encoded().unwrap();
        let decoded_input_share =
            SzkInputShare::get_decoded_with_param(&(&szk, false), &encoded_input_share[..])
                .unwrap();
        assert_eq!(h_input_share, decoded_input_share);
    }

    #[test]
    fn test_count_leader_input_share_roundtrip() {
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let szk = Szk::new(count, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = None;
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_input_share, _] = szk
            .prove(
                b"some application",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let encoded_input_share = l_input_share.get_encoded().unwrap();
        let decoded_input_share =
            SzkInputShare::get_decoded_with_param(&(&szk, true), &encoded_input_share[..]).unwrap();
        assert_eq!(l_input_share, decoded_input_share);
    }

    #[test]
    fn test_count_helper_input_share_roundtrip() {
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let szk = Szk::new(count, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = None;
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [_, h_input_share] = szk
            .prove(
                b"some application",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let encoded_input_share = h_input_share.get_encoded().unwrap();
        let decoded_input_share =
            SzkInputShare::get_decoded_with_param(&(&szk, false), &encoded_input_share[..])
                .unwrap();
        assert_eq!(h_input_share, decoded_input_share);
    }

    #[test]
    fn test_sumvec_leader_input_share_roundtrip() {
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let szk = Szk::new(sumvec, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = Some(Seed::generate().unwrap());
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_input_share, _] = szk
            .prove(
                b"some application",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let encoded_input_share = l_input_share.get_encoded().unwrap();
        let decoded_input_share =
            SzkInputShare::get_decoded_with_param(&(&szk, true), &encoded_input_share[..]).unwrap();
        assert_eq!(l_input_share, decoded_input_share);
    }

    #[test]
    fn test_sumvec_helper_input_share_roundtrip() {
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let szk = Szk::new(sumvec, 0);
        let prove_rand_seed = Seed::generate().unwrap();
        let helper_seed = Seed::generate().unwrap();
        let leader_seed_opt = Some(Seed::generate().unwrap());
        let helper_input_share = random_vector(szk.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [_, h_input_share] = szk
            .prove(
                b"some applicqation",
                &encoded_measurement,
                [prove_rand_seed, helper_seed],
                leader_seed_opt,
                &nonce,
            )
            .unwrap();

        let encoded_input_share = h_input_share.get_encoded().unwrap();
        let decoded_input_share =
            SzkInputShare::get_decoded_with_param(&(&szk, false), &encoded_input_share[..])
                .unwrap();
        assert_eq!(h_input_share, decoded_input_share);
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
