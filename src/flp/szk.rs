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
    field::{decode_fieldvec, encode_fieldvec, FieldElement},
    flp::{FlpError, Type},
    prng::{Prng, PrngError},
    vdaf::xof::{IntoFieldVec, Seed, Xof, XofTurboShake128},
};
use std::borrow::Cow;
use std::ops::BitAnd;
use std::{io::Cursor, marker::PhantomData};
use subtle::{Choice, ConstantTimeEq};

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
#[derive(Debug, Clone)]
pub enum SzkProofShare<F, const SEED_SIZE: usize> {
    /// Leader's proof share is uncompressed.
    Leader {
        /// Share of an FLP proof, as a vector of Field elements.
        uncompressed_proof_share: Vec<F>,
        /// Set only if joint randomness is needed. The first Seed is a blind, second
        /// is the helper's joint randomness part.
        leader_blind_and_helper_joint_rand_part_opt: Option<(Seed<SEED_SIZE>, Seed<SEED_SIZE>)>,
    },
    /// The Helper uses one seed for both its compressed proof share and as the blind for its joint
    /// randomness.
    Helper {
        /// The Seed that acts both as the compressed proof share and, optionally, as the blind.
        proof_share_seed_and_blind: Seed<SEED_SIZE>,
        /// The leader's joint randomness part, if needed.
        leader_joint_rand_part_opt: Option<Seed<SEED_SIZE>>,
    },
}

impl<F: FieldElement, const SEED_SIZE: usize> PartialEq for SzkProofShare<F, SEED_SIZE> {
    fn eq(&self, other: &SzkProofShare<F, SEED_SIZE>) -> bool {
        bool::from(self.ct_eq(other))
    }
}

impl<F: FieldElement, const SEED_SIZE: usize> ConstantTimeEq for SzkProofShare<F, SEED_SIZE> {
    fn ct_eq(&self, other: &SzkProofShare<F, SEED_SIZE>) -> Choice {
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

impl<F: FieldElement, const SEED_SIZE: usize> Encode for SzkProofShare<F, SEED_SIZE> {
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

impl<F: FieldElement + Decode, const SEED_SIZE: usize> ParameterizedDecode<(bool, usize, bool)>
    for SzkProofShare<F, SEED_SIZE>
{
    fn decode_with_param(
        (is_leader, proof_len, requires_joint_rand): &(bool, usize, bool),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *is_leader {
            Ok(SzkProofShare::Leader {
                uncompressed_proof_share: decode_fieldvec::<F>(*proof_len, bytes)?,
                leader_blind_and_helper_joint_rand_part_opt: if *requires_joint_rand {
                    Some((
                        Seed::<SEED_SIZE>::decode(bytes)?,
                        Seed::<SEED_SIZE>::decode(bytes)?,
                    ))
                } else {
                    None
                },
            })
        } else {
            Ok(SzkProofShare::Helper {
                proof_share_seed_and_blind: Seed::<SEED_SIZE>::decode(bytes)?,
                leader_joint_rand_part_opt: if *requires_joint_rand {
                    Some(Seed::<SEED_SIZE>::decode(bytes)?)
                } else {
                    None
                },
            })
        }
    }
}

/// A tuple containing the state and messages produced by an SZK query.
#[derive(Clone, Debug)]
pub struct SzkQueryShare<F: FieldElement, const SEED_SIZE: usize> {
    joint_rand_part_opt: Option<Seed<SEED_SIZE>>,
    pub(crate) flp_verifier: Vec<F>,
}

impl<F: FieldElement, const SEED_SIZE: usize> Encode for SzkQueryShare<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        if let Some(ref part) = self.joint_rand_part_opt {
            part.encode(bytes)?
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

impl<F: FieldElement + Decode, const SEED_SIZE: usize> ParameterizedDecode<(bool, usize)>
    for SzkQueryShare<F, SEED_SIZE>
{
    fn decode_with_param(
        (requires_joint_rand, verifier_len): &(bool, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *requires_joint_rand {
            Ok(SzkQueryShare {
                joint_rand_part_opt: Some(Seed::<SEED_SIZE>::decode(bytes)?),
                flp_verifier: decode_fieldvec(*verifier_len, bytes)?,
            })
        } else {
            Ok(SzkQueryShare {
                joint_rand_part_opt: None,
                flp_verifier: decode_fieldvec(*verifier_len, bytes)?,
            })
        }
    }
}

impl<F: FieldElement, const SEED_SIZE: usize> SzkQueryShare<F, SEED_SIZE> {
    pub(crate) fn merge_verifiers(
        mut leader_share: SzkQueryShare<F, SEED_SIZE>,
        helper_share: SzkQueryShare<F, SEED_SIZE>,
    ) -> SzkVerifier<F, SEED_SIZE> {
        for (x, y) in leader_share
            .flp_verifier
            .iter_mut()
            .zip(helper_share.flp_verifier)
        {
            *x += y;
        }
        SzkVerifier {
            flp_verifier: leader_share.flp_verifier,
            leader_joint_rand_part_opt: leader_share.joint_rand_part_opt,
            helper_joint_rand_part_opt: helper_share.joint_rand_part_opt,
        }
    }
}

/// Szk query state.
///
/// The state that needs to be stored by an Szk verifier between query() and decide().
pub type SzkQueryState<const SEED_SIZE: usize> = Option<Seed<SEED_SIZE>>;

/// Verifier type for the SZK proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SzkVerifier<F: FieldElement, const SEED_SIZE: usize> {
    flp_verifier: Vec<F>,
    leader_joint_rand_part_opt: Option<Seed<SEED_SIZE>>,
    helper_joint_rand_part_opt: Option<Seed<SEED_SIZE>>,
}

impl<F: FieldElement, const SEED_SIZE: usize> Encode for SzkVerifier<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        encode_fieldvec(&self.flp_verifier, bytes)?;
        if let Some(ref part) = self.leader_joint_rand_part_opt {
            part.encode(bytes)?
        };
        if let Some(ref part) = self.helper_joint_rand_part_opt {
            part.encode(bytes)?
        };
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.flp_verifier.len() * F::ENCODED_SIZE
                + match self.leader_joint_rand_part_opt {
                    Some(ref part) => part.encoded_len()?,
                    None => 0,
                }
                + match self.helper_joint_rand_part_opt {
                    Some(ref part) => part.encoded_len()?,
                    None => 0,
                },
        )
    }
}

impl<F: FieldElement + Decode, const SEED_SIZE: usize> ParameterizedDecode<(bool, usize)>
    for SzkVerifier<F, SEED_SIZE>
{
    fn decode_with_param(
        (requires_joint_rand, verifier_len): &(bool, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if *requires_joint_rand {
            Ok(SzkVerifier {
                flp_verifier: decode_fieldvec(*verifier_len, bytes)?,
                leader_joint_rand_part_opt: Some(Seed::<SEED_SIZE>::decode(bytes)?),
                helper_joint_rand_part_opt: Some(Seed::<SEED_SIZE>::decode(bytes)?),
            })
        } else {
            Ok(SzkVerifier {
                flp_verifier: decode_fieldvec(*verifier_len, bytes)?,
                leader_joint_rand_part_opt: None,
                helper_joint_rand_part_opt: None,
            })
        }
    }
}

/// Main struct encapsulating the shared zero-knowledge functionality. The type
/// T is the underlying FLP proof system. P is the XOF used to derive all random
/// coins (it should be indifferentiable from a random oracle for security.)
#[derive(Clone, Debug)]
pub struct Szk<T, P, const SEED_SIZE: usize>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// The Type representing the specific FLP system used to prove validity of an input.
    pub(crate) typ: T,
    algorithm_id: u32,
    phantom: PhantomData<P>,
}

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
    pub(crate) fn prove(
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
        let (leader_blind_and_helper_joint_rand_part_opt, leader_joint_rand_part_opt, joint_rand) =
            if let Some(leader_seed) = leader_seed_opt {
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

        let (joint_rand, joint_rand_seed, joint_rand_part) = if self.requires_joint_rand() {
            let ((joint_rand_seed, joint_rand), host_joint_rand_part) = match proof_share {
                SzkProofShare::Leader {
                    uncompressed_proof_share: _,
                    leader_blind_and_helper_joint_rand_part_opt,
                } => match leader_blind_and_helper_joint_rand_part_opt {
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
                    leader_joint_rand_part_opt,
                } => match leader_joint_rand_part_opt {
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
                joint_rand_part_opt: joint_rand_part,
                flp_verifier: verifier_share,
            },
            joint_rand_seed,
        ))
    }

    /// Returns true if the verifier message indicates that the input from which
    /// it was generated is valid.
    pub fn decide(
        &self,
        verifier: SzkVerifier<T::Field, SEED_SIZE>,
        query_state: SzkQueryState<SEED_SIZE>,
    ) -> Result<bool, SzkError> {
        // Check if underlying FLP proof validates
        let check_flp_proof = self.typ.decide(&verifier.flp_verifier)?;
        if !check_flp_proof {
            return Ok(false);
        }
        // Check that joint randomness was properly derived from both
        // aggregators' parts
        match (
            query_state,
            verifier.leader_joint_rand_part_opt,
            verifier.helper_joint_rand_part_opt,
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
        let leader_seed_opt = if szk_typ.requires_joint_rand() {
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

        let verifier = SzkQueryShare::merge_verifiers(l_query_share.clone(), h_query_share.clone());
        if let Ok(leader_decision) = szk_typ.decide(verifier.clone(), l_query_state.clone()) {
            assert_eq!(
                leader_decision, valid,
                "Leader incorrectly determined validity",
            );
        } else {
            panic!("Leader failed during decision");
        };
        if let Ok(helper_decision) = szk_typ.decide(verifier.clone(), h_query_state.clone()) {
            assert_eq!(
                helper_decision, valid,
                "Helper incorrectly determined validity",
            );
        } else {
            panic!("Helper failed during decision");
        };

        //test mutated jr seed
        if szk_typ.requires_joint_rand() {
            let joint_rand_seed_opt = Some(Seed::<16>::generate().unwrap());
            if let Ok(leader_decision) = szk_typ.decide(verifier, joint_rand_seed_opt.clone()) {
                assert!(!leader_decision, "Leader accepted wrong jr seed");
            };
        };

        // test mutated verifier
        let mut mutated_query_share = l_query_share.clone();
        for x in mutated_query_share.flp_verifier.iter_mut() {
            *x += T::Field::from(
                <T::Field as FieldElementWithInteger>::Integer::try_from(7).unwrap(),
            );
        }

        let verifier = SzkQueryShare::merge_verifiers(mutated_query_share, h_query_share.clone());

        let leader_decision = szk_typ.decide(verifier, l_query_state.clone()).unwrap();
        assert!(!leader_decision, "Leader validated after proof mutation");

        // test mutated input share
        let mut mutated_input = leader_input_share.clone();
        mutated_input[0] *=
            T::Field::from(<T::Field as FieldElementWithInteger>::Integer::try_from(23).unwrap());
        let (mutated_query_share, mutated_query_state) = szk_typ
            .query(&mutated_input, l_proof_share.clone(), &verify_key, &nonce)
            .unwrap();

        let verifier = SzkQueryShare::merge_verifiers(mutated_query_share, h_query_share.clone());

        if let Ok(leader_decision) = szk_typ.decide(verifier, mutated_query_state) {
            assert!(!leader_decision, "Leader validated after input mutation");
        };

        // test mutated proof share
        let (mut mutated_proof, leader_blind_and_helper_joint_rand_part_opt) = match l_proof_share {
            SzkProofShare::Leader {
                uncompressed_proof_share,
                leader_blind_and_helper_joint_rand_part_opt,
            } => (
                uncompressed_proof_share.clone(),
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
        let (l_query_share, l_query_state) = szk_typ
            .query(
                &leader_input_share,
                mutated_proof_share,
                &verify_key,
                &nonce,
            )
            .unwrap();
        let verifier = SzkQueryShare::merge_verifiers(l_query_share, h_query_share.clone());

        if let Ok(leader_decision) = szk_typ.decide(verifier, l_query_state) {
            assert!(!leader_decision, "Leader validated after proof mutation");
        };
    }

    #[test]
    fn test_sum_proof_share_encode() {
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(5).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(sum, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = Some(Seed::<16>::generate().unwrap());
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_proof_share, _] = szk_typ
            .prove(
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
    fn test_sumvec_proof_share_encode() {
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(sumvec, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = Some(Seed::<16>::generate().unwrap());
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_proof_share, _] = szk_typ
            .prove(
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
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(count, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = Some(Seed::<16>::generate().unwrap());
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_proof_share, _] = szk_typ
            .prove(
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
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(5).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(sum, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = Some(Seed::<16>::generate().unwrap());
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_proof_share, _] = szk_typ
            .prove(
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
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sum = Sum::<Field128>::new(5).unwrap();
        let encoded_measurement = sum.encode_measurement(&9).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(sum, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = Some(Seed::<16>::generate().unwrap());
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [_, h_proof_share] = szk_typ
            .prove(
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
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(count, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = None;
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_proof_share, _] = szk_typ
            .prove(
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
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let count = Count::<Field128>::new();
        let encoded_measurement = count.encode_measurement(&true).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(count, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = None;
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [_, h_proof_share] = szk_typ
            .prove(
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
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(sumvec, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = Some(Seed::<16>::generate().unwrap());
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [l_proof_share, _] = szk_typ
            .prove(
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
        let mut nonce = [0u8; 16];
        thread_rng().fill(&mut nonce[..]);
        let sumvec =
            SumVec::<Field128, ParallelSum<Field128, Mul<Field128>>>::new(5, 3, 3).unwrap();
        let encoded_measurement = sumvec.encode_measurement(&vec![1, 16, 0]).unwrap();
        let algorithm_id = 5;
        let szk_typ = Szk::new_turboshake128(sumvec, algorithm_id);
        let prove_rand_seed = Seed::<16>::generate().unwrap();
        let helper_seed = Seed::<16>::generate().unwrap();
        let leader_seed_opt = Some(Seed::<16>::generate().unwrap());
        let helper_input_share = random_vector(szk_typ.typ.input_len()).unwrap();
        let mut leader_input_share = encoded_measurement.clone().to_owned();
        for (x, y) in leader_input_share.iter_mut().zip(&helper_input_share) {
            *x -= *y;
        }

        let [_, h_proof_share] = szk_typ
            .prove(
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
        let sum = Sum::<Field128>::new(5).unwrap();

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
