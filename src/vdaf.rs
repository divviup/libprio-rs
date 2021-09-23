// SPDX-License-Identifier: MPL-2.0

//! This module constructs a Verifiable Distributed Aggregation Function (VDAF) from a
//! [`pcp::Value`].
//!
//! NOTE: This protocol implemented here is a prototype and has not undergone security analysis.
//! Use at your own risk.

pub mod suite;

use crate::field::FieldElement;
use crate::pcp::types::TypeError;
use crate::pcp::{decide, prove, query, PcpError, Proof, Value, Verifier};
use crate::prng::{Prng, PrngError};
use crate::vdaf::suite::{Key, KeyDeriver, KeyStream, Suite, SuiteError};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

/// Errors emitted by this module.
#[derive(Debug, thiserror::Error)]
pub enum VdafError {
    /// An error occurred.
    #[error("ppm error: {0}")]
    Uncategorized(String),

    /// The distributed input was deemed invalid.
    #[error("ppm error: invalid distributed input: {0}")]
    Validity(&'static str),

    /// PCP error.
    #[error("pcp error: {0}")]
    Pcp(#[from] PcpError),

    /// Type error.
    #[error("type error: {0}")]
    Type(#[from] TypeError),

    /// PRNG error.
    #[error("prng error: {0}")]
    Prng(#[from] PrngError),

    /// Suite error.
    #[error("suite error: {0}")]
    Suite(#[from] SuiteError),
}

/// A share of an input or proof for Prio.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Share<F: FieldElement> {
    /// An uncompressed share, typically sent to the leader.
    Leader(Vec<F>),

    /// A compressed share, typically sent to the helper.
    Helper {
        /// The seed for the pseudorandom generator.
        seed: Key,
        /// The length of the uncompressed share.
        length: usize,
    },
}

impl<F: FieldElement> TryFrom<Share<F>> for Vec<F> {
    type Error = VdafError;

    fn try_from(share: Share<F>) -> Result<Self, VdafError> {
        match share {
            Share::Leader(data) => Ok(data),
            Share::Helper { seed, length } => {
                let prng: Prng<F> = Prng::from_key_stream(KeyStream::from_key(&seed));
                Ok(prng.take(length).collect())
            }
        }
    }
}

/// The message sent by the client to each aggregator. This includes the client's input share and
/// the initial message of the input-validation protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputShareMessage<F: FieldElement> {
    /// The input share.
    pub input_share: Share<F>,

    /// The proof share.
    pub proof_share: Share<F>,

    /// The sum of the joint randomness seed shares sent to the other aggregators.
    //
    // TODO(cjpatton) If `input.joint_rand_len() == 0`, then we don't need to bother with the
    // joint randomness seed at all and make this optional.
    pub joint_rand_seed_hint: Key,

    /// The blinding factor, used to derive the aggregator's joint randomness seed share.
    pub blind: Key,
}

#[derive(Clone)]
struct HelperShare {
    input_share: Key,
    proof_share: Key,
    joint_rand_seed_hint: Key,
    blind: Key,
}

impl HelperShare {
    fn new(suite: Suite) -> Result<Self, VdafError> {
        Ok(HelperShare {
            input_share: Key::generate(suite)?,
            proof_share: Key::generate(suite)?,
            joint_rand_seed_hint: Key::uninitialized(suite),
            blind: Key::generate(suite)?,
        })
    }
}

/// The input-distribution algorithm of the VDAF. Run by the client, this generates the sequence of
/// [`InputShareMessage`] messages to send to the aggregators.
///
/// # Parameters
///
/// * `suite` is the cipher suite used for key derivation.
/// * `input` is the input to be secret shared.
/// * `num_shares` is the number of input shares (i.e., aggregators) to generate.
pub fn dist_input<V: Value>(
    suite: Suite,
    input: &V,
    num_shares: u8,
) -> Result<Vec<InputShareMessage<V::Field>>, VdafError> {
    if num_shares == 0 {
        return Err(VdafError::Uncategorized(format!(
            "upload(): at least one share is required; got {}",
            num_shares
        )));
    }

    let input_len = input.as_slice().len();
    let num_shares = num_shares as usize;

    // Generate the input shares and compute the joint randomness.
    let mut helper_shares = vec![HelperShare::new(suite)?; num_shares - 1];
    let mut leader_input_share = input.as_slice().to_vec();
    let mut joint_rand_seed = Key::uninitialized(suite);
    let mut aggregator_id = 1; // ID of the first helper
    for helper in helper_shares.iter_mut() {
        let mut deriver = KeyDeriver::from_key(&helper.blind);
        deriver.update(&[aggregator_id]);
        let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&helper.input_share));
        for (x, y) in leader_input_share.iter_mut().zip(prng).take(input_len) {
            *x -= y;
            deriver.update(&y.into());
        }

        helper.joint_rand_seed_hint = deriver.finish();
        for (x, y) in joint_rand_seed
            .as_mut_slice()
            .iter_mut()
            .zip(helper.joint_rand_seed_hint.as_slice().iter())
        {
            *x ^= y;
        }

        aggregator_id += 1; // ID of the next helper
    }

    let leader_blind = Key::generate(suite)?;

    let mut deriver = KeyDeriver::from_key(&leader_blind);
    deriver.update(&[0]); // ID of the leader
    for x in leader_input_share.iter() {
        deriver.update(&(*x).into());
    }

    let mut leader_joint_rand_seed_hint = deriver.finish();
    for (x, y) in joint_rand_seed
        .as_mut_slice()
        .iter_mut()
        .zip(leader_joint_rand_seed_hint.as_slice().iter())
    {
        *x ^= y;
    }

    // Run the proof-generation algorithm.
    let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&joint_rand_seed));
    let joint_rand: Vec<V::Field> = prng.take(input.joint_rand_len()).collect();
    let prng: Prng<V::Field> = Prng::generate(suite)?;
    let prove_rand: Vec<V::Field> = prng.take(input.prove_rand_len()).collect();
    let proof = prove(input, &prove_rand, &joint_rand)?;

    // Generate the proof shares and finalize the joint randomness seed hints.
    let proof_len = proof.as_slice().len();
    let mut leader_proof_share = proof.data;
    for helper in helper_shares.iter_mut() {
        let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&helper.proof_share));
        for (x, y) in leader_proof_share.iter_mut().zip(prng).take(proof_len) {
            *x -= y;
        }

        for (x, y) in helper
            .joint_rand_seed_hint
            .as_mut_slice()
            .iter_mut()
            .zip(joint_rand_seed.as_slice().iter())
        {
            *x ^= y;
        }
    }

    for (x, y) in leader_joint_rand_seed_hint
        .as_mut_slice()
        .iter_mut()
        .zip(joint_rand_seed.as_slice().iter())
    {
        *x ^= y;
    }

    // Prepare the output messages.
    let mut out = Vec::with_capacity(num_shares);
    out.push(InputShareMessage {
        input_share: Share::Leader(leader_input_share),
        proof_share: Share::Leader(leader_proof_share),
        joint_rand_seed_hint: leader_joint_rand_seed_hint,
        blind: leader_blind,
    });

    for helper in helper_shares.into_iter() {
        out.push(InputShareMessage {
            input_share: Share::Helper {
                seed: helper.input_share,
                length: input_len,
            },
            proof_share: Share::Helper {
                seed: helper.proof_share,
                length: proof_len,
            },
            joint_rand_seed_hint: helper.joint_rand_seed_hint,
            blind: helper.blind,
        });
    }

    Ok(out)
}

/// The message sent by an aggregator to every other aggregator. This is the final message of the
/// input-validation protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifierMessage<F: FieldElement> {
    /// The aggregator's share of the verifier message.
    pub verifier_share: Verifier<F>,

    /// The aggregator's share of the joint randomness, derived from its input share.
    //
    // TODO(cjpatton) If `input.joint_rand_len() == 0`, then we don't need to bother with the
    // joint randomness seed at all and make this optional.
    pub joint_rand_seed_share: Key,
}

/// The state of each aggregator.
#[derive(Clone, Debug)]
pub enum AggregatorState<V: Value> {
    /// Aggregator is waiting for its input share.
    Ready {
        /// Aggregator ID.
        aggregator_id: u8,

        /// Input parameter, needed to reconstruct a [`Value`] from a vector of field elements.
        input_param: V::Param,

        /// Query randomness seed shared by the aggregators and used to verify the input.
        query_rand_seed: Key,
    },
    /// Aggregator is waiting for the verifier messages.
    Wait {
        /// The input share.
        input_share: V,

        /// The joint randomness seed indicated by the client. The aggregators check that this
        /// indication matches the actual joint randomness seed.
        //
        // TODO(cjpatton) If `input.joint_rand_len() == 0`, then we don't need to bother with the
        // joint randomness seed at all and make this optional.
        joint_rand_seed: Key,
    },
}

/// The distributed state-initialization algorithm. This returns the initial state of each
/// aggregator for evaluating the VDAF.
///
/// # Parameters
///
/// * `suite` is the ciphersuite used for key derivation.
/// * `input_param` is the parameter used to reconstruct a [`Value`] from a vector of field
///   element.
/// * `num_shares` is the number of aggregators.
pub fn dist_init<V: Value>(
    suite: Suite,
    input_param: V::Param,
    num_shares: u8,
) -> Result<Vec<AggregatorState<V>>, VdafError> {
    let query_rand_seed = Key::generate(suite)?;
    Ok((0..num_shares)
        .map(|aggregator_id| AggregatorState::Ready {
            aggregator_id,
            input_param: input_param.clone(),
            query_rand_seed: query_rand_seed.clone(),
        })
        .collect())
}

/// The verify-start algorithm of the VDAF. Run by each aggregator, this consumes the
/// [`InputShareMessage`] message sent from the client and produces the [`VerifierMessage`] message
/// that will be broadcast to the other aggregators.
// TODO(cjpatton) Check for ciphersuite mismatch between `state` and `msg`.
pub fn verify_start<V: Value>(
    state: AggregatorState<V>,
    msg: InputShareMessage<V::Field>,
) -> Result<(AggregatorState<V>, VerifierMessage<V::Field>), VdafError> {
    let (aggregator_id, input_param, query_rand_seed) = match state {
        AggregatorState::Ready {
            aggregator_id,
            input_param,
            query_rand_seed,
        } => (aggregator_id, input_param, query_rand_seed),
        state => {
            return Err(VdafError::Uncategorized(format!(
                "verify_start(): called on incorrect state: {:?}",
                state
            )));
        }
    };

    let input_share_data: Vec<V::Field> = Vec::try_from(msg.input_share)?;
    let mut input_share = V::try_from((input_param, &input_share_data))?;
    input_share.set_leader(aggregator_id == 0);

    let proof_share_data: Vec<V::Field> = Vec::try_from(msg.proof_share)?;
    let proof_share = Proof::from(proof_share_data);

    // Compute the joint randomness.
    let mut deriver = KeyDeriver::from_key(&msg.blind);
    deriver.update(&[aggregator_id]);
    for x in input_share.as_slice() {
        deriver.update(&(*x).into());
    }
    let joint_rand_seed_share = deriver.finish();

    let mut joint_rand_seed = Key::uninitialized(query_rand_seed.suite());
    for (j, x) in joint_rand_seed.as_mut_slice().iter_mut().enumerate() {
        *x = msg.joint_rand_seed_hint.as_slice()[j] ^ joint_rand_seed_share.as_slice()[j];
    }

    let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&joint_rand_seed));
    let joint_rand: Vec<V::Field> = prng.take(input_share.joint_rand_len()).collect();

    // Compute the query randomness.
    let prng: Prng<V::Field> = Prng::from_key_stream(KeyStream::from_key(&query_rand_seed));
    let query_rand: Vec<V::Field> = prng.take(input_share.query_rand_len()).collect();

    // Run the query-generation algorithm.
    let verifier_share = query(&input_share, &proof_share, &query_rand, &joint_rand)?;

    // Prepare the output state and message.
    let state = AggregatorState::Wait {
        input_share,
        joint_rand_seed,
    };

    let out = VerifierMessage {
        verifier_share,
        joint_rand_seed_share,
    };

    Ok((state, out))
}

/// The verify-finish algorithm of the VDAF. Run by each aggregator, this consumes the
/// [`VerifierMessage`] messages broadcast by all of the aggregators and produces the aggregator's
/// input share.
// TODO(cjpatton) Check for ciphersuite mismatch between `state` and `msgs` and among `msgs`.
pub fn verify_finish<V: Value>(
    state: AggregatorState<V>,
    msgs: Vec<VerifierMessage<V::Field>>,
) -> Result<V, VdafError> {
    if msgs.is_empty() {
        return Err(VdafError::Uncategorized(
            "verify_finish(): expected at least one inbound messages; got none".to_string(),
        ));
    }

    let (input_share, mut joint_rand_seed) = match state {
        AggregatorState::Wait {
            input_share,
            joint_rand_seed,
        } => (input_share, joint_rand_seed),
        state => {
            return Err(VdafError::Uncategorized(format!(
                "verify_start(): called on incorrect state: {:?}",
                state
            )));
        }
    };

    // Combine the verifier messages.
    let mut verifier_data = vec![V::Field::zero(); msgs[0].verifier_share.as_slice().len()];
    for msg in msgs {
        if msg.verifier_share.as_slice().len() != verifier_data.len() {
            return Err(VdafError::Uncategorized(format!(
                "verify_finish(): expected verifier share of length {}; got {}",
                verifier_data.len(),
                msg.verifier_share.as_slice().len(),
            )));
        }

        for (x, y) in verifier_data.iter_mut().zip(msg.verifier_share.as_slice()) {
            *x += *y;
        }

        for (x, y) in joint_rand_seed
            .as_mut_slice()
            .iter_mut()
            .zip(msg.joint_rand_seed_share.as_slice())
        {
            *x ^= y;
        }
    }

    // Check that the joint randomness was correct.
    if joint_rand_seed != Key::uninitialized(joint_rand_seed.suite()) {
        return Err(VdafError::Validity("joint randomness check failed"));
    }

    // Check the proof.
    let verifier = Verifier::from(verifier_data);
    let result = decide(&input_share, &verifier)?;
    if !result {
        return Err(VdafError::Validity("proof check failed"));
    }

    Ok(input_share)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::Field64;
    use crate::pcp::types::Boolean;

    #[test]
    fn test_prio() {
        let suite = Suite::Blake3;
        let num_shares = 23;
        let input: Boolean<Field64> = Boolean::new(true);

        // Client runs the input and proof distribution algorithms.
        let input_shares = dist_input(suite, &input, num_shares as u8).unwrap();

        // Aggregators agree on query randomness.
        let states = dist_init(suite, (), num_shares).unwrap();

        // Aggregators receive their proof shares and broadcast their verifier messages.
        let (states, verifiers): (
            Vec<AggregatorState<Boolean<Field64>>>,
            Vec<VerifierMessage<Field64>>,
        ) = states
            .into_iter()
            .zip(input_shares.into_iter())
            .map(|(state, input_share)| verify_start(state, input_share).unwrap())
            .unzip();

        // Aggregators decide whether the input is valid based on the verifier messages.
        let mut output = vec![Field64::zero(); input.as_slice().len()];
        for state in states {
            let output_share = verify_finish(state, verifiers.clone()).unwrap();
            for (x, y) in output.iter_mut().zip(output_share.as_slice()) {
                *x += *y;
            }
        }

        assert_eq!(input.as_slice(), output.as_slice());
    }
}
