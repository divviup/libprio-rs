// SPDX-License-Identifier: MPL-2.0

//! This module constructs  a Verifiable Distributed Aggregation Function (VDAF) from a
//! [`pcp::Value`]. It consists of the following functions:
//!
//! * `upload` is run by the client to generate the input and proof shares.
//!
//! * `verify_start` is run by each aggregator upon receiving its input and proof share. The
//! output is broadcast to the other aggregators.
//!
//! * `verify_finish` is run by each aggregator to decide if the input was valid.
//!
//! In the special case of two aggregators (one leader, one helper), these calls map to HTTP
//! requests as follows (assuming they agree on the query randomness in advance):
//!
//! ```none
//! Client                 Leader                    Helper
//!   | upload()             |                         |
//!   |                      |                         |
//!   |    upload request    |                         |
//!   |--------------------->| verify_start()          |
//!   |                      |                         |
//!   |                      |    aggregate request    |
//!   |                      |------------------------>| verify_start()
//!   |                      |                         |
//!   |                      |                         |
//!   |                      |<------------------------|
//!   |                      | verify_finish()         | verify_finish()
//!   v                      v                         v
//!                        output share              output share
//! ```
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
use std::marker::PhantomData;

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
pub struct UploadMessage<F: FieldElement> {
    /// The input share.
    pub input_share: Share<F>,

    /// The proof share.
    pub proof_share: Share<F>,

    /// The sum of the joint randomness seed shares sent to the other aggregators.
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

/// Run by the client, this generates the sequence of [`UploadMessage`] messages to send to the
/// aggregators.
///
/// # Parameters
///
/// * `input` is the input to be secret shared.
/// * `num_shares` is the number of input shares (i.e., aggregators) to generate.
// TODO(cjpatton) Rename to dist_input to align with spec
pub fn upload<F, V>(
    suite: Suite,
    input: &V,
    num_shares: u8,
) -> Result<Vec<UploadMessage<F>>, VdafError>
where
    F: FieldElement,
    V: Value<F>,
{
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
        let prng: Prng<F> = Prng::from_key_stream(KeyStream::from_key(&helper.input_share));
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
    let prng: Prng<F> = Prng::from_key_stream(KeyStream::from_key(&joint_rand_seed));
    let joint_rand: Vec<F> = prng.take(input.joint_rand_len()).collect();
    let prng: Prng<F> = Prng::generate(suite)?;
    let prove_rand: Vec<F> = prng.take(input.prove_rand_len()).collect();
    let proof = prove(input, &prove_rand, &joint_rand)?;

    // Generate the proof shares and finalize the joint randomness seed hints.
    let proof_len = proof.as_slice().len();
    let mut leader_proof_share = proof.data;
    for helper in helper_shares.iter_mut() {
        let prng: Prng<F> = Prng::from_key_stream(KeyStream::from_key(&helper.proof_share));
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
    out.push(UploadMessage {
        input_share: Share::Leader(leader_input_share),
        proof_share: Share::Leader(leader_proof_share),
        joint_rand_seed_hint: leader_joint_rand_seed_hint,
        blind: leader_blind,
    });

    for helper in helper_shares.into_iter() {
        out.push(UploadMessage {
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
    pub joint_rand_seed_share: Key,
}

/// The initial state of an aggregate upon receiving the [`UploadMessage`] message from the client.
#[derive(Clone, Debug)]
pub struct AggregatorState<F, V>
where
    F: FieldElement,
    V: Value<F>,
{
    phantom: PhantomData<F>,
    input_share: V,
    joint_rand_seed: Key,
}

/// Run by each aggregator, this consumes the [`UploadMessage`] message sent from the client and
/// produces the [`VerifierMessage`] message that will be broadcast to the other aggregators. It also
/// returns the aggregator's state.
///
/// # Parameters
///
/// * `msg` is the message sent by the client. It contains, among other things, an input and proof
/// share.
/// * `param` is used to reconstruct the input share from the raw message.
/// * `query_rand_seed` is used to derive the query randomness shared by all the aggregators.
// TODO(cjpatton) Rename to dist_start to align with spec
pub fn verify_start<F, V>(
    suite: Suite,
    msg: UploadMessage<F>,
    param: V::Param,
    aggregator_id: u8,
    query_rand_seed: &Key,
) -> Result<(AggregatorState<F, V>, VerifierMessage<F>), VdafError>
where
    F: FieldElement,
    V: Value<F>,
{
    if query_rand_seed.suite() != suite {
        return Err(VdafError::Uncategorized(
            "verify_start(): joint rand seed type does not match suite".to_string(),
        ));
    }

    let input_share_data: Vec<F> = Vec::try_from(msg.input_share)?;
    let input_share = V::try_from((param, &input_share_data))?;

    let proof_share_data: Vec<F> = Vec::try_from(msg.proof_share)?;
    let proof_share = Proof::from(proof_share_data);

    // Compute the joint randomness.
    let mut deriver = KeyDeriver::from_key(&msg.blind);
    deriver.update(&[aggregator_id]);
    for x in input_share.as_slice() {
        deriver.update(&(*x).into());
    }
    let joint_rand_seed_share = deriver.finish();

    let mut joint_rand_seed = Key::uninitialized(suite);
    for (j, x) in joint_rand_seed.as_mut_slice().iter_mut().enumerate() {
        *x = msg.joint_rand_seed_hint.as_slice()[j] ^ joint_rand_seed_share.as_slice()[j];
    }

    let prng: Prng<F> = Prng::from_key_stream(KeyStream::from_key(&joint_rand_seed));
    let joint_rand: Vec<F> = prng.take(input_share.joint_rand_len()).collect();

    // Compute the query randomness.
    let prng: Prng<F> = Prng::from_key_stream(KeyStream::from_key(query_rand_seed));
    let query_rand: Vec<F> = prng.take(input_share.query_rand_len()).collect();

    // Run the query-generation algorithm.
    let verifier_share = query(&input_share, &proof_share, &query_rand, &joint_rand)?;

    // Prepare the output state and message.
    let state = AggregatorState {
        phantom: PhantomData,
        input_share,
        joint_rand_seed,
    };

    let out = VerifierMessage {
        verifier_share,
        joint_rand_seed_share,
    };

    Ok((state, out))
}

/// Run by each aggregator, this consumes the [`VerifierMessage`] messages broadcast by all of the
/// aggregators. It returns the aggregator's input share only if the input is valid.
// TODO(cjpatton) Rename to dist_finish to align with spec
pub fn verify_finish<F, V>(
    suite: Suite,
    state: AggregatorState<F, V>,
    msgs: Vec<VerifierMessage<F>>,
) -> Result<V, VdafError>
where
    F: FieldElement,
    V: Value<F>,
{
    if msgs.is_empty() {
        return Err(VdafError::Uncategorized(
            "verify_finish(): expected at least one inbound messages; got none".to_string(),
        ));
    }

    // Combine the verifier messages.
    let mut joint_rand_seed = state.joint_rand_seed;
    let mut verifier_data = vec![F::zero(); msgs[0].verifier_share.as_slice().len()];
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
    if joint_rand_seed != Key::uninitialized(suite) {
        return Err(VdafError::Validity("joint randomness check failed"));
    }

    // Check the proof.
    let verifier = Verifier::from(verifier_data);
    let result = decide(&state.input_share, &verifier)?;
    if !result {
        return Err(VdafError::Validity("proof check failed"));
    }

    Ok(state.input_share)
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
        let uploads = upload(suite, &input, num_shares as u8).unwrap();

        // Aggregators agree on query randomness.
        let query_rand_seed = Key::generate(suite).unwrap();

        // Aggregators receive their proof shares and broadcast their verifier messages.
        let mut states: Vec<AggregatorState<Field64, Boolean<Field64>>> =
            Vec::with_capacity(num_shares);
        let mut verifiers: Vec<VerifierMessage<Field64>> = Vec::with_capacity(num_shares);
        for (aggregator_id, upload) in (0..num_shares as u8).zip(uploads.into_iter()) {
            let (state, verifier) =
                verify_start(suite, upload, (), aggregator_id, &query_rand_seed).unwrap();
            states.push(state);
            verifiers.push(verifier);
        }

        // Aggregators decide whether the input is valid based on the verifier messages.
        let mut output = vec![Field64::zero(); input.as_slice().len()];
        for state in states {
            let output_share = verify_finish(suite, state, verifiers.clone()).unwrap();
            for (x, y) in output.iter_mut().zip(output_share.as_slice()) {
                *x += *y;
            }
        }

        assert_eq!(input.as_slice(), output.as_slice());
    }
}
