// SPDX-License-Identifier: MPL-2.0

//! This module implements core functionality of the PPM protocol for Prio. It consists of the
//! following functions:
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

use crate::field::FieldElement;
use crate::pcp::types::TypeError;
use crate::pcp::{decide, prove, query, PcpError, Proof, Value, Verifier};
use crate::prng::{Prng, PrngError, StreamCipher};
use aes::Aes128Ctr;
use getrandom::getrandom;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

/// Errors emitted by this module.
#[derive(Debug, thiserror::Error)]
pub enum PpmError {
    /// An error occurred.
    #[error("ppm error: {0}")]
    Ppm(String),

    /// The distributed input was deemed invalid.
    #[error("ppm error: invalid distributed input: {0}")]
    Validity(&'static str),

    /// PCP error.
    #[error("ppm error: pcp error: {0}")]
    Pcp(#[from] PcpError),

    /// Type error.
    #[error("ppm error: type error: {0}")]
    Type(#[from] TypeError),

    /// PRNG error.
    #[error("ppm error: prng error: {0}")]
    Prng(#[from] PrngError),

    /// Calling get_random() returned an error.
    #[error("ppm error: getrandom error: {0}")]
    GetRandom(#[from] getrandom::Error),
}

/// A share of an input or proof for Prio.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Share<F: FieldElement> {
    /// An uncompressed share, typically sent to the leader.
    Leader(Vec<F>),

    /// A compressed share, typically sent to the helper.
    Helper {
        /// The seed for the pseudorandom generator.
        seed: Vec<u8>,
        /// The length of the uncompressed share.
        length: usize,
    },
}

impl<F: FieldElement> TryFrom<Share<F>> for Vec<F> {
    type Error = PpmError;

    fn try_from(share: Share<F>) -> Result<Self, PpmError> {
        match share {
            Share::Leader(data) => Ok(data),
            Share::Helper { seed, length } => {
                let prng: Prng<F, Aes128Ctr> = Prng::new_with_seed(&seed)?;
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
    pub joint_rand_seed_hint: Vec<u8>,

    /// The blinding factor, used to derive the aggregator's joint randomness seed share.
    pub blind: Vec<u8>,
}

fn assert_safe() {
    // TODO(cjpatton) Replace this hack with a more consistent API for the Prio ciphersuite.Prng
    // should be constructed with an implementation of the following trait:
    //
    // pub trait Cipher {
    //    // Panics if `seed.len() != StreamCipher::KEY_LEN`.
    //    fn new_stream_cipher(seed: &[u8]) -> StreamCipher;
    //
    //    // Panics if `Hasher::OUT_LEN != StreamCipher::KEY_LEN`.
    //    fn new_hasher() -> Hasher;
    // }
    //
    // pub trait StreamCipher {
    //    const KEY_LEN: usize;
    //
    //    /// Fill `out` with the next `out.len()` bytes of the stream cipher.
    //    fn fill(out: &mut [u8]);
    // }
    //
    // pub trait Hasher<const OUT_LEN: usize> {
    //    const OUT_LEN: usize;
    //
    //    /// Appends `input` to the hash input.
    //    fn update(input: &[u8]);
    //
    //    /// Panics if `out.len() != Self::OUT_LEN`.
    //    fn finalize(out: &mut [u8]);
    // }
    const_assert_eq!(blake3::KEY_LEN, Aes128Ctr::STREAM_CIPHER_SEED_SIZE);
    const_assert_eq!(blake3::OUT_LEN, Aes128Ctr::STREAM_CIPHER_SEED_SIZE);
}

#[derive(Clone)]
struct HelperShare {
    input_share: Vec<u8>,
    proof_share: Vec<u8>,
    joint_rand_seed_hint: Vec<u8>,
    blind: Vec<u8>,
}

impl HelperShare {
    fn new() -> Self {
        HelperShare {
            input_share: vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE],
            proof_share: vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE],
            joint_rand_seed_hint: vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE],
            blind: vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE],
        }
    }
}

/// Run by the client, this generates the sequence of [`UploadMessage`] messages to send to the
/// aggregators.
///
/// # Parameters
///
/// * `input` is the input to be secret shared.
/// * `num_shares` is the number of input shares (i.e., aggregators) to generate.
pub fn upload<F, V>(input: &V, num_shares: u8) -> Result<Vec<UploadMessage<F>>, PpmError>
where
    F: FieldElement,
    V: Value<F>,
{
    assert_safe();

    if num_shares < 2 {
        return Err(PpmError::Ppm(format!(
            "upload(): at least 2 shares are required; got {}",
            num_shares
        )));
    }

    let input_len = input.as_slice().len();
    let num_shares = num_shares as usize;

    // Generate the input shares and compute the joint randomness.
    let mut helper_shares = vec![HelperShare::new(); num_shares - 1];
    let mut leader_input_share = input.as_slice().to_vec();
    let mut joint_rand_seed = vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE];
    let mut aggregator_id = 1; // ID of the first helper
    for helper in helper_shares.iter_mut() {
        getrandom(&mut helper.blind)?;
        getrandom(&mut helper.input_share)?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(&[aggregator_id]);
        hasher.update(&helper.blind);
        let prng: Prng<F, Aes128Ctr> = Prng::new_with_seed(&helper.input_share)?;
        for (x, y) in leader_input_share.iter_mut().zip(prng).take(input_len) {
            *x -= y;
            hasher.update(&y.into());
        }

        helper
            .joint_rand_seed_hint
            .copy_from_slice(hasher.finalize().as_bytes());
        for (x, y) in joint_rand_seed
            .iter_mut()
            .zip(helper.joint_rand_seed_hint.iter())
        {
            *x ^= y;
        }

        aggregator_id += 1; // ID of the next helper
    }

    let mut leader_blind = vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE];
    getrandom(&mut leader_blind)?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(&[0]); // ID of the leader
    hasher.update(&leader_blind);
    for x in leader_input_share.iter() {
        hasher.update(&(*x).into());
    }

    let mut leader_joint_rand_seed_hint = vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE];
    leader_joint_rand_seed_hint.copy_from_slice(hasher.finalize().as_bytes());
    for (x, y) in joint_rand_seed
        .iter_mut()
        .zip(leader_joint_rand_seed_hint.iter())
    {
        *x ^= y;
    }

    // Run the proof-generation algorithm.
    let prng: Prng<F, Aes128Ctr> = Prng::new_with_seed(&joint_rand_seed)?;
    let joint_rand: Vec<F> = prng.take(input.joint_rand_len()).collect();
    let prng: Prng<F, Aes128Ctr> = Prng::new()?;
    let prove_rand: Vec<F> = prng.take(input.prove_rand_len()).collect();
    let proof = prove(input, &prove_rand, &joint_rand)?;

    // Generate the proof shares and finalize the joint randomness seed hints.
    let proof_len = proof.as_slice().len();
    let mut leader_proof_share = proof.data;
    for helper in helper_shares.iter_mut() {
        getrandom(&mut helper.proof_share)?;

        let prng: Prng<F, Aes128Ctr> = Prng::new_with_seed(&helper.proof_share)?;
        for (x, y) in leader_proof_share.iter_mut().zip(prng).take(proof_len) {
            *x -= y;
        }

        for (x, y) in helper
            .joint_rand_seed_hint
            .iter_mut()
            .zip(joint_rand_seed.iter())
        {
            *x ^= y;
        }
    }

    for (x, y) in leader_joint_rand_seed_hint
        .iter_mut()
        .zip(joint_rand_seed.iter())
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
    pub joint_rand_seed_share: Vec<u8>,
}

/// The initial state of an aggregate upon receiving the [`UploadMessage`] message from the client.
#[derive(Clone, Debug)]
pub struct AggregatorState<F, V>
where
    F: FieldElement,
    V: Value<F>,
{
    input_share: V,
    verifier_share: Verifier<F>,
    joint_rand_seed_hint: Vec<u8>,
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
pub fn verify_start<F, V>(
    msg: UploadMessage<F>,
    param: V::Param,
    aggregator_id: u8,
    query_rand_seed: &[u8],
) -> Result<(AggregatorState<F, V>, VerifierMessage<F>), PpmError>
where
    F: FieldElement,
    V: Value<F>,
{
    assert_safe();

    let input_share_data: Vec<F> = Vec::try_from(msg.input_share)?;
    let input_share = V::try_from((param, &input_share_data))?;

    let proof_share_data: Vec<F> = Vec::try_from(msg.proof_share)?;
    let proof_share = Proof::from(proof_share_data);

    // Compute the joint randomness.
    let mut joint_rand_seed_share = vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE];
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[aggregator_id]);
    hasher.update(&msg.blind);
    for x in input_share.as_slice() {
        hasher.update(&(*x).into());
    }
    joint_rand_seed_share.copy_from_slice(hasher.finalize().as_bytes());

    let mut joint_rand_seed = vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE];
    for (j, x) in joint_rand_seed.iter_mut().enumerate() {
        *x = msg.joint_rand_seed_hint[j] ^ joint_rand_seed_share[j];
    }

    let prng: Prng<F, Aes128Ctr> = Prng::new_with_seed(&joint_rand_seed)?;
    let joint_rand: Vec<F> = prng.take(input_share.joint_rand_len()).collect();

    // Compute the query randomness.
    let prng: Prng<F, Aes128Ctr> = Prng::new_with_seed(query_rand_seed)?;
    let query_rand: Vec<F> = prng.take(input_share.query_rand_len()).collect();

    // Run the query-generation algorithm.
    let verifier_share = query(&input_share, &proof_share, &query_rand, &joint_rand)?;

    // Prepare the output state and message.
    let state = AggregatorState {
        input_share,
        verifier_share: verifier_share.clone(),
        joint_rand_seed_hint: msg.joint_rand_seed_hint,
    };

    let out = VerifierMessage {
        verifier_share,
        joint_rand_seed_share,
    };

    Ok((state, out))
}

/// Run by each aggregator, this consumes the [`VerifierMessage`] messages broadcast by each of the
/// other aggregators. It returns the aggregator's input share only if the input is valid.
pub fn verify_finish<F, V>(
    state: AggregatorState<F, V>,
    msgs: Vec<VerifierMessage<F>>,
) -> Result<V, PpmError>
where
    F: FieldElement,
    V: Value<F>,
{
    assert_safe();

    // Combine the verifier messages.
    let mut verifier_data = state.verifier_share.data;
    let mut joint_rand_seed_check = vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE];
    for msg in msgs {
        if msg.verifier_share.as_slice().len() != verifier_data.len() {
            return Err(PpmError::Ppm(format!(
                "verify_finish(): expected verifier share of length {}; got {}",
                verifier_data.len(),
                msg.verifier_share.as_slice().len(),
            )));
        }

        for (x, y) in verifier_data.iter_mut().zip(msg.verifier_share.data) {
            *x += y;
        }

        for (x, y) in joint_rand_seed_check
            .iter_mut()
            .zip(msg.joint_rand_seed_share)
        {
            *x ^= y;
        }
    }

    // Check that the joint randomness was correct.
    if joint_rand_seed_check != state.joint_rand_seed_hint {
        return Err(PpmError::Validity("joint randomness check failed"));
    }

    // Check the proof.
    let verifier = Verifier::from(verifier_data);
    let result = decide(&state.input_share, &verifier)?;
    if !result {
        return Err(PpmError::Validity("proof check failed"));
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
        let num_shares = 23;
        let input: Boolean<Field64> = Boolean::new(true);

        // Client runs the input and proof distribution algorithms.
        let uploads = upload(&input, num_shares as u8).unwrap();

        // Aggregators agree on query randomness.
        let mut query_rand_seed = vec![0; Aes128Ctr::STREAM_CIPHER_SEED_SIZE];
        getrandom(&mut query_rand_seed).unwrap();

        // Aggregators receive their proof shares and broadcast their verifier messages.
        let mut states: Vec<AggregatorState<Field64, Boolean<Field64>>> =
            Vec::with_capacity(num_shares);
        let mut verifiers: Vec<VerifierMessage<Field64>> = Vec::with_capacity(num_shares);
        for (aggregator_id, upload) in (0..num_shares as u8).zip(uploads.into_iter()) {
            let (state, verifier) =
                verify_start(upload, (), aggregator_id, &query_rand_seed).unwrap();
            states.push(state);
            verifiers.push(verifier);
        }

        // Aggregators decide whether the input is valid based on the verifier messages.
        let mut output = vec![Field64::zero(); input.as_slice().len()];
        for (i, state) in states.into_iter().enumerate() {
            // Gather the messages sent to aggregator i.
            let mut verifiers_for_aggregator: Vec<VerifierMessage<Field64>> =
                Vec::with_capacity(num_shares - 1);
            for (j, verifier) in verifiers.iter().enumerate() {
                if i != j {
                    verifiers_for_aggregator.push(verifier.clone());
                }
            }

            // Run aggregator i.
            let output_share = verify_finish(state, verifiers_for_aggregator).unwrap();
            for (x, y) in output.iter_mut().zip(output_share.as_slice()) {
                *x += *y;
            }
        }

        assert_eq!(input.as_slice(), output.as_slice());
    }
}
