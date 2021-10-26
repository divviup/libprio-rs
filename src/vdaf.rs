// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module
//! implements the prio3 Verifiable Distributed Aggregation Function specified in
//! [[VDAF](https://cjpatton.github.io/vdaf/draft-patton-cfrg-vdaf.html)]. It is constructed from a
//! [`crate::pcp::Value`].

use std::convert::TryFrom;

use crate::field::FieldElement;
use crate::pcp::types::TypeError;
use crate::pcp::PcpError;
use crate::prng::{Prng, PrngError};
use crate::vdaf::suite::{Key, KeyStream, SuiteError};
use serde::{Deserialize, Serialize};

/// Errors emitted by this module.
#[derive(Debug, thiserror::Error)]
pub enum VdafError {
    /// An error occurred.
    #[error("vdaf error: {0}")]
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

/// An additive share of a vector of field elements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Share<F> {
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

/// The Client's role in the execution of a VDAF.
pub trait Client {
    /// The raw measurement being aggregated.
    type Measurement;

    /// The type of input shares sento by the Client.
    type InputShare;

    /// Shards a measurement into a sequence of input shares, one for each Aggregator.
    fn shard(measurement: &Self::Measurement) -> Result<Vec<Self::InputShare>, VdafError>;
}

/// The Aggregator's role in the execution of a VDAF.
pub trait Aggregator {
    /// The number of rounds in the Verify process.
    const ROUNDS: usize;

    /// The type of input share received from each Client.
    type InputShare;

    /// The type of output share recovered from the input share.
    type OutputShare;

    /// The verification parameter, used to verify that the recovered output share is valid.
    type VerifyParam;

    /// Initial state of the Verify process.
    type VerifyStart: VerifyStart;

    /// The aggregation parameter, used to map the input share to an output share and to aggregate
    /// output shares.
    type AggregateParam;

    /// The Aggregator's share of the aggregate result.
    type AggregateShare;

    /// Begins the Verify process with the other Aggregators. The result of this process is
    /// the Aggregator's output share.
    fn verify(
        verify_param: &Self::VerifyParam,
        agg_param: &Self::AggregateParam,
        nonce: &[u8],
        input_share: &Self::InputShare,
    ) -> Result<Self::VerifyStart, VdafError>;

    /// Aggregates a sequence of output shares.
    fn aggregate<M: IntoIterator<Item = Self::OutputShare>>(
        agg_param: &Self::AggregateParam,
        output_shares: M,
    ) -> Result<Self::AggregateShare, VdafError>;
}

/// The state of an Aggregator during the Verify process.
pub trait VerifyStep {
    /// The type of output from this state. If this is the terminal state, then this is the
    /// Aggregator's output share.
    type Output;
}

/// The initial state of an Aggregator during the Verify process.
pub trait VerifyStart: VerifyStep {
    /// The Aggregator's next state.
    type Next: VerifyStep;

    /// Returns the Aggregator's output and next state.
    fn start(self) -> (Self::Next, Self::Output);
}

/// An intermediate state of an Aggregator during the Verify process.
pub trait VerifyNext: VerifyStep {
    /// The Aggregator's next state.
    type Next: VerifyStep;

    /// The output type from the previous round.
    type Input;

    fn next<M: IntoIterator<Item = Self::Input>>(
        self,
        inputs: M,
    ) -> Result<(Self::Next, Self::Output), VdafError>;
}

/// The terminal state of an Aggregator during the Verify process.
pub trait VerifyFinish: VerifyStep {
    /// The output type from the previous round.
    type Input;

    /// Returns the Aggregator's output share.
    fn finish<M: IntoIterator<Item = Self::Input>>(
        self,
        inputs: M,
    ) -> Result<Self::Output, VdafError>;
}

/// The Collector's role in the execution of a VDAF.
pub trait Collector {
    /// The aggregation parameter used to compute the aggregate result.
    type AggregateParam;

    /// A share of the aggregate result sent by each Aggregator.
    type AggregateShare;

    /// The aggregate result.
    type Aggregate;

    /// Combines the aggregate shares sent by the Aggregators into the aggregate result.
    fn unshard<M: IntoIterator<Item = Self::AggregateShare>>(
        agg_param: &Self::AggregateParam,
        agg_shares: M,
    ) -> Result<Self::Aggregate, VdafError>;
}

pub mod hits;
pub mod prio3;
pub mod suite;
