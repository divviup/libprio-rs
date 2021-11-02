// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module defines an
//! API for Verifiable Distributed Aggregation Functions (VDAFs) and implements two constructions
//! described in [[VDAF]].
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [BBCG+21]: https://ia.cr/2021/017
//! [VDAF]: https://datatracker.ietf.org/doc/draft-patton-cfrg-vdaf/

use std::convert::TryFrom;

use crate::field::FieldElement;
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

    /// PCP error.
    #[error("pcp error: {0}")]
    Pcp(#[from] PcpError),

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
        //
        // TODO(cjpatton) Avoid encoding the length of the uncompressed share on the wire. The VDAF
        // should always provide a way to compute this.
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

/// The base trait for VDAF schemes. This trait is inherited by traits [`Client`], [`Aggregator`],
/// and [`Collector`], which define the roles of the various parties involved in the execution of
/// the VDAF.
pub trait Vdaf {
    /// The type of Client measurement to be aggregated.
    type Measurement;

    /// The aggregate result of the VDAF execution.
    type AggregateResult;

    /// The aggregation parameter, used by the Aggregators to map their input shares to output
    /// shares.
    type AggregationParam;

    /// The public parameter used by Clients to shard their measurement into input shares.
    type PublicParam;

    /// A verification parameter, used by an Aggregator in the Prepare process to ensure that the
    /// Aggregators have recovered valid output shares.
    type VerifyParam;

    /// An input share sent by a Client.
    type InputShare;

    /// An output share recovered from an input share by an Aggregator.
    type OutputShare;

    /// An Aggregator's share of the aggregate result.
    type AggregateShare: Aggregatable;

    /// Generates the long-lived parameters used by the Clients and Aggregators.
    fn setup(&self) -> Result<(Self::PublicParam, Vec<Self::VerifyParam>), VdafError>;

    /// The number of Aggregators. The Client generates as many input shares as there are
    /// Aggregators.
    fn num_aggregators(&self) -> usize;
}

/// The Client's role in the execution of a VDAF.
pub trait Client: Vdaf {
    /// Shards a measurement into a sequence of input shares, one for each Aggregator.
    fn shard(
        &self,
        public_param: &Self::PublicParam,
        measurement: &Self::Measurement,
    ) -> Result<Vec<Self::InputShare>, VdafError>;
}

/// The Aggregator's role in the execution of a VDAF.
pub trait Aggregator: Vdaf {
    /// Initial state of the Aggregator during the Prepare process.
    type PrepareInit: PrepareStep<Self::OutputShare>;

    /// Begins the Prepare process with the other Aggregators. The result of this process is
    /// the Aggregator's output share.
    fn prepare(
        &self,
        verify_param: &Self::VerifyParam,
        agg_param: &Self::AggregationParam,
        nonce: &[u8],
        input_share: &Self::InputShare,
    ) -> Result<Self::PrepareInit, VdafError>;

    /// Aggregates a sequence of output shares into an aggregate share.
    fn aggregate<M: IntoIterator<Item = Self::OutputShare>>(
        &self,
        agg_param: &Self::AggregationParam,
        output_shares: M,
    ) -> Result<Self::AggregateShare, VdafError>;
}

/// The Collector's role in the execution of a VDAF.
pub trait Collector: Vdaf {
    /// Combines aggregate shares into the aggregate result.
    fn unshard<M: IntoIterator<Item = Self::AggregateShare>>(
        &self,
        agg_param: &Self::AggregationParam,
        agg_shares: M,
    ) -> Result<Self::AggregateResult, VdafError>;
}

/// Intermediate state of an Aggregator during the Prepare process.
pub trait PrepareStep<O>: Sized {
    /// The type of input messages.
    type Input;

    /// The type of output message.
    type Output;

    /// Compute the next state transition from the current state and the previous round of input
    /// messages.
    fn step<M: IntoIterator<Item = Self::Input>>(self, inputs: M) -> PrepareTransition<Self, O>;

    /// Compute the Aggregator's first message.
    fn start(self) -> Result<(Self, Self::Output), VdafError> {
        match self.step(std::iter::empty()) {
            PrepareTransition::Continue(new_state, output) => Ok((new_state, output)),
            PrepareTransition::Fail(err) => Err(err),
            PrepareTransition::Finish(_) => Err(VdafError::Uncategorized(
                "start() resulted in early Finish transition".to_string(),
            )),
        }
    }

    /// Compute the Aggregator's next message from the previous round of messages.
    fn next<M: IntoIterator<Item = Self::Input>>(
        self,
        inputs: M,
    ) -> Result<(Self, Self::Output), VdafError> {
        match self.step(inputs) {
            PrepareTransition::Continue(new_state, output) => Ok((new_state, output)),
            PrepareTransition::Fail(err) => Err(err),
            PrepareTransition::Finish(_) => Err(VdafError::Uncategorized(
                "next() resulted in early Finish transition".to_string(),
            )),
        }
    }

    /// Recover the Aggregator's output share.
    fn finish<M: IntoIterator<Item = Self::Input>>(self, inputs: M) -> Result<O, VdafError> {
        match self.step(inputs) {
            PrepareTransition::Continue(_, _) => Err(VdafError::Uncategorized(
                "finish() resulted in Continue transition".to_string(),
            )),
            PrepareTransition::Fail(err) => Err(err),
            PrepareTransition::Finish(output_share) => Ok(output_share),
        }
    }
}

/// A state transition of an Aggregator during the Prepare process.
pub enum PrepareTransition<S: PrepareStep<O>, O> {
    /// Continue processing.
    Continue(S, S::Output),

    /// Finish processing and return the output share.
    Finish(O),

    /// Fail and return an error.
    Fail(VdafError),
}

/// An aggregate share that can merged with aggregate shares of the same type.
pub trait Aggregatable {
    /// Update an aggregate share by merging it with another (`agg_share`).
    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError>;
}

pub mod hits;
pub mod prio3;
pub mod suite;
