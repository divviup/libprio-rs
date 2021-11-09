// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module defines an
//! API for Verifiable Distributed Aggregation Functions (VDAFs) and implements two constructions
//! described in [[VDAF]].
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [BBCG+21]: https://ia.cr/2021/017
//! [VDAF]: https://datatracker.ietf.org/doc/draft-patton-cfrg-vdaf/

use crate::field::FieldElement;
use crate::pcp::PcpError;
use crate::prng::PrngError;
use crate::vdaf::suite::{Key, SuiteError};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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
    Helper(Key),
}

/// The base trait for VDAF schemes. This trait is inherited by traits [`Client`], [`Aggregator`],
/// and [`Collector`], which define the roles of the various parties involved in the execution of
/// the VDAF.
pub trait Vdaf: Clone + Debug {
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
    type InputShare: Clone + Debug + Serialize + DeserializeOwned;

    /// An output share recovered from an input share by an Aggregator.
    type OutputShare: Clone + Debug + Serialize + DeserializeOwned;

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
    /// State of the Aggregator during the Prepare process.
    type PrepareStep: Clone + Debug;

    /// The type of messages exchanged among the Aggregators during the Prepare process.
    type PrepareMessage: Clone + Debug + Serialize + DeserializeOwned;

    /// Begins the Prepare process with the other Aggregators. The result of this process is
    /// the Aggregator's output share.
    fn prepare_init(
        &self,
        verify_param: &Self::VerifyParam,
        agg_param: &Self::AggregationParam,
        nonce: &[u8],
        input_share: &Self::InputShare,
    ) -> Result<Self::PrepareStep, VdafError>;

    /// Compute the next state transition from the current state and the previous round of input
    /// messages.
    fn prepare_step<M: IntoIterator<Item = Self::PrepareMessage>>(
        &self,
        state: Self::PrepareStep,
        inputs: M,
    ) -> PrepareTransition<Self::PrepareStep, Self::PrepareMessage, Self::OutputShare>;

    /// Compute the Aggregator's first message.
    fn prepare_start(
        &self,
        state: Self::PrepareStep,
    ) -> Result<(Self::PrepareStep, Self::PrepareMessage), VdafError> {
        match self.prepare_step(state, std::iter::empty()) {
            PrepareTransition::Continue(new_state, output) => Ok((new_state, output)),
            PrepareTransition::Fail(err) => Err(err),
            PrepareTransition::Finish(_) => Err(VdafError::Uncategorized(
                "start() resulted in early Finish transition".to_string(),
            )),
        }
    }

    /// Compute the Aggregator's next message from the previous round of messages.
    fn prepare_next<M: IntoIterator<Item = Self::PrepareMessage>>(
        &self,
        state: Self::PrepareStep,
        inputs: M,
    ) -> Result<(Self::PrepareStep, Self::PrepareMessage), VdafError> {
        match self.prepare_step(state, inputs) {
            PrepareTransition::Continue(new_state, output) => Ok((new_state, output)),
            PrepareTransition::Fail(err) => Err(err),
            PrepareTransition::Finish(_) => Err(VdafError::Uncategorized(
                "next() resulted in early Finish transition".to_string(),
            )),
        }
    }

    /// Recover the Aggregator's output share.
    fn prepare_finish<M: IntoIterator<Item = Self::PrepareMessage>>(
        &self,
        step: Self::PrepareStep,
        inputs: M,
    ) -> Result<Self::OutputShare, VdafError> {
        match self.prepare_step(step, inputs) {
            PrepareTransition::Continue(_, _) => Err(VdafError::Uncategorized(
                "finish() resulted in Continue transition".to_string(),
            )),
            PrepareTransition::Fail(err) => Err(err),
            PrepareTransition::Finish(output_share) => Ok(output_share),
        }
    }

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

/// A state transition of an Aggregator during the Prepare process.
pub enum PrepareTransition<S, M, O> {
    /// Continue processing.
    Continue(S, M),

    /// Finish processing and return the output share.
    Finish(O),

    /// Fail and return an error.
    Fail(VdafError),
}

/// An aggregate share that can merged with aggregate shares of the same type.
pub trait Aggregatable: Clone + Debug + Serialize + DeserializeOwned {
    /// Update an aggregate share by merging it with another (`agg_share`).
    fn merge(&mut self, agg_share: &Self) -> Result<(), VdafError>;
}

impl<F: FieldElement> Aggregatable for Vec<F> {
    fn merge(&mut self, agg_share: &Vec<F>) -> Result<(), VdafError> {
        if self.len() != agg_share.len() {
            return Err(VdafError::Uncategorized(format!(
                "cannot merge aggregate shares of different lengths (left = {}, right = {})",
                self.len(),
                agg_share.len()
            )));
        }

        for (x, y) in self.iter_mut().zip(agg_share.iter()) {
            *x += *y;
        }

        Ok(())
    }
}

pub mod hits;
pub mod prio3;
pub mod suite;
