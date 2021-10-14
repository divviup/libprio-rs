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

pub mod hits;
pub mod prio3;
pub mod suite;
