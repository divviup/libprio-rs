// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module
//! implements the prio3 Verifiable Distributed Aggregation Function specified in
//! [[VDAF](https://cjpatton.github.io/vdaf/draft-patton-cfrg-vdaf.html)]. It is constructed from a
//! [`crate::pcp::Value`].

pub mod prio3;
pub mod suite;
