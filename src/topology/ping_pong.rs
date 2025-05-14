// SPDX-License-Identifier: MPL-2.0

//! Implements the Ping-Pong Topology described in [VDAF]. This topology assumes there are exactly
//! two aggregators, designated "Leader" and "Helper". This topology is required for implementing
//! the [Distributed Aggregation Protocol][DAP].
//!
//! [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf#section-5.7.1
//! [DAP]: https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap

use crate::{
    codec::{decode_u32_items, encode_u32_items, CodecError, Decode, Encode, ParameterizedDecode},
    vdaf::{Aggregator, PrepareTransition, VdafError},
};
use std::fmt::Debug;

/// Errors emitted by this module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PingPongError {
    /// Error running prepare_init
    #[error("vdaf.prepare_init: {0}")]
    VdafPrepareInit(VdafError),

    /// Error running prepare_shares_to_prepare_message
    #[error("vdaf.prepare_shares_to_prepare_message {0}")]
    VdafPrepareSharesToPrepareMessage(VdafError),

    /// Error running prepare_next
    #[error("vdaf.prepare_next {0}")]
    VdafPrepareNext(VdafError),

    /// Error encoding or decoding a prepare share
    #[error("encode/decode prep share {0}")]
    CodecPrepShare(CodecError),

    /// Error encoding or decoding a prepare message
    #[error("encode/decode prep message {0}")]
    CodecPrepMessage(CodecError),

    /// Message from peer indicates it is in an unexpected state
    #[error("peer message mismatch: message is {found} expected {expected}")]
    PeerMessageMismatch {
        /// The state in the message from the peer.
        found: &'static str,
        /// The message expected from the peer.
        expected: &'static str,
    },
}

/// Corresponds to `struct Message` in [VDAF's Ping-Pong Topology][VDAF]. All of the fields of the
/// variants are opaque byte buffers. This is because the ping-pong routines take responsibility for
/// decoding preparation shares and messages, which usually requires having the preparation state.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf#section-5.7.1
#[derive(Clone, PartialEq, Eq)]
pub enum PingPongMessage {
    /// Corresponds to MessageType.continued.
    Continued {
        /// The current round's preparation message.
        prepare_message: Vec<u8>,
        /// The next round's preparation share.
        prepare_share: Vec<u8>,
    },
    /// Corresponds to MessageType.finish.
    Finish {
        /// The current round's preparation message.
        prepare_message: Vec<u8>,
    },
}

impl PingPongMessage {
    fn variant(&self) -> &'static str {
        match self {
            Self::Continued { .. } => "Continued",
            Self::Finish { .. } => "Finish",
        }
    }
}

impl Debug for PingPongMessage {
    // We want `PingPongMessage` to implement `Debug`, but we don't want that impl to print out
    // prepare shares or messages, because (1) their contents are sensitive and (2) their contents
    // are long and not intelligible to humans. For both reasons they generally shouldn't get
    // logged. Normally, we'd use the `derivative` crate to customize a derived `Debug`, but that
    // crate has not been audited (in the `cargo vet` sense) so we can't use it here unless we audit
    // 8,000+ lines of proc macros.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(self.variant()).finish()
    }
}

impl Encode for PingPongMessage {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        // The encoding includes an implicit discriminator byte, called MessageType in the VDAF
        // spec.
        match self {
            Self::Continued {
                prepare_message,
                prepare_share,
            } => {
                0u8.encode(bytes)?;
                encode_u32_items(bytes, &(), prepare_message)?;
                encode_u32_items(bytes, &(), prepare_share)?;
            }
            Self::Finish { prepare_message } => {
                1u8.encode(bytes)?;
                encode_u32_items(bytes, &(), prepare_message)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            Self::Continued {
                prepare_message,
                prepare_share,
            } => Some(1 + 4 + prepare_message.len() + 4 + prepare_share.len()),
            Self::Finish { prepare_message } => Some(1 + 4 + prepare_message.len()),
        }
    }
}

impl Decode for PingPongMessage {
    fn decode(bytes: &mut std::io::Cursor<&[u8]>) -> Result<Self, CodecError> {
        let message_type = u8::decode(bytes)?;
        Ok(match message_type {
            0 => {
                let prepare_message = decode_u32_items(&(), bytes)?;
                let prepare_share = decode_u32_items(&(), bytes)?;
                Self::Continued {
                    prepare_message,
                    prepare_share,
                }
            }
            1 => {
                let prepare_message = decode_u32_items(&(), bytes)?;
                Self::Finish { prepare_message }
            }
            _ => return Err(CodecError::UnexpectedValue),
        })
    }
}

/// Corresponds to the `State` enumeration implicitly defined in [VDAF's Ping-Pong Topology][VDAF].
/// VDAF describes `Start` and `Rejected` states, but the `Start` state is never instantiated in
/// code, and the `Rejected` state is represented as `std::result::Result::Err`, so this enum does
/// not include those variants.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf#section-5.7.1
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PingPongState<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
> {
    /// Preparation of the report will continue.
    Continued {
        /// The state to which the aggregator has advanced.
        prepare_state: A::PrepareState,
        /// A message which should be transmitted to the peer aggregator to continue preparing the
        /// report.
        message: PingPongMessage,
    },
    /// Preparation of the report is finished and has yielded an output share, but the peer
    /// aggregator is not finished.
    Finished {
        /// The output share this aggregator prepared.
        output_share: A::OutputShare,
        /// A message which should be transmitted to the peer aggregator so it can finish preparing
        /// the report.
        message: PingPongMessage,
    },
    /// The peer aggregator has finished and this aggregator has now finished preparing the report.
    /// No message need be transmitted to the peer.
    FinishedNoMessage {
        /// The output share this aggregator prepared.
        output_share: A::OutputShare,
    },
}

/// Extension trait on [`crate::vdaf::Aggregator`] which adds the [VDAF Ping-Pong Topology][VDAF].
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf#section-5.7.1
pub trait PingPongTopology<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>:
    Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    /// Specialization of [`PingPongState`] for this VDAF.
    type PingPongState;

    /// Initialize leader state using the leader's input share. Corresponds to
    /// `ping_pong_leader_init` in [VDAF].
    ///
    /// If successful, the returned `Self::PrepareShare` should be transmitted to the helper. The
    /// returned `Self::PrepareState` should be used by the leader along with the next
    /// [`PingPongMessage`] received from the Helper as input to [`Self::leader_continued`] to
    /// advance to the next round.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf#section-5.7.1
    fn leader_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::PrepareState, Self::PrepareShare), PingPongError>;

    /// Initialize helper state using the helper's input share and the leader's first prepare share.
    /// Corresponds to `ping_pong_helper_init` in [VDAF].
    ///
    /// The returned [`PrepareTransition`] will either be `PrepareTransition::Continue`, in which
    /// case the enclosed prepare state should be used by the helper along with the next
    /// `PingPongMessage` received from the leader as input to [`Self::helper_continued`] to advance
    /// to the next round, or it will be `PrepareTransition::Finish`, in which case preparation is
    /// finished and the output share may be accumulated.
    ///
    /// In either case, the returned [`PingPongMessage`] should be transmitted to the leader.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf#section-5.7.1
    #[allow(clippy::too_many_arguments)]
    fn helper_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
        leader_prepare_share: &Self::PrepareShare,
    ) -> Result<
        (
            PrepareTransition<Self, VERIFY_KEY_SIZE, NONCE_SIZE>,
            PingPongMessage,
        ),
        PingPongError,
    >;

    /// Continue preparation based on the leader's current state and an incoming [`PingPongMessage`]
    /// from the helper. Corresponds to `ping_pong_leader_continued` in [VDAF].
    ///
    /// If successful, the returned [`PingPongState`] will either be:
    ///
    /// - `PingPongState::Continued { prepare_state, message }`: `message` should be transmitted to
    ///   the helper. `prepare_state` should be used along with the next `PingPongMessage` received
    ///   from the helper as input to [`Self::leader_continued`] to advance to the next round.
    ///
    /// - `PingPongState::Finished { output_share, message }`: `message` should be transmitted to
    ///   the helper. `output_share` may be accumulated. Preparation is finished.
    ///
    /// - `PingPongState::FinishedNoMessage { output_share }`: `output_share` may be accumulated.
    ///   Preparation is finished. No message needs to be sent to the helper.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf#section-5.7.1
    fn leader_continued(
        &self,
        ctx: &[u8],
        leader_prepare_state: Self::PrepareState,
        aggregation_param: &Self::AggregationParam,
        helper_message: &PingPongMessage,
    ) -> Result<Self::PingPongState, PingPongError>;

    /// Continue preparation based on the helper's current state and an incoming [`PingPongMessage`]
    /// from the leader. Corresponds to `ping_pong_helper_contnued` in [VDAF].
    ///
    /// If successful, the returned [`PingPongState`] will either be:
    ///
    /// - `PingPongState::Continued { prepare_state, message }`: `message` should be transmitted to
    ///   the leader. `prepare_state` should be used along with the next `PingPongMessage` received
    ///   from the leader as input to [`Self::helper_continued`] to advance to the next round.
    ///
    /// - `PingPongState::Finished { output_share, message }`: `message` should be transmitted to
    ///   the leader. `output_share` may be accumulated. Preparation is finished.
    ///
    /// - `PingPongState::FinishedNoMessage { output_share }`: `output_share` may be accumulated.
    ///   Preparation is finished. No message needs to be sent to the leader.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf#section-5.7.1
    fn helper_continued(
        &self,
        ctx: &[u8],
        helper_prepare_state: Self::PrepareState,
        aggregation_param: &Self::AggregationParam,
        leader_message: &PingPongMessage,
    ) -> Result<Self::PingPongState, PingPongError>;
}

/// Private interfaces for implementing ping-pong
trait PingPongTopologyPrivate<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>:
    PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    fn continued(
        &self,
        ctx: &[u8],
        is_leader: bool,
        host_prepare_state: Self::PrepareState,
        aggregation_param: &Self::AggregationParam,
        peer_message: &PingPongMessage,
    ) -> Result<Self::PingPongState, PingPongError>;
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A>
    PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE> for A
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
{
    type PingPongState = PingPongState<VERIFY_KEY_SIZE, NONCE_SIZE, Self>;

    fn leader_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::PrepareState, Self::PrepareShare), PingPongError> {
        self.prepare_init(
            verify_key,
            ctx,
            /* Leader */ 0,
            aggregation_param,
            nonce,
            public_share,
            input_share,
        )
        .map_err(PingPongError::VdafPrepareInit)
    }

    fn helper_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
        leader_prepare_share: &Self::PrepareShare,
    ) -> Result<
        (
            PrepareTransition<Self, VERIFY_KEY_SIZE, NONCE_SIZE>,
            PingPongMessage,
        ),
        PingPongError,
    > {
        let (prepare_state, prepare_share) = self
            .prepare_init(
                verify_key,
                ctx,
                /* Helper */ 1,
                aggregation_param,
                nonce,
                public_share,
                input_share,
            )
            .map_err(PingPongError::VdafPrepareInit)?;

        let current_prepare_message = self
            .prepare_shares_to_prepare_message(
                ctx,
                aggregation_param,
                [leader_prepare_share.clone(), prepare_share],
            )
            .map_err(PingPongError::VdafPrepareSharesToPrepareMessage)?;

        let prepare_message = current_prepare_message
            .get_encoded()
            .map_err(PingPongError::CodecPrepMessage)?;

        self.prepare_next(ctx, prepare_state, current_prepare_message)
            .map_err(PingPongError::VdafPrepareNext)
            .and_then(|transition| match transition {
                PrepareTransition::Continue(_, ref prepare_share) => Ok((
                    transition.clone(),
                    PingPongMessage::Continued {
                        prepare_message,
                        prepare_share: prepare_share
                            .get_encoded()
                            .map_err(PingPongError::CodecPrepShare)?,
                    },
                )),
                PrepareTransition::Finish(_) => {
                    Ok((transition, PingPongMessage::Finish { prepare_message }))
                }
            })
    }

    fn leader_continued(
        &self,
        ctx: &[u8],
        leader_state: Self::PrepareState,
        agg_param: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::PingPongState, PingPongError> {
        self.continued(ctx, true, leader_state, agg_param, inbound)
    }

    fn helper_continued(
        &self,
        ctx: &[u8],
        helper_state: Self::PrepareState,
        agg_param: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::PingPongState, PingPongError> {
        self.continued(ctx, false, helper_state, agg_param, inbound)
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A>
    PingPongTopologyPrivate<VERIFY_KEY_SIZE, NONCE_SIZE> for A
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
{
    fn continued(
        &self,
        ctx: &[u8],
        is_leader: bool,
        host_prepare_state: Self::PrepareState,
        agg_param: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::PingPongState, PingPongError> {
        let (prepare_message, next_peer_prepare_share) = match inbound {
            PingPongMessage::Continued {
                prepare_message,
                prepare_share,
            } => (prepare_message, Some(prepare_share)),
            PingPongMessage::Finish { prepare_message } => (prepare_message, None),
        };

        let prepare_message =
            Self::PrepareMessage::get_decoded_with_param(&host_prepare_state, prepare_message)
                .map_err(PingPongError::CodecPrepMessage)?;
        let host_prepare_transition = self
            .prepare_next(ctx, host_prepare_state, prepare_message)
            .map_err(PingPongError::VdafPrepareNext)?;

        match (host_prepare_transition, next_peer_prepare_share) {
            (
                PrepareTransition::Continue(next_prepare_state, next_host_prepare_share),
                Some(next_peer_prepare_share),
            ) => {
                let next_peer_prepare_share = Self::PrepareShare::get_decoded_with_param(
                    &next_prepare_state,
                    next_peer_prepare_share,
                )
                .map_err(PingPongError::CodecPrepShare)?;
                let mut prepare_shares = [next_peer_prepare_share, next_host_prepare_share];
                if is_leader {
                    prepare_shares.reverse();
                }
                let current_prepare_message = self
                    .prepare_shares_to_prepare_message(ctx, agg_param, prepare_shares)
                    .map_err(PingPongError::VdafPrepareSharesToPrepareMessage)?;

                self.prepare_next(ctx, next_prepare_state, current_prepare_message.clone())
                    .map_err(PingPongError::VdafPrepareNext)
                    .and_then(|transition| match transition {
                        PrepareTransition::Continue(prepare_state, prepare_share) => {
                            Ok(PingPongState::Continued {
                                prepare_state,
                                message: PingPongMessage::Continued {
                                    prepare_message: current_prepare_message
                                        .get_encoded()
                                        .map_err(PingPongError::CodecPrepMessage)?,
                                    prepare_share: prepare_share
                                        .get_encoded()
                                        .map_err(PingPongError::CodecPrepShare)?,
                                },
                            })
                        }
                        PrepareTransition::Finish(output_share) => Ok(PingPongState::Finished {
                            output_share,
                            message: PingPongMessage::Finish {
                                prepare_message: current_prepare_message
                                    .get_encoded()
                                    .map_err(PingPongError::CodecPrepMessage)?,
                            },
                        }),
                    })
            }
            (PrepareTransition::Finish(output_share), None) => {
                Ok(PingPongState::FinishedNoMessage { output_share })
            }
            (PrepareTransition::Continue(_, _), None) => Err(PingPongError::PeerMessageMismatch {
                found: inbound.variant(),
                expected: "continue",
            }),
            (PrepareTransition::Finish(_), Some(_)) => Err(PingPongError::PeerMessageMismatch {
                found: inbound.variant(),
                expected: "finish",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::vdaf::dummy;
    use assert_matches::assert_matches;

    const CTX_STR: &[u8] = b"pingpong ctx";

    #[test]
    fn drive_vdaf_to_completion() {
        // This isn't a real test, but rather is intended just to concisely illustrate usage of the
        // API.
        let verify_key = [];
        let aggregation_param = dummy::AggregationParam(0);
        let nonce = [0; 16];
        #[allow(clippy::let_unit_value)]
        let public_share = ();
        let input_share = dummy::InputShare(0);

        let leader = dummy::Vdaf::new(16);
        let helper = dummy::Vdaf::new(16);

        // Leader inits into round 0
        let (mut current_leader_prepare_state, leader_prepare_share) = leader
            .leader_initialized(
                &verify_key,
                CTX_STR,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
            )
            .unwrap();

        // Helper inits into round 1
        let (helper_prepare_transition, mut current_helper_message) = helper
            .helper_initialized(
                &verify_key,
                CTX_STR,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
                &leader_prepare_share,
            )
            .unwrap();

        let mut current_leader_message = None;
        let mut leader_output_share = None;

        let mut current_helper_prepare_state = None;
        let mut helper_output_share = None;

        match helper_prepare_transition {
            PrepareTransition::Continue(helper_prepare_state, _) => {
                current_helper_prepare_state = Some(helper_prepare_state)
            }
            PrepareTransition::Finish(output_share) => helper_output_share = Some(output_share),
        }

        // Continue until both aggregators are finished
        loop {
            if leader_output_share.is_none() {
                let new_leader_state = leader
                    .leader_continued(
                        CTX_STR,
                        current_leader_prepare_state,
                        &aggregation_param,
                        &current_helper_message,
                    )
                    .unwrap();

                match new_leader_state {
                    PingPongState::Continued {
                        prepare_state,
                        message,
                    } => {
                        current_leader_prepare_state = prepare_state;
                        current_leader_message = Some(message);
                    }
                    PingPongState::Finished {
                        output_share,
                        message,
                    } => {
                        leader_output_share = Some(output_share);
                        current_leader_message = Some(message);
                    }
                    PingPongState::FinishedNoMessage { output_share } => {
                        leader_output_share = Some(output_share);
                    }
                }
            }

            if helper_output_share.is_none() {
                let new_helper_state = helper
                    .helper_continued(
                        CTX_STR,
                        current_helper_prepare_state.unwrap(),
                        &aggregation_param,
                        &current_leader_message.clone().unwrap(),
                    )
                    .unwrap();

                match new_helper_state {
                    PingPongState::Continued {
                        prepare_state,
                        message,
                    } => {
                        current_helper_prepare_state = Some(prepare_state);
                        current_helper_message = message;
                    }
                    PingPongState::Finished {
                        output_share,
                        message,
                    } => {
                        helper_output_share = Some(output_share);
                        current_helper_message = message;
                    }
                    PingPongState::FinishedNoMessage { output_share } => {
                        helper_output_share = Some(output_share);
                    }
                }
            }

            if leader_output_share.is_some() && helper_output_share.is_some() {
                break;
            }
        }
    }

    #[test]
    fn ping_pong_one_round() {
        let verify_key = [];
        let aggregation_param = dummy::AggregationParam(0);
        let nonce = [0; 16];
        #[allow(clippy::let_unit_value)]
        let public_share = ();
        let input_share = dummy::InputShare(0);

        let leader = dummy::Vdaf::new(1);
        let helper = dummy::Vdaf::new(1);

        // Leader inits into round 0
        let (leader_state, leader_prepare_share) = leader
            .leader_initialized(
                &verify_key,
                CTX_STR,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
            )
            .unwrap();

        // Helper inits into round 1
        let (helper_transition, helper_message) = helper
            .helper_initialized(
                &verify_key,
                CTX_STR,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
                &leader_prepare_share,
            )
            .unwrap();

        // 1 round VDAF: helper should finish immediately.
        assert_matches!(helper_transition, PrepareTransition::Finish(_));

        let leader_state = leader
            .leader_continued(CTX_STR, leader_state, &aggregation_param, &helper_message)
            .unwrap();
        // 1 round VDAF: leader should finish when it gets helper message and emit no message.
        assert_matches!(leader_state, PingPongState::FinishedNoMessage { .. });
    }

    #[test]
    fn ping_pong_two_rounds() {
        let verify_key = [];
        let aggregation_param = dummy::AggregationParam(0);
        let nonce = [0; 16];
        #[allow(clippy::let_unit_value)]
        let public_share = ();
        let input_share = dummy::InputShare(0);

        let leader = dummy::Vdaf::new(2);
        let helper = dummy::Vdaf::new(2);

        // Leader inits into round 0
        let (leader_state, leader_prepare_share) = leader
            .leader_initialized(
                &verify_key,
                CTX_STR,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
            )
            .unwrap();

        // Helper inits into round 1
        let (helper_state, helper_message) = helper
            .helper_initialized(
                &verify_key,
                CTX_STR,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
                &leader_prepare_share,
            )
            .unwrap();

        // 2 round VDAF, round 1: helper should continue.
        let helper_state = assert_matches!(helper_state, PrepareTransition::Continue(s, _) => s);

        let leader_state = leader
            .leader_continued(CTX_STR, leader_state, &aggregation_param, &helper_message)
            .unwrap();
        // 2 round VDAF, round 1: leader should finish and emit a finish message.
        let leader_message = assert_matches!(leader_state, PingPongState::Finished{ message, .. } => {
            message
        });

        let helper_state = helper
            .helper_continued(CTX_STR, helper_state, &aggregation_param, &leader_message)
            .unwrap();
        // 2 round vdaf, round 1: helper should finish and emit no message.
        assert_matches!(helper_state, PingPongState::FinishedNoMessage { .. });
    }

    #[test]
    fn ping_pong_three_rounds() {
        let verify_key = [];
        let aggregation_param = dummy::AggregationParam(0);
        let nonce = [0; 16];
        #[allow(clippy::let_unit_value)]
        let public_share = ();
        let input_share = dummy::InputShare(0);

        let leader = dummy::Vdaf::new(3);
        let helper = dummy::Vdaf::new(3);

        // Leader inits into round 0
        let (leader_state, leader_prepare_share) = leader
            .leader_initialized(
                &verify_key,
                CTX_STR,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
            )
            .unwrap();

        // Helper inits into round 1
        let (helper_state, helper_message) = helper
            .helper_initialized(
                &verify_key,
                CTX_STR,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
                &leader_prepare_share,
            )
            .unwrap();

        // 3 round VDAF, round 1: helper should continue.
        let helper_state = assert_matches!(helper_state, PrepareTransition::Continue(s, _) => s);

        let leader_state = leader
            .leader_continued(CTX_STR, leader_state, &aggregation_param, &helper_message)
            .unwrap();
        // 3 round VDAF, round 1: leader should continue and emit a continue message.
        let (leader_state, leader_message) = assert_matches!(
            leader_state, PingPongState::Continued{prepare_state, message} => (prepare_state, message)
        );

        let helper_state = helper
            .helper_continued(CTX_STR, helper_state, &aggregation_param, &leader_message)
            .unwrap();
        // 3 round vdaf, round 2: helper should finish and emit a finish message.
        let helper_message = assert_matches!(helper_state, PingPongState::Finished{message, ..} => {
            message
        });

        let leader_state = leader
            .leader_continued(CTX_STR, leader_state, &aggregation_param, &helper_message)
            .unwrap();
        // 3 round VDAF, round 2: leader should finish and emit no message.
        assert_matches!(leader_state, PingPongState::FinishedNoMessage { .. });
    }

    #[test]
    fn roundtrip_message() {
        let messages = [
            (
                PingPongMessage::Continued {
                    prepare_message: Vec::from("prepare message"),
                    prepare_share: Vec::from("prepare share"),
                },
                concat!(
                    "00", // enum discriminant
                    concat!(
                        // prepare_message
                        "0000000f",                       // length
                        "70726570617265206d657373616765", // contents
                    ),
                    concat!(
                        // prepare_share
                        "0000000d",                   // length
                        "70726570617265207368617265", // contents
                    ),
                ),
            ),
            (
                PingPongMessage::Finish {
                    prepare_message: Vec::from("prepare message"),
                },
                concat!(
                    "01", // enum discriminant
                    concat!(
                        // prepare_message
                        "0000000f",                       // length
                        "70726570617265206d657373616765", // contents
                    ),
                ),
            ),
        ];

        for (message, expected_hex) in messages {
            let mut encoded_val = Vec::new();
            message.encode(&mut encoded_val).unwrap();
            let got_hex = hex::encode(&encoded_val);
            assert_eq!(
                &got_hex, expected_hex,
                "Couldn't roundtrip (encoded value differs): {message:?}",
            );
            let decoded_val = PingPongMessage::decode(&mut Cursor::new(&encoded_val)).unwrap();
            assert_eq!(
                decoded_val, message,
                "Couldn't roundtrip (decoded value differs): {message:?}"
            );
            assert_eq!(
                encoded_val.len(),
                message.encoded_len().expect("No encoded length hint"),
                "Encoded length hint is incorrect: {message:?}"
            )
        }
    }
}
