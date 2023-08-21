// SPDX-License-Identifier: MPL-2.0

//! Implements the Ping-Pong Topology described in [VDAF]. This topology assumes there are exactly
//! two aggregators, designated "Leader" and "Helper". Note that while this implementation is
//! compatible with VDAF-06, it actually implements the ping-pong wrappers specified in VDAF-07
//! (forthcoming) since those line up better with the VDAF implementation in this crate.
//!
//! [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-06#section-5.8

use crate::{
    codec::{decode_u32_items, encode_u32_items, CodecError, Decode, Encode, ParameterizedDecode},
    vdaf::{Aggregator, PrepareTransition, VdafError},
};
use std::fmt::Debug;

/// Errors emitted by this module.
#[derive(Debug, thiserror::Error)]
pub enum PingPongError {
    /// Error running prepare_init
    #[error("vdaf.prepare_init: {0}")]
    VdafPrepareInit(VdafError),

    /// Error running prepare_preprocess
    #[error("vdaf.prepare_preprocess {0}")]
    VdafPreparePreprocess(VdafError),

    /// Error running prepare_step
    #[error("vdaf.prepare_step {0}")]
    VdafPrepareStep(VdafError),

    /// Error decoding a prepare share
    #[error("decode prep share {0}")]
    CodecPrepShare(CodecError),

    /// Error decoding a prepare message
    #[error("decode prep message {0}")]
    CodecPrepMessage(CodecError),

    /// State machine mismatch between peer and host.
    #[error("state mismatch: host state {0} peer state {0}")]
    StateMismatch(&'static str, &'static str),

    /// Internal error
    #[error("internal error: {0}")]
    InternalError(&'static str),
}

/// Distinguished aggregator roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The Leader aggregator.
    Leader,
    /// The Helper aggregator.
    Helper,
}

/// Corresponds to `struct Message` in [VDAF's Ping-Pong Topology][VDAF]. All of the fields of the
/// variants are opaque byte buffers. This is because the ping-pong routines take responsibility for
/// decoding preparation shares and messages, which usually requires having the preparation state.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-06#section-5.8
#[derive(Clone, PartialEq, Eq)]
pub enum Message {
    /// Corresponds to MessageType.initialize.
    Initialize {
        /// The leader's initial preparation share.
        prep_share: Vec<u8>,
    },
    /// Corresponds to MessageType.continue.
    Continue {
        /// The current round's preparation message.
        prep_msg: Vec<u8>,
        /// The next round's preparation share.
        prep_share: Vec<u8>,
    },
    /// Corresponds to MessageType.finish.
    Finish {
        /// The current round's preparation message.
        prep_msg: Vec<u8>,
    },
}

impl Message {
    fn state_name(&self) -> &'static str {
        match self {
            Self::Initialize { .. } => "Initialize",
            Self::Continue { .. } => "Continue",
            Self::Finish { .. } => "Finish",
        }
    }
}

impl Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Message::{}", self.state_name())
    }
}

impl Encode for Message {
    fn encode(&self, bytes: &mut Vec<u8>) {
        // The encoding includes an implicit discriminator byte, called MessageType in the VDAF
        // spec.
        match self {
            Self::Initialize { prep_share } => {
                0u8.encode(bytes);
                encode_u32_items(bytes, &(), prep_share);
            }
            Self::Continue {
                prep_msg,
                prep_share,
            } => {
                1u8.encode(bytes);
                encode_u32_items(bytes, &(), prep_msg);
                encode_u32_items(bytes, &(), prep_share);
            }
            Self::Finish { prep_msg } => {
                2u8.encode(bytes);
                encode_u32_items(bytes, &(), prep_msg);
            }
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            Self::Initialize { prep_share } => Some(1 + 4 + prep_share.len()),
            Self::Continue {
                prep_msg,
                prep_share,
            } => Some(1 + 4 + prep_msg.len() + 4 + prep_share.len()),
            Self::Finish { prep_msg } => Some(1 + 4 + prep_msg.len()),
        }
    }
}

impl Decode for Message {
    fn decode(bytes: &mut std::io::Cursor<&[u8]>) -> Result<Self, CodecError> {
        let message_type = u8::decode(bytes)?;
        Ok(match message_type {
            0 => {
                let prep_share = decode_u32_items(&(), bytes)?;
                Self::Initialize { prep_share }
            }
            1 => {
                let prep_msg = decode_u32_items(&(), bytes)?;
                let prep_share = decode_u32_items(&(), bytes)?;
                Self::Continue {
                    prep_msg,
                    prep_share,
                }
            }
            2 => {
                let prep_msg = decode_u32_items(&(), bytes)?;
                Self::Finish { prep_msg }
            }
            _ => return Err(CodecError::UnexpectedValue),
        })
    }
}

/// Corresponds to the `State` enumeration implicitly defined in [VDAF's Ping-Pong Topology][VDAF].
/// VDAF describes `Start` and `Rejected` states, but the `Start` state is never instantiated in
/// code, and the `Rejected` state is represented as `std::result::Result::Error`, so this enum does
/// not include those variants.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-06#section-5.8
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum State<PrepState, OutputShare> {
    /// Preparation of the report will continue with the enclosed state.
    Continued(PrepState),
    /// Preparation of the report is finished and has yielded the enclosed output share.
    Finished(OutputShare),
}

impl<PrepState: Encode, OutputShare: Encode> Encode for State<PrepState, OutputShare> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        match self {
            Self::Continued(prep_state) => {
                0u8.encode(bytes);
                prep_state.encode(bytes);
            }
            Self::Finished(output_share) => {
                1u8.encode(bytes);
                output_share.encode(bytes);
            }
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            1 + match self {
                Self::Continued(prep_state) => prep_state.encoded_len()?,
                Self::Finished(output_share) => output_share.encoded_len()?,
            },
        )
    }
}

/// Decoding parameter for [`State`].
pub struct StateDecodingParam<'a, PrepStateDecode, OutputShareDecode> {
    /// The decoding parameter for the preparation state.
    pub prep_state: &'a PrepStateDecode,
    /// The decoding parameter for the output share.
    pub output_share: &'a OutputShareDecode,
}

impl<
        'a,
        PrepStateDecode,
        OutputShareDecode,
        PrepState: ParameterizedDecode<PrepStateDecode>,
        OutputShare: ParameterizedDecode<OutputShareDecode>,
    > ParameterizedDecode<StateDecodingParam<'a, PrepStateDecode, OutputShareDecode>>
    for State<PrepState, OutputShare>
{
    fn decode_with_param(
        decoding_param: &StateDecodingParam<PrepStateDecode, OutputShareDecode>,
        bytes: &mut std::io::Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let variant = u8::decode(bytes)?;
        match variant {
            0 => Ok(Self::Continued(PrepState::decode_with_param(
                decoding_param.prep_state,
                bytes,
            )?)),
            1 => Ok(Self::Finished(OutputShare::decode_with_param(
                decoding_param.output_share,
                bytes,
            )?)),
            _ => Err(CodecError::UnexpectedValue),
        }
    }
}

/// Values returned by [`PingPongTopology::continued`].
///
/// Corresponds to the `State` enumeration implicitly defined in [VDAF's Ping-Pong Topology][VDAF],
/// but combined with the components of [`Message`] so that no impossible states may be represented.
/// For example, it's impossible to be in the `Continued` state but to send an outgoing `Message` of
/// type `Finished`.
///
/// VDAF describes `Start` and `Rejected` states, but the `Start` state is never instantiated in
/// code, and the `Rejected` state is represented as `std::result::Result::Error`, so this enum does
/// not include those variants.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-06#section-5.8
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateAndMessage<PrepareState, OutputShare> {
    /// The operation resulted in a new state and a message to transmit to the peer.
    WithMessage {
        /// The [`State`] to which we just advanced.
        state: State<PrepareState, OutputShare>,
        /// The [`Message`] which should now be transmitted to the peer.
        message: Message,
    },
    /// The operation caused the host to finish preparation of the input share, yielding an output
    /// share and no message for the peer.
    FinishedNoMessage {
        /// The output share which may now be accumulated.
        output_share: OutputShare,
    },
}

/// Extension trait on [`crate::vdaf::Aggregator`] which adds the [VDAF Ping-Pong Topology][VDAF].
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-06#section-5.8
pub trait PingPongTopology<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>:
    Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    /// Specialization of [`State`] for this VDAF.
    type State;
    /// Specialization of [`StateAndMessage`] for this VDAF.
    type StateAndMessage;

    /// Initialize leader state using the leader's input share. Corresponds to
    /// `ping_pong_leader_init` in the forthcoming `draft-irtf-cfrg-vdaf-07`.
    ///
    /// If successful, the returned [`State`] (which will always be `State::Continued`) should be
    /// stored by the leader for use in the next round, and the returned [`Message`] (which will
    /// always be `Message::Initialize`) should be transmitted to the helper.
    fn leader_initialize(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        agg_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::State, Message), PingPongError>;

    /// Initialize helper state using the helper's input share and the leader's first prepare share.
    /// Corresponds to `ping_pong_helper_init` in the forthcoming `draft-irtf-cfrg-vdaf-07`.
    ///
    /// If successful, the returned [`State`] will either be `State::Continued` and should be stored
    /// by the helper for use in the next round or `State::Finished`, in which case preparation is
    /// finished and the output share may be accumulated by the helper.
    ///
    /// Regardless of state, the returned [`Message`] should be transmitted to the leader.
    ///
    /// # Errors
    ///
    /// `inbound` must be `Message::Initialize` or the function will fail.
    fn helper_initialize(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        agg_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
        inbound: &Message,
    ) -> Result<(Self::State, Message), PingPongError>;

    /// Continue preparation based on the host's current state and an incoming [`Message`] from the
    /// peer. `role` is the host's [`Role`]. Corresponds to `ping_pong_contnued` in the forthcoming
    /// `draft-irtf-cfrg-vdaf-07`.
    ///
    /// If successful, the returned [`StateAndMessage`] will either be:
    ///
    /// - `StateAndMessage::WithMessage { state, message }`: the `state` will be either
    ///   `State::Continued` and should be stored by the host for use in the next round or
    ///   `State::Finished`, in which case preparation is finished and the output share may be
    ///   accumulated. Regardless of state, `message` should be transmitted to the peer.
    /// - `StateAndMessage::FinishedNoMessage`: preparation is finished and the output share may be
    ///   accumulated. No message needs to be send to the peer.
    ///
    /// # Errors
    ///
    /// `host_state` must be `State::Continued` or the function will fail.
    ///
    /// `inbound` must not be `Message::Initialize` or the function will fail.
    ///
    /// # Notes
    ///
    /// The specification of this function in [VDAF] takes the aggregation parameter. This version
    /// does not, because [`vdaf::Vdaf::prepare_preprocess`] does not take the aggregation
    /// parameter. This may change in the future if/when [#670][issue] is addressed.
    ///
    /// [issue]: https://github.com/divviup/libprio-rs/issues/670
    fn continued(
        &self,
        role: Role,
        host_state: Self::State,
        inbound: &Message,
    ) -> Result<StateAndMessage<Self::PrepareState, Self::OutputShare>, PingPongError>;
}

/// Private interfaces for VDAF Ping-Pong topology.
trait PingPongTopologyPrivate<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>:
    PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    /// Corresponds to `ping_pong_transition` in the forthcoming `draft-irtf-cfrg-vdaf-07`.
    /// `prep_shares` must be ordered so that the leader's prepare share is first.
    ///
    /// # Notes
    ///
    /// The specification of this function in [VDAF] takes the aggregation parameter. This version
    /// does not, because [`vdaf::Vdaf::prepare_preprocess`] does not take the aggregation
    /// parameter. This may change in the future if/when [#670][issue] is addressed.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-06#section-5.8
    /// [issue]: https://github.com/divviup/libprio-rs/issues/670
    fn transition(
        &self,
        prep_shares: [Self::PrepareShare; 2],
        host_prep_state: Self::PrepareState,
    ) -> Result<(Self::State, Message), PingPongError>;
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A>
    PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE> for A
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
{
    type State = State<A::PrepareState, A::OutputShare>;
    type StateAndMessage = StateAndMessage<A::PrepareState, A::OutputShare>;

    fn leader_initialize(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        agg_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::State, Message), PingPongError> {
        self.prepare_init(
            verify_key,
            /* Leader */ 0,
            agg_param,
            nonce,
            public_share,
            input_share,
        )
        .map(|(prep_state, prep_share)| {
            (
                State::Continued(prep_state),
                Message::Initialize {
                    prep_share: prep_share.get_encoded(),
                },
            )
        })
        .map_err(PingPongError::VdafPrepareInit)
    }

    fn helper_initialize(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        agg_param: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
        inbound: &Message,
    ) -> Result<(Self::State, Message), PingPongError> {
        let (prep_state, prep_share) = self
            .prepare_init(
                verify_key,
                /* Helper */ 1,
                agg_param,
                nonce,
                public_share,
                input_share,
            )
            .map_err(PingPongError::VdafPrepareInit)?;

        let inbound_prep_share = if let Message::Initialize { prep_share } = inbound {
            Self::PrepareShare::get_decoded_with_param(&prep_state, prep_share)
                .map_err(PingPongError::CodecPrepShare)?
        } else {
            return Err(PingPongError::StateMismatch(
                "initialize",
                inbound.state_name(),
            ));
        };

        self.transition([inbound_prep_share, prep_share], prep_state)
    }

    fn continued(
        &self,
        role: Role,
        host_state: Self::State,
        inbound: &Message,
    ) -> Result<Self::StateAndMessage, PingPongError> {
        let host_prep_state = if let State::Continued(state) = host_state {
            state
        } else {
            return Err(PingPongError::StateMismatch("finished", "continue"));
        };

        let (prep_msg, next_peer_prep_share) = match inbound {
            Message::Initialize { .. } => {
                return Err(PingPongError::StateMismatch(
                    "continue",
                    inbound.state_name(),
                ));
            }
            Message::Continue {
                prep_msg,
                prep_share,
            } => (prep_msg, Some(prep_share)),
            Message::Finish { prep_msg } => (prep_msg, None),
        };

        let prep_msg = Self::PrepareMessage::get_decoded_with_param(&host_prep_state, prep_msg)
            .map_err(PingPongError::CodecPrepMessage)?;
        let host_prep_transition = self
            .prepare_step(host_prep_state, prep_msg)
            .map_err(PingPongError::VdafPrepareStep)?;

        match (host_prep_transition, next_peer_prep_share) {
            (
                PrepareTransition::Continue(next_prep_state, next_host_prep_share),
                Some(next_peer_prep_share),
            ) => {
                let next_peer_prep_share = Self::PrepareShare::get_decoded_with_param(
                    &next_prep_state,
                    next_peer_prep_share,
                )
                .map_err(PingPongError::CodecPrepShare)?;
                let mut prep_shares = [next_peer_prep_share, next_host_prep_share];
                if role == Role::Leader {
                    prep_shares.reverse();
                }
                self.transition(prep_shares, next_prep_state)
                    .map(|(state, message)| StateAndMessage::WithMessage { state, message })
            }
            (PrepareTransition::Finish(output_share), None) => {
                Ok(StateAndMessage::FinishedNoMessage { output_share })
            }
            (transition, _) => {
                return Err(PingPongError::StateMismatch(
                    inbound.state_name(),
                    match transition {
                        PrepareTransition::Continue(_, _) => "continue",
                        PrepareTransition::Finish(_) => "finished",
                    },
                ))
            }
        }
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A>
    PingPongTopologyPrivate<VERIFY_KEY_SIZE, NONCE_SIZE> for A
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
{
    fn transition(
        &self,
        prep_shares: [Self::PrepareShare; 2],
        host_prep_state: Self::PrepareState,
    ) -> Result<(Self::State, Message), PingPongError> {
        let prep_message = self
            .prepare_preprocess(prep_shares)
            .map_err(PingPongError::VdafPreparePreprocess)?;

        self.prepare_step(host_prep_state, prep_message.clone())
            .map(|transition| match transition {
                PrepareTransition::Continue(prep_state, prep_share) => (
                    State::Continued(prep_state),
                    Message::Continue {
                        prep_msg: prep_message.get_encoded(),
                        prep_share: prep_share.get_encoded(),
                    },
                ),
                PrepareTransition::Finish(output_share) => (
                    State::Finished(output_share),
                    Message::Finish {
                        prep_msg: prep_message.get_encoded(),
                    },
                ),
            })
            .map_err(PingPongError::VdafPrepareStep)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;

    use crate::vdaf::dummy;

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
        let (leader_state, leader_message) = leader
            .leader_initialize(
                &verify_key,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
            )
            .unwrap();

        // Helper inits into round 1
        let (helper_state, helper_message) = helper
            .helper_initialize(
                &verify_key,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
                &leader_message,
            )
            .unwrap();

        // 1 round VDAF: helper should finish immediately.
        assert_matches!(helper_state, State::Finished(_));

        let leader_state = leader
            .continued(Role::Leader, leader_state, &helper_message)
            .unwrap();
        // 1 round VDAF: leader should finish when it gets helper message and emit no message.
        assert_matches!(leader_state, StateAndMessage::FinishedNoMessage { .. });
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
        let (leader_state, leader_message) = leader
            .leader_initialize(
                &verify_key,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
            )
            .unwrap();

        // Helper inits into round 1
        let (helper_state, helper_message) = helper
            .helper_initialize(
                &verify_key,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
                &leader_message,
            )
            .unwrap();

        // 2 round VDAF, round 1: helper should continue.
        assert_matches!(helper_state, State::Continued(_));

        let leader_state = leader
            .continued(Role::Leader, leader_state, &helper_message)
            .unwrap();
        // 2 round VDAF, round 1: leader should finish and emit a finish message.
        let leader_message = assert_matches!(
            leader_state, StateAndMessage::WithMessage { state: State::Finished(_), message } => message
        );

        let helper_state = helper
            .continued(Role::Helper, helper_state, &leader_message)
            .unwrap();
        // 2 round vdaf, round 1: helper should finish and emit no message.
        assert_matches!(helper_state, StateAndMessage::FinishedNoMessage { .. });
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
        let (leader_state, leader_message) = leader
            .leader_initialize(
                &verify_key,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
            )
            .unwrap();

        // Helper inits into round 1
        let (helper_state, helper_message) = helper
            .helper_initialize(
                &verify_key,
                &aggregation_param,
                &nonce,
                &public_share,
                &input_share,
                &leader_message,
            )
            .unwrap();

        // 3 round VDAF, round 1: helper should continue.
        assert_matches!(helper_state, State::Continued(_));

        let leader_state = leader
            .continued(Role::Leader, leader_state, &helper_message)
            .unwrap();
        // 3 round VDAF, round 1: leader should continue and emit a continue message.
        let (leader_state, leader_message) = assert_matches!(
            leader_state, StateAndMessage::WithMessage { ref message, state } => (state, message)
        );

        let helper_state = helper
            .continued(Role::Helper, helper_state, leader_message)
            .unwrap();
        // 3 round vdaf, round 2: helper should finish and emit a finish message.
        let helper_message = assert_matches!(
            helper_state, StateAndMessage::WithMessage { message, state: State::Finished(_) } => message
        );

        let leader_state = leader
            .continued(Role::Leader, leader_state, &helper_message)
            .unwrap();
        // 3 round VDAF, round 2: leader should finish and emit no message.
        assert_matches!(leader_state, StateAndMessage::FinishedNoMessage { .. });
    }
}
