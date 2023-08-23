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
        f.debug_tuple(self.state_name()).finish()
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

/// A transition in the pong-pong topology. This represents the `ping_pong_transition` function
/// defined in [VDAF].
///
/// # Discussion
///
/// The obvious implementation would of `ping_pong_transition` would be a method on trait
/// [`PingPongTopology`] that returns `(State, Message)`, and then `ContinuedValue::WithMessage`
/// would contain those values. But then DAP implementations would have to store relatively large
/// VDAF prepare shares between rounds of input preparation.
///
/// Instead, this structure stores just the previous round's prepare state and the current round's
/// preprocessed prepare message. Their encoding is much smaller than the `(State, Message)` tuple,
/// which can always be recomputed with [`Self::evaluate`].
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-06#section-5.8
#[derive(Clone, Debug, Eq)]
pub struct Transition<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
> {
    previous_prepare_state: A::PrepareState,
    current_prepare_message: A::PrepareMessage,
}

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > Transition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
{
    /// Evaluate this transition to obtain a new [`State`] and a [`Message`] which should be
    /// transmitted to the peer.
    #[allow(clippy::type_complexity)]
    pub fn evaluate(
        &self,
        vdaf: &A,
    ) -> Result<(State<VERIFY_KEY_SIZE, NONCE_SIZE, A>, Message), PingPongError> {
        let prep_msg = self.current_prepare_message.get_encoded();

        vdaf.prepare_step(
            self.previous_prepare_state.clone(),
            self.current_prepare_message.clone(),
        )
        .map(|transition| match transition {
            PrepareTransition::Continue(prep_state, prep_share) => (
                State::Continued(prep_state),
                Message::Continue {
                    prep_msg,
                    prep_share: prep_share.get_encoded(),
                },
            ),
            PrepareTransition::Finish(output_share) => {
                (State::Finished(output_share), Message::Finish { prep_msg })
            }
        })
        .map_err(PingPongError::VdafPrepareStep)
    }
}

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > PartialEq for Transition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
{
    fn eq(&self, other: &Self) -> bool {
        self.previous_prepare_state == other.previous_prepare_state
            && self.current_prepare_message == other.current_prepare_message
    }
}

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > Default for Transition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A::PrepareState: Default,
    A::PrepareMessage: Default,
{
    fn default() -> Self {
        Self {
            previous_prepare_state: A::PrepareState::default(),
            current_prepare_message: A::PrepareMessage::default(),
        }
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A> Encode
    for Transition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::PrepareState: Encode,
{
    fn encode(&self, bytes: &mut Vec<u8>) {
        self.previous_prepare_state.encode(bytes);
        self.current_prepare_message.encode(bytes);
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.previous_prepare_state.encoded_len()?
                + self.current_prepare_message.encoded_len()?,
        )
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A, PrepareStateDecode>
    ParameterizedDecode<PrepareStateDecode> for Transition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::PrepareState: ParameterizedDecode<PrepareStateDecode> + PartialEq,
    A::PrepareMessage: PartialEq,
{
    fn decode_with_param(
        decoding_param: &PrepareStateDecode,
        bytes: &mut std::io::Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let previous_prepare_state = A::PrepareState::decode_with_param(decoding_param, bytes)?;
        let current_prepare_message =
            A::PrepareMessage::decode_with_param(&previous_prepare_state, bytes)?;

        Ok(Self {
            previous_prepare_state,
            current_prepare_message,
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
pub enum State<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
> {
    /// Preparation of the report will continue with the enclosed state.
    Continued(A::PrepareState),
    /// Preparation of the report is finished and has yielded the enclosed output share.
    Finished(A::OutputShare),
}

/// Values returned by [`PingPongTopology::continued`].
///
/// Corresponds to the `State` enumeration implicitly defined in [VDAF's Ping-Pong Topology][VDAF],
/// but combined with the components of [`Message`] so that no impossible states may be represented.
/// For example, it's impossible to be in the `Continued` state but to send an outgoing `Message` of
/// type `Finished`.
///
/// VDAF describes `Start` and `Rejected` states, but the `Start` state is never instantiated in
/// code, and the `Rejected` state is represented as [`std::result::Result::Error`], so this enum
/// does not include those variants.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-06#section-5.8
#[derive(Clone, Debug)]
pub enum ContinuedValue<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
> {
    /// The operation resulted in a new state and a message to transmit to the peer.
    WithMessage {
        /// The transition that will be executed. Call `Transition::evaluate` to obtain the next
        /// [`State`] and a [`Message`] to transmit to the peer.
        transition: Transition<VERIFY_KEY_SIZE, NONCE_SIZE, A>,
    },
    /// The operation caused the host to finish preparation of the input share, yielding an output
    /// share and no message for the peer.
    FinishedNoMessage {
        /// The output share which may now be accumulated.
        output_share: A::OutputShare,
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
    /// Specialization of [`ContinuedValue`] for this VDAF.
    type ContinuedValue;

    /// Initialize leader state using the leader's input share. Corresponds to
    /// `ping_pong_leader_init` in the forthcoming `draft-irtf-cfrg-vdaf-07`.
    ///
    /// If successful, the returned [`Message`] (which will always be `Message::Initialize`) should
    /// be transmitted to the helper. The returned [`State`] (which will always be
    /// `State::Continued`) should be used by the leader along with the next [`Message`] received
    /// from the helper as input to [`Self::continued`] to advance to the next round.
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
    /// If successful, the returned [`Transition`] should be evaluated, yielding a [`Message`],
    /// which should be transmitted to the leader, and a [`State`].
    ///
    /// If the state is `State::Continued`, then it should be used by the helper along with the next
    /// `Message` received from the leader as input to [`Self::continued`] to advance to the next
    /// round. The helper may store the `Transition` between rounds of preparation instead of the
    /// `State` and `Message`.
    ///
    /// If the state is `State::Finished`, then preparation is finished and the output share may be
    /// accumulated.
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
    ) -> Result<Transition<VERIFY_KEY_SIZE, NONCE_SIZE, Self>, PingPongError>;

    /// Continue preparation based on the host's current state and an incoming [`Message`] from the
    /// peer. `role` is the host's [`Role`]. Corresponds to `ping_pong_contnued` in the forthcoming
    /// `draft-irtf-cfrg-vdaf-07`.
    ///
    /// If successful, the returned [`ContinuedValue`] will either be:
    ///
    /// - `ContinuedValue::WithMessage { transition }`: `transition` should be evaluated, yielding a
    ///   [`Message`], which should be transmitted to the peer, and a [`State`].
    ///
    ///   If the state is `State::Continued`, then it should be used by this aggregator along with
    ///   the next `Message` received from the peer as input to [`Self::continued`] to advance to
    ///   the next round. The aggregator may store the `Transition` between rounds of preparation
    ///   instead of the `State` and `Message`.
    ///
    ///   If the state is `State::Finished`, then preparation is finished and the output share may
    ///   be accumulated.
    ///
    /// - `ContinuedValue::FinishedNoMessage`: preparation is finished and the output share may be
    ///   accumulated. No message needs to be sent to the peer.
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
    ) -> Result<ContinuedValue<VERIFY_KEY_SIZE, NONCE_SIZE, Self>, PingPongError>;
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A>
    PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE> for A
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
{
    type State = State<VERIFY_KEY_SIZE, NONCE_SIZE, Self>;
    type ContinuedValue = ContinuedValue<VERIFY_KEY_SIZE, NONCE_SIZE, Self>;

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
    ) -> Result<Transition<VERIFY_KEY_SIZE, NONCE_SIZE, Self>, PingPongError> {
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

        let current_prepare_message = self
            .prepare_preprocess([inbound_prep_share, prep_share])
            .map_err(PingPongError::VdafPreparePreprocess)?;

        Ok(Transition {
            previous_prepare_state: prep_state,
            current_prepare_message,
        })
    }

    fn continued(
        &self,
        role: Role,
        host_state: Self::State,
        inbound: &Message,
    ) -> Result<Self::ContinuedValue, PingPongError> {
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
                let current_prepare_message = self
                    .prepare_preprocess(prep_shares)
                    .map_err(PingPongError::VdafPreparePreprocess)?;

                Ok(ContinuedValue::WithMessage {
                    transition: Transition {
                        previous_prepare_state: next_prep_state,
                        current_prepare_message,
                    },
                })
            }
            (PrepareTransition::Finish(output_share), None) => {
                Ok(ContinuedValue::FinishedNoMessage { output_share })
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
            .unwrap()
            .evaluate(&helper)
            .unwrap();

        // 1 round VDAF: helper should finish immediately.
        assert_matches!(helper_state, State::Finished(_));

        let leader_state = leader
            .continued(Role::Leader, leader_state, &helper_message)
            .unwrap();
        // 1 round VDAF: leader should finish when it gets helper message and emit no message.
        assert_matches!(leader_state, ContinuedValue::FinishedNoMessage { .. });
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
            .unwrap()
            .evaluate(&helper)
            .unwrap();

        // 2 round VDAF, round 1: helper should continue.
        assert_matches!(helper_state, State::Continued(_));

        let leader_state = leader
            .continued(Role::Leader, leader_state, &helper_message)
            .unwrap();
        // 2 round VDAF, round 1: leader should finish and emit a finish message.
        let leader_message = assert_matches!(
            leader_state, ContinuedValue::WithMessage { transition } => {
                let (state, message) = transition.evaluate(&leader).unwrap();
                assert_matches!(state, State::Finished(_));
                message
            }
        );

        let helper_state = helper
            .continued(Role::Helper, helper_state, &leader_message)
            .unwrap();
        // 2 round vdaf, round 1: helper should finish and emit no message.
        assert_matches!(helper_state, ContinuedValue::FinishedNoMessage { .. });
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
            .unwrap()
            .evaluate(&helper)
            .unwrap();

        // 3 round VDAF, round 1: helper should continue.
        assert_matches!(helper_state, State::Continued(_));

        let leader_state = leader
            .continued(Role::Leader, leader_state, &helper_message)
            .unwrap();
        // 3 round VDAF, round 1: leader should continue and emit a continue message.
        let (leader_state, leader_message) = assert_matches!(
            leader_state, ContinuedValue::WithMessage { transition } => {
                let (state, message) = transition.evaluate(&leader).unwrap();
                assert_matches!(state, State::Continued(_));
                (state, message)
            }
        );

        let helper_state = helper
            .continued(Role::Helper, helper_state, &leader_message)
            .unwrap();
        // 3 round vdaf, round 2: helper should finish and emit a finish message.
        let helper_message = assert_matches!(
            helper_state, ContinuedValue::WithMessage { transition } => {
                let (state, message) = transition.evaluate(&helper).unwrap();
                assert_matches!(state, State::Finished(_));
                message
            }
        );

        let leader_state = leader
            .continued(Role::Leader, leader_state, &helper_message)
            .unwrap();
        // 3 round VDAF, round 2: leader should finish and emit no message.
        assert_matches!(leader_state, ContinuedValue::FinishedNoMessage { .. });
    }
}
