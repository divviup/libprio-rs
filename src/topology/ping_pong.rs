// SPDX-License-Identifier: MPL-2.0

//! Implements the Ping-Pong Topology described in [VDAF]. This topology assumes there are exactly
//! two aggregators, designated "Leader" and "Helper". This topology is required for implementing
//! the [Distributed Aggregation Protocol][DAP].
//!
//! [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
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

    /// Host is in an unexpected state
    #[error("host state mismatch: in {found} expected {expected}")]
    HostStateMismatch {
        /// The state the host is in.
        found: &'static str,
        /// The state the host expected to be in.
        expected: &'static str,
    },

    /// Message from peer indicates it is in an unexpected state
    #[error("peer message mismatch: message is {found} expected {expected}")]
    PeerMessageMismatch {
        /// The state in the message from the peer.
        found: &'static str,
        /// The message expected from the peer.
        expected: &'static str,
    },

    /// Internal error
    #[error("internal error: {0}")]
    InternalError(&'static str),
}

/// Corresponds to `struct Message` in [VDAF's Ping-Pong Topology][VDAF]. All of the fields of the
/// variants are opaque byte buffers. This is because the ping-pong routines take responsibility for
/// decoding preparation shares and messages, which usually requires having the preparation state.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
#[derive(Clone, PartialEq, Eq)]
pub enum PingPongMessage {
    /// Corresponds to MessageType.initialize.
    Initialize {
        /// The leader's initial preparation share.
        prepare_share: Vec<u8>,
    },
    /// Corresponds to MessageType.continue.
    Continue {
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
            Self::Initialize { .. } => "Initialize",
            Self::Continue { .. } => "Continue",
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
            Self::Initialize { prepare_share } => {
                0u8.encode(bytes)?;
                encode_u32_items(bytes, &(), prepare_share)?;
            }
            Self::Continue {
                prepare_message,
                prepare_share,
            } => {
                1u8.encode(bytes)?;
                encode_u32_items(bytes, &(), prepare_message)?;
                encode_u32_items(bytes, &(), prepare_share)?;
            }
            Self::Finish { prepare_message } => {
                2u8.encode(bytes)?;
                encode_u32_items(bytes, &(), prepare_message)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            Self::Initialize { prepare_share } => Some(1 + 4 + prepare_share.len()),
            Self::Continue {
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
                let prepare_share = decode_u32_items(&(), bytes)?;
                Self::Initialize { prepare_share }
            }
            1 => {
                let prepare_message = decode_u32_items(&(), bytes)?;
                let prepare_share = decode_u32_items(&(), bytes)?;
                Self::Continue {
                    prepare_message,
                    prepare_share,
                }
            }
            2 => {
                let prepare_message = decode_u32_items(&(), bytes)?;
                Self::Finish { prepare_message }
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
/// The obvious implementation of `ping_pong_transition` would be a method on trait
/// [`PingPongTopology`] that returns `(State, Message)`, and then `ContinuedValue::WithMessage`
/// would contain those values. But then DAP implementations would have to store relatively large
/// VDAF prepare shares between rounds of input preparation.
///
/// Instead, this structure stores just the previous round's prepare state and the current round's
/// preprocessed prepare message. Their encoding is much smaller than the `(State, Message)` tuple,
/// which can always be recomputed with [`Self::evaluate`].
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
#[derive(Clone, Debug, Eq)]
pub struct PingPongTransition<
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
    > PingPongTransition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
{
    /// Evaluate this transition to obtain a new [`PingPongState`] and a [`PingPongMessage`] which
    /// should be transmitted to the peer.
    #[allow(clippy::type_complexity)]
    pub fn evaluate(
        &self,
        ctx: &[u8],
        vdaf: &A,
    ) -> Result<
        (
            PingPongState<VERIFY_KEY_SIZE, NONCE_SIZE, A>,
            PingPongMessage,
        ),
        PingPongError,
    > {
        let prepare_message = self
            .current_prepare_message
            .get_encoded()
            .map_err(PingPongError::CodecPrepMessage)?;

        vdaf.prepare_next(
            ctx,
            self.previous_prepare_state.clone(),
            self.current_prepare_message.clone(),
        )
        .map_err(PingPongError::VdafPrepareNext)
        .and_then(|transition| match transition {
            PrepareTransition::Continue(prepare_state, prepare_share) => Ok((
                PingPongState::Continued(prepare_state),
                PingPongMessage::Continue {
                    prepare_message,
                    prepare_share: prepare_share
                        .get_encoded()
                        .map_err(PingPongError::CodecPrepShare)?,
                },
            )),
            PrepareTransition::Finish(output_share) => Ok((
                PingPongState::Finished(output_share),
                PingPongMessage::Finish { prepare_message },
            )),
        })
    }
}

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > PartialEq for PingPongTransition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
{
    fn eq(&self, other: &Self) -> bool {
        self.previous_prepare_state == other.previous_prepare_state
            && self.current_prepare_message == other.current_prepare_message
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A> Encode
    for PingPongTransition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::PrepareState: Encode,
{
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.previous_prepare_state.encode(bytes)?;
        self.current_prepare_message.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.previous_prepare_state.encoded_len()?
                + self.current_prepare_message.encoded_len()?,
        )
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A, PrepareStateDecode>
    ParameterizedDecode<PrepareStateDecode> for PingPongTransition<VERIFY_KEY_SIZE, NONCE_SIZE, A>
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
/// code, and the `Rejected` state is represented as `std::result::Result::Err`, so this enum does
/// not include those variants.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PingPongState<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
> {
    /// Preparation of the report will continue with the enclosed state.
    Continued(A::PrepareState),
    /// Preparation of the report is finished and has yielded the enclosed output share.
    Finished(A::OutputShare),
}

/// Values returned by [`PingPongTopology::leader_continued`] or
/// [`PingPongTopology::helper_continued`].
#[derive(Clone, Debug)]
pub enum PingPongContinuedValue<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
> {
    /// The operation resulted in a new state and a message to transmit to the peer.
    WithMessage {
        /// The transition that will be executed. Call `PingPongTransition::evaluate` to obtain the
        /// next
        /// [`PingPongState`] and a [`PingPongMessage`] to transmit to the peer.
        transition: PingPongTransition<VERIFY_KEY_SIZE, NONCE_SIZE, A>,
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
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
pub trait PingPongTopology<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>:
    Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    /// Specialization of [`PingPongState`] for this VDAF.
    type State;
    /// Specialization of [`PingPongContinuedValue`] for this VDAF.
    type ContinuedValue;
    /// Specializaton of [`PingPongTransition`] for this VDAF.
    type Transition;

    /// Initialize leader state using the leader's input share. Corresponds to
    /// `ping_pong_leader_init` in [VDAF].
    ///
    /// If successful, the returned [`PingPongMessage`] (which will always be
    /// `PingPongMessage::Initialize`) should be transmitted to the helper. The returned
    /// [`PingPongState`] (which will always be `PingPongState::Continued`) should be used by the
    /// leader along with the next [`PingPongMessage`] received from the helper as input to
    /// [`Self::leader_continued`] to advance to the next round.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
    fn leader_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::State, PingPongMessage), PingPongError>;

    /// Initialize helper state using the helper's input share and the leader's first prepare share.
    /// Corresponds to `ping_pong_helper_init` in [VDAF].
    ///
    /// If successful, the returned [`PingPongTransition`] should be evaluated, yielding a
    /// [`PingPongMessage`], which should be transmitted to the leader, and a [`PingPongState`].
    ///
    /// If the state is `PingPongState::Continued`, then it should be used by the helper along with
    /// the next `PingPongMessage` received from the leader as input to [`Self::helper_continued`]
    /// to advance to the next round. The helper may store the `PingPongTransition` between rounds
    /// of preparation instead of the `PingPongState` and `PingPongMessage`.
    ///
    /// If the state is `PingPongState::Finished`, then preparation is finished and the output share
    /// may be accumulated.
    ///
    /// # Errors
    ///
    /// `inbound` must be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
    #[allow(clippy::too_many_arguments)]
    fn helper_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
        inbound: &PingPongMessage,
    ) -> Result<PingPongTransition<VERIFY_KEY_SIZE, NONCE_SIZE, Self>, PingPongError>;

    /// Continue preparation based on the leader's current state and an incoming [`PingPongMessage`]
    /// from the helper. Corresponds to `ping_pong_leader_continued` in [VDAF].
    ///
    /// If successful, the returned [`PingPongContinuedValue`] will either be:
    ///
    /// - `PingPongContinuedValue::WithMessage { transition }`: `transition` should be evaluated,
    ///   yielding a [`PingPongMessage`], which should be transmitted to the helper, and a
    ///   [`PingPongState`].
    ///
    ///   If the state is `PingPongState::Continued`, then it should be used by the leader along
    ///   with the next `PingPongMessage` received from the helper as input to
    ///   [`Self::leader_continued`] to advance to the next round. The leader may store the
    ///   `PingPongTransition` between rounds of preparation instead of of the `PingPongState` and
    ///   `PingPongMessage`.
    ///
    ///   If the state is `PingPongState::Finished`, then preparation is finished and the output
    ///   share may be accumulated.
    ///
    /// - `PingPongContinuedValue::FinishedNoMessage`: preparation is finished and the output share
    ///   may be accumulated. No message needs to be sent to the helper.
    ///
    /// # Errors
    ///
    /// `leader_state` must be `PingPongState::Continued` or the function will fail.
    ///
    /// `inbound` must not be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
    fn leader_continued(
        &self,
        ctx: &[u8],
        leader_state: Self::State,
        aggregation_parameter: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::ContinuedValue, PingPongError>;

    /// PingPongContinue preparation based on the helper's current state and an incoming
    /// [`PingPongMessage`] from the leader. Corresponds to `ping_pong_helper_contnued` in [VDAF].
    ///
    /// If successful, the returned [`PingPongContinuedValue`] will either be:
    ///
    /// - `PingPongContinuedValue::WithMessage { transition }`: `transition` should be evaluated,
    ///   yielding a [`PingPongMessage`], which should be transmitted to the leader, and a
    ///   [`PingPongState`].
    ///
    ///   If the state is `PingPongState::Continued`, then it should be used by the helper along
    ///   with the next `PingPongMessage` received from the leader as input to
    ///   [`Self::helper_continued`] to advance to the next round. The helper may store the
    ///   `PingPongTransition` between rounds of preparation instead of the `PingPongState` and
    ///   `PingPongMessage`.
    ///
    ///   If the state is `PingPongState::Finished`, then preparation is finished and the output
    ///   share may be accumulated.
    ///
    /// - `PingPongContinuedValue::FinishedNoMessage`: preparation is finished and the output share
    ///   may be accumulated. No message needs to be sent to the leader.
    ///
    /// # Errors
    ///
    /// `helper_state` must be `PingPongState::Continued` or the function will fail.
    ///
    /// `inbound` must not be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-08#section-5.8
    fn helper_continued(
        &self,
        ctx: &[u8],
        helper_state: Self::State,
        aggregation_parameter: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::ContinuedValue, PingPongError>;
}

/// Private interfaces for implementing ping-pong
trait PingPongTopologyPrivate<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>:
    PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    fn continued(
        &self,
        ctx: &[u8],
        is_leader: bool,
        host_state: Self::State,
        aggregation_parameter: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::ContinuedValue, PingPongError>;
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A>
    PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE> for A
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
{
    type State = PingPongState<VERIFY_KEY_SIZE, NONCE_SIZE, Self>;
    type ContinuedValue = PingPongContinuedValue<VERIFY_KEY_SIZE, NONCE_SIZE, Self>;
    type Transition = PingPongTransition<VERIFY_KEY_SIZE, NONCE_SIZE, Self>;

    fn leader_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::State, PingPongMessage), PingPongError> {
        self.prepare_init(
            verify_key,
            ctx,
            /* Leader */ 0,
            aggregation_parameter,
            nonce,
            public_share,
            input_share,
        )
        .map_err(PingPongError::VdafPrepareInit)
        .and_then(|(prepare_state, prepare_share)| {
            Ok((
                PingPongState::Continued(prepare_state),
                PingPongMessage::Initialize {
                    prepare_share: prepare_share
                        .get_encoded()
                        .map_err(PingPongError::CodecPrepShare)?,
                },
            ))
        })
    }

    fn helper_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
        inbound: &PingPongMessage,
    ) -> Result<Self::Transition, PingPongError> {
        let (prepare_state, prepare_share) = self
            .prepare_init(
                verify_key,
                ctx,
                /* Helper */ 1,
                aggregation_parameter,
                nonce,
                public_share,
                input_share,
            )
            .map_err(PingPongError::VdafPrepareInit)?;

        let inbound_prepare_share = if let PingPongMessage::Initialize { prepare_share } = inbound {
            Self::PrepareShare::get_decoded_with_param(&prepare_state, prepare_share)
                .map_err(PingPongError::CodecPrepShare)?
        } else {
            return Err(PingPongError::PeerMessageMismatch {
                found: inbound.variant(),
                expected: "initialize",
            });
        };

        let current_prepare_message = self
            .prepare_shares_to_prepare_message(
                ctx,
                aggregation_parameter,
                [inbound_prepare_share, prepare_share],
            )
            .map_err(PingPongError::VdafPrepareSharesToPrepareMessage)?;

        Ok(PingPongTransition {
            previous_prepare_state: prepare_state,
            current_prepare_message,
        })
    }

    fn leader_continued(
        &self,
        ctx: &[u8],
        leader_state: Self::State,
        aggregation_parameter: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::ContinuedValue, PingPongError> {
        self.continued(ctx, true, leader_state, aggregation_parameter, inbound)
    }

    fn helper_continued(
        &self,
        ctx: &[u8],
        helper_state: Self::State,
        aggregation_parameter: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::ContinuedValue, PingPongError> {
        self.continued(ctx, false, helper_state, aggregation_parameter, inbound)
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
        host_state: Self::State,
        aggregation_parameter: &Self::AggregationParam,
        inbound: &PingPongMessage,
    ) -> Result<Self::ContinuedValue, PingPongError> {
        let host_prepare_state = if let PingPongState::Continued(state) = host_state {
            state
        } else {
            return Err(PingPongError::HostStateMismatch {
                found: "finished",
                expected: "continue",
            });
        };

        let (prepare_message, next_peer_prepare_share) = match inbound {
            PingPongMessage::Initialize { .. } => {
                return Err(PingPongError::PeerMessageMismatch {
                    found: inbound.variant(),
                    expected: "continue",
                });
            }
            PingPongMessage::Continue {
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
                    .prepare_shares_to_prepare_message(ctx, aggregation_parameter, prepare_shares)
                    .map_err(PingPongError::VdafPrepareSharesToPrepareMessage)?;

                Ok(PingPongContinuedValue::WithMessage {
                    transition: PingPongTransition {
                        previous_prepare_state: next_prepare_state,
                        current_prepare_message,
                    },
                })
            }
            (PrepareTransition::Finish(output_share), None) => {
                Ok(PingPongContinuedValue::FinishedNoMessage { output_share })
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
                &leader_message,
            )
            .unwrap()
            .evaluate(CTX_STR, &helper)
            .unwrap();

        // 1 round VDAF: helper should finish immediately.
        assert_matches!(helper_state, PingPongState::Finished(_));

        let leader_state = leader
            .leader_continued(CTX_STR, leader_state, &aggregation_param, &helper_message)
            .unwrap();
        // 1 round VDAF: leader should finish when it gets helper message and emit no message.
        assert_matches!(
            leader_state,
            PingPongContinuedValue::FinishedNoMessage { .. }
        );
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
                &leader_message,
            )
            .unwrap()
            .evaluate(CTX_STR, &helper)
            .unwrap();

        // 2 round VDAF, round 1: helper should continue.
        assert_matches!(helper_state, PingPongState::Continued(_));

        let leader_state = leader
            .leader_continued(CTX_STR, leader_state, &aggregation_param, &helper_message)
            .unwrap();
        // 2 round VDAF, round 1: leader should finish and emit a finish message.
        let leader_message = assert_matches!(
            leader_state, PingPongContinuedValue::WithMessage { transition } => {
                let (state, message) = transition.evaluate(CTX_STR,&leader).unwrap();
                assert_matches!(state, PingPongState::Finished(_));
                message
            }
        );

        let helper_state = helper
            .helper_continued(CTX_STR, helper_state, &aggregation_param, &leader_message)
            .unwrap();
        // 2 round vdaf, round 1: helper should finish and emit no message.
        assert_matches!(
            helper_state,
            PingPongContinuedValue::FinishedNoMessage { .. }
        );
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
                &leader_message,
            )
            .unwrap()
            .evaluate(CTX_STR, &helper)
            .unwrap();

        // 3 round VDAF, round 1: helper should continue.
        assert_matches!(helper_state, PingPongState::Continued(_));

        let leader_state = leader
            .leader_continued(CTX_STR, leader_state, &aggregation_param, &helper_message)
            .unwrap();
        // 3 round VDAF, round 1: leader should continue and emit a continue message.
        let (leader_state, leader_message) = assert_matches!(
            leader_state, PingPongContinuedValue::WithMessage { transition } => {
                let (state, message) = transition.evaluate(CTX_STR,&leader).unwrap();
                assert_matches!(state, PingPongState::Continued(_));
                (state, message)
            }
        );

        let helper_state = helper
            .helper_continued(CTX_STR, helper_state, &aggregation_param, &leader_message)
            .unwrap();
        // 3 round vdaf, round 2: helper should finish and emit a finish message.
        let helper_message = assert_matches!(
            helper_state, PingPongContinuedValue::WithMessage { transition } => {
                let (state, message) = transition.evaluate(CTX_STR,&helper).unwrap();
                assert_matches!(state, PingPongState::Finished(_));
                message
            }
        );

        let leader_state = leader
            .leader_continued(CTX_STR, leader_state, &aggregation_param, &helper_message)
            .unwrap();
        // 3 round VDAF, round 2: leader should finish and emit no message.
        assert_matches!(
            leader_state,
            PingPongContinuedValue::FinishedNoMessage { .. }
        );
    }

    #[test]
    fn roundtrip_message() {
        let messages = [
            (
                PingPongMessage::Initialize {
                    prepare_share: Vec::from("prepare share"),
                },
                concat!(
                    "00", // enum discriminant
                    concat!(
                        // prepare_share
                        "0000000d",                   // length
                        "70726570617265207368617265", // contents
                    ),
                ),
            ),
            (
                PingPongMessage::Continue {
                    prepare_message: Vec::from("prepare message"),
                    prepare_share: Vec::from("prepare share"),
                },
                concat!(
                    "01", // enum discriminant
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
                    "02", // enum discriminant
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

    #[test]
    fn roundtrip_transition() {
        // VDAF implementations have tests for encoding/decoding their respective PrepareShare and
        // PrepareMessage types, so we test here using the dummy VDAF.
        let transition = PingPongTransition::<0, 16, dummy::Vdaf> {
            previous_prepare_state: dummy::PrepareState::default(),
            current_prepare_message: (),
        };

        let encoded = transition.get_encoded().unwrap();
        let hex_encoded = hex::encode(&encoded);

        assert_eq!(
            hex_encoded,
            concat!(
                concat!(
                    // previous_prepare_state
                    "00",       // input_share
                    "00000000", // current_round
                ),
                // current_prepare_message (0 length encoding)
            )
        );

        let decoded = PingPongTransition::get_decoded_with_param(&(), &encoded).unwrap();
        assert_eq!(transition, decoded);

        assert_eq!(
            encoded.len(),
            transition.encoded_len().expect("No encoded length hint"),
        );
    }
}
