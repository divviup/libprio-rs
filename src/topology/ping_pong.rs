// SPDX-License-Identifier: MPL-2.0

//! Implements the Ping-Pong Topology described in [VDAF]. This topology assumes there are exactly
//! two aggregators, designated "Leader" and "Helper". This topology is required for implementing
//! the [Distributed Aggregation Protocol][DAP].
//!
//! [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
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
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
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

/// A continuation of a state transition in the pong-pong topology. This mostly corresponds to the
/// `ping_pong_continue` and `ping_pong_transition` functions defined in [VDAF].
///
/// # Discussion
///
/// The obvious implementation of `ping_pong_transition` would be a method on [`PingPongTopology`]
/// that returns [`PingPongState`], and then other methods on `PingPongTopology` would use that. But
/// then DAP implementations would have to store relatively large VDAF prepare shares between rounds
/// of input preparation.
///
/// Instead, this structure stores just the previous round's prepare state and the current round's
/// preprocessed prepare message. Their encoding is much smaller than the `PingPongState`, which can
/// always be recomputed with [`Self::evaluate`]. Some motivating analysis of relative sizes of
/// protocol objects is [here][sizes].
///
/// If the `PingPongContinuation` evaluates to [`PingPongState::Finished`], then the output share
/// may be accumulated, no message need be sent to the peer aggregator and the
/// `PingPongContinuation` can be dropped.
///
/// If it evaluates to either [`PingPongState::FinishedWithOutbound`] or
/// [`PingPongState::Continued`], then the message should be sent to the peer aggregator. Clients
/// can encode the previously cloned `PingPongContinuation` so that preparation can be gracefully
/// resumed later.
///
/// See [`PingPongState`]'s documentation for detailed discussion of how to handle each of its
/// variants.
///
/// It is never necessary to encode or store a `PingPongContinuation` which evaluates to
/// `PingPongState::Finished`, so [`PingPongContinuation::encode`] fails in this case.
///
/// In VDAF's definition of `ping_pong_transition`, the function can only return states `Continued`
/// and `FinishedWithOutbound`, but because we need this to also yield `Finished` in some cases, it
/// also captures parts of `ping_pong_continued`.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
/// [sizes]: https://github.com/divviup/libprio-rs/pull/683/#issuecomment-1687210371
#[derive(Clone, Debug)]
pub struct PingPongContinuation<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
>(PingPongContinuationInner<VERIFY_KEY_SIZE, NONCE_SIZE, A>);

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
{
    /// Evaluate this continuation to obtain a new [`PingPongState`], which should be handled
    /// according to that item's documentation.
    pub fn evaluate(
        &self,
        ctx: &[u8],
        vdaf: &A,
    ) -> Result<PingPongState<A::PrepareState, A::OutputShare>, PingPongError> {
        match self.0 {
            PingPongContinuationInner::OutputShare(ref output_share) => {
                Ok(PingPongState::Finished {
                    output_share: output_share.clone(),
                })
            }
            PingPongContinuationInner::Transition {
                ref previous_prepare_state,
                ref current_prepare_message,
            } => Self::evaluate_transition(
                ctx,
                vdaf,
                previous_prepare_state,
                current_prepare_message,
            ),
        }
    }

    fn evaluate_transition(
        ctx: &[u8],
        vdaf: &A,
        previous_prepare_state: &A::PrepareState,
        current_prepare_message: &A::PrepareMessage,
    ) -> Result<PingPongState<A::PrepareState, A::OutputShare>, PingPongError> {
        let prepare_message = current_prepare_message
            .get_encoded()
            .map_err(PingPongError::CodecPrepMessage)?;

        vdaf.prepare_next(
            ctx,
            previous_prepare_state.clone(),
            current_prepare_message.clone(),
        )
        .map_err(PingPongError::VdafPrepareNext)
        .and_then(|transition| match transition {
            PrepareTransition::Continue(prepare_state, prepare_share) => {
                Ok(PingPongState::Continued(Continued {
                    prepare_state,
                    message: PingPongMessage::Continue {
                        prepare_message,
                        prepare_share: prepare_share
                            .get_encoded()
                            .map_err(PingPongError::CodecPrepShare)?,
                    },
                }))
            }
            PrepareTransition::Finish(output_share) => Ok(PingPongState::FinishedWithOutbound {
                output_share,
                message: PingPongMessage::Finish { prepare_message },
            }),
        })
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A> Encode
    for PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::PrepareState: Encode,
{
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match &self.0 {
            PingPongContinuationInner::Transition {
                previous_prepare_state,
                current_prepare_message,
            } => {
                previous_prepare_state.encode(bytes)?;
                current_prepare_message.encode(bytes)
            }
            _ => Err(CodecError::Other(
                "cannot encode anything but a transition".into(),
            )),
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        match &self.0 {
            PingPongContinuationInner::Transition {
                previous_prepare_state,
                current_prepare_message,
            } => Some(
                previous_prepare_state.encoded_len()? + current_prepare_message.encoded_len()?,
            ),
            _ => None,
        }
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A, PrepareStateDecode>
    ParameterizedDecode<PrepareStateDecode> for PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::PrepareState: ParameterizedDecode<PrepareStateDecode>,
{
    fn decode_with_param(
        decoding_param: &PrepareStateDecode,
        bytes: &mut std::io::Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let previous_prepare_state = A::PrepareState::decode_with_param(decoding_param, bytes)?;
        let current_prepare_message =
            A::PrepareMessage::decode_with_param(&previous_prepare_state, bytes)?;

        Ok(Self(PingPongContinuationInner::Transition {
            previous_prepare_state,
            current_prepare_message,
        }))
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A> PartialEq
    for PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::OutputShare: PartialEq + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (
                PingPongContinuationInner::OutputShare(self_share),
                PingPongContinuationInner::OutputShare(other_share),
            ) => self_share == other_share,
            (
                PingPongContinuationInner::Transition {
                    previous_prepare_state: lhs_state,
                    current_prepare_message: lhs_message,
                },
                PingPongContinuationInner::Transition {
                    previous_prepare_state: rhs_state,
                    current_prepare_message: rhs_message,
                },
            ) => lhs_state == rhs_state && lhs_message == rhs_message,
            _ => false,
        }
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A> Eq
    for PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::OutputShare: PartialEq + Eq,
{
}

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > From<PingPongContinuationInner<VERIFY_KEY_SIZE, NONCE_SIZE, A>>
    for PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
{
    fn from(value: PingPongContinuationInner<VERIFY_KEY_SIZE, NONCE_SIZE, A>) -> Self {
        Self(value)
    }
}

/// PingPongContinuationInner hides the internals of [`PingPongContinuation`] from external clients.
#[derive(Clone, Debug)]
enum PingPongContinuationInner<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
> {
    /// The continuation will yield `PingPongState::Finished`.
    OutputShare(A::OutputShare),
    /// The continuation will yield one of `PingPongState::Continued` or
    /// `PingPongState::FinishedWithOutbound`.
    Transition {
        /// The last round's prepare state.
        previous_prepare_state: A::PrepareState,
        /// The current round's prepare message.
        current_prepare_message: A::PrepareMessage,
    },
}

/// Preparation of the report will continue. Corresponds to the `Continued` state defined in
/// [VDAF's Ping-Pong Topology][VDAF].
///
/// The `message` should be transmitted to the peer aggregator so it can continue preparing the
/// report.
///
/// The `prepare_state` should be used along with the next [`PingPongMessage`] received from the
/// peer as input to the appropriate `PingPongTopology::{leader,helper}_continued` function to
/// advance to the next round.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Continued<P> {
    /// A message for the peer aggregator.
    pub message: PingPongMessage,
    /// The state to which the aggregator has advanced.
    pub prepare_state: P,
}

/// Corresponds to the `State` enumeration implicitly defined in [VDAF's Ping-Pong Topology][VDAF].
/// VDAF describes `Start` and `Rejected` states, but the `Start` state is never instantiated in
/// code, and the `Rejected` state is represented as `std::result::Result::Err`, so this enum does
/// not include those variants.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PingPongState<P, O> {
    /// Preparation of the report will continue.
    Continued(Continued<P>),

    /// Preparation of the report is finished. Corresponds to the `FinishedWithOutbound` state
    /// defined in [VDAF's Ping-Pong Topology][VDAF].
    ///
    /// The `message` should be transmitted to the peer aggregator so it can finish preparing the
    /// report.
    ///
    /// The `output_share` may be accumulated by the aggregator.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
    FinishedWithOutbound {
        /// The output share this aggregator prepared.
        output_share: O,
        /// A message for the peer aggregator.
        message: PingPongMessage,
    },

    /// Preparation of the report is finished. Corresponds to the `Finished` state defined in
    /// [VDAF's Ping-Pong Topology][VDAF].
    ///
    /// The `output_share` may be accumulated by the aggregator. No message need be transmitted to
    /// the peer, which has already finished preparing the report.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
    Finished {
        /// The output share this aggregator prepared.
        output_share: O,
    },
}

/// Extension trait on [`crate::vdaf::Aggregator`] which adds the [VDAF Ping-Pong Topology][VDAF].
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
pub trait PingPongTopology<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>:
    Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    /// Specialization of [`PingPongState`] for this VDAF.
    type PingPongState;

    /// Specialization of [`PingPongContinuation`] for this VDAF.
    type PingPongContinuation;

    /// Initialize leader state using the leader's input share. Corresponds to
    /// `ping_pong_leader_init` in [VDAF].
    ///
    /// On success, the leader has transitioned to the [`Continued`] state, which should be handled
    /// according to that item's documentation. On failure, the leader has transitioned to the
    /// `Rejected` state.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
    fn leader_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<Continued<Self::PrepareState>, PingPongError>;

    /// Initialize helper state using the helper's input share and the leader's first round prepare
    /// share. Corresponds to `ping_pong_helper_init` in [VDAF].
    ///
    /// On success, the returned [`PingPongContinuation`] should be evaluated, yielding a
    /// [`PingPongState`], which should be handled according to that item's documentation. On
    /// failure, the helper has transitioned to the `Rejected` state. The `PingPongContinuation` may
    /// be stored between rounds of preparation instead of the `PingPongState` it evaluates to.
    ///
    /// # Errors
    ///
    /// `leader_message` must be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
    #[allow(clippy::too_many_arguments)]
    fn helper_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
        leader_message: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError>;

    /// Continue preparation based on the leader's current state and an incoming [`PingPongMessage`]
    /// from the helper. Corresponds to `ping_pong_leader_continued` in [VDAF].
    ///
    /// On success, the returned [`PingPongContinuation`] should be evaluated, yielding a
    /// [`PingPongState`], which should be handled according to that item's documentation. On
    /// failure, the leader has transitioned to the `Rejected` state. The `PingPongContinuation` may
    /// be stored between rounds of preparation instead of the `PingPongState` it evaluates to.
    ///
    /// # Errors
    ///
    /// `helper_message` must not be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
    fn leader_continued(
        &self,
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        leader_prepare_state: Self::PrepareState,
        helper_message: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError>;

    /// Continue preparation based on the helper's current state and an incoming [`PingPongMessage`]
    /// from the leader. Corresponds to `ping_pong_helper_contnued` in [VDAF].
    ///
    /// On success, the returned [`PingPongContinuation`] should be evaluated, yielding a
    /// [`PingPongState`], which should be handled according to that item's documentation. On
    /// failure, the helper has transitioned to the `Rejected` state. The `PingPongContinuation` may
    /// be stored between rounds of preparation instead of the `PingPongState` it evaluates to.
    ///
    /// # Errors
    ///
    /// `leader_message` must not be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-15#section-5.7.1
    fn helper_continued(
        &self,
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        helper_prepare_state: Self::PrepareState,
        leader_message: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError>;
}

/// Private interfaces for implementing ping-pong
trait PingPongTopologyPrivate<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize>:
    PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE>
{
    fn continued(
        &self,
        ctx: &[u8],
        is_leader: bool,
        aggregation_parameter: &Self::AggregationParam,
        host_prepare_state: Self::PrepareState,
        peer_message: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError>;
}

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE> for A
{
    type PingPongState = PingPongState<A::PrepareState, A::OutputShare>;
    type PingPongContinuation = PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, Self>;

    fn leader_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<Continued<Self::PrepareState>, PingPongError> {
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
            Ok(Continued {
                prepare_state,
                message: PingPongMessage::Initialize {
                    prepare_share: prepare_share
                        .get_encoded()
                        .map_err(PingPongError::CodecPrepShare)?,
                },
            })
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
        leader_message: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError> {
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

        let leader_prepare_share =
            if let PingPongMessage::Initialize { prepare_share } = leader_message {
                Self::PrepareShare::get_decoded_with_param(&prepare_state, prepare_share)
                    .map_err(PingPongError::CodecPrepShare)?
            } else {
                return Err(PingPongError::PeerMessageMismatch {
                    found: leader_message.variant(),
                    expected: "initialize",
                });
            };

        let current_prepare_message = self
            .prepare_shares_to_prepare_message(
                ctx,
                aggregation_parameter,
                [leader_prepare_share, prepare_share],
            )
            .map_err(PingPongError::VdafPrepareSharesToPrepareMessage)?;

        Ok(PingPongContinuationInner::Transition {
            previous_prepare_state: prepare_state,
            current_prepare_message,
        }
        .into())
    }

    fn leader_continued(
        &self,
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        leader_state: Self::PrepareState,
        inbound: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError> {
        self.continued(ctx, true, aggregation_parameter, leader_state, inbound)
    }

    fn helper_continued(
        &self,
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        helper_state: Self::PrepareState,
        inbound: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError> {
        self.continued(ctx, false, aggregation_parameter, helper_state, inbound)
    }
}

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > PingPongTopologyPrivate<VERIFY_KEY_SIZE, NONCE_SIZE> for A
{
    fn continued(
        &self,
        ctx: &[u8],
        is_leader: bool,
        aggregation_parameter: &Self::AggregationParam,
        host_prepare_state: Self::PrepareState,
        inbound: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError> {
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

                Ok(PingPongContinuationInner::Transition {
                    previous_prepare_state: next_prepare_state,
                    current_prepare_message,
                }
                .into())
            }
            (PrepareTransition::Finish(output_share), None) => {
                Ok(PingPongContinuationInner::OutputShare(output_share).into())
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
        let Continued {
            prepare_state: leader_state,
            message: leader_message,
        } = leader
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
        let helper_state = helper
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
        let helper_message = assert_matches!(helper_state, PingPongState::FinishedWithOutbound {
            message, ..
        } => message);

        let leader_state = leader
            .leader_continued(CTX_STR, &aggregation_param, leader_state, &helper_message)
            .unwrap()
            .evaluate(CTX_STR, &leader)
            .unwrap();
        // 1 round VDAF: leader should finish when it gets helper message and emit no message.
        assert_matches!(leader_state, PingPongState::Finished { .. });
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
        let Continued {
            prepare_state: leader_state,
            message: leader_message,
        } = leader
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
        let helper_state = helper
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
        let (helper_state, helper_message) = assert_matches!(helper_state, PingPongState::Continued(
            Continued { prepare_state, message }
        ) => (prepare_state, message));

        let leader_state = leader
            .leader_continued(CTX_STR, &aggregation_param, leader_state, &helper_message)
            .unwrap()
            .evaluate(CTX_STR, &leader)
            .unwrap();
        // 2 round VDAF, round 1: leader should finish and emit a finish message.
        let leader_message = assert_matches!(leader_state, PingPongState::FinishedWithOutbound{
            message, ..
        } => message);

        let helper_state = helper
            .helper_continued(CTX_STR, &aggregation_param, helper_state, &leader_message)
            .unwrap()
            .evaluate(CTX_STR, &helper)
            .unwrap();
        // 2 round vdaf, round 1: helper should finish and emit no message.
        assert_matches!(helper_state, PingPongState::Finished { .. });
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
        let Continued {
            prepare_state: leader_state,
            message: leader_message,
        } = leader
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
        let helper_state = helper
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
        let (helper_state, helper_message) = assert_matches!(helper_state, PingPongState::Continued(
            Continued { prepare_state, message }
        ) => (prepare_state, message));

        let leader_state = leader
            .leader_continued(CTX_STR, &aggregation_param, leader_state, &helper_message)
            .unwrap()
            .evaluate(CTX_STR, &leader)
            .unwrap();
        // 3 round VDAF, round 1: leader should continue and emit a continue message.
        let (leader_state, leader_message) = assert_matches!(leader_state, PingPongState::Continued(
            Continued { prepare_state, message }
        ) => (prepare_state, message));

        let helper_state = helper
            .helper_continued(CTX_STR, &aggregation_param, helper_state, &leader_message)
            .unwrap()
            .evaluate(CTX_STR, &helper)
            .unwrap();
        // 3 round vdaf, round 2: helper should finish and emit a finish message.
        let helper_message = assert_matches!(helper_state, PingPongState::FinishedWithOutbound{
            message, ..
        } => message);

        let leader_state = leader
            .leader_continued(CTX_STR, &aggregation_param, leader_state, &helper_message)
            .unwrap()
            .evaluate(CTX_STR, &leader)
            .unwrap();
        // 3 round VDAF, round 2: leader should finish and emit no message.
        assert_matches!(leader_state, PingPongState::Finished { .. });
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
    fn roundtrip_continuation() {
        // VDAF implementations have tests for encoding/decoding their respective PrepareShare and
        // PrepareMessage types, so we test here using the dummy VDAF.
        let continuation =
            PingPongContinuation::<0, 16, dummy::Vdaf>(PingPongContinuationInner::Transition {
                previous_prepare_state: dummy::PrepareState::default(),
                current_prepare_message: (),
            });

        let encoded = continuation.get_encoded().unwrap();
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

        let decoded = PingPongContinuation::get_decoded_with_param(&(), &encoded).unwrap();
        assert_eq!(continuation, decoded);

        assert_eq!(
            encoded.len(),
            continuation.encoded_len().expect("No encoded length hint"),
        );
    }

    #[test]
    fn roundtrip_continuation_output_share() {
        let continuation = PingPongContinuation::<0, 16, dummy::Vdaf>(
            PingPongContinuationInner::OutputShare(dummy::OutputShare(0)),
        );

        // encoding an output share should fail
        continuation.get_encoded().unwrap_err();
    }
}
