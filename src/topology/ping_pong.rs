// SPDX-License-Identifier: MPL-2.0

//! Implements the Ping-Pong Topology described in [VDAF]. This topology assumes there are exactly
//! two aggregators, designated "Leader" and "Helper". This topology is required for implementing
//! the [Distributed Aggregation Protocol][DAP].
//!
//! [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
//! [DAP]: https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap

use crate::{
    codec::{decode_u32_items, encode_u32_items, CodecError, Decode, Encode, ParameterizedDecode},
    vdaf::{Aggregator, VdafError, VerifyTransition},
};
use std::fmt::Debug;

/// Errors emitted by this module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PingPongError {
    /// Error running verify_init
    #[error("vdaf.verify_init: {0}")]
    VdafVerifyInit(VdafError),

    /// Error running verifier_shares_to_message
    #[error("vdaf.verifier_shares_to_message {0}")]
    VdafVerifierSharesToMessage(VdafError),

    /// Error running verify_next
    #[error("vdaf.verify_next {0}")]
    VdafVerifyNext(VdafError),

    /// Error encoding or decoding a verifier share
    #[error("encode/decode verifier share {0}")]
    CodecVerifierShare(CodecError),

    /// Error encoding or decoding a verifier message
    #[error("encode/decode verifier message {0}")]
    CodecVerifierMessage(CodecError),

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
/// decoding verifier shares and messages, which usually requires having the verifier state.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
#[derive(Clone, PartialEq, Eq)]
pub enum PingPongMessage {
    /// Corresponds to MessageType.initialize.
    Initialize {
        /// The leader's initial verifier share.
        verifier_share: Vec<u8>,
    },
    /// Corresponds to MessageType.continue.
    Continue {
        /// The current round's verifier message.
        verifier_message: Vec<u8>,
        /// The next round's verifier share.
        verifier_share: Vec<u8>,
    },
    /// Corresponds to MessageType.finish.
    Finish {
        /// The current round's verifier message.
        verifier_message: Vec<u8>,
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
    // verifier shares or messages, because (1) their contents are sensitive and (2) their contents
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
            Self::Initialize { verifier_share } => {
                0u8.encode(bytes)?;
                encode_u32_items(bytes, &(), verifier_share)?;
            }
            Self::Continue {
                verifier_message,
                verifier_share,
            } => {
                1u8.encode(bytes)?;
                encode_u32_items(bytes, &(), verifier_message)?;
                encode_u32_items(bytes, &(), verifier_share)?;
            }
            Self::Finish { verifier_message } => {
                2u8.encode(bytes)?;
                encode_u32_items(bytes, &(), verifier_message)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            Self::Initialize { verifier_share } => Some(1 + 4 + verifier_share.len()),
            Self::Continue {
                verifier_message,
                verifier_share,
            } => Some(1 + 4 + verifier_message.len() + 4 + verifier_share.len()),
            Self::Finish { verifier_message } => Some(1 + 4 + verifier_message.len()),
        }
    }
}

impl Decode for PingPongMessage {
    fn decode(bytes: &mut std::io::Cursor<&[u8]>) -> Result<Self, CodecError> {
        let message_type = u8::decode(bytes)?;
        Ok(match message_type {
            0 => {
                let verifier_share = decode_u32_items(&(), bytes)?;
                Self::Initialize { verifier_share }
            }
            1 => {
                let verifier_message = decode_u32_items(&(), bytes)?;
                let verifier_share = decode_u32_items(&(), bytes)?;
                Self::Continue {
                    verifier_message,
                    verifier_share,
                }
            }
            2 => {
                let verifier_message = decode_u32_items(&(), bytes)?;
                Self::Finish { verifier_message }
            }
            _ => return Err(CodecError::UnexpectedValue),
        })
    }
}

/// A continuation of a state transition in the pong-pong topology. This mostly corresponds to the
/// `ping_pong_continued` and `ping_pong_transition` functions defined in [VDAF].
///
/// # Discussion
///
/// The obvious implementation of `ping_pong_transition` would be a method on [`PingPongTopology`]
/// that returns [`PingPongState`], and then other methods on `PingPongTopology` would use that. But
/// then DAP implementations would have to store relatively large VDAF verifier shares between
/// rounds of input verification.
///
/// Instead, this structure stores just the previous round's verifier state and the current round's
/// preprocessed verifier message. Their encoding is much smaller than the `PingPongState`, which
/// can always be recomputed with [`Self::evaluate`]. Some motivating analysis of relative sizes of
/// protocol objects is [here][sizes].
///
/// If the `PingPongContinuation` evaluates to [`PingPongState::Finished`], then the output share
/// may be accumulated, no message need be sent to the peer aggregator and the
/// `PingPongContinuation` can be dropped.
///
/// If it evaluates to either [`PingPongState::FinishedWithOutbound`] or
/// [`PingPongState::Continued`], then the message should be sent to the peer aggregator. Clients
/// can encode the previously cloned `PingPongContinuation` so that verification can be gracefully
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
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
/// [sizes]: https://github.com/divviup/libprio-rs/pull/683/#issuecomment-1687210371
#[derive(Clone)]
pub struct PingPongContinuation<
    const VERIFY_KEY_SIZE: usize,
    const NONCE_SIZE: usize,
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
>(PingPongContinuationInner<VERIFY_KEY_SIZE, NONCE_SIZE, A>);

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A> Debug
    for PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant = match &self.0 {
            PingPongContinuationInner::OutputShare(_) => "OutputShare",
            PingPongContinuationInner::Transition { .. } => "Transition",
        };
        f.debug_struct("PingPongContinuation")
            .field("variant", &variant)
            .finish_non_exhaustive()
    }
}

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
    ) -> Result<PingPongState<A::VerifyState, A::OutputShare>, PingPongError> {
        match self.0 {
            PingPongContinuationInner::OutputShare(ref output_share) => {
                Ok(PingPongState::Finished {
                    output_share: output_share.clone(),
                })
            }
            PingPongContinuationInner::Transition {
                ref previous_verifier_state,
                ref current_verifier_message,
            } => Self::evaluate_transition(
                ctx,
                vdaf,
                previous_verifier_state,
                current_verifier_message,
            ),
        }
    }

    fn evaluate_transition(
        ctx: &[u8],
        vdaf: &A,
        previous_verifier_state: &A::VerifyState,
        current_verifier_message: &A::VerifierMessage,
    ) -> Result<PingPongState<A::VerifyState, A::OutputShare>, PingPongError> {
        let verifier_message = current_verifier_message
            .get_encoded()
            .map_err(PingPongError::CodecVerifierMessage)?;

        vdaf.verify_next(
            ctx,
            previous_verifier_state.clone(),
            current_verifier_message.clone(),
        )
        .map_err(PingPongError::VdafVerifyNext)
        .and_then(|transition| match transition {
            VerifyTransition::Continue(verifier_state, verifier_share) => {
                Ok(PingPongState::Continued(Continued {
                    verifier_state,
                    message: PingPongMessage::Continue {
                        verifier_message,
                        verifier_share: verifier_share
                            .get_encoded()
                            .map_err(PingPongError::CodecVerifierShare)?,
                    },
                }))
            }
            VerifyTransition::Finish(output_share) => Ok(PingPongState::FinishedWithOutbound {
                output_share,
                message: PingPongMessage::Finish { verifier_message },
            }),
        })
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A> Encode
    for PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::VerifyState: Encode,
{
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match &self.0 {
            PingPongContinuationInner::Transition {
                previous_verifier_state,
                current_verifier_message,
            } => {
                previous_verifier_state.encode(bytes)?;
                current_verifier_message.encode(bytes)
            }
            _ => Err(CodecError::Other(
                "cannot encode anything but a transition".into(),
            )),
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        match &self.0 {
            PingPongContinuationInner::Transition {
                previous_verifier_state,
                current_verifier_message,
            } => Some(
                previous_verifier_state.encoded_len()? + current_verifier_message.encoded_len()?,
            ),
            _ => None,
        }
    }
}

impl<const VERIFY_KEY_SIZE: usize, const NONCE_SIZE: usize, A, VerifierStateDecode>
    ParameterizedDecode<VerifierStateDecode>
    for PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, A>
where
    A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    A::VerifyState: ParameterizedDecode<VerifierStateDecode>,
{
    fn decode_with_param(
        decoding_param: &VerifierStateDecode,
        bytes: &mut std::io::Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let previous_verifier_state = A::VerifyState::decode_with_param(decoding_param, bytes)?;
        let current_verifier_message =
            A::VerifierMessage::decode_with_param(&previous_verifier_state, bytes)?;

        Ok(Self(PingPongContinuationInner::Transition {
            previous_verifier_state,
            current_verifier_message,
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
                    previous_verifier_state: lhs_state,
                    current_verifier_message: lhs_message,
                },
                PingPongContinuationInner::Transition {
                    previous_verifier_state: rhs_state,
                    current_verifier_message: rhs_message,
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
#[derive(Clone)]
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
        /// The last round's verifier state.
        previous_verifier_state: A::VerifyState,
        /// The current round's verifier message.
        current_verifier_message: A::VerifierMessage,
    },
}

/// Verification of the report will continue. Corresponds to the `Continued` state defined in
/// [VDAF's Ping-Pong Topology][VDAF].
///
/// The `message` should be transmitted to the peer aggregator so it can continue verifying the
/// report.
///
/// The `verifier_state` should be used along with the next [`PingPongMessage`] received from the
/// peer as input to the appropriate `PingPongTopology::{leader,helper}_continued` function to
/// advance to the next round.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Continued<S> {
    /// A message for the peer aggregator.
    pub message: PingPongMessage,
    /// The state to which the aggregator has advanced.
    pub verifier_state: S,
}

/// Corresponds to the `State` enumeration implicitly defined in [VDAF's Ping-Pong Topology][VDAF].
/// VDAF describes `Start` and `Rejected` states, but the `Start` state is never instantiated in
/// code, and the `Rejected` state is represented as `std::result::Result::Err`, so this enum does
/// not include those variants.
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PingPongState<P, O> {
    /// Verification of the report will continue.
    Continued(Continued<P>),

    /// Verification of the report is finished. Corresponds to the `FinishedWithOutbound` state
    /// defined in [VDAF's Ping-Pong Topology][VDAF].
    ///
    /// The `message` should be transmitted to the peer aggregator so it can finish verifying the
    /// report.
    ///
    /// The `output_share` may be accumulated by the aggregator.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
    FinishedWithOutbound {
        /// The output share this aggregator verified.
        output_share: O,
        /// A message for the peer aggregator.
        message: PingPongMessage,
    },

    /// Verification of the report is finished. Corresponds to the `Finished` state defined in
    /// [VDAF's Ping-Pong Topology][VDAF].
    ///
    /// The `output_share` may be accumulated by the aggregator. No message need be transmitted to
    /// the peer, which has already finished verifying the report.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
    Finished {
        /// The output share this aggregator verified.
        output_share: O,
    },
}

/// Extension trait on [`crate::vdaf::Aggregator`] which adds the [VDAF Ping-Pong Topology][VDAF].
///
/// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
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
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
    fn leader_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<Continued<Self::VerifyState>, PingPongError>;

    /// Initialize helper state using the helper's input share and the leader's first round verifier
    /// share. Corresponds to `ping_pong_helper_init` in [VDAF].
    ///
    /// On success, the returned [`PingPongContinuation`] should be evaluated, yielding a
    /// [`PingPongState`], which should be handled according to that item's documentation. On
    /// failure, the helper has transitioned to the `Rejected` state. The `PingPongContinuation` may
    /// be stored between rounds of verification instead of the `PingPongState` it evaluates to.
    ///
    /// # Errors
    ///
    /// `leader_message` must be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
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

    /// Continue verification based on the leader's current state and an incoming
    /// [`PingPongMessage`] from the helper. Corresponds to `ping_pong_leader_continued` in [VDAF].
    ///
    /// On success, the returned [`PingPongContinuation`] should be evaluated, yielding a
    /// [`PingPongState`], which should be handled according to that item's documentation. On
    /// failure, the leader has transitioned to the `Rejected` state. The `PingPongContinuation` may
    /// be stored between rounds of verification instead of the `PingPongState` it evaluates to.
    ///
    /// # Errors
    ///
    /// `helper_message` must not be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
    fn leader_continued(
        &self,
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        leader_verifier_state: Self::VerifyState,
        helper_message: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError>;

    /// Continue verification based on the helper's current state and an incoming
    /// [`PingPongMessage`] from the leader. Corresponds to `ping_pong_helper_contnued` in [VDAF].
    ///
    /// On success, the returned [`PingPongContinuation`] should be evaluated, yielding a
    /// [`PingPongState`], which should be handled according to that item's documentation. On
    /// failure, the helper has transitioned to the `Rejected` state. The `PingPongContinuation` may
    /// be stored between rounds of verification instead of the `PingPongState` it evaluates to.
    ///
    /// # Errors
    ///
    /// `leader_message` must not be `PingPongMessage::Initialize` or the function will fail.
    ///
    /// [VDAF]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-5.7.1
    fn helper_continued(
        &self,
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        helper_verifier_state: Self::VerifyState,
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
        host_verifier_state: Self::VerifyState,
        peer_message: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError>;
}

impl<
        const VERIFY_KEY_SIZE: usize,
        const NONCE_SIZE: usize,
        A: Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE>,
    > PingPongTopology<VERIFY_KEY_SIZE, NONCE_SIZE> for A
{
    type PingPongState = PingPongState<A::VerifyState, A::OutputShare>;
    type PingPongContinuation = PingPongContinuation<VERIFY_KEY_SIZE, NONCE_SIZE, Self>;

    fn leader_initialized(
        &self,
        verify_key: &[u8; VERIFY_KEY_SIZE],
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        nonce: &[u8; NONCE_SIZE],
        public_share: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<Continued<Self::VerifyState>, PingPongError> {
        self.verify_init(
            verify_key,
            ctx,
            /* Leader */ 0,
            aggregation_parameter,
            nonce,
            public_share,
            input_share,
        )
        .map_err(PingPongError::VdafVerifyInit)
        .and_then(|(verifier_state, verifier_share)| {
            Ok(Continued {
                verifier_state,
                message: PingPongMessage::Initialize {
                    verifier_share: verifier_share
                        .get_encoded()
                        .map_err(PingPongError::CodecVerifierShare)?,
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
        let (verifier_state, verifier_share) = self
            .verify_init(
                verify_key,
                ctx,
                /* Helper */ 1,
                aggregation_parameter,
                nonce,
                public_share,
                input_share,
            )
            .map_err(PingPongError::VdafVerifyInit)?;

        let leader_verifier_share =
            if let PingPongMessage::Initialize { verifier_share } = leader_message {
                Self::VerifierShare::get_decoded_with_param(&verifier_state, verifier_share)
                    .map_err(PingPongError::CodecVerifierShare)?
            } else {
                return Err(PingPongError::PeerMessageMismatch {
                    found: leader_message.variant(),
                    expected: "initialize",
                });
            };

        let current_verifier_message = self
            .verifier_shares_to_message(
                ctx,
                aggregation_parameter,
                [leader_verifier_share, verifier_share],
            )
            .map_err(PingPongError::VdafVerifierSharesToMessage)?;

        Ok(PingPongContinuationInner::Transition {
            previous_verifier_state: verifier_state,
            current_verifier_message,
        }
        .into())
    }

    fn leader_continued(
        &self,
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        leader_state: Self::VerifyState,
        inbound: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError> {
        self.continued(ctx, true, aggregation_parameter, leader_state, inbound)
    }

    fn helper_continued(
        &self,
        ctx: &[u8],
        aggregation_parameter: &Self::AggregationParam,
        helper_state: Self::VerifyState,
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
        host_verifier_state: Self::VerifyState,
        inbound: &PingPongMessage,
    ) -> Result<Self::PingPongContinuation, PingPongError> {
        let (verifier_message, next_peer_verifier_share) = match inbound {
            PingPongMessage::Initialize { .. } => {
                return Err(PingPongError::PeerMessageMismatch {
                    found: inbound.variant(),
                    expected: "continue",
                });
            }
            PingPongMessage::Continue {
                verifier_message,
                verifier_share,
            } => (verifier_message, Some(verifier_share)),
            PingPongMessage::Finish { verifier_message } => (verifier_message, None),
        };

        let verifier_message =
            Self::VerifierMessage::get_decoded_with_param(&host_verifier_state, verifier_message)
                .map_err(PingPongError::CodecVerifierMessage)?;
        let host_verify_transition = self
            .verify_next(ctx, host_verifier_state, verifier_message)
            .map_err(PingPongError::VdafVerifyNext)?;

        match (host_verify_transition, next_peer_verifier_share) {
            (
                VerifyTransition::Continue(next_verifier_state, next_host_verifier_share),
                Some(next_peer_verifier_share),
            ) => {
                let next_peer_verifier_share = Self::VerifierShare::get_decoded_with_param(
                    &next_verifier_state,
                    next_peer_verifier_share,
                )
                .map_err(PingPongError::CodecVerifierShare)?;
                let mut verifier_shares = [next_peer_verifier_share, next_host_verifier_share];
                if is_leader {
                    verifier_shares.reverse();
                }
                let current_verifier_message = self
                    .verifier_shares_to_message(ctx, aggregation_parameter, verifier_shares)
                    .map_err(PingPongError::VdafVerifierSharesToMessage)?;

                Ok(PingPongContinuationInner::Transition {
                    previous_verifier_state: next_verifier_state,
                    current_verifier_message,
                }
                .into())
            }
            (VerifyTransition::Finish(output_share), None) => {
                Ok(PingPongContinuationInner::OutputShare(output_share).into())
            }
            (VerifyTransition::Continue(_, _), None) => Err(PingPongError::PeerMessageMismatch {
                found: inbound.variant(),
                expected: "continue",
            }),
            (VerifyTransition::Finish(_), Some(_)) => Err(PingPongError::PeerMessageMismatch {
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
            verifier_state: leader_state,
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
            verifier_state: leader_state,
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
            Continued { verifier_state, message }
        ) => (verifier_state, message));

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
            verifier_state: leader_state,
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
            Continued { verifier_state, message }
        ) => (verifier_state, message));

        let leader_state = leader
            .leader_continued(CTX_STR, &aggregation_param, leader_state, &helper_message)
            .unwrap()
            .evaluate(CTX_STR, &leader)
            .unwrap();
        // 3 round VDAF, round 1: leader should continue and emit a continue message.
        let (leader_state, leader_message) = assert_matches!(leader_state, PingPongState::Continued(
            Continued { verifier_state, message }
        ) => (verifier_state, message));

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
                    verifier_share: Vec::from("verifier share"),
                },
                concat!(
                    "00", // enum discriminant
                    concat!(
                        // verifier_share
                        "0000000e",                     // length
                        "7665726966696572207368617265", // contents
                    ),
                ),
            ),
            (
                PingPongMessage::Continue {
                    verifier_message: Vec::from("verifier message"),
                    verifier_share: Vec::from("verifier share"),
                },
                concat!(
                    "01", // enum discriminant
                    concat!(
                        // verifier_message
                        "00000010",                         // length
                        "7665726966696572206d657373616765", // contents
                    ),
                    concat!(
                        // verifier_share
                        "0000000e",                     // length
                        "7665726966696572207368617265", // contents
                    ),
                ),
            ),
            (
                PingPongMessage::Finish {
                    verifier_message: Vec::from("verifier message"),
                },
                concat!(
                    "02", // enum discriminant
                    concat!(
                        // verifier_message
                        "00000010",                         // length
                        "7665726966696572206d657373616765", // contents
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
        // VDAF implementations have tests for encoding/decoding their respective VerifierShare and
        // VerifierMessage types, so we test here using the dummy VDAF.
        let continuation =
            PingPongContinuation::<0, 16, dummy::Vdaf>(PingPongContinuationInner::Transition {
                previous_verifier_state: dummy::VerifierState::default(),
                current_verifier_message: (),
            });

        let encoded = continuation.get_encoded().unwrap();
        let hex_encoded = hex::encode(&encoded);

        assert_eq!(
            hex_encoded,
            concat!(
                concat!(
                    // previous_verifier_state
                    "00",       // input_share
                    "00000000", // current_round
                ),
                // current_verifier_message (0 length encoding)
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
