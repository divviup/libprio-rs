// SPDX-License-Identifier: MPL-2.0

//! Implementation of a dummy VDAF which conforms to the specification in [draft-irtf-cfrg-vdaf-18]
//! but does nothing. Useful for testing.
//!
//! [draft-irtf-cfrg-vdaf-18]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/18/

use crate::{
    codec::{CodecError, Decode, Encode},
    vdaf::{self, Aggregatable, VdafError, VerifyTransition},
};
use rand::random;
use std::{fmt::Debug, io::Cursor, sync::Arc};

/// The Dummy VDAF does summation modulus 256 so we can predict aggregation results.
const MODULUS: u64 = u8::MAX as u64 + 1;

type ArcVerifyInitFn =
    Arc<dyn Fn(&AggregationParam) -> Result<(), VdafError> + 'static + Send + Sync>;
type ArcVerifyNextFn = Arc<
    dyn Fn(&VerifierState) -> Result<VerifyTransition<Vdaf, 0, 16>, VdafError>
        + 'static
        + Send
        + Sync,
>;

/// Dummy VDAF that does nothing.
#[derive(Clone)]
pub struct Vdaf {
    verify_init_fn: ArcVerifyInitFn,
    verify_next_fn: ArcVerifyNextFn,
}

impl Debug for Vdaf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vdaf")
            .field("verify_init_fn", &"[redacted]")
            .field("verify_next_fn", &"[redacted]")
            .finish()
    }
}

impl Vdaf {
    /// The length of the verify key parameter for fake VDAF instantiations.
    pub const VERIFY_KEY_LEN: usize = 0;

    /// Construct a new instance of the dummy VDAF.
    pub fn new(rounds: u32) -> Self {
        Self {
            verify_init_fn: Arc::new(|_| -> Result<(), VdafError> { Ok(()) }),
            verify_next_fn: Arc::new(
                move |state| -> Result<VerifyTransition<Self, 0, 16>, VdafError> {
                    let new_round = state.current_round + 1;
                    if new_round == rounds {
                        Ok(VerifyTransition::Finish(OutputShare(u64::from(
                            state.input_share,
                        ))))
                    } else {
                        Ok(VerifyTransition::Continue(
                            VerifierState {
                                current_round: new_round,
                                ..*state
                            },
                            (),
                        ))
                    }
                },
            ),
        }
    }

    /// Provide an alternate implementation of [`vdaf::Aggregator::verify_init`].
    pub fn with_verify_init_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&AggregationParam) -> Result<(), VdafError> + Send + Sync + 'static,
    {
        self.verify_init_fn = Arc::new(f);
        self
    }

    /// Provide an alternate implementation of [`vdaf::Aggregator::verify_next`].
    pub fn with_verify_next_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&VerifierState) -> Result<VerifyTransition<Self, 0, 16>, VdafError>
            + Send
            + Sync
            + 'static,
    {
        self.verify_next_fn = Arc::new(f);
        self
    }
}

impl Default for Vdaf {
    fn default() -> Self {
        Self::new(1)
    }
}

impl vdaf::Vdaf for Vdaf {
    type Measurement = u8;
    type AggregateResult = u64;
    type AggregationParam = AggregationParam;
    type PublicShare = ();
    type InputShare = InputShare;
    type OutputShare = OutputShare;
    type AggregateShare = AggregateShare;

    fn algorithm_id(&self) -> u32 {
        0xFFFF0000
    }

    fn num_aggregators(&self) -> usize {
        2
    }
}

impl vdaf::Aggregator<0, 16> for Vdaf {
    type VerifyState = VerifierState;
    type VerifierShare = ();
    type VerifierMessage = ();

    fn verify_init(
        &self,
        _verify_key: &[u8; 0],
        _ctx: &[u8],
        _: usize,
        aggregation_param: &Self::AggregationParam,
        _nonce: &[u8; 16],
        _: &Self::PublicShare,
        input_share: &Self::InputShare,
    ) -> Result<(Self::VerifyState, Self::VerifierShare), VdafError> {
        (self.verify_init_fn)(aggregation_param)?;
        Ok((
            VerifierState {
                input_share: input_share.0,
                current_round: 0,
            },
            (),
        ))
    }

    fn verifier_shares_to_message<M: IntoIterator<Item = Self::VerifierShare>>(
        &self,
        _ctx: &[u8],
        _: &Self::AggregationParam,
        _: M,
    ) -> Result<Self::VerifierMessage, VdafError> {
        Ok(())
    }

    fn verify_next(
        &self,
        _ctx: &[u8],
        state: Self::VerifyState,
        _: Self::VerifierMessage,
    ) -> Result<VerifyTransition<Self, 0, 16>, VdafError> {
        (self.verify_next_fn)(&state)
    }

    fn aggregate_init(&self, _agg_param: &Self::AggregationParam) -> Self::AggregateShare {
        AggregateShare(0)
    }

    fn is_agg_param_valid(_cur: &Self::AggregationParam, _prev: &[Self::AggregationParam]) -> bool {
        true
    }
}

impl vdaf::Client<16> for Vdaf {
    fn shard(
        &self,
        _ctx: &[u8],
        measurement: &Self::Measurement,
        _nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Self::InputShare>), VdafError> {
        let first_input_share = random();
        let (second_input_share, _) = measurement.overflowing_sub(first_input_share);
        Ok((
            (),
            Vec::from([
                InputShare(first_input_share),
                InputShare(second_input_share),
            ]),
        ))
    }
}

impl vdaf::Collector for Vdaf {
    fn unshard<M: IntoIterator<Item = Self::AggregateShare>>(
        &self,
        aggregation_param: &Self::AggregationParam,
        agg_shares: M,
        _num_measurements: usize,
    ) -> Result<Self::AggregateResult, VdafError> {
        Ok(agg_shares
            .into_iter()
            .fold(0, |acc, share| (acc + share.0) % MODULUS)
            // Sum in the aggregation parameter so that collections over the same measurements with
            // varying parameters will yield predictable but distinct results.
            + u64::from(aggregation_param.0))
    }
}

/// A dummy input share.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct InputShare(pub u8);

impl Encode for InputShare {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.0.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.0.encoded_len()
    }
}

impl Decode for InputShare {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(u8::decode(bytes)?))
    }
}

/// Dummy aggregation parameter.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AggregationParam(pub u8);

impl Encode for AggregationParam {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.0.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.0.encoded_len()
    }
}

impl Decode for AggregationParam {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(u8::decode(bytes)?))
    }
}

/// Dummy output share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputShare(pub u64);

impl Decode for OutputShare {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(u64::decode(bytes)?))
    }
}

impl Encode for OutputShare {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.0.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.0.encoded_len()
    }
}

/// Dummy verifier state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VerifierState {
    input_share: u8,
    current_round: u32,
}

impl Encode for VerifierState {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.input_share.encode(bytes)?;
        self.current_round.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.input_share.encoded_len()? + self.current_round.encoded_len()?)
    }
}

impl Decode for VerifierState {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let input_share = u8::decode(bytes)?;
        let current_round = u32::decode(bytes)?;

        Ok(Self {
            input_share,
            current_round,
        })
    }
}

/// Dummy aggregate share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AggregateShare(pub u64);

impl Aggregatable for AggregateShare {
    type OutputShare = OutputShare;

    fn merge(&mut self, other: &Self) -> Result<(), VdafError> {
        self.0 = (self.0 + other.0) % MODULUS;
        Ok(())
    }

    fn accumulate(&mut self, out_share: &Self::OutputShare) -> Result<(), VdafError> {
        self.0 = (self.0 + out_share.0) % MODULUS;
        Ok(())
    }
}

impl From<OutputShare> for AggregateShare {
    fn from(out_share: OutputShare) -> Self {
        Self(out_share.0)
    }
}

impl Decode for AggregateShare {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(u64::decode(bytes)?))
    }
}

impl Encode for AggregateShare {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.0.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.0.encoded_len()
    }
}

/// Returns the aggregate result that the dummy VDAF would compute over the provided measurements,
/// for the provided aggregation parameter.
pub fn expected_aggregate_result<M>(aggregation_parameter: u8, measurements: M) -> u64
where
    M: IntoIterator<Item = u8>,
{
    (measurements.into_iter().map(u64::from).sum::<u64>()) % MODULUS
        + u64::from(aggregation_parameter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vdaf::{test_utils::run_vdaf_sharded, Client};
    use rand::{rng, RngExt};

    fn run_test(rounds: u32, aggregation_parameter: u8) {
        let vdaf = Vdaf::new(rounds);
        let mut verify_key = [0; 0];
        rng().fill(&mut verify_key[..]);
        let measurements = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

        let mut sharded_measurements = Vec::new();
        for measurement in measurements {
            let nonce = rng().random();
            let (public_share, input_shares) =
                vdaf.shard(b"dummy ctx", &measurement, &nonce).unwrap();

            sharded_measurements.push((public_share, nonce, input_shares));
        }

        let result = run_vdaf_sharded(
            b"dummy ctx",
            &vdaf,
            &AggregationParam(aggregation_parameter),
            sharded_measurements.clone(),
        )
        .unwrap();
        assert_eq!(
            result,
            expected_aggregate_result(aggregation_parameter, measurements)
        );
    }

    #[test]
    fn single_round_agg_param_10() {
        run_test(1, 10)
    }

    #[test]
    fn single_round_agg_param_20() {
        run_test(1, 20)
    }

    #[test]
    fn single_round_agg_param_32() {
        run_test(1, 32)
    }

    #[test]
    fn single_round_agg_param_u8_max() {
        run_test(1, u8::MAX)
    }

    #[test]
    fn two_round_agg_param_10() {
        run_test(2, 10)
    }

    #[test]
    fn two_round_agg_param_20() {
        run_test(2, 20)
    }

    #[test]
    fn two_round_agg_param_32() {
        run_test(2, 32)
    }

    #[test]
    fn two_round_agg_param_u8_max() {
        run_test(2, u8::MAX)
    }
}
