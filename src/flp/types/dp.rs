use crate::dp::{distributions::PureDpDiscreteLaplace, DifferentialPrivacyStrategy};
use crate::dp::{DifferentialPrivacyDistribution, DpError};
use crate::field::{FftFriendlyFieldElement, Field128, Field64};
use crate::flp::gadgets::{Mul, ParallelSumGadget};
use crate::flp::types::{Histogram, SumVec};
use crate::flp::{FlpError, TypeWithNoise};
use crate::vdaf::xof::SeedStreamTurboShake128;
use num_bigint::{BigInt, BigUint, TryFromBigIntError};
use num_integer::Integer;
use num_traits::{One, Pow};
use rand::{distributions::Distribution, Rng, SeedableRng};

// TODO(#1071): This is implemented for the concrete fields `Field64` and `Field128` in order to
// avoid imposing the `BigInt: From<F::Integer>` bound on all callers. In the future, we may want to
// impose this bound on `FieldElementWithInteger`, which would allow simple blanket implementations
// here.
impl<S> TypeWithNoise<PureDpDiscreteLaplace> for SumVec<Field64, S>
where
    S: ParallelSumGadget<Field64, Mul<Field64>> + Eq + 'static,
{
    fn add_noise_to_result(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [Self::Field],
        _num_measurements: usize,
    ) -> Result<(), FlpError> {
        self.add_noise(
            dp_strategy,
            agg_result,
            &mut SeedStreamTurboShake128::from_entropy(),
        )
    }
}

impl<S> TypeWithNoise<PureDpDiscreteLaplace> for SumVec<Field128, S>
where
    S: ParallelSumGadget<Field128, Mul<Field128>> + Eq + 'static,
{
    fn add_noise_to_result(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [Self::Field],
        _num_measurements: usize,
    ) -> Result<(), FlpError> {
        self.add_noise(
            dp_strategy,
            agg_result,
            &mut SeedStreamTurboShake128::from_entropy(),
        )
    }
}

impl<F, S> SumVec<F, S>
where
    F: FftFriendlyFieldElement,
    BigInt: From<F::Integer>,
    F::Integer: TryFrom<BigInt, Error = TryFromBigIntError<BigInt>>,
{
    fn add_noise<R>(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [F],
        rng: &mut R,
    ) -> Result<(), FlpError>
    where
        R: Rng,
    {
        // Compute the l1-sensitivity of the aggregation function (assuming the substitution-DP
        // model). The worst case is when one individual's measurement changes such that each vector
        // element flips from 0 to 2^bits - 1, or vice versa. Then, the l1 distance from the initial
        // query result to the new query result will be (2^bits - 1) * length.
        let two = BigUint::from(2u64);
        let bits = BigUint::from(self.bits);
        let length = BigUint::from(self.len);
        let sensitivity = (Pow::pow(two, &bits) - BigUint::one()) * length;

        // Initialize sampler.
        let sampler = dp_strategy.create_distribution(sensitivity.into())?;

        // Generate noise for each vector coordinate and apply it.
        add_iid_noise_to_field_vec(agg_result, rng, &sampler)
    }
}

impl<S> TypeWithNoise<PureDpDiscreteLaplace> for Histogram<Field64, S>
where
    S: ParallelSumGadget<Field64, Mul<Field64>> + Eq + 'static,
{
    fn add_noise_to_result(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [Self::Field],
        _num_measurements: usize,
    ) -> Result<(), FlpError> {
        self.add_noise(
            dp_strategy,
            agg_result,
            &mut SeedStreamTurboShake128::from_entropy(),
        )
    }
}

impl<S> TypeWithNoise<PureDpDiscreteLaplace> for Histogram<Field128, S>
where
    S: ParallelSumGadget<Field128, Mul<Field128>> + Eq + 'static,
{
    fn add_noise_to_result(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [Self::Field],
        _num_measurements: usize,
    ) -> Result<(), FlpError> {
        self.add_noise(
            dp_strategy,
            agg_result,
            &mut SeedStreamTurboShake128::from_entropy(),
        )
    }
}

impl<F, S> Histogram<F, S>
where
    F: FftFriendlyFieldElement,
    BigInt: From<F::Integer>,
    F::Integer: TryFrom<BigInt, Error = TryFromBigIntError<BigInt>>,
{
    fn add_noise<R>(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [F],
        rng: &mut R,
    ) -> Result<(), FlpError>
    where
        R: Rng,
    {
        // The l1-sensitivity of the aggregation function is two, assuming the substitution-DP
        // model. Substituting a measurement may, at worst, cause one cell of the query result
        // to be incremented by one, and another to be decremented by one.
        let sensitivity = BigUint::from(2u64);

        // Initialize sampler.
        let sampler = dp_strategy.create_distribution(sensitivity.into())?;

        // Generate noise for each vector coordinate and apply it.
        add_iid_noise_to_field_vec(agg_result, rng, &sampler)
    }
}

/// This generates independent, identically-distributed noise, and adds it to a vector of field
/// elements after projecting it into the field.
pub(super) fn add_iid_noise_to_field_vec<F, R, D>(
    field_vec: &mut [F],
    rng: &mut R,
    distribution: &D,
) -> Result<(), FlpError>
where
    F: FftFriendlyFieldElement,
    BigInt: From<F::Integer>,
    F::Integer: TryFrom<BigInt, Error = TryFromBigIntError<BigInt>>,
    R: Rng,
    D: Distribution<BigInt> + DifferentialPrivacyDistribution,
{
    // Note that reducing noise by the field modulus, adding it to an aggregate share, and then
    // summing the aggregate shares into an aggregate result is equivalent to a trusted curator
    // computing the query result directly, then adding noise, and then reducing the noised query
    // result by the field modulus, because addition modulo the field modulus is commutative. The
    // differential privacy guarantees obtained by adding the noise are preserved when the
    // hypothetical curator takes the modulus, because this is a post-processing step on a
    // differentially private query. Therefore, taking the modulus of the noise before adding it to
    // an aggregate share here is safe.

    let modulus = BigInt::from(F::modulus());
    for entry in field_vec.iter_mut() {
        // Generate noise.
        let noise = distribution.sample(rng);

        // Project it into the field by taking the modulus, converting to a fixed-precision
        // integer, and converting to a field element.
        let noise_wrapped = noise.mod_floor(&modulus);
        let noise_fieldint =
            F::Integer::try_from(noise_wrapped).map_err(DpError::BigIntConversion)?;
        let noise_field = F::from(noise_fieldint);

        // Add noise to each element of the aggregate share.
        *entry += noise_field;
    }

    Ok(())
}
