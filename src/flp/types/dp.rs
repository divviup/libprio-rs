// SPDX-License-Identifier: MPL-2.0

use crate::dp::{distributions::PureDpDiscreteLaplace, DifferentialPrivacyStrategy};
use crate::dp::{DifferentialPrivacyDistribution, DpError};
use crate::field::{Field128, Field64, NttFriendlyFieldElement};
use crate::flp::gadgets::{Mul, ParallelSumGadget};
use crate::flp::types::{Histogram, SumVec};
use crate::flp::{FlpError, TypeWithNoise};
use crate::vdaf::xof::SeedStreamTurboShake128;
use num_bigint::{BigInt, BigUint, TryFromBigIntError};
use num_integer::Integer;
use rand::{distr::Distribution, make_rng, Rng};

// TODO(#1071): This is implemented for the concrete fields `Field64` and `Field128` in order to
// avoid imposing the `BigInt: From<F::Integer>` bound on all callers. In the future, we may want to
// impose this bound on `FieldElementWithInteger`, which would allow simple blanket implementations
// here.
impl<S> TypeWithNoise<PureDpDiscreteLaplace> for SumVec<Field64, S>
where
    S: ParallelSumGadget<Field64, Mul<Field64>> + Eq + 'static,
{
    fn add_noise_to_agg_share(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [Self::Field],
        _num_measurements: usize,
    ) -> Result<(), FlpError> {
        self.add_noise(
            dp_strategy,
            agg_result,
            &mut make_rng::<SeedStreamTurboShake128>(),
        )
    }
}

impl<S> TypeWithNoise<PureDpDiscreteLaplace> for SumVec<Field128, S>
where
    S: ParallelSumGadget<Field128, Mul<Field128>> + Eq + 'static,
{
    fn add_noise_to_agg_share(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [Self::Field],
        _num_measurements: usize,
    ) -> Result<(), FlpError> {
        self.add_noise(
            dp_strategy,
            agg_result,
            &mut make_rng::<SeedStreamTurboShake128>(),
        )
    }
}

impl<F, S> SumVec<F, S>
where
    F: NttFriendlyFieldElement,
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
        // Compute the global sensitivity of the aggregation function, using the L1 norm as a
        // distance metric, and using the substitution-DP model. The worst case is when one
        // individual's measurement changes such that each vector element flips from 0 to 2^bits -
        // 1, or vice versa. Then, the l1 distance from the initial query result to the new query
        // result will be (2^bits - 1) * length.
        let length = BigUint::from(self.len);
        let sensitivity = BigUint::from(
            1u128
                .checked_shl(self.bits as u32)
                .ok_or(FlpError::InvalidParameter(
                    "bits must be less than 128".into(),
                ))?
                - 1,
        ) * length;

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
    fn add_noise_to_agg_share(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [Self::Field],
        _num_measurements: usize,
    ) -> Result<(), FlpError> {
        self.add_noise(
            dp_strategy,
            agg_result,
            &mut make_rng::<SeedStreamTurboShake128>(),
        )
    }
}

impl<S> TypeWithNoise<PureDpDiscreteLaplace> for Histogram<Field128, S>
where
    S: ParallelSumGadget<Field128, Mul<Field128>> + Eq + 'static,
{
    fn add_noise_to_agg_share(
        &self,
        dp_strategy: &PureDpDiscreteLaplace,
        agg_result: &mut [Self::Field],
        _num_measurements: usize,
    ) -> Result<(), FlpError> {
        self.add_noise(
            dp_strategy,
            agg_result,
            &mut make_rng::<SeedStreamTurboShake128>(),
        )
    }
}

impl<F, S> Histogram<F, S>
where
    F: NttFriendlyFieldElement,
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
        // The global sensitivity of the aggregation function is two, using the L1 norm as a
        // distance metric, and using the substitution-DP model. Substituting a measurement may, at
        // worst, cause one cell of the query result to be incremented by one, and another to be
        // decremented by one.
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
    F: NttFriendlyFieldElement,
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

#[cfg(test)]
mod tests {
    use crate::{
        dp::{
            distributions::PureDpDiscreteLaplace, DifferentialPrivacyStrategy, PureDpBudget,
            Rational,
        },
        field::{merge_vector, split_vector, Field128, FieldElement},
        flp::{
            gadgets::ParallelSum,
            types::{Histogram, SumVec},
        },
        vdaf::xof::{Xof, XofTurboShake128},
    };

    #[test]
    fn sumvec_laplace_noise() {
        let dp_strategy = PureDpDiscreteLaplace::from_budget(
            PureDpBudget::new(Rational::from_unsigned(2u8, 1u8).unwrap()).unwrap(),
        );
        const SIZE: usize = 10;

        {
            let mut rng = XofTurboShake128::init(&[0; 32], &[]).into_seed_stream();
            let [mut share1, mut share2]: [Vec<Field128>; 2] =
                split_vector(&[Field128::zero(); SIZE], 2)
                    .try_into()
                    .unwrap();

            let sumvec: SumVec<_, ParallelSum<_, _>> = SumVec::new(1, SIZE, 1).unwrap();
            sumvec
                .add_noise(&dp_strategy, share1.as_mut_slice(), &mut rng)
                .unwrap();
            sumvec
                .add_noise(&dp_strategy, share2.as_mut_slice(), &mut rng)
                .unwrap();

            let mut aggregate_result = share1;
            merge_vector(&mut aggregate_result, &share2).unwrap();

            assert_eq!(
                aggregate_result,
                [
                    Field128::from(9),
                    Field128::from(5),
                    Field128::from(15),
                    Field128::from(3),
                    Field128::from(5),
                    Field128::from(0),
                    -Field128::from(3),
                    -Field128::from(30),
                    Field128::from(2),
                    -Field128::from(7),
                ]
            );
        }

        {
            let mut rng = XofTurboShake128::init(&[1; 32], &[]).into_seed_stream();
            let [mut share1, mut share2]: [Vec<Field128>; 2] =
                split_vector(&[Field128::zero(); SIZE], 2)
                    .try_into()
                    .unwrap();

            let sumvec: SumVec<_, ParallelSum<_, _>> = SumVec::new(2, SIZE, 1).unwrap();
            sumvec
                .add_noise(&dp_strategy, &mut share1, &mut rng)
                .unwrap();
            sumvec
                .add_noise(&dp_strategy, &mut share2, &mut rng)
                .unwrap();

            let mut aggregate_result = share1;
            merge_vector(&mut aggregate_result, &share2).unwrap();

            assert_eq!(
                aggregate_result,
                [
                    -Field128::from(36),
                    -Field128::from(8),
                    Field128::from(24),
                    Field128::from(32),
                    Field128::from(9),
                    -Field128::from(7),
                    -Field128::from(4),
                    Field128::from(9),
                    -Field128::from(8),
                    -Field128::from(14),
                ]
            );
        }
    }

    #[test]
    fn histogram_laplace_noise() {
        let dp_strategy = PureDpDiscreteLaplace::from_budget(
            PureDpBudget::new(Rational::from_unsigned(2u8, 1u8).unwrap()).unwrap(),
        );
        const SIZE: usize = 10;

        let mut rng = XofTurboShake128::init(&[2; 32], &[]).into_seed_stream();
        let [mut share1, mut share2]: [Vec<Field128>; 2] =
            split_vector(&[Field128::zero(); SIZE], 2)
                .try_into()
                .unwrap();

        let histogram: Histogram<_, ParallelSum<_, _>> = Histogram::new(SIZE, 1).unwrap();
        histogram
            .add_noise(&dp_strategy, &mut share1, &mut rng)
            .unwrap();
        histogram
            .add_noise(&dp_strategy, &mut share2, &mut rng)
            .unwrap();

        let mut aggregate_result = share1;
        merge_vector(&mut aggregate_result, &share2).unwrap();

        assert_eq!(
            aggregate_result,
            [
                Field128::from(2),
                Field128::from(1),
                -Field128::from(1),
                Field128::from(1),
                Field128::from(3),
                Field128::from(1),
                Field128::from(0),
                Field128::from(4),
                Field128::from(3),
                -Field128::from(2),
            ]
        );
    }
}
