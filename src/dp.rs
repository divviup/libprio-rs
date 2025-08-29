// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
//!
//! There are three main traits defined in this module:
//!
//!  - `DifferentialPrivacyBudget`: Implementors should be types of DP-budgets,
//!    i.e., methods to measure the amount of privacy provided by DP-mechanisms.
//!    Examples: zCDP, ApproximateDP (Epsilon-Delta), PureDP
//!
//!  - `DifferentialPrivacyDistribution`: Distribution from which noise is sampled.
//!    Examples: DiscreteGaussian, DiscreteLaplace
//!
//!  - `DifferentialPrivacyStrategy`: This is a combination of choices for budget and distribution.
//!    Examples: zCDP-DiscreteGaussian, EpsilonDelta-DiscreteGaussian
//!
use num_bigint::{BigInt, BigUint, TryFromBigIntError};
use num_rational::{BigRational, Ratio};
use serde::Serialize;

/// Errors propagated by methods in this module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DpError {
    /// Tried to use an invalid float as privacy parameter.
    #[error(
        "DP error: input value was not a valid privacy parameter. \
             It should to be a non-negative, finite float."
    )]
    InvalidFloat,

    /// Tried to convert BigInt into something incompatible.
    #[error("DP error: {0}")]
    BigIntConversion(#[from] TryFromBigIntError<BigInt>),

    /// Invalid parameter value.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Positive arbitrary precision rational number to represent DP and noise distribution parameters in
/// protocol messages and manipulate them without rounding errors.
#[derive(Clone, Debug)]
pub struct Rational(Ratio<BigUint>);

impl Rational {
    /// Construct a [`Rational`] number from numerator `n` and denominator `d`. Errors if denominator is zero.
    pub fn from_unsigned<T>(n: T, d: T) -> Result<Self, DpError>
    where
        T: Into<u128>,
    {
        // we don't want to expose BigUint in the public api, hence the Into<u128> bound
        let d = d.into();
        if d == 0 {
            Err(DpError::InvalidParameter(
                "input denominator was zero".to_owned(),
            ))
        } else {
            Ok(Rational(Ratio::<BigUint>::new(n.into().into(), d.into())))
        }
    }
}

impl TryFrom<f32> for Rational {
    type Error = DpError;
    /// Constructs a `Rational` from a given `f32` value.
    ///
    /// The special float values (NaN, positive and negative infinity) result in
    /// an error. All other values are represented exactly, without rounding errors.
    fn try_from(value: f32) -> Result<Self, DpError> {
        match BigRational::from_float(value) {
            Some(y) => Ok(Rational(Ratio::<BigUint>::new(
                y.numer().clone().try_into()?,
                y.denom().clone().try_into()?,
            ))),
            None => Err(DpError::InvalidFloat)?,
        }
    }
}

/// Marker trait for differential privacy budgets (regardless of the specific accounting method).
pub trait DifferentialPrivacyBudget {}

/// Marker trait for differential privacy scalar noise distributions.
pub trait DifferentialPrivacyDistribution {}

/// Zero-concentrated differential privacy (ZCDP) budget as defined in [[BS16]].
///
/// [BS16]: https://arxiv.org/pdf/1605.02065.pdf
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Ord, PartialOrd)]
pub struct ZCdpBudget {
    epsilon: Ratio<BigUint>,
}

impl ZCdpBudget {
    /// Create a budget for parameter `epsilon`, using the notation from [[CKS20]] where `rho = (epsilon**2)/2`
    /// for a `rho`-ZCDP budget.
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub fn new(epsilon: Rational) -> Result<Self, DpError> {
        if epsilon.0.numer() == &BigUint::ZERO {
            return Err(DpError::InvalidParameter("epsilon cannot be zero".into()));
        }
        Ok(Self { epsilon: epsilon.0 })
    }
}

impl DifferentialPrivacyBudget for ZCdpBudget {}

/// Pure differential privacy budget. (&epsilon;-DP or (&epsilon;, 0)-DP)
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Ord, PartialOrd)]
pub struct PureDpBudget {
    epsilon: Ratio<BigUint>,
}

impl PureDpBudget {
    /// Create a budget for parameter `epsilon`.
    pub fn new(epsilon: Rational) -> Result<Self, DpError> {
        if epsilon.0.numer() == &BigUint::ZERO {
            return Err(DpError::InvalidParameter("epsilon cannot be zero".into()));
        }
        Ok(Self { epsilon: epsilon.0 })
    }
}

impl DifferentialPrivacyBudget for PureDpBudget {}

/// This module encapsulates deserialization helper structs. It is needed so we can wrap derived
/// `Deserialize` implementations in customized `Deserialize` implementations, which make use of
/// constructor associated methods for budgets to enforce input validation invariants.
mod budget_serde {
    use num_bigint::BigUint;
    use num_rational::Ratio;
    use serde::{de, Deserialize};

    use super::Rational;

    #[derive(Deserialize)]
    pub struct ZCdpBudget {
        epsilon: Ratio<BigUint>,
    }

    impl<'de> Deserialize<'de> for super::ZCdpBudget {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let helper = ZCdpBudget::deserialize(deserializer)?;
            super::ZCdpBudget::new(Rational(helper.epsilon))
                .map_err(|_| de::Error::custom("epsilon cannot be zero"))
        }
    }

    #[derive(Deserialize)]
    pub struct PureDpBudget {
        epsilon: Ratio<BigUint>,
    }

    impl<'de> Deserialize<'de> for super::PureDpBudget {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let helper = PureDpBudget::deserialize(deserializer)?;
            super::PureDpBudget::new(Rational(helper.epsilon))
                .map_err(|_| de::Error::custom("epsilon cannot be zero"))
        }
    }
}

/// Strategy to make aggregate results differentially private, e.g. by adding noise from a specific
/// type of distribution instantiated with a given DP budget.
pub trait DifferentialPrivacyStrategy {
    /// The type of the DP budget, i.e. the variant of differential privacy that can be obtained
    /// by using this strategy.
    type Budget: DifferentialPrivacyBudget;

    /// The distribution type this strategy will use to generate the noise.
    type Distribution: DifferentialPrivacyDistribution;

    /// The type the sensitivity used for privacy analysis has.
    type Sensitivity;

    /// Create a strategy from a differential privacy budget. The distribution created with
    /// `create_distribution` should provide the amount of privacy specified here.
    fn from_budget(b: Self::Budget) -> Self;

    /// Create a new distribution parametrized s.t. adding samples to the result of a function
    /// with sensitivity `s` will yield differential privacy of the DP variant given in the
    /// `Budget` type. Can error upon invalid parameters.
    fn create_distribution(&self, s: Self::Sensitivity) -> Result<Self::Distribution, DpError>;
}

pub mod distributions;
mod rand_bigint;

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{PureDpBudget, Rational, ZCdpBudget};

    #[test]
    fn budget_deserialization() {
        serde_json::from_value::<ZCdpBudget>(json!({"epsilon": [[1], [1]]})).unwrap();
        serde_json::from_value::<ZCdpBudget>(json!({"epsilon": [[0], [1]]})).unwrap_err();
        serde_json::from_value::<ZCdpBudget>(json!({"epsilon": [[1], [0]]})).unwrap_err();

        serde_json::from_value::<PureDpBudget>(json!({"epsilon": [[1], [1]]})).unwrap();
        serde_json::from_value::<PureDpBudget>(json!({"epsilon": [[0], [1]]})).unwrap_err();
        serde_json::from_value::<PureDpBudget>(json!({"epsilon": [[1], [0]]})).unwrap_err();
    }

    #[test]
    fn bad_budgets() {
        ZCdpBudget::new(Rational::from_unsigned(0u128, 1).unwrap()).unwrap_err();
        PureDpBudget::new(Rational::from_unsigned(0u128, 1).unwrap()).unwrap_err();
    }
}
