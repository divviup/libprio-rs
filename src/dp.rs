// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use std::fmt::Debug;

/// Positive rational number to represent DP and noise distribution parameters in protocol messages
/// and manipulate them without rounding errors.
#[derive(Clone, Debug)]
pub struct Rational {
    /// Numerator.
    pub numerator: u32,
    /// Denominator.
    pub denominator: u32,
}

/// Marker trait for differential privacy budgets (regardless of the specific accounting method).
pub trait DifferentialPrivacyBudget {}

/// Marker trait for differential privacy scalar noise distributions.
pub trait DifferentialPrivacyDistribution {}

/// Zero-concentrated differential privacy (zCDP) budget as defined in [[BS16]].
///
/// [BS16]: https://arxiv.org/pdf/1605.02065.pdf
pub struct ZeroConcentratedDifferentialPrivacyBudget {
    /// Parameter `epsilon`, using the notation from [[CKS20]] where `rho = (epsilon**2)/2`
    /// for a `rho`-zCDP budget.
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub epsilon: Rational,
}

impl DifferentialPrivacyBudget for ZeroConcentratedDifferentialPrivacyBudget {}

/// Zero-mean Discrete Gaussian noise distribution.
///
/// The distribution is defined over the integers, represented by arbitrary-precision integers.
#[derive(Clone, Debug)]
pub struct DiscreteGaussian {
    /// Standard deviation of the distribution.
    pub sigma: Rational,
}

impl DifferentialPrivacyDistribution for DiscreteGaussian {}

/// Strategy to make aggregate shares differentially private, e.g. by adding noise from a specific
/// type of distribution instantiated with a given DP budget
pub trait DifferentialPrivacyStrategy {}

/// A zCDP budget used to create a Discrete Gaussian distribution
pub struct ZCdpDiscreteGaussian {
    budget: ZeroConcentratedDifferentialPrivacyBudget,
}

impl DifferentialPrivacyStrategy for ZCdpDiscreteGaussian {}

impl ZCdpDiscreteGaussian {
    /// Creates a new Discrete Gaussian by following Theorem 4 from [[CKS20]]
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub fn create_distribution(&self, sensitivity: Rational) -> DiscreteGaussian {
        let sigma = Rational {
            numerator: self.budget.epsilon.denominator * sensitivity.numerator,
            denominator: self.budget.epsilon.numerator * sensitivity.denominator,
        };
        DiscreteGaussian { sigma }
    }
}
