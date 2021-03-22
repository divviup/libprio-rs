// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic over a prime field using a 32bit prime.

use crate::fp::FieldParameters;

/// Possible errors from finite field operations.
#[derive(Debug, thiserror::Error)]
pub enum FiniteFieldError {
    /// Input sizes do not match
    #[error("input sizes do not match")]
    InputSizeMismatch,
}

/// Newtype wrapper over u128
///
/// Implements the arithmetic over the finite prime field
#[derive(Clone, Copy, Debug, PartialOrd, Ord, Hash, Default)]
pub struct Field(u128);

/// Parameters for GF(2^32 - 2^20 + 1).
pub(crate) const SMALL_FP: FieldParameters = FieldParameters {
    p: 4293918721,
    p2: 8587837442,
    mu: 17302828673139736575,
    r2: 1676699750,
};

/// Modulus for the field, a FFT friendly prime: 2^32 - 2^20 + 1
pub const MODULUS: u32 = SMALL_FP.p as u32;
/// Generator for the multiplicative subgroup
pub(crate) const GENERATOR: u32 = 3925978153;
/// Number of primitive roots
pub(crate) const N_ROOTS: u32 = 1 << 20; // number of primitive roots

impl std::ops::Add for Field {
    type Output = Field;

    fn add(self, rhs: Self) -> Self {
        Self(SMALL_FP.add(self.0, rhs.0))
    }
}

impl std::ops::AddAssign for Field {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Sub for Field {
    type Output = Field;

    fn sub(self, rhs: Self) -> Self {
        Self(SMALL_FP.sub(self.0, rhs.0))
    }
}

impl std::ops::SubAssign for Field {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl std::ops::Mul for Field {
    type Output = Field;

    fn mul(self, rhs: Self) -> Self {
        Self(SMALL_FP.mul(self.0, rhs.0))
    }
}

impl std::ops::MulAssign for Field {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl std::ops::Div for Field {
    type Output = Field;

    fn div(self, rhs: Self) -> Self {
        self * rhs.inv()
    }
}

impl std::ops::DivAssign for Field {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl PartialEq for Field {
    fn eq(&self, rhs: &Self) -> bool {
        SMALL_FP.from_elem(self.0) == SMALL_FP.from_elem(rhs.0)
    }
}

impl Eq for Field {}

impl Field {
    /// Modular exponentation
    pub fn pow(self, exp: Self) -> Self {
        Self(SMALL_FP.pow(self.0, SMALL_FP.from_elem(exp.0)))
    }

    /// Modular inverse
    ///
    /// Note: inverse of 0 is defined as 0.
    pub fn inv(self) -> Self {
        Self(SMALL_FP.inv(self.0))
    }
}

impl From<u32> for Field {
    fn from(x: u32) -> Self {
        Field(SMALL_FP.elem(x as u128))
    }
}

impl From<Field> for u32 {
    fn from(x: Field) -> Self {
        SMALL_FP.from_elem(x.0) as u32
    }
}

impl PartialEq<u32> for Field {
    fn eq(&self, rhs: &u32) -> bool {
        SMALL_FP.from_elem(self.0) == *rhs as u128
    }
}

impl std::fmt::Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", SMALL_FP.from_elem(self.0))
    }
}

#[test]
fn test_small_fp() {
    assert_eq!(SMALL_FP.check(), Ok(()));
}

#[test]
fn test_arithmetic() {
    use rand::prelude::*;

    // add
    assert_eq!(Field::from(MODULUS - 1) + Field::from(1), 0);
    assert_eq!(Field::from(MODULUS - 2) + Field::from(2), 0);
    assert_eq!(Field::from(MODULUS - 2) + Field::from(3), 1);
    assert_eq!(Field::from(1) + Field::from(1), 2);
    assert_eq!(Field::from(2) + Field::from(MODULUS), 2);
    assert_eq!(Field::from(3) + Field::from(MODULUS - 1), 2);

    // sub
    assert_eq!(Field::from(0) - Field::from(1), MODULUS - 1);
    assert_eq!(Field::from(1) - Field::from(2), MODULUS - 1);
    assert_eq!(Field::from(15) - Field::from(3), 12);
    assert_eq!(Field::from(1) - Field::from(1), 0);
    assert_eq!(Field::from(2) - Field::from(MODULUS), 2);
    assert_eq!(Field::from(3) - Field::from(MODULUS - 1), 4);

    // add + sub
    for _ in 0..100 {
        let f = Field::from(random::<u32>());
        let g = Field::from(random::<u32>());
        assert_eq!(f + g - f - g, 0);
        assert_eq!(f + g - g, f);
        assert_eq!(f + g - f, g);
    }

    // mul
    assert_eq!(Field::from(35) * Field::from(123), 4305);
    assert_eq!(Field::from(1) * Field::from(MODULUS), 0);
    assert_eq!(Field::from(0) * Field::from(123), 0);
    assert_eq!(Field::from(123) * Field::from(0), 0);
    assert_eq!(Field::from(123123123) * Field::from(123123123), 1237630077);

    // div
    assert_eq!(Field::from(35) / Field::from(5), 7);
    assert_eq!(Field::from(35) / Field::from(0), 0);
    assert_eq!(Field::from(0) / Field::from(5), 0);
    assert_eq!(Field::from(1237630077) / Field::from(123123123), 123123123);

    assert_eq!(Field::from(0).inv(), 0);

    // mul and div
    let uniform = rand::distributions::Uniform::from(1..MODULUS);
    let mut rng = thread_rng();
    for _ in 0..100 {
        // non-zero element
        let f = Field::from(uniform.sample(&mut rng));
        assert_eq!(f * f.inv(), 1);
        assert_eq!(f.inv() * f, 1);
    }

    // pow
    assert_eq!(Field::from(2).pow(3.into()), 8);
    assert_eq!(Field::from(3).pow(9.into()), 19683);
    assert_eq!(Field::from(51).pow(27.into()), 3760729523);
    assert_eq!(Field::from(432).pow(0.into()), 1);
    assert_eq!(Field(0).pow(123.into()), 0);
}

/// Merge two vectors of fields by summing other_vector into accumulator.
///
/// # Errors
///
/// Fails if the two vectors do not have the same length.
pub fn merge_vector(
    accumulator: &mut [Field],
    other_vector: &[Field],
) -> Result<(), FiniteFieldError> {
    if accumulator.len() != other_vector.len() {
        return Err(FiniteFieldError::InputSizeMismatch);
    }
    for (a, o) in accumulator.iter_mut().zip(other_vector.iter()) {
        *a += *o;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::vector_with_length;
    use assert_matches::assert_matches;

    #[test]
    fn test_accumulate() {
        let mut lhs = vector_with_length(10);
        lhs.iter_mut().for_each(|f| *f = Field(1));
        let mut rhs = vector_with_length(10);
        rhs.iter_mut().for_each(|f| *f = Field(2));

        merge_vector(&mut lhs, &rhs).unwrap();

        lhs.iter().for_each(|f| assert_eq!(*f, Field(3)));
        rhs.iter().for_each(|f| assert_eq!(*f, Field(2)));

        let wrong_len = vector_with_length(9);
        let result = merge_vector(&mut lhs, &wrong_len);
        assert_matches!(result, Err(FiniteFieldError::InputSizeMismatch));
    }
}
