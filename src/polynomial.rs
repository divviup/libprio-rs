// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Functions for polynomial interpolation and evaluation

use crate::field::NttFriendlyFieldElement;
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use crate::ntt::{ntt, ntt_inv_finish};

use std::convert::TryFrom;

/// Evaluate a polynomial using Horner's method.
pub fn poly_eval<F: NttFriendlyFieldElement>(poly: &[F], eval_at: F) -> F {
    if poly.is_empty() {
        return F::zero();
    }

    let mut result = poly[poly.len() - 1];
    for i in (0..poly.len() - 1).rev() {
        result *= eval_at;
        result += poly[i];
    }

    result
}

/// Returns the degree of polynomial `p`.
pub fn poly_deg<F: NttFriendlyFieldElement>(p: &[F]) -> usize {
    let mut d = p.len();
    while d > 0 && p[d - 1] == F::zero() {
        d -= 1;
    }
    d.saturating_sub(1)
}

/// Multiplies polynomials `p` and `q` and returns the result.
pub fn poly_mul<F: NttFriendlyFieldElement>(p: &[F], q: &[F]) -> Vec<F> {
    let p_size = poly_deg(p) + 1;
    let q_size = poly_deg(q) + 1;
    let mut out = vec![F::zero(); p_size + q_size];
    for i in 0..p_size {
        for j in 0..q_size {
            out[i + j] += p[i] * q[j];
        }
    }
    out.truncate(poly_deg(&out) + 1);
    out
}

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[inline]
pub fn poly_interpret_eval<F: NttFriendlyFieldElement>(
    points: &[F],
    eval_at: F,
    tmp_coeffs: &mut [F],
) -> F {
    let size_inv = F::from(F::Integer::try_from(points.len()).unwrap()).inv();
    ntt(tmp_coeffs, points, points.len()).unwrap();
    ntt_inv_finish(tmp_coeffs, points.len(), size_inv);
    poly_eval(&tmp_coeffs[..points.len()], eval_at)
}

/// Returns a polynomial that evaluates to `0` if the input is in range `[start, end)`. Otherwise,
/// the output is not `0`.
pub(crate) fn poly_range_check<F: NttFriendlyFieldElement>(start: usize, end: usize) -> Vec<F> {
    let mut p = vec![F::one()];
    let mut q = [F::zero(), F::one()];
    for i in start..end {
        q[0] = -F::from(F::Integer::try_from(i).unwrap());
        p = poly_mul(&p, &q);
    }
    p
}

#[cfg(test)]
mod tests {
    use crate::{
        field::{Field64, FieldElement, FieldPrio2, NttFriendlyFieldElement},
        polynomial::{poly_deg, poly_eval, poly_mul, poly_range_check},
    };
    use std::convert::TryFrom;

    #[test]
    fn test_eval() {
        let mut poly = [FieldPrio2::from(0); 4];
        poly[0] = 2.into();
        poly[1] = 1.into();
        poly[2] = 5.into();
        // 5*3^2 + 3 + 2 = 50
        assert_eq!(poly_eval(&poly[..3], 3.into()), 50);
        poly[3] = 4.into();
        // 4*3^3 + 5*3^2 + 3 + 2 = 158
        assert_eq!(poly_eval(&poly[..4], 3.into()), 158);
    }

    #[test]
    fn test_poly_deg() {
        let zero = FieldPrio2::zero();
        let one = FieldPrio2::root(0).unwrap();
        assert_eq!(poly_deg(&[zero]), 0);
        assert_eq!(poly_deg(&[one]), 0);
        assert_eq!(poly_deg(&[zero, one]), 1);
        assert_eq!(poly_deg(&[zero, zero, one]), 2);
        assert_eq!(poly_deg(&[zero, one, one]), 2);
        assert_eq!(poly_deg(&[zero, one, one, one]), 3);
        assert_eq!(poly_deg(&[zero, one, one, one, zero]), 3);
        assert_eq!(poly_deg(&[zero, one, one, one, zero, zero]), 3);
    }

    #[test]
    fn test_poly_mul() {
        let p = [
            Field64::from(u64::try_from(2).unwrap()),
            Field64::from(u64::try_from(3).unwrap()),
        ];

        let q = [
            Field64::one(),
            Field64::zero(),
            Field64::from(u64::try_from(5).unwrap()),
        ];

        let want = [
            Field64::from(u64::try_from(2).unwrap()),
            Field64::from(u64::try_from(3).unwrap()),
            Field64::from(u64::try_from(10).unwrap()),
            Field64::from(u64::try_from(15).unwrap()),
        ];

        let got = poly_mul(&p, &q);
        assert_eq!(&got, &want);
    }

    #[test]
    fn test_poly_range_check() {
        let start = 74;
        let end = 112;
        let p = poly_range_check(start, end);

        // Check each number in the range.
        for i in start..end {
            let x = Field64::from(i as u64);
            let y = poly_eval(&p, x);
            assert_eq!(y, Field64::zero(), "range check failed for {i}");
        }

        // Check the number below the range.
        let x = Field64::from((start - 1) as u64);
        let y = poly_eval(&p, x);
        assert_ne!(y, Field64::zero());

        // Check a number above the range.
        let x = Field64::from(end as u64);
        let y = poly_eval(&p, x);
        assert_ne!(y, Field64::zero());
    }
}
