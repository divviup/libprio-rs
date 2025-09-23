// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Functions for polynomial interpolation and evaluation

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use crate::ntt::{ntt, ntt_inv_finish};
use crate::{
    field::{FieldElement, NttFriendlyFieldElement},
    fp::log2,
};

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

/// Returns the element `1/n` on `F`, where `n` must be a power of two.
#[inline]
fn inv_pow2<F: FieldElement>(n: usize) -> F {
    let log2_n = usize::try_from(log2(n as u128)).unwrap();
    assert_eq!(n, 1 << log2_n);

    let half = F::half();
    let mut x = F::one();
    for _ in 0..log2_n {
        x *= half
    }
    x
}

/// Evaluates multiple polynomials given in the Lagrange basis.
///
/// This is Algorithm 7 of rhizomes paper.
/// <https://eprint.iacr.org/2025/1727>.
pub(crate) fn poly_eval_batched<F: FieldElement>(
    polynomials: &[Vec<F>],
    roots: &[F],
    x: F,
) -> Vec<F> {
    let mut l = F::one();
    let mut u = Vec::with_capacity(polynomials.len());
    u.extend(polynomials.iter().map(|poly| poly[0]));
    let mut d = roots[0] - x;
    for (i, wn_i) in (1..).zip(&roots[1..]) {
        l *= d;
        d = *wn_i - x;
        let t = l * *wn_i;
        for (u_j, poly) in u.iter_mut().zip(polynomials) {
            *u_j *= d;
            if let Some(yi) = poly.get(i) {
                *u_j += t * *yi;
            }
        }
    }

    if roots.len() > 1 {
        let num_roots_inv = -inv_pow2::<F>(roots.len());
        u.iter_mut().for_each(|u_j| *u_j *= num_roots_inv);
    }

    u
}

/// Generates the powers of the primitive n-th root of unity.
///
/// Returns
///   roots\[i\] = w_n^i for 0 ≤ i < n,
/// where
///   w_n is the primitive n-th root of unity in `F`, and
///   `n` must be a power of two.
pub(crate) fn nth_root_powers<F: NttFriendlyFieldElement>(n: usize) -> Vec<F> {
    let log2_n = usize::try_from(log2(n as u128)).unwrap();
    assert_eq!(n, 1 << log2_n);

    let mut roots = vec![F::zero(); n];
    roots[0] = F::one();
    if n > 1 {
        roots[1] = -F::one();
        for i in 2..=log2_n {
            let mid = 1 << (i - 1);
            // Due to w_{2n}^{2j} = w_{n}^j
            for j in (1..mid).rev() {
                roots[j << 1] = roots[j]
            }

            let wn = F::root(i).unwrap();
            roots[1] = wn;
            roots[1 + mid] = -wn;

            // Due to w_{n}^{j} = -w_{n}^{j+n/2}
            for j in (3..mid).step_by(2) {
                roots[j] = wn * roots[j - 1];
                roots[j + mid] = -roots[j]
            }
        }
    }

    roots
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
    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    use crate::polynomial::{poly_eval_batched, poly_interpret_eval};
    use crate::{
        field::{Field64, FieldElement, FieldPrio2, NttFriendlyFieldElement},
        fp::log2,
        polynomial::{nth_root_powers, poly_deg, poly_eval, poly_mul, poly_range_check},
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

    /// Generates the powers of the primitive n-th root of unity.
    ///
    /// Returns
    ///   roots\[i\] = w_n^i for 0 ≤ i < n,
    /// where
    ///   w_n is the primitive n-th root of unity in `F`, and
    ///   `n` must be a power of two.
    ///
    /// This is the iterative method.
    fn nth_root_powers_slow<F: NttFriendlyFieldElement>(n: usize) -> Vec<F> {
        let log2_n = usize::try_from(log2(n as u128)).unwrap();
        let wn = F::root(log2_n).unwrap();
        core::iter::successors(Some(F::one()), |&x| Some(x * wn))
            .take(n)
            .collect()
    }

    #[test]
    fn test_nth_root_powers() {
        for i in 0..8 {
            assert_eq!(
                nth_root_powers::<Field64>(1 << i),
                nth_root_powers_slow::<Field64>(1 << i)
            );
        }
    }

    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    #[test]
    fn test_poly_eval_batched() {
        // Single polynomial with constant terms
        test_poly_eval_batched_with_lengths(&[1]);
        // Constant terms
        test_poly_eval_batched_with_lengths(&[1, 1]);
        // Powers of two
        test_poly_eval_batched_with_lengths(&[1, 2, 4, 16, 64]);
        // arbitrary
        test_poly_eval_batched_with_lengths(&[1, 6, 3, 9]);
    }

    #[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
    fn test_poly_eval_batched_with_lengths(lengths: &[usize]) {
        let sizes = lengths
            .iter()
            .map(|s| s.next_power_of_two())
            .collect::<Vec<_>>();

        let polynomials = sizes
            .iter()
            .map(|&size| Field64::random_vector(size))
            .collect::<Vec<_>>();
        let x = Field64::random_vector(1)[0];

        let &n = sizes.iter().max().unwrap();
        let mut ntt_mem = vec![Field64::zero(); n];
        let roots = nth_root_powers(n);

        // Evaluates several polynomials converting them to the monomial basis (iteratively).
        let want = polynomials
            .iter()
            .map(|poly| {
                let extended_poly = [poly.clone(), vec![Field64::zero(); n - poly.len()]].concat();
                poly_interpret_eval(&extended_poly, x, &mut ntt_mem)
            })
            .collect::<Vec<_>>();

        // Simultaneouly evaluates several polynomials directly in the Lagrange basis (batched).
        let got = poly_eval_batched(&polynomials, &roots, x);
        assert_eq!(got, want, "sizes: {sizes:?} x: {x} P: {polynomials:?}");
    }
}
