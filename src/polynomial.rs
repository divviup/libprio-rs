// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Functions for polynomial interpolation and evaluation

use crate::{
    field::{FieldElement, NttFriendlyFieldElement},
    fp::log2,
    ntt::{ntt_inv, ntt_set_s, NttError},
};

use std::convert::TryFrom;

/// Evaluate a polynomial in the monomial basis using Horner's method.
pub fn poly_eval_monomial<F: NttFriendlyFieldElement>(poly: &[F], eval_at: F) -> F {
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

/// Multiplies polynomials `p` and `q`, given in the monomial basis.
pub fn poly_mul_monomial<F: NttFriendlyFieldElement>(p: &[F], q: &[F]) -> Vec<F> {
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

/// Interpolate a polynomial from the provided points and evaluate it, using `tmp_coeffs` as scratch
/// space for the NTT.
#[cfg(any(test, all(feature = "crypto-dependencies", feature = "experimental")))]
#[inline]
pub fn poly_interpret_eval<F: NttFriendlyFieldElement>(
    points: &[F],
    eval_at: F,
    tmp_coeffs: &mut [F],
) -> F {
    use crate::ntt::{ntt, ntt_inv_finish};

    let size_inv = F::from(F::Integer::try_from(points.len()).unwrap()).inv();
    ntt(tmp_coeffs, points, points.len()).unwrap();
    ntt_inv_finish(tmp_coeffs, points.len(), size_inv);
    poly_eval_monomial(&tmp_coeffs[..points.len()], eval_at)
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

/// Multiplies polynomials `p` and `q`, given in the Lagrange basis. The polynomials must have the
/// same length, which must be a power of 2. For input polynomials of length `k` and degree `k - 1`,
/// the output polynomial will have length `2k` and degree `2k - 2`, containing one excess
/// coordinate. This is necessary for compatibility with NTT algorithms.
///
/// Implements `Lagrange.poly_mul` of [6.1.3.2][2], using the polynomial multiplication technique of
/// [Faz25 section 3.3][1].
///
/// [1]: https://eprint.iacr.org/2025/1727
/// [2]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-6.1.3.2
pub(crate) fn poly_mul_lagrange<F: NttFriendlyFieldElement>(
    output: &mut [F],
    p: &[F],
    q: &[F],
) -> Result<(), NttError> {
    assert_eq!(p.len(), q.len());
    assert!(p.len().is_power_of_two());

    double_evaluations(output, p)?;

    for (p_element, q_element) in output.iter_mut().zip(get_double_evaluations(q)?) {
        *p_element *= q_element;
    }

    Ok(())
}

/// Evaluates multiple polynomials given in the Lagrange basis. All the polynomials must have the
/// same length `n`, which must be a power of 2.
///
/// Implements `Lagrange.poly_eval_batched` of [6.1.3.2][2], using algorithm 7 from [Faz25][1].
/// `Lagrange.poly_eval` can be realized by passing a slice containing a single polynomial.
///
/// [1]: https://eprint.iacr.org/2025/1727
/// [2]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-6.1.3.2
pub(crate) fn poly_eval_lagrange_batched<F: NttFriendlyFieldElement, P: AsRef<[F]>>(
    polynomials: &[P],
    x: F,
) -> Vec<F> {
    let poly_len = polynomials[0].as_ref().len();
    assert!(
        polynomials.iter().all(|p| p.as_ref().len() == poly_len),
        "polynomials must be of equal length"
    );
    assert!(polynomials[0].as_ref().len().is_power_of_two());
    let roots = nth_root_powers(poly_len);

    let mut l = F::one();
    let mut u: Vec<F> = polynomials
        .iter()
        .map(|p| p.as_ref().first().copied().unwrap_or_else(F::zero))
        .collect();
    let mut d = roots[0] - x;
    for (i, wn_i) in (1..).zip(&roots[1..]) {
        l *= d;
        d = *wn_i - x;
        let t = l * *wn_i;
        for (u_j, poly) in u.iter_mut().zip(polynomials) {
            *u_j *= d;
            if let Some(yi) = poly.as_ref().get(i) {
                *u_j += t * *yi;
            }
        }
    }

    let mut num_roots_inv = inv_pow2::<F>(roots.len());
    if roots.len() > 1 {
        num_roots_inv = -num_roots_inv;
    }
    u.iter_mut().for_each(|u_j| *u_j *= num_roots_inv);

    u
}

/// Generates the powers of the primitive n-th root of unity.
///
/// Returns
///   roots\[i\] = w_n^i for 0 ≤ i < n,
/// where
///   w_n is the primitive n-th root of unity in `F`, and
///   `n` must be a power of two.
///
/// # Panics
///
/// It panics when `n` is not a power of two.
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

/// Appends evaluations of the polynomial to the provided slice until it is full. The length of the
/// slice must be a power of 2. The slice must contain `num_values` evaluations of the polynomial.
/// The remaining values in the slice are overwritten.
///
/// Corresponds to `extend_values_to_power_of_2` of [6.1.3.2][1].
///
/// [1]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-6.1.3.2
pub(crate) fn extend_values_to_power_of_2<F: NttFriendlyFieldElement>(
    polynomial: &mut [F],
    num_values: usize,
) {
    let desired_values = polynomial.len();
    assert!(desired_values.is_power_of_two());
    assert!(num_values <= polynomial.len());

    let root_powers: Vec<F> = nth_root_powers(desired_values);

    let mut w = vec![F::zero(); desired_values];
    for i in 0..num_values {
        w[i] = (0..num_values)
            .filter(|j| i != *j)
            .fold(F::one(), |acc, j| acc * (root_powers[i] - root_powers[j]));
    }

    for k in num_values..desired_values {
        for i in 0..k {
            w[i] *= root_powers[i] - root_powers[k];
        }

        let mut y_numerator = F::zero();
        let mut y_denominator = F::one();
        for (i, value) in polynomial[..k].iter().enumerate() {
            y_numerator = y_numerator * w[i] + y_denominator * *value;
            y_denominator *= w[i];
        }

        w[k] = (0..k).fold(F::one(), |acc, j| acc * (root_powers[k] - root_powers[j]));
        polynomial[k] = -w[k] * y_numerator * y_denominator.inv();
    }
}

/// Compute `2n` evaluations of the polynomial interpolated from `evaluations`, which consists of
/// `n` Lagrange basis evaluations. `n` must be a power of 2. The `2n` evaluations are written to
/// `output`.
///
/// Corresponds to `double_evaluations` of [6.1.3.2][1].
///
/// [1]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-6.1.3.2
pub(crate) fn double_evaluations<F: NttFriendlyFieldElement>(
    output: &mut [F],
    evaluations: &[F],
) -> Result<(), NttError> {
    if !evaluations.len().is_power_of_two() {
        return Err(NttError::SizeInvalid);
    }
    if output.len() != 2 * evaluations.len() {
        return Err(NttError::SizeInvalid);
    }

    // Do inverse NTT into the front half of output, then forward NTT into the back half to get odd
    // indices of output
    let (front, back) = output.split_at_mut(evaluations.len());
    ntt_inv(front, evaluations, evaluations.len())?;
    ntt_set_s(back, front, evaluations.len())?;

    // Interleave the input (even indices) with the back half of output (odd indices), into output.
    // This is safe to do because any element of pre-overwrite output can only contribute to a
    // smaller index post-overwrite, and thus overwriting doesn't destroy any information we need.
    for output_position in 0..output.len() {
        output[output_position] = if output_position % 2 == 0 {
            evaluations[output_position / 2]
        } else {
            output[evaluations.len() + output_position / 2]
        };
    }

    Ok(())
}

/// Same as [`double_evaluations`], but returns the result.
pub(crate) fn get_double_evaluations<F: NttFriendlyFieldElement>(
    evaluations: &[F],
) -> Result<Vec<F>, NttError> {
    let mut output = vec![F::zero(); evaluations.len() * 2];
    double_evaluations(&mut output, evaluations)?;

    Ok(output)
}

/// Returns a polynomial in the monomial basis that evaluates to `0` if the input is in range
/// `[start, end)`. Otherwise, the output is not `0`.
pub(crate) fn poly_range_check<F: NttFriendlyFieldElement>(start: usize, end: usize) -> Vec<F> {
    let mut p = vec![F::one()];
    let mut q = [F::zero(), F::one()];
    for i in start..end {
        q[0] = -F::from(F::Integer::try_from(i).unwrap());
        p = poly_mul_monomial(&p, &q);
    }
    p
}

#[cfg(test)]
mod tests {
    use crate::{
        field::{Field64, FieldElement, FieldPrio2, NttFriendlyFieldElement},
        fp::log2,
        ntt::get_ntt,
        polynomial::{
            extend_values_to_power_of_2, get_double_evaluations, nth_root_powers, poly_deg,
            poly_eval_lagrange_batched, poly_eval_monomial, poly_interpret_eval, poly_mul_lagrange,
            poly_mul_monomial, poly_range_check,
        },
    };
    use std::convert::TryFrom;

    #[test]
    fn test_eval() {
        let mut poly = [FieldPrio2::from(0); 4];
        poly[0] = 2.into();
        poly[1] = 1.into();
        poly[2] = 5.into();
        // 5*3^2 + 3 + 2 = 50
        assert_eq!(poly_eval_monomial(&poly[..3], 3.into()), 50);
        poly[3] = 4.into();
        // 4*3^3 + 5*3^2 + 3 + 2 = 158
        assert_eq!(poly_eval_monomial(&poly[..4], 3.into()), 158);
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

        let got = poly_mul_monomial(&p, &q);
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
            let y = poly_eval_monomial(&p, x);
            assert_eq!(y, Field64::zero(), "range check failed for {i}");
        }

        // Check the number below the range.
        let x = Field64::from((start - 1) as u64);
        let y = poly_eval_monomial(&p, x);
        assert_ne!(y, Field64::zero());

        // Check a number above the range.
        let x = Field64::from(end as u64);
        let y = poly_eval_monomial(&p, x);
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

    #[test]
    fn test_poly_eval_lagrange_batched() {
        // Single polynomial with constant terms
        test_poly_eval_lagrange_batched_with_lengths(&[1]);
        // Constant terms
        test_poly_eval_lagrange_batched_with_lengths(&[1, 1]);
        // Powers of two
        test_poly_eval_lagrange_batched_with_lengths(&[64, 64, 64, 64, 64]);
    }

    fn test_poly_eval_lagrange_batched_with_lengths(lengths: &[usize]) {
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

        // Evaluates several polynomials converting them to the monomial basis (iteratively).
        let want = polynomials
            .iter()
            .map(|poly| {
                let extended_poly = [poly.clone(), vec![Field64::zero(); n - poly.len()]].concat();
                poly_interpret_eval(&extended_poly, x, &mut ntt_mem)
            })
            .collect::<Vec<_>>();

        // Simultaneously evaluates several polynomials directly in the Lagrange basis (batched).
        let got = poly_eval_lagrange_batched(&polynomials, x);
        assert_eq!(got, want, "sizes: {sizes:?} x: {x} P: {polynomials:?}");
    }

    #[test]
    fn test_poly_mul_lagrange() {
        for log_n in 0..8 {
            let n = 1 << log_n;

            let p_monomial = Field64::random_vector(n);
            let q_monomial = Field64::random_vector(n);

            let p_lagrange = get_ntt(&p_monomial, n).unwrap();
            let q_lagrange = get_ntt(&q_monomial, n).unwrap();

            let mut product_lagrange = vec![Field64::zero(); 2 * n];
            poly_mul_lagrange(&mut product_lagrange, &p_lagrange, &q_lagrange).unwrap();
            let product_monomial = poly_mul_monomial(&p_monomial, &q_monomial);
            let product_monomial_ntt = get_ntt(&product_monomial, 2 * n).unwrap();
            assert_eq!(product_lagrange, product_monomial_ntt);
        }
    }

    #[test]
    fn test_extend_values_to_power_of_2() {
        for log_n in 0..7 {
            let n = 1 << log_n;
            for k in 0..n + 1 {
                // Random monomial polynomial of degree k - 1
                let mut p_monomial = Field64::random_vector(k);
                p_monomial.extend_from_slice(&vec![Field64::zero(); n - k]);

                // Convert to Lagrange basis
                let p_lagrange = get_ntt(&p_monomial, n).unwrap();

                // Truncate to k values
                let mut p_lagrange_truncated = p_lagrange.clone();
                for element in p_lagrange_truncated.iter_mut().skip(k) {
                    *element = Field64::zero();
                }

                // Recover the n Lagrange basis values
                extend_values_to_power_of_2(&mut p_lagrange_truncated, k);

                assert_eq!(p_lagrange_truncated, p_lagrange, "log_n = {log_n} k = {k}");
            }
        }
    }

    #[test]
    fn test_double_evaluations() {
        for log_n in 0..8 {
            let n = 1 << log_n;
            // Random monomial polynomial
            let p_monomial = Field64::random_vector(n);

            // Convert to Lagrange basis
            let p_lagrange = get_ntt(&p_monomial, n).unwrap();

            assert_eq!(
                get_double_evaluations(&p_lagrange).unwrap(),
                get_ntt(&p_monomial, 2 * n).unwrap()
            );
        }
    }
}
