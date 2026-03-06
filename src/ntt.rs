// SPDX-License-Identifier: MPL-2.0

//! This module implements an iterative NTT (Number Theoretic Transform) algorithm.

use crate::field::NttFriendlyFieldElement;
use crate::fp::{log2, MAX_ROOTS};

use std::convert::TryFrom;

/// An error returned by an NTT operation.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum NttError {
    /// The output is too small.
    #[error("output slice is smaller than specified size")]
    OutputTooSmall,
    /// The specified size is too large.
    #[error("size is larger than than maximum permitted")]
    SizeTooLarge,
    /// The specified size is not a power of 2.
    #[error("size is not a power of 2")]
    SizeInvalid,
}

/// Sets `outp` to the NTT of `inp`, converting a polynomial in the monomial basis to the Lagrange
/// basis.
///
/// Interpreting the input as the coefficients of a polynomial, the output is equal to the input
/// evaluated at points `p^0, p^1, ... p^(size-1)`, where `p` is the `size`-th principal root of
/// unity.
///
/// This corresponds to the `Field.ntt` interface of [6.1.2][1], with `set_s = false`, and uses
/// Algorithm 4 of [Faz25][2].
///
/// [1]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-6.1.2
/// [2]: https://eprint.iacr.org/2025/1727.pdf
pub(crate) fn ntt<F: NttFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
) -> Result<(), NttError> {
    ntt_internal(outp, inp, size, false)
}

/// Sets `outp` to the NTT of `inp`.
///
/// Interpreting the input as the coefficients of a polynomial, the output is equal to the input
/// evaluated at points `s * p^0, s * p^1, ... s * p^(size-1)`, where `p` is the size-th principal
/// root of unity and `s` is a (2 * size)-th root of unity.
///
/// This corresponds to the `Field.ntt` interface of [6.1.2][1], with `set_s = true` and uses
/// Algorithm 4 of [Faz25][2].
///
/// [1]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vdaf-18#section-6.1.2
/// [2]: https://eprint.iacr.org/2025/1727.pdf
pub(crate) fn ntt_set_s<F: NttFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
) -> Result<(), NttError> {
    ntt_internal(outp, inp, size, true)
}

#[allow(clippy::many_single_char_names)]
fn ntt_internal<F: NttFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
    set_s: bool,
) -> Result<(), NttError> {
    let d = usize::try_from(log2(size as u128)).map_err(|_| NttError::SizeTooLarge)?;

    if size > outp.len() {
        return Err(NttError::OutputTooSmall);
    }

    if (set_s && size > 1 << (MAX_ROOTS - 1)) || size > 1 << MAX_ROOTS {
        return Err(NttError::SizeTooLarge);
    }

    if size != 1 << d {
        return Err(NttError::SizeInvalid);
    }

    if d > 0 {
        for (i, outp_val) in outp[..size].iter_mut().enumerate() {
            let j = bitrev(d, i);
            *outp_val = if j < inp.len() { inp[j] } else { F::zero() };
        }
    } else {
        outp[0] = inp[0];
    }

    let mut w: F;
    for l in 1..d + 1 {
        w = if set_s {
            // Unwrap safety: we ensure above that size is small enough to ensure that we have all
            // the roots we need.
            F::root(l + 1).unwrap()
        } else {
            F::one()
        };
        let r = F::root(l).unwrap();
        let y = 1 << (l - 1);
        let chunk = (size / y) >> 1;

        // unrolling first iteration of i-loop.
        for j in 0..chunk {
            let x = j << l;
            let u = outp[x];
            let v = w * outp[x + y];
            outp[x] = u + v;
            outp[x + y] = u - v;
        }

        for i in 1..y {
            w *= r;
            for j in 0..chunk {
                let x = (j << l) + i;
                let u = outp[x];
                let v = w * outp[x + y];
                outp[x] = u + v;
                outp[x + y] = u - v;
            }
        }
    }

    Ok(())
}

/// Does the same thing as [`ntt`], but returns the output.
pub(crate) fn get_ntt<F: NttFriendlyFieldElement>(
    input: &[F],
    size: usize,
) -> Result<Vec<F>, NttError> {
    let mut output = vec![F::zero(); size];
    ntt(&mut output, input, size)?;

    Ok(output)
}

/// Sets `outp` to the inverse of the NTT of `inp`.
pub(crate) fn ntt_inv<F: NttFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
) -> Result<(), NttError> {
    let size_inv = F::from(F::Integer::try_from(size).unwrap()).inv();
    ntt(outp, inp, size)?;
    ntt_inv_finish(outp, size, size_inv);
    Ok(())
}

/// Does the same thing as [`ntt_inv`], but returns the output.
pub(crate) fn get_ntt_inv<F: NttFriendlyFieldElement>(
    inp: &[F],
    size: usize,
) -> Result<Vec<F>, NttError> {
    let mut output = vec![F::zero(); size];
    ntt_inv(&mut output, inp, size)?;

    Ok(output)
}

/// An intermediate step in the computation of the inverse DFT. Exposing this function allows us to
/// amortize the cost the modular inverse across multiple inverse DFT operations.
pub(crate) fn ntt_inv_finish<F: NttFriendlyFieldElement>(outp: &mut [F], size: usize, size_inv: F) {
    let mut tmp: F;
    outp[0] *= size_inv;
    outp[size >> 1] *= size_inv;
    for i in 1..size >> 1 {
        tmp = outp[i] * size_inv;
        outp[i] = outp[size - i] * size_inv;
        outp[size - i] = tmp;
    }
}

/// Returns the first d bits of x in reverse order. (Thanks, OEIS! <https://oeis.org/A030109>)
fn bitrev(d: usize, x: usize) -> usize {
    x.reverse_bits() >> (usize::BITS - d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        field::{
            split_vector, Field128, Field64, FieldElement, FieldElementWithInteger, FieldPrio2,
        },
        polynomial::poly_eval_monomial,
    };

    fn ntt_then_inv_test<F: NttFriendlyFieldElement>() -> Result<(), NttError> {
        let test_sizes = [1, 2, 4, 8, 16, 256, 1024, 2048];

        for size in test_sizes.iter() {
            let want = F::random_vector(*size);

            let tmp = get_ntt(&want, *size)?;
            let got = get_ntt_inv(&tmp, tmp.len())?;
            assert_eq!(got, want);
        }

        Ok(())
    }

    fn test_ntt_set_s<F: NttFriendlyFieldElement>() {
        for log_n in 0..8 {
            let n = 1 << log_n;
            let nth_root = F::root(log_n).unwrap();
            let two_nth_root = F::root(log_n + 1).unwrap();

            // Random polynomial in monomial basis
            let p_monomial = F::random_vector(n);

            // Evaluate the polynomial at the powers of an n-th root of unity multiplied by a 2n-th
            // root of unity
            let monomial_evaluations = (0..n)
                .map(|power| F::Integer::try_from(power).unwrap())
                .map(|power| poly_eval_monomial(&p_monomial, two_nth_root * nth_root.pow(power)))
                .collect::<Vec<_>>();

            let mut ntt = vec![F::zero(); n];
            ntt_set_s(&mut ntt, &p_monomial, n).unwrap();

            // Monomial evaluations should match NTT with set_s = true
            assert_eq!(monomial_evaluations, ntt);
        }
    }

    #[test]
    fn test_priov2_field32() {
        ntt_then_inv_test::<FieldPrio2>().expect("unexpected error");
    }

    #[test]
    fn test_field64() {
        ntt_then_inv_test::<Field64>().expect("unexpected error");
        test_ntt_set_s::<Field64>();
    }

    #[test]
    fn test_field128() {
        ntt_then_inv_test::<Field128>().expect("unexpected error");
        test_ntt_set_s::<Field128>();
    }

    // This test demonstrates a consequence of \[BBG+19, Fact 4.4\]: interpolating a polynomial
    // over secret shares and summing up the coefficients is equivalent to interpolating a
    // polynomial over the plaintext data.
    #[test]
    fn test_ntt_linearity() {
        let len = 16;
        let num_shares = 3;
        let x = Field64::random_vector(len);
        let mut x_shares = split_vector(&x, num_shares);

        // Just for fun, let's do something different with a subset of the inputs. For the first
        // share, every odd element is set to the plaintext value. For all shares but the first,
        // every odd element is set to 0.
        for (i, x_val) in x.iter().enumerate() {
            if i % 2 != 0 {
                x_shares[0][i] = *x_val;
                for x_share in x_shares[1..num_shares].iter_mut() {
                    x_share[i] = Field64::zero();
                }
            }
        }

        let mut got = vec![Field64::zero(); len];
        let mut buf = vec![Field64::zero(); len];
        for share in x_shares {
            ntt_inv(&mut buf, &share, len).unwrap();
            for i in 0..len {
                got[i] += buf[i];
            }
        }

        let want = get_ntt_inv(&x, len).unwrap();

        assert_eq!(got, want);
    }

    #[test]
    fn test_ntt_interpolation() {
        let count = 128;
        let points = Field128::random_vector(count);
        let poly = get_ntt(&points, count).unwrap();
        let principal_root = Field128::root(7).unwrap(); // log_2(128);
        for (power, poly_coeff) in poly.iter().enumerate() {
            let expected = points
                .iter()
                .enumerate()
                .map(|(j, point_j)| {
                    principal_root
                        .pow(power.try_into().unwrap())
                        .pow(j.try_into().unwrap())
                        * *point_j
                })
                .reduce(|f, g| f + g)
                .unwrap();
            assert_eq!(expected, *poly_coeff);
        }
    }
}
