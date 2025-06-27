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

/// Sets `outp` to the NTT of `inp`.
///
/// Interpreting the input as the coefficients of a polynomial, the output is equal to the input
/// evaluated at points `p^0, p^1, ... p^(size-1)`, where `p` is the `2^size`-th principal root of
/// unity.
#[allow(clippy::many_single_char_names)]
pub fn ntt<F: NttFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
) -> Result<(), NttError> {
    let d = usize::try_from(log2(size as u128)).map_err(|_| NttError::SizeTooLarge)?;

    if size > outp.len() {
        return Err(NttError::OutputTooSmall);
    }

    if size > 1 << MAX_ROOTS {
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
        w = F::one();
        let r = F::root(l).unwrap();
        let y = 1 << (l - 1);
        let chunk = (size / y) >> 1;

        // unrolling first iteration of i-loop.
        for j in 0..chunk {
            let x = j << l;
            let u = outp[x];
            let v = outp[x + y];
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

/// Sets `outp` to the inverse of the DFT of `inp`.
#[cfg(any(test, all(feature = "crypto-dependencies", feature = "experimental")))]
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
    use crate::field::{split_vector, Field128, Field64, FieldElement, FieldPrio2};
    use crate::polynomial::{poly_ntt, TestPolyAuxMemory};

    fn ntt_then_inv_test<F: NttFriendlyFieldElement>() -> Result<(), NttError> {
        let test_sizes = [1, 2, 4, 8, 16, 256, 1024, 2048];

        for size in test_sizes.iter() {
            let mut tmp = vec![F::zero(); *size];
            let mut got = vec![F::zero(); *size];
            let want = F::random_vector(*size);

            ntt(&mut tmp, &want, want.len())?;
            ntt_inv(&mut got, &tmp, tmp.len())?;
            assert_eq!(got, want);
        }

        Ok(())
    }

    #[test]
    fn test_priov2_field32() {
        ntt_then_inv_test::<FieldPrio2>().expect("unexpected error");
    }

    #[test]
    fn test_field64() {
        ntt_then_inv_test::<Field64>().expect("unexpected error");
    }

    #[test]
    fn test_field128() {
        ntt_then_inv_test::<Field128>().expect("unexpected error");
    }

    #[test]
    fn test_recursive_ntt() {
        let size = 128;
        let mut mem = TestPolyAuxMemory::new(size / 2);

        let inp = FieldPrio2::random_vector(size);
        let mut want = vec![FieldPrio2::zero(); size];
        let mut got = vec![FieldPrio2::zero(); size];

        ntt::<FieldPrio2>(&mut want, &inp, inp.len()).unwrap();

        poly_ntt(
            &mut got,
            &inp,
            &mem.roots_2n,
            size,
            false,
            &mut mem.ntt_memory,
        );

        assert_eq!(got, want);
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

        let mut want = vec![Field64::zero(); len];
        ntt_inv(&mut want, &x, len).unwrap();

        assert_eq!(got, want);
    }
}
