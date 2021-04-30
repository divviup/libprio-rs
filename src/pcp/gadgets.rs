// SPDX-License-Identifier: MPL-2.0

//! A collection of gadgets.

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish};
use crate::field::FieldElement;
use crate::pcp::{Gadget, GadgetCallOnly, PcpError};
use crate::polynomial::{poly_deg, poly_eval, poly_mul};

use rayon::prelude::*;

use std::convert::TryFrom;
use std::marker::PhantomData;

/// For input polynomials larger than or equal to this threshold, gadgets will use FFT for
/// polynomial multiplication. Otherwise, the gadget uses direct multiplication.
const FFT_THRESHOLD: usize = 60;

/// An arity-2 gadget that multiples its inputs.
#[derive(Clone)]
pub struct Mul<F: FieldElement> {
    /// Size of buffer for FFT operations.
    n: usize,
    /// Inverse of `n` in `F`.
    n_inv: F,
}

impl<F: FieldElement> Mul<F> {
    /// Return a new multiplier gadget.
    pub fn new(in_len: usize) -> Self {
        let n = (2 * in_len).next_power_of_two();
        let n_inv = F::from(F::Integer::try_from(n).unwrap()).inv();
        Self { n, n_inv }
    }

    // Multiply input polynomials directly.
    pub(crate) fn call_poly_direct<V: AsRef<[F]>>(
        &mut self,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        let v = poly_mul(inp[0].as_ref(), inp[1].as_ref());
        for i in 0..v.len() {
            outp[i] = v[i];
        }
        Ok(())
    }

    // Multiply input polynomials using FFT.
    pub(crate) fn call_poly_fft<V: AsRef<[F]>>(
        &mut self,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        let n = self.n;
        let mut buf = vec![F::zero(); n];

        discrete_fourier_transform(&mut buf, inp[0].as_ref(), n)?;
        discrete_fourier_transform(outp, inp[1].as_ref(), n)?;

        for i in 0..n {
            buf[i] *= outp[i];
        }

        discrete_fourier_transform(outp, &buf, n)?;
        discrete_fourier_transform_inv_finish(outp, n, self.n_inv);
        Ok(())
    }
}

impl<F: FieldElement> GadgetCallOnly<F> for Mul<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, inp.len())?;
        Ok(inp[0] * inp[1])
    }

    fn call_in_len(&self) -> usize {
        2
    }
}

impl<F: FieldElement> Gadget<F> for Mul<F> {
    fn call_poly<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;
        if inp[0].as_ref().len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
    }

    fn call_poly_out_len(&self, in_len: usize) -> usize {
        2 * in_len
    }
}

/// An arity-1 gadget that evaluates its input on some polynomial.
#[derive(Clone)]
pub struct PolyEval<F: FieldElement> {
    poly: Vec<F>,
    /// Size of buffer for FFT operations.
    n: usize,
    /// Inverse of `n` in `F`.
    n_inv: F,
}

impl<F: FieldElement> PolyEval<F> {
    /// Returns a gadget that evaluates its input on `poly`. The parameter `in_len` denotes the
    /// length of each input passed to `call_poly`.
    pub fn new(poly: Vec<F>, in_len: usize) -> Self {
        let n = (in_len * poly_deg(&poly)).next_power_of_two();
        let n_inv = F::from(F::Integer::try_from(n).unwrap()).inv();
        Self { poly, n, n_inv }
    }

    fn call_poly_direct<V: AsRef<[F]>>(
        &mut self,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        outp[0] = self.poly[0];
        let x = inp[0].as_ref();
        let mut z = x.to_vec();
        for i in 1..self.poly.len() {
            for j in 0..z.len() {
                outp[j] += self.poly[i] * z[j];
            }

            if i < self.poly.len() - 1 {
                z = poly_mul(&z, x);
            }
        }
        Ok(())
    }

    fn call_poly_fft<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        let n = self.n;
        let x = inp[0].as_ref();

        let mut x_vals = vec![F::zero(); n];
        discrete_fourier_transform(&mut x_vals, x, n)?;

        let mut z_vals = x_vals.clone();
        let mut z = vec![F::zero(); n];
        let mut z_len = x.len();
        z[..x.len()].clone_from_slice(x);

        outp[0] = self.poly[0];
        for i in 1..self.poly.len() {
            for j in 0..z_len {
                outp[j] += self.poly[i] * z[j];
            }

            if i < self.poly.len() - 1 {
                for j in 0..n {
                    z_vals[j] *= x_vals[j];
                }

                discrete_fourier_transform(&mut z, &z_vals, n)?;
                discrete_fourier_transform_inv_finish(&mut z, n, self.n_inv);
                z_len += x.len();
            }
        }
        Ok(())
    }
}

impl<F: FieldElement> GadgetCallOnly<F> for PolyEval<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, inp.len())?;
        Ok(poly_eval(&self.poly, inp[0]))
    }

    fn call_in_len(&self) -> usize {
        1
    }
}

impl<F: FieldElement> Gadget<F> for PolyEval<F> {
    fn call_poly<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;

        for i in 0..outp.len() {
            outp[i] = F::zero();
        }

        if inp[0].as_ref().len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
    }

    fn call_poly_out_len(&self, in_len: usize) -> usize {
        poly_deg(&self.poly) * in_len
    }
}

/// An arity-2 gadget that returns `poly(in[0]) * in[1]` for some polynomial `poly`.
#[derive(Clone)]
pub struct BlindPolyEval<F: FieldElement> {
    poly: Vec<F>,
    /// Size of buffer for the outer FFT multiplication.
    n: usize,
    /// Inverse of `n` in `F`.
    n_inv: F,
}

impl<F: FieldElement> BlindPolyEval<F> {
    /// Returns a `BlindPolyEval` gadget for polynomial `poly`. Integer `in_len` is the length of
    /// the input polynomials on which `call_poly` will be called.
    pub fn new(poly: Vec<F>, in_len: usize) -> Self {
        let n = (in_len * (poly_deg(&poly) + 1)).next_power_of_two();
        let n_inv = F::from(F::Integer::try_from(n).unwrap()).inv();
        Self { poly, n, n_inv }
    }

    fn call_poly_direct<V: AsRef<[F]>>(
        &mut self,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        let x = inp[0].as_ref();
        let y = inp[1].as_ref();

        let mut z = y.to_vec();
        for i in 0..self.poly.len() {
            for j in 0..z.len() {
                outp[j] += self.poly[i] * z[j];
            }

            if i < self.poly.len() - 1 {
                z = poly_mul(&z, x);
            }
        }
        Ok(())
    }

    fn call_poly_fft<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        let n = self.n;
        let x = inp[0].as_ref();
        let y = inp[1].as_ref();

        let mut x_vals = vec![F::zero(); n];
        discrete_fourier_transform(&mut x_vals, x, n)?;

        let mut z_vals = vec![F::zero(); n];
        discrete_fourier_transform(&mut z_vals, y, n)?;

        let mut z = vec![F::zero(); n];
        let mut z_len = y.len();
        z[..y.len()].clone_from_slice(y);

        for i in 0..self.poly.len() {
            for j in 0..z_len {
                outp[j] += self.poly[i] * z[j];
            }

            if i < self.poly.len() - 1 {
                for j in 0..n {
                    z_vals[j] *= x_vals[j];
                }

                discrete_fourier_transform(&mut z, &z_vals, n)?;
                discrete_fourier_transform_inv_finish(&mut z, n, self.n_inv);
                z_len += x.len();
            }
        }
        Ok(())
    }
}

impl<F: FieldElement> GadgetCallOnly<F> for BlindPolyEval<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, inp.len())?;
        Ok(inp[1] * poly_eval(&self.poly, inp[0]))
    }

    fn call_in_len(&self) -> usize {
        2
    }
}

impl<F: FieldElement> Gadget<F> for BlindPolyEval<F> {
    fn call_poly<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;

        for i in 0..outp.len() {
            outp[i] = F::zero();
        }

        if inp[0].as_ref().len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
    }

    fn call_poly_out_len(&self, in_len: usize) -> usize {
        (poly_deg(&self.poly) + 1) * in_len
    }
}

/// A wrapper gadget that applies the inner gadget to chunks of input and returns the sum of the
/// outputs. The arity is equal to the arity of the inner gadget times the number of chunks.
#[derive(Clone)]
pub struct ParallelSum<F: FieldElement, G: Gadget<F>> {
    inner: G,
    chunks: usize,
    phantom: PhantomData<F>,
}

impl<F: FieldElement, G: Gadget<F>> ParallelSum<F, G> {
    /// Wraps `inner` into a parallel sum gadget with `chunks` chunks.
    pub fn new(inner: G, chunks: usize) -> Self {
        Self {
            inner,
            chunks,
            phantom: PhantomData,
        }
    }
}

impl<F: FieldElement, G: Gadget<F>> GadgetCallOnly<F> for ParallelSum<F, G> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, inp.len())?;
        let mut outp = F::zero();
        for chunk in inp.chunks(self.inner.call_in_len()) {
            outp += self.inner.call(chunk)?;
        }
        Ok(outp)
    }

    fn call_in_len(&self) -> usize {
        self.chunks * self.inner.call_in_len()
    }
}

impl<F: FieldElement, G: Gadget<F>> Gadget<F> for ParallelSum<F, G> {
    fn call_poly<V: Sync + AsRef<[F]>>(
        &mut self,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;

        let res = inp
            .par_chunks(self.inner.call_in_len())
            .map(|chunk| {
                let mut inner = self.inner.clone();
                let mut partial_outp = vec![F::zero(); outp.len()];
                inner.call_poly(&mut partial_outp, chunk).unwrap();
                partial_outp
            })
            .reduce(
                || vec![F::zero(); outp.len()],
                |mut x, y| {
                    for i in 0..x.len() {
                        x[i] += y[i];
                    }
                    x
                },
            );

        for i in 0..outp.len() {
            outp[i] = res[i];
        }

        Ok(())
    }

    fn call_poly_out_len(&self, in_len: usize) -> usize {
        self.inner.call_poly_out_len(in_len)
    }
}

// Check that the input parameters of g.call() are wll-formed.
fn gadget_call_check<F: FieldElement, G: GadgetCallOnly<F>>(
    g: &G,
    in_len: usize,
) -> Result<(), PcpError> {
    if in_len != g.call_in_len() {
        return Err(PcpError::CircuitInLen);
    }

    if in_len == 0 {
        return Err(PcpError::CircuitIn("can't call an arity-0 gadget"));
    }

    Ok(())
}

// Check that the input parameters of g.call_poly() are well-formed.
fn gadget_call_poly_check<F: FieldElement, G: GadgetCallOnly<F>, V: AsRef<[F]>>(
    g: &G,
    outp: &[F],
    inp: &[V],
) -> Result<(), PcpError>
where
    G: Gadget<F>,
{
    gadget_call_check(g, inp.len())?;

    for i in 1..inp.len() {
        if inp[i].as_ref().len() != inp[0].as_ref().len() {
            return Err(PcpError::GadgetPolyInLen);
        }
    }

    if outp.len() < g.call_poly_out_len(inp[0].as_ref().len()) {
        return Err(PcpError::GadgetPolyOutLen);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::{rand_vec, Field80 as TestField};
    use crate::polynomial::poly_range_check;

    #[test]
    fn test_mul() {
        // Test the gadget with input polynomials shorter than `FFT_THRESHOLD`. This exercises the
        // naive multiplication code path.
        let in_len = FFT_THRESHOLD - 1;
        let mut g: Mul<TestField> = Mul::new(in_len);
        gadget_test(&mut g, in_len);

        // Test the gadget with input polynomials longer than `FFT_THRESHOLD`. This exercises
        // FFT-based polynomial multiplication.
        let in_len = FFT_THRESHOLD.next_power_of_two();
        let mut g: Mul<TestField> = Mul::new(in_len);
        gadget_test(&mut g, in_len);
    }

    #[test]
    fn test_poly_eval() {
        let poly = rand_vec(10);

        let in_len = FFT_THRESHOLD - 1;
        let mut g: PolyEval<TestField> = PolyEval::new(poly.clone(), in_len);
        gadget_test(&mut g, in_len);

        let in_len = FFT_THRESHOLD.next_power_of_two();
        let mut g: PolyEval<TestField> = PolyEval::new(poly.clone(), in_len);
        gadget_test(&mut g, in_len);
    }

    #[test]
    fn test_blind_poly_eval() {
        let poly = rand_vec(10);

        let in_len = FFT_THRESHOLD - 1;
        let mut g: BlindPolyEval<TestField> = BlindPolyEval::new(poly.clone(), in_len);
        gadget_test(&mut g, in_len);

        let in_len = FFT_THRESHOLD.next_power_of_two();
        let mut g: BlindPolyEval<TestField> = BlindPolyEval::new(poly.clone(), in_len);
        gadget_test(&mut g, in_len);
    }

    #[test]
    fn test_parallel_sum() {
        let in_len = 10;
        let mut g: ParallelSum<TestField, BlindPolyEval<TestField>> =
            ParallelSum::new(BlindPolyEval::new(poly_range_check(0, 2), in_len), 23);
        gadget_test(&mut g, in_len);
    }

    // Test that calling g.call_poly() and evaluating the output at a given point is equivalent
    // to evaluating each of the inputs at the same point and applying g.call() on the results.
    fn gadget_test<F: FieldElement, G: GadgetCallOnly<F>>(g: &mut G, in_len: usize)
    where
        G: Gadget<F>,
    {
        let l = g.call_in_len();
        let mut inp = vec![F::zero(); l];
        let mut poly_outp = vec![F::zero(); g.call_poly_out_len(in_len)];
        let mut poly_inp = vec![vec![F::zero(); in_len]; l];

        let r = F::rand();
        for i in 0..l {
            poly_inp[i] = rand_vec(in_len);
            inp[i] = poly_eval(&poly_inp[i], r);
        }

        g.call_poly(&mut poly_outp, &poly_inp).unwrap();
        let got = poly_eval(&poly_outp, r);
        let want = g.call(&inp).unwrap();
        assert_eq!(got, want);

        // Repeat the call to make sure that the gadget's memory is reset properly between calls.
        g.call_poly(&mut poly_outp, &poly_inp).unwrap();
        let got = poly_eval(&poly_outp, r);
        let want = g.call(&inp).unwrap();
        assert_eq!(got, want);
    }
}
