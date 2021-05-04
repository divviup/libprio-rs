// SPDX-License-Identifier: MPL-2.0

//! A collection of gadgets.

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish};
use crate::field::FieldElement;
use crate::pcp::{Gadget, GadgetCallOnly, PcpError};
use crate::polynomial::{poly_deg, poly_eval, poly_mul, poly_range_check};

use std::convert::TryFrom;

/// For input polynomials larger than or equal to this threshold, gadgets will use FFT for
/// polynomial multiplication. Otherwise, the gadget uses direct multiplication.
const FFT_THRESHOLD: usize = 60;

/// An arity-2 gadget that multiples its inputs.
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
}

impl<F: FieldElement> PolyEval<F> {
    // Multiply input polynomials directly.
    fn call_poly_direct<V: AsRef<[F]>>(
        &mut self,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        outp[0] = self.poly[0];
        let mut x = inp[0].as_ref().to_vec();
        for i in 1..self.poly.len() {
            for j in 0..x.len() {
                outp[j] += self.poly[i] * x[j];
            }

            if i < self.poly.len() - 1 {
                x = poly_mul(&x, inp[0].as_ref());
            }
        }
        Ok(())
    }

    // Multiply input polynomials using FFT.
    fn call_poly_fft<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        let n = self.n;
        let inp = inp[0].as_ref();

        let mut inp_vals = vec![F::zero(); n];
        discrete_fourier_transform(&mut inp_vals, inp, n)?;

        let mut x_vals = inp_vals.clone();
        let mut x = vec![F::zero(); n];
        x[..inp.len()].clone_from_slice(inp);

        outp[0] = self.poly[0];
        for i in 1..self.poly.len() {
            for j in 0..outp.len() {
                outp[j] += self.poly[i] * x[j];
            }

            if i < self.poly.len() - 1 {
                for j in 0..n {
                    x_vals[j] *= inp_vals[j];
                }

                discrete_fourier_transform(&mut x, &x_vals, n)?;
                discrete_fourier_transform_inv_finish(&mut x, n, self.n_inv);
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

/// XXX ...
pub struct MeanVarUnsigned<F: FieldElement> {
    poly: [F; 3],
    /// Size of buffer for FFT operations.
    n: usize,
    /// Inverse of `n` in `F`.
    n_inv: F,
    /// XXX
    bits: usize,
}

impl<F: FieldElement> MeanVarUnsigned<F> {
    /// XXX
    pub fn new(bits: usize, in_len: usize) -> Self {
        let poly: Vec<F> = poly_range_check(0, 2);
        let n = (in_len * 3).next_power_of_two();
        let n_inv = F::from(F::Integer::try_from(n).unwrap()).inv();

        Self {
            poly: [poly[0], poly[1], poly[2]],
            n,
            n_inv,
            bits,
        }
    }

    pub(crate) fn call_poly_direct<V: AsRef<[F]>>(
        &mut self,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        let bits = self.bits;
        let r_vec = &inp[..bits];
        let x_vec = &inp[bits..2 * bits];
        let x = inp[2 * bits].as_ref();

        let z = poly_mul(x, x);
        for i in 0..z.len() {
            outp[i] = z[i];
        }
        for i in z.len()..outp.len() {
            outp[i] = F::zero();
        }

        for l in 0..bits {
            let mut z = r_vec[l].as_ref().to_vec();
            for i in 0..3 {
                for j in 0..z.len() {
                    outp[j] += self.poly[i] * z[j];
                }

                if i < 2 {
                    z = poly_mul(&z, x_vec[l].as_ref());
                }
            }
        }

        Ok(())
    }

    pub(crate) fn call_poly_fft<V: AsRef<[F]>>(
        &mut self,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        let bits = self.bits;
        let n = self.n;
        let r_vec = &inp[..bits];
        let x_vec = &inp[bits..2 * bits];
        let x = inp[2 * bits].as_ref();

        let mut x_vals = vec![F::zero(); n];
        let mut z_vals = vec![F::zero(); n];
        let mut z = vec![F::zero(); n];

        let m = n / 2;
        let m_inv = self.n_inv * F::from(F::Integer::try_from(2).unwrap());
        discrete_fourier_transform(&mut x_vals, x, m)?;
        for j in 0..m {
            z_vals[j] = x_vals[j] * x_vals[j];
        }
        discrete_fourier_transform(&mut z, &z_vals, m)?;
        discrete_fourier_transform_inv_finish(&mut z, m, m_inv);
        for i in 0..outp.len() {
            outp[i] = z[i];
        }

        for l in 0..bits {
            let x = x_vec[l].as_ref();
            let y = r_vec[l].as_ref();
            z[..y.len()].clone_from_slice(y);

            discrete_fourier_transform(&mut x_vals, x, n)?;
            discrete_fourier_transform(&mut z_vals, y, n)?;

            let mut z_len = y.len();
            for i in 0..3 {
                for j in 0..z_len {
                    outp[j] += self.poly[i] * z[j];
                }

                if i < 2 {
                    for j in 0..n {
                        z_vals[j] *= x_vals[j];
                    }

                    discrete_fourier_transform(&mut z, &z_vals, n)?;
                    discrete_fourier_transform_inv_finish(&mut z, n, self.n_inv);
                    z_len += x.len();
                }
            }
        }

        Ok(())
    }
}

impl<F: FieldElement> GadgetCallOnly<F> for MeanVarUnsigned<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, inp.len())?;
        let bits = self.bits;
        let r_vec = &inp[..bits];
        let x_vec = &inp[bits..2 * bits];
        let x = inp[2 * bits];

        let mut res = x * x;

        // Check that `x_vec` is a bit vector.
        for l in 0..bits {
            res += r_vec[l] * poly_eval(&self.poly, x_vec[l]);
        }

        Ok(res)
    }

    fn call_in_len(&self) -> usize {
        2 * self.bits + 1
    }
}

impl<F: FieldElement> Gadget<F> for MeanVarUnsigned<F> {
    fn call_poly<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;
        if inp[0].as_ref().len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
    }

    fn call_poly_out_len(&self, in_len: usize) -> usize {
        in_len * 3
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
    fn test_mean_var_int() {
        let in_len = FFT_THRESHOLD - 1;
        let mut g: MeanVarUnsigned<TestField> = MeanVarUnsigned::new(12, in_len);
        gadget_test(&mut g, in_len);

        let in_len = FFT_THRESHOLD.next_power_of_two();
        let mut g: MeanVarUnsigned<TestField> = MeanVarUnsigned::new(5, in_len);
        gadget_test(&mut g, in_len);
    }

    // Test that calling g.call_poly() and evaluating the output at a given point is equivalent
    // to evaluating each of the inputs at the same point and applying g.call() on the results.
    fn gadget_test<F: FieldElement, G: GadgetCallOnly<F>>(g: &mut G, in_len: usize)
    where
        G: Gadget<F>,
    {
        let mut inp = vec![F::zero(); g.call_in_len()];
        let mut poly_outp = vec![F::zero(); g.call_poly_out_len(in_len)];
        let mut poly_inp = vec![vec![F::zero(); in_len]; g.call_in_len()];

        let r = F::rand();
        for i in 0..g.call_in_len() {
            for j in 0..in_len {
                poly_inp[i][j] = F::rand();
            }
            inp[i] = poly_eval(&poly_inp[i], r);
        }

        g.call_poly(&mut poly_outp, &poly_inp).unwrap();
        let got = poly_eval(&poly_outp, r);
        let want = g.call(&inp).unwrap();
        assert_eq!(got, want);

        // Repeat the call to make sure that the gadget's memory is reset properly between calls.
        g.call_poly(&mut poly_outp, &poly_inp).unwrap();
        let got = poly_eval(&poly_outp, r);
        assert_eq!(got, want);
    }
}
