// SPDX-License-Identifier: MPL-2.0

//! A collection of gadgets.

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish};
use crate::field::FieldElement;
use crate::pcp::{Gadget, PcpError};
use crate::polynomial::{poly_deg, poly_eval, poly_mul, poly_range_check};

use std::any::Any;
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
    /// Return a new multiplier gadget. `num_calls` is the number of times this gadget will be
    /// called by the validity circuit.
    pub fn new(num_calls: usize) -> Self {
        // Compute the amount of memory that will be needed for the output of `call_poly`. The
        // degree of this gadget is `2`, so this is `2 * (1 + num_calls).next_power_of_two()`.
        // (We round up to the next power of two in order to make room for FFT.)
        let n = (2 * (1 + num_calls).next_power_of_two()).next_power_of_two();
        let n_inv = F::from(F::Integer::try_from(n).unwrap()).inv();
        Self { n, n_inv }
    }

    // Multiply input polynomials directly.
    pub(crate) fn call_poly_direct(
        &mut self,
        outp: &mut [F],
        inp: &[Vec<F>],
    ) -> Result<(), PcpError> {
        let v = poly_mul(&inp[0], &inp[1]);
        outp[..v.len()].clone_from_slice(&v);
        Ok(())
    }

    // Multiply input polynomials using FFT.
    pub(crate) fn call_poly_fft(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError> {
        let n = self.n;
        let mut buf = vec![F::zero(); n];

        discrete_fourier_transform(&mut buf, &inp[0], n)?;
        discrete_fourier_transform(outp, &inp[1], n)?;

        for i in 0..n {
            buf[i] *= outp[i];
        }

        discrete_fourier_transform(outp, &buf, n)?;
        discrete_fourier_transform_inv_finish(outp, n, self.n_inv);
        Ok(())
    }
}

impl<F: FieldElement> Gadget<F> for Mul<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, inp.len())?;
        Ok(inp[0] * inp[1])
    }

    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;
        if inp[0].len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
    }

    fn arity(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        2
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
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
    /// Returns a gadget that evaluates its input on `poly`. `num_calls` is the number of times
    /// this gadget is called by the validity circuit.
    pub fn new(poly: Vec<F>, num_calls: usize) -> Self {
        let n = (poly_deg(&poly) * (1 + num_calls).next_power_of_two()).next_power_of_two();
        let n_inv = F::from(F::Integer::try_from(n).unwrap()).inv();
        Self { poly, n, n_inv }
    }
}

impl<F: FieldElement> PolyEval<F> {
    // Multiply input polynomials directly.
    fn call_poly_direct(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError> {
        outp[0] = self.poly[0];
        let mut x = inp[0].to_vec();
        for i in 1..self.poly.len() {
            for j in 0..x.len() {
                outp[j] += self.poly[i] * x[j];
            }

            if i < self.poly.len() - 1 {
                x = poly_mul(&x, &inp[0]);
            }
        }
        Ok(())
    }

    // Multiply input polynomials using FFT.
    fn call_poly_fft(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError> {
        let n = self.n;
        let inp = &inp[0];

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

impl<F: FieldElement> Gadget<F> for PolyEval<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, inp.len())?;
        Ok(poly_eval(&self.poly, inp[0]))
    }

    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;

        for item in outp.iter_mut() {
            *item = F::zero();
        }

        if inp[0].len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
    }

    fn arity(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        poly_deg(&self.poly)
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// The gadget for the MeanVarUnsignedVector type. It is not designed for general use.
///
/// MeanVarUnsigned is parameterized by a positive integer `bits`. Its arity is `2*bits+1`:
///
///  * The first `bits` values are interpreted as a vector `r_vec`.
///  * The next `bits` values are interpreted as a vector `x_vec`.
///  * The last value is interpreted as a singleton `x`.
///
/// The gadget is designed to output `x^2` *as long as* `x_vec` is a vector comprised of 0s and 1s.
/// It first computes `w[l] = p(x_vec[l]) * r_vec[l]` for each `0 <= l < bits`, where `p(x) =
/// x(x-1)`. It then computes `w = w[0] + ... + w[bits-1]` and returns `w + x^2`.
///
/// If `vec_x[l] == 1` or `vec_x[l] == 0`, then `p(vec_x[l]) == 0`; otherwise, `p(vec_x[l]) != 0`.
/// The validity circuit for MeanVarUnsignedVector sets `r_vec[l] = r^(l+1)`, where `r` is a
/// uniform random field element. This is ensures that, if `p(vec_x[l]) != 0` for some `l`, then
/// `w != 0` with high probability.
pub struct MeanVarUnsigned<F: FieldElement> {
    /// Polynomial used to check that each element of `x_vec` is either `0` or `1`.
    poly: [F; 3],
    /// Size of buffer for FFT operations.
    n: usize,
    /// Inverse of `n` in `F`.
    n_inv: F,
    /// The parameter that determines the circuit's arity.
    bits: usize,
}

impl<F: FieldElement> MeanVarUnsigned<F> {
    /// Constructs a MeanVarUnsigned gadget with parameter `bits`. `num_calls` is the number of
    /// times this gadget is called by the validity circuit.
    pub fn new(bits: usize, num_calls: usize) -> Self {
        let poly: Vec<F> = poly_range_check(0, 2);
        let n = (3 * (1 + num_calls).next_power_of_two()).next_power_of_two();
        let n_inv = F::from(F::Integer::try_from(n).unwrap()).inv();

        Self {
            poly: [poly[0], poly[1], poly[2]],
            n,
            n_inv,
            bits,
        }
    }

    pub(crate) fn call_poly_direct(
        &mut self,
        outp: &mut [F],
        inp: &[Vec<F>],
    ) -> Result<(), PcpError> {
        let bits = self.bits;
        let r_vec = &inp[..bits];
        let x_vec = &inp[bits..2 * bits];
        let x = &inp[2 * bits];

        let z = poly_mul(x, x);
        outp[..z.len()].clone_from_slice(&z[..]);
        for item in outp.iter_mut().skip(z.len()) {
            *item = F::zero();
        }

        for l in 0..bits {
            let mut z = r_vec[l].to_vec();
            for i in 0..3 {
                for j in 0..z.len() {
                    outp[j] += self.poly[i] * z[j];
                }

                if i < 2 {
                    z = poly_mul(&z, &x_vec[l]);
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::many_single_char_names)]
    pub(crate) fn call_poly_fft(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError> {
        let bits = self.bits;
        let n = self.n;
        let r_vec = &inp[..bits];
        let x_vec = &inp[bits..2 * bits];
        let x = &inp[2 * bits];

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
        outp.clone_from_slice(&z[..outp.len()]);

        for l in 0..bits {
            let x = &x_vec[l];
            let y = &r_vec[l];
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

impl<F: FieldElement> Gadget<F> for MeanVarUnsigned<F> {
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

    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;
        if inp[0].len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
    }

    fn arity(&self) -> usize {
        2 * self.bits + 1
    }

    fn degree(&self) -> usize {
        3
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

// Check that the input parameters of g.call() are wll-formed.
fn gadget_call_check<F: FieldElement, G: Gadget<F>>(
    gadget: &G,
    in_len: usize,
) -> Result<(), PcpError> {
    if in_len != gadget.arity() {
        return Err(PcpError::CircuitInLen);
    }

    if in_len == 0 {
        return Err(PcpError::CircuitIn("can't call an arity-0 gadget"));
    }

    Ok(())
}

// Check that the input parameters of g.call_poly() are well-formed.
fn gadget_call_poly_check<F: FieldElement, G: Gadget<F>>(
    gadget: &G,
    outp: &[F],
    inp: &[Vec<F>],
) -> Result<(), PcpError>
where
    G: Gadget<F>,
{
    gadget_call_check(gadget, inp.len())?;

    for i in 1..inp.len() {
        if inp[i].len() != inp[0].len() {
            return Err(PcpError::GadgetPolyInLen);
        }
    }

    if outp.len() < gadget.degree() * inp[0].len() {
        return Err(PcpError::GadgetPolyOutLen);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::{rand, Field80 as TestField};
    use crate::prng::{Prng, STREAM_CIPHER_AES128CTR_KEY_LENGTH};

    use aes::Aes128Ctr;

    #[test]
    fn test_mul() {
        // Test the gadget with input polynomials shorter than `FFT_THRESHOLD`. This exercises the
        // naive multiplication code path.
        let num_calls = FFT_THRESHOLD / 2;
        let mut g: Mul<TestField> = Mul::new(num_calls);
        gadget_test(&mut g, num_calls);

        // Test the gadget with input polynomials longer than `FFT_THRESHOLD`. This exercises
        // FFT-based polynomial multiplication.
        let num_calls = FFT_THRESHOLD;
        let mut g: Mul<TestField> = Mul::new(num_calls);
        gadget_test(&mut g, num_calls);
    }

    #[test]
    fn test_poly_eval() {
        let poly = rand(10).unwrap();

        let num_calls = FFT_THRESHOLD / 2;
        let mut g: PolyEval<TestField> = PolyEval::new(poly.clone(), num_calls);
        gadget_test(&mut g, num_calls);

        let num_calls = FFT_THRESHOLD;
        let mut g: PolyEval<TestField> = PolyEval::new(poly, num_calls);
        gadget_test(&mut g, num_calls);
    }

    #[test]
    fn test_mean_var_unsigned() {
        let num_calls = FFT_THRESHOLD / 2;
        let mut g: MeanVarUnsigned<TestField> = MeanVarUnsigned::new(12, num_calls);
        gadget_test(&mut g, num_calls);

        let num_calls = FFT_THRESHOLD;
        let mut g: MeanVarUnsigned<TestField> = MeanVarUnsigned::new(5, num_calls);
        gadget_test(&mut g, num_calls);
    }

    // Test that calling g.call_poly() and evaluating the output at a given point is equivalent
    // to evaluating each of the inputs at the same point and applying g.call() on the results.
    fn gadget_test<F: FieldElement, G: Gadget<F>>(g: &mut G, num_calls: usize) {
        let mut prng = Prng::<F, Aes128Ctr, STREAM_CIPHER_AES128CTR_KEY_LENGTH>::new().unwrap();
        let mut inp = vec![F::zero(); g.arity()];
        let mut poly_outp = vec![F::zero(); (g.degree() * (1 + num_calls)).next_power_of_two()];
        let mut poly_inp = vec![vec![F::zero(); 1 + num_calls]; g.arity()];

        let r = prng.next().unwrap();
        for i in 0..g.arity() {
            for j in 0..num_calls {
                poly_inp[i][j] = prng.next().unwrap();
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
