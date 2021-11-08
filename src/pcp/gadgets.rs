// SPDX-License-Identifier: MPL-2.0

//! A collection of gadgets.

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish};
use crate::field::FieldElement;
use crate::pcp::{Gadget, PcpError};
use crate::polynomial::{poly_deg, poly_eval, poly_mul};

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
    /// The number of times this gadget will be called.
    num_calls: usize,
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
        Self {
            n,
            n_inv,
            num_calls,
        }
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

    fn calls(&self) -> usize {
        self.num_calls
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// An arity-1 gadget that evaluates its input on some polynomial.
//
// TODO Make `poly` an array of length determined by a const generic.
pub struct PolyEval<F: FieldElement> {
    poly: Vec<F>,
    /// Size of buffer for FFT operations.
    n: usize,
    /// Inverse of `n` in `F`.
    n_inv: F,
    /// The number of times this gadget will be called.
    num_calls: usize,
}

impl<F: FieldElement> PolyEval<F> {
    /// Returns a gadget that evaluates its input on `poly`. `num_calls` is the number of times
    /// this gadget is called by the validity circuit.
    pub fn new(poly: Vec<F>, num_calls: usize) -> Self {
        let n = (poly_deg(&poly) * (1 + num_calls).next_power_of_two()).next_power_of_two();
        let n_inv = F::from(F::Integer::try_from(n).unwrap()).inv();
        Self {
            poly,
            n,
            n_inv,
            num_calls,
        }
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

    fn calls(&self) -> usize {
        self.num_calls
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
        return Err(PcpError::Gadget(format!(
            "unexpected number of inputs: got {}; want {}",
            in_len,
            gadget.arity()
        )));
    }

    if in_len == 0 {
        return Err(PcpError::Gadget("can't call an arity-0 gadget".to_string()));
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
            return Err(PcpError::Gadget(
                "gadget called on polynomials with different lengths".to_string(),
            ));
        }
    }

    if outp.len() < gadget.degree() * inp[0].len() {
        return Err(PcpError::Gadget(
            "slice allocated for gadget output polynomial is too small".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::Field96 as TestField;
    use crate::prng::Prng;
    use crate::vdaf::suite::Suite;

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
        let poly: Vec<TestField> = Prng::generate(Suite::Blake3).unwrap().take(10).collect();

        let num_calls = FFT_THRESHOLD / 2;
        let mut g: PolyEval<TestField> = PolyEval::new(poly.clone(), num_calls);
        gadget_test(&mut g, num_calls);

        let num_calls = FFT_THRESHOLD;
        let mut g: PolyEval<TestField> = PolyEval::new(poly, num_calls);
        gadget_test(&mut g, num_calls);
    }

    // Test that calling g.call_poly() and evaluating the output at a given point is equivalent
    // to evaluating each of the inputs at the same point and applying g.call() on the results.
    fn gadget_test<F: FieldElement, G: Gadget<F>>(g: &mut G, num_calls: usize) {
        let mut prng: Prng<F> = Prng::generate(Suite::Blake3).unwrap();
        let mut inp = vec![F::zero(); g.arity()];
        let mut poly_outp = vec![F::zero(); (g.degree() * (1 + num_calls)).next_power_of_two()];
        let mut poly_inp = vec![vec![F::zero(); 1 + num_calls]; g.arity()];

        let r = prng.get();
        for i in 0..g.arity() {
            for j in 0..num_calls {
                poly_inp[i][j] = prng.get();
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
