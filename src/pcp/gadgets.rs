// SPDX-License-Identifier: MPL-2.0

//! A collection of gadgets.

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv};
use crate::field::FieldElement;
use crate::pcp::{Gadget, GadgetCallOnly, PcpError};

/// An arity-2 gadget that multiples its inputs.
pub struct Mul<F: FieldElement> {
    buf: Vec<F>,
}

impl<F: FieldElement> Mul<F> {
    /// Return a new multiplier gadget.
    pub fn new(in_len: usize) -> Self {
        Self {
            buf: vec![F::zero(); 2 * in_len],
        }
    }

    // Use FTT to multiply the polynomials. This method is much faster than naive multiplication
    // for larger inputs.
    fn fft_call_poly<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        let n = 2 * inp[0].as_ref().len();

        discrete_fourier_transform(&mut self.buf, inp[0].as_ref(), n)?;
        discrete_fourier_transform(outp, inp[1].as_ref(), n)?;

        for i in 0..n {
            self.buf[i] *= outp[i];
        }

        Ok(discrete_fourier_transform_inv(outp, &self.buf, n)?)
    }
}

impl<F: FieldElement> GadgetCallOnly<F> for Mul<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, inp)?;
        Ok(inp[0] * inp[1])
    }

    fn call_in_len(&self) -> usize {
        2
    }
}

impl<F: FieldElement> Gadget<F> for Mul<F> {
    fn call_poly<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError> {
        gadget_call_poly_check(self, outp, inp)?;
        // TODO(cjpatton): For samll enough inputs, naive multiplication is actually faster than
        // using FFT. Figutre out what this threshold is.
        self.fft_call_poly(outp, inp)
    }

    fn call_poly_out_len(&self, in_len: usize) -> usize {
        2 * in_len
    }
}

// Check that the input parameters of g.call() are wll-formed.
fn gadget_call_check<F: FieldElement, G: GadgetCallOnly<F>>(
    g: &G,
    inp: &[F],
) -> Result<(), PcpError> {
    if inp.len() != g.call_in_len() {
        return Err(PcpError::CircuitInLen);
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
    if inp.len() != g.call_in_len() {
        return Err(PcpError::CircuitInLen);
    }

    if inp.len() == 0 {
        return Ok(());
    }

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

    use crate::field::Field80 as TestField;
    use crate::polynomial::poly_eval;

    // Test that calling g.call_poly() and evaluating the output at a given point is equivalent
    // to evaluating each of the inputs at the same point and aplying g.call() on the results.
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
    }

    #[test]
    fn test_mul_gadget() {
        let in_len = 128;
        let mut g: Mul<TestField> = Mul::new(in_len);
        gadget_test(&mut g, in_len);
    }
}
