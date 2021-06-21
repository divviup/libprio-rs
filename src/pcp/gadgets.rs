// SPDX-License-Identifier: MPL-2.0

//! A collection of gadgets.

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish};
use crate::field::FieldElement;
use crate::pcp::{Gadget, GadgetCallOnly, PcpError};
use crate::polynomial::{poly_deg, poly_eval, poly_mul, poly_range_check};

use std::convert::TryFrom;
use std::marker::PhantomData;

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
    fn call(&mut self, idx: usize, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, idx, inp.len())?;
        Ok(inp[0] * inp[1])
    }

    fn arity(&self, idx: usize) -> usize {
        if idx >= self.len() {
            return 0;
        }

        2
    }

    fn deg(&self, idx: usize) -> usize {
        if idx >= self.len() {
            return 0;
        }

        2
    }

    fn len(&self) -> usize {
        1
    }
}

impl<F: FieldElement> Gadget<F> for Mul<F> {
    fn call_poly<V: AsRef<[F]>>(
        &mut self,
        idx: usize,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        gadget_call_poly_check(self, idx, outp, inp)?;
        if inp[0].as_ref().len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
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
    fn call(&mut self, idx: usize, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, idx, inp.len())?;
        Ok(poly_eval(&self.poly, inp[0]))
    }

    fn arity(&self, idx: usize) -> usize {
        if idx >= self.len() {
            return 0;
        }

        1
    }

    fn deg(&self, idx: usize) -> usize {
        if idx >= self.len() {
            return 0;
        }

        poly_deg(&self.poly)
    }

    fn len(&self) -> usize {
        1
    }
}

impl<F: FieldElement> Gadget<F> for PolyEval<F> {
    fn call_poly<V: AsRef<[F]>>(
        &mut self,
        idx: usize,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        gadget_call_poly_check(self, idx, outp, inp)?;

        for i in 0..outp.len() {
            outp[i] = F::zero();
        }

        if inp[0].as_ref().len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
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
    /// Constructs a MeanVarUnsigned gadget with parameter `bits`.
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
    fn call(&mut self, idx: usize, inp: &[F]) -> Result<F, PcpError> {
        gadget_call_check(self, idx, inp.len())?;
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

    fn arity(&self, idx: usize) -> usize {
        if idx >= self.len() {
            return 0;
        }

        2 * self.bits + 1
    }

    fn deg(&self, idx: usize) -> usize {
        if idx >= self.len() {
            return 0;
        }

        3
    }

    fn len(&self) -> usize {
        1
    }
}

impl<F: FieldElement> Gadget<F> for MeanVarUnsigned<F> {
    fn call_poly<V: AsRef<[F]>>(
        &mut self,
        idx: usize,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        gadget_call_poly_check(self, idx, outp, inp)?;
        if inp[0].as_ref().len() >= FFT_THRESHOLD {
            self.call_poly_fft(outp, inp)
        } else {
            self.call_poly_direct(outp, inp)
        }
    }
}

/// XXX
/// XXX Constructor should check that G1 and G2 have output size of 1
pub struct Pair<F: FieldElement, G1: Gadget<F>, G2: Gadget<F>> {
    g1: G1,
    g2: G2,
    phantom: PhantomData<F>,
}

impl<F, G1, G2> Pair<F, G1, G2>
where
    F: FieldElement,
    G1: Gadget<F>,
    G2: Gadget<F>,
{
    /// XXX
    pub fn new(g1: G1, g2: G2) -> Self {
        Self {
            g1,
            g2,
            phantom: PhantomData,
        }
    }
}

impl<F, G1, G2> GadgetCallOnly<F> for Pair<F, G1, G2>
where
    F: FieldElement,
    G1: Gadget<F>,
    G2: Gadget<F>,
{
    fn call(&mut self, idx: usize, inp: &[F]) -> Result<F, PcpError> {
        match idx {
            0 => self.g1.call(0, inp),
            1 => self.g2.call(0, inp),
            _ => panic!("XXX"),
        }
    }

    fn arity(&self, idx: usize) -> usize {
        match idx {
            0 => self.g1.arity(0),
            1 => self.g2.arity(0),
            _ => 0,
        }
    }

    fn deg(&self, idx: usize) -> usize {
        match idx {
            0 => self.g1.deg(0),
            1 => self.g2.deg(0),
            _ => 0,
        }
    }

    fn len(&self) -> usize {
        2
    }
}

impl<F, G1, G2> Gadget<F> for Pair<F, G1, G2>
where
    F: FieldElement,
    G1: Gadget<F>,
    G2: Gadget<F>,
{
    fn call_poly<V: AsRef<[F]>>(
        &mut self,
        idx: usize,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError> {
        match idx {
            0 => self.g1.call_poly(0, outp, inp),
            1 => self.g2.call_poly(0, outp, inp),
            _ => panic!("XXX"),
        }
    }
}

// Check that the input parameters of g.call() are wll-formed.
fn gadget_call_check<F: FieldElement, G: GadgetCallOnly<F>>(
    g: &G,
    idx: usize,
    in_len: usize,
) -> Result<(), PcpError> {
    if idx >= g.len() {
        panic!("XXX index check");
    }

    if in_len != g.arity(idx) {
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
    idx: usize,
    outp: &[F],
    inp: &[V],
) -> Result<(), PcpError>
where
    G: Gadget<F>,
{
    gadget_call_check(g, idx, inp.len())?;

    for i in 1..inp.len() {
        if inp[i].as_ref().len() != inp[0].as_ref().len() {
            return Err(PcpError::GadgetPolyInLen);
        }
    }

    if outp.len() < g.deg(idx) * inp[0].as_ref().len() {
        return Err(PcpError::GadgetPolyOutLen);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::{rand, Field80 as TestField};
    use crate::prng::Prng;

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
        let poly = rand(10).unwrap();

        let in_len = FFT_THRESHOLD - 1;
        let mut g: PolyEval<TestField> = PolyEval::new(poly.clone(), in_len);
        gadget_test(&mut g, in_len);

        let in_len = FFT_THRESHOLD.next_power_of_two();
        let mut g: PolyEval<TestField> = PolyEval::new(poly.clone(), in_len);
        gadget_test(&mut g, in_len);
    }

    #[test]
    fn test_mean_var_unsigned() {
        let in_len = FFT_THRESHOLD - 1;
        let mut g: MeanVarUnsigned<TestField> = MeanVarUnsigned::new(12, in_len);
        gadget_test(&mut g, in_len);

        let in_len = FFT_THRESHOLD.next_power_of_two();
        let mut g: MeanVarUnsigned<TestField> = MeanVarUnsigned::new(5, in_len);
        gadget_test(&mut g, in_len);
    }

    #[test]
    fn test_pair() {
        let poly = rand(10).unwrap();
        let in_len = FFT_THRESHOLD - 1;

        let mut g = Pair::new(
            Mul::<TestField>::new(in_len),
            PolyEval::<TestField>::new(poly.clone(), in_len),
        );
        gadget_test(&mut g, in_len);
    }

    // Test that calling g.call_poly() and evaluating the output at a given point is equivalent
    // to evaluating each of the inputs at the same point and applying g.call() on the results.
    fn gadget_test<F: FieldElement, G: Gadget<F>>(g: &mut G, in_len: usize) {
        let mut prng = Prng::new().unwrap();

        // Calling arity or deg on an out-of-range index should always return 0.
        assert_eq!(g.arity(g.len()), 0);
        assert_eq!(g.deg(g.len()), 0);

        for idx in 0..g.len() {
            let mut inp = vec![F::zero(); g.arity(idx)];
            let mut poly_outp = vec![F::zero(); g.deg(idx) * in_len];
            let mut poly_inp = vec![vec![F::zero(); in_len]; g.arity(idx)];

            let r = prng.next().unwrap();
            for i in 0..g.arity(idx) {
                for j in 0..in_len {
                    poly_inp[i][j] = prng.next().unwrap();
                }
                inp[i] = poly_eval(&poly_inp[i], r);
            }

            g.call_poly(idx, &mut poly_outp, &poly_inp).unwrap();
            let got = poly_eval(&poly_outp, r);
            let want = g.call(idx, &inp).unwrap();
            assert_eq!(got, want);

            // Repeat the call to make sure that the gadget's memory is reset properly between calls.
            g.call_poly(idx, &mut poly_outp, &poly_inp).unwrap();
            let got = poly_eval(&poly_outp, r);
            assert_eq!(got, want);
        }
    }
}
