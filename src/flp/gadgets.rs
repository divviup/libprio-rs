// SPDX-License-Identifier: MPL-2.0

//! A collection of gadgets.

#[cfg(feature = "multithreaded")]
use crate::field::add_vector;
use crate::field::NttFriendlyFieldElement;
use crate::flp::{gadget_poly_len, wire_poly_len, FlpError, Gadget};
use crate::ntt::{get_ntt, get_ntt_inv};
use crate::polynomial::{poly_deg, poly_eval_monomial, poly_mul_lagrange};

#[cfg(feature = "multithreaded")]
use rayon::prelude::*;

use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;

/// An arity-2 gadget that multiples its inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mul {
    /// The number of times this gadget will be called.
    num_calls: usize,
}

impl Mul {
    /// Return a new multiplier gadget. `num_calls` is the number of times this gadget will be
    /// called by the validity circuit.
    pub fn new(num_calls: usize) -> Self {
        Self { num_calls }
    }
}

impl<F: NttFriendlyFieldElement> Gadget<F> for Mul {
    fn eval(&mut self, inp: &[F]) -> Result<F, FlpError> {
        gadget_eval_check::<F, _>(self, inp.len())?;
        Ok(inp[0] * inp[1])
    }

    fn eval_poly(&self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), FlpError> {
        gadget_eval_poly_check(self, outp, inp)?;

        poly_mul_lagrange(outp, &inp[0], &inp[1])?;

        Ok(())
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyEval<F: NttFriendlyFieldElement> {
    /// The coefficients of the polynomial, in the monomial basis.
    poly: Vec<F>,
    /// The number of times this gadget will be called.
    num_calls: usize,
}

impl<F: NttFriendlyFieldElement> PolyEval<F> {
    /// Returns a gadget that evaluates its input on `poly`, a polynomial in the monomial basis.
    /// `num_calls` is the number of times this gadget is called by the validity circuit.
    pub fn new(poly: Vec<F>, num_calls: usize) -> Self {
        Self { poly, num_calls }
    }
}

impl<F: NttFriendlyFieldElement> Gadget<F> for PolyEval<F> {
    fn eval(&mut self, inp: &[F]) -> Result<F, FlpError> {
        gadget_eval_check(self, inp.len())?;
        Ok(poly_eval_monomial(&self.poly, inp[0]))
    }

    fn eval_poly(&self, outp: &mut [F], inp_lagrange: &[Vec<F>]) -> Result<(), FlpError> {
        gadget_eval_poly_check(self, outp, inp_lagrange)?;

        // Convert input polynomial from Lagrange to monomial basis
        let inp_monomial = get_ntt_inv(&inp_lagrange[0], inp_lagrange[0].len())?;
        // Extend to n evaluations
        let n = gadget_poly_len(self.degree(), wire_poly_len(self.num_calls)).next_power_of_two();
        let inp_lagrange_extended = get_ntt(&inp_monomial, n)?;
        // Compose input polynomial with this gadget's polynomial by evaluating self.poly at each
        // point of inp
        for (x, outp_element) in inp_lagrange_extended.into_iter().zip(outp.iter_mut()) {
            *outp_element = poly_eval_monomial(&self.poly, x);
        }

        Ok(())
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

/// Trait for abstracting over [`ParallelSum`].
pub trait ParallelSumGadget<F: NttFriendlyFieldElement, G>: Gadget<F> + Debug {
    /// Wraps `inner` into a sum gadget that calls it `chunks` many times, and adds the reuslts.
    fn new(inner: G, chunks: usize) -> Self;
}

/// A wrapper gadget that applies the inner gadget to chunks of input and returns the sum of the
/// outputs. The arity is equal to the arity of the inner gadget times the number of times it is
/// called.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelSum<F: NttFriendlyFieldElement, G: Gadget<F>> {
    inner: G,
    chunks: usize,
    phantom: PhantomData<F>,
}

impl<F: NttFriendlyFieldElement, G: 'static + Gadget<F>> ParallelSumGadget<F, G>
    for ParallelSum<F, G>
{
    fn new(inner: G, chunks: usize) -> Self {
        Self {
            inner,
            chunks,
            phantom: PhantomData,
        }
    }
}

impl<F: NttFriendlyFieldElement, G: 'static + Gadget<F>> Gadget<F> for ParallelSum<F, G> {
    fn eval(&mut self, inp: &[F]) -> Result<F, FlpError> {
        gadget_eval_check(self, inp.len())?;
        let mut outp = F::zero();
        for chunk in inp.chunks(self.inner.arity()) {
            outp += self.inner.eval(chunk)?;
        }
        Ok(outp)
    }

    fn eval_poly(&self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), FlpError> {
        gadget_eval_poly_check(self, outp, inp)?;

        for x in outp.iter_mut() {
            *x = F::zero();
        }

        let mut partial_outp = vec![F::zero(); outp.len()];

        for chunk in inp.chunks(self.inner.arity()) {
            self.inner.eval_poly(&mut partial_outp, chunk)?;
            for i in 0..outp.len() {
                outp[i] += partial_outp[i]
            }
        }

        Ok(())
    }

    fn arity(&self) -> usize {
        self.chunks * self.inner.arity()
    }

    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn calls(&self) -> usize {
        self.inner.calls()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// A wrapper gadget that applies the inner gadget to chunks of input and returns the sum of the
/// outputs. The arity is equal to the arity of the inner gadget times the number of chunks. The sum
/// evaluation is multithreaded.
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelSumMultithreaded<F: NttFriendlyFieldElement, G: Gadget<F>> {
    serial_sum: ParallelSum<F, G>,
}

#[cfg(feature = "multithreaded")]
impl<F, G> ParallelSumGadget<F, G> for ParallelSumMultithreaded<F, G>
where
    F: NttFriendlyFieldElement + Sync + Send,
    G: 'static + Gadget<F> + Clone + Sync + Send,
{
    fn new(inner: G, chunks: usize) -> Self {
        Self {
            serial_sum: ParallelSum::new(inner, chunks),
        }
    }
}

/// Data structures passed between fold operations in [`ParallelSumMultithreaded`].
#[cfg(feature = "multithreaded")]
struct ParallelSumFoldState<F, G> {
    /// Inner gadget.
    inner: G,
    /// Output buffer for `eval_poly()`.
    partial_output: Vec<F>,
    /// Sum accumulator.
    partial_sum: Vec<F>,
}

#[cfg(feature = "multithreaded")]
impl<F, G> ParallelSumFoldState<F, G> {
    fn new(gadget: &G, length: usize) -> ParallelSumFoldState<F, G>
    where
        G: Clone,
        F: NttFriendlyFieldElement,
    {
        ParallelSumFoldState {
            inner: gadget.clone(),
            partial_output: vec![F::zero(); length],
            partial_sum: vec![F::zero(); length],
        }
    }
}

#[cfg(feature = "multithreaded")]
impl<F, G> Gadget<F> for ParallelSumMultithreaded<F, G>
where
    F: NttFriendlyFieldElement + Sync + Send,
    G: 'static + Gadget<F> + Clone + Sync + Send,
{
    fn eval(&mut self, inp: &[F]) -> Result<F, FlpError> {
        self.serial_sum.eval(inp)
    }

    fn eval_poly(&self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), FlpError> {
        gadget_eval_poly_check(self, outp, inp)?;

        // Create a copy of the inner gadget and two working buffers on each thread. Evaluate the
        // gadget on each input polynomial, using the first temporary buffer as an output buffer.
        // Then accumulate that result into the second temporary buffer, which acts as a running
        // sum. Then, discard everything but the partial sums, add them, and finally copy the sum
        // to the output parameter. This is equivalent to the single threaded calculation in
        // ParallelSum, since we only rearrange additions, and field addition is associative.
        let res = inp
            .par_chunks(self.serial_sum.inner.arity())
            .fold(
                || ParallelSumFoldState::new(&self.serial_sum.inner, outp.len()),
                |mut state, chunk| {
                    state
                        .inner
                        .eval_poly(&mut state.partial_output, chunk)
                        .unwrap();
                    for (sum_elem, output_elem) in state
                        .partial_sum
                        .iter_mut()
                        .zip(state.partial_output.iter())
                    {
                        *sum_elem += *output_elem;
                    }
                    state
                },
            )
            .map(|state| state.partial_sum)
            .reduce(|| vec![F::zero(); outp.len()], add_vector);

        outp.copy_from_slice(&res[..]);
        Ok(())
    }

    fn arity(&self) -> usize {
        self.serial_sum.arity()
    }

    fn degree(&self) -> usize {
        self.serial_sum.degree()
    }

    fn calls(&self) -> usize {
        self.serial_sum.calls()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// Check that the input parameters of `G::eval` are well-formed.
fn gadget_eval_check<F: NttFriendlyFieldElement, G: Gadget<F>>(
    gadget: &G,
    in_len: usize,
) -> Result<(), FlpError> {
    if in_len != gadget.arity() {
        return Err(FlpError::Gadget(format!(
            "unexpected number of inputs: got {}; want {}",
            in_len,
            gadget.arity()
        )));
    }

    if in_len == 0 {
        return Err(FlpError::Gadget("can't call an arity-0 gadget".to_string()));
    }

    Ok(())
}

/// Check that the input parameters of `G::eval_poly` are well-formed.
fn gadget_eval_poly_check<F: NttFriendlyFieldElement, G: Gadget<F>, P: AsRef<[F]>>(
    gadget: &G,
    outp: &[F],
    inp: &[P],
) -> Result<(), FlpError> {
    gadget_eval_check(gadget, inp.len())?;

    for i in 1..inp.len() {
        if inp[i].as_ref().len() != inp[0].as_ref().len() {
            return Err(FlpError::Gadget(
                "gadget called on wire polynomials with different lengths".to_string(),
            ));
        }
    }

    let expected = gadget_poly_len(gadget.degree(), inp[0].as_ref().len()).next_power_of_two();
    if outp.len() != expected {
        return Err(FlpError::Gadget(format!(
            "incorrect output length: got {}; want {}",
            outp.len(),
            expected
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_with;

    use super::*;

    use crate::{
        field::{Field64 as TestField, FieldElement},
        polynomial::poly_eval_lagrange_batched,
    };

    #[test]
    fn test_mul() {
        gadget_test::<TestField, _>(&mut Mul::new(20));
    }

    #[test]
    fn test_poly_eval() {
        gadget_test::<TestField, _>(&mut PolyEval::new(TestField::random_vector(10), 30));
    }

    #[test]
    fn test_parallel_sum() {
        let num_calls = 10;
        let chunks = 23;

        let mut g = ParallelSum::<TestField, _>::new(Mul::new(num_calls), chunks);
        gadget_test(&mut g);
    }

    #[test]
    #[cfg(feature = "multithreaded")]
    fn test_parallel_sum_multithreaded() {
        for num_calls in [1, 10, 100] {
            let chunks = 23;

            let mut g = ParallelSumMultithreaded::new(Mul::new(num_calls), chunks);
            gadget_test(&mut g);

            // Test that the multithreaded version has the same output as the normal version.
            let mut g_serial = ParallelSum::new(Mul::new(num_calls), chunks);
            assert_eq!(g.arity(), g_serial.arity());
            assert_eq!(g.degree(), g_serial.degree());
            assert_eq!(g.calls(), g_serial.calls());

            let arity = g.arity();
            let degree = g.degree();

            // Test that both gadgets evaluate to the same value when run on scalar inputs.
            let inp = TestField::random_vector(arity);
            let result = g.eval(&inp).unwrap();
            let result_serial = g_serial.eval(&inp).unwrap();
            assert_eq!(result, result_serial);

            // Test that both gadgets evaluate to the same value when run on polynomial inputs.
            let mut poly_outp =
                vec![TestField::zero(); (degree * num_calls + 1).next_power_of_two()];
            let mut poly_outp_serial =
                vec![TestField::zero(); (degree * num_calls + 1).next_power_of_two()];
            let poly_inp: Vec<_> =
                std::iter::repeat_with(|| TestField::random_vector(wire_poly_len(num_calls)))
                    .take(arity)
                    .collect();

            g.eval_poly(&mut poly_outp, &poly_inp).unwrap();
            g_serial
                .eval_poly(&mut poly_outp_serial, &poly_inp)
                .unwrap();
            assert_eq!(poly_outp, poly_outp_serial);
        }
    }

    /// Test that calling g.eval_poly() and evaluating the output at a given point is equivalent
    /// to evaluating each of the inputs at the same point and applying g.eval() on the results.
    fn gadget_test<F: NttFriendlyFieldElement, G: Gadget<F>>(g: &mut G) {
        let random_point = F::random_vector(1)[0];

        let wire_polys: Vec<_> = repeat_with(|| F::random_vector(wire_poly_len(g.calls())))
            .take(g.arity())
            .collect();

        // Evaluate each wire polynomial at the random point and call the gadget on the result.
        let wire_poly_evaluations = poly_eval_lagrange_batched(&wire_polys, random_point);
        let want = g.eval(&wire_poly_evaluations).unwrap();

        // Compute the gadget polynomial and evaluate it at the random point.
        let mut gadget_poly = vec![
            F::zero();
            gadget_poly_len(g.degree(), wire_poly_len(g.calls()))
                .next_power_of_two()
        ];

        g.eval_poly(&mut gadget_poly, &wire_polys).unwrap();
        let gadget_poly_evals = poly_eval_lagrange_batched(&[&gadget_poly], random_point);
        let gadget_poly_eval = gadget_poly_evals[0];

        assert_eq!(want, gadget_poly_eval, "num calls: {}", g.calls());

        // Repeat the call to make sure that the gadget's memory is reset properly between calls.
        g.eval_poly(&mut gadget_poly, &wire_polys).unwrap();
        let gadget_poly_evals = poly_eval_lagrange_batched(&[&gadget_poly], random_point);
        let gadget_poly_eval = gadget_poly_evals[0];

        assert_eq!(want, gadget_poly_eval);
    }
}
