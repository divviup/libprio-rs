// SPDX-License-Identifier: MPL-2.0

//! This module implements the fully linear PCP ("Probabilistically Checkable Proof") system
//! described in \[BBG+19, Theorem 4.3\]. This is the core component of Prio's input-validation
//! protocol \[GB17\].
//!
//! # Overview
//!
//! The proof system is comprised of three algorithms. The first, `prove`, is run by the prover in
//! order to generate a proof of a statement's validity. The second and third, `query` and
//! `decide`, are run by the verifier in order to check the proof. In our setting, the proof
//! asserts that the prover's is a valid encoding of a value of a given type. For example:
//!
//! ```
//! use prio::pcp::types::Boolean;
//! use prio::pcp::{decide, prove, query};
//! use prio::field::{FieldElement, Field64};
//!
//! // The prover generates a proof pf that its input x is a valid encoding
//! // of a boolean (either "true" or "false"). Both the input and proof are
//! // vectors over the finite field specified by Field64.
//! let x: Boolean<Field64> = Boolean::new(false);
//! let pf = prove(&x).unwrap();
//!
//! // The verifier "queries" the proof pf and input x, getting a "verification
//! // message" in response. It uses this message to decide if the input is
//! // valid.
//! let vf = query(&x, &pf, &[Field64::rand()]).unwrap();
//! let res = decide(&x, &vf).unwrap();
//! assert_eq!(res, true);
//! ```
//!
//! If an input is _not_ valid, then the verification step will fail with high probability:
//!
//! ```
//! use prio::pcp::types::Boolean;
//! use prio::pcp::{decide, prove, query, Value};
//! use prio::field::{FieldElement, Field64};
//!
//! let x = Boolean::from(Field64::from(23));
//! let pf = prove(&x).unwrap();
//! let vf = query(&x, &pf, &[Field64::rand()]).unwrap();
//! let res = decide(&x, &vf).unwrap();
//! assert_eq!(res, false);
//! ```
//!
//! The "fully linear" property of the proof system allows the protocol to be executed over
//! secret-shared data. In this setting, the prover uses an additive secret sharing scheme to
//! "split" its input and proof into a number of shares and distributes the shares among a set of
//! verifiers. Each verifier queries its input and proof share locally. One of the verifiers
//! collects the outputs and uses them to decide if the input was valid. This procedure allows the
//! verifiers to validate a user's input without ever seeing the input in the clear:
//!
//! ```
//! use prio::pcp::types::Boolean;
//! use prio::pcp::{decide, prove, query, Value, Proof, Verifier};
//! use prio::field::{split, FieldElement, Field64};
//!
//! use std::convert::TryFrom;
//!
//! // The prover encodes its input and splits it into two secret shares. It
//! // sends each share to two aggregators.
//! let x: Boolean<Field64>= Boolean::new(true);
//! let x_shares: Vec<Boolean<Field64>> = split(x.as_slice(), 2)
//!     .into_iter()
//!     .map(|data| x.new_with(data))
//!     .collect();
//!
//! // The prover generates a proof of its input's validity and splits the proof
//! // into two shares. It sends each share to one of two aggregators.
//! let pf = prove(&x).unwrap();
//! let pf_shares: Vec<Proof<Field64>> = split(pf.as_slice(), 2)
//!     .into_iter()
//!     .map(|data| Proof::from(data))
//!     .collect();
//!
//! // The verifiers agree on the randomness they'll use to jointly verify the
//! // proof.
//! let rand = [Field64::rand()];
//!
//! // Each verifier queries its shares of the input and proof and sends its
//! // share of the verification message to the leader.
//! let vf_shares = vec![
//!     query(&x_shares[0], &pf_shares[0], &rand).unwrap(),
//!     query(&x_shares[1], &pf_shares[1], &rand).unwrap(),
//! ];
//!
//! // The leader collects the verifier shares and decides if the input is valid.
//! let vf = Verifier::try_from(vf_shares.as_slice()).unwrap();
//! let res = decide(&x_shares[0], &vf).unwrap();
//! assert_eq!(res, true);
//! ```
//!
//! The fully linear PCP system of [BBG+19, Theorem 4.3] applies to languages recognized by
//! arithmetic circuits over finite fields that have a particular structure. Namely, all gates in
//! the circuit are either affine (i.e., addition or scalar multiplication) or invoke a special
//! sub-circuit, called the "gadget", which may contain non-affine operations (i.e.,
//! multiplication). For example, the `Boolean` type uses the `Mul` gadget, an arity-2 circuit that
//! simply multiples its inputs and outputs the result.
//!
//! A concrete system is instantiated by implementing the `Value` trait. This includes specifying
//! the validity circuit, as well as the underlying gadget (the `Gadget` trait).
//!
//! # References
//!
//! - \[GB17\] H. Corrigan-Gibbs and D. Boneh. "[Prio: Private, Robust, and Scalable Computation of
//! Aggregate Statistics.](https://crypto.stanford.edu/prio/paper.pdf)" NSDI 2017.
//! - \[BBG+19\] Boneh et al. "[Zero-Knowledge Proofs on Secret-Shared Data via Fully Linear
//! PCPs.](https://eprint.iacr.org/2019/188)" CRYPTO 2019.

use std::convert::TryFrom;

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish, FftError};
use crate::field::{rand_vec, FieldElement};
use crate::fp::log2;
use crate::polynomial::{poly_deg, poly_eval};

pub mod gadgets;
pub mod types;

/// Errors propagagted by methods in this module.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum PcpError {
    /// The caller of an arithmetic circuit provided the wrong number of inputs. This error may
    /// occur when evaluating a validity circuit or gadget.
    #[error("wrong number of inputs to arithmetic circuit")]
    CircuitInLen,

    /// This error is returned by `collect` if the input slice is empty.
    #[error("collect requires at least one input")]
    CollectInLen,

    /// This error is returned by `collect` if the two or more verifier shares have different
    /// gadget arities.
    #[error("collect inputs have mismatched gadget arity")]
    CollectGadgetInLenMismatch,

    /// Returned if an FFT operation propagates an error.
    #[error("FFT error")]
    Fft(#[from] FftError),

    /// When evaluating a gadget on polynomials, this error is returned if the input polynomials
    /// don't all have the same length.
    #[error("gadget called on polynomials with different lengths")]
    GadgetPolyInLen,

    /// When evaluating a gadget on polynomials, this error is returned if the slice allocated for
    /// the output polynomial is too small.
    #[error("sliice allocated for gadget output is too small")]
    GadgetPolyOutLen,

    /// The proof string is either too large or too short for the given type.
    #[error("attempted query on proof with invalid length")]
    QueryProofLen,
}

/// A value of a certain type. Implementations of this trait specify an arithmetic circuit that
/// determines whether a given value is valid.
pub trait Value<F, G>: Sized
where
    F: FieldElement,
    G: Gadget<F>,
{
    /// Evalauates the validity circuit on the given input (i.e., `self`) and returns the output.
    /// Slice `rand` is the random input consumed by the validity circuit.
    ///
    /// ```
    /// use prio::pcp::types::Boolean;
    /// use prio::pcp::Value;
    /// use prio::field::{FieldElement, Field64};
    ///
    /// type F = Field64;
    /// type T = Boolean<F>;
    ///
    /// let x = T::new(false);
    ///
    /// let mut rand: Vec<F> = Vec::with_capacity(x.valid_rand_len());
    /// for _ in 0..rand.len() {
    ///     rand.push(F::rand());
    /// }
    ///
    /// let v = x.valid(&mut x.gadget(0), &rand).unwrap();
    /// assert_eq!(v, F::zero());
    /// ```
    fn valid(&self, g: &mut dyn GadgetCallOnly<F>, rand: &[F]) -> Result<F, PcpError>;

    /// Returns a reference to the underlying data.
    fn as_slice(&self) -> &[F];

    /// Constructs a value of this type from the given data without checking whether the data is
    /// valid. This method takes ownership of `data`.
    fn new_with(&self, data: Vec<F>) -> Self;

    /// The length of the random input expected by the validity circuit.
    fn valid_rand_len(&self) -> usize;

    /// The number of calls to the gadget made when evaluating the validity circuit.
    fn valid_gadget_calls(&self) -> usize;

    /// Returns an instance of the gadget associated with the validity circuit. `in_len` is the
    /// maximum degree of each of the polynomials passed into `call_poly`. If `call_poly` is never
    /// used by the caller, then it is safe to set `in_len == 0`.
    fn gadget(&self, in_len: usize) -> G;
}

/// The gadget functionality required for evaluating a validity circuit. The `Gadget` trait
/// inherits this trait.
pub trait GadgetCallOnly<F: FieldElement> {
    /// Evaluates the gadget on input `inp` and returns the output.
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError>;

    /// Returns the circuit's arity, i.e., the expected length of the input to `call`.
    fn call_in_len(&self) -> usize;
}

/// The sub-circuit associated with some validity circuit. A gadget is called either on a sequence
/// of finite field elements or a sequence of polynomials over a finite field.
pub trait Gadget<F: FieldElement>: GadgetCallOnly<F> {
    /// Evaluate the gadget on input of a sequence of polynomials. The output is written to `outp`.
    fn call_poly<V: AsRef<[F]>>(&mut self, outp: &mut [F], inp: &[V]) -> Result<(), PcpError>;

    /// The maximum degree of the polynomial output by `call_poly` as a function of the maximum
    /// degree of each input polynomial.
    fn call_poly_out_len(&self, in_len: usize) -> usize;
}

/// Generate a proof of an input's validity.
pub fn prove<F, G, V>(x: &V) -> Result<Proof<F>, PcpError>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
{
    let g_calls = x.valid_gadget_calls();
    let m = (g_calls + 1).next_power_of_two();
    let mut g = x.gadget(m);
    let p = g.call_poly_out_len(m);
    let l = g.call_in_len();
    let mut data = vec![F::zero(); l + p];

    // Run the validity circuit with a "shim" gadget that records the value of each input wire of
    // each gadget evaluation.
    let mut shim = ProveShimGadget::new(&mut g, g_calls);
    let _ = x.valid(&mut shim, &rand_vec(x.valid_rand_len()));

    // Construct the intermediate proof polynomials `f[0], ..., f[l-1]`. Also, record in the slice
    // `data[...l]` the value of `f[i](1)`, i.e., the first point at which polynomial `f[i]` was
    // interpolated.
    let mut f = vec![vec![F::zero(); m]; l];
    let m_inv = F::from(F::Integer::try_from(m).unwrap()).inv();
    for i in 0..l {
        data[i] = shim.f_vals[i][0];
        discrete_fourier_transform(&mut f[i], &shim.f_vals[i], m)?;
        discrete_fourier_transform_inv_finish(&mut f[i], m, m_inv);
    }

    // Construct the proof polynomial `data[l..] = G(f[0], ..., f[l-1])`.
    g.call_poly(&mut data[l..], &f)?;

    let poly_len = poly_deg(&data[l..]);
    data.truncate(l + poly_len + 1);
    Ok(Proof { data })
}

// A "shim" gadget used during proof generation to record the points at which the intermediate
// proof polynomials are interpolated.
struct ProveShimGadget<'a, F, G>
where
    F: FieldElement,
    G: Gadget<F>,
{
    inner: &'a mut G,
    /// Points at which intermediate proof polynomials are interpolated.
    f_vals: Vec<Vec<F>>,
    /// The number of times the gadget has been called so far.
    ct: usize,
}

impl<'a, F, G> ProveShimGadget<'a, F, G>
where
    F: FieldElement,
    G: Gadget<F>,
{
    fn new(inner: &'a mut G, gadget_calls: usize) -> Self {
        let mut f_vals = vec![vec![F::zero(); gadget_calls + 1]; inner.call_in_len()];
        for i in 0..f_vals.len() {
            // Choose a random field element as first point on the i-th proof polynomial.
            f_vals[i][0] = F::rand();
        }
        Self {
            inner,
            f_vals,
            ct: 1,
        }
    }
}

impl<'a, F, G> GadgetCallOnly<F> for ProveShimGadget<'a, F, G>
where
    F: FieldElement,
    G: Gadget<F>,
{
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        for i in 0..inp.len() {
            self.f_vals[i][self.ct] = inp[i];
        }
        self.ct += 1;
        self.inner.call(inp)
    }

    fn call_in_len(&self) -> usize {
        self.inner.call_in_len()
    }
}

/// The output of `prove`, a proof of an input's validity.
#[derive(Clone, Debug)]
pub struct Proof<F: FieldElement> {
    data: Vec<F>,
}

impl<F: FieldElement> Proof<F> {
    /// Returns a reference to the underlying data.
    pub fn as_slice(&self) -> &[F] {
        &self.data
    }
}

impl<F: FieldElement> From<Vec<F>> for Proof<F> {
    fn from(data: Vec<F>) -> Self {
        Self { data }
    }
}

/// Generate the verifier for an input and proof (or the verifier share for an input share and
/// proof share).
pub fn query<F, G, V>(x: &V, pf: &Proof<F>, rand: &[F]) -> Result<Verifier<F>, PcpError>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
{
    let g_calls = x.valid_gadget_calls();
    let m = (g_calls + 1).next_power_of_two();
    let g = x.gadget(m);
    let l = g.call_in_len();
    let r = rand[0];

    // Run the validity circuit with a "shim" gadget that records each input to each gadget.
    // Record the output of the circuit.
    //
    // NOTE The proof of [BBC+19, Theorem 4.3] assumes that the output of the validity circuit is
    // equal to the output of the last gadget evaluation. Here we relax this assumption, This
    // should be ok, since it's possible to transform any circuit into one that for which this is
    // true. (Needs security analysis.)
    let mut shim = QueryShimGadget::new(&g, g_calls, pf)?;
    let v = x.valid(&mut shim, &rand[1..])?;

    // Reconstruct the intermediate proof polynomials `f[0], ..., f[l-1]` and evaluate each
    // polynomial at input `r`.
    let mut f = vec![F::zero(); m];
    let mut f_at_r = vec![F::zero(); l];
    let m_inv = F::from(F::Integer::try_from(m).unwrap()).inv();
    for i in 0..l {
        for j in 0..m {
            f[j] = F::zero();
        }
        discrete_fourier_transform(&mut f, &shim.f_vals[i], m)?;
        discrete_fourier_transform_inv_finish(&mut f, m, m_inv);
        f_at_r[i] = poly_eval(&f, r);
    }

    // Evaluate `p` at `r`.
    //
    // NOTE Usually `r` is sampled uniformly form the field. Technically speaking, [BBC+19, Theorem
    // 4.3] requires that r be sampled from the set of field elements *minus* the roots of unity at
    // which the polynomials are interpolated. This relaxation is fine, but results in a modest
    // loss of concrete security. (Needs security analysis.)
    let p_at_r = poly_eval(&pf.data[l..], r);

    Ok(Verifier { v, p_at_r, f_at_r })
}

// A "shim" gadget used during proof verification to record the points at which the intermediate
// proof polynomials are evaluated.
struct QueryShimGadget<F: FieldElement> {
    /// Points at which intermediate proof polynomials are interpolated.
    f_vals: Vec<Vec<F>>,
    /// Points at which the proof polynomial is interpolated.
    p_vals: Vec<F>,
    /// Used to compute an index into `p_val`.
    step: usize,
    /// The number of times the gadget has been called so far.
    ct: usize,
    /// The arity of the inner gadget.
    l: usize,
}

impl<F: FieldElement> QueryShimGadget<F> {
    fn new<G: Gadget<F>>(inner: &G, g_calls: usize, pf: &Proof<F>) -> Result<Self, PcpError> {
        let m = (g_calls + 1).next_power_of_two();
        let p = inner.call_poly_out_len(m);
        let l = inner.call_in_len();

        if pf.data.len() < l || pf.data.len() > l + p {
            return Err(PcpError::QueryProofLen);
        }

        // Record the intermediate polynomial seeds.
        let mut f_vals = vec![vec![F::zero(); g_calls + 1]; l];
        for i in 0..l {
            f_vals[i][0] = pf.data[i]
        }

        // Evaluate the proof polynomial at roots of unity.
        let size = p.next_power_of_two();
        let mut p_vals = vec![F::zero(); size];
        discrete_fourier_transform(&mut p_vals, &pf.data[l..], size)?;

        let step = (1 << (log2(p as u128) - log2(m as u128))) as usize;
        Ok(QueryShimGadget {
            f_vals,
            p_vals,
            step,
            ct: 1,
            l,
        })
    }
}

impl<F: FieldElement> GadgetCallOnly<F> for QueryShimGadget<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        for i in 0..inp.len() {
            self.f_vals[i][self.ct] = inp[i];
        }
        let outp = self.p_vals[self.ct * self.step];
        self.ct += 1;
        Ok(outp)
    }

    fn call_in_len(&self) -> usize {
        self.l
    }
}

/// The output of `query`, the verifier message generated for a proof.
#[derive(Debug)]
pub struct Verifier<F: FieldElement> {
    /// Output of the validity circuit.
    v: F,
    /// The proof polynomial evluated at `r`.
    p_at_r: F,
    /// The intermediate proof polynomials evaluated at `r`.
    f_at_r: Vec<F>,
}

impl<F: FieldElement> TryFrom<&[Verifier<F>]> for Verifier<F> {
    type Error = PcpError;

    /// Returns the verifier corresponding to a sequence of verifier shares.
    fn try_from(vf_shares: &[Verifier<F>]) -> Result<Verifier<F>, PcpError> {
        if vf_shares.len() == 0 {
            return Err(PcpError::CollectInLen);
        }

        let l = vf_shares[0].f_at_r.len();
        let mut vf = Verifier {
            v: F::zero(),
            p_at_r: F::zero(),
            f_at_r: vec![F::zero(); l],
        };

        for i in 0..vf_shares.len() {
            if vf_shares[i].f_at_r.len() != l {
                return Err(PcpError::CollectGadgetInLenMismatch);
            }

            vf.v += vf_shares[i].v;
            vf.p_at_r += vf_shares[i].p_at_r;
            for j in 0..l {
                vf.f_at_r[j] += vf_shares[i].f_at_r[j];
            }
        }

        Ok(vf)
    }
}

/// Decide if the input (or input share) is valid using the given verifier.
pub fn decide<F, G, V>(x: &V, vf: &Verifier<F>) -> Result<bool, PcpError>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
{
    let e = x.gadget(0).call(&vf.f_at_r)?;
    if e == vf.p_at_r && vf.v == F::zero() {
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{split, Field126};
    use crate::pcp::types::Boolean;

    // Simple integration test for the core PCP logic. You'll find more extensive unit tests for
    // each implemented data type in src/types.rs.
    #[test]
    fn test_pcp() {
        type F = Field126;
        type T = Boolean<F>;

        let x: T = Boolean::new(false);
        let x_shares: Vec<T> = split(x.as_slice(), 2)
            .into_iter()
            .map(|data| x.new_with(data))
            .collect();

        let pf = prove(&x).unwrap();
        let pf_shares: Vec<Proof<F>> = split(pf.as_slice(), 2)
            .into_iter()
            .map(|data| Proof::from(data))
            .collect();

        let rand = rand_vec(1 + x.valid_rand_len());
        let vf_shares = vec![
            query(&x_shares[0], &pf_shares[0], &rand).unwrap(),
            query(&x_shares[1], &pf_shares[1], &rand).unwrap(),
        ];

        let vf = Verifier::try_from(vf_shares.as_slice()).unwrap();
        let res = decide(&x, &vf).unwrap();
        assert_eq!(res, true);
    }
}
