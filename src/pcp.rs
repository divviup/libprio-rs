// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module
//! implements a fully linear PCP ("Probabilistically Checkable Proof") system based on
//! \[[BBC+19](https://eprint.iacr.org/2019/188), Theorem 4.3\].
//!
//! # Overview
//!
//! The proof system is comprised of three algorithms. The first, `prove`, is run by the prover in
//! order to generate a proof of a statement's validity. The second and third, `query` and
//! `decide`, are run by the verifier in order to check the proof. The proof asserts that the input
//! is an element of a language recognized by an arithmetic circuit. For example:
//!
//! ```
//! use prio::pcp::types::Boolean;
//! use prio::pcp::{decide, prove, query, Value};
//! use prio::field::{rand, FieldElement, Field64};
//!
//! // The prover generates a proof `pf` that its input `x` is a valid encoding
//! // of a boolean (either `true` or `false`). Both the input and proof are
//! // vectors over a finite field.
//! let x: Boolean<Field64> = Boolean::new(false);
//!
//! // The verifier chooses "joint randomness" that that will be used to
//! // generate and verify a proof of `x`'s validity. In proof systems like
//! // [BBC+19, Theorem 5.3], the verifier sends the prover a random challenge
//! // in the first round, which the prover uses to construct the proof.
//! let joint_rand = rand(x.valid_rand_len()).unwrap();
//!
//! // The verifier chooses local randomness it uses to check the proof.
//! let query_rand = rand(x.valid_gadget_len()).unwrap();
//!
//! // The prover generates the proof.
//! let pf = prove(&x, &joint_rand).unwrap();
//!
//! // The verifier queries the proof `pf` and input `x`, getting a
//! // "verification message" in response. It uses this message to decide if
//! // the input is valid.
//! let vf = query(&x, &pf, &query_rand, &joint_rand).unwrap();
//! let res = decide(&x, &vf).unwrap();
//! assert_eq!(res, true);
//! ```
//!
//! If an input is _not_ valid, then the verification step will fail with high probability:
//!
//! ```
//! use prio::pcp::types::Boolean;
//! use prio::pcp::{decide, prove, query, Value};
//! use prio::field::{rand, FieldElement, Field64};
//!
//! use std::convert::TryFrom;
//!
//! let x = Boolean::try_from(((), vec![Field64::from(23)])).unwrap(); // Invalid input
//! let joint_rand = rand(x.valid_rand_len()).unwrap();
//! let query_rand = rand(x.valid_gadget_len()).unwrap();
//! let pf = prove(&x, &joint_rand).unwrap();
//! let vf = query(&x, &pf, &query_rand, &joint_rand).unwrap();
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
//! use prio::field::{rand, split, FieldElement, Field64};
//!
//! use std::convert::TryFrom;
//!
//! // The prover encodes its input and splits it into two secret shares. It
//! // sends each share to two aggregators.
//! let x: Boolean<Field64>= Boolean::new(true);
//! let x_par = x.param();
//! let x_shares: Vec<Boolean<Field64>> = split(x.as_slice(), 2)
//!     .unwrap()
//!     .into_iter()
//!     .map(|data| Boolean::try_from((x_par, data)).unwrap())
//!     .collect();
//!
//! let joint_rand = rand(x.valid_rand_len()).unwrap();
//! let query_rand = rand(x.valid_gadget_len()).unwrap();
//!
//! // The prover generates a proof of its input's validity and splits the proof
//! // into two shares. It sends each share to one of two aggregators.
//! let pf = prove(&x, &joint_rand).unwrap();
//! let pf_shares: Vec<Proof<Field64>> = split(pf.as_slice(), 2)
//!     .unwrap()
//!     .into_iter()
//!     .map(Proof::from)
//!     .collect();
//!
//! // Each verifier queries its shares of the input and proof and sends its
//! // share of the verification message to the leader.
//! let vf_shares = vec![
//!     query(&x_shares[0], &pf_shares[0], &query_rand, &joint_rand).unwrap(),
//!     query(&x_shares[1], &pf_shares[1], &query_rand, &joint_rand).unwrap(),
//! ];
//!
//! // The leader collects the verifier shares and decides if the input is valid.
//! let vf = Verifier::try_from(vf_shares.as_slice()).unwrap();
//! let res = decide(&x_shares[0], &vf).unwrap();
//! assert_eq!(res, true);
//! ```
//!
//! The fully linear PCP system of [BBC+19, Theorem 4.3] applies to languages recognized by
//! arithmetic circuits over finite fields that have a particular structure. Namely, all gates in
//! the circuit are either affine (i.e., addition or scalar multiplication) or invoke a special
//! sub-circuit, called the "gadget", which may contain non-affine operations (i.e.,
//! multiplication). For example, the `Boolean` type uses the `Mul` gadget, an arity-2 circuit that
//! simply multiples its inputs and outputs the result.
//!
//! # References
//!
//! - \[GB17\] H. Corrigan-Gibbs and D. Boneh. "[Prio: Private, Robust, and Scalable Computation of
//! Aggregate Statistics.](https://crypto.stanford.edu/prio/paper.pdf)" NSDI 2017.
//! - \[BBC+19\] Boneh et al. "[Zero-Knowledge Proofs on Secret-Shared Data via Fully Linear
//! PCPs.](https://eprint.iacr.org/2019/188)" CRYPTO 2019.

use std::any::Any;
use std::convert::TryFrom;
use std::fmt::Debug;

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish, FftError};
use crate::field::{FieldElement, FieldError};
use crate::fp::log2;
use crate::polynomial::poly_eval;
use crate::prng::Prng;

pub mod gadgets;
pub mod types;

/// Errors propagated by methods in this module.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum PcpError {
    /// The caller of an arithmetic circuit provided the wrong number of inputs. This error may
    /// occur when evaluating a validity circuit or gadget.
    #[error("wrong number of inputs to arithmetic circuit")]
    CircuitInLen,

    /// The caller of an arithmetic circuit provided malformed input.
    #[error("malformed input to circuit")]
    CircuitIn(&'static str),

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
    #[error("slice allocated for gadget output is too small")]
    GadgetPolyOutLen,

    /// Calling `query` returned an error.
    #[error("query error: {0}")]
    Query(&'static str),

    /// Returned by `query` if one of the elements of the query randomness vector is invalid. An
    /// element is invalid if using it to generate the verification message would result in a
    /// privacy violation.
    ///
    /// If this error is returned, the caller may generate fresh randomness and retry.
    #[error("query error: invalid query randomness")]
    QueryRandInvalid,

    /// Calling `decide` returned an error.
    #[error("decide error: {0}")]
    Decide(&'static str),

    /// The validity circuit was called with the wrong amount of randomness.
    #[error("incorrect amount of randomness")]
    ValidRandLen,

    /// Encountered an error while evaluating a validity circuit.
    #[error("failed to run validity circuit: {0}")]
    Valid(&'static str),

    /// Returned if a field operation encountered an error.
    #[error("Field error")]
    Field(#[from] FieldError),

    /// Failure when calling getrandom().
    #[error("getrandom: {0}")]
    GetRandom(#[from] getrandom::Error),
}

/// A value of a certain type. Implementations of this trait specify an arithmetic circuit that
/// determines whether a given value is valid.
pub trait Value<F: FieldElement>:
    Sized
    + PartialEq
    + Eq
    + Debug
    + TryFrom<(<Self as Value<F>>::Param, Vec<F>), Error = <Self as Value<F>>::TryFromError>
{
    /// Parameters used to construct a value of this type from a vector of field elements.
    type Param;

    /// Error returned when converting a `(Param, Vec<F>)` to a `Value<F>` fails.
    type TryFromError: Debug;

    /// Evaluates the validity circuit on the given input (i.e., `self`) and returns the output.
    /// `joint_rand` is the joint randomness shared by the prover and verifier. `g` is the sequence
    /// of gadgets called by the circuit.
    ///
    /// ```
    /// use prio::pcp::types::Boolean;
    /// use prio::pcp::Value;
    /// use prio::field::{rand, FieldElement, Field64};
    ///
    /// let x: Boolean<Field64> = Boolean::new(false);
    /// let joint_rand = rand(x.valid_rand_len()).unwrap();
    /// let v = x.valid(&mut x.gadget(), &joint_rand).unwrap();
    /// assert_eq!(v, Field64::zero());
    /// ```
    fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, joint_rand: &[F]) -> Result<F, PcpError>;

    /// Returns a reference to the underlying data.
    fn as_slice(&self) -> &[F];

    /// The length of the random input expected by the validity circuit.
    fn valid_rand_len(&self) -> usize;

    /// The number of gadgets expected by the validity circuit.
    fn valid_gadget_len(&self) -> usize;

    /// The number of calls to the gadget made when evaluating the validity circuit.
    ///
    /// TODO(cjpatton) Consider consolidating this and `gadget` into one call. The benefit would be
    /// that there is one less thing to worry about when implementing a Value<F>. We would need to
    /// extend Gadget<F> so that it tells you how many times it gets called.
    fn valid_gadget_calls(&self) -> Vec<usize>;

    /// Returns the sequence of gadgets associated with the validity circuit.
    ///
    /// NOTE The construction of [BBC+19, Theorem 4.3] uses a single gadget rather than many. The
    /// idea to generalize the proof system to allow multiple gadgets is discussed briefly in
    /// [BBC+19, Remark 4.5], but no construction is given. The construction implemented here
    /// requires security analysis.
    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>>;

    /// Returns a copy of the associated type parameters for this value.
    fn param(&self) -> Self::Param;

    /// When verifying a proof over secret shared data, this method may be used to distinguish the
    /// "leader" share from the others. This is useful, for example, when some of the gadget inputs
    /// are constants used for both proof generation and verification.
    ///
    /// ```
    /// use prio::pcp::types::MeanVarUnsignedVector;
    /// use prio::pcp::{decide, prove, query, Value, Proof, Verifier};
    /// use prio::field::{split, rand, FieldElement, Field64};
    ///
    /// use std::convert::TryFrom;
    ///
    /// let measurement = [1, 2, 3];
    /// let bits = 8;
    /// let x: MeanVarUnsignedVector<Field64> =
    ///     MeanVarUnsignedVector::new(bits, &measurement).unwrap();
    /// let x_shares: Vec<MeanVarUnsignedVector<Field64>> = split(x.as_slice(), 2)
    ///     .unwrap()
    ///     .into_iter()
    ///     .enumerate()
    ///     .map(|(i, data)| {
    ///         let mut share =
    ///             MeanVarUnsignedVector::try_from((x.param(), data)).unwrap();
    ///         share.set_leader(i == 0);
    ///         share
    ///     })
    ///     .collect();
    ///
    /// let joint_rand = rand(x.valid_rand_len()).unwrap();
    /// let query_rand = rand(x.valid_gadget_len()).unwrap();
    ///
    /// let pf = prove(&x, &joint_rand).unwrap();
    /// let pf_shares: Vec<Proof<Field64>> = split(pf.as_slice(), 2)
    ///     .unwrap()
    ///     .into_iter()
    ///     .map(Proof::from)
    ///     .collect();
    ///
    /// let vf_shares = vec![
    ///     query(&x_shares[0], &pf_shares[0], &query_rand, &joint_rand).unwrap(),
    ///     query(&x_shares[1], &pf_shares[1], &query_rand, &joint_rand).unwrap(),
    /// ];
    ///
    /// let vf = Verifier::try_from(vf_shares.as_slice()).unwrap();
    /// let res = decide(&x_shares[0], &vf).unwrap();
    /// assert_eq!(res, true);
    /// ```
    fn set_leader(&mut self, _is_leader: bool) {
        // No-op by default.
    }
}

/// A gadget, a non-affine arithmetic circuit that is called when evaluating a validity circuit.
///
/// TODO(cjpatton) Consider extending this API with a `Param` associated type and have it implement
/// a constructor from an instance of `Param` and the number of times the gadget gets called.
pub trait Gadget<F: FieldElement> {
    /// Evaluates the gadget on input `inp` and returns the output.
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError>;

    /// Evaluate the gadget on input of a sequence of polynomials. The output is written to `outp`.
    fn call_poly(&mut self, outp: &mut [F], inp: &Vec<Vec<F>>) -> Result<(), PcpError>;

    /// Returns the arity of the gadget. This is the length of `inp` passed to `call` or
    /// `call_poly`.
    fn arity(&self) -> usize;

    /// Returns the circuit's arithmetic degree. This determines the minimum length the `outp`
    /// buffer passed to `call_poly`.
    fn deg(&self) -> usize;

    /// This call is used to downcast a `Box<dyn Gadget<F>>` to a concrete type.
    fn as_any(&mut self) -> &mut dyn Any;
}

/// Generate a proof of an input's validity.
pub fn prove<F, V>(x: &V, joint_rand: &[F]) -> Result<Proof<F>, PcpError>
where
    F: FieldElement,
    V: Value<F>,
{
    let g_calls = x.valid_gadget_calls();

    let mut shim = x
        .gadget()
        .into_iter()
        .enumerate()
        .map(|(idx, g)| ProveShimGadget::new(g, g_calls[idx]))
        .collect::<Result<Vec<_>, _>>()?;

    // Create a buffer for storing the proof. The buffer is longer than the proof itself; the extra
    // length is to accommodate the computation of each gadget polynomial.
    let data_len = (0..shim.len())
        .map(|idx| shim[idx].arity() + shim[idx].deg() * (1 + g_calls[idx]).next_power_of_two())
        .sum();
    let mut data = vec![F::zero(); data_len];

    // Run the validity circuit with a sequence of "shim" gadgets that record the value of each
    // input wire of each gadget evaluation. These values are used to construct the wire
    // polynomials for each gadget in the next step.
    let _ = x.valid(&mut shim, joint_rand);

    // Fill the buffer with the proof. `proof_len` keeps track of the amount of data written to the
    // buffer so far.
    let mut proof_len = 0;
    for idx in 0..shim.len() {
        let g = shim[idx]
            .as_any()
            .downcast_mut::<ProveShimGadget<F>>()
            .unwrap();

        let g_deg = g.deg();
        let g_arity = g.arity();

        // Interpolate the wire polynomials `f[0], ..., f[g_arity-1]` from the input wires of each
        // evaluation of the gadget.
        let m = (1 + g_calls[idx]).next_power_of_two();
        let m_inv = F::from(F::Integer::try_from(m).unwrap()).inv();
        let mut f = vec![vec![F::zero(); m]; g_arity];
        for wire in 0..g_arity {
            discrete_fourier_transform(&mut f[wire], &g.f_vals[wire], m)?;
            discrete_fourier_transform_inv_finish(&mut f[wire], m, m_inv);

            // The first point on each wire polynomial is a random value chosen by the prover. This
            // point is stored in the proof so that the verifier can reconstruct the wire
            // polynomials.
            data[proof_len + wire] = g.f_vals[wire][0];
        }

        // Construct the gadget polynomial `G(f[0], ..., f[g_arity-1])` and append it to `data`.
        g.call_poly(&mut data[proof_len + g_arity..], &f)?;
        proof_len += g_arity + g_deg * (m - 1) + 1;
    }

    // Truncate the buffer to the size of the proof.
    data.truncate(proof_len);
    Ok(Proof { data })
}

// A "shim" gadget used during proof generation to record the input wires each time a gadget is
// evaluated.
struct ProveShimGadget<F: FieldElement> {
    inner: Box<dyn Gadget<F>>,

    /// Points at which the wire polynomials are interpolated.
    f_vals: Vec<Vec<F>>,

    /// The number of times the gadget has been called so far.
    ct: usize,
}

impl<F: FieldElement> ProveShimGadget<F> {
    fn new(
        inner: Box<dyn Gadget<F>>,
        g_calls: usize,
    ) -> Result<Box<dyn Gadget<F>>, getrandom::Error> {
        let mut f_vals = vec![vec![F::zero(); 1 + g_calls]; inner.arity()];
        let mut prng = Prng::new_with_length(f_vals.len())?;

        for wire in 0..f_vals.len() {
            // Choose a random field element as the first point on the wire polynomial.
            f_vals[wire][0] = prng.next().unwrap();
        }

        Ok(Box::new(Self {
            inner,
            f_vals,
            ct: 1,
        }))
    }
}

impl<F: FieldElement> Gadget<F> for ProveShimGadget<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        for wire in 0..inp.len() {
            self.f_vals[wire][self.ct] = inp[wire];
        }
        self.ct += 1;
        self.inner.call(inp)
    }

    fn call_poly(&mut self, outp: &mut [F], inp: &Vec<Vec<F>>) -> Result<(), PcpError> {
        self.inner.call_poly(outp, inp)
    }

    fn arity(&self) -> usize {
        self.inner.arity()
    }

    fn deg(&self) -> usize {
        self.inner.deg()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
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

/// Generate a verifier message for an input and proof (or the verifier share for an input share
/// and proof share).
///
/// Parameters:
/// * `x` is the input.
/// * `pf` is the proof.
/// * `query_rand` is the verifier's randomness.
/// * `joint_rand` is the randomness shared by the prover and verifier.
pub fn query<F, V>(
    x: &V,
    pf: &Proof<F>,
    query_rand: &[F],
    joint_rand: &[F],
) -> Result<Verifier<F>, PcpError>
where
    F: FieldElement,
    V: Value<F>,
{
    let g_calls = x.valid_gadget_calls();

    let mut proof_len = 0;
    let mut shim = x
        .gadget()
        .into_iter()
        .enumerate()
        .map(|(idx, g)| {
            if idx >= query_rand.len() {
                return Err(PcpError::Query("short query randomness"));
            }

            let g_deg = g.deg();
            let g_arity = g.arity();
            let m = (1 + g_calls[idx]).next_power_of_two();
            let r = query_rand[idx];

            // Make sure the query randomness isn't a root of unity. Evaluating the gadget
            // polynomial at any of these points would be a privacy violation, since these points
            // were used by the prover to construct the wire polynomials.
            if r.pow(F::Integer::try_from(m).unwrap()) == F::one() {
                return Err(PcpError::QueryRandInvalid);
            }

            // Compute the length of the sub-proof corresponding to the `idx`-th gadget.
            let next_len = g_arity + g_deg * (m - 1) + 1;
            if proof_len + next_len > pf.data.len() {
                return Err(PcpError::Query("short proof"));
            }

            let proof_data = &pf.data[proof_len..proof_len + next_len];
            proof_len += next_len;

            QueryShimGadget::new(g, r, proof_data, g_calls[idx])
        })
        .collect::<Result<Vec<_>, _>>()?;

    if proof_len < pf.data.len() {
        return Err(PcpError::Query("long proof"));
    }

    if query_rand.len() > shim.len() {
        return Err(PcpError::Query("long joint randomness"));
    }

    // Create a buffer for the verifier data. This includes the output of the validity circuit and,
    // for each gadget `shim[idx].inner`, the wire polynomials evaluated at the query randomness
    // `query_rand[idx]` and the gadget polynomial evaluated at `query_rand[idx]`.
    let data_len = 1
        + (0..shim.len())
            .map(|idx| shim[idx].arity() + 1)
            .sum::<usize>();
    let mut data = Vec::with_capacity(data_len);

    // Run the validity circuit with a sequence of "shim" gadgets that record the inputs to each
    // wire for each gadget call. Record the output of the circuit and append it to the verifier
    // message.
    //
    // NOTE The proof of [BBC+19, Theorem 4.3] assumes that the output of the validity circuit is
    // equal to the output of the last gadget evaluation. Here we relax this assumption. This
    // should be OK, since it's possible to transform any circuit into one for which this is true.
    // (Needs security analysis.)
    let v = x.valid(&mut shim, joint_rand)?;
    data.push(v);

    // Fill the buffer with the verifier message.
    for idx in 0..shim.len() {
        let r = query_rand[idx];
        let g = shim[idx]
            .as_any()
            .downcast_ref::<QueryShimGadget<F>>()
            .unwrap();

        // Reconstruct the wire polynomials `f[0], ..., f[g_arity-1]` and evaluate each wire
        // polynomial at query randomness `r`.
        let m = (1 + g_calls[idx]).next_power_of_two();
        let m_inv = F::from(F::Integer::try_from(m).unwrap()).inv();
        let mut f = vec![F::zero(); m];
        for wire in 0..g.arity() {
            discrete_fourier_transform(&mut f, &g.f_vals[wire], m)?;
            discrete_fourier_transform_inv_finish(&mut f, m, m_inv);
            data.push(poly_eval(&f, r));
        }

        // Add the value of the gadget polynomial evaluated at `r`.
        data.push(g.p_at_r);
    }

    Ok(Verifier { data })
}

// A "shim" gadget used during proof verification to record the points at which the intermediate
// proof polynomials are evaluated.
struct QueryShimGadget<F: FieldElement> {
    inner: Box<dyn Gadget<F>>,

    /// Points at which intermediate proof polynomials are interpolated.
    f_vals: Vec<Vec<F>>,

    /// Points at which the gadget polynomial is interpolated.
    p_vals: Vec<F>,

    /// The gadget polynomial evaluated on a random input `r`.
    p_at_r: F,

    /// Used to compute an index into `p_val`.
    step: usize,

    /// The number of times the gadget has been called so far.
    ct: usize,
}

impl<F: FieldElement> QueryShimGadget<F> {
    fn new(
        inner: Box<dyn Gadget<F>>,
        r: F,
        proof_data: &[F],
        g_calls: usize,
    ) -> Result<Box<dyn Gadget<F>>, PcpError> {
        let g_deg = inner.deg();
        let g_arity = inner.arity();
        let m = (1 + g_calls).next_power_of_two();
        let p = m * g_deg;

        // Each call to this gadget records the values at which intermediate proof polynomials were
        // interpolated. The first point was a random value chosen by the prover and transmitted in
        // the proof.
        let mut f_vals = vec![vec![F::zero(); 1 + g_calls]; g_arity];
        for wire in 0..g_arity {
            f_vals[wire][0] = proof_data[wire];
        }

        // Evaluate the gadget polynomial at roots of unity.
        let size = p.next_power_of_two();
        let mut p_vals = vec![F::zero(); size];
        discrete_fourier_transform(&mut p_vals, &proof_data[g_arity..], size)?;

        // The step is used to compute the element of `p_val` that will be returned by a call to
        // the gadget.
        let step = (1 << (log2(p as u128) - log2(m as u128))) as usize;

        // Evaluate the gadget polynomial `p` at query randomness `r`.
        let p_at_r = poly_eval(&proof_data[g_arity..], r);

        Ok(Box::new(QueryShimGadget {
            inner,
            f_vals,
            p_vals,
            p_at_r,
            step,
            ct: 1,
        }))
    }
}

impl<F: FieldElement> Gadget<F> for QueryShimGadget<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        for wire in 0..inp.len() {
            self.f_vals[wire][self.ct] = inp[wire];
        }
        let outp = self.p_vals[self.ct * self.step];
        self.ct += 1;
        Ok(outp)
    }

    fn call_poly(&mut self, _outp: &mut [F], _inp: &Vec<Vec<F>>) -> Result<(), PcpError> {
        panic!("no-op");
    }

    fn arity(&self) -> usize {
        self.inner.arity()
    }

    fn deg(&self) -> usize {
        self.inner.deg()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// The output of `query`, the verifier message generated for a proof.
#[derive(Debug)]
pub struct Verifier<F: FieldElement> {
    data: Vec<F>,
}

impl<F: FieldElement> Verifier<F> {
    /// Returns a reference to the underlying data. The first element of the output is the output
    /// of the validity circuit. The remainder is a sequence of chunks, where the `idx`-th chunk
    /// corresponds to the `idx`-th gadget for the validity circuit. The last element of a chunk is
    /// the gadget polynomial evaluated on a random input `r`; the rest are the intermediate proof
    /// polynomials evaluated at `r`.
    pub fn as_slice(&self) -> &[F] {
        &self.data
    }
}

impl<F: FieldElement> From<Vec<F>> for Verifier<F> {
    fn from(data: Vec<F>) -> Self {
        Self { data }
    }
}

impl<F: FieldElement> TryFrom<&[Verifier<F>]> for Verifier<F> {
    type Error = PcpError;

    /// Returns the verifier corresponding to a sequence of verifier shares.
    fn try_from(vf_shares: &[Verifier<F>]) -> Result<Verifier<F>, PcpError> {
        if vf_shares.len() == 0 {
            return Err(PcpError::CollectInLen);
        }

        let mut vf = Verifier {
            data: vec![F::zero(); vf_shares[0].data.len()],
        };

        for i in 0..vf_shares.len() {
            if vf_shares[i].data.len() != vf.data.len() {
                return Err(PcpError::CollectGadgetInLenMismatch);
            }

            for j in 0..vf.data.len() {
                vf.data[j] += vf_shares[i].data[j];
            }
        }

        Ok(vf)
    }
}

/// Decide if the input (or input share) is valid using the given verifier.
pub fn decide<F, V>(x: &V, vf: &Verifier<F>) -> Result<bool, PcpError>
where
    F: FieldElement,
    V: Value<F>,
{
    let mut g = x.gadget();

    if vf.data.len() == 0 {
        return Err(PcpError::Decide("zero-length verifier"));
    }

    // Check if the output of the circuit is 0.
    if vf.data[0] != F::zero() {
        return Ok(false);
    }

    // Check that each of the proof polynomials are well-formed.
    let mut verifier_len = 1;
    for idx in 0..g.len() {
        let next_len = 1 + g[idx].arity();
        if verifier_len + next_len > vf.data.len() {
            return Err(PcpError::Decide("short verifier"));
        }

        let e = g[idx].call(&vf.data[verifier_len..verifier_len + next_len - 1])?;
        if e != vf.data[verifier_len + next_len - 1] {
            return Ok(false);
        }

        verifier_len += next_len;
    }

    if verifier_len != vf.data.len() {
        return Err(PcpError::Decide("long verifier"));
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{rand, split, Field126};
    use crate::pcp::gadgets::{Mul, PolyEval};
    use crate::pcp::types::Boolean;
    use crate::polynomial::poly_range_check;

    use std::convert::Infallible;

    // Simple integration test for the core PCP logic. You'll find more extensive unit tests for
    // each implemented data type in src/types.rs.
    #[test]
    fn test_pcp() {
        type F = Field126;
        type T = TestValue<F>;
        const NUM_SHARES: usize = 2;

        let inp = F::from(3);
        let x: T = TestValue::new(inp);
        let x_par = x.param();
        let x_shares: Vec<T> = split(x.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, data)| {
                let mut share = T::try_from((x_par, data)).unwrap();
                share.set_leader(i == 0);
                share
            })
            .collect();

        let joint_rand = rand(x.valid_rand_len()).unwrap();
        let pf = prove(&x, &joint_rand).unwrap();
        let pf_shares: Vec<Proof<F>> = split(pf.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .map(Proof::from)
            .collect();

        let query_rand = rand(2).unwrap(); // Length is the same as the length of the gadget
        let vf_shares: Vec<Verifier<F>> = (0..NUM_SHARES)
            .map(|i| query(&x_shares[i], &pf_shares[i], &query_rand, &joint_rand).unwrap())
            .collect();
        let vf = Verifier::try_from(vf_shares.as_slice()).unwrap();
        let res = decide(&x, &vf).unwrap();
        assert_eq!(res, true, "{:?}", vf);
    }

    #[test]
    fn test_decide() {
        let query_rand = rand(1).unwrap();
        let joint_rand = vec![];
        let x: Boolean<Field126> = Boolean::new(true);

        let ok_vf = query(
            &x,
            &prove(&x, &joint_rand).unwrap(),
            &query_rand,
            &joint_rand,
        )
        .unwrap();
        assert!(decide(&x, &ok_vf).is_ok());

        let vf_len = ok_vf.as_slice().len();

        let bad_vf = Verifier::from(ok_vf.as_slice()[..vf_len - 1].to_vec());
        assert!(decide(&x, &bad_vf).is_err());

        let bad_vf = Verifier::from(ok_vf.as_slice()[..2].to_vec());
        assert!(decide(&x, &bad_vf).is_err());

        let bad_vf = Verifier::from(vec![]);
        assert!(decide(&x, &bad_vf).is_err());
    }

    /// A toy type used for testing the functionality in this module. Valid inputs of this type
    /// consist of a pair of field elements `(x, y)` where `2 <= x < 5` and `x^3 == y`.
    #[derive(Debug, PartialEq, Eq)]
    pub struct TestValue<F: FieldElement> {
        data: Vec<F>, // The encoded input
    }

    impl<F: FieldElement> TestValue<F> {
        pub fn new(inp: F) -> Self {
            Self {
                data: vec![inp, inp * inp * inp],
            }
        }
    }

    impl<F: FieldElement> Value<F> for TestValue<F> {
        type Param = ();
        type TryFromError = Infallible;

        fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, joint_rand: &[F]) -> Result<F, PcpError> {
            if joint_rand.len() != self.valid_rand_len() {
                return Err(PcpError::ValidRandLen);
            }

            if self.data.len() != 2 {
                return Err(PcpError::CircuitInLen);
            }

            let r = joint_rand[0];
            let mut res = F::zero();

            // Check that `data[0]^3 == data[1]`.
            let mut inp = [self.data[0], self.data[0]];
            inp[0] = g[0].call(&inp)?;
            inp[0] = g[0].call(&inp)?;
            let x3_diff = inp[0] - self.data[1];
            res += r * x3_diff;

            // Check that `data[0]` is in the correct range.
            let x_checked = g[1].call(&[self.data[0]])?;
            res += (r * r) * x_checked;

            Ok(res)
        }

        fn valid_gadget_calls(&self) -> Vec<usize> {
            vec![2, 1]
        }

        fn valid_rand_len(&self) -> usize {
            1
        }

        fn valid_gadget_len(&self) -> usize {
            2
        }

        fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
            vec![
                Box::new(Mul::new(2)),
                Box::new(PolyEval::new(poly_range_check(2, 5), 1)),
            ]
        }

        fn as_slice(&self) -> &[F] {
            &self.data
        }

        fn param(&self) -> Self::Param {
            ()
        }
    }

    impl<F: FieldElement> TryFrom<((), Vec<F>)> for TestValue<F> {
        type Error = Infallible;

        fn try_from(val: ((), Vec<F>)) -> Result<Self, Infallible> {
            Ok(Self { data: val.1 })
        }
    }
}
