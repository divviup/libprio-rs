// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This
//! module implements the fully linear PCP ("Probabilistically Checkable Proof") system described
//! in \[[BBC+19](https://eprint.iacr.org/2019/188), Theorem 4.3\].
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
//! use prio::field::{rand, FieldElement, Field64};
//!
//! // The randomness shared by the prover and verifier. In proof systems like [BCG+19, Theorem
//! // 5.3], the verifier sends the prover a random challenge in the first round, which the
//! // prover uses to construct the proof. (This is not used for this example, so `joint_rand'
//! // is empty.)
//! let joint_rand = [];
//!
//! // The randomness used by the verifier.
//! let query_rand = rand(1).unwrap();
//!
//! // The prover generates a proof pf that its input x is a valid encoding
//! // of a boolean (either "true" or "false"). Both the input and proof are
//! // vectors over the finite field specified by Field64.
//! let x: Boolean<Field64> = Boolean::new(false);
//! let pf = prove(&x, &joint_rand).unwrap();
//!
//! // The verifier "queries" the proof pf and input x, getting a "verification
//! // message" in response. It uses this message to decide if the input is
//! // valid.
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
//! let joint_rand = [];
//! let query_rand = rand(1).unwrap();
//! let x = Boolean::from(Field64::from(23));
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
//! let joint_rand = [];
//! let query_rand = rand(1).unwrap();
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
//! A concrete system is instantiated by implementing the `Value` trait. This includes specifying
//! the validity circuit, as well as the underlying gadget (the `Gadget` trait).
//!
//! # References
//!
//! - \[GB17\] H. Corrigan-Gibbs and D. Boneh. "[Prio: Private, Robust, and Scalable Computation of
//! Aggregate Statistics.](https://crypto.stanford.edu/prio/paper.pdf)" NSDI 2017.
//! - \[BBC+19\] Boneh et al. "[Zero-Knowledge Proofs on Secret-Shared Data via Fully Linear
//! PCPs.](https://eprint.iacr.org/2019/188)" CRYPTO 2019.

use std::convert::TryFrom;
use std::fmt::Debug;

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish, FftError};
use crate::field::{FieldElement, FieldError};
use crate::fp::log2;
use crate::polynomial::poly_eval;
use crate::prng::Prng;

pub mod gadgets;
pub mod types;

/// Errors propagagted by methods in this module.
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
pub trait Value<F, G>:
    Sized
    + PartialEq
    + Eq
    + Debug
    + TryFrom<(<Self as Value<F, G>>::Param, Vec<F>), Error = <Self as Value<F, G>>::TryFromError>
where
    F: FieldElement,
    G: Gadget<F>,
{
    /// Parameters used to construct a value of this type from a vector of field elements.
    type Param;

    /// Error returned when converting a `(Vec<F>, Param)` to a `Value<F, G>` fails.
    type TryFromError: Debug;

    /// Evaluates the validity circuit on the given input (i.e., `self`) and returns the output.
    /// Slice `rand` is the random input consumed by the validity circuit.
    ///
    /// ```
    /// use prio::pcp::types::Boolean;
    /// use prio::pcp::Value;
    /// use prio::field::{rand, FieldElement, Field64};
    ///
    /// type F = Field64;
    /// type T = Boolean<F>;
    ///
    /// let x = T::new(false);
    ///
    /// let joint_rand = rand(x.valid_rand_len()).unwrap();
    /// let v = x.valid(&mut x.gadget(), &joint_rand).unwrap();
    /// assert_eq!(v, F::zero());
    /// ```
    fn valid(&self, g: &mut dyn GadgetCallOnly<F>, joint_rand: &[F]) -> Result<F, PcpError>;

    /// Returns a reference to the underlying data.
    fn as_slice(&self) -> &[F];

    /// The length of the random input expected by the validity circuit.
    fn valid_rand_len(&self) -> usize;

    /// The number of calls to the gadget made when evaluating the validity circuit.
    fn valid_gadget_calls(&self, idx: usize) -> usize;

    /// Returns an instance of the gadget associated with the validity circuit. `in_len` is the
    /// maximum degree of each of the polynomials passed into `call_poly`. If `call_poly` is never
    /// used by the caller, then it is safe to set `in_len == 0`.
    fn gadget(&self) -> G;

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
    ///
    /// let joint_rand = rand(3).unwrap();
    /// let query_rand = rand(1).unwrap();
    ///
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

/// The gadget functionality required for evaluating a validity circuit. The `Gadget` trait
/// inherits this trait.
pub trait GadgetCallOnly<F: FieldElement> {
    /// Evaluates the gadget at index `idx` on input `inp` and returns the output.
    fn call(&mut self, idx: usize, inp: &[F]) -> Result<F, PcpError>;

    /// Returns the arity of the gadget at index `idx`, i.e., the expected length of the input to
    /// the corresponding call to `call`.
    fn arity(&self, idx: usize) -> usize;

    /// XXX
    fn len(&self) -> usize;

    /// XXX
    fn deg(&self, idx: usize) -> usize;
}

/// The sub-circuit associated with some validity circuit. A gadget is called either on a sequence
/// of finite field elements or a sequence of polynomials over a finite field.
pub trait Gadget<F: FieldElement>: GadgetCallOnly<F> {
    /// Evaluate the gadget at index `idx` on input of a sequence of polynomials. The output is
    /// written to `outp`.
    fn call_poly<V: AsRef<[F]>>(
        &mut self,
        idx: usize,
        outp: &mut [F],
        inp: &[V],
    ) -> Result<(), PcpError>;
}

/// Generate a proof of an input's validity.
pub fn prove<F, G, V>(x: &V, joint_rand: &[F]) -> Result<Proof<F>, PcpError>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
{
    // Run the validity circuit with a "shim" gadget that records the value of each input wire of
    // each gadget evaluation.
    let mut g = x.gadget();
    let mut shim = ProveShimGadget::new(&mut g, x)?;
    let _ = x.valid(&mut shim, joint_rand);

    let data_len = (0..shim.len())
        .map(|idx| {
            shim.arity(idx) + shim.deg(idx) * (x.valid_gadget_calls(idx) + 1).next_power_of_two()
        })
        .reduce(|a, b| a + b)
        .unwrap();
    let mut data = vec![F::zero(); data_len];

    let mut proof_len = 0;
    for idx in 0..shim.len() {
        let g_deg = shim.deg(idx);
        let g_arity = shim.arity(idx);
        let g_calls = x.valid_gadget_calls(idx);

        // Construct the intermediate proof polynomials `f[0], ..., f[g_arity-1]`. Also, append
        // the value of `f[j](1)` for each `0 <= j < g_arity[i]` to `data`.
        let m = (g_calls + 1).next_power_of_two();
        let mut f = vec![vec![F::zero(); m]; g_arity];
        let m_inv = F::from(F::Integer::try_from(m).unwrap()).inv();
        for wire in 0..g_arity {
            data[proof_len + wire] = shim.f_vals[idx][wire][0];
            discrete_fourier_transform(&mut f[wire], &shim.f_vals[idx][wire], m)?;
            discrete_fourier_transform_inv_finish(&mut f[wire], m, m_inv);
        }

        // Construct the proof polynomial G(f[0], ..., f[g_arity-1])` and append it to `data`.
        shim.inner
            .call_poly(idx, &mut data[proof_len + g_arity..], &f)?;
        proof_len += g_arity + g_deg * (m - 1) + 1;
    }

    data.truncate(proof_len);
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
    f_vals: Vec<Vec<Vec<F>>>,

    /// The number of times each subgadget has been called so far.
    ct: Vec<usize>,
}

impl<'a, F, G> ProveShimGadget<'a, F, G>
where
    F: FieldElement,
    G: Gadget<F>,
{
    fn new<V: Value<F, G>>(inner: &'a mut G, x: &V) -> Result<Self, getrandom::Error> {
        let mut f_vals: Vec<Vec<Vec<F>>> = (0..inner.len())
            .map(|idx| vec![vec![F::zero(); x.valid_gadget_calls(idx) + 1]; inner.arity(idx)])
            .collect();

        let ct = vec![1; inner.len()];

        let mut prng = Prng::new()?;
        for idx in 0..f_vals.len() {
            for wire in 0..f_vals[idx].len() {
                // Choose a random field element as first point on the proof polynomial.
                f_vals[idx][wire][0] = prng.next().unwrap();
            }
        }

        Ok(Self { inner, f_vals, ct })
    }
}

impl<'a, F, G> GadgetCallOnly<F> for ProveShimGadget<'a, F, G>
where
    F: FieldElement,
    G: Gadget<F>,
{
    fn call(&mut self, idx: usize, inp: &[F]) -> Result<F, PcpError> {
        for wire in 0..inp.len() {
            self.f_vals[idx][wire][self.ct[idx]] = inp[wire];
        }
        self.ct[idx] += 1;
        self.inner.call(idx, inp)
    }

    fn arity(&self, idx: usize) -> usize {
        self.inner.arity(idx)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn deg(&self, idx: usize) -> usize {
        self.inner.deg(idx)
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
///
/// Parameters:
/// * `x` is the input.
/// * `pf` is the proof.
/// * `query_rand` is the verifier's randomness.
/// * `joint_rand` is the randomness shared by the prover and verifier.
pub fn query<F, G, V>(
    x: &V,
    pf: &Proof<F>,
    query_rand: &[F],
    joint_rand: &[F],
) -> Result<Verifier<F>, PcpError>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
{
    // Run the validity circuit with a "shim" gadget that records each input to each gadget.
    //
    // NOTE The proof of [BBC+19, Theorem 4.3] assumes that the output of the validity circuit is
    // equal to the output of the last gadget evaluation. Here we relax this assumption. This
    // should be OK, since it's possible to transform any circuit into one for which this is true.
    // (Needs security analysis.)
    let mut g = x.gadget();
    let mut shim = QueryShimGadget::new(&mut g, x, pf)?;
    let v = x.valid(&mut shim, joint_rand)?;

    if query_rand.len() != shim.len() {
        return Err(PcpError::Query("incorrect length of query randomness"));
    }

    // Allocate space for the verifier data. This includes the output of the validity circuit, the
    // intermediate polynomials evaluated at the query randomness `r`, and the proof polynomial
    // evaluated at `r`.
    let data_len = (0..shim.len())
        .map(|idx| shim.arity(idx) + 2)
        .reduce(|a, b| a + b)
        .unwrap();
    let mut data = Vec::with_capacity(data_len);

    // Record the output of the circuit.
    data.push(v);

    let mut proof_len = 0;
    for idx in 0..shim.len() {
        let g_deg = shim.deg(idx);
        let g_arity = shim.arity(idx);
        let g_calls = x.valid_gadget_calls(idx);

        // Reconstruct the intermediate proof polynomials `f[0], ..., f[l-1]` and evaluate each
        // polynomial at input `r`.
        let m = (g_calls + 1).next_power_of_two();
        let m_inv = F::from(F::Integer::try_from(m).unwrap()).inv();
        let mut f = vec![F::zero(); m];
        for wire in 0..g_arity {
            discrete_fourier_transform(&mut f, &shim.f_vals[idx][wire], m)?;
            discrete_fourier_transform_inv_finish(&mut f, m, m_inv);
            data.push(poly_eval(&f, query_rand[idx]));
        }

        // Evaluate the proof polynomial `p` at `r`.
        //
        // NOTE Usually `r` is sampled uniformly from the field. Strictly speaking, [BBC+19, Theorem
        // 4.3] requires that `r` be sampled from the set of field elements *minus* the roots of unity
        // at which the polynomials are interpolated. This relaxation is fine, but results in a
        // modest loss of concrete security. (Needs security analysis.)
        let next_len = g_arity + g_deg * (m - 1) + 1;
        data.push(poly_eval(
            &pf.data[proof_len + g_arity..proof_len + next_len],
            query_rand[idx],
        ));
        proof_len += next_len;
    }

    Ok(Verifier { data })
}

// A "shim" gadget used during proof verification to record the points at which the intermediate
// proof polynomials are evaluated.
struct QueryShimGadget<'a, F, G>
where
    F: FieldElement,
    G: Gadget<F>,
{
    inner: &'a mut G,

    /// Points at which intermediate proof polynomials are interpolated.
    f_vals: Vec<Vec<Vec<F>>>,

    /// Points at which the proof polynomial is interpolated.
    p_vals: Vec<Vec<F>>,

    /// Used to compute an index into `p_val`.
    step: Vec<usize>,

    /// The number of times the gadget has been called so far.
    ct: Vec<usize>,
}

impl<'a, F, G> QueryShimGadget<'a, F, G>
where
    F: FieldElement,
    G: Gadget<F>,
{
    fn new<V: Value<F, G>>(inner: &'a mut G, x: &V, pf: &Proof<F>) -> Result<Self, PcpError> {
        let mut f_vals: Vec<Vec<Vec<F>>> = (0..inner.len())
            .map(|i| vec![vec![F::zero(); x.valid_gadget_calls(i) + 1]; inner.arity(i)])
            .collect();

        let mut p_vals: Vec<Vec<F>> = (0..inner.len())
            .map(|idx| {
                let g_calls = x.valid_gadget_calls(idx);
                let g_deg = inner.deg(idx);
                let m = (g_calls + 1).next_power_of_two();
                let p = m * g_deg;
                let size = p.next_power_of_two();
                vec![F::zero(); size]
            })
            .collect();

        let mut step: Vec<usize> = vec![0; inner.len()];

        let ct: Vec<usize> = vec![1; inner.len()];

        let mut proof_len = 0;
        for idx in 0..inner.len() {
            let g_calls = x.valid_gadget_calls(idx);
            let g_arity = inner.arity(idx);
            let g_deg = inner.deg(idx);
            let m = (g_calls + 1).next_power_of_two();
            let p = m * g_deg;

            let next_len = g_arity + g_deg * (m - 1) + 1;
            if pf.data.len() < proof_len + next_len {
                return Err(PcpError::Query("proof too short"));
            }

            // Record the intermediate polynomial seeds.
            for wire in 0..g_arity {
                f_vals[idx][wire][0] = pf.data[proof_len + wire];
            }

            // Evaluate the proof polynomial at roots of unity.
            let size = p_vals[idx].len();
            discrete_fourier_transform(
                &mut p_vals[idx],
                &pf.data[proof_len + g_arity..proof_len + next_len],
                size,
            )?;

            step[idx] = (1 << (log2(p as u128) - log2(m as u128))) as usize;

            proof_len += next_len;
        }

        if proof_len < pf.data.len() {
            return Err(PcpError::Query("proof too long"));
        }

        Ok(QueryShimGadget {
            inner,
            f_vals,
            p_vals,
            step,
            ct,
        })
    }
}

impl<'a, F, G> GadgetCallOnly<F> for QueryShimGadget<'a, F, G>
where
    F: FieldElement,
    G: Gadget<F>,
{
    fn call(&mut self, idx: usize, inp: &[F]) -> Result<F, PcpError> {
        for j in 0..inp.len() {
            self.f_vals[idx][j][self.ct[idx]] = inp[j];
        }
        let outp = self.p_vals[idx][self.ct[idx] * self.step[idx]];
        self.ct[idx] += 1;
        Ok(outp)
    }

    fn arity(&self, idx: usize) -> usize {
        self.inner.arity(idx)
    }

    fn deg(&self, idx: usize) -> usize {
        self.inner.deg(idx)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// The output of `query`, the verifier message generated for a proof.
#[derive(Debug)]
pub struct Verifier<F: FieldElement> {
    /// `data` consists of the following values:
    ///
    ///  * `data[0]` is the output of the validity circuit;
    ///  * `data[1..l+1]` are the outputs of the intermediate proof polynomials evaluated at the
    ///     query randomness `r` (`l` denotes the arity of the gadget); and
    ///  * `data[l+1]` is the output of the proof polynomial evaluated at `r`.
    data: Vec<F>,
}

impl<F: FieldElement> Verifier<F> {
    /// Returns a reference to the underlying data.
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
pub fn decide<F, G, V>(x: &V, vf: &Verifier<F>) -> Result<bool, PcpError>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
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
    for i in 0..g.len() {
        let next_len = 1 + g.arity(i);
        if verifier_len + next_len > vf.data.len() {
            return Err(PcpError::Decide("short verifier"));
        }

        let e = g.call(i, &vf.data[verifier_len..verifier_len + next_len - 1])?;
        println!("{}", e - vf.data[verifier_len + next_len - 1]);
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
    use crate::pcp::gadgets::{Mul, Pair, PolyEval};
    use crate::pcp::types::{call_poly_out_len, Boolean};
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

    #[derive(Debug, PartialEq, Eq)]
    pub struct TestValue<F: FieldElement> {
        is_leader: bool, // Whether this is the leader's input share XXX delete me?
        data: Vec<F>,    // The encoded input
    }

    impl<F: FieldElement> TestValue<F> {
        pub fn new(inp: F) -> Self {
            Self {
                is_leader: true,
                data: vec![inp, inp * inp * inp],
            }
        }
    }

    impl<F: FieldElement> Value<F, Pair<F, Mul<F>, PolyEval<F>>> for TestValue<F> {
        type Param = ();
        type TryFromError = Infallible;

        fn valid(&self, g: &mut dyn GadgetCallOnly<F>, joint_rand: &[F]) -> Result<F, PcpError> {
            if joint_rand.len() != self.valid_rand_len() {
                return Err(PcpError::ValidRandLen);
            }

            if self.data.len() != 2 {
                return Err(PcpError::CircuitInLen);
            }

            let r = joint_rand[0];
            let mut res = F::zero();

            // Check that `data[0] == data[1]^3`.
            let mut inp = [self.data[0], self.data[0]];
            inp[0] = g.call(0, &inp)?;
            inp[0] = g.call(0, &inp)?;
            let x3_diff = inp[0] - self.data[1];
            res += r * x3_diff;

            // Check that `data[0]` is in the correct range.
            let x_checked = g.call(1, &[self.data[0]])?;
            res += (r * r) * x_checked;

            Ok(res)
        }

        fn valid_gadget_calls(&self, idx: usize) -> usize {
            match idx {
                0 => 2,
                1 => 1,
                _ => 0,
            }
        }

        fn valid_rand_len(&self) -> usize {
            1
        }

        fn gadget(&self) -> Pair<F, Mul<F>, PolyEval<F>> {
            Pair::new(
                Mul::new(call_poly_out_len(self.valid_gadget_calls(0))),
                PolyEval::new(
                    poly_range_check(2, 5),
                    call_poly_out_len(self.valid_gadget_calls(1)),
                ),
            )
        }

        fn as_slice(&self) -> &[F] {
            &self.data
        }

        fn param(&self) -> Self::Param {
            ()
        }

        fn set_leader(&mut self, is_leader: bool) {
            self.is_leader = is_leader
        }
    }

    impl<F: FieldElement> TryFrom<((), Vec<F>)> for TestValue<F> {
        type Error = Infallible;

        fn try_from(val: ((), Vec<F>)) -> Result<Self, Infallible> {
            Ok(Self {
                is_leader: true,
                data: val.1,
            })
        }
    }
}
