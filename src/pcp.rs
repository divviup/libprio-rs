// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module
//! defines the [`Value`] trait, the main building block of the [`prio3`](crate::vdaf::prio3) VDAF.
//! Implementations of this trait for various measurement types can be found in [`types`].
//!
//! This module also implements a fully linear PCP ("Probabilistically Checkable Proof") system
//! based on [[BBCG+19], Theorem 4.3] suitable for implementations of [`Value`]. This proof system
//! is used in `prio3` to ensure validity of the recovered output shares. Most applications will
//! not need to use this module directly. Typically the VDAF will provide the functionality you
//! need.
//!
//! # Overview
//!
//! The proof system is comprised of three algorithms. The first, `prove`, is run by the prover in
//! order to generate a proof of a statement's validity. The second and third, `query` and
//! `decide`, are run by the verifier in order to check the proof. The proof asserts that the input
//! is an element of a language recognized by an arithmetic circuit. If an input is _not_ valid,
//! then the verification step will fail with high probability. For example:
//!
//! ```
//! use prio::pcp::types::Count;
//! use prio::pcp::{decide, prove, query, Value};
//! use prio::field::{random_vector, FieldElement, Field64};
//!
//! // The prover chooses a measurement.
//! let input: Count<Field64> = Count::new(0).unwrap();
//!
//! // The prover and verifier agree on "joint randomness" used to generate and
//! // check the proof. The application needs to ensure that the prover
//! // "commits" to the input before this point. In the `prio3` VDAF, the joint
//! // randomness is derived from additive shares of the input.
//! let joint_rand = random_vector(input.joint_rand_len()).unwrap();
//!
//! // The prover generates the proof.
//! let prove_rand = random_vector(input.prove_rand_len()).unwrap();
//! let proof = prove(&input, &prove_rand, &joint_rand).unwrap();
//!
//! // The verifier checks the proof.
//! let query_rand = random_vector(input.query_rand_len()).unwrap();
//! let verifier = query(&input, &proof, &query_rand, &joint_rand).unwrap();
//! assert!(decide(&input, &verifier).unwrap());
//! ```
//!
//! The proof system implemented here lifts [[BBCG+19], Theorem 4.3] to a 1.5-round, public-coin,
//! interactive oracle proof system (see [[BBCG+19], Definition 3.11]). The main difference is that
//! the arithmetic circuit may include an additional, random input (called the "joint randomness"
//! above). This allows us to express proof systems like the SIMD circuit of [[BBCG+19]] that
//! trade a modest amount of soundness error for efficiency.
//!
//! Another improvement made here is to allow validity circuits with multiple repeating
//! sub-components (called "gadgets" here, see [`gadgets`]) instead of just one. This idea comes
//! from [[BBCG+19], Remark 4.5].
//!
//! WARNING: These modifications have not yet undergone significant security analysis. As such,
//! this proof system should not be considered suitable for production use.
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [CGB17]: https://crypto.stanford.edu/prio

use crate::fft::{discrete_fourier_transform, discrete_fourier_transform_inv_finish, FftError};
use crate::field::{FieldElement, FieldError};
use crate::fp::log2;
use crate::polynomial::poly_eval;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::convert::TryFrom;
use std::fmt::Debug;

pub mod gadgets;
pub mod types;

/// Errors propagated by methods in this module.
#[derive(Debug, thiserror::Error)]
pub enum PcpError {
    /// Calling [`prove`] returned an error.
    #[error("prove error: {0}")]
    Prove(String),

    /// Calling [`query`] returned an error.
    #[error("query error: {0}")]
    Query(String),

    /// Calling [`decide`] returned an error.
    #[error("decide error: {0}")]
    Decide(String),

    /// Gadget returned an error.
    #[error("gadget error: {0}")]
    Gadget(String),

    /// Validity circuit returned an error.
    #[error("validity circuit error: {0}")]
    Valid(String),

    /// Value error.
    #[error("value error: {0}")]
    Value(String),

    /// Returned if an FFT operation propagates an error.
    #[error("FFT error: {0}")]
    Fft(#[from] FftError),

    /// Returned if a field operation encountered an error.
    #[error("Field error: {0}")]
    Field(#[from] FieldError),
}

/// A value of a certain type. Implementations of this trait specify an arithmetic circuit that
/// determines whether a given value is valid.
pub trait Value:
    Sized
    + PartialEq
    + Eq
    + Debug
    + for<'a> TryFrom<(<Self as Value>::Param, &'a [Self::Field]), Error = PcpError>
{
    /// The finite field used for this type.
    type Field: FieldElement;

    /// Parameters used to construct a value of this type from a vector of field elements.
    type Param: Clone + Debug;

    /// Evaluates the validity circuit on the given input (i.e., `self`) and returns the output.
    /// `joint_rand` is the joint randomness shared by the prover and verifier. `g` is the sequence
    /// of gadgets called by the circuit.
    ///
    /// ```
    /// use prio::pcp::types::Count;
    /// use prio::pcp::Value;
    /// use prio::field::{random_vector, FieldElement, Field64};
    ///
    /// let x: Count<Field64> = Count::new(1).unwrap();
    /// let joint_rand = random_vector(x.joint_rand_len()).unwrap();
    /// let v = x.valid(&mut x.gadget(), &joint_rand).unwrap();
    /// assert_eq!(v, Field64::zero());
    /// ```
    fn valid(
        &self,
        gadgets: &mut Vec<Box<dyn Gadget<Self::Field>>>,
        joint_rand: &[Self::Field],
    ) -> Result<Self::Field, PcpError>;

    /// Returns a reference to the underlying data.
    fn as_slice(&self) -> &[Self::Field];

    /// The length of the random input used by both the prover and the verifier.
    fn joint_rand_len(&self) -> usize;

    /// The length of the random input consumed by the prover to generate a proof. This is the same
    /// as the sum of the arity of each gadget in the validity circuit.
    fn prove_rand_len(&self) -> usize;

    /// The length of the random input consumed by the verifier to make queries against inputs and
    /// proofs. This is the same as the number of gadgets in the validity circuit.
    fn query_rand_len(&self) -> usize;

    /// The number of calls to the gadget made when evaluating the validity circuit.
    //
    // TODO(cjpatton) Consider consolidating this and `gadget` into one call. The benefit would be
    // that there is one less thing to worry about when implementing a Value<F>. We would need to
    // extend Gadget<F> so that it tells you how many times it gets called.
    fn valid_gadget_calls(&self) -> Vec<usize>;

    /// Returns the sequence of gadgets associated with the validity circuit.
    ///
    /// NOTE The construction of [[BBCG+19], Theorem 4.3] uses a single gadget rather than many.
    /// The idea to generalize the proof system to allow multiple gadgets is discussed briefly in
    /// [[BBCG+19], Remark 4.5], but no construction is given. The construction implemented here
    /// requires security analysis.
    ///
    /// [BBCG+19]: https://ia.cr/2019/188
    fn gadget(&self) -> Vec<Box<dyn Gadget<Self::Field>>>;

    /// Returns a copy of the associated type parameters for this value.
    fn param(&self) -> Self::Param;

    /// When verifying a proof over secret shared data, this method may be used to distinguish the
    /// "leader" share from the others. This is useful, for example, when some of the gadget inputs
    /// are constants used for both proof generation and verification.
    //
    // TODO(cjpatton) Deprecate this method. The proper way to do this should be to normalizing by
    // dividing by the number of shares.
    fn set_leader(&mut self, _is_leader: bool) {
        // No-op by default.
    }
}

/// A gadget, a non-affine arithmetic circuit that is called when evaluating a validity circuit.
pub trait Gadget<F: FieldElement> {
    /// Evaluates the gadget on input `inp` and returns the output.
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError>;

    /// Evaluate the gadget on input of a sequence of polynomials. The output is written to `outp`.
    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError>;

    /// Returns the arity of the gadget. This is the length of `inp` passed to `call` or
    /// `call_poly`.
    fn arity(&self) -> usize;

    /// Returns the circuit's arithmetic degree. This determines the minimum length the `outp`
    /// buffer passed to `call_poly`.
    fn degree(&self) -> usize;

    /// This call is used to downcast a `Box<dyn Gadget<F>>` to a concrete type.
    fn as_any(&mut self) -> &mut dyn Any;
}

/// Generate a proof of an input's validity.
///
/// # Parameters
///
/// * `input` is the input.
/// * `prove_rand` is the prover' randomness.
/// * `joint_rand` is the randomness shared by the prover and verifier.
pub fn prove<V: Value>(
    input: &V,
    prove_rand: &[V::Field],
    joint_rand: &[V::Field],
) -> Result<Proof<V::Field>, PcpError> {
    let gadget_calls = input.valid_gadget_calls();

    let mut prove_rand_len = 0;
    let mut shim = input
        .gadget()
        .into_iter()
        .enumerate()
        .map(|(idx, inner)| {
            let inner_arity = inner.arity();
            if prove_rand_len + inner_arity > prove_rand.len() {
                return Err(PcpError::Prove(format!(
                    "short prove randomness: got {}; want {}",
                    prove_rand.len(),
                    input.prove_rand_len()
                )));
            }

            let gadget = Box::new(ProveShimGadget::new(
                inner,
                gadget_calls[idx],
                &prove_rand[prove_rand_len..prove_rand_len + inner_arity],
            )?) as Box<dyn Gadget<V::Field>>;
            prove_rand_len += inner_arity;

            Ok(gadget)
        })
        .collect::<Result<Vec<_>, PcpError>>()?;

    // Create a buffer for storing the proof. The buffer is longer than the proof itself; the extra
    // length is to accommodate the computation of each gadget polynomial.
    let data_len = (0..shim.len())
        .map(|idx| {
            shim[idx].arity() + shim[idx].degree() * (1 + gadget_calls[idx]).next_power_of_two()
        })
        .sum();
    let mut data = vec![V::Field::zero(); data_len];

    // Run the validity circuit with a sequence of "shim" gadgets that record the value of each
    // input wire of each gadget evaluation. These values are used to construct the wire
    // polynomials for each gadget in the next step.
    let _ = input.valid(&mut shim, joint_rand);

    // Fill the buffer with the proof. `proof_len` keeps track of the amount of data written to the
    // buffer so far.
    let mut proof_len = 0;
    for idx in 0..shim.len() {
        let gadget = shim[idx]
            .as_any()
            .downcast_mut::<ProveShimGadget<V::Field>>()
            .unwrap();

        // Interpolate the wire polynomials `f[0], ..., f[g_arity-1]` from the input wires of each
        // evaluation of the gadget.
        let m = (1 + gadget_calls[idx]).next_power_of_two();
        let m_inv =
            V::Field::from(<<V as Value>::Field as FieldElement>::Integer::try_from(m).unwrap())
                .inv();
        let mut f = vec![vec![V::Field::zero(); m]; gadget.arity()];
        for wire in 0..gadget.arity() {
            discrete_fourier_transform(&mut f[wire], &gadget.f_vals[wire], m)?;
            discrete_fourier_transform_inv_finish(&mut f[wire], m, m_inv);

            // The first point on each wire polynomial is a random value chosen by the prover. This
            // point is stored in the proof so that the verifier can reconstruct the wire
            // polynomials.
            data[proof_len + wire] = gadget.f_vals[wire][0];
        }

        // Construct the gadget polynomial `G(f[0], ..., f[g_arity-1])` and append it to `data`.
        gadget.call_poly(&mut data[proof_len + gadget.arity()..], &f)?;
        proof_len += gadget.arity() + gadget.degree() * (m - 1) + 1;
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
        gadget_calls: usize,
        prove_rand: &[F],
    ) -> Result<Self, PcpError> {
        let mut f_vals = vec![vec![F::zero(); 1 + gadget_calls]; inner.arity()];

        #[allow(clippy::needless_range_loop)]
        for wire in 0..f_vals.len() {
            // Choose a random field element as the first point on the wire polynomial.
            f_vals[wire][0] = prove_rand[wire];
        }

        Ok(Self {
            inner,
            f_vals,
            ct: 1,
        })
    }
}

impl<F: FieldElement> Gadget<F> for ProveShimGadget<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        #[allow(clippy::needless_range_loop)]
        for wire in 0..inp.len() {
            self.f_vals[wire][self.ct] = inp[wire];
        }
        self.ct += 1;
        self.inner.call(inp)
    }

    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), PcpError> {
        self.inner.call_poly(outp, inp)
    }

    fn arity(&self) -> usize {
        self.inner.arity()
    }

    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// The output of `prove`, a proof of an input's validity.
#[derive(Clone, Debug)]
pub struct Proof<F: FieldElement> {
    pub(crate) data: Vec<F>,
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

impl<F: FieldElement> From<Proof<F>> for Vec<u8> {
    fn from(proof: Proof<F>) -> Self {
        F::slice_into_byte_vec(&proof.data)
    }
}

/// Generate a verifier message for an input and proof (or the verifier share for an input share
/// and proof share).
///
/// # Parameters
///
/// * `input` is the input or input share.
/// * `proof` is the proof or proof share.
/// * `query_rand` is the verifier's randomness.
/// * `joint_rand` is the randomness shared by the prover and verifier.
pub fn query<V: Value>(
    input: &V,
    proof: &Proof<V::Field>,
    query_rand: &[V::Field],
    joint_rand: &[V::Field],
) -> Result<Verifier<V::Field>, PcpError> {
    let gadget_calls = input.valid_gadget_calls();

    let mut proof_len = 0;
    let mut shim = input
        .gadget()
        .into_iter()
        .enumerate()
        .map(|(idx, gadget)| {
            if idx >= query_rand.len() {
                return Err(PcpError::Query(format!(
                    "short query randomness: got {}; want {}",
                    query_rand.len(),
                    input.query_rand_len()
                )));
            }

            let gadget_degree = gadget.degree();
            let gadget_arity = gadget.arity();
            let m = (1 + gadget_calls[idx]).next_power_of_two();
            let r = query_rand[idx];

            // Make sure the query randomness isn't a root of unity. Evaluating the gadget
            // polynomial at any of these points would be a privacy violation, since these points
            // were used by the prover to construct the wire polynomials.
            if r.pow(<<V as Value>::Field as FieldElement>::Integer::try_from(m).unwrap())
                == V::Field::one()
            {
                return Err(PcpError::Query(format!(
                    "invalid query randomness: encountered 2^{}-th root of unity",
                    m
                )));
            }

            // Compute the length of the sub-proof corresponding to the `idx`-th gadget.
            let next_len = gadget_arity + gadget_degree * (m - 1) + 1;
            if proof_len + next_len > proof.data.len() {
                // TODO(cjpatton) Compute the expected proof length and print it in error message.
                return Err(PcpError::Query("short proof".to_string()));
            }

            let proof_data = &proof.data[proof_len..proof_len + next_len];
            proof_len += next_len;

            Ok(Box::new(QueryShimGadget::new(
                gadget,
                r,
                proof_data,
                gadget_calls[idx],
            )?) as Box<dyn Gadget<V::Field>>)
        })
        .collect::<Result<Vec<_>, _>>()?;

    if proof_len < proof.data.len() {
        // TODO(cjpatton) Compute the expected proof length and print it in error message.
        return Err(PcpError::Query("long proof".to_string()));
    }

    if query_rand.len() > shim.len() {
        return Err(PcpError::Query(format!(
            "long query randomness: got {}; want {}",
            query_rand.len(),
            input.query_rand_len()
        )));
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
    let validity = input.valid(&mut shim, joint_rand)?;
    data.push(validity);

    // Fill the buffer with the verifier message.
    for idx in 0..shim.len() {
        let r = query_rand[idx];
        let gadget = shim[idx]
            .as_any()
            .downcast_ref::<QueryShimGadget<V::Field>>()
            .unwrap();

        // Reconstruct the wire polynomials `f[0], ..., f[g_arity-1]` and evaluate each wire
        // polynomial at query randomness `r`.
        let m = (1 + gadget_calls[idx]).next_power_of_two();
        let m_inv =
            V::Field::from(<<V as Value>::Field as FieldElement>::Integer::try_from(m).unwrap())
                .inv();
        let mut f = vec![V::Field::zero(); m];
        for wire in 0..gadget.arity() {
            discrete_fourier_transform(&mut f, &gadget.f_vals[wire], m)?;
            discrete_fourier_transform_inv_finish(&mut f, m, m_inv);
            data.push(poly_eval(&f, r));
        }

        // Add the value of the gadget polynomial evaluated at `r`.
        data.push(gadget.p_at_r);
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
        gadget_calls: usize,
    ) -> Result<Self, PcpError> {
        let gadget_degree = inner.degree();
        let gadget_arity = inner.arity();
        let m = (1 + gadget_calls).next_power_of_two();
        let p = m * gadget_degree;

        // Each call to this gadget records the values at which intermediate proof polynomials were
        // interpolated. The first point was a random value chosen by the prover and transmitted in
        // the proof.
        let mut f_vals = vec![vec![F::zero(); 1 + gadget_calls]; gadget_arity];
        for wire in 0..gadget_arity {
            f_vals[wire][0] = proof_data[wire];
        }

        // Evaluate the gadget polynomial at roots of unity.
        let size = p.next_power_of_two();
        let mut p_vals = vec![F::zero(); size];
        discrete_fourier_transform(&mut p_vals, &proof_data[gadget_arity..], size)?;

        // The step is used to compute the element of `p_val` that will be returned by a call to
        // the gadget.
        let step = (1 << (log2(p as u128) - log2(m as u128))) as usize;

        // Evaluate the gadget polynomial `p` at query randomness `r`.
        let p_at_r = poly_eval(&proof_data[gadget_arity..], r);

        Ok(Self {
            inner,
            f_vals,
            p_vals,
            p_at_r,
            step,
            ct: 1,
        })
    }
}

impl<F: FieldElement> Gadget<F> for QueryShimGadget<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, PcpError> {
        #[allow(clippy::needless_range_loop)]
        for wire in 0..inp.len() {
            self.f_vals[wire][self.ct] = inp[wire];
        }
        let outp = self.p_vals[self.ct * self.step];
        self.ct += 1;
        Ok(outp)
    }

    fn call_poly(&mut self, _outp: &mut [F], _inp: &[Vec<F>]) -> Result<(), PcpError> {
        panic!("no-op");
    }

    fn arity(&self) -> usize {
        self.inner.arity()
    }

    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// The output of `query`, the verifier message generated for a proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Verifier<F> {
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

/// Decide if the input (or input share) is valid using the given verifier.
///
/// # Parameters
///
/// * `input` is the input or input share.
/// * `verifier` is the verifier message or a share of the verifier message.
pub fn decide<V: Value>(input: &V, verifier: &Verifier<V::Field>) -> Result<bool, PcpError> {
    let mut gadgets = input.gadget();

    if verifier.data.is_empty() {
        // TODO(cjpatton) Compute expected verifier length and output it here
        return Err(PcpError::Decide("zero-length verifier".to_string()));
    }

    // Check if the output of the circuit is 0.
    if verifier.data[0] != V::Field::zero() {
        return Ok(false);
    }

    // Check that each of the proof polynomials are well-formed.
    let mut verifier_len = 1;
    #[allow(clippy::needless_range_loop)]
    for idx in 0..gadgets.len() {
        let next_len = 1 + gadgets[idx].arity();
        if verifier_len + next_len > verifier.data.len() {
            // TODO(cjpatton) Compute expected verifier length and output it here
            return Err(PcpError::Decide("short verifier".to_string()));
        }

        let e = gadgets[idx].call(&verifier.data[verifier_len..verifier_len + next_len - 1])?;
        if e != verifier.data[verifier_len + next_len - 1] {
            return Ok(false);
        }

        verifier_len += next_len;
    }

    if verifier_len != verifier.data.len() {
        // TODO(cjpatton) Compute expected verifier length and output it here
        return Err(PcpError::Decide("long verifier".to_string()));
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{random_vector, split_vector, Field126};
    use crate::pcp::gadgets::{Mul, PolyEval};
    use crate::pcp::types::Count;
    use crate::polynomial::poly_range_check;

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
        let x_shares: Vec<T> = split_vector(x.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, data)| {
                let mut share = T::try_from((x_par, data.as_slice())).unwrap();
                share.set_leader(i == 0);
                share
            })
            .collect();

        let joint_rand = random_vector(x.joint_rand_len()).unwrap();
        let prove_rand = random_vector(x.prove_rand_len()).unwrap();
        let query_rand = random_vector(x.query_rand_len()).unwrap();

        let pf = prove(&x, &prove_rand, &joint_rand).unwrap();
        let pf_shares: Vec<Proof<F>> = split_vector(pf.as_slice(), NUM_SHARES)
            .unwrap()
            .into_iter()
            .map(Proof::from)
            .collect();

        let vf: Verifier<F> = (0..NUM_SHARES)
            .map(|i| query(&x_shares[i], &pf_shares[i], &query_rand, &joint_rand).unwrap())
            .reduce(|mut left, right| {
                for (x, y) in left.data.iter_mut().zip(right.data.iter()) {
                    *x += *y;
                }
                Verifier { data: left.data }
            })
            .unwrap();
        assert!(decide(&x, &vf).unwrap());
    }

    #[test]
    fn test_decide() {
        let x: Count<Field126> = Count::new(1).unwrap();
        let joint_rand = random_vector(x.joint_rand_len()).unwrap();
        let prove_rand = random_vector(x.prove_rand_len()).unwrap();
        let query_rand = random_vector(x.query_rand_len()).unwrap();

        let ok_vf = query(
            &x,
            &prove(&x, &prove_rand, &joint_rand).unwrap(),
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

    impl<F: FieldElement> Value for TestValue<F> {
        type Field = F;
        type Param = ();

        fn valid(&self, g: &mut Vec<Box<dyn Gadget<F>>>, joint_rand: &[F]) -> Result<F, PcpError> {
            if joint_rand.len() != self.joint_rand_len() {
                return Err(PcpError::Valid(format!(
                    "unexpected joint randomness length: got {}; want {}",
                    joint_rand.len(),
                    self.joint_rand_len()
                )));
            }

            if self.data.len() != 2 {
                return Err(PcpError::Valid(format!(
                    "unexpected input length: got {}; want {}",
                    self.data.len(),
                    2
                )));
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

        fn joint_rand_len(&self) -> usize {
            1
        }

        fn prove_rand_len(&self) -> usize {
            3
        }

        fn query_rand_len(&self) -> usize {
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

        fn param(&self) -> Self::Param {}
    }

    impl<F: FieldElement> TryFrom<((), &[F])> for TestValue<F> {
        type Error = PcpError;

        fn try_from(val: ((), &[F])) -> Result<Self, PcpError> {
            Ok(Self {
                data: val.1.to_vec(),
            })
        }
    }
}
