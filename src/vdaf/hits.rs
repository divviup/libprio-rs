// SPDX-License-Identifier: MPL-2.0

//! **(NOTE: This module is experimental. Applications should not use it yet.)** This module
//! implements the VDAF [[VDAF]] used for the privacy-preserving heavy hitters protocol of
//! [[BBCG+21]]. It is constructed from an Incremental Distributed Point Function (IDPF).
//!
//! [VDAF]: https://cjpatton.github.io/vdaf/draft-patton-cfrg-vdaf.html
//! [BBCG+21]: https://eprint.iacr.org/2021/017

use std::array::IntoIter;
use std::collections::HashSet;
use std::convert::TryFrom;

use crate::field::{split_vector, FieldElement};
use crate::fp::log2;
use crate::prng::Prng;
use crate::vdaf::suite::{Key, KeyDeriver, KeyStream, Suite};
use crate::vdaf::{Share, VdafError};

/// An input for an IDPF ([`Idpf`]).
#[derive(Clone, Debug, Hash, Eq)]
pub struct IdpfInput<'a> {
    data: &'a [u8],
    len: usize,
}

impl<'a> IdpfInput<'a> {
    /// Constructs an IDPF input using the first `len` bits of `data`.
    pub fn new(data: &'a [u8], len: usize) -> Result<Self, VdafError> {
        if len > data.len() << 3 {
            return Err(VdafError::Uncategorized(format!(
                "desired bit length ({} bits) exceeds data length ({} bytes)",
                len,
                data.len()
            )));
        }

        Ok(Self { data, len })
    }

    /// Return the `bit`-th bit of the input. Bounds checking is performed by the caller.
    fn bit(&self, bit: usize) -> usize {
        let i = bit >> 3;
        let j = ((i << 3) ^ bit) as u8;
        (self.data[i] >> j) as usize & 1
    }

    /// Return the index of input `self` in the look-up table of a `ToyIdpf` instance.
    fn index(&self) -> usize {
        let mut index = 1 << self.len;
        for i in 0..self.len {
            index += self.bit(i) << i;
        }
        index - 1
    }

    /// Construct a new input that is a prefix of `self`. Bounds checking is performed by the
    /// caller.
    fn prefix(&self, len: usize) -> Self {
        Self {
            data: self.data,
            len,
        }
    }
}

impl<'a> PartialEq<IdpfInput<'a>> for IdpfInput<'a> {
    fn eq(&self, other: &IdpfInput) -> bool {
        if self.len != other.len {
            return false;
        }

        // Do a constant-time comparison, masking the last byte to fit the input length (in bits).
        let i = self.len >> 3;
        let j = ((i << 3) ^ self.len) as u8;
        let mut out = 0;
        for (r, (a, b)) in self.data.iter().zip(other.data).enumerate() {
            let (mut x, mut y) = (*a, *b);
            if r == i {
                x &= j;
                y &= j;
            }

            if r > i {
                break;
            }

            out |= x ^ y;
        }
        out == 0
    }
}

/// An Incremental Distributed Point Function (IDPF), as defined by [[BBCG+21]].
///
/// [BBCG+21]: https://eprint.iacr.org/2021/017
//
// NOTE(cjpatton) The real IDPF API probably needs to be stateful.
pub trait Idpf<const KEY_LEN: usize, const OUT_LEN: usize>: Sized {
    /// The finite field over which the IDPF is defined.
    //
    // NOTE(cjpatton) The IDPF of [BBCG+21] might use different fields for different levels of the
    // prefix tree.
    type Field: FieldElement;

    /// Generate and return a sequence of IDPF shares for `input`. Parameter `output` is an
    /// iterator that is invoked to get the output value for each successive level of the prefix
    /// tree.
    fn gen<O: IntoIterator<Item = Self::Field>>(
        input: &IdpfInput,
        output: O,
    ) -> Result<[Self; KEY_LEN], VdafError>;

    /// Evaluate an IDPF share on `prefix`.
    fn eval(&self, prefix: &IdpfInput) -> Result<[Self::Field; OUT_LEN], VdafError>;
}

/// A "toy" IPDF used for demonstration purposes. The space consumed by each share is `O(2^n)`,
/// where `n` is the length of the input. The size of each share is restricted to 1MB, so this IDPF
/// is only suitable for very short inputs.
//
// NOTE(cjpatton) It would be straight-forward to generalize this construction to any `KEY_LEN` and
// `OUT_LEN`.
pub struct ToyIdpf<F> {
    data: Vec<F>,
    len: usize,
}

impl<F: FieldElement> Idpf<2, 1> for ToyIdpf<F> {
    type Field = F;

    fn gen<M: IntoIterator<Item = Self::Field>>(
        input: &IdpfInput,
        output: M,
    ) -> Result<[Self; 2], VdafError> {
        const MAX_DATA_BYTES: usize = 1024 * 1024; // 1MB

        let max_input_len =
            usize::try_from(log2((MAX_DATA_BYTES / F::ENCODED_SIZE) as u128)).unwrap();
        if input.len > max_input_len {
            return Err(VdafError::Uncategorized(format!(
                "input length ({}) exceeds maximum of ({})",
                input.len, max_input_len
            )));
        }

        let data_len = 1 << (input.len + 1);
        let mut data = vec![F::zero(); data_len];
        let mut output = output.into_iter();
        for len in 0..input.len + 1 {
            data[input.prefix(len).index()] = output.next().unwrap();
        }

        // NOTE(cjpatton) We could save some space by representing one of the key shares as a PRNG
        // seed. Of course, there's not much point, since this is a toy.
        let mut data = split_vector(&data, 2)?.into_iter();
        Ok([
            ToyIdpf {
                data: data.next().unwrap(),
                len: input.len,
            },
            ToyIdpf {
                data: data.next().unwrap(),
                len: input.len,
            },
        ])
    }

    fn eval(&self, prefix: &IdpfInput) -> Result<[F; 1], VdafError> {
        if prefix.len > self.len {
            return Err(VdafError::Uncategorized(format!(
                "prefix length ({}) exceeds input length ({})",
                prefix.len, self.len
            )));
        }

        Ok([self.data[prefix.index()]])
    }
}

/// An input share for the heavy hitters VDAF.
pub struct InputShareMessage<I: Idpf<2, 1>> {
    /// IDPF share of input
    pub data: I,

    /// IDPF share of the authentication vector
    pub auth: I,

    /// PRNG seed used to generate the aggregator's share of the randomness used in the first part
    /// of the sketching protocol.
    pub sketch_start_seed: Key,

    /// Aggregator's share of the randomness used in the second part of the sketching protocol.
    pub sketch_next: Share<I::Field>,
}

/// The VDAF input-distribution algorithm for heavy hitters.
#[allow(clippy::many_single_char_names)]
pub fn hits_input<I: Idpf<2, 1>>(
    suite: Suite,
    input: &IdpfInput,
) -> Result<[InputShareMessage<I>; 2], VdafError> {
    let auth_rand: Vec<I::Field> = Prng::generate(suite)?.take(input.len + 1).collect();

    // For each level of the prefix tree, generate correlated randomness that the aggregators use
    // to validate the output. See [BBCG+21, Appendix C.4].
    let leader_sketch_start_seed = Key::generate(suite)?;
    let helper_sketch_start_seed = Key::generate(suite)?;
    let helper_sketch_next_seed = Key::generate(suite)?;
    let mut leader_sketch_start_prng: Prng<I::Field> =
        Prng::from_key_stream(KeyStream::from_key(&leader_sketch_start_seed));
    let mut helper_sketch_start_prng: Prng<I::Field> =
        Prng::from_key_stream(KeyStream::from_key(&helper_sketch_start_seed));
    let mut helper_sketch_next_prng: Prng<I::Field> =
        Prng::from_key_stream(KeyStream::from_key(&helper_sketch_next_seed));
    let mut leader_sketch_next: Vec<I::Field> = Vec::with_capacity(2 * input.len);
    for k in auth_rand.iter() {
        // [BBCG+21, Appendix C.4]
        //
        // $(a, b, c)$
        let a = leader_sketch_start_prng.next().unwrap() + helper_sketch_start_prng.next().unwrap();
        let b = leader_sketch_start_prng.next().unwrap() + helper_sketch_start_prng.next().unwrap();
        let c = leader_sketch_start_prng.next().unwrap() + helper_sketch_start_prng.next().unwrap();

        // $A = -2a + k$
        // $B = a^2 + b + -ak + c$
        let d = *k - (a + a);
        let e = (a * a) + b - (a * *k) + c;
        leader_sketch_next.push(d - helper_sketch_next_prng.next().unwrap());
        leader_sketch_next.push(e - helper_sketch_next_prng.next().unwrap());
    }

    // Generate IDPF shares of the data and authentication vectors.
    let mut data = IntoIter::new(I::gen(input, std::iter::repeat(I::Field::one()))?);
    let mut auth = IntoIter::new(I::gen(input, auth_rand)?);

    Ok([
        InputShareMessage {
            data: data.next().unwrap(),
            auth: auth.next().unwrap(),
            sketch_start_seed: leader_sketch_start_seed,
            sketch_next: Share::Leader(leader_sketch_next),
        },
        InputShareMessage {
            data: data.next().unwrap(),
            auth: auth.next().unwrap(),
            sketch_start_seed: helper_sketch_start_seed,
            sketch_next: Share::Helper {
                seed: helper_sketch_next_seed,
                length: 2 * input.len,
            },
        },
    ])
}

/// The verification parameter used by the aggregators to evaluate the VDAF on a distributed input.
pub struct VerifyParam {
    /// Key used to derive the verification randomness from the nonce.
    pub verify_rand_init: Key,
}

/// The VDAF setup algorithm that generates the verification parameters.
#[cfg(test)]
fn hits_setup(suite: Suite) -> Result<[VerifyParam; 2], VdafError> {
    let verify_rand_init = Key::generate(suite)?;
    Ok([
        VerifyParam {
            verify_rand_init: verify_rand_init.clone(),
        },
        VerifyParam { verify_rand_init },
    ])
}

/// The VDAF aggregation parameter, a sequence of equal-length candidate prefixes.
#[derive(Debug, PartialEq)]
pub struct AggregationParam<'a> {
    prefixes: Vec<IdpfInput<'a>>,
    len: usize,
}

impl<'a> AggregationParam<'a> {
    /// Constructs an aggregation parameter from a sequence of prefixes. An error is returned if
    /// the prefixes aren't equal length, a prefix appears twice in the set of prefixes, or the
    /// set of prefixes is empty.
    pub fn new<P: IntoIterator<Item = IdpfInput<'a>>>(
        prefixes: P,
    ) -> Result<AggregationParam<'a>, VdafError> {
        let mut unique_prefixes = HashSet::new();
        let mut len = None;
        let prefixes = prefixes
            .into_iter()
            .map(|prefix| {
                if unique_prefixes.contains(&prefix) {
                    return Err(VdafError::Uncategorized(format!(
                        "prefix {:?} occurs twice in prefix set",
                        prefix
                    )));
                }

                if let Some(l) = len {
                    if prefix.len != l {
                        return Err(VdafError::Uncategorized(
                            "prefixes must all have the same length".to_string(),
                        ));
                    }
                } else {
                    len = Some(prefix.len);
                }
                unique_prefixes.insert(prefix.clone());
                Ok(prefix)
            })
            .collect::<Result<Vec<_>, VdafError>>()?;

        match len {
            Some(len) => Ok(AggregationParam { prefixes, len }),
            None => Err(VdafError::Uncategorized("prefix set is empty".to_string())),
        }
    }
}

/// The state of an aggregator while evaluating the VDAF.
pub enum VerifyState<F: FieldElement> {
    /// State after first round of VDAF evaluation.
    Start {
        /// The aggregator's share of the output.
        output_share: Vec<F>,

        /// Aggregator's share of $A = -2a + k$ (see [[BBDG+21](https://eprint.iacr.org/2021/017),
        /// Appendix C.4]).
        d: F,

        /// Aggregator's share of $B = a^2 + b -ak + c$.
        e: F,
    },

    /// State after secon round of VDAF evaluation.
    Next {
        /// The aggregator's share of the output.
        output_share: Vec<F>,
    },
}

/// Structure of the verification messages produced by the aggregators while evaluating the VDAF.
#[derive(Clone)]
pub enum VerifyMessage<F: FieldElement> {
    /// First round message.
    Start {
        /// Shares of the blinded polynomial coefficients (see
        /// [[BBDG+21](https://eprint.iacr.org/2021/017), Appendix C.4]).
        z: [F; 3],
    },

    /// Second round message.
    Next {
        /// Share of the polynomial evaluation.
        y: F,
    },
}

/// The VDAF verify-start algorithm for heavy hitters.
#[allow(clippy::type_complexity)]
#[allow(clippy::many_single_char_names)]
pub fn hits_start<I: Idpf<2, 1>>(
    verify_param: &VerifyParam,
    agg_param: &AggregationParam,
    nonce: &[u8],
    input_share: &InputShareMessage<I>,
) -> Result<(VerifyState<I::Field>, VerifyMessage<I::Field>), VdafError> {
    let i = agg_param.len;

    // Derive the verification randomness.
    let mut deriver = KeyDeriver::from_key(&verify_param.verify_rand_init);
    deriver.update(nonce);
    let verify_rand_seed = deriver.finish();
    let mut verify_rand_prng: Prng<I::Field> =
        Prng::from_key_stream(KeyStream::from_key(&verify_rand_seed));

    // Evaluate the IDPF shares and compute the polynomial coefficients.
    let mut z = [I::Field::zero(); 3];
    let mut output_share = Vec::with_capacity(agg_param.prefixes.len());
    for prefix in agg_param.prefixes.iter() {
        let v = input_share.data.eval(prefix)?[0];
        let k = input_share.auth.eval(prefix)?[0];
        let r = verify_rand_prng.next().unwrap();

        // [BBCG+21, Appendix C.4]
        //
        // $(z_\sigma, z^*_\sigma, z^{**}_\sigma)$
        let tmp = r * v;
        z[0] += tmp;
        z[1] += r * tmp;
        z[2] += r * k;
        output_share.push(v);
    }

    // [BBCG+21, Appendix C.4]
    //
    // Add blind shares $(a_\sigma b_\sigma, c_\sigma)$
    //
    // NOTE(cjpatton) We can make this faster by a factor of 3 by using three seed shares instead
    // of one.
    let mut prng =
        Prng::<I::Field>::from_key_stream(KeyStream::from_key(&input_share.sketch_start_seed))
            .skip(3 * i);
    z[0] += prng.next().unwrap();
    z[1] += prng.next().unwrap();
    z[2] += prng.next().unwrap();

    let (d, e) = match &input_share.sketch_next {
        Share::Leader(data) => (data[2 * i], data[2 * i + 1]),
        Share::Helper { seed, length: _ } => {
            // NOTE(cjpatton) We can make this faster by a factor of 2 by using two seed shares
            // instead of one.
            let mut prng = Prng::<I::Field>::from_key_stream(KeyStream::from_key(seed)).skip(2 * i);
            (prng.next().unwrap(), prng.next().unwrap())
        }
    };

    Ok((
        VerifyState::Start { output_share, d, e },
        VerifyMessage::Start { z },
    ))
}

/// The VDAF verify-next algorithm for heavy hitters. The protocol has two rounds, so this is only
/// called once by each aggregator.
#[allow(clippy::many_single_char_names)]
pub fn hits_next<F, M>(
    state: VerifyState<F>,
    msg: M,
) -> Result<(VerifyState<F>, VerifyMessage<F>), VdafError>
where
    F: FieldElement,
    M: IntoIterator<Item = VerifyMessage<F>>,
{
    // Compute polynomial coefficients.
    let z = msg
        .into_iter()
        .map(|msg| match msg {
            VerifyMessage::Start { z } => Ok(z),
            _ => Err(VdafError::Uncategorized(
                "hits_next() called on unexpected message".to_string(),
            )),
        })
        .reduce(|z_left, z_right| {
            let z_left = z_left?;
            let z_right = z_right?;
            Ok([
                z_left[0] + z_right[0],
                z_left[1] + z_right[1],
                z_left[2] + z_right[2],
            ])
        })
        .unwrap_or_else(|| {
            Err(VdafError::Uncategorized(
                "hits_next(): empty verify message iterator".to_string(),
            ))
        })?;

    let (output_share, d, e) = match state {
        VerifyState::Start { output_share, d, e } => (output_share, d, e),
        _ => {
            return Err(VdafError::Uncategorized(
                "hits_next() called in unexpected state".to_string(),
            ));
        }
    };

    // Compute our share of the polynomial evaluation.
    //
    // NOTE(cjpatton) This differs slightly from [BBCG+21] in that the first three terms are
    // scaled by the number of shares of the output.
    let x = ((z[0] * z[0]) - z[1] - z[2]) / (F::one() + F::one());
    let y = x + (d * z[0]) + e;
    Ok((
        VerifyState::Next { output_share },
        VerifyMessage::Next { y },
    ))
}

/// The VDAF verify-finish algorithm for heavy hitters.
pub fn hits_finish<F, M>(state: VerifyState<F>, msg: M) -> Result<Vec<F>, VdafError>
where
    F: FieldElement,
    M: IntoIterator<Item = VerifyMessage<F>>,
{
    let y = msg
        .into_iter()
        .map(|msg| match msg {
            VerifyMessage::Next { y } => Ok(y),
            _ => Err(VdafError::Uncategorized(
                "hits_finish() called on unexpected message".to_string(),
            )),
        })
        .reduce(|y_left, y_right| Ok(y_left? + y_right?))
        .unwrap_or_else(|| {
            Err(VdafError::Uncategorized(
                "hits_next(): empty verify message iterator".to_string(),
            ))
        })?;

    let output_share = match state {
        VerifyState::Next { output_share } => (output_share),
        _ => {
            return Err(VdafError::Uncategorized(
                "hits_finish() called in unexpected state".to_string(),
            ));
        }
    };

    if y != F::zero() {
        return Err(VdafError::Uncategorized(format!(
            "hits_finish(): output is invalid: got {}; expected {}",
            y,
            F::zero(),
        )));
    }

    Ok(output_share)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::Field126;

    #[test]
    fn test_ipdf() {
        // IDPF input equality tests.
        assert_eq!(
            IdpfInput::new(b"hello", 40).unwrap(),
            IdpfInput::new(b"hello", 40).unwrap()
        );
        assert_eq!(
            IdpfInput::new(b"hi", 9).unwrap(),
            IdpfInput::new(b"ha", 9).unwrap(),
        );
        assert_eq!(
            IdpfInput::new(b"hello", 25).unwrap(),
            IdpfInput::new(b"help", 25).unwrap()
        );
        assert_ne!(
            IdpfInput::new(b"hello", 40).unwrap(),
            IdpfInput::new(b"hello", 39).unwrap()
        );
        assert_ne!(
            IdpfInput::new(b"hello", 40).unwrap(),
            IdpfInput::new(b"hell-", 40).unwrap()
        );

        // Generate IPDF keys.
        let input = IdpfInput::new(b"hi", 16).unwrap();
        let keys = ToyIdpf::<Field126>::gen(&input, std::iter::repeat(Field126::one())).unwrap();

        // Try evaluating the IPDF keys on all prefixes.
        for prefix_len in 0..input.len + 1 {
            let res = eval_idpf(&keys, &input.prefix(prefix_len), &[Field126::one()]);
            assert!(res.is_ok(), "prefix_len={} error: {:?}", prefix_len, res);
        }

        // Try evaluating the IPDF keys on incorrect prefixes.
        let res = eval_idpf(
            &keys,
            &IdpfInput::new(&[2], 2).unwrap(),
            &[Field126::zero()],
        );
        assert!(res.is_ok(), "{:?}", res);

        let res = eval_idpf(
            &keys,
            &IdpfInput::new(&[23, 1], 12).unwrap(),
            &[Field126::zero()],
        );
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_hits() {
        // Try constructing an invalid aggregation parameters.
        //
        // The parameter is invalid if the same prefix appears twice.
        assert!(AggregationParam::new([
            IdpfInput::new(b"hello", 40).unwrap(),
            IdpfInput::new(b"hello", 40).unwrap()
        ])
        .is_err());

        // The parameter is invalid if there is a mixture of prefix lengths.
        assert!(AggregationParam::new([
            IdpfInput::new(b"hi", 10).unwrap(),
            IdpfInput::new(b"xx", 12).unwrap(),
        ])
        .is_err());

        let suite = Suite::Blake3;
        let verify_params = hits_setup(suite).unwrap();

        // Run the VDAF input-distribution algorithm.
        let input = IdpfInput::new(b"hi", 16).unwrap();
        let input_shares = hits_input::<ToyIdpf<Field126>>(suite, &input).unwrap();
        let nonce = b"This is a nonce";

        // Try evaluating the VDAF on each prefix of the input.
        for prefix_len in 0..input.len + 1 {
            let res = eval_vdaf(
                &verify_params,
                &nonce[..],
                &input_shares,
                &AggregationParam::new([input.prefix(prefix_len)]).unwrap(),
                Some(&[Field126::one()]),
            );
            assert!(res.is_ok(), "prefix_len={} error: {:?}", prefix_len, res);
        }

        // Try various prefixes.
        let prefix_len = 9;
        let res = eval_vdaf(
            &verify_params,
            &nonce[..],
            &input_shares,
            &AggregationParam::new([
                IdpfInput::new(b"xx", prefix_len).unwrap(),
                IdpfInput::new(b"x-", prefix_len).unwrap(),
                IdpfInput::new(b"ho", prefix_len).unwrap(),
                IdpfInput::new(&[23, 1], prefix_len).unwrap(),
                IdpfInput::new(&[16, 1], prefix_len).unwrap(),
                IdpfInput::new(&[44, 1], prefix_len).unwrap(),
                IdpfInput::new(&[13, 1], prefix_len).unwrap(),
                IdpfInput::new(&[0, 1], prefix_len).unwrap(),
                IdpfInput::new(&[0, 0], prefix_len).unwrap(),
            ])
            .unwrap(),
            Some(&[
                Field126::zero(),
                Field126::zero(),
                Field126::one(),
                Field126::zero(),
                Field126::zero(),
                Field126::zero(),
                Field126::zero(),
                Field126::zero(),
                Field126::zero(),
            ]),
        );
        assert!(res.is_ok(), "{:?}", res);

        // Try evaluating the VDAF with malformed inputs.
        //
        // This IDPF key pair evaluates to 1 everywhere, which is illegal.
        let mut input_shares = hits_input::<ToyIdpf<Field126>>(suite, &input).unwrap();
        for (i, x) in input_shares[0].data.data.iter_mut().enumerate() {
            if i != input.index() {
                *x += Field126::one();
            }
        }
        let prefix_len = 16;
        let res = eval_vdaf(
            &verify_params,
            &nonce[..],
            &input_shares,
            &AggregationParam::new([IdpfInput::new(b"xx", prefix_len).unwrap()]).unwrap(),
            None,
        );
        assert!(res.is_err());

        // This IDPF key pair has a garbled authentication vector.
        let mut input_shares = hits_input::<ToyIdpf<Field126>>(suite, &input).unwrap();
        for x in input_shares[0].data.data.iter_mut() {
            *x = Field126::zero();
        }
        let prefix_len = 16;
        let res = eval_vdaf(
            &verify_params,
            &nonce[..],
            &input_shares,
            &AggregationParam::new([IdpfInput::new(b"xx", prefix_len).unwrap()]).unwrap(),
            None,
        );
        assert!(res.is_err());
    }

    fn eval_idpf<I, const KEY_LEN: usize, const OUT_LEN: usize>(
        keys: &[I; KEY_LEN],
        input: &IdpfInput,
        expected_output: &[I::Field; OUT_LEN],
    ) -> Result<(), VdafError>
    where
        I: Idpf<KEY_LEN, OUT_LEN>,
    {
        let mut output = [I::Field::zero(); OUT_LEN];
        for key in keys {
            let output_share = key.eval(input)?;
            for (x, y) in output.iter_mut().zip(output_share) {
                *x += y;
            }
        }

        if expected_output != &output {
            return Err(VdafError::Uncategorized(format!(
                "eval_idpf(): unexpected output: got {:?}; want {:?}",
                output, expected_output
            )));
        }

        Ok(())
    }

    fn eval_vdaf<I: Idpf<2, 1>>(
        verify_params: &[VerifyParam],
        nonce: &[u8],
        input_shares: &[InputShareMessage<I>],
        agg_param: &AggregationParam,
        expected_output: Option<&[I::Field]>,
    ) -> Result<(), VdafError> {
        // Each aggregator runs verify-start.
        let mut state1: Vec<VerifyState<I::Field>> = Vec::with_capacity(2);
        let mut round1: Vec<VerifyMessage<I::Field>> = Vec::with_capacity(2);
        for (verify_param, input_share) in verify_params.iter().zip(input_shares.iter()) {
            let (state, msg) = hits_start(verify_param, agg_param, nonce, input_share)?;
            state1.push(state);
            round1.push(msg);
        }

        // Aggregators exchange round-1 messages and run verify-next.
        let mut state2: Vec<VerifyState<I::Field>> = Vec::with_capacity(2);
        let mut round2: Vec<VerifyMessage<I::Field>> = Vec::with_capacity(2);
        for state in state1.into_iter() {
            let (state, msg) = hits_next(state, round1.clone())?;
            state2.push(state);
            round2.push(msg);
        }

        // Aggregators exchange round-2 messages and run verify-finish.
        let mut output = vec![I::Field::zero(); agg_param.prefixes.len()];
        for state in state2.into_iter() {
            let output_share = hits_finish(state, round2.clone())?;
            for (x, y) in output.iter_mut().zip(output_share.as_slice()) {
                *x += *y;
            }
        }

        if let Some(want) = expected_output {
            if want != output {
                return Err(VdafError::Uncategorized(format!(
                    "eval_vdaf(): unexpected output: got {:?}; want {:?}",
                    output, want
                )));
            }
        }

        Ok(())
    }
}
