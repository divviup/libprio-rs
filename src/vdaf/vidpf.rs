// SPDX-License-Identifier: MPL-2.0

//! Implementations of VIDPF specified in [[draft-mouris-cfrg-mastic]].
//!
//! [draft-mouris-cfrg-mastic]: https://datatracker.ietf.org/doc/draft-mouris-cfrg-mastic/01/

use std::{
    error::Error,
    iter::zip,
    marker::PhantomData,
    ops::{Add, BitAnd, BitXor, ControlFlow, Sub},
};

use rand_core::RngCore;
use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable};

use super::xof::Xof;
use crate::{
    codec::Decode, field::FieldElement, field::FieldElementExt, vdaf::xof::Seed as XofSeed,
};

/// VERSION is a tag.
pub static VERSION: &str = "MyVIDPF";

/// Params defines the global parameters of VIDPF instance.
pub struct Params<
    // Field for operations.
    F: FieldElement,
    // Primitive to construct a pseudorandom generator.
    P: GetPRG,
> {
    // Number of bits for alpha.
    n: usize,
    // Dimension of weights.
    m: usize,
    // Security parameter in bytes.
    k: usize,
    // Constructor of a pseudorandom generator.
    get_prg: P,

    _pd: PhantomData<F>,
}

impl<F: FieldElement, P: GetPRG> Params<F, P> {
    /// new
    pub fn new(sec_param: usize, n: usize, m: usize, get_prg: P) -> Self {
        Self {
            n,
            m,
            k: sec_param / 8,
            get_prg,
            _pd: PhantomData::<F>,
        }
    }

    /// gen
    pub fn gen(
        &self,
        alpha: &[u8],
        beta: &Weight<F>,
        binder: &[u8],
    ) -> Result<(Public<F>, Key, Key), Box<dyn Error>> {
        assert!(alpha.len() == (self.n + 7) / 8, "bad alpha size");
        assert!(beta.0.len() == self.m, "bad beta size");

        // Key Generation.
        let k0 = Key::gen(self.k)?;
        let k1 = Key::gen(self.k)?;

        let mut cw = Vec::with_capacity(self.n);
        let mut cs = Vec::with_capacity(self.n);

        const L: usize = 0;
        const R: usize = 1;

        let mut t_i = [ControlBit(0), ControlBit(1)];
        let mut s_i = [Seed::from(&k0), Seed::from(&k1)];
        for i in 0..self.n {
            let val = [self.prg(&s_i[0], binder), self.prg(&s_i[1], binder)];
            let sl = [&val[0].0, &val[1].0];
            let tl = [val[0].1, val[1].1];
            let sr = [&val[0].2, &val[1].2];
            let tr = [val[0].3, val[1].3];

            let s = [sl, sr];
            let t = [tl, tr];

            let alpha_i = ControlBit((alpha[i / 8] >> (i % 8)) & 0x1);
            let (diff, same) = if alpha_i == ControlBit(0) {
                (L, R)
            } else {
                (R, L)
            };

            let s_cw = s[same][0] ^ s[same][1];
            let t_cw_l = tl[0] ^ tl[1] ^ alpha_i ^ ControlBit(1);
            let t_cw_r = tr[0] ^ tr[1] ^ alpha_i;
            let t_cw = [t_cw_l, t_cw_r];

            let s_tilde_i = [s[diff][0] ^ (&s_cw, t_i[0]), s[diff][1] ^ (&s_cw, t_i[1])];

            t_i[0] = t[diff][0] ^ (t_i[0] & t_cw[diff]);
            t_i[1] = t[diff][1] ^ (t_i[1] & t_cw[diff]);

            let mut w_i = [
                Weight(vec![F::zero(); self.m]),
                Weight(vec![F::zero(); self.m]),
            ];

            (s_i[0], w_i[0]) = self.convert(&s_tilde_i[0], binder);
            (s_i[1], w_i[1]) = self.convert(&s_tilde_i[1], binder);

            let mut w_cw = beta - &(&w_i[0] + &w_i[1]);
            (&mut w_cw).conditional_negate(t_i[1].into());

            cw.push(CorrWord {
                s_cw,
                t_cw_l,
                t_cw_r,
                w_cw,
            });

            let pi_0 = self.hash_one(alpha, i, &s_i[0]);
            let pi_1 = self.hash_one(alpha, i, &s_i[1]);
            let pi = &pi_0 ^ &pi_1;
            cs.push(pi);
        }

        Ok((Public { cw, cs }, k0, k1))
    }

    fn prg(&self, seed: &Seed, binder: &[u8]) -> Sequence {
        let dst = "100".as_bytes();
        let mut sl = Seed::new(self.k);
        let mut sr = Seed::new(self.k);
        let mut tl = ControlBit(0);
        let mut tr = ControlBit(0);
        let mut prg = self.get_prg.get(seed, dst, binder);
        sl.fill(&mut prg);
        sr.fill(&mut prg);
        tl.fill(&mut prg);
        tr.fill(&mut prg);

        Sequence(sl, tl, sr, tr)
    }

    fn convert(&self, seed: &Seed, binder: &[u8]) -> (Seed, Weight<F>) {
        let dst = "101".as_bytes();
        let mut out_seed = Seed::new(self.k);
        let mut weight = Weight::new(self.m);
        let mut prg = self.get_prg.get(seed, dst, binder);
        out_seed.fill(&mut prg);
        weight.fill(&mut prg);

        (out_seed, weight)
    }

    fn hash_one(&self, alpha: &[u8], level: usize, seed: &Seed) -> Proof {
        let dst = "vidpf cs proof".as_bytes();
        let mut binder = Vec::new();
        binder.extend(self.n.to_le_bytes());
        binder.extend_from_slice(alpha);
        binder.extend(level.to_le_bytes());

        let mut proof = Proof::new(2 * self.k);
        let mut prg = self.get_prg.get(seed, dst, &binder);
        proof.fill(&mut prg);

        proof
    }
}

/// Sequence
pub struct Sequence(Seed, ControlBit, Seed, ControlBit);

/// Fill
pub trait Fill {
    /// fill
    fn fill(&mut self, r: &mut impl RngCore);
}

/// GetPRG
pub trait GetPRG {
    /// get
    fn get(&self, seed: &Seed, dst: &[u8], binder: &[u8]) -> impl RngCore;
}

struct PrngFromXof<const SEED_SIZE: usize, X: Xof<SEED_SIZE>>(PhantomData<X>);

impl<const SEED_SIZE: usize, X: Xof<SEED_SIZE>> Default for PrngFromXof<SEED_SIZE, X> {
    fn default() -> Self {
        Self(PhantomData::<X>)
    }
}

impl<const SEED_SIZE: usize, X: Xof<SEED_SIZE>> GetPRG for PrngFromXof<SEED_SIZE, X> {
    fn get(&self, seed: &Seed, dst: &[u8], binder: &[u8]) -> impl RngCore {
        let xof_seed = XofSeed::get_decoded(&seed.0).unwrap();
        X::seed_stream(&xof_seed, dst, binder)
    }
}

#[derive(Debug)]
/// CorrWords
pub struct CorrWord<F: FieldElement> {
    /// s_cw
    s_cw: Seed,
    /// t_cw_l
    t_cw_l: ControlBit,
    /// t_cw_r
    t_cw_r: ControlBit,
    /// w_cw
    w_cw: Weight<F>,
}

#[derive(Debug)]
/// Proof
pub struct Proof(Vec<u8>);

impl Proof {
    /// new
    pub fn new(n: usize) -> Self {
        Self(vec![0; n])
    }
}

impl Fill for Proof {
    fn fill(&mut self, r: &mut impl RngCore) {
        r.fill_bytes(&mut self.0)
    }
}

impl<'a> BitXor<&'a Proof> for &'a Proof {
    type Output = Proof;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Proof(zip(&self.0, &rhs.0).map(|(a, b)| *a ^ *b).collect())
    }
}

#[derive(Debug)]
/// Public information for aggregation.
pub struct Public<F: FieldElement> {
    cw: Vec<CorrWord<F>>,
    cs: Vec<Proof>,
}

#[derive(Debug)]
/// Key is the server's private key.
pub struct Key(Vec<u8>);

impl Key {
    /// gen
    pub fn gen(n: usize) -> Result<Self, Box<dyn Error>> {
        let mut k = vec![0u8; n];
        getrandom::getrandom(&mut k)?;
        Ok(Key(k))
    }
}

impl From<&Key> for Seed {
    fn from(k: &Key) -> Self {
        Seed(k.0.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// ControlBit
pub struct ControlBit(u8);

impl From<ControlBit> for Choice {
    fn from(value: ControlBit) -> Self {
        Choice::from(value.0)
    }
}

impl Fill for ControlBit {
    fn fill(&mut self, r: &mut impl RngCore) {
        let mut b = [0u8; 1];
        r.fill_bytes(&mut b);
        *self = ControlBit(b[0] & 0x1);
    }
}

impl BitAnd<ControlBit> for ControlBit {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        ControlBit(self.0 & rhs.0)
    }
}

impl BitXor<ControlBit> for ControlBit {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        ControlBit(self.0 ^ rhs.0)
    }
}

#[derive(Debug)]
/// Seed
pub struct Seed(Vec<u8>);

impl Seed {
    /// new
    pub fn new(n: usize) -> Self {
        Self(vec![0; n])
    }
}

impl Fill for Seed {
    fn fill(&mut self, r: &mut impl RngCore) {
        r.fill_bytes(&mut self.0)
    }
}

impl<'a> BitXor<&'a Seed> for &'a Seed {
    type Output = Seed;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Seed(zip(&self.0, &rhs.0).map(|(a, b)| *a ^ *b).collect())
    }
}

impl<'a> BitXor<(&'a Seed, ControlBit)> for &'a Seed {
    type Output = Seed;

    fn bitxor(self, (rhs, bit): (&'a Seed, ControlBit)) -> Self::Output {
        Seed(
            zip(&self.0, &rhs.0)
                .map(|(a, b)| u8::conditional_select(&(*a ^ *b), b, bit.into()))
                .collect(),
        )
    }
}

#[derive(Debug)]
/// Weight
pub struct Weight<F: FieldElement>(Vec<F>);

impl<F: FieldElement> Weight<F> {
    /// new
    pub fn new(n: usize) -> Self {
        Self(vec![F::zero(); n])
    }
}

impl<F: FieldElement> Fill for Weight<F> {
    fn fill(&mut self, r: &mut impl RngCore) {
        let mut bytes = vec![0u8; F::ENCODED_SIZE];
        self.0.iter_mut().for_each(|i| loop {
            r.fill_bytes(&mut bytes);
            if let ControlFlow::Break(x) = F::from_random_rejection(&bytes) {
                *i = x;
                break;
            }
        });
    }
}

impl<F: FieldElement> ConditionallyNegatable for &mut Weight<F> {
    fn conditional_negate(&mut self, choice: Choice) {
        self.0.iter_mut().for_each(|a| a.conditional_negate(choice));
    }
}

impl<'a, F: FieldElement> Add<&'a Weight<F>> for &'a Weight<F> {
    type Output = Weight<F>;

    fn add(self, rhs: Self) -> Self::Output {
        Weight(zip(&self.0, &rhs.0).map(|(a, b)| *a + *b).collect())
    }
}

impl<'a, F: FieldElement> Sub<&'a Weight<F>> for &'a Weight<F> {
    type Output = Weight<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        Weight(zip(&self.0, &rhs.0).map(|(a, b)| *a - *b).collect())
    }
}

//
//
//
//
//
//
//
//

#[cfg(test)]
mod tests {
    use crate::{
        field::Field128,
        vdaf::{
            vidpf::{PrngFromXof, Weight},
            xof::XofTurboShake128,
        },
    };

    use super::{Params, VERSION};

    #[test]
    fn devel() {
        const SEC_PARAM: usize = 128;
        let n: usize = 2;
        let m: usize = 3;
        let prng = PrngFromXof::<16, XofTurboShake128>::default();
        let alpha = "1".as_bytes();
        let beta = Weight::<Field128>([21.into(), 22.into(), 23.into()].into());
        let binder = "protocol using a vidpf".as_bytes();

        let vidpf = Params::new(SEC_PARAM, n, m, prng);
        match vidpf.gen(&alpha, &beta, &binder) {
            Ok((pp, k0, k1)) => {
                println!("public: {:?}", pp);
                println!("key0: {:?}", k0);
                println!("key1: {:?}", k1);
            }
            Err(e) => println!("error: {:?}", e),
        };
    }

    #[test]
    fn version() {
        assert_eq!(VERSION, "MyVIDPF")
    }
}
