// SPDX-License-Identifier: MPL-2.0

//! Implementations of VIDPF specified in [[draft-mouris-cfrg-mastic]].
//!
//! [draft-mouris-cfrg-mastic]: https://datatracker.ietf.org/doc/draft-mouris-cfrg-mastic/01/

use std::{
    borrow::{Borrow, BorrowMut},
    error::Error,
    iter::zip,
    marker::PhantomData,
    ops::{Add, BitAnd, BitXor, ControlFlow, Mul, Sub},
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
        let k0 = Key::gen_det(0xA, ControlBit(0), self.k)?;
        let k1 = Key::gen_det(0xA, ControlBit(1), self.k)?;

        let s_i = [&mut Seed::from(&k0), &mut Seed::from(&k1)];
        let t_i = &mut [ControlBit(0), ControlBit(1)];
        let w_i = [&mut Weight::new(self.m), &mut Weight::new(self.m)];

        let mut cw = Vec::with_capacity(self.n);
        let mut cs = Vec::with_capacity(self.n);

        for i in 0..self.n {
            let Sequence(sl_0, tl_0, sr_0, tr_0) = self.prg(s_i[0], binder);
            let Sequence(sl_1, tl_1, sr_1, tr_1) = self.prg(s_i[1], binder);

            let s = [[&sl_0, &sl_1], [&sr_0, &sr_1]];
            let t = [[tl_0, tl_1], [tr_0, tr_1]];

            const L: usize = 0;
            const R: usize = 1;

            let alpha_i = ControlBit((alpha[i / 8] >> (i % 8)) & 0x1);
            let (diff, same) = if alpha_i == ControlBit(0) {
                (L, R)
            } else {
                (R, L)
            };

            let s_cw = s[same][0] ^ s[same][1];
            let t_cw_l = tl_0 ^ tl_1 ^ alpha_i ^ ControlBit(1);
            let t_cw_r = tr_0 ^ tr_1 ^ alpha_i;
            let t_cw = [t_cw_l, t_cw_r];

            let s_tilde_i_0 = s[diff][0] ^ (t_i[0] & s_cw.borrow()).borrow();
            let s_tilde_i_1 = s[diff][1] ^ (t_i[1] & s_cw.borrow()).borrow();

            t_i[0] = t[diff][0] ^ (t_i[0] & t_cw[diff]);
            t_i[1] = t[diff][1] ^ (t_i[1] & t_cw[diff]);

            (*s_i[0], *w_i[0]) = self.convert(&s_tilde_i_0, binder);
            (*s_i[1], *w_i[1]) = self.convert(&s_tilde_i_1, binder);

            let mut w_cw = (beta - w_i[0]).borrow() + w_i[1];
            w_cw.borrow_mut().conditional_negate(t_i[1].into());

            cw.push(CorrWord {
                s_cw,
                t_cw_l,
                t_cw_r,
                w_cw,
            });

            let pi = [
                &self.hash_one(alpha, i, s_i[0]),
                &self.hash_one(alpha, i, s_i[1]),
            ];
            cs.push(pi[0] ^ pi[1]);
        }

        Ok((Public { cw, cs }, k0, k1))
    }

    fn eval_next(
        &self,
        b: ControlBit,
        public: &Public<F>,
        i: usize,
        s_i_1: Seed,
        t_i_1: ControlBit,
        pi: Proof,
        alpha: &[u8],
        binder: &[u8],
    ) -> (Seed, ControlBit, Weight<F>, Proof) {
        const L: usize = 0;
        const R: usize = 1;

        let s_cw = &public.cw[i].s_cw;
        let t_cw_l = public.cw[i].t_cw_l;
        let t_cw_r = public.cw[i].t_cw_r;
        let w_cw = &public.cw[i].w_cw;
        let cs = &public.cs[i];

        let seq_tilde = &self.prg(&s_i_1, binder);
        let seq_cw = Sequence(s_cw.clone(), t_cw_l, s_cw.clone(), t_cw_r);
        let tau = seq_tilde ^ (t_i_1 & seq_cw.borrow()).borrow();
        let Sequence(sl, tl, sr, tr) = tau;

        let s = [&sl, &sr];
        let t = [tl, tr];

        let alpha_i = ControlBit((alpha[i / 8] >> (i % 8)) & 0x1);
        let (s_tilde_i, t_i) = if alpha_i == ControlBit(0) {
            (s[L], t[L])
        } else {
            (s[R], t[R])
        };

        let (s_i, w_i) = self.convert(s_tilde_i, binder);

        let mut y_i = w_i.borrow() + (t_i * w_cw).borrow();
        y_i.borrow_mut().conditional_negate(b.into());

        let pi_tilde = &self.hash_one(alpha, i, &s_i);
        let h2_input = pi.borrow() ^ (pi_tilde ^ (t_i & cs).borrow()).borrow();
        let out_pi = pi.borrow() ^ self.hash_two(&h2_input).borrow();

        (s_i, t_i, y_i, out_pi)
    }

    /// eval
    pub fn eval(
        &self,
        alpha: &[u8],
        key: &Key,
        public: &Public<F>,
        binder: &[u8],
    ) -> (Share<F>, Aux<F>) {
        assert!(alpha.len() == (self.n + 7) / 8, "bad alpha size");
        assert!(key.1.len() == self.k, "bad key size");
        assert!(public.cw.len() == self.n, "bad public key size");
        assert!(public.cs.len() == self.n, "bad public key size");

        let b = key.0;
        let mut s_i_1 = Seed::from(key);
        let mut t_i_1 = b;
        let mut pi_i_1 = Proof::new(2 * self.k);

        let mut list_y = Vec::<Weight<F>>::with_capacity(self.n);

        for i in 0..self.n {
            let (s_i, t_i, y, pi_i) =
                self.eval_next(b, public, i, s_i_1, t_i_1, pi_i_1, alpha, binder);
            s_i_1 = s_i;
            t_i_1 = t_i;
            pi_i_1 = pi_i;
            println!("b: {:?} y_i: {:?}", b, &y);
            list_y.push(y);
        }
        let yy = list_y[self.n - 1].clone();
        println!("b: {:?} y_i: {:?}\n", b, yy);

        (Share(yy, pi_i_1), Aux(list_y))
    }

    /// verify checks that the proofs are equal and that the shares of beta add up to beta.
    pub fn verify(&self, a: &Share<F>, b: &Share<F>, beta: &Weight<F>) -> bool {
        &a.0 + &b.0 == *beta && a.1 .0 == b.1 .0
    }

    fn prg(&self, seed: &Seed, binder: &[u8]) -> Sequence {
        let dst = "100".as_bytes();
        let mut sl = Seed::new(self.k);
        let mut sr = Seed::new(self.k);
        let mut tl = ControlBit(0);
        let mut tr = ControlBit(0);
        let mut prg = self.get_prg.get(seed, dst, binder);
        sl.fill(&mut prg);
        tl.fill(&mut prg);
        sr.fill(&mut prg);
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

    fn hash_two(&self, proof: &Proof) -> Proof {
        let seed = Seed::new(self.k);
        let dst = "vidpf proof adjustment".as_bytes();
        let binder = &proof.0;

        let mut out_proof = Proof::new(2 * self.k);
        let mut prg = self.get_prg.get(&seed, dst, binder);
        out_proof.fill(&mut prg);

        out_proof
    }
}

/// Aux
pub struct Aux<F: FieldElement>(Vec<Weight<F>>);

/// Share
pub struct Share<F: FieldElement>(Weight<F>, Proof);

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

/// PrngFromXof is a helper to specify a Xof implementer.
pub struct PrngFromXof<const SEED_SIZE: usize, X: Xof<SEED_SIZE>>(PhantomData<X>);

impl<const SEED_SIZE: usize, X: Xof<SEED_SIZE>> Default for PrngFromXof<SEED_SIZE, X> {
    fn default() -> Self {
        Self(PhantomData::<X>)
    }
}

impl<const SEED_SIZE: usize, X: Xof<SEED_SIZE>> GetPRG for PrngFromXof<SEED_SIZE, X> {
    fn get(&self, seed: &Seed, dst: &[u8], binder: &[u8]) -> impl RngCore {
        let xof_seed = XofSeed::<SEED_SIZE>::get_decoded(&seed.0).unwrap();
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
pub struct Key(ControlBit, Vec<u8>);

impl Key {
    /// gen
    pub fn gen(id: ControlBit, n: usize) -> Result<Self, Box<dyn Error>> {
        let mut k = vec![0u8; n];
        getrandom::getrandom(&mut k)?;
        Ok(Key(id, k))
    }
    /// gen_det
    pub fn gen_det(t: u8, id: ControlBit, n: usize) -> Result<Self, Box<dyn Error>> {
        let k = vec![t + id.0; n];
        Ok(Key(id, k))
    }
}

impl From<&Key> for Seed {
    fn from(k: &Key) -> Self {
        Seed(k.1.clone())
    }
}

/// Sequence
pub struct Sequence(Seed, ControlBit, Seed, ControlBit);

impl<'a> BitXor<&'a Sequence> for &'a Sequence {
    type Output = Sequence;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Sequence(
            &self.0 ^ &rhs.0,
            self.1 ^ rhs.1,
            &self.2 ^ &rhs.2,
            self.3 ^ rhs.3,
        )
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

impl BitAnd<&Seed> for ControlBit {
    type Output = Seed;

    fn bitand(self, rhs: &Seed) -> Self::Output {
        Seed(
            rhs.0
                .iter()
                .map(|x| u8::conditional_select(&0, x, self.into()))
                .collect(),
        )
    }
}

impl BitAnd<&Sequence> for ControlBit {
    type Output = Sequence;

    fn bitand(self, rhs: &Sequence) -> Self::Output {
        Sequence(self & &rhs.0, self & rhs.1, self & &rhs.2, self & rhs.3)
    }
}

impl BitAnd<&Proof> for ControlBit {
    type Output = Proof;

    fn bitand(self, rhs: &Proof) -> Self::Output {
        Proof(
            rhs.0
                .iter()
                .map(|x| u8::conditional_select(&0, x, self.into()))
                .collect(),
        )
    }
}

impl<F: FieldElement> Mul<&Weight<F>> for ControlBit {
    type Output = Weight<F>;

    fn mul(self, rhs: &Weight<F>) -> Self::Output {
        Weight(
            rhs.0
                .iter()
                .map(|x| F::conditional_select(&F::zero(), x, self.into()))
                .collect(),
        )
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

impl<F: FieldElement> ConditionallyNegatable for Weight<F> {
    fn conditional_negate(&mut self, choice: Choice) {
        self.0.iter_mut().for_each(|a| a.conditional_negate(choice));
    }
}

impl<'a, F: FieldElement> Add<&'a Weight<F>> for &'a Weight<F> {
    type Output = Weight<F>;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.0.len() == rhs.0.len(), "weight add");
        Weight(zip(&self.0, &rhs.0).map(|(a, b)| *a + *b).collect())
    }
}

impl<'a, F: FieldElement> Sub<&'a Weight<F>> for &'a Weight<F> {
    type Output = Weight<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.0.len() == rhs.0.len(), "weight sub");
        Weight(zip(&self.0, &rhs.0).map(|(a, b)| *a - *b).collect())
    }
}

impl<F: FieldElement> PartialEq for Weight<F> {
    fn eq(&self, rhs: &Self) -> bool {
        self.0.len() == rhs.0.len() && zip(&self.0, &rhs.0).all(|(a, b)| a == b)
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
    use std::iter::zip;

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
        let n: usize = 3;
        let m: usize = 1;
        let prng = PrngFromXof::<16, XofTurboShake128>::default();
        let alpha = "1".as_bytes();
        let beta = Weight::<Field128>([21.into()].into());
        let binder = "protocol using a vidpf".as_bytes();
        println!("alpha: {:?}", alpha);
        println!("beta: {:?}", beta);

        let vidpf = Params::new(SEC_PARAM, n, m, prng);
        let Ok((public, k0, k1)) = vidpf.gen(alpha, &beta, binder) else {
            panic!("error: gen");
        };

        println!("public: {:?}", public);
        println!("key0: {:?}", k0);
        println!("key1: {:?}", k1);

        let (share0, aux0) = vidpf.eval(alpha, &k0, &public, binder);
        let (share1, aux1) = vidpf.eval(alpha, &k1, &public, binder);

        println!(
            "aux: {:?}",
            zip(aux0.0, aux1.0)
                .map(|(a, b)| &a + &b)
                .collect::<Vec<Weight<Field128>>>()
        );

        println!("y0: {:?}", share0.0);
        println!("y1: {:?}", share1.0);
        println!("y: {:?}", &share0.0 + &share1.0);
        println!("y_ok: {}", beta == &share0.0 + &share1.0);
        println!("p0: {:?}", share0.1);
        println!("p1: {:?}", share1.1);
        println!("p: {:?}", share0.1 .0 == share1.1 .0);
        println!("verify: {:?}", vidpf.verify(&share0, &share1, &beta));
    }

    #[test]
    fn version() {
        assert_eq!(VERSION, "MyVIDPF")
    }
}
