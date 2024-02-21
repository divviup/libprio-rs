// SPDX-License-Identifier: MPL-2.0

//! Implementations of VIDPF specified in [[draft-mouris-cfrg-mastic]].
//!
//! [draft-mouris-cfrg-mastic]: https://datatracker.ietf.org/doc/draft-mouris-cfrg-mastic/01/

use std::{
    borrow::Borrow,
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

/// Params defines the global parameters of a VIDPF instance.
pub struct Params<P, C>
where
    // Primitive to construct a pseudorandom generator.
    P: GetPRG,
    // Type used for Codomain of a DPF.
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>,
{
    // Key size in bytes.
    key_size: usize,
    // Proof size in bytes.
    proof_size: usize,
    // Seed size in bytes.
    seed_size: usize,
    // Constructor of a pseudorandom generator.
    get_prg: P,
    // Used to cryptographically bind some information.
    bind_info: Vec<u8>,

    _cd: PhantomData<C>,
}

impl<P: GetPRG, C> Params<P, C>
where
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>,
{
    /// new
    pub fn new(sec_param: usize, get_prg: P, bind_info: Vec<u8>) -> Self {
        Self {
            key_size: sec_param / 8,
            seed_size: sec_param / 8,
            proof_size: 2 * (sec_param / 8),
            get_prg,
            bind_info,
            _cd: PhantomData,
        }
    }

    /// gen
    pub fn gen(&self, alpha: &[u8], beta: &C) -> Result<(Public<C>, Key, Key), Box<dyn Error>> {
        let n = 8 * alpha.len();

        // Key Generation.
        let k0 = Key::gen_det(0xA, ServerID::S0, self.key_size)?;
        let k1 = Key::gen_det(0xA, ServerID::S1, self.key_size)?;

        let mut s_i = [Seed::from(&k0), Seed::from(&k1)];
        let mut t_i = [ControlBit::from(&k0), ControlBit::from(&k1)];

        let mut cw = Vec::with_capacity(n);
        let mut cs = Vec::with_capacity(n);

        for i in 0..n {
            let alpha_i = ControlBit::from((alpha[i / 8] >> (i % 8)) & 0x1);

            let Sequence(sl_0, tl_0, sr_0, tr_0) = self.prg(&s_i[0]);
            let Sequence(sl_1, tl_1, sr_1, tr_1) = self.prg(&s_i[1]);

            let s = [[&sl_0, &sl_1], [&sr_0, &sr_1]];
            let t = [[tl_0, tl_1], [tr_0, tr_1]];

            const L: u8 = 0;
            const R: u8 = 1;

            let diff = u8::conditional_select(&L, &R, alpha_i.0) as usize;
            let same = u8::conditional_select(&R, &L, alpha_i.0) as usize;

            let s_cw = s[same][0] ^ s[same][1];
            let t_cw_l = tl_0 ^ tl_1 ^ alpha_i ^ ControlBit::from(1);
            let t_cw_r = tr_0 ^ tr_1 ^ alpha_i;
            let t_cw = [t_cw_l, t_cw_r];

            let s_tilde_i_0 = s[diff][0] ^ (t_i[0] & s_cw.borrow()).borrow();
            let s_tilde_i_1 = s[diff][1] ^ (t_i[1] & s_cw.borrow()).borrow();

            t_i[0] = t[diff][0] ^ (t_i[0] & t_cw[diff]);
            t_i[1] = t[diff][1] ^ (t_i[1] & t_cw[diff]);

            let w_i0: C;
            let w_i1: C;
            (s_i[0], w_i0) = self.convert(&s_tilde_i_0);
            (s_i[1], w_i1) = self.convert(&s_tilde_i_1);

            let mut w_cw = (beta - w_i0.borrow()).borrow() + w_i1.borrow();
            w_cw.conditional_negate(t_i[1].0);

            let cw_i = CorrectionWord {
                s_cw,
                t_cw_l,
                t_cw_r,
                w_cw,
            };

            let pi_i = self.gen_proof(alpha, i, &s_i);

            cw.push(cw_i);
            cs.push(pi_i);
        }

        Ok((Public { cw, cs }, k0, k1))
    }

    fn gen_proof(&self, alpha: &[u8], level: usize, s_i: &[Seed; 2]) -> Proof {
        let pi_0 = &self.hash_one(alpha, level, &s_i[0]);
        let pi_1 = &self.hash_one(alpha, level, &s_i[1]);
        pi_0 ^ pi_1
    }

    fn eval_next(
        &self,
        b: ServerID,
        alpha_i: ControlBit,
        s_i: &mut Seed,
        t_i: &mut ControlBit,
        cw: &CorrectionWord<C>,
    ) -> C {
        let CorrectionWord {
            s_cw,
            t_cw_l,
            t_cw_r,
            w_cw,
        } = cw;

        let seq_tilde = &self.prg(s_i);
        let seq_cw = &Sequence(s_cw.clone(), *t_cw_l, s_cw.clone(), *t_cw_r);
        let Sequence(sl, tl, sr, tr) = seq_tilde ^ (*t_i & seq_cw).borrow();

        let s_tilde_i = Seed::conditional_select(&sl, &sr, alpha_i.0);
        *t_i = ControlBit::conditional_select(&tl, &tr, alpha_i.0);

        let w_i: C;
        (*s_i, w_i) = self.convert(&s_tilde_i);

        let mut y_i = w_i.borrow() + (w_cw * (*t_i)).borrow();
        y_i.conditional_negate(Choice::from(b as u8));

        y_i
    }

    fn proof_next(
        &self,
        pi: &Proof,
        alpha: &[u8],
        level: usize,
        si: &Seed,
        ti: ControlBit,
        cs: &Proof,
    ) -> Proof {
        let pi_tilde = &self.hash_one(alpha, level, si);
        let h2_input = pi ^ (pi_tilde ^ (ti & cs).borrow()).borrow();
        let out_pi = pi ^ self.hash_two(&h2_input).borrow();
        out_pi
    }

    /// eval
    pub fn eval(&self, alpha: &[u8], key: &Key, public: &Public<C>) -> Share<C> {
        assert!(key.value.len() == self.key_size, "bad key size");

        let n = 8 * alpha.len();
        assert!(public.cw.len() >= n, "bad public key size");
        assert!(public.cs.len() >= n, "bad public key size");

        let mut s_i = Seed::from(key);
        let mut t_i = ControlBit::from(key);

        let mut y = C::new();
        let mut pi = Proof::new(self.proof_size);

        for i in 0..n {
            let alpha_i = ControlBit::from((alpha[i / 8] >> (i % 8)) & 0x1);
            y = self.eval_next(key.id, alpha_i, &mut s_i, &mut t_i, &public.cw[i]);
            pi = self.proof_next(&pi, alpha, i, &s_i, t_i, &public.cs[i]);
        }

        Share(y, pi)
    }

    /// verify checks that the proofs are equal and that the shares of beta add up to beta.
    pub fn verify(&self, a: &Share<C>, b: &Share<C>, beta: &C) -> bool {
        a.0.borrow() + b.0.borrow() == *beta && a.1 .0 == b.1 .0
    }

    fn prg(&self, seed: &Seed) -> Sequence {
        let dst = "100".as_bytes();
        let mut sl = Seed::new(self.seed_size);
        let mut sr = Seed::new(self.seed_size);
        let mut tl = ControlBit::from(0);
        let mut tr = ControlBit::from(0);
        let mut prg = self.get_prg.get(seed, dst, &self.bind_info);
        sl.fill(&mut prg);
        tl.fill(&mut prg);
        sr.fill(&mut prg);
        tr.fill(&mut prg);

        Sequence(sl, tl, sr, tr)
    }

    fn convert(&self, seed: &Seed) -> (Seed, C) {
        let dst = "101".as_bytes();
        let mut out_seed = Seed::new(self.seed_size);
        let mut value = C::new();
        let mut prg = self.get_prg.get(seed, dst, &self.bind_info);
        out_seed.fill(&mut prg);
        value.fill(&mut prg);

        (out_seed, value)
    }

    fn hash_one(&self, alpha: &[u8], level: usize, seed: &Seed) -> Proof {
        let dst = "vidpf cs proof".as_bytes();
        let mut binder = Vec::new();
        binder.extend_from_slice(alpha);
        binder.extend(level.to_le_bytes());

        let mut proof = Proof::new(self.proof_size);
        let mut prg = self.get_prg.get(seed, dst, &binder);
        proof.fill(&mut prg);

        proof
    }

    fn hash_two(&self, proof: &Proof) -> Proof {
        let seed = Seed::new(self.seed_size);
        let dst = "vidpf proof adjustment".as_bytes();
        let binder = &proof.0;

        let mut out_proof = Proof::new(self.proof_size);
        let mut prg = self.get_prg.get(&seed, dst, binder);
        out_proof.fill(&mut prg);

        out_proof
    }
}

#[derive(Debug)]
/// Share
pub struct Share<C>(C, Proof)
where
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>;

/// Fill
pub trait Fill {
    /// fill
    fn fill(&mut self, r: &mut impl RngCore);
}

/// CodomainOps
pub trait CodomainOps<C: ?Sized>:
    Sized + Add<Output = C> + Sub<Output = C> + Mul<ControlBit, Output = C>
{
}

/// Codomain
pub trait Codomain: Fill + PartialEq + ConditionallyNegatable
where
    for<'a> &'a Self: CodomainOps<Self>,
{
    /// new
    fn new() -> Self;
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
/// CorrectionWord
pub struct CorrectionWord<C>
where
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>,
{
    /// s_cw
    s_cw: Seed,
    /// t_cw_l
    t_cw_l: ControlBit,
    /// t_cw_r
    t_cw_r: ControlBit,
    /// w_cw
    w_cw: C,
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
pub struct Public<C>
where
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>,
{
    cw: Vec<CorrectionWord<C>>,
    cs: Vec<Proof>,
}

#[derive(Debug)]
/// Key is the server's private key.
pub struct Key {
    id: ServerID,
    value: Vec<u8>,
}

impl Key {
    /// gen
    pub fn gen(id: ServerID, n: usize) -> Result<Self, Box<dyn Error>> {
        let mut value = vec![0u8; n];
        getrandom::getrandom(&mut value)?;
        Ok(Key { id, value })
    }
    /// gen_det
    pub fn gen_det(t: u8, id: ServerID, n: usize) -> Result<Self, Box<dyn Error>> {
        let value = vec![t + id as u8; n];
        Ok(Key { id, value })
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

#[derive(Debug, Clone, Copy)]
/// ServerID
pub enum ServerID {
    /// S0
    S0 = 0,
    /// S1
    S1 = 1,
}

#[derive(Debug, Clone, Copy)]
/// ControlBit
pub struct ControlBit(Choice);

impl From<u8> for ControlBit {
    fn from(b: u8) -> Self {
        ControlBit(Choice::from(b))
    }
}

impl ConditionallySelectable for ControlBit {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        ControlBit((a.0 & !choice) | (b.0 & choice))
    }
}

impl From<&Key> for ControlBit {
    fn from(k: &Key) -> Self {
        ControlBit::from(k.id as u8)
    }
}

impl Fill for ControlBit {
    fn fill(&mut self, r: &mut impl RngCore) {
        let mut b = [0u8; 1];
        r.fill_bytes(&mut b);
        *self = ControlBit::from(b[0] & 0x1)
    }
}

impl BitAnd<&Seed> for ControlBit {
    type Output = Seed;
    fn bitand(self, rhs: &Seed) -> Self::Output {
        Seed(
            rhs.0
                .iter()
                .map(|x| u8::conditional_select(&0, x, self.0))
                .collect(),
        )
    }
}

impl BitAnd<&Proof> for ControlBit {
    type Output = Proof;
    fn bitand(self, rhs: &Proof) -> Self::Output {
        Proof(
            rhs.0
                .iter()
                .map(|x| u8::conditional_select(&0, x, self.0))
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

impl BitAnd for ControlBit {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        ControlBit(self.0 & rhs.0)
    }
}

impl BitXor for ControlBit {
    type Output = Self;

    #[inline]
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

    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Seed(
            zip(&a.0, &b.0)
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect(),
        )
    }
}

impl From<&Key> for Seed {
    fn from(k: &Key) -> Self {
        Seed(k.value.clone())
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
pub struct Weight<F: FieldElement, const N: usize>(pub [F; N]);

impl<F: FieldElement, const N: usize> Weight<F, N> {}

impl<'a, F: FieldElement, const N: usize> CodomainOps<Weight<F, N>> for &'a Weight<F, N> {}

impl<F: FieldElement, const N: usize> Codomain for Weight<F, N> {
    fn new() -> Self {
        Self([F::zero(); N])
    }
}

impl<F: FieldElement, const N: usize> Fill for Weight<F, N> {
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

impl<F: FieldElement, const N: usize> ConditionallyNegatable for Weight<F, N> {
    fn conditional_negate(&mut self, choice: Choice) {
        self.0.iter_mut().for_each(|a| a.conditional_negate(choice));
    }
}

impl<'a, F: FieldElement, const N: usize> Add for &'a Weight<F, N> {
    type Output = Weight<F, N>;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.0.len() == rhs.0.len(), "Weight add");
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, zip(&self.0, &rhs.0)).for_each(|(c, (a, b))| *c = *a + *b);
        out
    }
}

impl<'a, F: FieldElement, const N: usize> Sub for &'a Weight<F, N> {
    type Output = Weight<F, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.0.len() == rhs.0.len(), "Weight sub");
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, zip(&self.0, &rhs.0)).for_each(|(c, (a, b))| *c = *a - *b);
        out
    }
}

impl<'a, F: FieldElement, const N: usize> Mul<ControlBit> for &'a Weight<F, N> {
    type Output = Weight<F, N>;

    fn mul(self, bit: ControlBit) -> Self::Output {
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, &self.0)
            .for_each(|(a, b)| *a = F::conditional_select(&F::zero(), b, bit.0));
        out
    }
}
impl<F: FieldElement, const N: usize> PartialEq for Weight<F, N> {
    fn eq(&self, rhs: &Self) -> bool {
        self.0.len() == rhs.0.len() && zip(&self.0, &rhs.0).all(|(a, b)| a == b)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        field::Field128,
        vdaf::{
            vidpf::{Codomain, PrngFromXof, Weight},
            xof::XofTurboShake128,
        },
    };

    use super::Params;

    #[test]
    fn devel() {
        const SEC_PARAM: usize = 128;
        let prng = PrngFromXof::<16, XofTurboShake128>::default();
        let alpha = "1".as_bytes();
        let beta = Weight::<Field128, 3>([21.into(), 22.into(), 23.into()]);
        let binder = "protocol using a vidpf".as_bytes().to_vec();
        println!("alpha: {:?}", alpha);
        println!("beta: {:?}", beta);

        let vidpf: Params<PrngFromXof<16, XofTurboShake128>, Weight<Field128, 3>> =
            Params::new(SEC_PARAM, prng, binder);
        let Ok((public, k0, k1)) = vidpf.gen(alpha, &beta) else {
            panic!("error: gen");
        };

        println!("public: {:?}", public);
        println!("key0: {:?}", k0);
        println!("key1: {:?}", k1);

        let share0 = vidpf.eval(alpha, &k0, &public);
        let share1 = vidpf.eval(alpha, &k1, &public);

        // println!(
        //     "aux: {:?}",
        //     zip(aux0.0, aux1.0)
        //         .map(|(a, b)| &a + &b)
        //         .collect::<Vec<Weight<Field128>>>()
        // );

        println!("y0: {:?}", share0.0);
        println!("y1: {:?}", share1.0);
        println!("y: {:?}", &share0.0 + &share1.0);
        println!("y_ok: {}", beta == &share0.0 + &share1.0);
        println!("p0: {:?}", share0.1);
        println!("p1: {:?}", share1.1);
        println!("p: {:?}", share0.1 .0 == share1.1 .0);
        println!("verify: {:?}", vidpf.verify(&share0, &share1, &beta));
    }

    fn setup() -> Params<PrngFromXof<16, XofTurboShake128>, Weight<Field128, 3>> {
        let prng = PrngFromXof::<16, XofTurboShake128>::default();
        const SEC_PARAM: usize = 128;
        let binder = "Mock Protocol uses a VIDPF".as_bytes().to_vec();
        Params::new(SEC_PARAM, prng, binder)
    }

    #[test]
    fn happy_path() {
        let vidpf = setup();
        let alpha: &[u8] = &[0xF];
        let beta = Weight([21.into(), 22.into(), 23.into()]);

        let (public, k0, k1) = vidpf.gen(alpha, &beta).unwrap();
        let share0 = vidpf.eval(alpha, &k0, &public);
        let share1 = vidpf.eval(alpha, &k1, &public);

        assert!(
            vidpf.verify(&share0, &share1, &beta) == true,
            "verification failed"
        )
    }

    #[test]
    fn bad_query() {
        let vidpf = setup();
        let alpha: &[u8] = &[0xF];
        let alpha_bad: &[u8] = &[0x0];
        let beta = Weight([21.into(), 22.into(), 23.into()]);

        let (public, k0, k1) = vidpf.gen(alpha, &beta).unwrap();
        let share0 = vidpf.eval(alpha_bad, &k0, &public);
        let share1 = vidpf.eval(alpha_bad, &k1, &public);

        assert!(
            &share0.0 + &share1.0 == Weight::new(),
            "shares must add up to zero"
        );
        assert!(
            vidpf.verify(&share0, &share1, &beta) == false,
            "verification passed, but it should failed"
        )
    }

    #[cfg(feature = "count-allocations")]
    #[test]
    pub fn mem_alloc() {
        let vidpf = setup();
        let alpha: &[u8] = &[0xF];
        let beta = Weight([21.into(), 22.into(), 23.into()]);
        // Gen:  AllocationInfo { count_total: 166, count_current: 0, count_max: 29, bytes_total: 3712, bytes_current: 0, bytes_max: 1376 }
        // Eval: AllocationInfo { count_total: 154, count_current: 0, count_max: 10, bytes_total: 3184, bytes_current: 0, bytes_max: 176 }

        println!(
            "Gen:  {:?}",
            allocation_counter::measure(|| {
                let _ = vidpf.gen(alpha, &beta).unwrap();
            })
        );

        let (public, k0, _) = vidpf.gen(alpha, &beta).unwrap();
        println!(
            "Eval: {:?}",
            allocation_counter::measure(|| {
                let _ = vidpf.eval(alpha, &k0, &public);
            })
        )
    }
}
