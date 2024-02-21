// SPDX-License-Identifier: MPL-2.0

//! Implementations of VIDPF specified in [[draft-mouris-cfrg-mastic]].
//!
//! [draft-mouris-cfrg-mastic]: https://datatracker.ietf.org/doc/draft-mouris-cfrg-mastic/01/

use std::{
    borrow::Borrow,
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

/// Params defines the global parameters of a VIDPF instance.
pub struct Params<P, C>
where
    // Primitive to construct a pseudorandom generator.
    P: GetPRNG,
    // Represents the codomain of a Distributed Point Function (DPF).
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>,
{
    // Key size in bytes.
    key_size: usize,
    // Proof size in bytes.
    proof_size: usize,
    // Seed size in bytes.
    seed_size: usize,
    // Constructor of a seeded pseudorandom generator.
    prng_ctor: P,
    // Used to cryptographically bind some information.
    bind_info: Vec<u8>,

    _cd: PhantomData<C>,
}

impl<P: GetPRNG, C> Params<P, C>
where
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>,
{
    /// new
    pub fn new(sec_param: usize, prng_ctor: P, bind_info: Vec<u8>) -> Self {
        Self {
            key_size: sec_param / 8,
            seed_size: sec_param / 8,
            proof_size: 2 * (sec_param / 8),
            prng_ctor,
            bind_info,
            _cd: PhantomData,
        }
    }

    /// gen
    pub fn gen(&self, alpha: &[u8], beta: &C) -> Result<(PublicKey<C>, Key, Key), Box<dyn Error>> {
        // Key Generation.
        let k0 = Key::gen(ServerID::S0, self.key_size)?;
        let k1 = Key::gen(ServerID::S1, self.key_size)?;

        let mut s_i = [Seed::from(&k0), Seed::from(&k1)];
        let mut t_i = [ControlBit::from(&k0), ControlBit::from(&k1)];

        let n = 8 * alpha.len();
        let mut cw = Vec::with_capacity(n);
        let mut cs = Vec::with_capacity(n);

        for i in 0..n {
            let alpha_i = ControlBit::from((alpha[i / 8] >> (i % 8)) & 0x1);

            let Sequence(sl_0, tl_0, sr_0, tr_0) = self.prg(&s_i[0]);
            let Sequence(sl_1, tl_1, sr_1, tr_1) = self.prg(&s_i[1]);

            let s_same_0 = &Seed::conditional_select(&sl_0, &sr_0, !alpha_i.0);
            let s_same_1 = &Seed::conditional_select(&sl_1, &sr_1, !alpha_i.0);

            let s_cw = s_same_0 ^ s_same_1;
            let t_cw_l = tl_0 ^ tl_1 ^ alpha_i ^ ControlBit::from(1);
            let t_cw_r = tr_0 ^ tr_1 ^ alpha_i;
            let t_cw_diff = ControlBit::conditional_select(&t_cw_l, &t_cw_r, alpha_i.0);

            let s_diff_0 = &Seed::conditional_select(&sl_0, &sr_0, alpha_i.0);
            let s_diff_1 = &Seed::conditional_select(&sl_1, &sr_1, alpha_i.0);
            let s_tilde_i_0 = s_diff_0 ^ (t_i[0] & s_cw.borrow()).borrow();
            let s_tilde_i_1 = s_diff_1 ^ (t_i[1] & s_cw.borrow()).borrow();

            let t_diff_0 = ControlBit::conditional_select(&tl_0, &tr_0, alpha_i.0);
            let t_diff_1 = ControlBit::conditional_select(&tl_1, &tr_1, alpha_i.0);
            t_i[0] = t_diff_0 ^ (t_i[0] & t_cw_diff);
            t_i[1] = t_diff_1 ^ (t_i[1] & t_cw_diff);

            let (w_i_0, w_i_1): (C, C);
            (s_i[0], w_i_0) = self.convert(&s_tilde_i_0);
            (s_i[1], w_i_1) = self.convert(&s_tilde_i_1);

            let mut w_cw = (beta - w_i_0.borrow()).borrow() + w_i_1.borrow();
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

        Ok((PublicKey { cw, cs }, k0, k1))
    }

    fn gen_proof(&self, alpha: &[u8], level: usize, s_i: &[Seed; 2]) -> Proof {
        let pi_0 = &self.hash_one(alpha, level, &s_i[0]);
        let pi_1 = &self.hash_one(alpha, level, &s_i[1]);
        pi_0 ^ pi_1
    }

    /// eval_next
    pub fn eval_next(
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

        let mut y_i = C::conditional_select(&w_i, (&w_i + w_cw).borrow(), t_i.0);
        y_i.conditional_negate(Choice::from(b as u8));

        y_i
    }

    /// proof_next
    pub fn proof_next(
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
    pub fn eval(&self, alpha: &[u8], key: &Key, pk: &PublicKey<C>) -> Share<C> {
        assert!(key.value.len() == self.key_size, "bad key size");

        let n = 8 * alpha.len();
        assert!(pk.cw.len() >= n, "bad public key size of cw field");
        assert!(pk.cs.len() >= n, "bad public key size of cs field");

        let mut s_i = Seed::from(key);
        let mut t_i = ControlBit::from(key);

        let mut y = C::new();
        let mut pi = Proof::new(self.proof_size);

        for i in 0..n {
            let alpha_i = ControlBit::from((alpha[i / 8] >> (i % 8)) & 0x1);
            y = self.eval_next(key.id, alpha_i, &mut s_i, &mut t_i, &pk.cw[i]);
            pi = self.proof_next(&pi, alpha, i, &s_i, t_i, &pk.cs[i]);
        }

        Share { y, pi }
    }

    /// verify checks that the proofs are equal and that the shares of beta add up to beta.
    pub fn verify(&self, a: &Share<C>, b: &Share<C>, beta: &C) -> bool {
        assert!(a.pi.0.len() == self.proof_size);
        assert!(b.pi.0.len() == self.proof_size);
        &a.y + &b.y == *beta && a.pi.0 == b.pi.0
    }

    fn prg(&self, seed: &Seed) -> Sequence {
        let dst = "100".as_bytes();
        let mut prg = self.prng_ctor.new_prng(seed, dst, &self.bind_info);

        let mut sl = Seed::new(self.seed_size);
        let mut sr = Seed::new(self.seed_size);
        let mut tl = ControlBit::from(0);
        let mut tr = ControlBit::from(0);
        sl.fill(&mut prg);
        sr.fill(&mut prg);
        tl.fill(&mut prg);
        tr.fill(&mut prg);

        Sequence(sl, tl, sr, tr)
    }

    fn convert(&self, seed: &Seed) -> (Seed, C) {
        let dst = "101".as_bytes();
        let mut prg = self.prng_ctor.new_prng(seed, dst, &self.bind_info);

        let mut out_seed = Seed::new(self.seed_size);
        let mut value = C::new();
        out_seed.fill(&mut prg);
        value.fill(&mut prg);

        (out_seed, value)
    }

    fn hash_one(&self, alpha: &[u8], level: usize, seed: &Seed) -> Proof {
        let dst = "vidpf cs proof".as_bytes();
        let mut binder = Vec::new();
        binder.extend_from_slice(alpha);
        binder.extend(level.to_le_bytes());
        let mut prg = self.prng_ctor.new_prng(seed, dst, &binder);

        let mut proof = Proof::new(self.proof_size);
        proof.fill(&mut prg);

        proof
    }

    fn hash_two(&self, proof: &Proof) -> Proof {
        let dst = "vidpf proof adjustment".as_bytes();
        let seed = Seed::new(self.seed_size);
        let binder = &proof.0;
        let mut prg = self.prng_ctor.new_prng(&seed, dst, binder);

        let mut out_proof = Proof::new(self.proof_size);
        out_proof.fill(&mut prg);

        out_proof
    }
}

#[derive(Debug, Clone, Copy)]
/// ServerID used to identify two aggregation servers.
pub enum ServerID {
    /// S0 is the first server.
    S0 = 0,
    /// S1 is the second server.
    S1 = 1,
}

#[derive(Debug)]
/// Share
pub struct Share<C>
where
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>,
{
    /// y is the sharing of the output.
    pub y: C,
    /// pi is a proof used to verify the share.
    pub pi: Proof,
}

/// Fill
pub trait Fill {
    /// fill
    fn fill(&mut self, r: &mut impl RngCore);
}

/// CodomainOps
pub trait CodomainOps<C: ?Sized>: Sized + Add<Output = C> + Sub<Output = C> {}

/// Codomain
pub trait Codomain: Fill + PartialEq + ConditionallyNegatable + ConditionallySelectable
where
    for<'a> &'a Self: CodomainOps<Self>,
{
    /// new
    fn new() -> Self;
}

/// GetPRG
pub trait GetPRNG {
    /// new_prng
    fn new_prng(&self, seed: &Seed, dst: &[u8], binder: &[u8]) -> impl RngCore;
}

/// PrngFromXof is a helper to create a PRNG from any [crate::vdaf::xof::Xof] implementer.
pub struct PrngFromXof<const SEED_SIZE: usize, X: Xof<SEED_SIZE>>(PhantomData<X>);

impl<const SEED_SIZE: usize, X: Xof<SEED_SIZE>> Default for PrngFromXof<SEED_SIZE, X> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<const SEED_SIZE: usize, X: Xof<SEED_SIZE>> GetPRNG for PrngFromXof<SEED_SIZE, X> {
    fn new_prng(&self, seed: &Seed, dst: &[u8], binder: &[u8]) -> impl RngCore {
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

impl BitXor for &Proof {
    type Output = Proof;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Proof(zip(&self.0, &rhs.0).map(|(a, b)| a ^ b).collect())
    }
}

#[derive(Debug)]
/// PublicKey is used by aggregation servers.
pub struct PublicKey<C>
where
    C: Codomain,
    for<'a> &'a C: CodomainOps<C>,
{
    cw: Vec<CorrectionWord<C>>,
    cs: Vec<Proof>,
}

#[derive(Debug)]
/// Key is the aggreagation server's private key.
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
}

struct Sequence(Seed, ControlBit, Seed, ControlBit);

impl BitXor for &Sequence {
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
/// ControlBit
pub struct ControlBit(Choice);

impl From<u8> for ControlBit {
    fn from(b: u8) -> Self {
        ControlBit(Choice::from(b))
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

impl ConditionallySelectable for ControlBit {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        ControlBit((a.0 & !choice) | (b.0 & choice))
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

    fn bitand(self, rhs: Self) -> Self::Output {
        ControlBit(self.0 & rhs.0)
    }
}

impl BitXor for ControlBit {
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

impl BitXor for &Seed {
    type Output = Seed;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Seed(zip(&self.0, &rhs.0).map(|(a, b)| a ^ b).collect())
    }
}

#[derive(Debug, Clone, Copy)]
/// Weight
pub struct Weight<F: FieldElement, const N: usize>([F; N]);

impl<F: FieldElement, const N: usize> CodomainOps<Weight<F, N>> for &Weight<F, N> {}

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

impl<F: FieldElement, const N: usize> ConditionallySelectable for Weight<F, N> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, zip(&a.0, &b.0))
            .for_each(|(c, (a, b))| *c = F::conditional_select(a, b, choice));
        out
    }
}
impl<F: FieldElement, const N: usize> ConditionallyNegatable for Weight<F, N> {
    fn conditional_negate(&mut self, choice: Choice) {
        self.0.iter_mut().for_each(|a| a.conditional_negate(choice));
    }
}

impl<F: FieldElement, const N: usize> PartialEq for Weight<F, N> {
    fn eq(&self, rhs: &Self) -> bool {
        N == self.0.len() && N == rhs.0.len() && zip(self.0, rhs.0).all(|(a, b)| a == b)
    }
}

impl<F: FieldElement, const N: usize> Add for &Weight<F, N> {
    type Output = Weight<F, N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, zip(self.0, rhs.0)).for_each(|(c, (a, b))| *c = a + b);
        out
    }
}

impl<F: FieldElement, const N: usize> Sub for &Weight<F, N> {
    type Output = Weight<F, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = Weight::<F, N>::new();
        zip(&mut out.0, zip(self.0, rhs.0)).for_each(|(c, (a, b))| *c = a - b);
        out
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

        println!("y0: {:?}", share0.y);
        println!("y1: {:?}", share1.y);
        println!("y: {:?}", &share0.y + &share1.y);
        println!("y_ok: {}", beta == &share0.y + &share1.y);
        println!("p0: {:?}", share0.pi);
        println!("p1: {:?}", share1.pi);
        println!("p: {:?}", share0.pi.0 == share1.pi.0);
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
            &share0.y + &share1.y == Weight::new(),
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
        // Gen:  AllocationInfo { count_total: 198, count_current: 0, count_max: 34, bytes_total: 4224, bytes_current: 0, bytes_max: 1504 }
        // Eval: AllocationInfo { count_total: 162, count_current: 0, count_max: 11, bytes_total: 3312, bytes_current: 0, bytes_max: 192 }
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
