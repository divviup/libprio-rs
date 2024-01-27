// SPDX-License-Identifier: MPL-2.0

//! Implementations of VIDPF specified in [[draft-mouris-cfrg-mastic]].
//!
//! [draft-mouris-cfrg-mastic]: https://datatracker.ietf.org/doc/draft-mouris-cfrg-mastic/01/

use std::{
    error::Error,
    io::Cursor,
    iter::zip,
    marker::PhantomData,
    ops::BitXor,
    ops::Sub,
    ops::{Add, BitAnd},
};

use rand_core::RngCore;
use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable};

use crate::{codec::Decode, field::FieldElement, vdaf::xof::Seed as XofSeed};

use super::xof::{Xof, XofTurboShake128};

/// VERSION is a tag.
pub static VERSION: &str = "MyVIDPF";

/// Params defines the global parameters of VIDPF instance.
pub struct Params<
    // Field for operations.
    F: FieldElement,
> {
    // Number of bits for alpha.
    n: usize,
    // Dimension of weights.
    m: usize,
    // Security parameter in bytes.
    k: usize,
    // Pseudorandom generator based on an eXtendable-Output Function.
    prg: Box<dyn Prng>,

    _pd: PhantomData<F>,
}

impl<F: FieldElement> Params<F> {
    /// new
    pub fn new(sec_param: usize, n: usize, m: usize, prng: impl Prng + 'static) -> Self {
        assert!(
            prng.size() == (sec_param / 8),
            "security param does not match PRNG input size"
        );

        Self {
            n,
            m,
            k: sec_param / 8,
            prg: Box::new(prng),
            _pd: PhantomData::<F>,
        }
    }

    /// gen
    pub fn gen(
        &self,
        alpha: &[u8],
        beta: &VecFieldElement<F>,
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
            let val = [self.prg.gen(&s_i[0])?, self.prg.gen(&s_i[1])?];
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

            let (s_i_0, w_i_0) = self.convert(&s_tilde_i[0]);
            let (s_i_1, w_i_1) = self.convert(&s_tilde_i[1]);
            s_i[0] = s_i_0;
            s_i[1] = s_i_1;

            let mut w_cw = beta - &(&w_i_0 + &w_i_1);
            (&mut w_cw).conditional_negate(t_i[1].into());

            cw.push(CorrWord {
                s_cw,
                t_cw_l,
                t_cw_r,
                w_cw,
            });

            let pi_0 = self.hash_one(alpha, i, &s_i[0])?;
            let pi_1 = self.hash_one(alpha, i, &s_i[1])?;
            let pi = &pi_0 ^ &pi_1;
            cs.push(pi);
        }

        Ok((Public { cw, cs }, k0, k1))
    }

    fn convert(&self, _seed_in: &Seed) -> (Seed, VecFieldElement<F>) {
        (Seed(Vec::new()), VecFieldElement(vec![F::one(); self.m]))
    }

    fn hash_one(&self, alpha: &[u8], level: usize, seed: &Seed) -> Result<Proof, Box<dyn Error>> {
        let dst = vec![0u8; 10];
        let mut binder = Vec::new();
        binder.extend(self.n.to_le_bytes());
        binder.extend_from_slice(alpha);
        binder.extend(level.to_le_bytes());

        let seed = XofSeed::decode(&mut Cursor::new(&seed.0))?;
        let mut pi = Proof(vec![0u8; 2 * self.k]);
        XofTurboShake128::seed_stream(&seed, &dst, &binder).fill_bytes(&mut pi.0);
        Ok(pi)
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
    w_cw: VecFieldElement<F>,
}

#[derive(Debug)]
/// Proof
pub struct Proof(Vec<u8>);

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

impl<T: RngCore> From<&mut T> for ControlBit {
    fn from(r: &mut T) -> Self {
        let mut b = [0u8; 1];
        r.fill_bytes(&mut b);
        Self(b[0] & 0x1)
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

impl<T: RngCore> From<(usize, &mut T)> for Seed {
    fn from((n, r): (usize, &mut T)) -> Self {
        let mut k = vec![0u8; n];
        r.fill_bytes(&mut k);
        Seed(k)
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
/// VecFieldElement
pub struct VecFieldElement<F: FieldElement>(Vec<F>);

impl<F: FieldElement> ConditionallyNegatable for &mut VecFieldElement<F> {
    fn conditional_negate(&mut self, choice: Choice) {
        self.0.iter_mut().for_each(|a| a.conditional_negate(choice));
    }
}

impl<'a, F: FieldElement> Add<&'a VecFieldElement<F>> for &'a VecFieldElement<F> {
    type Output = VecFieldElement<F>;

    fn add(self, rhs: Self) -> Self::Output {
        VecFieldElement(zip(&self.0, &rhs.0).map(|(a, b)| *a + *b).collect())
    }
}

impl<'a, F: FieldElement> Sub<&'a VecFieldElement<F>> for &'a VecFieldElement<F> {
    type Output = VecFieldElement<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        VecFieldElement(zip(&self.0, &rhs.0).map(|(a, b)| *a - *b).collect())
    }
}

/// Prng
pub trait Prng {
    /// size
    fn size(&self) -> usize;
    /// gen
    fn gen(&self, input: &Seed) -> Result<Sequence, Box<dyn Error>>;
}

/// Sequence
pub struct Sequence(Seed, ControlBit, Seed, ControlBit);

struct PrngFromXof<const K: usize, X: Xof<K>>(PhantomData<X>);

impl<const K: usize, X: Xof<K>> PrngFromXof<K, X> {
    pub fn new() -> Self {
        Self(PhantomData::<X>)
    }
}

impl<const K: usize, X: Xof<K>> Prng for PrngFromXof<K, X> {
    fn size(&self) -> usize {
        K
    }
    // fn gen(&self, buffer: &mut [u8], input: &[u8]) -> Result<(), Box<dyn Error>> {
    //     let dst = vec![0u8; K];
    //     let binder = vec![0u8; K];
    //     let seed = XofSeed::decode(&mut Cursor::new(input))?;
    //     X::seed_stream(&seed, &dst, &binder).fill_bytes(buffer);
    //     Ok(())
    // }
    fn gen(&self, input: &Seed) -> Result<Sequence, Box<dyn Error>> {
        let dst = vec![0u8; K];
        let binder = vec![0u8; K];
        let input_seed = XofSeed::decode(&mut Cursor::new(&input.0))?;

        let mut rng = X::seed_stream(&input_seed, &dst, &binder);
        let sl = Seed::from((K, &mut rng));
        let tl = ControlBit::from(&mut rng);
        let sr = Seed::from((K, &mut rng));
        let tr = ControlBit::from(&mut rng);

        Ok(Sequence(sl, tl, sr, tr))
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
            vidpf::{PrngFromXof, VecFieldElement},
            xof::XofTurboShake128,
        },
    };

    use super::{Params, VERSION};

    #[test]
    fn devel() {
        const SEC_PARAM: usize = 128;
        let n: usize = 2;
        let m: usize = 3;
        let prng = PrngFromXof::<16, XofTurboShake128>::new();
        let vidpf = Params::<Field128>::new(SEC_PARAM, n, m, prng);
        let beta = VecFieldElement([21.into(), 22.into(), 23.into()].into());

        match vidpf.gen("1".as_bytes(), &beta) {
            Ok(data) => {
                println!("data: {:?}", data.0);
                println!("key0: {:?}", data.1);
                println!("key1: {:?}", data.2);
            }
            Err(e) => println!("error: {:?}", e),
        };
    }

    #[test]
    fn version() {
        assert_eq!(VERSION, "MyVIDPF")
    }
}
