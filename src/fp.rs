// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for any field GF(p) for which p < 2^126.

use rand::{prelude::*, Rng};

/// For each set of field parameters we pre-compute the 1st, 2nd, 4th, ..., 2^20-th principal roots
/// of unity. The largest of these is used to run the FFT algorithm on an input of size 2^20. This
/// is the largest input size we would ever need for the cryptographic applications in this crate.
pub(crate) const MAX_ROOTS: usize = 20;

/// This structure represents the parameters of a finite field GF(p) for which p < 2^126.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FieldParameters {
    /// The prime modulus `p`.
    pub p: u128,
    /// `p * 2`.
    pub p2: u128,
    /// `mu = -p^(-1) mod 2^64`.
    pub mu: u64,
    /// `r2 = (2^128)^2 mod p`.
    pub r2: u128,
    /// The `2^num_roots`-th -principal root of unity. This element is used to generate the
    /// elements of `roots`.
    pub g: u128,
    /// The number of principal roots of unity in `roots`.
    pub num_roots: usize,
    /// `roots[l]` is the `2^l`-th principal root of unity, i.e., `roots[l]` has order `2^l` in the
    /// multiplicative group. `root[l]` is equal to one by definition.
    pub roots: [u128; MAX_ROOTS + 1],
}

impl FieldParameters {
    /// Addition.
    pub fn add(&self, x: u128, y: u128) -> u128 {
        let (z, carry) = x.wrapping_add(y).overflowing_sub(self.p2);
        let m = 0u128.wrapping_sub(carry as u128);
        z.wrapping_add(m & self.p2)
    }

    /// Subtraction.
    pub fn sub(&self, x: u128, y: u128) -> u128 {
        let (z, carry) = x.overflowing_sub(y);
        let m = 0u128.wrapping_sub(carry as u128);
        z.wrapping_add(m & self.p2)
    }

    /// Multiplication of field elements in the Montgomery domain. This uses the REDC algorithm
    /// described
    /// [here](https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdfA).
    ///
    /// Example usage:
    /// assert_eq!(fp.from_elem(fp.mul(fp.elem(23), fp.elem(2))), 46);
    pub fn mul(&self, x: u128, y: u128) -> u128 {
        let x = [lo64(x), hi64(x)];
        let y = [lo64(y), hi64(y)];
        let p = [lo64(self.p), hi64(self.p)];
        let mut zz = [0; 4];
        let mut result: u128;
        let mut carry: u128;
        let mut hi: u128;
        let mut lo: u128;
        let mut cc: u128;

        // Integer multiplication
        result = x[0] * y[0];
        carry = hi64(result);
        zz[0] = lo64(result);
        result = x[0] * y[1];
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        zz[2] = lo64(result);

        result = x[1] * y[0];
        hi = hi64(result);
        lo = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = x[1] * y[1];
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[2] + lo;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        zz[3] = lo64(result);

        // Reduction
        let w = self.mu.wrapping_mul(zz[0] as u64);
        result = p[0] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = zz[0] + lo;
        zz[0] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = p[1] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = zz[2] + hi + cc;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = zz[3] + cc;
        zz[3] = lo64(result);

        let w = self.mu.wrapping_mul(zz[1] as u64);
        result = p[0] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = p[1] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[2] + lo;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = zz[3] + hi + cc;
        zz[3] = lo64(result);

        zz[2] | (zz[3] << 64)
    }

    /// Modular exponentiation, i.e., `x^exp (mod p)` where `p` is the modulus. Note that the
    /// runtime of this algorithm is linear in the bit length of `exp`.
    pub fn pow(&self, x: u128, exp: u128) -> u128 {
        let mut t = self.elem(1);
        for i in (0..128 - exp.leading_zeros()).rev() {
            t = self.mul(t, t);
            if (exp >> i) & 1 != 0 {
                t = self.mul(t, x);
            }
        }
        t
    }

    /// Modular inversion, i.e., x^-1 (mod p) where `p` is the modulu. Note that the runtime of
    /// this algorithm is linear in the bit length of `p`.
    pub fn inv(&self, x: u128) -> u128 {
        self.pow(x, self.p - 2)
    }

    /// Negation, i.e., `-x (mod p)` where `p` is the modulus.
    pub fn neg(&self, x: u128) -> u128 {
        self.sub(0, x)
    }

    /// Maps an integer to its internal representation. Field elements are mapped to the Montgomery
    /// domain in order to carry out field arithmetic.
    ///
    /// Example usage:
    /// let integer = 1; // Standard integer representation
    /// let elem = fp.elem(integer); // Internal representation in the Montgomery domain
    /// assert_eq!(elem, 2564090464);
    pub fn elem(&self, x: u128) -> u128 {
        modp(self.mul(x, self.r2), self.p)
    }

    /// Returns a random field element mapped.
    pub fn rand_elem<R: Rng + ?Sized>(&self, rng: &mut R) -> u128 {
        let uniform = rand::distributions::Uniform::from(0..self.p);
        self.elem(uniform.sample(rng))
    }

    /// Maps a field element to its representation as an integer.
    ///
    /// Example usage:
    /// let elem = 2564090464; // Internal representation in the Montgomery domain
    /// let integer = fp.from_elem(elem); // Standard integer representation
    /// assert_eq!(integer, 1);
    pub fn from_elem(&self, x: u128) -> u128 {
        modp(self.mul(x, 1), self.p)
    }

    #[cfg(test)]
    pub fn check(&self, p: u128, g: u128, order: u128) {
        use modinverse::modinverse;
        use num_bigint::{BigInt, ToBigInt};
        use std::cmp::max;

        if let Some(x) = p.checked_next_power_of_two() {
            if x > 1 << 126 {
                panic!("p >= 2^126");
            }
        } else {
            panic!("p >= 2^126");
        }
        assert_eq!(self.p, p, "p mismatch");
        assert_eq!(self.p2, p << 1, "p2 mismatch");

        let mu = match modinverse((-(p as i128)).rem_euclid(1 << 64), 1 << 64) {
            Some(mu) => mu as u64,
            None => panic!("inverse of -p (mod 2^64) is undefined"),
        };
        assert_eq!(self.mu, mu, "mu mismatch");

        let big_p = &p.to_bigint().unwrap();
        let big_r: &BigInt = &(&(BigInt::from(1) << 128) % big_p);
        let big_r2: &BigInt = &(&(big_r * big_r) % big_p);
        let mut it = big_r2.iter_u64_digits();
        let mut r2 = 0;
        r2 |= it.next().unwrap() as u128;
        if let Some(x) = it.next() {
            r2 |= (x as u128) << 64;
        }
        assert_eq!(self.r2, r2, "r2 mismatch");

        assert_eq!(self.g, self.elem(g), "g mismatch");
        assert_eq!(
            self.from_elem(self.pow(self.g, order)),
            1,
            "g order incorrect"
        );

        let num_roots = log2(order) as usize;
        assert_eq!(order, 1 << num_roots, "order not a power of 2");
        assert_eq!(self.num_roots, num_roots, "num_roots mismatch");

        let mut roots = vec![0; max(num_roots, MAX_ROOTS) + 1];
        roots[num_roots] = self.elem(g);
        for i in (0..num_roots).rev() {
            roots[i] = self.mul(roots[i + 1], roots[i + 1]);
        }
        assert_eq!(&self.roots, &roots[..MAX_ROOTS + 1], "roots mismatch");
        assert_eq!(self.from_elem(self.roots[0]), 1, "first root is not one");
    }
}

fn lo64(x: u128) -> u128 {
    x & ((1 << 64) - 1)
}

fn hi64(x: u128) -> u128 {
    x >> 64
}

fn modp(x: u128, p: u128) -> u128 {
    let (z, carry) = x.overflowing_sub(p);
    let m = 0u128.wrapping_sub(carry as u128);
    z.wrapping_add(m & p)
}

pub(crate) const FP32: FieldParameters = FieldParameters {
    p: 4293918721, // 32-bit prime
    p2: 8587837442,
    mu: 17302828673139736575,
    r2: 1676699750,
    g: 1074114499,
    num_roots: 20,
    roots: [
        2564090464, 1729828257, 306605458, 2294308040, 1648889905, 57098624, 2788941825,
        2779858277, 368200145, 2760217336, 594450960, 4255832533, 1372848488, 721329415,
        3873251478, 1134002069, 7138597, 2004587313, 2989350643, 725214187, 1074114499,
    ],
};

pub(crate) const FP64: FieldParameters = FieldParameters {
    p: 15564440312192434177, // 64-bit prime
    p2: 31128880624384868354,
    mu: 15564440312192434175,
    r2: 13031533328350459868,
    g: 8693478717884812021,
    num_roots: 59,
    roots: [
        3501465310287461188,
        12062975001904972989,
        14847933823983913979,
        5743743733744043357,
        12036183376424650304,
        1310071208805268988,
        351359342873885390,
        760642505652925971,
        8075457983432319221,
        14554120515039960006,
        9277695709938157757,
        5146332056710123439,
        9547487945110664452,
        1379816102304800478,
        8461341165309158767,
        12152693588256515089,
        9516424165972384563,
        8278889272850348764,
        6847784946159064188,
        875721217475244711,
        3028669228647031529,
    ],
};

pub(crate) const FP80: FieldParameters = FieldParameters {
    p: 779190469673491460259841, // 80-bit prime
    p2: 1558380939346982920519682,
    mu: 18446744073709551615,
    r2: 699883506621195336351723,
    g: 470015708362303528848629,
    num_roots: 72,
    roots: [
        146393360532246310485619,
        632797109141245149774222,
        671768715528862959481,
        155287852188866912681838,
        84398650169430234366422,
        591732619446824370107997,
        369489067863767193117628,
        65351307276236357745139,
        250263845222966534834802,
        615370028124972287024172,
        428271082931219526829234,
        82144483146855494501530,
        655790508505248218964487,
        715547187733913654852114,
        29653674159319497967645,
        208078234303463777930443,
        495449125070884366403280,
        409220521346165172951210,
        134217175002192449913815,
        87718316256013518265593,
        261278801525790549618040,
    ],
};

pub(crate) const FP126: FieldParameters = FieldParameters {
    p: 74769074762901517850839147140769382401, // 126-bit prime
    p2: 149538149525803035701678294281538764802,
    mu: 18446744073709551615,
    r2: 27801541991839173768379182336352451464,
    g: 63245316532470582112420298384754157617,
    num_roots: 118,
    roots: [
        41206067869332392060018018868690681852,
        33563006893569125790821128272078700549,
        9969209968386869007425498928188874206,
        26245577744033816872542400585149646017,
        53536320213034809573447803273264211942,
        27613962195776955012920796583378240442,
        32365734403831264958530930421153577004,
        13579354626561224539372784961933801433,
        57316758837288076943811104544124917759,
        70913423672054213072910590891105064074,
        71265034959502540558500186666669444000,
        34207722676470700263211551887273866594,
        37340170148681921863826823402458410577,
        35009585531414332540073382665488435215,
        70329412074928482115163094157328536788,
        39119429759852994810554872198104013087,
        47573549675073661838420354629772140200,
        77849817677037388106638164185970185092,
        37853717993704464400736177978677308170,
        83509620839139853788963077680031940984,
        64573608437864873942981348294630891347,
    ],
};

// Compute the ceiling of the base-2 logarithm of `x`.
pub(crate) fn log2(x: u128) -> u128 {
    let y = (127 - x.leading_zeros()) as u128;
    y + ((x > 1 << y) as u128)
}

#[cfg(test)]
mod tests {
    use super::*;
    use modinverse::modinverse;
    use num_bigint::ToBigInt;

    #[test]
    fn test_log2() {
        assert_eq!(log2(1), 0);
        assert_eq!(log2(2), 1);
        assert_eq!(log2(3), 2);
        assert_eq!(log2(4), 2);
        assert_eq!(log2(15), 4);
        assert_eq!(log2(16), 4);
        assert_eq!(log2(30), 5);
        assert_eq!(log2(32), 5);
        assert_eq!(log2(1 << 127), 127);
        assert_eq!(log2((1 << 127) + 13), 128);
    }

    struct TestFieldParametersData {
        fp: FieldParameters,  // The paramters being tested
        expected_p: u128,     // Expected fp.p
        expected_g: u128,     // Expected fp.from_elem(fp.g)
        expected_order: u128, // Expect fp.from_elem(fp.pow(fp.g, expected_order)) == 1
    }

    #[test]
    fn test_fp() {
        let test_fps = vec![
            TestFieldParametersData {
                fp: FP32,
                expected_p: 4293918721,
                expected_g: 3925978153,
                expected_order: 1 << 20,
            },
            TestFieldParametersData {
                fp: FP64,
                expected_p: 15564440312192434177,
                expected_g: 7450580596923828125,
                expected_order: 1 << 59,
            },
            TestFieldParametersData {
                fp: FP80,
                expected_p: 779190469673491460259841,
                expected_g: 41782115852031095118226,
                expected_order: 1 << 72,
            },
            TestFieldParametersData {
                fp: FP126,
                expected_p: 74769074762901517850839147140769382401,
                expected_g: 43421413544015439978138831414974882540,
                expected_order: 1 << 118,
            },
        ];

        for t in test_fps.into_iter() {
            //  Check that the field parameters have been constructed properly.
            t.fp.check(t.expected_p, t.expected_g, t.expected_order);

            // Check that the generator has the correct order.
            assert_eq!(t.fp.from_elem(t.fp.pow(t.fp.g, t.expected_order)), 1);

            // Test arithmetic using the field parameters.
            arithmetic_test(&t.fp);
        }
    }

    fn arithmetic_test(fp: &FieldParameters) {
        let mut rng = rand::thread_rng();
        let big_p = &fp.p.to_bigint().unwrap();

        for _ in 0..100 {
            let x = fp.rand_elem(&mut rng);
            let y = fp.rand_elem(&mut rng);
            let big_x = &fp.from_elem(x).to_bigint().unwrap();
            let big_y = &fp.from_elem(y).to_bigint().unwrap();

            // Test addition.
            let got = fp.add(x, y);
            let want = (big_x + big_y) % big_p;
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test subtraction.
            let got = fp.sub(x, y);
            let want = if big_x >= big_y {
                big_x - big_y
            } else {
                big_p - big_y + big_x
            };
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test multiplication.
            let got = fp.mul(x, y);
            let want = (big_x * big_y) % big_p;
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test inversion.
            let got = fp.inv(x);
            let want = modinverse(fp.from_elem(x) as i128, fp.p as i128).unwrap();
            assert_eq!(fp.from_elem(got) as i128, want);
            assert_eq!(fp.from_elem(fp.mul(got, x)), 1);

            // Test negation.
            let got = fp.neg(x);
            let want = (-(fp.from_elem(x) as i128)).rem_euclid(fp.p as i128);
            assert_eq!(fp.from_elem(got) as i128, want);
            assert_eq!(fp.from_elem(fp.add(got, x)), 0);
        }
    }
}
