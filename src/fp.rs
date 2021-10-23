// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for any field GF(p) for which p < 2^126.

#[cfg(test)]
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
    /// Equal to `2^b - 1`, where `b` is the length of `p` in bits.
    pub bit_mask: u128,
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
    #[cfg(test)]
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

        let bit_mask: u128 = p.next_power_of_two() - 1;
        assert_eq!(self.bit_mask, bit_mask, "bit_mask mismatch");
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
    bit_mask: 4294967295,
    roots: [
        2564090464, 1729828257, 306605458, 2294308040, 1648889905, 57098624, 2788941825,
        2779858277, 368200145, 2760217336, 594450960, 4255832533, 1372848488, 721329415,
        3873251478, 1134002069, 7138597, 2004587313, 2989350643, 725214187, 1074114499,
    ],
};

pub(crate) const FP64: FieldParameters = FieldParameters {
    p: 18446744069414584321, // 64-bit prime
    p2: 36893488138829168642,
    mu: 18446744069414584319,
    r2: 4294967295,
    g: 959634606461954525,
    num_roots: 32,
    bit_mask: 18446744073709551615,
    roots: [
        18446744065119617025,
        4294967296,
        18446462594437939201,
        72057594037927936,
        1152921504338411520,
        16384,
        18446743519658770561,
        18446735273187346433,
        6519596376689022014,
        9996039020351967275,
        15452408553935940313,
        15855629130643256449,
        8619522106083987867,
        13036116919365988132,
        1033106119984023956,
        16593078884869787648,
        16980581328500004402,
        12245796497946355434,
        8709441440702798460,
        8611358103550827629,
        8120528636261052110,
    ],
};

pub(crate) const FP96: FieldParameters = FieldParameters {
    p: 79228148845226978974766202881, // 96-bit prime
    p2: 158456297690453957949532405762,
    mu: 18446744073709551615,
    r2: 69162923446439011319006025217,
    g: 11329412859948499305522312170,
    num_roots: 64,
    bit_mask: 79228162514264337593543950335,
    roots: [
        10128756682736510015896859,
        79218020088544242464750306022,
        9188608122889034248261485869,
        10170869429050723924726258983,
        36379376833245035199462139324,
        20898601228930800484072244511,
        2845758484723985721473442509,
        71302585629145191158180162028,
        76552499132904394167108068662,
        48651998692455360626769616967,
        36570983454832589044179852640,
        72716740645782532591407744342,
        73296872548531908678227377531,
        14831293153408122430659535205,
        61540280632476003580389854060,
        42256269782069635955059793151,
        51673352890110285959979141934,
        43102967204983216507957944322,
        3990455111079735553382399289,
        68042997008257313116433801954,
        44344622755749285146379045633,
    ],
};

pub(crate) const FP126: FieldParameters = FieldParameters {
    p: 85070591730234613043491808580380655617, // 126-bit prime
    p2: 170141183460469226086983617160761311234,
    mu: 18446744073709551615,
    r2: 4228289479895218941917209552,
    g: 64132991420990267358698281226844529554,
    num_roots: 64,
    bit_mask: 85070591730234615865843651857942052863,
    roots: [
        85070591730234624332899181690626244605,
        85070591730234601754084435470135066629,
        38137617727210006121801958903729016440,
        8688340227897707017283728584977967178,
        50356378283019641467766742940691527450,
        33550949333748454978718892503921695778,
        47992546207817536319642646308549364948,
        40842705777252249605217191958775728119,
        89236517585491498085722663868093517978,
        63067092524540968535007362257738426824,
        89031567408593428196542541013322467020,
        41527756442199241501616896479704917226,
        27942344825010058474020272302955536377,
        36893509478604971204958162271142169833,
        77313788438105850413543617102369725114,
        66303272676638993803236953822843236868,
        7505587139693091429406100867350473894,
        24726071555371336713058093105331355614,
        14146496786913858079253200761825179484,
        66750945294819701976251551632766384780,
        39883619804504909766529131424051012626,
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
                expected_p: 18446744069414584321,
                expected_g: 1753635133440165772,
                expected_order: 1 << 32,
            },
            TestFieldParametersData {
                fp: FP96,
                expected_p: 79228148845226978974766202881,
                expected_g: 34233996298771126927060021012,
                expected_order: 1 << 64,
            },
            TestFieldParametersData {
                fp: FP126,
                expected_p: 85070591730234613043491808580380655617,
                expected_g: 31797354429379572309620216906993098145,
                expected_order: 1 << 64,
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
