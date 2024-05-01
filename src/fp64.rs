// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for any field GF(p) for which p < 2^64.

use crate::fp::{hi64, lo64, MAX_ROOTS};

/// This structure represents the parameters of a finite field GF(p) for which p < 2^64.
///
/// See also [`FieldParameters`](crate::fp::FieldParameters).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FieldParameters64 {
    /// The prime modulus `p`.
    pub p: u64,
    /// `mu = -p^(-1) mod 2^64`.
    pub mu: u64,
    /// `r2 = (2^64)^2 mod p`.
    pub r2: u64,
    /// The `2^num_roots`-th -principal root of unity. This element is used to generate the
    /// elements of `roots`.
    pub g: u64,
    /// The number of principal roots of unity in `roots`.
    pub num_roots: usize,
    /// Equal to `2^b - 1`, where `b` is the length of `p` in bits.
    pub bit_mask: u64,
    /// `roots[l]` is the `2^l`-th principal root of unity, i.e., `roots[l]` has order `2^l` in the
    /// multiplicative group. `roots[0]` is equal to one by definition.
    pub roots: [u64; MAX_ROOTS + 1],
}

impl FieldParameters64 {
    /// Addition. The result will be in [0, p), so long as both x and y are as well.
    #[inline(always)]
    pub fn add(&self, x: u64, y: u64) -> u64 {
        //   0,x
        // + 0,y
        // =====
        //   c,z
        let (z, carry) = x.overflowing_add(y);
        //     c, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = z.overflowing_sub(self.p);
        let (_s1, b1) = (carry as u64).overflowing_sub(b0 as u64);
        // if b1 == 1: return z
        // else:       return s0
        let m = 0u64.wrapping_sub(b1 as u64);
        (z & m) | (s0 & !m)
    }

    /// Subtraction. The result will be in [0, p), so long as both x and y are as well.
    #[inline(always)]
    pub fn sub(&self, x: u64, y: u64) -> u64 {
        //        x
        // -      y
        // ========
        //    b0,z0
        let (z0, b0) = x.overflowing_sub(y);
        let m = 0u64.wrapping_sub(b0 as u64);
        //      z0
        // +     p
        // ========
        //   s1,s0
        z0.wrapping_add(m & self.p)
        // if b1 == 1: return s0
        // else:       return z0
    }

    /// Multiplication of field elements in the Montgomery domain. This uses the REDC algorithm
    /// described [here][montgomery]. The result will be in [0, p).
    ///
    /// # Example usage
    /// ```text
    /// assert_eq!(fp.residue(fp.mul(fp.montgomery(23), fp.montgomery(2))), 46);
    /// ```
    ///
    /// [montgomery]: https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdf
    #[inline(always)]
    pub fn mul(&self, x: u64, y: u64) -> u64 {
        let mut zz = [0; 2];

        // Integer multiplication
        // z = x * y

        //     x
        // *   y
        // =====
        // z1,z0
        let result = (x as u128) * (y as u128);
        zz[0] = lo64(result) as u64;
        zz[1] = hi64(result) as u64;

        // Montgomery Reduction
        // z = z + p * mu*(z mod 2^64), where mu = (-p)^(-1) mod 2^64.

        // z1,z0
        // +   p
        // *   w = mu*z0
        // =====
        // z1, 0
        let w = self.mu.wrapping_mul(zz[0]);
        let result = (self.p as u128) * (w as u128);
        let hi = hi64(result);
        let lo = lo64(result) as u64;
        let (result, carry) = zz[0].overflowing_add(lo);
        zz[0] = result;
        let result = zz[1] as u128 + hi + carry as u128;
        zz[1] = lo64(result) as u64;
        let cc = hi64(result) as u64;

        // z = (z1)
        let prod = zz[1];

        // Final subtraction
        // If z >= p, then z = z - p

        //    cc, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = prod.overflowing_sub(self.p);
        let (_s1, b1) = cc.overflowing_sub(b0 as u64);
        // if b1 == 1: return z
        // else:       return s0
        let mask = 0u64.wrapping_sub(b1 as u64);
        (prod & mask) | (s0 & !mask)
    }

    /// Modular exponentiation, i.e., `x^exp (mod p)` where `p` is the modulus. Note that the
    /// runtime of this algorithm is linear in the bit length of `exp`.
    pub fn pow(&self, x: u64, exp: u64) -> u64 {
        let mut t = self.montgomery(1);
        for i in (0..64 - exp.leading_zeros()).rev() {
            t = self.mul(t, t);
            if (exp >> i) & 1 != 0 {
                t = self.mul(t, x);
            }
        }
        t
    }

    /// Modular inversion, i.e., x^-1 (mod p) where `p` is the modulus. Note that the runtime of
    /// this algorithm is linear in the bit length of `p`.
    #[inline(always)]
    pub fn inv(&self, x: u64) -> u64 {
        self.pow(x, self.p - 2)
    }

    /// Negation, i.e., `-x (mod p)` where `p` is the modulus.
    #[inline(always)]
    pub fn neg(&self, x: u64) -> u64 {
        self.sub(0, x)
    }

    /// Maps an integer to its internal representation. Field elements are mapped to the Montgomery
    /// domain in order to carry out field arithmetic. The result will be in [0, p).
    ///
    /// # Example usage
    /// ```text
    /// let integer = 1; // Standard integer representation
    /// let elem = fp.montgomery(integer); // Internal representation in the Montgomery domain
    /// assert_eq!(elem, 2564090464);
    /// ```
    #[inline(always)]
    pub fn montgomery(&self, x: u64) -> u64 {
        modp(self.mul(x, self.r2), self.p)
    }

    /// Maps a field element to its representation as an integer. The result will be in [0, p).
    ///
    /// #Example usage
    /// ```text
    /// let elem = 2564090464; // Internal representation in the Montgomery domain
    /// let integer = fp.residue(elem); // Standard integer representation
    /// assert_eq!(integer, 1);
    /// ```
    #[inline(always)]
    pub fn residue(&self, x: u64) -> u64 {
        modp(self.mul(x, 1), self.p)
    }
}

#[inline(always)]
fn modp(x: u64, p: u64) -> u64 {
    let (z, carry) = x.overflowing_sub(p);
    let m = 0u64.wrapping_sub(carry as u64);
    z.wrapping_add(m & p)
}

pub(crate) const FP64: FieldParameters64 = FieldParameters64 {
    p: 18446744069414584321, // 64-bit prime
    mu: 18446744069414584319,
    r2: 18446744065119617025,
    g: 15733474329512464024,
    num_roots: 32,
    bit_mask: 18446744073709551615,
    roots: [
        4294967295,
        18446744065119617026,
        18446744069414518785,
        18374686475393433601,
        268435456,
        18446673700670406657,
        18446744069414584193,
        576460752303421440,
        16576810576923738718,
        6647628942875889800,
        10087739294013848503,
        2135208489130820273,
        10781050935026037169,
        3878014442329970502,
        1205735313231991947,
        2523909884358325590,
        13797134855221748930,
        12267112747022536458,
        430584883067102937,
        10135969988448727187,
        6815045114074884550,
    ],
};

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use rand::{distributions::Distribution, thread_rng};

    use crate::fp::tests::{
        all_field_parameters_tests, TestFieldParameters, TestFieldParametersData,
    };

    use super::*;

    impl TestFieldParameters for FieldParameters64 {
        fn p(&self) -> u128 {
            self.p.into()
        }

        fn g(&self) -> u128 {
            self.g as u128
        }

        fn r2(&self) -> u128 {
            self.r2 as u128
        }

        fn mu(&self) -> u64 {
            self.mu
        }

        fn bit_mask(&self) -> u128 {
            self.bit_mask as u128
        }

        fn num_roots(&self) -> usize {
            self.num_roots
        }

        fn roots(&self) -> Vec<u128> {
            self.roots.iter().map(|x| *x as u128).collect()
        }

        fn montgomery(&self, x: u128) -> u128 {
            FieldParameters64::montgomery(self, x.try_into().unwrap()).into()
        }

        fn residue(&self, x: u128) -> u128 {
            FieldParameters64::residue(self, x.try_into().unwrap()).into()
        }

        fn add(&self, x: u128, y: u128) -> u128 {
            FieldParameters64::add(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
        }

        fn sub(&self, x: u128, y: u128) -> u128 {
            FieldParameters64::sub(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
        }

        fn neg(&self, x: u128) -> u128 {
            FieldParameters64::neg(self, x.try_into().unwrap()).into()
        }

        fn mul(&self, x: u128, y: u128) -> u128 {
            FieldParameters64::mul(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
        }

        fn pow(&self, x: u128, exp: u128) -> u128 {
            FieldParameters64::pow(self, x.try_into().unwrap(), exp.try_into().unwrap()).into()
        }

        fn inv(&self, x: u128) -> u128 {
            FieldParameters64::inv(self, x.try_into().unwrap()).into()
        }

        fn rand_elem(&self) -> u128 {
            let uniform = rand::distributions::Uniform::from(0..self.p);
            self.montgomery(uniform.sample(&mut thread_rng())).into()
        }

        fn radix(&self) -> BigInt {
            BigInt::from(1) << 64
        }
    }

    #[test]
    fn test_fp64_u64() {
        all_field_parameters_tests(TestFieldParametersData {
            fp: Box::new(FP64),
            expected_p: 18446744069414584321,
            expected_g: 1753635133440165772,
            expected_order: 1 << 32,
        })
    }
}
