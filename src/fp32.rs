// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for any field GF(p) for which p < 2^32.

use crate::fp::{hi32, lo32, MAX_ROOTS};

/// This structure represents the parameters of a finite field GF(p) for which p < 2^32.
///
/// See also [`FieldParameters`](crate::fp::FieldParameters).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FieldParameters32 {
    /// The prime modulus `p`.
    pub p: u32,
    /// `mu = -p^(-1) mod 2^32`.
    pub mu: u32,
    /// `r2 = (2^32)^2 mod p`.
    pub r2: u32,
    /// The `2^num_roots`-th -principal root of unity. This element is used to generate the
    /// elements of `roots`.
    pub g: u32,
    /// The number of principal roots of unity in `roots`.
    pub num_roots: usize,
    /// Equal to `2^b - 1`, where `b` is the length of `p` in bits.
    pub bit_mask: u32,
    /// `roots[l]` is the `2^l`-th principal root of unity, i.e., `roots[l]` has order `2^l` in the
    /// multiplicative group. `roots[0]` is equal to one by definition.
    pub roots: [u32; MAX_ROOTS + 1],
}

impl FieldParameters32 {
    /// Addition. The result will be in [0, p), so long as both x and y are as well.
    #[inline(always)]
    pub fn add(&self, x: u32, y: u32) -> u32 {
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
        let (_s1, b1) = (carry as u32).overflowing_sub(b0 as u32);
        // if b1 == 1: return z
        // else:       return s0
        let m = 0u32.wrapping_sub(b1 as u32);
        (z & m) | (s0 & !m)
    }

    /// Subtraction. The result will be in [0, p), so long as both x and y are as well.
    #[inline(always)]
    pub fn sub(&self, x: u32, y: u32) -> u32 {
        //        x
        // -      y
        // ========
        //    b0,z0
        let (z0, b0) = x.overflowing_sub(y);
        let m = 0u32.wrapping_sub(b0 as u32);
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
    pub fn mul(&self, x: u32, y: u32) -> u32 {
        let mut zz = [0; 2];

        // Integer multiplication
        // z = x * y

        //     x
        // *   y
        // =====
        // z1,z0
        let result = (x as u64) * (y as u64);
        zz[0] = lo32(result) as u32;
        zz[1] = hi32(result) as u32;

        // Montgomery Reduction
        // z = z + p * mu*(z mod 2^32), where mu = (-p)^(-1) mod 2^32.

        // z1,z0
        // +   p
        // *   w = mu*z0
        // =====
        // z1, 0
        let w = self.mu.wrapping_mul(zz[0]);
        let result = (self.p as u64) * (w as u64);
        let hi = hi32(result);
        let lo = lo32(result) as u32;
        let (result, carry) = zz[0].overflowing_add(lo);
        zz[0] = result;
        let result = zz[1] as u64 + hi + carry as u64;
        zz[1] = lo32(result) as u32;
        let cc = hi32(result) as u32;

        // z = (z1)
        let prod = zz[1];

        // Final subtraction
        // If z >= p, then z = z - p

        //    cc, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = prod.overflowing_sub(self.p);
        let (_s1, b1) = cc.overflowing_sub(b0 as u32);
        // if b1 == 1: return z
        // else:       return s0
        let mask = 0u32.wrapping_sub(b1 as u32);
        (prod & mask) | (s0 & !mask)
    }

    /// Modular exponentiation, i.e., `x^exp (mod p)` where `p` is the modulus. Note that the
    /// runtime of this algorithm is linear in the bit length of `exp`.
    pub fn pow(&self, x: u32, exp: u32) -> u32 {
        let mut t = self.montgomery(1);
        for i in (0..32 - exp.leading_zeros()).rev() {
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
    pub fn inv(&self, x: u32) -> u32 {
        self.pow(x, self.p - 2)
    }

    /// Negation, i.e., `-x (mod p)` where `p` is the modulus.
    #[inline(always)]
    pub fn neg(&self, x: u32) -> u32 {
        self.sub(0, x)
    }

    /// Maps an integer to its internal representation. Field elements are mapped to the Montgomery
    /// domain in order to carry out field arithmetic. The result will be in [0, p).
    ///
    /// # Example usage
    /// ```text
    /// let integer = 1; // Standard integer representation
    /// let elem = fp.montgomery(integer); // Internal representation in the Montgomery domain
    /// assert_eq!(elem, 1048575);
    /// ```
    #[inline(always)]
    pub fn montgomery(&self, x: u32) -> u32 {
        modp(self.mul(x, self.r2), self.p)
    }

    /// Maps a field element to its representation as an integer. The result will be in [0, p).
    ///
    /// #Example usage
    /// ```text
    /// let elem = 1048575; // Internal representation in the Montgomery domain
    /// let integer = fp.residue(elem); // Standard integer representation
    /// assert_eq!(integer, 1);
    /// ```
    #[inline(always)]
    pub fn residue(&self, x: u32) -> u32 {
        modp(self.mul(x, 1), self.p)
    }
}

#[inline(always)]
fn modp(x: u32, p: u32) -> u32 {
    let (z, carry) = x.overflowing_sub(p);
    let m = 0u32.wrapping_sub(carry as u32);
    z.wrapping_add(m & p)
}

pub(crate) const FP32: FieldParameters32 = FieldParameters32 {
    p: 4293918721, // 32-bit prime
    mu: 4293918719,
    r2: 266338049,
    g: 3903828692,
    num_roots: 20,
    bit_mask: 4294967295,
    roots: [
        1048575, 4292870146, 1189722990, 3984864191, 2523259768, 2828840154, 1658715539,
        1534972560, 3732920810, 3229320047, 2836564014, 2170197442, 3760663902, 2144268387,
        3849278021, 1395394315, 574397626, 125025876, 3755041587, 2680072542, 3903828692,
    ],
};

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use crate::fp::tests::{
        all_field_parameters_tests, TestFieldParameters, TestFieldParametersData,
    };

    use super::*;

    impl TestFieldParameters for FieldParameters32 {
        fn p(&self) -> u128 {
            self.p.into()
        }

        fn g(&self) -> u128 {
            self.g as u128
        }

        fn base(&self) -> u128 {
            1u128 << 32
        }

        fn r2(&self) -> u128 {
            self.r2 as u128
        }

        fn mu(&self) -> u64 {
            self.mu as u64
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
            FieldParameters32::montgomery(self, x.try_into().unwrap()).into()
        }

        fn residue(&self, x: u128) -> u128 {
            FieldParameters32::residue(self, x.try_into().unwrap()).into()
        }

        fn add(&self, x: u128, y: u128) -> u128 {
            FieldParameters32::add(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
        }

        fn sub(&self, x: u128, y: u128) -> u128 {
            FieldParameters32::sub(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
        }

        fn neg(&self, x: u128) -> u128 {
            FieldParameters32::neg(self, x.try_into().unwrap()).into()
        }

        fn mul(&self, x: u128, y: u128) -> u128 {
            FieldParameters32::mul(self, x.try_into().unwrap(), y.try_into().unwrap()).into()
        }

        fn pow(&self, x: u128, exp: u128) -> u128 {
            FieldParameters32::pow(self, x.try_into().unwrap(), exp.try_into().unwrap()).into()
        }

        fn inv(&self, x: u128) -> u128 {
            FieldParameters32::inv(self, x.try_into().unwrap()).into()
        }

        fn radix(&self) -> BigInt {
            BigInt::from(self.base())
        }
    }

    #[test]
    fn test_fp32_u32() {
        all_field_parameters_tests(TestFieldParametersData {
            fp: Box::new(FP32),
            expected_p: 4293918721,
            expected_g: 3925978153,
            expected_order: 1 << 20,
        });
    }
}
