// SPDX-License-Identifier: MPL-2.0

//! Prime field arithmetic for any Galois field GF(p) for which `p < 2^W`,
//! where `W ≤ 128` is a specified word size.

use num_traits::{
    ops::overflowing::{OverflowingAdd, OverflowingSub},
    AsPrimitive, ConstOne, ConstZero, PrimInt, Unsigned, WrappingAdd, WrappingMul, WrappingSub,
};

use crate::fp::MAX_ROOTS;

/// `Word` is the datatype used for storing and operating with
/// field elements.
///
/// The types `u32`, `u64`, and `u128` implement this trait.
pub trait Word:
    'static
    + Unsigned
    + PrimInt
    + OverflowingAdd
    + OverflowingSub
    + WrappingAdd
    + WrappingSub
    + WrappingMul
    + ConstZero
    + ConstOne
    + From<bool>
{
    /// Number of bits of word size.
    const BITS: usize;
}

impl Word for u32 {
    const BITS: usize = Self::BITS as usize;
}

impl Word for u64 {
    const BITS: usize = Self::BITS as usize;
}

impl Word for u128 {
    const BITS: usize = Self::BITS as usize;
}

/// `FieldParameters` sets the parameters to implement a prime field
/// GF(p) for which the prime modulus `p` fits in one word of
/// `W ≤ 128` bits.
pub trait FieldParameters<W: Word> {
    /// The prime modulus `p`.
    const PRIME: W;
    /// `mu = -p^(-1) mod 2^LOG2_BASE`.
    const MU: W;
    /// `r2 = R^2 mod p`.
    const R2: W;
    /// The `2^num_roots`-th -principal root of unity. This element
    /// is used to generate the elements of `ROOTS`.
    const G: W;
    /// The number of principal roots of unity in `ROOTS`.
    const NUM_ROOTS: usize;
    /// Equal to `2^b - 1`, where `b` is the length of `p` in bits.
    const BIT_MASK: W;
    /// `ROOTS[l]` is the `2^l`-th principal root of unity, i.e.,
    /// `ROOTS[l]` has order `2^l` in the multiplicative group.
    /// `ROOTS[0]` is equal to one by definition.
    const ROOTS: [W; MAX_ROOTS + 1];
    /// The multiplicative inverse of 2.
    const HALF: W;
    /// The log2(base) for the base used for multiprecision arithmetic.
    /// So, `LOG2_BASE ≤ 64` as processors have at most a 64-bit
    /// integer multiplier.
    #[cfg(test)]
    const LOG2_BASE: usize;
    /// The log2(R) where R is the machine word-friendly modulus
    /// used in the Montgomery representation.
    #[cfg(test)]
    const LOG2_RADIX: usize;
}

/// `FieldOps` provides arithmetic operations over GF(p).
///
/// The multiplication method is required as it admits different
/// implementations.
pub trait FieldOps<W: Word>: FieldParameters<W> {
    /// Addition. The result will be in [0, p), so long as both x
    /// and y are as well.
    #[inline(always)]
    fn add(x: W, y: W) -> W {
        //   0,x
        // + 0,y
        // =====
        //   c,z
        let (z, carry) = x.overflowing_add(&y);

        //     c, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = z.overflowing_sub(&Self::PRIME);
        let (_s1, b1) =
            <W as From<bool>>::from(carry).overflowing_sub(&<W as From<bool>>::from(b0));
        // if b1 == 1: return z
        // else:       return s0
        let mask = W::ZERO.wrapping_sub(&<W as From<bool>>::from(b1));
        (z & mask) | (s0 & !mask)
    }

    /// Subtraction. The result will be in [0, p), so long as both x
    /// and y are as well.
    #[inline(always)]
    fn sub(x: W, y: W) -> W {
        //        x
        // -      y
        // ========
        //    b0,z0
        let (z0, b0) = x.overflowing_sub(&y);
        let mask = W::ZERO.wrapping_sub(&<W as From<bool>>::from(b0));
        //      z0
        // +     p
        // ========
        //   s1,s0
        z0.wrapping_add(&(mask & Self::PRIME))
        // z0 + (m & self.p)
        // if b1 == 1: return s0
        // else:       return z0
    }

    /// Negation, i.e., `-x (mod p)` where `p` is the modulus.
    #[inline(always)]
    fn neg(x: W) -> W {
        Self::sub(W::ZERO, x)
    }

    #[inline(always)]
    fn modp(x: W) -> W {
        Self::sub(x, Self::PRIME)
    }

    /// Multiplication. The result will be in [0, p), so long as both x
    /// and y are as well.
    fn mul(x: W, y: W) -> W;

    /// Modular exponentiation, i.e., `x^exp (mod p)` where `p` is
    /// the modulus. Note that the runtime of this algorithm is
    /// linear in the bit length of `exp`.
    fn pow(x: W, exp: W) -> W {
        let mut t = Self::ROOTS[0];
        for i in (0..W::BITS - (exp.leading_zeros() as usize)).rev() {
            t = Self::mul(t, t);
            if (exp >> i) & W::ONE != W::ZERO {
                t = Self::mul(t, x);
            }
        }
        t
    }

    /// Modular inversion, i.e., x^-1 (mod p) where `p` is the
    /// modulus. Note that the runtime of this algorithm is
    /// linear in the bit length of `p`.
    #[inline(always)]
    fn inv(x: W) -> W {
        Self::pow(x, Self::PRIME - W::ONE - W::ONE)
    }

    /// Maps an integer to its internal representation. Field
    /// elements are mapped to the Montgomery domain in order to
    /// carry out field arithmetic. The result will be in [0, p).
    #[inline(always)]
    fn montgomery(x: W) -> W {
        Self::modp(Self::mul(x, Self::R2))
    }

    /// Maps a field element to its representation as an integer.
    /// The result will be in [0, p).
    #[inline(always)]
    fn residue(x: W) -> W {
        Self::modp(Self::mul(x, W::ONE))
    }
}

/// `FieldMulOpsSingleWord` implements prime field multiplication.
///
/// The implementation assumes that the modulus `p` fits in one word of
/// 'W' bits, and that the product of two integers fits in a datatype
/// (`Self::DoubleWord`) of exactly `2*W` bits.
pub(crate) trait FieldMulOpsSingleWord<W>: FieldParameters<W>
where
    W: Word + AsPrimitive<Self::DoubleWord>,
{
    type DoubleWord: Word + AsPrimitive<W>;

    /// Multiplication of field elements in the Montgomery domain.
    /// This uses the Montgomery's [REDC algorithm][montgomery].
    /// The result will be in [0, p).
    ///
    /// [montgomery]: https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdf
    fn mul(x: W, y: W) -> W {
        let hi_lo = |v: Self::DoubleWord| -> (W, W) { ((v >> W::BITS).as_(), v.as_()) };

        // Integer multiplication
        // z = x * y

        //     x
        // *   y
        // =====
        // z1,z0
        let (z1, z0) = hi_lo(x.as_() * y.as_());

        // Montgomery Reduction
        // z = z + p * mu*(z mod 2^W), where mu = (-p)^(-1) mod 2^W.
        //   = z + p * w, where w = mu*z0
        let w = Self::MU.wrapping_mul(&z0);
        let (r1, r0) = hi_lo(Self::PRIME.as_() * w.as_());

        //    z1,z0
        // +  r1,r0
        //    =====
        // cc, z, 0
        let (_zero, carry) = z0.overflowing_add(&r0);
        let (cc, z) = hi_lo(z1.as_() + r1.as_() + <Self::DoubleWord as From<bool>>::from(carry));

        // Final subtraction
        // If z >= p, then z = z - p

        //    cc, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = z.overflowing_sub(&Self::PRIME);
        let (_s1, b1) = cc.overflowing_sub(&<W as From<bool>>::from(b0));
        // if b1 == 1: return z
        // else:       return s0
        let mask = W::ZERO.wrapping_sub(&<W as From<bool>>::from(b1));
        (z & mask) | (s0 & !mask)
    }
}

/// `FieldMulOpsSplitWord` implements prime field multiplication.
///
/// The implementation assumes that the modulus `p` fits in one word of
/// 'W' bits, but the product of two integers does not fit in any primitive
/// integer. Thus, multiplication is processed splitting integers in two
/// words.
pub(crate) trait FieldMulOpsSplitWord<W>: FieldParameters<W>
where
    W: Word + AsPrimitive<Self::HalfWord>,
{
    type HalfWord: Word + AsPrimitive<W>;
    const MU: Self::HalfWord;
    /// Multiplication of field elements in the Montgomery domain.
    /// This uses the Montgomery's [REDC algorithm][montgomery].
    /// The result will be in [0, p).
    ///
    /// [montgomery]: https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdf
    fn mul(x: W, y: W) -> W {
        let high = |v: W| v >> (W::BITS / 2);
        let low = |v: W| v & ((W::ONE << (W::BITS / 2)) - W::ONE);

        let (x1, x0) = (high(x), low(x));
        let (y1, y0) = (high(y), low(y));

        // Integer multiplication
        // z = x * y

        //       x1,x0
        // *     y1,y0
        // ===========
        // z3,z2,z1,z0
        let mut result = x0 * y0;
        let mut carry = high(result);
        let z0 = low(result);
        result = x0 * y1;
        let mut hi = high(result);
        let mut lo = low(result);
        result = lo + carry;
        let mut z1 = low(result);
        let mut cc = high(result);
        result = hi + cc;
        let mut z2 = low(result);

        result = x1 * y0;
        hi = high(result);
        lo = low(result);
        result = z1 + lo;
        z1 = low(result);
        cc = high(result);
        result = hi + cc;
        carry = low(result);

        result = x1 * y1;
        hi = high(result);
        lo = low(result);
        result = lo + carry;
        lo = low(result);
        cc = high(result);
        result = hi + cc;
        hi = low(result);
        result = z2 + lo;
        z2 = low(result);
        cc = high(result);
        result = hi + cc;
        let mut z3 = low(result);

        // Montgomery Reduction
        // z = z + p * mu*(z mod 2^64), where mu = (-p)^(-1) mod 2^64.

        // z3,z2,z1,z0
        // +     p1,p0
        // *         w = mu*z0
        // ===========
        // z3,z2,z1, 0
        let mut w = <Self as FieldMulOpsSplitWord<W>>::MU.wrapping_mul(&z0.as_());
        let p0 = low(Self::PRIME);
        result = p0 * w.as_();
        hi = high(result);
        lo = low(result);
        result = z0 + lo;
        cc = high(result);
        result = hi + cc;
        carry = low(result);

        let p1 = high(Self::PRIME);
        result = p1 * w.as_();
        hi = high(result);
        lo = low(result);
        result = lo + carry;
        lo = low(result);
        cc = high(result);
        result = hi + cc;
        hi = low(result);
        result = z1 + lo;
        z1 = low(result);
        cc = high(result);
        result = z2 + hi + cc;
        z2 = low(result);
        cc = high(result);
        result = z3 + cc;
        z3 = low(result);

        //    z3,z2,z1
        // +     p1,p0
        // *         w = mu*z1
        // ===========
        //    z3,z2, 0
        w = <Self as FieldMulOpsSplitWord<W>>::MU.wrapping_mul(&z1.as_());
        result = p0 * w.as_();
        hi = high(result);
        lo = low(result);
        result = z1 + lo;
        cc = high(result);
        result = hi + cc;
        carry = low(result);

        result = p1 * w.as_();
        hi = high(result);
        lo = low(result);
        result = lo + carry;
        lo = low(result);
        cc = high(result);
        result = hi + cc;
        hi = low(result);
        result = z2 + lo;
        z2 = low(result);
        cc = high(result);
        result = z3 + hi + cc;
        z3 = low(result);
        cc = high(result);

        // z = (z3,z2)
        let prod = z2 | (z3 << (W::BITS / 2));

        // Final subtraction
        // If z >= p, then z = z - p

        //    cc, z
        // -   0, p
        // ========
        // b1,s1,s0
        let (s0, b0) = prod.overflowing_sub(&Self::PRIME);
        let (_s1, b1) = cc.overflowing_sub(&<W as From<bool>>::from(b0));
        // if b1 == 1: return z
        // else:       return s0
        let mask = W::ZERO.wrapping_sub(&<W as From<bool>>::from(b1));
        (prod & mask) | (s0 & !mask)
    }
}

/// `impl_field_ops_single_word` helper to implement prime field operations.
///
/// The implementation assumes that the modulus `p` fits in one word of
/// 'W' bits, and that the product of two integers fits in a datatype
/// (`Self::DoubleWord`) of exactly `2*W` bits.
macro_rules! impl_field_ops_single_word {
    ($struct_name:ident, $W:ty, $W2:ty) => {
        const _: () = assert!(<$W2>::BITS == 2 * <$W>::BITS);
        impl $crate::fp::ops::FieldMulOpsSingleWord<$W> for $struct_name {
            type DoubleWord = $W2;
        }
        impl $crate::fp::ops::FieldOps<$W> for $struct_name {
            #[inline(always)]
            fn mul(x: $W, y: $W) -> $W {
                <Self as $crate::fp::ops::FieldMulOpsSingleWord<_>>::mul(x, y)
            }
        }
    };
}

/// `impl_field_ops_split_word` helper to implement prime field operations.
///
/// The implementation assumes that the modulus `p` fits in one word of
/// 'W' bits, but the product of two integers does not fit. Thus,
/// multiplication is processed splitting integers in two words.
macro_rules! impl_field_ops_split_word {
    ($struct_name:ident, $W:ty, $W2:ty) => {
        const _: () = assert!(2 * <$W2>::BITS == <$W>::BITS);
        impl $crate::fp::ops::FieldMulOpsSplitWord<$W> for $struct_name {
            type HalfWord = $W2;
            const MU: Self::HalfWord = {
                let mu = <$struct_name as FieldParameters<$W>>::MU;
                assert!(mu <= (<$W2>::MAX as $W));
                mu as $W2
            };
        }
        impl $crate::fp::ops::FieldOps<$W> for $struct_name {
            #[inline(always)]
            fn mul(x: $W, y: $W) -> $W {
                <Self as $crate::fp::ops::FieldMulOpsSplitWord<_>>::mul(x, y)
            }
        }
    };
}
