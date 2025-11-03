// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for any field GF(p) for which p < 2^128.

#[macro_use]
mod ops;

pub use ops::{FieldOps, FieldParameters};

/// For each set of field parameters we pre-compute the 1st, 2nd, 4th, ..., 2^20-th principal roots
/// of unity. The largest of these is used to run the NTT algorithm on an input of size 2^20. This
/// is the largest input size we would ever need for the cryptographic applications in this crate.
pub(crate) const MAX_ROOTS: usize = 20;

/// FP32 implements operations over GF(p) for which the prime
/// modulus `p` fits in a u32 word.
pub(crate) struct FP32;

impl_field_ops_single_word!(FP32, u32, u64);

impl FieldParameters<u32> for FP32 {
    const PRIME: u32 = 4293918721;
    const MU: u32 = 4293918719;
    const R2: u32 = 266338049;
    const G: u32 = 3903828692;
    const NUM_ROOTS: usize = 20;
    const BIT_MASK: u32 = 4294967295;
    const ROOTS: [u32; MAX_ROOTS + 1] = [
        1048575, 4292870146, 1189722990, 3984864191, 2523259768, 2828840154, 1658715539,
        1534972560, 3732920810, 3229320047, 2836564014, 2170197442, 3760663902, 2144268387,
        3849278021, 1395394315, 574397626, 125025876, 3755041587, 2680072542, 3903828692,
    ];
    const HALF: u32 = 2147483648;
    #[cfg(test)]
    const LOG2_BASE: usize = 32;
    #[cfg(test)]
    const LOG2_RADIX: usize = 32;
}

/// FP64 implements operations over GF(p) for which the prime
/// modulus `p` fits in a u64 word.
pub(crate) struct FP64;

impl_field_ops_single_word!(FP64, u64, u128);

impl FieldParameters<u64> for FP64 {
    const PRIME: u64 = 18446744069414584321;
    const MU: u64 = 18446744069414584319;
    const R2: u64 = 18446744065119617025;
    const G: u64 = 15733474329512464024;
    const NUM_ROOTS: usize = 32;
    const BIT_MASK: u64 = 18446744073709551615;
    const ROOTS: [u64; MAX_ROOTS + 1] = [
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
    ];
    const HALF: u64 = 9223372036854775808;
    #[cfg(test)]
    const LOG2_BASE: usize = 64;
    #[cfg(test)]
    const LOG2_RADIX: usize = 64;
}

/// FP128 implements operations over GF(p) for which the prime
/// modulus `p` fits in a u128 word.
pub(crate) struct FP128;

impl_field_ops_split_word!(FP128, u128, u64);

impl FieldParameters<u128> for FP128 {
    const PRIME: u128 = 340282366920938462946865773367900766209;
    const MU: u128 = 18446744073709551615;
    const R2: u128 = 403909908237944342183153;
    const G: u128 = 107630958476043550189608038630704257141;
    const NUM_ROOTS: usize = 66;
    const BIT_MASK: u128 = 340282366920938463463374607431768211455;
    const ROOTS: [u128; MAX_ROOTS + 1] = [
        516508834063867445247,
        340282366920938462430356939304033320962,
        129526470195413442198896969089616959958,
        169031622068548287099117778531474117974,
        81612939378432101163303892927894236156,
        122401220764524715189382260548353967708,
        199453575871863981432000940507837456190,
        272368408887745135168960576051472383806,
        24863773656265022616993900367764287617,
        257882853788779266319541142124730662203,
        323732363244658673145040701829006542956,
        57532865270871759635014308631881743007,
        149571414409418047452773959687184934208,
        177018931070866797456844925926211239962,
        268896136799800963964749917185333891349,
        244556960591856046954834420512544511831,
        118945432085812380213390062516065622346,
        202007153998709986841225284843501908420,
        332677126194796691532164818746739771387,
        258279638927684931537542082169183965856,
        148221243758794364405224645520862378432,
    ];
    const HALF: u128 = 170141183460469231731687303715884105728;
    #[cfg(test)]
    const LOG2_BASE: usize = 64;
    #[cfg(test)]
    const LOG2_RADIX: usize = 128;
}

/// Compute the ceiling of the base-2 logarithm of `x`.
pub(crate) fn log2(x: u128) -> u128 {
    let y = (127 - x.leading_zeros()) as u128;
    y + ((x > 1 << y) as u128)
}

#[cfg(test)]
pub(crate) mod tests {
    use core::{cmp::max, fmt::Debug, marker::PhantomData};
    use modinverse::modinverse;
    use num_bigint::{BigInt, ToBigInt};
    use num_traits::AsPrimitive;
    use rand::{distr::Distribution, rng, Rng};

    use super::ops::Word;
    use crate::fp::{log2, FieldOps, FP128, FP32, FP64, MAX_ROOTS};

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

    struct TestFieldParametersData<T: FieldOps<W>, W: Word> {
        /// Expected fp.p
        pub expected_p: W,
        /// Expected fp.residue(fp.g)
        pub expected_g: W,
        /// Expect fp.residue(fp.pow(fp.g, 1 << expected_log2_order)) == 1
        pub expected_log2_order: usize,

        phantom: PhantomData<T>,
    }

    impl<T, W> TestFieldParametersData<T, W>
    where
        T: FieldOps<W>,
        W: Word + AsPrimitive<u128> + ToBigInt + for<'a> TryFrom<&'a BigInt> + Debug,
        for<'a> <W as TryFrom<&'a BigInt>>::Error: Debug,
    {
        fn all_field_parameters_tests(&self) {
            self.check_generator();
            self.check_consistency();
            self.arithmetic_test();
        }

        // Check that the generator has the correct order.
        fn check_generator(&self) {
            assert_eq!(
                T::residue(T::pow(T::G, W::ONE << self.expected_log2_order)),
                W::ONE
            );
            assert_ne!(
                T::residue(T::pow(T::G, W::ONE << (self.expected_log2_order / 2))),
                W::ONE
            );
        }

        // Check that the field parameters have been constructed properly.
        fn check_consistency(&self) {
            assert_eq!(T::PRIME, self.expected_p, "p mismatch");

            let u128_p = T::PRIME.as_();
            let base = 1i128 << T::LOG2_BASE;
            let mu = match modinverse((-(u128_p as i128)).rem_euclid(base), base) {
                Some(mu) => mu as u128,
                None => panic!("inverse of -p (mod base) is undefined"),
            };
            assert_eq!(T::MU.as_(), mu, "mu mismatch");

            let big_p = &u128_p.to_bigint().unwrap();
            let big_radix = BigInt::from(1) << T::LOG2_RADIX;
            let big_r: &BigInt = &(big_radix % big_p);
            let big_r2: &BigInt = &(&(big_r * big_r) % big_p);
            let mut it = big_r2.iter_u64_digits();
            let mut r2 = 0;
            r2 |= it.next().unwrap() as u128;
            if let Some(x) = it.next() {
                r2 |= (x as u128) << 64;
            }
            assert_eq!(T::R2.as_(), r2, "r2 mismatch");

            assert_eq!(T::G, T::montgomery(self.expected_g), "g mismatch");
            assert_eq!(
                T::residue(T::pow(T::G, W::ONE << self.expected_log2_order)),
                W::ONE,
                "g order incorrect"
            );

            let num_roots = self.expected_log2_order;
            assert_eq!(T::NUM_ROOTS, num_roots, "num_roots mismatch");

            let mut roots = vec![W::ZERO; max(num_roots, MAX_ROOTS) + 1];
            roots[num_roots] = T::montgomery(self.expected_g);
            for i in (0..num_roots).rev() {
                roots[i] = T::mul(roots[i + 1], roots[i + 1]);
            }
            assert_eq!(T::ROOTS, &roots[..MAX_ROOTS + 1], "roots mismatch");
            assert_eq!(T::residue(T::ROOTS[0]), W::ONE, "first root is not one");

            let bit_mask = (BigInt::from(1) << big_p.bits()) - BigInt::from(1);
            assert_eq!(
                T::BIT_MASK.to_bigint().unwrap(),
                bit_mask,
                "bit_mask mismatch"
            );
        }

        // Test arithmetic using the field parameters.
        fn arithmetic_test(&self) {
            let u128_p = T::PRIME.as_();
            let big_p = &u128_p.to_bigint().unwrap();
            let big_zero = &BigInt::from(0);
            let uniform = rand::distr::Uniform::try_from(0..u128_p).unwrap();
            let mut rng = rng();

            let mut weird_ints = Vec::from([
                0,
                1,
                T::BIT_MASK.as_() - u128_p,
                T::BIT_MASK.as_() - u128_p + 1,
                u128_p - 1,
            ]);
            if u128_p > u64::MAX as u128 {
                weird_ints.extend_from_slice(&[
                    u64::MAX as u128,
                    1 << 64,
                    u128_p & u64::MAX as u128,
                    u128_p & !u64::MAX as u128,
                    u128_p & !u64::MAX as u128 | 1,
                ]);
            }

            let mut generate_random = || -> (W, BigInt) {
                // Add bias to random element generation, to explore "interesting" inputs.
                let intu128 = if rng.random_ratio(1, 4) {
                    weird_ints[rng.random_range(0..weird_ints.len())]
                } else {
                    uniform.sample(&mut rng)
                };
                let bigint = intu128.to_bigint().unwrap();
                let int = W::try_from(&bigint).unwrap();
                let montgomery_domain = T::montgomery(int);
                (montgomery_domain, bigint)
            };

            for _ in 0..1000 {
                let (x, ref big_x) = generate_random();
                let (y, ref big_y) = generate_random();

                // Test addition.
                let got = T::add(x, y);
                let want = (big_x + big_y) % big_p;
                assert_eq!(T::residue(got).to_bigint().unwrap(), want);

                // Test subtraction.
                let got = T::sub(x, y);
                let want = if big_x >= big_y {
                    big_x - big_y
                } else {
                    big_p - big_y + big_x
                };
                assert_eq!(T::residue(got).to_bigint().unwrap(), want);

                // Test multiplication.
                let got = T::mul(x, y);
                let want = (big_x * big_y) % big_p;
                assert_eq!(T::residue(got).to_bigint().unwrap(), want);

                // Test inversion.
                let got = T::inv(x);
                let want = big_x.modpow(&(big_p - 2), big_p);
                assert_eq!(T::residue(got).to_bigint().unwrap(), want);
                if big_x == big_zero {
                    assert_eq!(T::residue(T::mul(got, x)), W::ZERO);
                } else {
                    assert_eq!(T::residue(T::mul(got, x)), W::ONE);
                }

                // Test negation.
                let got = T::neg(x);
                let want = (big_p - big_x) % big_p;
                assert_eq!(T::residue(got).to_bigint().unwrap(), want);
                assert_eq!(T::residue(T::add(got, x)), W::ZERO);
            }
        }
    }

    mod fp32 {
        #[test]
        fn check_field_parameters() {
            use super::*;

            TestFieldParametersData::<FP32, _> {
                expected_p: 4293918721,
                expected_g: 3925978153,
                expected_log2_order: 20,
                phantom: PhantomData,
            }
            .all_field_parameters_tests();
        }
    }

    mod fp64 {
        #[test]
        fn check_field_parameters() {
            use super::*;

            TestFieldParametersData::<FP64, _> {
                expected_p: 18446744069414584321,
                expected_g: 1753635133440165772,
                expected_log2_order: 32,
                phantom: PhantomData,
            }
            .all_field_parameters_tests();
        }
    }

    mod fp128 {
        #[test]
        fn check_field_parameters() {
            use super::*;

            TestFieldParametersData::<FP128, _> {
                expected_p: 340282366920938462946865773367900766209,
                expected_g: 145091266659756586618791329697897684742,
                expected_log2_order: 66,
                phantom: PhantomData,
            }
            .all_field_parameters_tests();
        }
    }
}
