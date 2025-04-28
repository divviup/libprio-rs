//! Helper functions to randomly generate big integers, sampled from uniform distributions.
//!
//! These were vendored from num-biguint.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::Zero;
use rand::Rng;

/// Uniform distribution producing [`BigUint`].
pub(super) struct UniformBigUint {
    len: BigUint,
    base: BigUint,
}

impl UniformBigUint {
    pub(super) fn new(low: &BigUint, high: &BigUint) -> Self {
        assert!(low < high);
        Self {
            len: high - low,
            base: low.clone(),
        }
    }

    pub(super) fn new_inclusive(low: &BigUint, high: &BigUint) -> Self {
        assert!(low <= high);
        Self {
            len: high - low + 1u32,
            base: low.clone(),
        }
    }

    pub(super) fn sample<R>(&self, rng: &mut R) -> BigUint
    where
        R: Rng + ?Sized,
    {
        &self.base + gen_biguint_below(rng, &self.len)
    }
}

/// Uniformly generate a random [`BigUint`] in the range \[0, `bound`).
fn gen_biguint_below<R>(rng: &mut R, bound: &BigUint) -> BigUint
where
    R: Rng + ?Sized,
{
    assert!(!bound.is_zero());
    let bits = bound.bits();
    loop {
        let n = gen_biguint(rng, bits);
        if n < *bound {
            return n;
        }
    }
}

/// Uniformly generate a random [`BigUint`] in the range \[0, 2^`bits`).
fn gen_biguint<R>(rng: &mut R, bit_size: u64) -> BigUint
where
    R: Rng + ?Sized,
{
    let (digits, rem) = bit_size.div_rem(&32);
    let len = usize::try_from(digits + (rem > 0) as u64).expect("capacity overflow");
    let mut data = vec![0u32; len];
    gen_bits(rng, &mut data, rem);
    BigUint::new(data)
}

/// Use [`Rng::fill`] to generate random bits.
///
/// If `rem` is greater than zero, than only the lowest `rem` bits of the last u32 are filled with
/// random data.
fn gen_bits<R>(rng: &mut R, data: &mut [u32], rem: u64)
where
    R: Rng + ?Sized,
{
    // `fill` is faster than many `gen::<u32>` calls
    rng.fill(data);
    if rem > 0 {
        let last = data.len() - 1;
        data[last] >>= 32 - rem;
    }
}
