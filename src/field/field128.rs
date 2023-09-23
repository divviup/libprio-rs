// Copyright (c) 2023 ISRG
// Copyright (c) 2023 Cloudflare Inc.
//
// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for `GF((2^62-2^3+1)*2^66+1)`.

use super::{
    fiat_crypto_fp128::{
        fp128_add, fp128_from_bytes, fp128_from_montgomery, fp128_mul, fp128_opp, fp128_sub,
        fp128_to_bytes, fp128_to_montgomery, Fp128MontgomeryDomainFieldElement,
        Fp128NonMontgomeryDomainFieldElement,
    },
    FftFriendlyFieldElement, FieldElement, FieldElementVisitor, FieldElementWithInteger,
    FieldError,
};
use crate::codec::{CodecError, Decode, Encode};
use impl_ops::impl_op;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    convert::{TryFrom, TryInto},
    fmt::{Debug, Display, Formatter},
    hash::{Hash, Hasher},
    io::{Cursor, Read},
    marker::PhantomData,
    ops,
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

impl From<Fp128NonMontgomeryDomainFieldElement> for u128 {
    fn from(x: Fp128NonMontgomeryDomainFieldElement) -> Self {
        ((x.0[1] as u128) << 64) | (x.0[0] as u128)
    }
}

impl From<u128> for Fp128NonMontgomeryDomainFieldElement {
    // This From function tolerates values in the full range of u128,
    // those above the prime modulus are reduced modulo p.
    fn from(x: u128) -> Self {
        let mut v = x;
        if x >= FIELD128_PRIME {
            v = x % FIELD128_PRIME;
        }
        Self([v as u64, (v >> 64) as u64])
    }
}

impl From<Fp128MontgomeryDomainFieldElement> for Fp128NonMontgomeryDomainFieldElement {
    fn from(x: Fp128MontgomeryDomainFieldElement) -> Self {
        let mut z = Self(Default::default());
        fp128_from_montgomery(&mut z, &x);
        z
    }
}

impl From<Fp128NonMontgomeryDomainFieldElement> for Fp128MontgomeryDomainFieldElement {
    fn from(x: Fp128NonMontgomeryDomainFieldElement) -> Self {
        let mut z = Self(Default::default());
        fp128_to_montgomery(&mut z, &x);
        z
    }
}

impl From<Fp128MontgomeryDomainFieldElement> for u128 {
    fn from(x: Fp128MontgomeryDomainFieldElement) -> Self {
        u128::from(Fp128NonMontgomeryDomainFieldElement::from(x))
    }
}

impl From<u128> for Fp128MontgomeryDomainFieldElement {
    fn from(x: u128) -> Self {
        Self::from(Fp128NonMontgomeryDomainFieldElement::from(x))
    }
}

impl PartialEq for Fp128MontgomeryDomainFieldElement {
    fn eq(&self, rhs: &Self) -> bool {
        self.0[0] == rhs.0[0] && self.0[1] == rhs.0[1]
    }
}

impl Eq for Fp128MontgomeryDomainFieldElement {}

impl Hash for Fp128MontgomeryDomainFieldElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]

/// Field128 represents an element of the prime field GF((2^62-2^3+1)*2^66+1).
///
/// Internally, elements use Montgomery representation and its
/// implementation is provided by fiat-crypto tool that produces
/// formally-verified prime field arithmetic.
pub struct Field128(Fp128MontgomeryDomainFieldElement);

/// safeFpFromConst constructs a Field128 given constant values that
/// represent an element in Montgomery representation.
macro_rules! safeFpFromConst {
    ($u0:expr, $u1:expr) => {
        Field128(Fp128MontgomeryDomainFieldElement([$u0, $u1]))
    };
}

impl Default for Field128 {
    #[inline(always)]
    fn default() -> Self {
        safeFpFromConst!(u64::default(), u64::default()) // returns zero.
    }
}

impl From<Fp128MontgomeryDomainFieldElement> for Field128 {
    #[inline(always)]
    fn from(x: Fp128MontgomeryDomainFieldElement) -> Self {
        Self(x)
    }
}

impl ConstantTimeEq for Field128 {
    #[inline(always)]
    fn ct_eq(&self, rhs: &Self) -> Choice {
        u64::ct_eq(&self.0[0], &rhs.0[0]) & u64::ct_eq(&self.0[1], &rhs.0[1])
    }
}

impl ConditionallySelectable for Field128 {
    #[inline(always)]
    fn conditional_select(a: &Self, b: &Self, c: Choice) -> Self {
        safeFpFromConst!(
            u64::conditional_select(&a.0[0], &b.0[0], c),
            u64::conditional_select(&a.0[1], &b.0[1], c)
        )
    }
}

impl_op!(-|a: Field128| -> Field128 {
    let mut out = Self::default();
    fp128_opp(&mut out.0, &a.0);
    out
});

impl_op!(+|a:Field128, b:Field128| -> Field128 {
    let mut out = Self::default();
    fp128_add(&mut out.0, &a.0, &b.0);
    out
});

impl_op!(-|a: Field128, b: Field128| -> Field128 {
    let mut out = Self::default();
    fp128_sub(&mut out.0, &a.0, &b.0);
    out
});

impl_op!(*|a: Field128, b: Field128| -> Field128 {
    let mut out = Self::default();
    fp128_mul(&mut out.0, &a.0, &b.0);
    out
});

impl_op!(/|a: Field128, b: Field128| -> Field128 { a * b.inv() });

impl_op!(-|a: &Field128| -> Field128 { -(*a) });
impl_op!(+|a: &Field128, b: &Field128| -> Field128 { *a + *b });
impl_op!(-|a: &Field128, b: &Field128| -> Field128 { *a - *b });
impl_op!(*|a: &Field128, b: &Field128| -> Field128 { *a * *b });
impl_op!(/|a: &Field128, b: &Field128| -> Field128 { *a / *b });

impl_op!(+= |a:&mut Field128, b:Field128| {*a = *a + b});
impl_op!(-= |a:&mut Field128, b:Field128| {*a = *a - b});
impl_op!(*= |a:&mut Field128, b:Field128| {*a = *a * b});
impl_op!(/= |a:&mut Field128, b:Field128| {*a = *a / b});

impl From<u128> for Field128 {
    fn from(x: u128) -> Self {
        Self::from(Fp128MontgomeryDomainFieldElement::from(x))
    }
}

impl From<Field128> for u128 {
    fn from(x: Field128) -> Self {
        u128::from(Fp128NonMontgomeryDomainFieldElement::from(x.0))
    }
}

impl<'a> TryFrom<&'a [u8]> for Field128 {
    type Error = FieldError;
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let array: [u8; FIELD128_ENCODED_SIZE] =
            bytes.try_into().map_err(|_| FieldError::ShortRead)?;

        if u128::from_le_bytes(array) >= FIELD128_PRIME {
            return Err(FieldError::ModulusOverflow);
        }

        let mut non_mont = Fp128NonMontgomeryDomainFieldElement(Default::default());
        fp128_from_bytes(&mut non_mont.0, &array);
        Ok(Field128(Fp128MontgomeryDomainFieldElement::from(non_mont)))
    }
}

impl From<Field128> for [u8; FIELD128_ENCODED_SIZE] {
    fn from(elem: Field128) -> Self {
        let mut slice = Self::default();
        let non_mont: Fp128NonMontgomeryDomainFieldElement = elem.0.into();
        fp128_to_bytes(&mut slice, &non_mont.0);
        slice
    }
}

impl From<Field128> for Vec<u8> {
    fn from(elem: Field128) -> Self {
        <[u8; FIELD128_ENCODED_SIZE]>::from(elem).to_vec()
    }
}

impl Display for Field128 {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", u128::from(*self))
    }
}

impl Debug for Field128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", u128::from(*self))
    }
}

impl Serialize for Field128 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes: [u8; FIELD128_ENCODED_SIZE] = (*self).into();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Field128 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_bytes(FieldElementVisitor {
            phantom: PhantomData,
        })
    }
}

impl Encode for Field128 {
    fn encode(&self, bytes: &mut Vec<u8>) {
        let slice = <[u8; FIELD128_ENCODED_SIZE]>::from(*self);
        bytes.extend_from_slice(&slice);
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(FIELD128_ENCODED_SIZE)
    }
}

impl Decode for Field128 {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut value = [0u8; FIELD128_ENCODED_SIZE];
        bytes.read_exact(&mut value)?;
        Field128::try_from(value.as_slice()).map_err(|e| {
            CodecError::Other(Box::new(e) as Box<dyn std::error::Error + 'static + Send + Sync>)
        })
    }
}

impl FieldElement for Field128 {
    const ENCODED_SIZE: usize = FIELD128_ENCODED_SIZE;

    fn inv(&self) -> Self {
        self.pow(FIELD128_PRIME - 2)
    }

    fn try_from_random(bytes: &[u8]) -> Result<Self, FieldError> {
        Field128::try_from(bytes)
    }

    fn zero() -> Self {
        Self::default()
    }

    fn one() -> Self {
        // By definition, FIELD128_UROOTS[0] = 1.
        FIELD128_UROOTS[0]
    }
}

impl FieldElementWithInteger for Field128 {
    type Integer = u128;
    type IntegerTryFromError = <Self::Integer as TryFrom<usize>>::Error;
    type TryIntoU64Error = <Self::Integer as TryInto<u64>>::Error;

    // pow is a non-constant-time operation with respect to the bits of the exponent.
    fn pow(&self, exp: Self::Integer) -> Self {
        let mut t = Self::one();
        for i in (0..u128::BITS - exp.leading_zeros()).rev() {
            t *= t;
            if (exp >> i) & 1 != 0 {
                t *= *self;
            }
        }
        t
    }

    fn modulus() -> Self::Integer {
        FIELD128_PRIME
    }
}

impl FftFriendlyFieldElement for Field128 {
    fn generator() -> Self {
        FIELD128_PRIMITIVE_UROOT
    }

    fn generator_order() -> Self::Integer {
        1 << FIELD128_NUM_UROOTS
    }

    fn root(l: usize) -> Option<Self> {
        if l < FIELD128_UROOTS.len() {
            Some(FIELD128_UROOTS[l])
        } else {
            None
        }
    }
}

/// The prime modulus `p=(2^62-2^3+1)*2^66+1`.
const FIELD128_PRIME: u128 = 340282366920938462946865773367900766209;

/// Size in bytes used to store a Field128 element.
const FIELD128_ENCODED_SIZE: usize = 16;

/// The `2^num_roots`-th -principal root of unity. This element is used to generate the
/// elements of `roots`.
///
/// In sage this is calculated as:
///   PRIMITIVE_UROOT = GF(p).primitive_element()^(2^62-2^3+1)
///                   = 7^(2^62-2^3+1)
///                   = 145091266659756586618791329697897684742
/// Then, converted to Montgomery domain with R=2^128.
///   toMont = lambda x: x*2**128
///   FIELD128_PRIMITIVE_UROOT = toMont(PRIMITIVE_UROOT)
///                            = 0x50f8f7f554db309cf0111fb98c6b9875
static FIELD128_PRIMITIVE_UROOT: Field128 =
    safeFpFromConst!(0xf0111fb98c6b9875, 0x50f8f7f554db309c);

/// The number of principal roots of unity in `roots`.
const FIELD128_NUM_UROOTS: usize = 66;

/// `FIELD128_UROOTS[l]` is the `2^l`-th principal root of unity, i.e., `roots[l]` has order `2^l` in the
/// multiplicative group. `roots[0]` is equal to one by definition.
///
/// In sage this is calculated as:
///   PRIMITIVE_UROOT = GF(p).primitive_element()^(2^62-2^3+1)
///   toMont = lambda x: x*2**128
///   toHex = lambda x: list(map(hex, ZZ(x).digits(2**64)))
///   FIELD128_UROOTS = [ toHex(toMont(PRIMITIVE_UROOT**(2**i))) for i in range(66,66-21,-1) ]
static FIELD128_UROOTS: [Field128; 20 + 1] = [
    safeFpFromConst!(0xffffffffffffffff, 0x000000000000001b),
    safeFpFromConst!(0x0000000000000002, 0xffffffffffffffc8),
    safeFpFromConst!(0x8ff94ea745b7d9d6, 0x6171e408747992c7),
    safeFpFromConst!(0x1323bb095fba9556, 0x7f2a4e6655e5a49c),
    safeFpFromConst!(0xd1c455956aafabfc, 0x3d661493c5e89442),
    safeFpFromConst!(0x416d53fbcdfcc65c, 0x5c159e1cffd5eca0),
    safeFpFromConst!(0x1428d15e766f5f3e, 0x960d5c8696ec3aa3),
    safeFpFromConst!(0x8f0cef59f8c23f3e, 0xcce83f596bb28730),
    safeFpFromConst!(0x432d8d01ae187081, 0x12b496afe629224c),
    safeFpFromConst!(0x0edc26fa686e3d3b, 0xc2026e57b1554ea9),
    safeFpFromConst!(0x79663ccecfe8c86c, 0xf38c95d1e57405ea),
    safeFpFromConst!(0xfecc377cf0f47a9f, 0x2b486d42d73283bd),
    safeFpFromConst!(0x0ab1068497116540, 0x70866815dbf52bac),
    safeFpFromConst!(0x2d9420dc5196c01a, 0x852c9b234b09c7df),
    safeFpFromConst!(0x1a491a6cf3399115, 0xca4b831e2e621692),
    safeFpFromConst!(0xb99152aabeebd757, 0xb7fbf514f82e2269),
    safeFpFromConst!(0xb8da2f851dfd594a, 0x597c0e93a246d640),
    safeFpFromConst!(0xf2888c210ef9f1c4, 0x97f929a5d52ab886),
    safeFpFromConst!(0xe6a0ccbc956fc7fb, 0xfa4748aea960d0eb),
    safeFpFromConst!(0xe53d6d96e1ec92a0, 0xc24ed95c3c013bcc),
    safeFpFromConst!(0x6cc6e9129e1679c0, 0x6f825f88648b270c),
];
