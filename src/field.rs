// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic.
//!
//! Each field has an associated parameter called the "generator" that generates a multiplicative
//! subgroup of order `2^n` for some `n`.

use crate::{
    fp::{FP128, FP32, FP64, FP96},
    prng::{Prng, PrngError},
    vdaf::suite::Suite,
};
use serde::{
    de::{DeserializeOwned, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::{
    cmp::min,
    convert::{TryFrom, TryInto},
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    io::{Cursor, Read},
    marker::PhantomData,
    ops::{Add, AddAssign, BitAnd, Div, DivAssign, Mul, MulAssign, Neg, Shl, Shr, Sub, SubAssign},
};

/// Possible errors from finite field operations.
#[derive(Debug, thiserror::Error)]
pub enum FieldError {
    /// Input sizes do not match.
    #[error("input sizes do not match")]
    InputSizeMismatch,
    /// Returned when decoding a `FieldElement` from a short byte string.
    #[error("short read from bytes")]
    ShortRead,
    /// Returned when decoding a `FieldElement` from a byte string encoding an integer larger than
    /// or equal to the field modulus.
    #[error("read from byte slice exceeds modulus")]
    ModulusOverflow,
    /// Error while performing I/O.
    #[error("I/O error")]
    Io(#[from] std::io::Error),
}

/// Byte order for encoding FieldElement values into byte sequences.
#[derive(Clone, Copy, Debug)]
enum ByteOrder {
    /// Big endian byte order.
    BigEndian,
    /// Little endian byte order.
    LittleEndian,
}

/// Objects with this trait represent an element of `GF(p)` for some prime `p`.
pub trait FieldElement:
    Sized
    + Debug
    + Copy
    + PartialEq
    + Eq
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
    + Display
    + From<<Self as FieldElement>::Integer>
    + for<'a> TryFrom<&'a [u8], Error = FieldError>
    // NOTE Ideally we would require `Into<[u8; Self::ENCODED_SIZE]>` instead of `Into<Vec<u8>>`,
    // since the former avoids a heap allocation and can easily be converted into Vec<u8>, but that
    // isn't possible yet[1]. However we can provide the impl on FieldElement implementations.
    // [1]: https://github.com/rust-lang/rust/issues/60551
    + Into<Vec<u8>>
    + Serialize
    + DeserializeOwned
    + 'static // NOTE This bound is needed for downcasting a `dyn Gadget<F>>` to a concrete type.
{
    /// Size in bytes of the encoding of a value.
    const ENCODED_SIZE: usize;

    /// The error returned if converting `usize` to an `Integer` fails.
    type IntegerTryFromError: std::error::Error;

    /// The error returend if converting an `Integer` to a `u64` fails.
    type TryIntoU64Error: std::error::Error;

    /// The integer representation of the field element.
    type Integer: Copy
        + Debug
        + Eq
        + Ord
        + BitAnd<Output = <Self as FieldElement>::Integer>
        + Div<Output = <Self as FieldElement>::Integer>
        + Shl<Output = <Self as FieldElement>::Integer>
        + Shr<Output = <Self as FieldElement>::Integer>
        + Sub<Output = <Self as FieldElement>::Integer>
        + From<Self>
        + TryFrom<usize, Error = Self::IntegerTryFromError>
        + TryInto<u64, Error = Self::TryIntoU64Error>;

    /// Modular exponentation, i.e., `self^exp (mod p)`.
    fn pow(&self, exp: Self::Integer) -> Self;

    /// Modular inversion, i.e., `self^-1 (mod p)`. If `self` is 0, then the output is undefined.
    fn inv(&self) -> Self;

    /// Returns the prime modulus `p`.
    fn modulus() -> Self::Integer;

    /// Interprets the next [`Self::ENCODED_SIZE`] bytes from the input slice as an element of the
    /// field. The `m` most significant bits are cleared, where `m` is equal to the length of
    /// [`Self::Integer`] in bits minus the length of the modulus in bits.
    ///
    /// # Errors
    ///
    /// An error is returned if the provided slice is too small to encode a field element or if the
    /// result encodes an integer larger than the field modulus.
    ///
    /// # Warnings
    ///
    /// This function should only be used within [`prng::Prng`] to convert a random byte string into
    /// a field element. Use [`Self::try_from_reader`] to deserialize field elements. Use
    /// [`field::rand`] or [`prng::Prng`] to randomly generate field elements.
    #[doc(hidden)]
    fn try_from_random(bytes: &[u8]) -> Result<Self, FieldError>;

    /// Interprets the next [`Self::ENCODED_SIZE`] bytes from the input slice as an element of the
    /// field.
    ///
    /// # Errors
    ///
    /// An error is returned if the provided slice is too small to encode a field element or if the
    /// result encodes an integer larger than the field modulus.
    ///
    /// # Notes
    ///
    /// Ideally we would implement `TryFrom<R: Read> for FieldElement` but the stdlib's blanket
    /// implementation of `TryFrom` forbids this: <https://github.com/rust-lang/rust/issues/50133>
    fn try_from_reader<R: Read>(reader: &mut R) -> Result<Self, FieldError>;

    /// Returns the size of the multiplicative subgroup generated by `generator()`.
    fn generator_order() -> Self::Integer;

    /// Returns the generator of the multiplicative subgroup of size `generator_order()`.
    fn generator() -> Self;

    /// Returns the `2^l`-th principal root of unity for any `l <= 20`. Note that the `2^0`-th
    /// prinicpal root of unity is 1 by definition.
    fn root(l: usize) -> Option<Self>;

    /// Returns the additive identity.
    fn zero() -> Self;

    /// Returns the multiplicative identity.
    fn one() -> Self;

    /// Convert a slice of field elements into a vector of bytes.
    ///
    /// # Notes
    ///
    /// Ideally we would implement `From<&[F: FieldElement]> for Vec<u8>` or the corresponding
    /// `Into`, but the orphan rule and the stdlib's blanket implementations of `Into` make this
    /// impossible.
    fn slice_into_byte_vec(values: &[Self]) -> Vec<u8> {
        let mut vec = Vec::with_capacity(values.len() * Self::ENCODED_SIZE);
        for elem in values {
            vec.append(&mut (*elem).into());
        }
        vec
    }

    /// Convert a slice of bytes into a vector of field elements. The slice is interpreted as a
    /// sequence of [`Self::ENCODED_SIZE`]-byte sequences.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the provided byte slice is not a multiple of the size of a
    /// field element, or if any of the values in the byte slice are invalid encodings of a field
    /// element as documented in [`Self::try_from_reader`].
    ///
    /// # Notes
    ///
    /// Ideally we would implement `From<&[u8]> for Vec<F: FieldElement>` or the corresponding
    /// `Into`, but the orphan rule and the stdlib's blanket implementations of `Into` make this
    /// impossible.
    fn byte_slice_into_vec(bytes: &[u8]) -> Result<Vec<Self>, FieldError> {
        if bytes.len() % Self::ENCODED_SIZE != 0 {
            return Err(FieldError::ShortRead);
        }
        let mut vec = Vec::with_capacity(bytes.len() / Self::ENCODED_SIZE);
        for chunk in bytes.chunks_exact(Self::ENCODED_SIZE) {
            vec.push(Self::try_from_reader(&mut Cursor::new(chunk))?);
        }
        Ok(vec)
    }
}

/// serde Visitor implementation used to generically deserialize `FieldElement`
/// values from byte arrays.
struct FieldElementVisitor<F: FieldElement> {
    phantom: PhantomData<F>,
}

impl<'de, F: FieldElement> Visitor<'de> for FieldElementVisitor<F> {
    type Value = F;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_fmt(format_args!("an array of {} bytes", F::ENCODED_SIZE))
    }

    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Self::Value::try_from(v).map_err(E::custom)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut bytes = vec![];
        while let Some(byte) = seq.next_element()? {
            bytes.push(byte);
        }

        self.visit_bytes(&bytes)
    }
}

macro_rules! make_field {
    (
        $(#[$meta:meta])*
        $elem:ident, $int:ident, $fp:ident, $encoding_size:literal, $encoding_order:expr,
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialOrd, Ord, Default)]
        pub struct $elem(u128);

        impl $elem {
            /// Attempts to instantiate an `$elem` from the first `Self::ENCODED_SIZE` bytes in the
            /// provided slice. The decoded value will be bitwise-ANDed with `mask` before reducing
            /// it using the field modulus.
            ///
            /// # Errors
            ///
            /// An error is returned if the provided slice is not long enough to encode a field
            /// element or if the decoded value is greater than the field prime.
            ///
            /// # Notes
            ///
            /// We cannot use `u128::from_le_bytes` or `u128::from_be_bytes` because those functions
            /// expect inputs to be exactly 16 bytes long. Our encoding of most field elements is
            /// more compact, and does not have to correspond to the size of an integer type. For
            /// instance,`Field96`'s encoding is 12 bytes, even though it is a 16 byte `u128` in
            /// memory.
            fn try_from_bytes(bytes: &[u8], mask: u128) -> Result<Self, FieldError> {
                if Self::ENCODED_SIZE > bytes.len() {
                    return Err(FieldError::ShortRead);
                }

                let mut int = 0;
                for i in 0..Self::ENCODED_SIZE {
                    let j = match $encoding_order {
                        ByteOrder::LittleEndian => i,
                        ByteOrder::BigEndian => Self::ENCODED_SIZE - i - 1,
                    };

                    int |= (bytes[j] as u128) << (i << 3);
                }

                int &= mask;

                if int >= $fp.p {
                    return Err(FieldError::ModulusOverflow);
                }
                Ok(Self($fp.elem(int)))
            }
        }

        impl PartialEq for $elem {
            fn eq(&self, rhs: &Self) -> bool {
                // The fields included in this comparison MUST match the fields
                // used in Hash::hash
                // https://doc.rust-lang.org/std/hash/trait.Hash.html#hash-and-eq
                $fp.from_elem(self.0) == $fp.from_elem(rhs.0)
            }
        }

        impl Hash for $elem {
            fn hash<H: Hasher>(&self, state: &mut H) {
                // The fields included in this hash MUST match the fields used
                // in PartialEq::eq
                // https://doc.rust-lang.org/std/hash/trait.Hash.html#hash-and-eq
                $fp.from_elem(self.0).hash(state);
            }
        }

        impl Eq for $elem {}

        impl Add for $elem {
            type Output = $elem;
            fn add(self, rhs: Self) -> Self {
                Self($fp.add(self.0, rhs.0))
            }
        }

        impl Add for &$elem {
            type Output = $elem;
            fn add(self, rhs: Self) -> $elem {
                *self + *rhs
            }
        }

        impl AddAssign for $elem {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $elem {
            type Output = $elem;
            fn sub(self, rhs: Self) -> Self {
                Self($fp.sub(self.0, rhs.0))
            }
        }

        impl Sub for &$elem {
            type Output = $elem;
            fn sub(self, rhs: Self) -> $elem {
                *self - *rhs
            }
        }

        impl SubAssign for $elem {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl Mul for $elem {
            type Output = $elem;
            fn mul(self, rhs: Self) -> Self {
                Self($fp.mul(self.0, rhs.0))
            }
        }

        impl Mul for &$elem {
            type Output = $elem;
            fn mul(self, rhs: Self) -> $elem {
                *self * *rhs
            }
        }

        impl MulAssign for $elem {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Div for $elem {
            type Output = $elem;
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, rhs: Self) -> Self {
                self * rhs.inv()
            }
        }

        impl Div for &$elem {
            type Output = $elem;
            fn div(self, rhs: Self) -> $elem {
                *self / *rhs
            }
        }

        impl DivAssign for $elem {
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl Neg for $elem {
            type Output = $elem;
            fn neg(self) -> Self {
                Self($fp.neg(self.0))
            }
        }

        impl Neg for &$elem {
            type Output = $elem;
            fn neg(self) -> $elem {
                -(*self)
            }
        }

        impl From<$int> for $elem {
            fn from(x: $int) -> Self {
                Self($fp.elem(u128::try_from(x).unwrap()))
            }
        }

        impl From<$elem> for $int {
            fn from(x: $elem) -> Self {
                $int::try_from($fp.from_elem(x.0)).unwrap()
            }
        }

        impl PartialEq<$int> for $elem {
            fn eq(&self, rhs: &$int) -> bool {
                $fp.from_elem(self.0) == u128::try_from(*rhs).unwrap()
            }
        }

        impl<'a> TryFrom<&'a [u8]> for $elem {
            type Error = FieldError;

            fn try_from(bytes: &[u8]) -> Result<Self, FieldError> {
                Self::try_from_bytes(bytes, u128::MAX)
            }
        }

        impl From<$elem> for [u8; $elem::ENCODED_SIZE] {
            fn from(elem: $elem) -> Self {
                let int = $fp.from_elem(elem.0);
                let mut slice = [0; $elem::ENCODED_SIZE];
                for i in 0..$elem::ENCODED_SIZE {
                    let j = match $encoding_order {
                        ByteOrder::LittleEndian => i,
                        ByteOrder::BigEndian => $elem::ENCODED_SIZE - i - 1,
                    };

                    slice[j] = ((int >> (i << 3)) & 0xff) as u8;
                }
                slice
            }
        }

        impl From<$elem> for Vec<u8> {
            fn from(elem: $elem) -> Self {
                <[u8; $elem::ENCODED_SIZE]>::from(elem).to_vec()
            }
        }

        impl Display for $elem {
            fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
                write!(f, "{}", $fp.from_elem(self.0))
            }
        }

        impl Debug for $elem {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", $fp.from_elem(self.0))
            }
        }

        // We provide custom [`serde::Serialize`] and [`serde::Deserialize`] implementations because
        // the derived implementations would represent `FieldElement` values as the backing `u128`,
        // which is not what we want because (1) we can be more efficient in all cases and (2) in
        // some circumstances, [some serializers don't support `u128`](https://github.com/serde-rs/json/issues/625).
        impl Serialize for $elem {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let bytes: [u8; $elem::ENCODED_SIZE] = (*self).into();
                serializer.serialize_bytes(&bytes)
            }
        }

        impl<'de> Deserialize<'de> for $elem {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<$elem, D::Error> {
                deserializer.deserialize_bytes(FieldElementVisitor { phantom: PhantomData })
            }
        }

        impl FieldElement for $elem {
            const ENCODED_SIZE: usize = $encoding_size;
            type Integer = $int;
            type IntegerTryFromError = <Self::Integer as TryFrom<usize>>::Error;
            type TryIntoU64Error = <Self::Integer as TryInto<u64>>::Error;

            fn pow(&self, exp: Self::Integer) -> Self {
                Self($fp.pow(self.0, u128::try_from(exp).unwrap()))
            }

            fn inv(&self) -> Self {
                Self($fp.inv(self.0))
            }

            fn modulus() -> Self::Integer {
                $fp.p as $int
            }

            fn try_from_random(bytes: &[u8]) -> Result<Self, FieldError> {
                $elem::try_from_bytes(bytes, $fp.bit_mask)
            }

            fn try_from_reader<R: Read>(reader: &mut R) -> Result<Self, FieldError> {
                let mut bytes = [0u8; $elem::ENCODED_SIZE];
                reader.read_exact(&mut bytes)?;
                $elem::try_from_bytes(&bytes, u128::MAX)
            }

            fn generator() -> Self {
                Self($fp.g)
            }

            fn generator_order() -> Self::Integer {
                1 << (Self::Integer::try_from($fp.num_roots).unwrap())
            }

            fn root(l: usize) -> Option<Self> {
                if l < min($fp.roots.len(), $fp.num_roots+1) {
                    Some(Self($fp.roots[l]))
                } else {
                    None
                }
            }

            fn zero() -> Self {
                Self(0)
            }

            fn one() -> Self {
                Self($fp.roots[0])
            }
        }
    };
}

make_field!(
    /// `GF(4293918721)`, a 32-bit field.
    Field32,
    u32,
    FP32,
    4,
    ByteOrder::BigEndian,
);

make_field!(
    /// Same as Field32, but encoded in little endian for compatibility with Prio v2.
    FieldPriov2,
    u32,
    FP32,
    4,
    ByteOrder::LittleEndian,
);

make_field!(
    /// `GF(18446744069414584321)`, a 64-bit field.
    Field64,
    u64,
    FP64,
    8,
    ByteOrder::BigEndian,
);

make_field!(
    /// `GF(79228148845226978974766202881)`, a 96-bit field.
    Field96,
    u128,
    FP96,
    12,
    ByteOrder::BigEndian,
);

make_field!(
    /// `GF(340282366920938462946865773367900766209)`, a 128-bit field.
    Field128,
    u128,
    FP128,
    16,
    ByteOrder::BigEndian,
);

/// Merge two vectors of fields by summing other_vector into accumulator.
///
/// # Errors
///
/// Fails if the two vectors do not have the same length.
pub(crate) fn merge_vector<F: FieldElement>(
    accumulator: &mut [F],
    other_vector: &[F],
) -> Result<(), FieldError> {
    if accumulator.len() != other_vector.len() {
        return Err(FieldError::InputSizeMismatch);
    }
    for (a, o) in accumulator.iter_mut().zip(other_vector.iter()) {
        *a += *o;
    }

    Ok(())
}

/// Outputs an additive secret sharing of the input.
pub(crate) fn split_vector<F: FieldElement>(
    inp: &[F],
    num_shares: usize,
) -> Result<Vec<Vec<F>>, PrngError> {
    if num_shares == 0 {
        return Ok(vec![]);
    }

    let mut outp = Vec::with_capacity(num_shares);
    outp.push(inp.to_vec());

    for _ in 1..num_shares {
        let share: Vec<F> = random_vector(inp.len())?;
        for (x, y) in outp[0].iter_mut().zip(&share) {
            *x -= *y;
        }
        outp.push(share);
    }

    Ok(outp)
}

/// Generate a vector of uniform random field elements.
pub fn random_vector<F: FieldElement>(len: usize) -> Result<Vec<F>, PrngError> {
    Ok(Prng::generate(Suite::Aes128CtrHmacSha256)?
        .take(len)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::MAX_ROOTS;
    use crate::prng::Prng;
    use assert_matches::assert_matches;
    use std::io::{Cursor, Write};

    #[test]
    fn test_endianness() {
        let little_endian_encoded: [u8; FieldPriov2::ENCODED_SIZE] =
            FieldPriov2(0x12_34_56_78).into();

        let mut big_endian_encoded: [u8; Field32::ENCODED_SIZE] = Field32(0x12_34_56_78).into();
        big_endian_encoded.reverse();

        assert_eq!(little_endian_encoded, big_endian_encoded);
    }

    #[test]
    fn test_accumulate() {
        let mut lhs = vec![Field32(1); 10];
        let rhs = vec![Field32(2); 10];

        merge_vector(&mut lhs, &rhs).unwrap();

        lhs.iter().for_each(|f| assert_eq!(*f, Field32(3)));
        rhs.iter().for_each(|f| assert_eq!(*f, Field32(2)));

        let wrong_len = vec![Field32::zero(); 9];
        let result = merge_vector(&mut lhs, &wrong_len);
        assert_matches!(result, Err(FieldError::InputSizeMismatch));
    }

    // Some of the checks in this function, like `assert_eq!(one - one, zero)`
    // or `assert_eq!(two / two, one)` trip this clippy lint for tautological
    // comparisons, but we have a legitimate need to verify these basics. We put
    // the #[allow] on the whole function since "attributes on expressions are
    // experimental" https://github.com/rust-lang/rust/issues/15701
    #[allow(clippy::eq_op)]
    fn field_element_test<F: FieldElement>() {
        let mut prng: Prng<F> = Prng::generate(Suite::Aes128CtrHmacSha256).unwrap();
        let int_modulus = F::modulus();
        let int_one = F::Integer::try_from(1).unwrap();
        let zero = F::zero();
        let one = F::one();
        let two = F::from(F::Integer::try_from(2).unwrap());
        let four = F::from(F::Integer::try_from(4).unwrap());

        // add
        assert_eq!(F::from(int_modulus - int_one) + one, zero);
        assert_eq!(one + one, two);
        assert_eq!(two + F::from(int_modulus), two);

        // sub
        assert_eq!(zero - one, F::from(int_modulus - int_one));
        assert_eq!(one - one, zero);
        assert_eq!(two - F::from(int_modulus), two);
        assert_eq!(one - F::from(int_modulus - int_one), two);

        // add + sub
        for _ in 0..100 {
            let f = prng.get();
            let g = prng.get();
            assert_eq!(f + g - f - g, zero);
            assert_eq!(f + g - g, f);
            assert_eq!(f + g - f, g);
        }

        // mul
        assert_eq!(two * two, four);
        assert_eq!(two * one, two);
        assert_eq!(two * zero, zero);
        assert_eq!(one * F::from(int_modulus), zero);

        // div
        assert_eq!(four / two, two);
        assert_eq!(two / two, one);
        assert_eq!(zero / two, zero);
        assert_eq!(two / zero, zero); // Undefined behavior
        assert_eq!(zero.inv(), zero); // Undefined behavior

        // mul + div
        for _ in 0..100 {
            let f = prng.get();
            if f == zero {
                continue;
            }
            assert_eq!(f * f.inv(), one);
            assert_eq!(f.inv() * f, one);
        }

        // pow
        assert_eq!(two.pow(F::Integer::try_from(0).unwrap()), one);
        assert_eq!(two.pow(int_one), two);
        assert_eq!(two.pow(F::Integer::try_from(2).unwrap()), four);
        assert_eq!(two.pow(int_modulus - int_one), one);
        assert_eq!(two.pow(int_modulus), two);

        // roots
        let mut int_order = F::generator_order();
        for l in 0..MAX_ROOTS + 1 {
            assert_eq!(
                F::generator().pow(int_order),
                F::root(l).unwrap(),
                "failure for F::root({})",
                l
            );
            int_order = int_order >> int_one;
        }

        // serialization
        let test_inputs = vec![zero, one, prng.get(), F::from(int_modulus - int_one)];
        for want in test_inputs.iter() {
            println!("check {:?}", want);
            let mut bytes: Vec<u8> = vec![];
            bytes.write_all(&(*want).into()).unwrap();

            assert_eq!(bytes.len(), F::ENCODED_SIZE);

            let got = F::try_from_reader(&mut Cursor::new(&bytes)).unwrap();
            assert_eq!(got, *want);
            assert_eq!(bytes.len(), F::ENCODED_SIZE);
        }

        let serialized_vec = F::slice_into_byte_vec(&test_inputs);
        let deserialized = F::byte_slice_into_vec(&serialized_vec).unwrap();
        assert_eq!(deserialized, test_inputs);
    }

    #[test]
    fn test_field32() {
        field_element_test::<Field32>();
    }

    #[test]
    fn test_field_priov2() {
        field_element_test::<FieldPriov2>();
    }

    #[test]
    fn test_field64() {
        field_element_test::<Field64>();
    }

    #[test]
    fn test_field96() {
        field_element_test::<Field96>();
    }

    #[test]
    fn test_field128() {
        field_element_test::<Field128>();
    }
}
