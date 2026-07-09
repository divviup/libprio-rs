// SPDX-License-Identifier: MPL-2.0

//! Support for encoding and decoding messages to or from the TLS wire encoding, as specified in
//! [RFC 8446, Section 3][1].
//!
//! The [`Encode`], [`Decode`], [`ParameterizedEncode`] and [`ParameterizedDecode`] traits can be
//! implemented on values that need to be encoded or decoded. Utility functions are provided to
//! encode or decode sequences of values.
//!
//! [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3

use byteorder::{BigEndian, ReadBytesExt};
use num_traits::{bounds::UpperBounded, ConstZero, ToBytes};
use std::{
    convert::TryInto,
    error::Error,
    io::{Cursor, Read},
    marker::PhantomData,
    mem::size_of,
};

/// An error that occurred during decoding.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CodecError {
    /// An I/O error.
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    /// Extra data remained in the input after decoding a value.
    #[error("{0} bytes left in buffer after decoding value")]
    BytesLeftOver(usize),

    /// The length prefix of an encoded vector exceeds the amount of remaining input.
    #[error("length prefix of encoded vector overflows buffer: {0}")]
    LengthPrefixTooBig(usize),

    /// The byte length of a vector exceeded the range of its length prefix.
    #[error("vector length exceeded range of length prefix")]
    LengthPrefixOverflow,

    /// The number of items in a variable-length vector is outside of the type's range.
    ///
    /// In TLS presentation language, variable-length vectors declare a subrange of legal lengths
    /// ([1]). Vectors of items whose length is outside this range are invalid.
    ///
    /// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
    #[error("vector length outside of type's range")]
    VectorLengthOutsideOfRange,

    /// Custom errors from [`Decode`] implementations.
    #[error("other error: {0}")]
    Other(#[source] Box<dyn Error + 'static + Send + Sync>),

    /// An invalid value was decoded.
    #[error("unexpected value")]
    UnexpectedValue,
}

/// Describes how to decode an object from a byte sequence.
pub trait Decode: Sized {
    /// Read and decode an encoded object from `bytes`. On success, the decoded value is returned
    /// and `bytes` is advanced by the encoded size of the value. On failure, an error is returned
    /// and no further attempt to read from `bytes` should be made.
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError>;

    /// Convenience method to get a decoded value. Returns an error if [`Self::decode`] fails, or if
    /// there are any bytes left in `bytes` after decoding a value.
    fn get_decoded(bytes: &[u8]) -> Result<Self, CodecError> {
        Self::get_decoded_with_param(&(), bytes)
    }
}

/// Describes how to decode an object from a byte sequence and a decoding parameter that provides
/// additional context.
pub trait ParameterizedDecode<P>: Sized {
    /// Read and decode an encoded object from `bytes`. `decoding_parameter` provides details of the
    /// wire encoding such as lengths of different portions of the message. On success, the decoded
    /// value is returned and `bytes` is advanced by the encoded size of the value. On failure, an
    /// error is returned and no further attempt to read from `bytes` should be made.
    fn decode_with_param(
        decoding_parameter: &P,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError>;

    /// Convenience method to get a decoded value. Returns an error if [`Self::decode_with_param`]
    /// fails, or if there are any bytes left in `bytes` after decoding a value.
    fn get_decoded_with_param(decoding_parameter: &P, bytes: &[u8]) -> Result<Self, CodecError> {
        let mut cursor = Cursor::new(bytes);
        let decoded = Self::decode_with_param(decoding_parameter, &mut cursor)?;
        if cursor.position() as usize != bytes.len() {
            return Err(CodecError::BytesLeftOver(
                bytes.len() - cursor.position() as usize,
            ));
        }

        Ok(decoded)
    }
}

/// Provide a blanket implementation so that any [`Decode`] can be used as a
/// `ParameterizedDecode<T>` for any `T`.
impl<D: Decode, T> ParameterizedDecode<T> for D {
    fn decode_with_param(
        _decoding_parameter: &T,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        Self::decode(bytes)
    }
}

/// Describes how to encode objects into a byte sequence.
pub trait Encode {
    /// Append the encoded form of this object to the end of `bytes`, growing the vector as needed.
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError>;

    /// Convenience method to encode a value into a new `Vec<u8>`.
    fn get_encoded(&self) -> Result<Vec<u8>, CodecError> {
        self.get_encoded_with_param(&())
    }

    /// Returns an optional hint indicating how many bytes will be required to encode this value, or
    /// `None` by default.
    fn encoded_len(&self) -> Option<usize> {
        None
    }
}

/// Describes how to encode objects into a byte sequence.
pub trait ParameterizedEncode<P> {
    /// Append the encoded form of this object to the end of `bytes`, growing the vector as needed.
    /// `encoding_parameter` provides details of the wire encoding, used to control how the value
    /// is encoded.
    fn encode_with_param(
        &self,
        encoding_parameter: &P,
        bytes: &mut Vec<u8>,
    ) -> Result<(), CodecError>;

    /// Convenience method to encode a value into a new `Vec<u8>`.
    fn get_encoded_with_param(&self, encoding_parameter: &P) -> Result<Vec<u8>, CodecError> {
        let mut ret = if let Some(length) = self.encoded_len_with_param(encoding_parameter) {
            Vec::with_capacity(length)
        } else {
            Vec::new()
        };
        self.encode_with_param(encoding_parameter, &mut ret)?;
        Ok(ret)
    }

    /// Returns an optional hint indicating how many bytes will be required to encode this value, or
    /// `None` by default.
    fn encoded_len_with_param(&self, _encoding_parameter: &P) -> Option<usize> {
        None
    }
}

/// Provide a blanket implementation so that any [`Encode`] can be used as a
/// `ParameterizedEncode<T>` for any `T`.
impl<E: Encode + ?Sized, T> ParameterizedEncode<T> for E {
    fn encode_with_param(
        &self,
        _encoding_parameter: &T,
        bytes: &mut Vec<u8>,
    ) -> Result<(), CodecError> {
        self.encode(bytes)
    }

    fn encoded_len_with_param(&self, _encoding_parameter: &T) -> Option<usize> {
        <Self as Encode>::encoded_len(self)
    }
}

impl Decode for () {
    fn decode(_bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(())
    }
}

impl Encode for () {
    fn encode(&self, _bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(0)
    }
}

impl Decode for u8 {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut value = [0u8; size_of::<u8>()];
        bytes.read_exact(&mut value)?;
        Ok(value[0])
    }
}

impl Encode for u8 {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.push(*self);
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(1)
    }
}

impl Decode for u16 {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(bytes.read_u16::<BigEndian>()?)
    }
}

impl Encode for u16 {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&u16::to_be_bytes(*self));
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(2)
    }
}

impl Decode for u32 {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(bytes.read_u32::<BigEndian>()?)
    }
}

impl Encode for u32 {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&u32::to_be_bytes(*self));
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(4)
    }
}

impl Decode for u64 {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(bytes.read_u64::<BigEndian>()?)
    }
}

impl Encode for u64 {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&u64::to_be_bytes(*self));
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(8)
    }
}

/// A variable-length vector of encoded items.
///
/// The vector has a length prefix of type `LEN`, which will be one of `u8`, `u16` or `u32`. The
/// maximum length of the vector is the maximum value of the length prefix.
///
/// A TLS presentation language declaration like `opaque buf<11..2^16-1>` would be represented by
/// `VariableLengthVector<11, u16, u8>`.
///
/// `ExampleType values<0..2^32-1>` would be `VariableLengthVector<0, u32, ExampleType>`, assuming a
/// Rust type `ExampleType` corresponding to the TLS presentation language type of that name exists.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableLengthVector<const MIN_LEN: usize, LEN, E> {
    /// Contents of the vector.
    contents: Vec<E>,
    phantom: PhantomData<LEN>,
}

impl<const MIN_LEN: usize, LEN, E> VariableLengthVector<MIN_LEN, LEN, E> {
    /// Make a new variable length vector.
    pub fn new(contents: Vec<E>) -> Self {
        Self {
            contents,
            phantom: PhantomData,
        }
    }

    /// Length of the vector
    pub fn len(&self) -> usize {
        self.contents.len()
    }

    /// Whether this vector is empty.
    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}

impl<const MIN_LEN: usize, LEN: LengthPrefix, E: Encode> Encode
    for VariableLengthVector<MIN_LEN, LEN, E>
{
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        if self.contents.len() < MIN_LEN
            || self.contents.len()
                > LEN::max_value()
                    .try_into()
                    .map_err(|_| CodecError::Other("prefix max length too big for usize".into()))?
        {
            return Err(CodecError::VectorLengthOutsideOfRange);
        }

        // Reserve space to later write length
        let len_offset = bytes.len();
        LEN::ZERO.encode(bytes)?;

        for item in &self.contents {
            item.encode(bytes)?;
        }

        let len = LEN::try_from(bytes.len() - len_offset - LEN::ENCODED_LEN)
            .map_err(|_| CodecError::LengthPrefixOverflow)?;
        bytes[len_offset..len_offset + LEN::ENCODED_LEN]
            .copy_from_slice(len.to_be_bytes().as_ref());
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            LEN::ENCODED_LEN
                + self
                    .contents
                    .iter()
                    .map(|item| item.encoded_len().unwrap_or(0))
                    .sum::<usize>(),
        )
    }
}

impl<const MIN_LEN: usize, LEN: LengthPrefix, D: Decode> Decode
    for VariableLengthVector<MIN_LEN, LEN, D>
{
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        // Read two bytes to get length of opaque byte vector
        let length: usize = LEN::decode(bytes)?
            .try_into()
            .map_err(|_| CodecError::Other("prefix max length too big for usize".into()))?;

        if length < MIN_LEN {
            return Err(CodecError::VectorLengthOutsideOfRange);
        }

        let contents = decode_fixlen_items(length, &(), bytes)?;

        Ok(contents.into())
    }
}

impl<const MIN_LEN: usize, LEN, E> AsRef<[E]> for VariableLengthVector<MIN_LEN, LEN, E> {
    fn as_ref(&self) -> &[E] {
        &self.contents
    }
}

impl<const MIN_LEN: usize, LEN, E> AsMut<[E]> for VariableLengthVector<MIN_LEN, LEN, E> {
    fn as_mut(&mut self) -> &mut [E] {
        &mut self.contents
    }
}

impl<const MIN_LEN: usize, LEN, E> From<Vec<E>> for VariableLengthVector<MIN_LEN, LEN, E> {
    fn from(contents: Vec<E>) -> Self {
        Self::new(contents)
    }
}

/// Marker trait for types that can be the length prefix of a [`VariableLengthVector`].
trait LengthPrefix:
    Encode + Decode + TryFrom<usize> + TryInto<usize> + ConstZero + UpperBounded + ToBytes
{
    /// Length of an encoded length prefix.
    const ENCODED_LEN: usize;
}

impl LengthPrefix for u8 {
    const ENCODED_LEN: usize = 1;
}

impl LengthPrefix for u16 {
    const ENCODED_LEN: usize = 2;
}

impl LengthPrefix for u32 {
    const ENCODED_LEN: usize = 4;
}

/// Encode `items` into `bytes` as a [fixed-length vector][1], with no length tag.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn encode_fixlen_items<E: Encode>(bytes: &mut Vec<u8>, items: &[E]) -> Result<(), CodecError> {
    for item in items {
        item.encode(bytes)?;
    }

    Ok(())
}

/// Decode `bytes` as a [fixed-length vector][1] into as many instances of `D` as possible.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn decode_fixlen_items<P, D: ParameterizedDecode<P>>(
    length: usize,
    decoding_parameter: &P,
    bytes: &mut Cursor<&[u8]>,
) -> Result<Vec<D>, CodecError> {
    let mut decoded = Vec::new();
    let initial_position = bytes.position() as usize;

    // Create cursor over specified portion of provided cursor to ensure we can't read past length.
    let inner = bytes.get_ref();

    // Make sure encoded length doesn't overflow usize or go past the end of provided byte buffer.
    let (items_end, overflowed) = initial_position.overflowing_add(length);
    if overflowed || items_end > inner.len() {
        return Err(CodecError::LengthPrefixTooBig(length));
    }

    let mut sub = Cursor::new(&bytes.get_ref()[initial_position..items_end]);

    while sub.position() < length as u64 {
        decoded.push(D::decode_with_param(decoding_parameter, &mut sub)?);
    }

    // Advance outer cursor by the amount read in the inner cursor
    bytes.set_position(initial_position as u64 + sub.position());

    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use super::*;
    use assert_matches::assert_matches;

    #[test]
    fn encode_nothing() {
        let mut bytes = vec![];
        ().encode(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 0);
    }

    #[test]
    fn roundtrip_u8() {
        let value = 100u8;

        let mut bytes = vec![];
        value.encode(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 1);

        let decoded = u8::decode(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn roundtrip_u16() {
        let value = 1000u16;

        let mut bytes = vec![];
        value.encode(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 2);
        // Check endianness of encoding
        assert_eq!(bytes, vec![3, 232]);

        let decoded = u16::decode(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn roundtrip_u32() {
        let value = 134_217_728u32;

        let mut bytes = vec![];
        value.encode(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 4);
        // Check endianness of encoding
        assert_eq!(bytes, vec![8, 0, 0, 0]);

        let decoded = u32::decode(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn roundtrip_u64() {
        let value = 137_438_953_472u64;

        let mut bytes = vec![];
        value.encode(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 8);
        // Check endianness of encoding
        assert_eq!(bytes, vec![0, 0, 0, 32, 0, 0, 0, 0]);

        let decoded = u64::decode(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(value, decoded);
    }

    #[derive(Debug, Eq, PartialEq)]
    struct TestMessage {
        field_u8: u8,
        field_u16: u16,
        field_u32: u32,
        field_u64: u64,
    }

    impl Encode for TestMessage {
        fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
            self.field_u8.encode(bytes)?;
            self.field_u16.encode(bytes)?;
            self.field_u32.encode(bytes)?;
            self.field_u64.encode(bytes)
        }

        fn encoded_len(&self) -> Option<usize> {
            Some(
                self.field_u8.encoded_len()?
                    + self.field_u16.encoded_len()?
                    + self.field_u32.encoded_len()?
                    + self.field_u64.encoded_len()?,
            )
        }
    }

    impl Decode for TestMessage {
        fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
            let field_u8 = u8::decode(bytes)?;
            let field_u16 = u16::decode(bytes)?;
            let field_u32 = u32::decode(bytes)?;
            let field_u64 = u64::decode(bytes)?;

            Ok(TestMessage {
                field_u8,
                field_u16,
                field_u32,
                field_u64,
            })
        }
    }

    impl TestMessage {
        fn encoded_length() -> usize {
            // u8 field
            1 +
            // u16 field
            2 +
            // u32 field
            4 +
            // u64 field
            8
        }
    }

    #[test]
    fn roundtrip_message() {
        let value = TestMessage {
            field_u8: 0,
            field_u16: 300,
            field_u32: 134_217_728,
            field_u64: 137_438_953_472,
        };

        let mut bytes = vec![];
        value.encode(&mut bytes).unwrap();
        assert_eq!(bytes.len(), TestMessage::encoded_length());
        assert_eq!(value.encoded_len().unwrap(), TestMessage::encoded_length());

        let decoded = TestMessage::decode(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn empty_variable_length_vector() {
        assert!(VariableLengthVector::<0, u8, u8>::new(vec![]).is_empty())
    }

    fn messages_vec() -> Vec<TestMessage> {
        vec![
            TestMessage {
                field_u8: 0,
                field_u16: 300,
                field_u32: 134_217_728,
                field_u64: 137_438_953_472,
            },
            TestMessage {
                field_u8: 0,
                field_u16: 300,
                field_u32: 134_217_728,
                field_u64: 137_438_953_472,
            },
            TestMessage {
                field_u8: 0,
                field_u16: 300,
                field_u32: 134_217_728,
                field_u64: 137_438_953_472,
            },
        ]
    }

    #[test]
    fn roundtrip_variable_length_u8() {
        let values = VariableLengthVector::<0, u8, _>::new(messages_vec());
        let mut bytes = vec![];
        values.encode(&mut bytes).unwrap();

        assert_eq!(
            bytes.len(),
            // Length of opaque vector
            1 +
            // 3 TestMessage values
            3 * TestMessage::encoded_length()
        );

        let decoded = VariableLengthVector::get_decoded(&bytes).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn roundtrip_variable_length_u16() {
        let values = VariableLengthVector::<0, u16, _>::new(messages_vec());
        let mut bytes = vec![];
        values.encode(&mut bytes).unwrap();

        assert_eq!(
            bytes.len(),
            // Length of opaque vector
            2 +
            // 3 TestMessage values
            3 * TestMessage::encoded_length()
        );

        // Check endianness of encoded length
        assert_eq!(bytes[0..2], [0, 3 * TestMessage::encoded_length() as u8]);

        let decoded = VariableLengthVector::get_decoded(&bytes).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn roundtrip_variable_length_u32() {
        let values = VariableLengthVector::<0, u32, _>::new(messages_vec());
        let mut bytes = Vec::new();
        values.encode(&mut bytes).unwrap();

        assert_eq!(bytes.len(), 4 + 3 * TestMessage::encoded_length());

        // Check endianness of encoded length.
        assert_eq!(
            bytes[0..4],
            [0, 0, 0, 3 * TestMessage::encoded_length() as u8]
        );

        let decoded = VariableLengthVector::get_decoded(&bytes).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn variable_length_vector_reject_too_short() {
        assert_matches!(
            VariableLengthVector::<2, u16, _>::new(vec![0u8])
                .get_encoded()
                .unwrap_err(),
            CodecError::VectorLengthOutsideOfRange
        );
    }

    #[test]
    fn variable_length_vector_reject_too_long() {
        assert_matches!(
            VariableLengthVector::<2, u16, _>::new(vec![0u8; usize::from(u16::MAX) + 1])
                .get_encoded()
                .unwrap_err(),
            CodecError::VectorLengthOutsideOfRange
        );
    }

    #[test]
    fn variable_length_vector_u8_encoded_len() {
        let vlv = VariableLengthVector::<1, u8, _>::new(messages_vec());
        assert_eq!(
            vlv.encoded_len().unwrap(),
            1 + 3 * TestMessage::encoded_length()
        );
    }

    #[test]
    fn variable_length_vector_u16_encoded_len() {
        let vlv = VariableLengthVector::<1, u16, _>::new(messages_vec());
        assert_eq!(
            vlv.encoded_len().unwrap(),
            2 + 3 * TestMessage::encoded_length()
        );
    }

    #[test]
    fn variable_length_vector_u32_encoded_len() {
        let vlv = VariableLengthVector::<1, u32, _>::new(messages_vec());
        assert_eq!(
            vlv.encoded_len().unwrap(),
            4 + 3 * TestMessage::encoded_length()
        );
    }

    #[test]
    fn roundtrip_fixlen_vector() {
        let values = messages_vec();
        let mut bytes = Vec::new();
        encode_fixlen_items(&mut bytes, &values).unwrap();

        let decoded = decode_fixlen_items(bytes.len(), &(), &mut Cursor::new(&bytes)).unwrap();
        assert_eq!(values, decoded);

        // too short
        assert_matches!(
            decode_fixlen_items::<_, TestMessage>(bytes.len() - 1, &(), &mut Cursor::new(&bytes))
                .unwrap_err(),
            CodecError::Io(e) => assert_eq!(e.kind(), ErrorKind::UnexpectedEof)
        );

        // too long
        assert_matches!(
            decode_fixlen_items::<_, TestMessage>(bytes.len() + 1, &(), &mut Cursor::new(&bytes))
                .unwrap_err(),
            CodecError::LengthPrefixTooBig(_)
        );
    }

    #[test]
    fn decode_too_short() {
        let values = VariableLengthVector::<0, u32, _>::new(messages_vec());
        let encoded = values.get_encoded().unwrap();

        let error =
            VariableLengthVector::<0, u32, TestMessage>::get_decoded(&encoded[..3]).unwrap_err();
        assert_matches!(error, CodecError::Io(e) => assert_eq!(e.kind(), ErrorKind::UnexpectedEof));

        let error =
            VariableLengthVector::<0, u32, TestMessage>::get_decoded(&encoded[..4]).unwrap_err();
        assert_matches!(error, CodecError::LengthPrefixTooBig(_));
    }

    #[test]
    fn decode_items_overflow() {
        let encoded = vec![1u8];

        let mut cursor = Cursor::new(encoded.as_slice());
        cursor.set_position(1);

        assert_matches!(
            decode_fixlen_items::<(), u8>(usize::MAX, &(), &mut cursor).unwrap_err(),
            CodecError::LengthPrefixTooBig(usize::MAX)
        );
    }

    #[test]
    fn decode_items_too_big() {
        let encoded = vec![1u8];

        let mut cursor = Cursor::new(encoded.as_slice());
        cursor.set_position(1);

        assert_matches!(
            decode_fixlen_items::<(), u8>(2, &(), &mut cursor).unwrap_err(),
            CodecError::LengthPrefixTooBig(2)
        );
    }

    #[test]
    fn length_hint_correctness() {
        assert_eq!(().encoded_len().unwrap(), ().get_encoded().unwrap().len());
        assert_eq!(0u8.encoded_len().unwrap(), 0u8.get_encoded().unwrap().len());
        assert_eq!(
            0u16.encoded_len().unwrap(),
            0u16.get_encoded().unwrap().len()
        );
        assert_eq!(
            0u32.encoded_len().unwrap(),
            0u32.get_encoded().unwrap().len()
        );
        assert_eq!(
            0u64.encoded_len().unwrap(),
            0u64.get_encoded().unwrap().len()
        );
    }

    #[test]
    fn get_decoded_leftover() {
        let encoded_good = [1, 2, 3, 4];
        assert_matches!(u32::get_decoded(&encoded_good).unwrap(), 0x01020304u32);

        let encoded_bad = [1, 2, 3, 4, 5];
        let error = u32::get_decoded(&encoded_bad).unwrap_err();
        assert_matches!(error, CodecError::BytesLeftOver(1));
    }

    #[test]
    fn encoded_len_backwards_compatibility() {
        struct MyMessage;

        impl Encode for MyMessage {
            fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
                bytes.extend_from_slice(b"Hello, world");
                Ok(())
            }
        }

        assert_eq!(MyMessage.encoded_len(), None);

        assert_eq!(MyMessage.get_encoded().unwrap(), b"Hello, world");
    }
}
