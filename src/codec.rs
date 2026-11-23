// SPDX-License-Identifier: MPL-2.0

//! Support for encoding and decoding messages to or from the TLS wire encoding, as specified in
//! [RFC 8446, Section 3][1].
//!
//! The [`Encode`], [`Decode`], [`ParameterizedEncode`] and [`ParameterizedDecode`] traits can be
//! implemented on values that need to be encoded or decoded. Utility functions are provided to
//! encode or decode sequences of values.
//!
//! [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3

use alloc::{boxed::Box, vec::Vec};
use core::{convert::TryInto, error::Error, num::TryFromIntError};

/// An error that occurred during decoding.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CodecError {
    /// Extra data remained in the input after decoding a value.
    #[error("{0} bytes left in buffer after decoding value")]
    BytesLeftOver(usize),

    /// The length prefix of an encoded vector exceeds the amount of remaining input.
    #[error("length prefix of encoded vector overflows buffer: {0}")]
    LengthPrefixTooBig(usize),

    /// The byte length of a vector exceeded the range of its length prefix.
    #[error("vector length exceeded range of length prefix")]
    LengthPrefixOverflow,

    /// Custom errors from [`Decode`] implementations.
    #[error("other error: {0}")]
    Other(#[source] Box<dyn Error + 'static + Send + Sync>),

    /// An invalid value was decoded.
    #[error("unexpected value")]
    UnexpectedValue,

    /// The input was too short to decode as a value.
    #[error("unexpected EOF")]
    UnexpectedEof,
}

/// Describes how to decode an object from a byte sequence.
pub trait Decode: Sized {
    /// Read and decode an encoded object from `bytes`. On success, the decoded value is returned
    /// and `bytes` is advanced by the encoded size of the value. On failure, an error is returned
    /// and no further attempt to read from `bytes` should be made.
    fn decode(bytes: &mut Input) -> Result<Self, CodecError>;

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
    fn decode_with_param(decoding_parameter: &P, bytes: &mut Input) -> Result<Self, CodecError>;

    /// Convenience method to get a decoded value. Returns an error if [`Self::decode_with_param`]
    /// fails, or if there are any bytes left in `bytes` after decoding a value.
    fn get_decoded_with_param(decoding_parameter: &P, bytes: &[u8]) -> Result<Self, CodecError> {
        let mut input = Input::new(bytes);
        let decoded = Self::decode_with_param(decoding_parameter, &mut input)?;
        if !input.is_empty() {
            return Err(CodecError::BytesLeftOver(input.len()));
        }

        Ok(decoded)
    }
}

/// Provide a blanket implementation so that any [`Decode`] can be used as a
/// `ParameterizedDecode<T>` for any `T`.
impl<D: Decode, T> ParameterizedDecode<T> for D {
    fn decode_with_param(_decoding_parameter: &T, bytes: &mut Input) -> Result<Self, CodecError> {
        Self::decode(bytes)
    }
}

/// Input slice for decoding operations.
///
/// This is a newtype around a byte slice, with convenience methods that split off the front of the
/// slice, and update `self` to be the rest of the slice.
pub struct Input<'a>(&'a [u8]);

impl<'a> Input<'a> {
    /// Construct an input from a byte slice.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }

    /// Split off some number of bytes from the front of the input, and return them.
    pub fn split(&mut self, length: usize) -> Result<&'a [u8], CodecError> {
        let Some((chunk, rest)) = self.0.split_at_checked(length) else {
            return Err(CodecError::UnexpectedEof);
        };
        self.0 = rest;
        Ok(chunk)
    }

    /// Split of a constant number of bytes from the front of the input, and return them.
    pub fn split_const<const N: usize>(&mut self) -> Result<&'a [u8; N], CodecError> {
        let Some((chunk, rest)) = self.0.split_first_chunk::<N>() else {
            return Err(CodecError::UnexpectedEof);
        };
        self.0 = rest;
        Ok(chunk)
    }

    /// Check if the input is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the length of the remaining input.
    pub fn len(&self) -> usize {
        self.0.len()
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
    fn decode(_bytes: &mut Input) -> Result<Self, CodecError> {
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
    fn decode(bytes: &mut Input) -> Result<Self, CodecError> {
        let value = bytes.split_const::<1>()?;
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
    fn decode(bytes: &mut Input) -> Result<Self, CodecError> {
        let value = bytes.split_const::<2>()?;
        Ok(u16::from_be_bytes(*value))
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
    fn decode(bytes: &mut Input) -> Result<Self, CodecError> {
        let value = bytes.split_const::<4>()?;
        Ok(u32::from_be_bytes(*value))
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
    fn decode(bytes: &mut Input) -> Result<Self, CodecError> {
        let value = bytes.split_const::<8>()?;
        Ok(u64::from_be_bytes(*value))
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

/// Encode `items` into `bytes` as a [fixed-length vector][1], with no length tag.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn encode_fixlen_items<E: Encode>(bytes: &mut Vec<u8>, items: &[E]) -> Result<(), CodecError> {
    for item in items {
        item.encode(bytes)?;
    }

    Ok(())
}

/// Encode `items` into `bytes` as a [variable-length vector][1] with a maximum length of `0xff`.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn encode_u8_items<P, E: ParameterizedEncode<P>>(
    bytes: &mut Vec<u8>,
    encoding_parameter: &P,
    items: &[E],
) -> Result<(), CodecError> {
    // Reserve space to later write length
    let len_offset = bytes.len();
    bytes.push(0);

    for item in items {
        item.encode_with_param(encoding_parameter, bytes)?;
    }

    let len =
        u8::try_from(bytes.len() - len_offset - 1).map_err(|_| CodecError::LengthPrefixOverflow)?;
    bytes[len_offset] = len;
    Ok(())
}

/// Decode `bytes` into a vector of `D` values, treating `bytes` as a [variable-length vector][1] of
/// maximum length `0xff`.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn decode_u8_items<P, D: ParameterizedDecode<P>>(
    decoding_parameter: &P,
    bytes: &mut Input,
) -> Result<Vec<D>, CodecError> {
    // Read one byte to get length of opaque byte vector
    let length = usize::from(u8::decode(bytes)?);

    decode_fixlen_items(length, decoding_parameter, bytes)
}

/// Encode `items` into `bytes` as a [variable-length vector][1] with a maximum length of `0xffff`.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn encode_u16_items<P, E: ParameterizedEncode<P>>(
    bytes: &mut Vec<u8>,
    encoding_parameter: &P,
    items: &[E],
) -> Result<(), CodecError> {
    // Reserve space to later write length
    let len_offset = bytes.len();
    0u16.encode(bytes)?;

    for item in items {
        item.encode_with_param(encoding_parameter, bytes)?;
    }

    let len = u16::try_from(bytes.len() - len_offset - 2)
        .map_err(|_| CodecError::LengthPrefixOverflow)?;
    bytes[len_offset..len_offset + 2].copy_from_slice(&len.to_be_bytes());
    Ok(())
}

/// Decode `bytes` into a vector of `D` values, treating `bytes` as a [variable-length vector][1] of
/// maximum length `0xffff`.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn decode_u16_items<P, D: ParameterizedDecode<P>>(
    decoding_parameter: &P,
    bytes: &mut Input,
) -> Result<Vec<D>, CodecError> {
    // Read two bytes to get length of opaque byte vector
    let length = usize::from(u16::decode(bytes)?);

    decode_fixlen_items(length, decoding_parameter, bytes)
}

/// Encode `items` into `bytes` as a [variable-length vector][1] with a maximum length of
/// `0xffffffff`.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn encode_u32_items<P, E: ParameterizedEncode<P>>(
    bytes: &mut Vec<u8>,
    encoding_parameter: &P,
    items: &[E],
) -> Result<(), CodecError> {
    // Reserve space to later write length
    let len_offset = bytes.len();
    0u32.encode(bytes)?;

    for item in items {
        item.encode_with_param(encoding_parameter, bytes)?;
    }

    let len = u32::try_from(bytes.len() - len_offset - 4)
        .map_err(|_| CodecError::LengthPrefixOverflow)?;
    bytes[len_offset..len_offset + 4].copy_from_slice(&len.to_be_bytes());
    Ok(())
}

/// Decode `bytes` into a vector of `D` values, treating `bytes` as a [variable-length vector][1] of
/// maximum length `0xffffffff`.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn decode_u32_items<P, D: ParameterizedDecode<P>>(
    decoding_parameter: &P,
    bytes: &mut Input,
) -> Result<Vec<D>, CodecError> {
    // Read four bytes to get length of opaque byte vector.
    let len: usize = u32::decode(bytes)?
        .try_into()
        .map_err(|err: TryFromIntError| CodecError::Other(err.into()))?;

    decode_fixlen_items(len, decoding_parameter, bytes)
}

/// Decode `bytes` as a [fixed-length vector][1] into as many instances of `D` as possible.
///
/// [1]: https://datatracker.ietf.org/doc/html/rfc8446#section-3.4
pub fn decode_fixlen_items<P, D: ParameterizedDecode<P>>(
    length: usize,
    decoding_parameter: &P,
    bytes: &mut Input,
) -> Result<Vec<D>, CodecError> {
    let mut decoded = Vec::new();

    // Split the specified protion of the input buffer to ensure we can't read past length.
    let mut contents = Input::new(
        bytes
            .split(length)
            .map_err(|_| CodecError::LengthPrefixTooBig(length))?,
    );

    while !contents.is_empty() {
        decoded.push(D::decode_with_param(decoding_parameter, &mut contents)?);
    }

    Ok(decoded)
}

#[cfg(test)]
mod tests {
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

        let decoded = u8::decode(&mut Input::new(&bytes)).unwrap();
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

        let decoded = u16::decode(&mut Input::new(&bytes)).unwrap();
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

        let decoded = u32::decode(&mut Input::new(&bytes)).unwrap();
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

        let decoded = u64::decode(&mut Input::new(&bytes)).unwrap();
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
        fn decode(bytes: &mut Input) -> Result<Self, CodecError> {
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

        let decoded = TestMessage::decode(&mut Input::new(&bytes)).unwrap();
        assert_eq!(value, decoded);
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
        let values = messages_vec();
        let mut bytes = vec![];
        encode_u8_items(&mut bytes, &(), &values).unwrap();

        assert_eq!(
            bytes.len(),
            // Length of opaque vector
            1 +
            // 3 TestMessage values
            3 * TestMessage::encoded_length()
        );

        let decoded = decode_u8_items(&(), &mut Input::new(&bytes)).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn roundtrip_variable_length_u16() {
        let values = messages_vec();
        let mut bytes = vec![];
        encode_u16_items(&mut bytes, &(), &values).unwrap();

        assert_eq!(
            bytes.len(),
            // Length of opaque vector
            2 +
            // 3 TestMessage values
            3 * TestMessage::encoded_length()
        );

        // Check endianness of encoded length
        assert_eq!(bytes[0..2], [0, 3 * TestMessage::encoded_length() as u8]);

        let decoded = decode_u16_items(&(), &mut Input::new(&bytes)).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn roundtrip_variable_length_u32() {
        let values = messages_vec();
        let mut bytes = Vec::new();
        encode_u32_items(&mut bytes, &(), &values).unwrap();

        assert_eq!(bytes.len(), 4 + 3 * TestMessage::encoded_length());

        // Check endianness of encoded length.
        assert_eq!(
            bytes[0..4],
            [0, 0, 0, 3 * TestMessage::encoded_length() as u8]
        );

        let decoded = decode_u32_items(&(), &mut Input::new(&bytes)).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn roundtrip_fixlen_vector() {
        let values = messages_vec();
        let mut bytes = Vec::new();
        encode_fixlen_items(&mut bytes, &values).unwrap();

        let decoded = decode_fixlen_items(bytes.len(), &(), &mut Input::new(&bytes)).unwrap();
        assert_eq!(values, decoded);

        // too short
        assert_matches!(
            decode_fixlen_items::<_, TestMessage>(bytes.len() - 1, &(), &mut Input::new(&bytes))
                .unwrap_err(),
            CodecError::UnexpectedEof
        );

        // too long
        assert_matches!(
            decode_fixlen_items::<_, TestMessage>(bytes.len() + 1, &(), &mut Input::new(&bytes))
                .unwrap_err(),
            CodecError::LengthPrefixTooBig(_)
        );
    }

    #[test]
    fn decode_too_short() {
        let values = messages_vec();
        let mut bytes = Vec::new();
        encode_u32_items(&mut bytes, &(), &values).unwrap();

        let error =
            decode_u32_items::<_, TestMessage>(&(), &mut Input::new(&bytes[..3])).unwrap_err();
        assert_matches!(error, CodecError::UnexpectedEof);

        let error =
            decode_u32_items::<_, TestMessage>(&(), &mut Input::new(&bytes[..4])).unwrap_err();
        assert_matches!(error, CodecError::LengthPrefixTooBig(_));
    }

    #[test]
    fn decode_items_overflow() {
        let mut input = Input::new(&[]);

        assert_matches!(
            decode_fixlen_items::<(), u8>(usize::MAX, &(), &mut input).unwrap_err(),
            CodecError::LengthPrefixTooBig(usize::MAX)
        );
    }

    #[test]
    fn decode_items_too_big() {
        let mut input = Input::new(&[]);

        assert_matches!(
            decode_fixlen_items::<(), u8>(2, &(), &mut input).unwrap_err(),
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

    #[test]
    fn encode_length_prefix_overflow() {
        let mut bytes = Vec::new();
        let error = encode_u8_items(&mut bytes, &(), &[1u8; u8::MAX as usize + 1]).unwrap_err();
        assert_matches!(error, CodecError::LengthPrefixOverflow);
    }
}
