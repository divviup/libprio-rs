// SPDX-License-Identifier: MPL-2.0

//! This module implements the cryptographic dependencies for our VDAF implementation. This
//! includes a stream cipher (here we call it a [`KeyStream`]) and a pseudorandom function whose
//! output size is the same as the key size (we call this a [`KeyDeriver`]).
//
// TODO(cjpatton) Add unit test with test vectors.

use crate::codec::{CodecError, Decode, Encode};
use aes::cipher::generic_array::GenericArray;
use aes::cipher::{FromBlockCipher, NewBlockCipher, StreamCipher as AesStreamCipher};
use aes::{Aes128, Aes128Ctr};
use getrandom::getrandom;
use ring::hmac;
use std::io::{Cursor, Read};

const BLAKE3_DERIVE_PREFIX: &[u8] = b"blake3 key derive";
const BLAKE3_STREAM_PREFIX: &[u8] = b"blake3 key stream";

/// Errors propagated by methods in this module.
#[derive(Debug, thiserror::Error)]
pub enum SuiteError {
    /// Failure when calling getrandom().
    #[error("getrandom: {0}")]
    GetRandom(#[from] getrandom::Error),
    /// Failure performing I/O
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A suite uniquely determines a [`KeyStream`] and [`KeyDeriver`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Suite {
    /// The [`KeyStream`] is implemented from AES128 in CTR mode and the [`KeyDeriver`] is
    /// implemented from HMAC-SHA256.
    Aes128CtrHmacSha256,

    /// Both primitives are implemented using the BLAKE3 keyed hash function. The [`KeyStream']
    /// uses the XOF mode and the [`KeyDeriver`] uses the standard, fixed-sized output mode.
    Blake3,
}

/// A Key used to instantiate a [`KeyStream`] or [`KeyDeriver`].
#[derive(Clone, Debug, Eq)]
pub enum Key {
    #[allow(missing_docs)]
    Aes128CtrHmacSha256([u8; 32]),

    #[allow(missing_docs)]
    Blake3([u8; 32]),
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Aes128CtrHmacSha256(left), Self::Aes128CtrHmacSha256(right))
            | (Self::Blake3(left), Self::Blake3(right)) => {
                // Do constant-time compare.
                let mut r = 0;
                for (x, y) in (&left[..]).iter().zip(&right[..]) {
                    r |= x ^ y;
                }
                r == 0
            }
            _ => false,
        }
    }
}

impl Key {
    /// Generates a uniform random key of the type determined by `suite`.
    pub fn generate(suite: Suite) -> Result<Self, SuiteError> {
        match suite {
            Suite::Aes128CtrHmacSha256 => {
                let mut key = [0; 32];
                getrandom(&mut key)?;
                Ok(Key::Aes128CtrHmacSha256(key))
            }
            Suite::Blake3 => {
                let mut key = [0; 32];
                getrandom(&mut key)?;
                Ok(Key::Blake3(key))
            }
        }
    }

    /// Return the length in bytes of the key.
    pub fn size(&self) -> usize {
        match self {
            Self::Aes128CtrHmacSha256(_) | Self::Blake3(_) => 32,
        }
    }

    /// Returns an uninitialized (i.e., zero-valued) key. The caller is expected to initialize the
    /// key with a (pseudo)random input.
    pub(crate) fn uninitialized(suite: Suite) -> Self {
        match suite {
            Suite::Aes128CtrHmacSha256 => Key::Aes128CtrHmacSha256([0; 32]),
            Suite::Blake3 => Key::Blake3([0; 32]),
        }
    }

    /// Returns a reference to the underlying data.
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Aes128CtrHmacSha256(key) => &key[..],
            Self::Blake3(key) => &key[..],
        }
    }

    /// Returns a mutable reference to the underlying data.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Aes128CtrHmacSha256(key) => &mut key[..],
            Self::Blake3(key) => &mut key[..],
        }
    }

    /// Returns the suite for this key.
    pub(crate) fn suite(&self) -> Suite {
        match self {
            Key::Aes128CtrHmacSha256(_) => Suite::Aes128CtrHmacSha256,
            Key::Blake3(_) => Suite::Blake3,
        }
    }
}

impl Encode for Key {
    fn encode(&self, bytes: &mut Vec<u8>) {
        let seed = match self {
            Self::Aes128CtrHmacSha256(entropy) => entropy,
            Self::Blake3(entropy) => entropy,
        };

        bytes.extend_from_slice(seed);
    }
}

impl Decode<Suite> for Key {
    fn decode(decoding_parameter: &Suite, bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut seed = [0u8; 32];
        bytes.read_exact(&mut seed)?;

        match decoding_parameter {
            Suite::Aes128CtrHmacSha256 => Ok(Self::Aes128CtrHmacSha256(seed)),
            Suite::Blake3 => Ok(Self::Blake3(seed)),
        }
    }
}

/// A KeyStream expands a key into a stream of pseudorandom bytes.
#[derive(Debug)]
pub enum KeyStream {
    #[allow(missing_docs)]
    Aes128CtrHmacSha256(Aes128Ctr),

    #[allow(missing_docs)]
    Blake3(blake3::OutputReader),
}

impl KeyStream {
    /// Constructs a key expander from a key.
    pub fn from_key(key: &Key) -> Self {
        match key {
            Key::Aes128CtrHmacSha256(key) => {
                // The first 16 bytes of the key and the last 16 bytes of the key are used, respectively,
                // for the key and initialization vector for AES128 in CTR mode.
                let aes_ctr_key = GenericArray::from_slice(&key[..16]);
                let aes_ctr_iv = GenericArray::from_slice(&key[16..]);
                Self::Aes128CtrHmacSha256(Aes128Ctr::from_block_cipher(
                    Aes128::new(aes_ctr_key),
                    aes_ctr_iv,
                ))
            }
            Key::Blake3(key) => {
                const_assert_eq!(BLAKE3_STREAM_PREFIX.len(), BLAKE3_DERIVE_PREFIX.len());
                let mut hasher = blake3::Hasher::new_keyed(key);
                hasher.update(BLAKE3_STREAM_PREFIX);
                Self::Blake3(hasher.finalize_xof())
            }
        }
    }

    /// Fills the buffer `out` with the next `out.len()` bytes of the key stream.
    pub fn fill(&mut self, out: &mut [u8]) {
        match self {
            Self::Aes128CtrHmacSha256(aes128_ctr) => aes128_ctr.apply_keystream(out),
            Self::Blake3(output_reader) => output_reader.fill(out),
        }
    }
}

/// A KeyDeriver is a pseudorandom function whose output is a [`Key`] object.
#[derive(Debug)]
pub enum KeyDeriver {
    #[allow(missing_docs)]
    Blake3(blake3::Hasher),

    #[allow(missing_docs)]
    Aes128CtrHmacSha256(hmac::Context),
}

impl KeyDeriver {
    /// Initializes the function with the given key.
    pub fn from_key(key: &Key) -> Self {
        match key {
            Key::Aes128CtrHmacSha256(key) => {
                let key = hmac::Key::new(hmac::HMAC_SHA256, &key[..]);
                let context = hmac::Context::with_key(&key);
                Self::Aes128CtrHmacSha256(context)
            }
            Key::Blake3(key) => {
                let mut hasher = blake3::Hasher::new_keyed(key);
                hasher.update(BLAKE3_DERIVE_PREFIX);
                Self::Blake3(hasher)
            }
        }
    }

    /// Appends `input` to the function's input.
    pub fn update(&mut self, input: &[u8]) {
        match self {
            Self::Aes128CtrHmacSha256(context) => {
                context.update(input);
            }
            Self::Blake3(hasher) => {
                hasher.update(input);
            }
        }
    }

    /// Returns the output of the function.
    pub fn finish(&self) -> Key {
        match self {
            Self::Aes128CtrHmacSha256(context) => {
                let context = context.clone();
                let tag = context.sign();
                let tag = tag.as_ref();
                if tag.len() != 32 {
                    // This should never happen.
                    panic!("tag length is {}; expected 32", tag.len());
                }

                let mut key = [0; 32];
                key.copy_from_slice(tag);
                Key::Aes128CtrHmacSha256(key)
            }
            Self::Blake3(hasher) => Key::Blake3(*hasher.finalize().as_bytes()),
        }
    }
}
