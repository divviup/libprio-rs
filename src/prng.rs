// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

use super::field::{FieldElement, FieldError};
use aes::{
    cipher::{
        generic_array::GenericArray, FromBlockCipher, NewBlockCipher,
        StreamCipher as AesStreamCipher,
    },
    Aes128, Aes128Ctr,
};
use getrandom::getrandom;

use std::marker::PhantomData;

const DEFAULT_BUFFER_SIZE_IN_ELEMENTS: usize = 128;
const MAXIMUM_BUFFER_SIZE_IN_ELEMENTS: usize = 4096;

pub(crate) fn secret_share<F: FieldElement>(share1: &mut [F]) -> Result<Vec<u8>, getrandom::Error> {
    let mut seed = [0; STREAM_CIPHER_AES128CTR_KEY_LENGTH];
    getrandom(&mut seed)?;

    // get prng array
    let data: Vec<F> =
        Prng::<_, Aes128Ctr, STREAM_CIPHER_AES128CTR_KEY_LENGTH>::new_with_seed_and_optional_length(
            seed,
            Some(share1.len()),
        )
        .collect();

    // secret share
    for (s1, d) in share1.iter_mut().zip(data.iter()) {
        *s1 -= *d;
    }

    Ok(seed.to_vec())
}

pub(crate) fn extract_share_from_seed<F: FieldElement>(
    length: usize,
    seed: &[u8],
) -> Result<Vec<F>, PrngError> {
    if seed.len() != STREAM_CIPHER_AES128CTR_KEY_LENGTH {
        return Err(PrngError::SeedLen);
    }

    let mut seed_copy = [0; STREAM_CIPHER_AES128CTR_KEY_LENGTH];
    seed_copy.clone_from_slice(seed);
    Ok(
        Prng::<_, Aes128Ctr, STREAM_CIPHER_AES128CTR_KEY_LENGTH>::new_with_seed_and_optional_length(
            seed_copy,
            Some(length),
        )
        .collect(),
    )
}

/// A stream cipher maps a short key to a stream of pseudorandom bytes.
pub(crate) trait StreamCipher<const KEY_LENGTH: usize> {
    /// Returns a new stream ciphered keyed by `key`.
    fn new(key: [u8; KEY_LENGTH]) -> Self;

    /// Fills the input buffer `buf` with the next `buf.len()` bytes of the stream.
    fn fill(&mut self, buf: &mut [u8]);
}

pub(crate) const STREAM_CIPHER_AES128CTR_KEY_LENGTH: usize = 2 * aes::BLOCK_SIZE;
pub(crate) const STREAM_CIPHER_BLAKE3_KEY_LENGTH: usize = 32;

impl StreamCipher<STREAM_CIPHER_AES128CTR_KEY_LENGTH> for Aes128Ctr {
    fn new(key: [u8; STREAM_CIPHER_AES128CTR_KEY_LENGTH]) -> Self {
        // The first 16 bytes of the key and the last 16 bytes of the key are used, respectively,
        // for the key and initialization vector for AES128 in CTR mode.
        let aes_ctr_key = GenericArray::from_slice(&key[..aes::BLOCK_SIZE]);
        let aes_ctr_iv = GenericArray::from_slice(&key[aes::BLOCK_SIZE..]);
        Self::from_block_cipher(Aes128::new(aes_ctr_key), aes_ctr_iv)
    }

    fn fill(&mut self, buf: &mut [u8]) {
        self.apply_keystream(buf);
    }
}

impl StreamCipher<STREAM_CIPHER_BLAKE3_KEY_LENGTH> for blake3::OutputReader {
    fn new(key: [u8; STREAM_CIPHER_BLAKE3_KEY_LENGTH]) -> Self {
        let mut hasher = blake3::Hasher::new_keyed(&key);
        hasher.update(b"prio-v3-prng-stream-cipher");
        hasher.finalize_xof()
    }

    fn fill(&mut self, buf: &mut [u8]) {
        Self::fill(self, buf)
    }
}

/// Errors propagated by methods in this module.
#[derive(Debug, PartialEq, thiserror::Error)]
pub(crate) enum PrngError {
    #[error("invalid seed length")]
    SeedLen,
}

/// This type implements an iterator that generates a pseudorandom sequence of field elements. The
/// sequence is derived from the key stream of AES-128 in CTR mode with a random IV.
#[derive(Debug)]
pub(crate) struct Prng<F, X, const KEY_LENGTH: usize>
where
    F: FieldElement,
    X: StreamCipher<KEY_LENGTH>,
{
    phantom: PhantomData<F>,
    key_stream: X,
    length: Option<usize>,
    buffer: Vec<u8>,
    buffer_index: usize,
    output_written: usize,
}

impl<F, X, const KEY_LENGTH: usize> Prng<F, X, KEY_LENGTH>
where
    F: FieldElement,
    X: StreamCipher<KEY_LENGTH>,
{
    /// Generates a seed and constructs an iterator over an infinite sequence of pseudorandom field
    /// elements.
    pub(crate) fn new() -> Result<Self, getrandom::Error> {
        let mut seed = [0; KEY_LENGTH];
        getrandom(&mut seed)?;
        Ok(Self::new_with_seed_and_optional_length(seed, None))
    }

    /// Generates a seed and constructs an iterator over a pseudorandom sequence of field elements
    /// of length `length`.
    pub(crate) fn new_with_length(length: usize) -> Result<Self, getrandom::Error> {
        let mut seed = [0; KEY_LENGTH];
        getrandom(&mut seed)?;
        Ok(Self::new_with_seed_and_optional_length(seed, Some(length)))
    }

    fn new_with_seed_and_optional_length(seed: [u8; KEY_LENGTH], length: Option<usize>) -> Self {
        let mut key_stream = X::new(seed);

        let buf_len_in_elems = match length {
            Some(length) => std::cmp::min(length + 1, MAXIMUM_BUFFER_SIZE_IN_ELEMENTS),
            None => DEFAULT_BUFFER_SIZE_IN_ELEMENTS,
        };
        let mut buffer = vec![0; buf_len_in_elems * F::ENCODED_SIZE];
        key_stream.fill(&mut buffer);

        Self {
            phantom: PhantomData::<F>,
            key_stream,
            length,
            buffer,
            buffer_index: 0,
            output_written: 0,
        }
    }
}

impl<F, X, const KEY_LENGTH: usize> Iterator for Prng<F, X, KEY_LENGTH>
where
    F: FieldElement,
    X: StreamCipher<KEY_LENGTH>,
{
    type Item = F;

    fn next(&mut self) -> Option<F> {
        if let Some(length) = self.length {
            if self.output_written >= length {
                return None;
            }
        }

        loop {
            // Seek to the next chunk of the buffer that encodes an element of F.
            for i in (self.buffer_index..self.buffer.len()).step_by(F::ENCODED_SIZE) {
                let j = i + F::ENCODED_SIZE;
                if let Some(x) = match F::try_from_random(&self.buffer[i..j]) {
                    Ok(x) => Some(x),
                    Err(FieldError::ModulusOverflow) => None, // reject this sample
                    Err(err) => panic!("unexpected error: {}", err),
                } {
                    // Set the buffer index to the next chunk.
                    self.buffer_index = j;
                    self.output_written += 1;
                    return Some(x);
                }
            }

            // Refresh buffer with the next chunk of PRG output.
            for b in &mut self.buffer {
                *b = 0;
            }
            self.key_stream.fill(&mut self.buffer);
            self.buffer_index = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field32, FieldPriov2};

    #[test]
    fn secret_sharing() {
        let mut data = vec![Field32::from(0); 123];
        data[3] = 23.into();

        let data_clone = data.clone();

        let seed = secret_share(&mut data).unwrap();
        assert_ne!(data, data_clone);

        let share2 = extract_share_from_seed(data.len(), &seed).unwrap();

        assert_eq!(data.len(), share2.len());

        // recombine
        for (d, d2) in data.iter_mut().zip(share2.iter()) {
            *d += *d2;
        }

        assert_eq!(data, data_clone);
    }

    #[test]
    fn secret_sharing_interop() {
        let seed = [
            0xcd, 0x85, 0x5b, 0xd4, 0x86, 0x48, 0xa4, 0xce, 0x52, 0x5c, 0x36, 0xee, 0x5a, 0x71,
            0xf3, 0x0f, 0x66, 0x80, 0xd3, 0x67, 0x53, 0x9a, 0x39, 0x6f, 0x12, 0x2f, 0xad, 0x94,
            0x4d, 0x34, 0xcb, 0x58,
        ];

        let reference = [
            0xd0056ec5, 0xe23f9c52, 0x47e4ddb4, 0xbe5dacf6, 0x4b130aba, 0x530c7a90, 0xe8fc4ee5,
            0xb0569cb7, 0x7774cd3c, 0x7f24e6a5, 0xcc82355d, 0xc41f4f13, 0x67fe193c, 0xc94d63a4,
            0x5d7b474c, 0xcc5c9f5f, 0xe368e1d5, 0x020fa0cf, 0x9e96aa2a, 0xe924137d, 0xfa026ab9,
            0x8ebca0cc, 0x26fc58a5, 0x10a7b173, 0xb9c97291, 0x53ef0e28, 0x069cfb8e, 0xe9383cae,
            0xacb8b748, 0x6f5b9d49, 0x887d061b, 0x86db0c58,
        ];

        let share2 = extract_share_from_seed::<FieldPriov2>(reference.len(), &seed).unwrap();

        assert_eq!(share2, reference);
    }

    /// takes a seed and hash as base64 encoded strings
    fn random_data_interop(seed_base64: &str, hash_base64: &str, len: usize) {
        let seed = base64::decode(seed_base64).unwrap();
        let random_data = extract_share_from_seed::<FieldPriov2>(len, &seed).unwrap();

        let random_bytes = FieldPriov2::slice_into_byte_vec(&random_data);

        let digest = ring::digest::digest(&ring::digest::SHA256, &random_bytes);
        assert_eq!(base64::encode(digest), hash_base64);
    }

    #[test]
    fn test_hash_interop() {
        random_data_interop(
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
            "RtzeQuuiWdD6bW2ZTobRELDmClz1wLy3HUiKsYsITOI=",
            100_000,
        );

        // zero seed
        random_data_interop(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "3wHQbSwAn9GPfoNkKe1qSzWdKnu/R+hPPyRwwz6Di+w=",
            100_000,
        );
        // 0, 1, 2 ... seed
        random_data_interop(
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
            "RtzeQuuiWdD6bW2ZTobRELDmClz1wLy3HUiKsYsITOI=",
            100_000,
        );
        // one arbirtary fixed seed
        random_data_interop(
            "rkLrnVcU8ULaiuXTvR3OKrfpMX0kQidqVzta1pleKKg=",
            "b1fMXYrGUNR3wOZ/7vmUMmY51QHoPDBzwok0fz6xC0I=",
            100_000,
        );
        // all bits set seed
        random_data_interop(
            "//////////////////////////////////////////8=",
            "iBiDaqLrv7/rX/+vs6akPiprGgYfULdh/XhoD61HQXA=",
            100_000,
        );
    }
}
