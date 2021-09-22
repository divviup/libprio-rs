// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Tool for generating pseudorandom field elements.
//!
//! NOTE: The public API for this module is a work in progress.

use super::field::{FieldElement, FieldError};
use crate::vdaf::suite::{Key, KeyStream, Suite, SuiteError};

use std::marker::PhantomData;

const BUFFER_SIZE_IN_ELEMENTS: usize = 128;

pub(crate) fn secret_share<F: FieldElement>(share1: &mut [F]) -> Result<Vec<u8>, PrngError> {
    let key = Key::generate(Suite::Aes128CtrHmacSha256)?;

    // get prng array
    let data: Vec<F> = Prng::from_key_stream(KeyStream::from_key(&key))
        .take(share1.len())
        .collect();

    // secret share
    for (s1, d) in share1.iter_mut().zip(data.iter()) {
        *s1 -= *d;
    }

    Ok(key.as_slice().to_vec())
}

pub(crate) fn extract_share_from_seed<F: FieldElement>(
    length: usize,
    seed: &[u8],
) -> Result<Vec<F>, PrngError> {
    if seed.len() != aes::BLOCK_SIZE * 2 {
        return Err(PrngError::SeedSize);
    }

    let mut key = [0; aes::BLOCK_SIZE * 2];
    key.copy_from_slice(seed);
    let key_stream = KeyStream::from_key(&Key::Aes128CtrHmacSha256(key));
    Ok(Prng::from_key_stream(key_stream).take(length).collect())
}

/// Errors propagated by methods in this module.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum PrngError {
    /// Tried to construct a PRNG from a seed of invalid length.
    #[error("invalid seed length")]
    SeedSize,

    /// VDAF suite error.
    #[error("vdaf suite error: {0}")]
    VdafSuite(#[from] SuiteError),
}

/// This type implements an iterator that generates a pseudorandom sequence of field elements. The
/// sequence is derived from the key stream of AES-128 in CTR mode with a random IV.
#[derive(Debug)]
pub(crate) struct Prng<F: FieldElement> {
    phantom: PhantomData<F>,
    key_stream: KeyStream,
    buffer: Vec<u8>,
    buffer_index: usize,
    output_written: usize,
}

impl<F: FieldElement> Prng<F> {
    /// Generates a seed and constructs an iterator over an infinite sequence of pseudorandom field
    /// elements.
    pub(crate) fn generate(suite: Suite) -> Result<Self, PrngError> {
        let key = Key::generate(suite)?;
        let key_stream = KeyStream::from_key(&key);
        Ok(Self::from_key_stream(key_stream))
    }

    pub(crate) fn from_key_stream(mut key_stream: KeyStream) -> Self {
        let mut buffer = vec![0; BUFFER_SIZE_IN_ELEMENTS * F::ENCODED_SIZE];
        key_stream.fill(&mut buffer);

        Self {
            phantom: PhantomData::<F>,
            key_stream,
            buffer,
            buffer_index: 0,
            output_written: 0,
        }
    }
}

impl<F: FieldElement> Iterator for Prng<F> {
    type Item = F;

    fn next(&mut self) -> Option<F> {
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
