// SPDX-License-Identifier: MPL-2.0

//! This module implements PRGs as specified in draft-patton-cfrg-vdaf-01.

use crate::vdaf::{CodecError, Decode, Encode};
use aes::{
    cipher::{
        generic_array::GenericArray, FromBlockCipher, NewBlockCipher,
        StreamCipher as AesStreamCipher,
    },
    Aes128, Aes128Ctr,
};
use cmac::{Cmac, Mac, NewMac};
use std::{
    fmt::Debug,
    io::{Cursor, Read},
};

/// Input of [`Prg`].
#[derive(Clone, Debug, Eq)]
pub struct Seed<const L: usize>(pub(crate) [u8; L]);

impl<const L: usize> Seed<L> {
    /// Generate a uniform random seed.
    pub fn generate() -> Result<Self, getrandom::Error> {
        let mut seed = [0; L];
        getrandom::getrandom(&mut seed)?;
        Ok(Self(seed))
    }

    pub(crate) fn uninitialized() -> Self {
        Self([0; L])
    }

    pub(crate) fn xor_accumulate(&mut self, other: &Self) {
        for i in 0..L {
            self.0[i] ^= other.0[i]
        }
    }

    pub(crate) fn xor(&mut self, left: &Self, right: &Self) {
        for i in 0..L {
            self.0[i] = left.0[i] ^ right.0[i]
        }
    }
}

impl<const L: usize> PartialEq for Seed<L> {
    fn eq(&self, other: &Self) -> bool {
        // Do constant-time compare.
        let mut r = 0;
        for (x, y) in (&self.0[..]).iter().zip(&other.0[..]) {
            r |= x ^ y;
        }
        r == 0
    }
}

impl<const L: usize> Encode for Seed<L> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.0[..]);
    }
}

impl<const L: usize> Decode<()> for Seed<L> {
    fn decode(_decoding_parameter: &(), bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut seed = [0; L];
        bytes.read_exact(&mut seed)?;
        Ok(Seed(seed))
    }
}

/// A stream of pseudorandom bytes derived from a seed.
pub trait SeedStream {
    /// Fill `buf` with the next `buf.len()` bytes of output.
    fn fill(&mut self, buf: &mut [u8]);
}

/// A pseudorandom generator (PRG) with the interface specified in
/// [VDAF](https://datatracker.ietf.org/doc/draft-patton-cfrg-vdaf/).
pub trait Prg<const L: usize>: Clone + Debug {
    /// The type of stream produced by this PRG.
    type SeedStream: SeedStream;

    /// Construct an instance of [`Prg`] with the given seed.
    fn init(seed: &Seed<L>) -> Self;

    /// Update the PRG state by passing in the next fragment of the info string. The final info
    /// string is assembled from the concatenation of sequence of fragments passed to this method.
    fn update(&mut self, data: &[u8]);

    /// Finalize the PRG state, producing a seed stream.
    fn into_seed_stream(self) -> Self::SeedStream;

    /// Finalize the PRG state, producing a seed.
    fn into_seed(self) -> Seed<L> {
        let mut new_seed = [0; L];
        let mut seed_stream = self.into_seed_stream();
        seed_stream.fill(&mut new_seed);
        Seed(new_seed)
    }

    /// Construct a seed stream from the given seed and info string.
    fn seed_stream(seed: &Seed<L>, info: &[u8]) -> Self::SeedStream {
        let mut prg = Self::init(seed);
        prg.update(info);
        prg.into_seed_stream()
    }
}

/// The PRG based on AES128 as specifed in
/// [VDAF](https://datatracker.ietf.org/doc/draft-patton-cfrg-vdaf/).
#[derive(Clone, Debug)]
pub struct PrgAes128(Cmac<Aes128>);

impl Prg<16> for PrgAes128 {
    type SeedStream = SeedStreamAes128;

    fn init(seed: &Seed<16>) -> Self {
        Self(Cmac::new_from_slice(&seed.0).unwrap())
    }

    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    fn into_seed_stream(self) -> SeedStreamAes128 {
        let key = self.0.finalize().into_bytes();
        SeedStreamAes128::new(&key, &[0; 16])
    }
}

/// The key stream produced by AES128 in CTR-mode.
#[derive(Debug)]
pub struct SeedStreamAes128(Aes128Ctr);

impl SeedStreamAes128 {
    pub(crate) fn new(key: &[u8], iv: &[u8]) -> Self {
        SeedStreamAes128(Aes128Ctr::from_block_cipher(
            Aes128::new(GenericArray::from_slice(key)),
            GenericArray::from_slice(iv),
        ))
    }
}

impl SeedStream for SeedStreamAes128 {
    fn fill(&mut self, buf: &mut [u8]) {
        buf.fill(0);
        self.0.apply_keystream(buf);
    }
}
