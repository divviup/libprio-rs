// SPDX-License-Identifier: MPL-2.0

//! Implementations of XOFs specified in [[draft-irtf-cfrg-vdaf-07]].
//!
//! [draft-irtf-cfrg-vdaf-07]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/07/

use crate::{
    field::FieldElement,
    prng::Prng,
    vdaf::{CodecError, Decode, Encode},
};
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
use aes::{
    cipher::{generic_array::GenericArray, BlockEncrypt, KeyInit},
    Block,
};
#[cfg(feature = "crypto-dependencies")]
use aes::{
    cipher::{KeyIvInit, StreamCipher},
    Aes128,
};
#[cfg(feature = "crypto-dependencies")]
use ctr::Ctr64BE;
use rand_core::{
    impls::{next_u32_via_fill, next_u64_via_fill},
    RngCore, SeedableRng,
};
#[cfg(feature = "crypto-dependencies")]
use sha2::{Digest, Sha256};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128, Shake128Core, Shake128Reader, TurboShake128, TurboShake128Core, TurboShake128Reader,
};
#[cfg(feature = "crypto-dependencies")]
use std::fmt::Formatter;
use std::{
    fmt::Debug,
    io::{Cursor, Read},
};
use subtle::{Choice, ConstantTimeEq};

/// Input of [`Xof`].
#[derive(Clone, Debug)]
pub struct Seed<const SEED_SIZE: usize>(pub(crate) [u8; SEED_SIZE]);

impl<const SEED_SIZE: usize> Seed<SEED_SIZE> {
    /// Generate a uniform random seed.
    pub fn generate() -> Result<Self, getrandom::Error> {
        let mut seed = [0; SEED_SIZE];
        getrandom::getrandom(&mut seed)?;
        Ok(Self::from_bytes(seed))
    }

    /// Construct seed from a byte slice.
    pub(crate) fn from_bytes(seed: [u8; SEED_SIZE]) -> Self {
        Self(seed)
    }
}

impl<const SEED_SIZE: usize> AsRef<[u8; SEED_SIZE]> for Seed<SEED_SIZE> {
    fn as_ref(&self) -> &[u8; SEED_SIZE] {
        &self.0
    }
}

impl<const SEED_SIZE: usize> PartialEq for Seed<SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<const SEED_SIZE: usize> Eq for Seed<SEED_SIZE> {}

impl<const SEED_SIZE: usize> ConstantTimeEq for Seed<SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl<const SEED_SIZE: usize> Encode for Seed<SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.0[..]);
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(SEED_SIZE)
    }
}

impl<const SEED_SIZE: usize> Decode for Seed<SEED_SIZE> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut seed = [0; SEED_SIZE];
        bytes.read_exact(&mut seed)?;
        Ok(Seed(seed))
    }
}

/// Trait for deriving a vector of field elements.
pub trait IntoFieldVec: RngCore + Sized {
    /// Generate a finite field vector from the seed stream.
    fn into_field_vec<F: FieldElement>(self, length: usize) -> Vec<F>;
}

impl<S: RngCore> IntoFieldVec for S {
    fn into_field_vec<F: FieldElement>(self, length: usize) -> Vec<F> {
        Prng::from_seed_stream(self).take(length).collect()
    }
}

/// An extendable output function (XOF) with the interface specified in [[draft-irtf-cfrg-vdaf-07]].
///
/// [draft-irtf-cfrg-vdaf-07]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/07/
pub trait Xof<const SEED_SIZE: usize>: Clone + Debug {
    /// The type of stream produced by this XOF.
    type SeedStream: RngCore + Sized;

    /// Construct an instance of [`Xof`] with the given seed.
    fn init(seed_bytes: &[u8; SEED_SIZE], dst: &[u8]) -> Self;

    /// Update the XOF state by passing in the next fragment of the info string. The final info
    /// string is assembled from the concatenation of sequence of fragments passed to this method.
    fn update(&mut self, data: &[u8]);

    /// Finalize the XOF state, producing a seed stream.
    fn into_seed_stream(self) -> Self::SeedStream;

    /// Finalize the XOF state, producing a seed.
    fn into_seed(self) -> Seed<SEED_SIZE> {
        let mut new_seed = [0; SEED_SIZE];
        let mut seed_stream = self.into_seed_stream();
        seed_stream.fill_bytes(&mut new_seed);
        Seed(new_seed)
    }

    /// Construct a seed stream from the given seed and info string.
    fn seed_stream(seed: &Seed<SEED_SIZE>, dst: &[u8], binder: &[u8]) -> Self::SeedStream {
        let mut xof = Self::init(seed.as_ref(), dst);
        xof.update(binder);
        xof.into_seed_stream()
    }
}

/// The key stream produced by AES128 in CTR-mode.
#[cfg(feature = "crypto-dependencies")]
#[cfg_attr(docsrs, doc(cfg(feature = "crypto-dependencies")))]
pub struct SeedStreamAes128(Ctr64BE<Aes128>);

#[cfg(feature = "crypto-dependencies")]
impl SeedStreamAes128 {
    pub(crate) fn new(key: &[u8], iv: &[u8]) -> Self {
        SeedStreamAes128(<Ctr64BE<Aes128> as KeyIvInit>::new(key.into(), iv.into()))
    }

    fn fill(&mut self, buf: &mut [u8]) {
        buf.fill(0);
        self.0.apply_keystream(buf);
    }
}

#[cfg(feature = "crypto-dependencies")]
impl RngCore for SeedStreamAes128 {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.fill(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill(dest);
        Ok(())
    }

    fn next_u32(&mut self) -> u32 {
        next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        next_u64_via_fill(self)
    }
}

#[cfg(feature = "crypto-dependencies")]
impl Debug for SeedStreamAes128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Ctr64BE<Aes128> does not implement Debug, but [`ctr::CtrCore`][1] does, and we get that
        // with [`cipher::StreamCipherCoreWrapper::get_core`][2].
        //
        // [1]: https://docs.rs/ctr/latest/ctr/struct.CtrCore.html
        // [2]: https://docs.rs/cipher/latest/cipher/struct.StreamCipherCoreWrapper.html
        self.0.get_core().fmt(f)
    }
}

/// The XOF based on SHA-3 as specified in [[draft-irtf-cfrg-vdaf-07]].
///
/// [draft-irtf-cfrg-vdaf-07]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/07/
#[derive(Clone, Debug)]
pub struct XofShake128(Shake128);

impl Xof<16> for XofShake128 {
    type SeedStream = SeedStreamSha3;

    fn init(seed_bytes: &[u8; 16], dst: &[u8]) -> Self {
        let mut xof = Self(Shake128::from_core(Shake128Core::default()));
        Update::update(
            &mut xof.0,
            &[dst.len().try_into().expect("dst must be at most 255 bytes")],
        );
        Update::update(&mut xof.0, dst);
        Update::update(&mut xof.0, seed_bytes);
        xof
    }

    fn update(&mut self, data: &[u8]) {
        Update::update(&mut self.0, data);
    }

    fn into_seed_stream(self) -> SeedStreamSha3 {
        SeedStreamSha3::new(self.0.finalize_xof())
    }
}

/// The key stream produced by the cSHAKE128 XOF.
pub struct SeedStreamSha3(Shake128Reader);

impl SeedStreamSha3 {
    pub(crate) fn new(reader: Shake128Reader) -> Self {
        Self(reader)
    }
}

impl RngCore for SeedStreamSha3 {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        XofReader::read(&mut self.0, dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        XofReader::read(&mut self.0, dest);
        Ok(())
    }

    fn next_u32(&mut self) -> u32 {
        next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        next_u64_via_fill(self)
    }
}

/// A `rand`-compatible interface to construct XofShake128 seed streams, with the domain separation tag
/// and binder string both fixed as the empty string.
impl SeedableRng for SeedStreamSha3 {
    type Seed = [u8; 16];

    fn from_seed(seed: Self::Seed) -> Self {
        XofShake128::init(&seed, b"").into_seed_stream()
    }
}

/// XOF based on TurboSHAKE128 from [[draft-irtf-cfrg-kangarootwelve-11]].
///
/// TODO(cjpatton) Make sure the implementation is in compliance with the spec.
///
/// [draft-irtf-cfrg-kangarootwelve-11](https://datatracker.ietf.org/doc/draft-irtf-cfrg-kangarootwelve/11/)
#[derive(Clone, Debug)]
pub struct XofTurboShake128(TurboShake128);

impl Xof<16> for XofTurboShake128 {
    type SeedStream = SeedStreamTurboShake128;

    fn init(seed_bytes: &[u8; 16], dst: &[u8]) -> Self {
        let mut xof = Self(TurboShake128::from_core(TurboShake128Core::new(1)));
        Update::update(
            &mut xof.0,
            &[dst.len().try_into().expect("dst must be at most 255 bytes")],
        );
        Update::update(&mut xof.0, dst);
        Update::update(&mut xof.0, seed_bytes);
        xof
    }

    fn update(&mut self, data: &[u8]) {
        Update::update(&mut self.0, data);
    }

    fn into_seed_stream(self) -> SeedStreamTurboShake128 {
        SeedStreamTurboShake128::new(self.0.finalize_xof())
    }
}

/// Seed stream for [`XofTurboShake128`].
pub struct SeedStreamTurboShake128(TurboShake128Reader);

impl SeedStreamTurboShake128 {
    pub(crate) fn new(reader: TurboShake128Reader) -> Self {
        Self(reader)
    }
}

impl RngCore for SeedStreamTurboShake128 {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        XofReader::read(&mut self.0, dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }

    fn next_u32(&mut self) -> u32 {
        next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        next_u64_via_fill(self)
    }
}

/// XOF based on SHA-256.
///
/// This is a tweaked version of expand_message_xmd from RFC 9380, copied here for reference:
///
/// ```text
/// expand_message_xmd(msg, DST, len_in_bytes)
///
///    Parameters:
///    - H, a hash function (see requirements above).
///    - b_in_bytes, b / 8 for b the output size of H in bits.
///      For example, for b = 256, b_in_bytes = 32.
///    - s_in_bytes, the input block size of H, measured in bytes (see
///      discussion above). For example, for SHA-256, s_in_bytes = 64.
///
///    Input:
///    - msg, a byte string.
///    - DST, a byte string of at most 255 bytes.
///      See below for information on using longer DSTs.
///    - len_in_bytes, the length of the requested output in bytes,
///      not greater than the lesser of (255 * b_in_bytes) or 2^16-1.
///
///    Output:
///    - uniform_bytes, a byte string.
///
///    Steps:
///    1.  ell = ceil(len_in_bytes / b_in_bytes)
///    2.  ABORT if ell > 255 or len_in_bytes > 65535 or len(DST) > 255
///    3.  DST_prime = DST || I2OSP(len(DST), 1)
///    4.  Z_pad = I2OSP(0, s_in_bytes)
///    5.  l_i_b_str = I2OSP(len_in_bytes, 2)
///    6.  msg_prime = Z_pad || msg || l_i_b_str || I2OSP(0, 1) || DST_prime
///    7.  b_0 = H(msg_prime)
///    8.  b_1 = H(b_0 || I2OSP(1, 1) || DST_prime)
///    9.  for i in (2, ..., ell):
///    10.    b_i = H(strxor(b_0, b_(i - 1)) || I2OSP(i, 1) || DST_prime)
///    11. uniform_bytes = b_1 || ... || b_ell
///    12. return substr(uniform_bytes, 0, len_in_bytes)
/// ```
///
/// Changes:
///
///  * We don't know the length of the output `len_in_bytes` in advance, so use `0` instead. As a
///    result, the requested output length can't be used for domain separation.
///
///  * Use 4 bytes to encode the block ID instead of one. This is necessary to get the longer
///    outputs that we need in this application.

#[derive(Clone, Debug)]
pub struct XofSha2 {
    b0_hasher: Sha256,
    dst_prime: Vec<u8>,
}

impl Xof<16> for XofSha2 {
    type SeedStream = SeedStreamSha2;

    fn init(seed_bytes: &[u8; 16], dst: &[u8]) -> Self {
        const S_IN_BYTES: usize = 64;
        let mut dst_prime = Vec::with_capacity(dst.len() + 1);
        dst_prime.extend_from_slice(dst);
        dst_prime.push(dst.len().try_into().expect("dst too long"));

        let mut b0_hasher = <Sha256 as Digest>::new();
        Digest::update(&mut b0_hasher, [0; S_IN_BYTES]); // Z_pad

        // msg = seed_bytes || binder
        Digest::update(&mut b0_hasher, seed_bytes); // msg part

        Self {
            b0_hasher,
            dst_prime,
        }
    }

    fn update(&mut self, data: &[u8]) {
        Digest::update(&mut self.b0_hasher, data); // msg part
    }

    fn into_seed_stream(self) -> SeedStreamSha2 {
        let Self {
            mut b0_hasher,
            dst_prime,
        } = self;
        Digest::update(&mut b0_hasher, [0; 4]); // tweaked l_i_b_str
        Digest::update(&mut b0_hasher, 0_u32.to_be_bytes()); // tweaked block ID
        Digest::update(&mut b0_hasher, &dst_prime); // finish msg_prime
        let b0 = b0_hasher.finalize().into();

        let mut b1_hasher = <Sha256 as Digest>::new();
        Digest::update(&mut b1_hasher, b0);
        Digest::update(&mut b1_hasher, 1_u32.to_be_bytes()); // tweaked block ID
        Digest::update(&mut b1_hasher, &dst_prime);
        let b1 = b1_hasher.finalize().into();

        SeedStreamSha2 {
            b0,
            bcurr: b1,
            index: 1,
            offset: 0,
            dst_prime,
        }
    }
}

/// Seed stream for [`XofSha2`].
pub struct SeedStreamSha2 {
    b0: [u8; 32],
    bcurr: [u8; 32],
    index: u32,
    offset: usize,
    dst_prime: Vec<u8>,
}

impl RngCore for SeedStreamSha2 {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        if dest.is_empty() {
            return;
        }

        let mut read = 0;
        while read < dest.len() {
            let read_from_bcurr = std::cmp::min(dest.len() - read, 32 - self.offset);
            dest[..read_from_bcurr]
                .copy_from_slice(&self.bcurr[self.offset..self.offset + read_from_bcurr]);
            self.offset += read_from_bcurr;
            read += read_from_bcurr;

            if self.offset == 32 {
                self.offset = 0;
                self.index += 1;
                let mut bnext_hasher = <Sha256 as Digest>::new();
                for (b0, bcurr) in self.b0.iter().zip(self.bcurr.iter_mut()) {
                    *bcurr ^= *b0;
                }
                Digest::update(&mut bnext_hasher, self.bcurr);
                Digest::update(&mut bnext_hasher, self.index.to_be_bytes()); // tweaked block ID
                Digest::update(&mut bnext_hasher, &self.dst_prime);
                self.bcurr = bnext_hasher.finalize().into();
            }
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }

    fn next_u32(&mut self) -> u32 {
        next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        next_u64_via_fill(self)
    }
}

/// Factory to produce multiple [`XofFixedKeyAes128`] instances with the same fixed key and
/// different seeds.
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "crypto-dependencies", feature = "experimental")))
)]
pub struct XofFixedKeyAes128Key {
    cipher: Aes128,
}

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
impl XofFixedKeyAes128Key {
    /// Derive the fixed key from the domain separation tag and binder string.
    pub fn new(dst: &[u8], binder: &[u8]) -> Self {
        let mut fixed_key_deriver = Shake128::from_core(Shake128Core::default());
        Update::update(
            &mut fixed_key_deriver,
            &[dst.len().try_into().expect("dst must be at most 255 bytes")],
        );
        Update::update(&mut fixed_key_deriver, dst);
        Update::update(&mut fixed_key_deriver, binder);
        let mut key = GenericArray::from([0; 16]);
        XofReader::read(&mut fixed_key_deriver.finalize_xof(), key.as_mut());
        Self {
            cipher: Aes128::new(&key),
        }
    }

    /// Combine a fixed key with a seed to produce a new stream of bytes.
    pub fn with_seed(&self, seed: &[u8; 16]) -> SeedStreamFixedKeyAes128 {
        SeedStreamFixedKeyAes128 {
            cipher: self.cipher.clone(),
            base_block: (*seed).into(),
            length_consumed: 0,
        }
    }
}

/// XofFixedKeyAes128 as specified in [[draft-irtf-cfrg-vdaf-07]]. This XOF is NOT RECOMMENDED for
/// general use; see Section 9 ("Security Considerations") for details.
///
/// This XOF combines SHA-3 and a fixed-key mode of operation for AES-128. The key is "fixed" in
/// the sense that it is derived (using cSHAKE128) from the domain separation tag and binder
/// strings, and depending on the application, these strings can be hard-coded. The seed is used to
/// construct each block of input passed to a hash function built from AES-128.
///
/// [draft-irtf-cfrg-vdaf-07]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/07/
#[derive(Clone, Debug)]
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "crypto-dependencies", feature = "experimental")))
)]
pub struct XofFixedKeyAes128 {
    fixed_key_deriver: Shake128,
    base_block: Block,
}

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
impl Xof<16> for XofFixedKeyAes128 {
    type SeedStream = SeedStreamFixedKeyAes128;

    fn init(seed_bytes: &[u8; 16], dst: &[u8]) -> Self {
        let mut fixed_key_deriver = Shake128::from_core(Shake128Core::default());
        Update::update(
            &mut fixed_key_deriver,
            &[dst.len().try_into().expect("dst must be at most 255 bytes")],
        );
        Update::update(&mut fixed_key_deriver, dst);
        Self {
            fixed_key_deriver,
            base_block: (*seed_bytes).into(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        Update::update(&mut self.fixed_key_deriver, data);
    }

    fn into_seed_stream(self) -> SeedStreamFixedKeyAes128 {
        let mut fixed_key = GenericArray::from([0; 16]);
        XofReader::read(
            &mut self.fixed_key_deriver.finalize_xof(),
            fixed_key.as_mut(),
        );
        SeedStreamFixedKeyAes128 {
            base_block: self.base_block,
            cipher: Aes128::new(&fixed_key),
            length_consumed: 0,
        }
    }
}

/// Seed stream for [`XofFixedKeyAes128`].
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "crypto-dependencies", feature = "experimental")))
)]
pub struct SeedStreamFixedKeyAes128 {
    cipher: Aes128,
    base_block: Block,
    length_consumed: u64,
}

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
impl SeedStreamFixedKeyAes128 {
    fn hash_block(&self, block: &mut Block) {
        let sigma = Block::from([
            // hi
            block[8],
            block[9],
            block[10],
            block[11],
            block[12],
            block[13],
            block[14],
            block[15],
            // xor(hi, lo)
            block[8] ^ block[0],
            block[9] ^ block[1],
            block[10] ^ block[2],
            block[11] ^ block[3],
            block[12] ^ block[4],
            block[13] ^ block[5],
            block[14] ^ block[6],
            block[15] ^ block[7],
        ]);
        self.cipher.encrypt_block_b2b(&sigma, block);
        for (b, s) in block.iter_mut().zip(sigma.iter()) {
            *b ^= s;
        }
    }

    fn fill(&mut self, buf: &mut [u8]) {
        let next_length_consumed = self.length_consumed + u64::try_from(buf.len()).unwrap();
        let mut offset = usize::try_from(self.length_consumed % 16).unwrap();
        let mut index = 0;
        let mut block = Block::from([0; 16]);

        // NOTE(cjpatton) We might be able to speed this up by unrolling this loop and encrypting
        // multiple blocks at the same time via `self.cipher.encrypt_blocks()`.
        for block_counter in self.length_consumed / 16..(next_length_consumed + 15) / 16 {
            block.clone_from(&self.base_block);
            for (b, i) in block.iter_mut().zip(block_counter.to_le_bytes().iter()) {
                *b ^= i;
            }
            self.hash_block(&mut block);
            let read = std::cmp::min(16 - offset, buf.len() - index);
            buf[index..index + read].copy_from_slice(&block[offset..offset + read]);
            offset = 0;
            index += read;
        }

        self.length_consumed = next_length_consumed;
    }
}

#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
impl RngCore for SeedStreamFixedKeyAes128 {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.fill(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill(dest);
        Ok(())
    }

    fn next_u32(&mut self) -> u32 {
        next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        next_u64_via_fill(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{field::Field128, vdaf::equality_comparison_test};
    use serde::{Deserialize, Serialize};
    use std::{convert::TryInto, io::Cursor};

    #[derive(Deserialize, Serialize)]
    struct XofTestVector {
        #[serde(with = "hex")]
        seed: Vec<u8>,
        #[serde(with = "hex")]
        dst: Vec<u8>,
        #[serde(with = "hex")]
        binder: Vec<u8>,
        length: usize,
        #[serde(with = "hex")]
        derived_seed: Vec<u8>,
        #[serde(with = "hex")]
        expanded_vec_field128: Vec<u8>,
    }

    // Test correctness of dervied methods.
    fn test_xof<P, const SEED_SIZE: usize>()
    where
        P: Xof<SEED_SIZE>,
    {
        let seed = Seed::generate().unwrap();
        let dst = b"algorithm and usage";
        let binder = b"bind to artifact";

        let mut xof = P::init(seed.as_ref(), dst);
        xof.update(binder);

        let mut want = Seed([0; SEED_SIZE]);
        xof.clone().into_seed_stream().fill_bytes(&mut want.0[..]);
        let got = xof.clone().into_seed();
        assert_eq!(got, want);

        let mut want = [0; 45];
        xof.clone().into_seed_stream().fill_bytes(&mut want);
        let mut got = [0; 45];
        P::seed_stream(&seed, dst, binder).fill_bytes(&mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn xof_shake128() {
        let t: XofTestVector =
            serde_json::from_str(include_str!("test_vec/07/XofShake128.json")).unwrap();
        let mut xof = XofShake128::init(&t.seed.try_into().unwrap(), &t.dst);
        xof.update(&t.binder);

        assert_eq!(
            xof.clone().into_seed(),
            Seed(t.derived_seed.try_into().unwrap())
        );

        let mut bytes = Cursor::new(t.expanded_vec_field128.as_slice());
        let mut want = Vec::with_capacity(t.length);
        while (bytes.position() as usize) < t.expanded_vec_field128.len() {
            want.push(Field128::decode(&mut bytes).unwrap())
        }
        let got: Vec<Field128> = xof.clone().into_seed_stream().into_field_vec(t.length);
        assert_eq!(got, want);

        test_xof::<XofShake128, 16>();
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn xof_fixed_key_aes128() {
        let t: XofTestVector =
            serde_json::from_str(include_str!("test_vec/07/XofFixedKeyAes128.json")).unwrap();
        let mut xof = XofFixedKeyAes128::init(&t.seed.try_into().unwrap(), &t.dst);
        xof.update(&t.binder);

        assert_eq!(
            xof.clone().into_seed(),
            Seed(t.derived_seed.try_into().unwrap())
        );

        let mut bytes = Cursor::new(t.expanded_vec_field128.as_slice());
        let mut want = Vec::with_capacity(t.length);
        while (bytes.position() as usize) < t.expanded_vec_field128.len() {
            want.push(Field128::decode(&mut bytes).unwrap())
        }
        let got: Vec<Field128> = xof.clone().into_seed_stream().into_field_vec(t.length);
        assert_eq!(got, want);

        test_xof::<XofFixedKeyAes128, 16>();
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn xof_fixed_key_aes128_incomplete_block() {
        let seed = Seed::generate().unwrap();
        let mut expected = [0; 32];
        XofFixedKeyAes128::seed_stream(&seed, b"dst", b"binder").fill(&mut expected);

        for len in 0..=32 {
            let mut buf = vec![0; len];
            XofFixedKeyAes128::seed_stream(&seed, b"dst", b"binder").fill(&mut buf);
            assert_eq!(buf, &expected[..len]);
        }
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn xof_fixed_key_aes128_alternate_apis() {
        let dst = b"domain separation tag";
        let binder = b"AAAAAAAAAAAAAAAAAAAAAAAA";
        let seed_1 = Seed::generate().unwrap();
        let seed_2 = Seed::generate().unwrap();

        let mut stream_1_trait_api = XofFixedKeyAes128::seed_stream(&seed_1, dst, binder);
        let mut output_1_trait_api = [0u8; 32];
        stream_1_trait_api.fill(&mut output_1_trait_api);
        let mut stream_2_trait_api = XofFixedKeyAes128::seed_stream(&seed_2, dst, binder);
        let mut output_2_trait_api = [0u8; 32];
        stream_2_trait_api.fill(&mut output_2_trait_api);

        let fixed_key = XofFixedKeyAes128Key::new(dst, binder);
        let mut stream_1_alternate_api = fixed_key.with_seed(seed_1.as_ref());
        let mut output_1_alternate_api = [0u8; 32];
        stream_1_alternate_api.fill(&mut output_1_alternate_api);
        let mut stream_2_alternate_api = fixed_key.with_seed(seed_2.as_ref());
        let mut output_2_alternate_api = [0u8; 32];
        stream_2_alternate_api.fill(&mut output_2_alternate_api);

        assert_eq!(output_1_trait_api, output_1_alternate_api);
        assert_eq!(output_2_trait_api, output_2_alternate_api);
    }

    #[test]
    fn seed_equality_test() {
        equality_comparison_test(&[Seed([1, 2, 3]), Seed([3, 2, 1])])
    }
}
