// SPDX-License-Identifier: MPL-2.0

//! Implementations of XOFs specified in [[draft-irtf-cfrg-vdaf-08]].
//!
//! [draft-irtf-cfrg-vdaf-08]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/08/

/// Value of the domain separation byte "D" used by XofTurboShake128 when invoking TurboSHAKE128.
const XOF_TURBO_SHAKE_128_DOMAIN_SEPARATION: u8 = 1;
/// Value of the domain separation byte "D" used by XofFixedKeyAes128 when invoking TurboSHAKE128.
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
const XOF_FIXED_KEY_AES_128_DOMAIN_SEPARATION: u8 = 2;

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
#[cfg(feature = "crypto-dependencies")]
use hmac::{Hmac, Mac};
use rand_core::{
    impls::{next_u32_via_fill, next_u64_via_fill},
    RngCore, SeedableRng,
};

use rand::distr::{Distribution, StandardUniform};
#[cfg(feature = "crypto-dependencies")]
use sha2::Sha256;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    TurboShake128, TurboShake128Core, TurboShake128Reader,
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

impl<const SEED_SIZE: usize> Distribution<Seed<SEED_SIZE>> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Seed<SEED_SIZE> {
        let mut seed_bytes = [0; SEED_SIZE];
        rng.fill(&mut seed_bytes[..]);
        Seed(seed_bytes)
    }
}

impl<const SEED_SIZE: usize> Seed<SEED_SIZE> {
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
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&self.0[..]);
        Ok(())
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

/// An extendable output function (XOF) with the interface specified in [[draft-irtf-cfrg-vdaf-08]].
///
/// [draft-irtf-cfrg-vdaf-08]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/08/
pub trait Xof<const SEED_SIZE: usize>: Clone + Debug {
    /// The type of stream produced by this XOF.
    type SeedStream: RngCore + Sized;

    /// Construct an instance of [`Xof`] with the given seed.
    fn init(seed_bytes: &[u8; SEED_SIZE], dst_parts: &[&[u8]]) -> Self;

    /// Update the XOF state by passing in the next fragment of the info string. The final info
    /// string is assembled from the concatenation of sequence of fragments passed to this method.
    fn update(&mut self, binder_part: &[u8]);

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
    fn seed_stream(
        seed: &[u8; SEED_SIZE],
        dst_parts: &[&[u8]],
        binder_parts: &[&[u8]],
    ) -> Self::SeedStream {
        let mut xof = Self::init(seed, dst_parts);
        for binder_part in binder_parts {
            xof.update(binder_part);
        }
        xof.into_seed_stream()
    }
}

/// The key stream produced by AES128 in CTR-mode.
#[cfg(feature = "crypto-dependencies")]
#[cfg_attr(docsrs, doc(cfg(feature = "crypto-dependencies")))]
pub struct SeedStreamAes128(Ctr64BE<Aes128>);

#[cfg(feature = "crypto-dependencies")]
impl SeedStreamAes128 {
    /// Construct an instance of the seed stream with the given AES key `key` and initialization
    /// vector `iv`.
    pub fn new(key: &[u8], iv: &[u8]) -> Self {
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

/// The XOF based on TurboSHAKE128 as specified in [[draft-irtf-cfrg-vdaf-08]].
///
/// [draft-irtf-cfrg-vdaf-08]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/08/
#[derive(Clone, Debug)]
pub struct XofTurboShake128(TurboShake128);

impl XofTurboShake128 {
    pub(crate) fn from_seed_slice(seed_bytes: &[u8], dst_parts: &[&[u8]]) -> Self {
        let mut xof = Self(TurboShake128::from_core(TurboShake128Core::new(
            XOF_TURBO_SHAKE_128_DOMAIN_SEPARATION,
        )));

        let dst_len = dst_parts
            .iter()
            .map(|dst_part| dst_part.len())
            .sum::<usize>();
        let Ok(dst_len) = u16::try_from(dst_len) else {
            panic!("dst must not exceed 65535 bytes");
        };

        let Ok(seed_len) = u8::try_from(seed_bytes.len()) else {
            panic!("seed must not exceed 255 bytes");
        };

        Update::update(&mut xof.0, &dst_len.to_le_bytes());
        for dst_part in dst_parts {
            Update::update(&mut xof.0, dst_part);
        }
        Update::update(&mut xof.0, &seed_len.to_le_bytes());
        Update::update(&mut xof.0, seed_bytes);
        xof
    }
}

impl Xof<32> for XofTurboShake128 {
    type SeedStream = SeedStreamTurboShake128;

    fn init(seed_bytes: &[u8; 32], dst_parts: &[&[u8]]) -> Self {
        Self::from_seed_slice(&seed_bytes[..], dst_parts)
    }

    fn update(&mut self, data: &[u8]) {
        Update::update(&mut self.0, data);
    }

    fn into_seed_stream(self) -> SeedStreamTurboShake128 {
        SeedStreamTurboShake128::new(self.0.finalize_xof())
    }
}

/// The seed stream produced by TurboSHAKE128.
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

    fn next_u32(&mut self) -> u32 {
        next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        next_u64_via_fill(self)
    }
}

/// A `rand`-compatible interface to construct XofTurboShake128 seed streams, with the domain
/// separation tag and binder string both fixed as the empty string.
impl SeedableRng for SeedStreamTurboShake128 {
    type Seed = [u8; 32];

    fn from_seed(seed: Self::Seed) -> Self {
        XofTurboShake128::init(&seed, &[]).into_seed_stream()
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
    /// Derive the fixed key from the binder string and the domain separator, which is concatenation
    /// of all the items in `dst`.
    ///
    /// # Panics
    /// Panics if the total length of all elements of `dst` exceeds `u16::MAX`.
    pub fn new(dst: &[&[u8]], binder: &[u8]) -> Self {
        let mut fixed_key_deriver = TurboShake128::from_core(TurboShake128Core::new(
            XOF_FIXED_KEY_AES_128_DOMAIN_SEPARATION,
        ));
        let tot_dst_len: usize = dst
            .iter()
            .map(|s| {
                let len = s.len();
                assert!(len <= u16::MAX as usize, "dst must be at most 65535 bytes");
                len
            })
            .sum();

        // Feed the dst length, dst, and binder into the XOF
        fixed_key_deriver.update(
            u16::try_from(tot_dst_len)
                .expect("dst must be at most 65535 bytes")
                .to_le_bytes()
                .as_slice(),
        );
        dst.iter().for_each(|s| fixed_key_deriver.update(s));
        fixed_key_deriver.update(binder);

        // Squeeze out the key
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

/// XofFixedKeyAes128 as specified in [[draft-irtf-cfrg-vdaf-08]]. This XOF is NOT RECOMMENDED for
/// general use; see Section 9 ("Security Considerations") for details.
///
/// This XOF combines TurboSHAKE128 and a fixed-key mode of operation for AES-128. The key is
/// "fixed" in the sense that it is derived (using TurboSHAKE128) from the domain separation tag and
/// binder strings, and depending on the application, these strings can be hard-coded. The seed is
/// used to construct each block of input passed to a hash function built from AES-128.
///
/// [draft-irtf-cfrg-vdaf-08]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/08/
#[derive(Clone, Debug)]
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "crypto-dependencies", feature = "experimental")))
)]
pub struct XofFixedKeyAes128 {
    fixed_key_deriver: TurboShake128,
    base_block: Block,
}

// This impl is only used by Mastic right now. The XofFixedKeyAes128Key impl is used in cases where
// the base XOF can be reused with different contexts. This is the case in VDAF IDPF computation.
//  TODO(#1147): try to remove the duplicated code below. init() It's mostly the same as
// XofFixedKeyAes128Key::new() above
#[cfg(all(feature = "crypto-dependencies", feature = "experimental"))]
impl Xof<16> for XofFixedKeyAes128 {
    type SeedStream = SeedStreamFixedKeyAes128;

    fn init(seed_bytes: &[u8; 16], dst_parts: &[&[u8]]) -> Self {
        let mut fixed_key_deriver = TurboShake128::from_core(TurboShake128Core::new(2u8));
        let dst_len = dst_parts
            .iter()
            .map(|dst_part| dst_part.len())
            .sum::<usize>();
        Update::update(
            &mut fixed_key_deriver,
            u16::try_from(dst_len)
                .expect("dst must be at most 65535 bytes")
                .to_le_bytes()
                .as_slice(),
        );
        for dst_part in dst_parts {
            Update::update(&mut fixed_key_deriver, dst_part);
        }
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
        for block_counter in self.length_consumed / 16..next_length_consumed.div_ceil(16) {
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

    fn next_u32(&mut self) -> u32 {
        next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        next_u64_via_fill(self)
    }
}

/// XOF based on HMAC-SHA256 and AES128. This XOF is not part of the VDAF spec.
#[cfg(feature = "crypto-dependencies")]
#[cfg_attr(docsrs, doc(cfg(feature = "crypto-dependencies")))]
#[derive(Clone, Debug)]
pub struct XofHmacSha256Aes128(Hmac<Sha256>);

#[cfg(feature = "crypto-dependencies")]
impl Xof<32> for XofHmacSha256Aes128 {
    type SeedStream = SeedStreamAes128;

    fn init(seed_bytes: &[u8; 32], dst_parts: &[&[u8]]) -> Self {
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(seed_bytes).unwrap();
        let dst_len = dst_parts
            .iter()
            .map(|dst_part| dst_part.len())
            .sum::<usize>();
        Mac::update(
            &mut mac,
            &[dst_len.try_into().expect("dst must be at most 255 bytes")],
        );
        for dst_part in dst_parts {
            Mac::update(&mut mac, dst_part);
        }
        Self(mac)
    }

    fn update(&mut self, data: &[u8]) {
        Mac::update(&mut self.0, data);
    }

    fn into_seed_stream(self) -> SeedStreamAes128 {
        let tag = Mac::finalize(self.0).into_bytes();
        let (key, iv) = tag.split_at(16);
        SeedStreamAes128::new(key, iv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{field::Field128, vdaf::equality_comparison_test};
    use rand::{rng, Rng, RngCore};
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

    /// Test correctness of dervied methods.
    fn test_xof<P, const SEED_SIZE: usize>()
    where
        P: Xof<SEED_SIZE>,
    {
        let mut rng = rng();
        let seed = rng.random::<Seed<SEED_SIZE>>();
        let dst = b"algorithm and usage";
        let binder = b"bind to artifact";

        let mut xof = P::init(seed.as_ref(), &[dst]);
        xof.update(binder);

        let mut want = Seed([0; SEED_SIZE]);
        xof.clone().into_seed_stream().fill_bytes(&mut want.0[..]);
        let got = xof.clone().into_seed();
        assert_eq!(got, want);

        let mut want = [0; 45];
        xof.clone().into_seed_stream().fill_bytes(&mut want);
        let mut got = [0; 45];
        P::seed_stream(seed.as_ref(), &[dst], &[binder]).fill_bytes(&mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn xof_turboshake128() {
        let t: XofTestVector =
            serde_json::from_str(include_str!("test_vec/15/XofTurboShake128.json")).unwrap();
        let mut xof = XofTurboShake128::init(&t.seed.try_into().unwrap(), &[&t.dst]);
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

        test_xof::<XofTurboShake128, 32>();
    }

    #[test]
    fn xof_hmac_sha256_aes128() {
        let t: XofTestVector =
            serde_json::from_str(include_str!("test_vec/XofHmacSha256Aes128.json")).unwrap();

        let mut xof = XofHmacSha256Aes128::init(&t.seed.try_into().unwrap(), &[&t.dst]);
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

        test_xof::<XofHmacSha256Aes128, 32>();
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn xof_fixed_key_aes128() {
        let t: XofTestVector =
            serde_json::from_str(include_str!("test_vec/15/XofFixedKeyAes128.json")).unwrap();
        let mut xof = XofFixedKeyAes128::init(&t.seed.try_into().unwrap(), &[&t.dst]);
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
        let mut rng = rng();
        let seed = rng.random::<Seed<16>>();
        let mut expected = [0; 32];
        XofFixedKeyAes128::seed_stream(seed.as_ref(), &[b"dst"], &[b"binder"]).fill(&mut expected);

        for len in 0..=32 {
            let mut buf = vec![0; len];
            XofFixedKeyAes128::seed_stream(seed.as_ref(), &[b"dst"], &[b"binder"]).fill(&mut buf);
            assert_eq!(buf, &expected[..len]);
        }
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn xof_fixed_key_aes128_alternate_apis() {
        let mut rng = rng();
        let fixed_dst = b"domain separation tag";
        let ctx = b"context string";
        let binder = b"AAAAAAAAAAAAAAAAAAAAAAAA";
        let seed_1 = rng.random::<Seed<16>>();
        let seed_2 = rng.random::<Seed<16>>();

        let mut stream_1_trait_api =
            XofFixedKeyAes128::seed_stream(seed_1.as_ref(), &[fixed_dst, ctx], &[binder]);
        let mut output_1_trait_api = [0u8; 32];
        stream_1_trait_api.fill(&mut output_1_trait_api);
        let mut stream_2_trait_api =
            XofFixedKeyAes128::seed_stream(seed_2.as_ref(), &[fixed_dst, ctx], &[binder]);
        let mut output_2_trait_api = [0u8; 32];
        stream_2_trait_api.fill(&mut output_2_trait_api);

        let fixed_key = XofFixedKeyAes128Key::new(&[fixed_dst, ctx], binder);
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
