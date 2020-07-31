// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

use crate::finite_field::Field;
use aes_ctr::stream_cipher::generic_array::GenericArray;
use aes_ctr::stream_cipher::NewStreamCipher;
use aes_ctr::stream_cipher::SyncStreamCipher;
use aes_ctr::Aes128Ctr;

const BLOCK_SIZE: usize = 16;
pub const SEED_LENGTH: usize = 2 * BLOCK_SIZE;

pub fn secret_share(share1: &mut [Field]) -> Vec<u8> {
    let field_size = std::mem::size_of::<Field>();
    // get prng array
    let (data, seed) = random_data_and_seed(share1.len() * field_size);

    use std::convert::TryInto;

    for (s1, d) in share1.iter_mut().zip(data.chunks_exact(field_size)) {
        let integer = u32::from_le_bytes(d.try_into().unwrap());
        let field = Field::from(integer);
        *s1 -= field;
    }

    seed
}

pub fn extract_share_from_seed(length: usize, seed: &[u8]) -> Vec<Field> {
    assert_eq!(seed.len(), SEED_LENGTH);
    let field_size = std::mem::size_of::<Field>();
    let data = random_data_from_seed(seed, length * field_size);

    use std::convert::TryInto;

    let mut share = Vec::with_capacity(length);
    for d in data.chunks_exact(field_size) {
        let integer = u32::from_le_bytes(d.try_into().unwrap());
        share.push(Field::from(integer));
    }

    share
}

fn random_data_and_seed(length: usize) -> (Vec<u8>, Vec<u8>) {
    let mut seed = vec![0u8; SEED_LENGTH];
    use rand::RngCore;
    rand::thread_rng().fill_bytes(&mut seed);
    let data = random_data_from_seed(&seed, length);
    (data, seed)
}

fn random_data_from_seed(seed: &[u8], length: usize) -> Vec<u8> {
    let key = GenericArray::from_slice(&seed[..BLOCK_SIZE]);
    let nonce = GenericArray::from_slice(&seed[BLOCK_SIZE..]);
    let mut cipher = Aes128Ctr::new(&key, &nonce);
    let mut data = vec![0; length];
    cipher.apply_keystream(&mut data);
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_data_zero_seed() {
        let data = random_data_from_seed(&[0u8; SEED_LENGTH], 32);

        let reference = [
            0x66, 0xe9, 0x4b, 0xd4, 0xef, 0x8a, 0x2c, 0x3b, 0x88, 0x4c, 0xfa, 0x59, 0xca, 0x34,
            0x2b, 0x2e, 0x58, 0xe2, 0xfc, 0xce, 0xfa, 0x7e, 0x30, 0x61, 0x36, 0x7f, 0x1d, 0x57,
            0xa4, 0xe7, 0x45, 0x5a,
        ];

        assert_eq!(data, reference);
    }

    #[test]
    fn test_random_data_fixed_seed() {
        let seed = [
            0x1c, 0x67, 0x4b, 0xc0, 0x68, 0x98, 0xc5, 0x61, 0xf3, 0xba, 0xa4, 0xef, 0xbc, 0x61,
            0x98, 0xef, 0xe9, 0xdc, 0xc7, 0xfe, 0x3f, 0xba, 0x4f, 0x13, 0x08, 0xcb, 0x7f, 0xc9,
            0x45, 0xe1, 0x4f, 0xdd,
        ];

        let data = random_data_from_seed(&seed, 32);

        let reference = [
            0x73, 0x16, 0xe1, 0xd5, 0x5d, 0xd0, 0xd6, 0xeb, 0x4d, 0x7a, 0xde, 0x89, 0xb1, 0x83,
            0x1c, 0xef, 0x1e, 0xe1, 0x52, 0x51, 0xa2, 0x3c, 0x11, 0x34, 0xd2, 0x66, 0x77, 0x55,
            0xf8, 0xc4, 0xdc, 0x85,
        ];

        assert_eq!(data, reference);
    }

    #[test]
    fn secret_sharing() {
        let mut data = vec![Field::from(0); 123];
        data[3] = 23.into();

        let data_clone = data.clone();

        let seed = secret_share(&mut data);
        assert_ne!(data, data_clone);

        let share2 = extract_share_from_seed(data.len(), &seed);

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

        let share2 = extract_share_from_seed(reference.len(), &seed);

        assert_eq!(share2, reference);
    }
}
