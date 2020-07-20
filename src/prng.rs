use super::finite_field::Field;
use aes_ctr::stream_cipher::generic_array::GenericArray;
use aes_ctr::stream_cipher::NewStreamCipher;
use aes_ctr::stream_cipher::SyncStreamCipher;
use aes_ctr::Aes128Ctr;

const BLOCK_SIZE: usize = 16;
pub const SEED_LENGTH: usize = BLOCK_SIZE;

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
    let nonce = GenericArray::from_slice(&[0u8; BLOCK_SIZE]);
    let key_array = GenericArray::from_slice(&seed);
    let mut cipher = Aes128Ctr::new(&key_array, &nonce);
    let mut data = vec![0; length];
    cipher.apply_keystream(&mut data);
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_data_zero_seed() {
        let data = random_data_from_seed(&[0u8; 16], 32);

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
            0x55, 0x57, 0x6b, 0x04, 0xfe, 0x74, 0x9c, 0x4d, 0x25, 0x9b, 0x36, 0xec, 0x4f, 0x62,
            0xb8, 0x20,
        ];

        let data = random_data_from_seed(&seed, 32);

        let reference = [
            0xe4, 0xd2, 0x1d, 0xd9, 0xd3, 0xbb, 0xab, 0xdf, 0x2b, 0x51, 0x98, 0x79, 0x16, 0xc6,
            0x14, 0x9f, 0xa6, 0x86, 0xc9, 0x3c, 0xd3, 0x3a, 0x4a, 0xe4, 0x64, 0x07, 0x1d, 0x56,
            0xed, 0x8f, 0x05, 0x38,
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
            0x84, 0xfd, 0xfd, 0x87, 0x38, 0xe9, 0x8b, 0xf1, 0xf6, 0xe4, 0x36, 0xec, 0x7c, 0xc8,
            0xe0, 0xd4,
        ];

        let reference = [
            0xcdbb43cfu32,
            0x01e71d83,
            0x4352fd19,
            0x4e1f5785,
            0x8dee2ad7,
            0xe7d067ca,
            0xd4cbf324,
            0x4f9ab2dc,
            0x260ee94e,
            0x8be00cba,
            0xd169bb1d,
            0xe05ab3b7,
            0x8bacdb03,
            0x2b5f90e3,
            0x90adf992,
            0x105255be,
            0xb9be822d,
            0x9d96a1a1,
        ];

        let share2 = extract_share_from_seed(reference.len(), &seed);

        assert_eq!(share2, reference);
    }
}
