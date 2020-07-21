use crate::encrypt::*;
use crate::finite_field::*;
use crate::polynomial::*;
use crate::util::*;

pub struct Client {
    dimension: usize,
    points_f: Vec<Field>,
    points_g: Vec<Field>,
    evals_f: Vec<Field>,
    evals_g: Vec<Field>,
    poly_mem: PolyAuxMemory,
    public_key1: PublicKey,
    public_key2: PublicKey,
}

impl Client {
    pub fn new(dimension: usize, public_key1: PublicKey, public_key2: PublicKey) -> Option<Self> {
        let n = (dimension + 1).next_power_of_two();

        if 2 * n > N_ROOTS as usize {
            // too many elements for this field, not enough roots of unity
            return None;
        }

        Some(Client {
            dimension,
            points_f: vector_with_length(n),
            points_g: vector_with_length(n),
            evals_f: vector_with_length(2 * n),
            evals_g: vector_with_length(2 * n),
            poly_mem: PolyAuxMemory::new(n),
            public_key1,
            public_key2,
        })
    }

    pub fn encode_simple(&mut self, data: &[Field]) -> Result<(Vec<u8>, Vec<u8>), EncryptError> {
        let copy_data = |share_data: &mut [Field]| {
            share_data[..].clone_from_slice(data);
        };
        self.encode_with(copy_data)
    }

    pub fn encode_with<F>(&mut self, init_function: F) -> Result<(Vec<u8>, Vec<u8>), EncryptError>
    where
        F: FnOnce(&mut [Field]),
    {
        let mut proof = vector_with_length(proof_length(self.dimension));
        // unpack one long vector to different subparts
        let mut unpacked = unpack_proof_mut(&mut proof, self.dimension).unwrap();
        // initialize the data part
        init_function(&mut unpacked.data);
        // fill in the rest
        construct_proof(
            &unpacked.data,
            self.dimension,
            &mut unpacked.f0,
            &mut unpacked.g0,
            &mut unpacked.h0,
            &mut unpacked.points_h_packed,
            self,
        );

        // use prng to share the proof: share2 is the PRNG seed, and proof is mutated in place
        let share2 = crate::prng::secret_share(&mut proof);
        let share1 = serialize(&proof);
        // encrypt shares with respective keys
        let encrypted_share1 = encrypt_share(&share1, &self.public_key1)?;
        let encrypted_share2 = encrypt_share(&share2, &self.public_key2)?;
        Ok((encrypted_share1, encrypted_share2))
    }
}

pub fn encode_simple(
    data: &[Field],
    public_key1: PublicKey,
    public_key2: PublicKey,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let dimension = data.len();
    let mut client_memory = Client::new(dimension, public_key1, public_key2)?;
    client_memory.encode_simple(data).ok()
}

fn interpolate_and_evaluate_at_2n(
    n: usize,
    points_in: &[Field],
    evals_out: &mut [Field],
    mem: &mut PolyAuxMemory,
) {
    // interpolate through roots of unity
    poly_fft(
        &mut mem.coeffs,
        points_in,
        &mem.roots_n_inverted,
        n,
        true,
        &mut mem.fft_memory,
    );
    // evaluate at 2N roots of unity
    poly_fft(
        evals_out,
        &mem.coeffs,
        &mem.roots_2n,
        2 * n,
        false,
        &mut mem.fft_memory,
    );
}

fn construct_proof(
    data: &[Field],
    dimension: usize,
    f0: &mut Field,
    g0: &mut Field,
    h0: &mut Field,
    points_h_packed: &mut [Field],
    mem: &mut Client,
) {
    let n = (dimension + 1).next_power_of_two();

    // set zero terms to random
    *f0 = Field::from(rand::random::<u32>());
    *g0 = Field::from(rand::random::<u32>());
    mem.points_f[0] = *f0;
    mem.points_g[0] = *g0;

    // set zero term for the proof polynomial
    *h0 = *f0 * *g0;

    // set f_i = data_(i - 1)
    // set g_i = f_i - 1
    for i in 0..dimension {
        mem.points_f[i + 1] = data[i];
        mem.points_g[i + 1] = data[i] - 1.into();
    }

    // interpolate and evaluate at roots of unity
    interpolate_and_evaluate_at_2n(n, &mem.points_f, &mut mem.evals_f, &mut mem.poly_mem);
    interpolate_and_evaluate_at_2n(n, &mem.points_g, &mut mem.evals_g, &mut mem.poly_mem);

    // calculate the proof polynomial as evals_f(r) * evals_g(r)
    // only add non-zero points
    let mut j: usize = 0;
    let mut i: usize = 1;
    while i < 2 * n {
        points_h_packed[j] = mem.evals_f[i] * mem.evals_g[i];
        j += 1;
        i += 2;
    }
}

#[test]
fn test_encode() {
    let pub_key1 = PublicKey::from_base64(
        "BIl6j+J6dYttxALdjISDv6ZI4/VWVEhUzaS05LgrsfswmbLOgNt9HUC2E0w+9RqZx3XMkdEHBHfNuCSMpOwofVQ=",
    )
    .unwrap();
    let pub_key2 = PublicKey::from_base64(
        "BNNOqoU54GPo+1gTPv+hCgA9U2ZCKd76yOMrWa1xTWgeb4LhFLMQIQoRwDVaW64g/WTdcxT4rDULoycUNFB60LE=",
    )
    .unwrap();

    let data_u32 = [0u32, 1, 0, 1, 1, 0, 0, 0, 1];
    let data = data_u32
        .iter()
        .map(|x| Field::from(*x))
        .collect::<Vec<Field>>();
    let encoded_shares = encode_simple(&data, pub_key1, pub_key2);
    assert_eq!(encoded_shares.is_some(), true);
}
