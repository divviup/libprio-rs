use crate::finite_field::*;
use crate::polynomial::*;
use crate::util::*;

pub struct ClientMemory {
    dimension: usize,
    points_f: Vec<Field>,
    points_g: Vec<Field>,
    evals_f: Vec<Field>,
    evals_g: Vec<Field>,
    poly_mem: PolyTempMemory,
}

impl ClientMemory {
    pub fn new(dimension: usize) -> Option<Self> {
        let n = (dimension + 1).next_power_of_two();

        if 2 * n > N_ROOTS as usize {
            // too many elements for this field size
            return None;
        }

        Some(ClientMemory {
            dimension,
            points_f: vector_with_length(n),
            points_g: vector_with_length(n),
            evals_f: vector_with_length(2 * n),
            evals_g: vector_with_length(2 * n),
            poly_mem: PolyTempMemory::new(2 * n),
        })
    }

    pub fn encode_simple(&mut self, data: &[Field]) -> (Vec<u8>, Vec<u8>) {
        let copy_data = |share_data: &mut [Field]| {
            share_data[..].clone_from_slice(data);
        };
        self.encode_with(copy_data)
    }

    pub fn encode_with<F>(&mut self, init_function: F) -> (Vec<u8>, Vec<u8>)
    where
        F: FnOnce(&mut [Field]),
    {
        let mut data_and_proof = vector_with_length(share_length(self.dimension));
        // unpack one long vector to different subparts
        let mut unpacked = unpack_share_mut(&mut data_and_proof, self.dimension).unwrap();
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

        // use prng to share the
        let share2 = crate::prng::secret_share(&mut data_and_proof);
        let share1 = serialize(&data_and_proof);
        (share1, share2)
    }
}

pub fn encode_simple(data: &[Field]) -> Option<(Vec<u8>, Vec<u8>)> {
    let dimension = data.len();
    let mut client_memory = ClientMemory::new(dimension)?;
    Some(client_memory.encode_simple(data))
}

fn poly_interpolate_eval_2n(
    n: usize,
    points_in: &[Field],
    evals_out: &mut [Field],
    mem: &mut PolyTempMemory,
) {
    // interpolate through roots of unity
    poly_fft(
        &mut mem.coeffs,
        points_in,
        &mem.roots_half_inverted,
        n,
        true,
        &mut mem.fft_memory,
    );
    // evaluate at 2N roots of unity
    poly_fft(
        evals_out,
        &mem.coeffs,
        &mem.roots,
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
    mem: &mut ClientMemory,
) {
    let proof_length = 2 * (dimension + 1).next_power_of_two();

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
    poly_interpolate_eval_2n(
        proof_length / 2,
        &mem.points_f,
        &mut mem.evals_f,
        &mut mem.poly_mem,
    );
    poly_interpolate_eval_2n(
        proof_length / 2,
        &mem.points_g,
        &mut mem.evals_g,
        &mut mem.poly_mem,
    );

    // calculate the proof polynomial as evals_f(r) * evals_g(r)
    // only add non-zero points
    let mut j: usize = 0;
    let mut i: usize = 1;
    while i < proof_length {
        points_h_packed[j] = mem.evals_f[i] * mem.evals_g[i];
        j += 1;
        i += 2;
    }
}

#[test]
fn test_encode() {
    let data_u32 = [0u32, 1, 0, 1, 1, 0, 0, 0, 1];
    let data = data_u32
        .iter()
        .map(|x| Field::from(*x))
        .collect::<Vec<Field>>();
    let encoded_shares = encode_simple(&data);
    assert_eq!(encoded_shares.is_some(), true);
}
