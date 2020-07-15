use crate::finite_field::*;
use crate::polynomial::*;
use crate::util::*;

pub struct ClientMemory {
    coeffs: Vec<Field>,
    points_f: Vec<Field>,
    points_g: Vec<Field>,
    evals_f: Vec<Field>,
    evals_g: Vec<Field>,
    poly_mem: PolyTempMemory,
}

impl ClientMemory {
    fn new(dimension: usize) -> Option<Self> {
        let n = dimension.next_power_of_two();
        if 2 * n > N_ROOTS as usize {
            return None;
        }

        Some(ClientMemory {
            coeffs: vector_with_length(2 * n),
            points_f: vector_with_length(n),
            points_g: vector_with_length(n),
            evals_f: vector_with_length(2 * n),
            evals_g: vector_with_length(2 * n),
            poly_mem: PolyTempMemory::new(2 * n),
        })
    }
}

fn poly_interpolate_eval_2n(
    n: usize,
    points_in: &[Field],
    evals_out: &mut [Field],
    coeffs: &mut [Field],
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

pub fn encode(dimension: usize, share: &mut [Field], mem: &mut ClientMemory) {
    let mut unpacked = unpack_share_mut(share, dimension).unwrap();

    construct_proof(
        &unpacked.data,
        dimension,
        &mut unpacked.f0,
        &mut unpacked.g0,
        &mut unpacked.h0,
        &mut unpacked.points_h_packed,
        mem,
    );
}

fn construct_proof(
    data: &[Field],
    dimension: usize,
    f0: &mut Field,
    g0: &mut Field,
    h0: &mut Field,
    points_h_packed: &mut [Field],
    mut mem: &mut ClientMemory,
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
        &mut mem.points_f,
        &mut mem.evals_f,
        &mut mem.coeffs,
        &mut mem.poly_mem,
    );
    poly_interpolate_eval_2n(
        proof_length / 2,
        &mut mem.points_g,
        &mut mem.evals_g,
        &mut mem.coeffs,
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
    let dimension = 80;
    let mut mem = ClientMemory::new(dimension).unwrap();
    let mut share = vector_with_length(share_length(dimension));

    encode(dimension, &mut share, &mut mem);
}
