use crate::finite_field::*;
use crate::polynomial::*;
use crate::util::*;

pub struct ValidationMemory {
    points_f: Vec<Field>,
    points_g: Vec<Field>,
    points_h: Vec<Field>,
    poly_mem: PolyTempMemory,
}

impl ValidationMemory {
    fn new(dimension: usize) -> Self {
        let n: usize = (dimension + 1).next_power_of_two();
        ValidationMemory {
            points_f: vector_with_length(n),
            points_g: vector_with_length(n),
            points_h: vector_with_length(2 * n),
            poly_mem: PolyTempMemory::new(2 * n),
        }
    }
}

pub struct Server {
    dimension: usize,
    is_first_server: bool,
    accumulator: Vec<Field>,
    validation_mem: ValidationMemory,
}

impl Server {
    fn new(dimension: usize, is_first_server: bool) -> Server {
        Server {
            dimension,
            is_first_server,
            accumulator: vector_with_length(dimension),
            validation_mem: ValidationMemory::new(dimension),
        }
    }

    fn generate_verification_message(
        &mut self,
        eval_at: Field,
        share: &[Field],
    ) -> Option<VerificationMessage> {
        generate_verification_message(
            self.dimension,
            eval_at,
            share,
            self.is_first_server,
            &mut self.validation_mem,
        )
    }

    fn aggregate(
        &mut self,
        share: &[Field],
        v1: VerificationMessage,
        v2: VerificationMessage,
    ) -> bool {
        if share_length(self.dimension) != share.len() {
            return false;
        }
        let is_valid = is_valid_share(v1, v2);
        if is_valid {
            // add to the accumulator
            for (i, a) in self.accumulator.iter_mut().enumerate() {
                *a = *a + share[i];
            }
        }

        is_valid
    }
}

pub struct VerificationMessage {
    fR: Field,
    gR: Field,
    hR: Field,
}

pub fn generate_verification_message(
    dimension: usize,
    eval_at: Field,
    share: &[Field],
    is_first_server: bool,
    mem: &mut ValidationMemory,
) -> Option<VerificationMessage> {
    let unpacked = unpack_share(share, dimension)?;
    let proof_length = 2 * (dimension + 1).next_power_of_two();

    // set zero terms
    mem.points_f[0] = *unpacked.f0;
    mem.points_g[0] = *unpacked.g0;
    mem.points_h[0] = *unpacked.h0;

    // set points_f and points_g
    for (i, x) in unpacked.data.iter().enumerate() {
        mem.points_f[i + 1] = *x;

        if is_first_server {
            // only one server needs to subtract one
            mem.points_g[i + 1] = *x - 1.into();
        } else {
            mem.points_g[i + 1] = *x;
        }
    }

    // set points_h, skipping over elements that should be zero
    let mut i = 1;
    let mut j = 0;
    while i < proof_length {
        mem.points_h[i] = unpacked.points_h_packed[j];
        j += 1;
        i += 2;
    }

    // evaluate polynomials at random point
    let fR = poly_interpret_eval(
        &mem.points_f,
        &mem.poly_mem.roots_half_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );
    let gR = poly_interpret_eval(
        &mem.points_g,
        &mem.poly_mem.roots_half_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );
    let hR = poly_interpret_eval(
        &mem.points_h,
        &mem.poly_mem.roots_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );

    let vm = VerificationMessage { fR, gR, hR };
    Some(vm)
}

pub fn is_valid_share(v1: VerificationMessage, v2: VerificationMessage) -> bool {
    // reconstruct fR, gR, hR
    let fR = v1.fR + v2.fR;
    let gR = v1.gR + v2.gR;
    let hR = v1.hR + v2.hR;
    // validity check
    fR * gR == hR
}
