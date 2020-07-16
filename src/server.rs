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
    pub fn new(dimension: usize, is_first_server: bool) -> Server {
        Server {
            dimension,
            is_first_server,
            accumulator: vector_with_length(dimension),
            validation_mem: ValidationMemory::new(dimension),
        }
    }

    pub fn generate_verification_message(
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

    pub fn aggregate(
        &mut self,
        share: &[Field],
        v1: &VerificationMessage,
        v2: &VerificationMessage,
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

    pub fn total_shares(&self) -> &[Field] {
        &self.accumulator
    }

    pub fn choose_eval_at(&self) -> Field {
        loop {
            let eval_at = Field::from(rand::random::<u32>());
            if !self.validation_mem.poly_mem.roots.contains(&eval_at) {
                break eval_at;
            }
        }
    }
}

pub struct VerificationMessage {
    f_r: Field,
    g_r: Field,
    h_r: Field,
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
    let f_r = poly_interpret_eval(
        &mem.points_f,
        &mem.poly_mem.roots_half_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );
    let g_r = poly_interpret_eval(
        &mem.points_g,
        &mem.poly_mem.roots_half_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );
    let h_r = poly_interpret_eval(
        &mem.points_h,
        &mem.poly_mem.roots_inverted,
        eval_at,
        &mut mem.poly_mem.coeffs,
        &mut mem.poly_mem.fft_memory,
    );

    let vm = VerificationMessage { f_r, g_r, h_r };
    Some(vm)
}

pub fn is_valid_share(v1: &VerificationMessage, v2: &VerificationMessage) -> bool {
    // reconstruct f_r, g_r, h_r
    let f_r = v1.f_r + v2.f_r;
    let g_r = v1.g_r + v2.g_r;
    let h_r = v1.h_r + v2.h_r;
    // validity check
    f_r * g_r == h_r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation() {
        let dim = 8;
        let share_u32: Vec<u32> = vec![
            1, 0, 0, 0, 0, 0, 0, 0, 2052337230, 3217065186, 1886032198, 2533724497, 397524722,
            3820138372, 1535223968, 4291254640, 3565670552, 2447741959, 163741941, 335831680,
            2567182742, 3542857140, 124017604, 4201373647, 431621210, 1618555683, 267689149,
        ];

        let mut share1: Vec<Field> = share_u32.iter().map(|x| Field::from(*x)).collect();
        let share2 = secret_share(&mut share1);

        let mut server1 = Server::new(dim, true);
        let mut server2 = Server::new(dim, false);
        let eval_at = Field::from(12313);
        let v1 = server1
            .generate_verification_message(eval_at, &share1)
            .unwrap();
        let v2 = server2
            .generate_verification_message(eval_at, &share2)
            .unwrap();
        assert_eq!(is_valid_share(&v1, &v2), true);
    }
}
