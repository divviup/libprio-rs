//! Implementation of the Prio3L1BoundSum VDAF defined in [1].
//!
//! [1]: https://martinthomson.github.io/prio-l1-bound-sum/draft-thomson-ppm-l1-bound-sum.html

use crate::{
    field::Field128,
    flp::{
        gadgets::{Mul, ParallelSum},
        types::L1BoundSum,
    },
    vdaf::{prio3::Prio3, xof::XofTurboShake128, VdafError},
};

/// A VDAF which sums over vectors, checking that each element of each vector is within some range
/// and that the L1 norm of each vector matches a claimed weight encoded into the input, which
/// itself must be within the same range as the vector elements, effectively bounding the L1 norm
/// to 2^bits - 1.
pub type Prio3L1BoundSum =
    Prio3<L1BoundSum<Field128, ParallelSum<Field128, Mul<Field128>>>, XofTurboShake128, 32>;

impl Prio3L1BoundSum {
    /// Construct an instance of Prio3L1BoundSum with the given number of aggregators. `bits` defines
    /// the bit width of each summand of the measurement; `len` defines the length of the
    /// measurement vector.
    pub fn new_l1_bound_sum(
        num_aggregators: u8,
        bits: usize,
        len: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(
            num_aggregators,
            1,
            // TODO: use real codepoint once assigned
            0xFFFF0000,
            L1BoundSum::new(bits, len, chunk_length)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vdaf::test_utils::run_vdaf;

    #[test]
    fn test() {
        let l1_bound_sum = Prio3::new_l1_bound_sum(2, 3, 4, 3).unwrap();

        assert_eq!(
            run_vdaf(
                b"context",
                &l1_bound_sum,
                &(),
                [
                    [7, 0, 0, 0].to_vec(),
                    [0, 0, 0, 7].to_vec(),
                    [0, 0, 0, 0].to_vec(),
                    [1, 2, 2, 2].to_vec(),
                    [0, 1, 0, 0].to_vec(),
                ]
            )
            .unwrap(),
            vec![8, 3, 2, 9]
        );
    }
}
