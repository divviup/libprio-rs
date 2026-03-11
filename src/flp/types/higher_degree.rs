//! Implements a VDAF using a degree 3 gadget, for test purposes.

use crate::{
    field::Field64 as TestField,
    flp::{gadgets::PolyEval, types::decode_result, Flp, FlpError, Gadget, Type},
    vdaf::{prio3::Prio3, xof::XofTurboShake128, VdafError},
};

/// A VDAF that runs the [`HigherDegree`] type. The extensible output function used and the
/// algorithm ID match the reference implementation ([1]) used to compute test vectors.
///
/// [1]: https://github.com/cfrg/draft-irtf-cfrg-vdaf/blob/9825c287f1588248539748c9ff434ad966102bb1/poc/tests/test_vdaf_prio3.py#L33
pub(crate) type Prio3HigherDegree = Prio3<HigherDegree, XofTurboShake128, 32>;

impl Prio3HigherDegree {
    pub(crate) fn new_higher_degree(num_aggregators: u8) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, 1, 0xFFFFFFFF, HigherDegree::new())
    }
}

/// Validity circuit using a degree 3 gadget, for test purposes.
///
/// This circuit evaluates x * (x - 1) * (x - 2), and only accepts the measurements 0, 1, and 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HigherDegree;

impl HigherDegree {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Flp for HigherDegree {
    type Field = TestField;

    fn gadget(&self) -> Vec<Box<dyn Gadget<Self::Field>>> {
        vec![Box::new(PolyEval::new(
            vec![
                TestField::from(0),
                TestField::from(2),
                -TestField::from(3),
                TestField::from(1),
            ],
            1,
        ))]
    }

    fn num_gadgets(&self) -> usize {
        1
    }

    fn valid(
        &self,
        gadgets: &mut Vec<Box<dyn Gadget<Self::Field>>>,
        input: &[Self::Field],
        joint_rand: &[Self::Field],
        _num_shares: usize,
    ) -> Result<Vec<Self::Field>, FlpError> {
        self.valid_call_check(input, joint_rand)?;
        let check = gadgets[0].eval(input)?;
        Ok(vec![check])
    }

    fn input_len(&self) -> usize {
        1
    }

    fn proof_len(&self) -> usize {
        let gadget_arity = 1;
        let gadget_degree = 3;
        let gadget_calls = 1usize;
        let p = (gadget_calls + 1).next_power_of_two();
        gadget_arity + gadget_degree * (p - 1) + 1
    }

    fn verifier_len(&self) -> usize {
        3
    }

    fn joint_rand_len(&self) -> usize {
        0
    }

    fn eval_output_len(&self) -> usize {
        1
    }

    fn prove_rand_len(&self) -> usize {
        1
    }
}

impl Type for HigherDegree {
    type Measurement = u64;

    type AggregateResult = u64;

    fn encode_measurement(
        &self,
        measurement: &Self::Measurement,
    ) -> Result<Vec<Self::Field>, FlpError> {
        Ok(vec![TestField::from(*measurement)])
    }

    fn truncate(&self, input: Vec<Self::Field>) -> Result<Vec<Self::Field>, FlpError> {
        self.truncate_call_check(&input)?;
        Ok(input)
    }

    fn decode_result(
        &self,
        data: &[Self::Field],
        _num_measurements: usize,
    ) -> Result<Self::AggregateResult, FlpError> {
        decode_result(data)
    }

    fn output_len(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use crate::flp::test_utils::TypeTest;

    use super::*;

    #[test]
    fn test_degree_3() {
        let typ = HigherDegree::new();
        TypeTest::expect_valid::<2>(&typ, &[TestField::from(0)], &[TestField::from(0)]);
        TypeTest::expect_valid::<2>(&typ, &[TestField::from(1)], &[TestField::from(1)]);
        TypeTest::expect_valid::<2>(&typ, &[TestField::from(2)], &[TestField::from(2)]);
        TypeTest::expect_valid::<3>(&typ, &[TestField::from(2)], &[TestField::from(2)]);
        TypeTest::expect_invalid::<2>(&typ, &[TestField::from(3)]);
        TypeTest::expect_invalid::<3>(&typ, &[TestField::from(3)]);
    }
}
