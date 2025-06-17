// SPDX-License-Identifier: MPL-2.0

//! Tools for evaluating Prio3 test vectors.

use crate::{
    field::NttFriendlyFieldElement,
    flp::{
        gadgets::{Mul, ParallelSumGadget},
        types::SumVec,
    },
    vdaf::{
        prio3::{Prio3, Prio3Count, Prio3Histogram, Prio3MultihotCountVec, Prio3Sum},
        test_utils::TestVectorVdaf,
        xof::Xof,
    },
};
use serde_json::Value;
use std::{collections::HashMap, convert::TryInto};

impl TestVectorVdaf for Prio3Count {
    fn new(shares: u8, _parameters: &HashMap<String, Value>) -> Self {
        Prio3::new_count(shares).unwrap()
    }

    fn deserialize_measurement(measurement: &Value) -> Self::Measurement {
        measurement.as_u64().unwrap() != 0
    }

    fn deserialize_aggregate_result(aggregate_result: &Value) -> Self::AggregateResult {
        aggregate_result.as_u64().unwrap()
    }
}

impl TestVectorVdaf for Prio3Sum {
    fn new(shares: u8, parameters: &HashMap<String, Value>) -> Self {
        let max_measurement = parameters["max_measurement"].as_u64().unwrap();
        Prio3::new_sum(shares, max_measurement).unwrap()
    }

    fn deserialize_measurement(measurement: &Value) -> Self::Measurement {
        measurement.as_u64().unwrap()
    }

    fn deserialize_aggregate_result(aggregate_result: &Value) -> Self::AggregateResult {
        aggregate_result.as_u64().unwrap()
    }
}

impl<F, S, P, const SEED_SIZE: usize> TestVectorVdaf for Prio3<SumVec<F, S>, P, SEED_SIZE>
where
    F: NttFriendlyFieldElement,
    S: ParallelSumGadget<F, Mul<F>> + Eq + 'static,
    P: Xof<SEED_SIZE>,
{
    fn new(shares: u8, parameters: &HashMap<String, Value>) -> Self {
        let bits = parameters["bits"].as_u64().unwrap().try_into().unwrap();
        let length = parameters["length"].as_u64().unwrap().try_into().unwrap();
        let chunk_length = parameters["chunk_length"]
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap();
        let sum_vec = SumVec::new(bits, length, chunk_length).unwrap();
        Prio3::new(shares, 1, 0x00000003, sum_vec).unwrap()
    }

    fn deserialize_measurement(measurement: &Value) -> Self::Measurement {
        measurement
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                usize::try_from(value.as_u64().unwrap())
                    .unwrap()
                    .try_into()
                    .unwrap()
            })
            .collect()
    }

    fn deserialize_aggregate_result(aggregate_result: &Value) -> Self::AggregateResult {
        aggregate_result
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                usize::try_from(value.as_u64().unwrap())
                    .unwrap()
                    .try_into()
                    .unwrap()
            })
            .collect()
    }
}

impl TestVectorVdaf for Prio3Histogram {
    fn new(shares: u8, parameters: &HashMap<String, Value>) -> Self {
        let length = parameters["length"].as_u64().unwrap().try_into().unwrap();
        let chunk_length = parameters["chunk_length"]
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap();
        Prio3::new_histogram(shares, length, chunk_length).unwrap()
    }

    fn deserialize_measurement(measurement: &Value) -> Self::Measurement {
        measurement.as_u64().unwrap().try_into().unwrap()
    }

    fn deserialize_aggregate_result(aggregate_result: &Value) -> Self::AggregateResult {
        aggregate_result
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap().into())
            .collect()
    }
}

impl TestVectorVdaf for Prio3MultihotCountVec {
    fn new(shares: u8, parameters: &HashMap<String, Value>) -> Self {
        let length = parameters["length"].as_u64().unwrap().try_into().unwrap();
        let max_weight = parameters["max_weight"]
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap();
        let chunk_length = parameters["chunk_length"]
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap();
        Prio3::new_multihot_count_vec(shares, length, max_weight, chunk_length).unwrap()
    }

    fn deserialize_measurement(measurement: &Value) -> Self::Measurement {
        measurement
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_bool().unwrap())
            .collect()
    }

    fn deserialize_aggregate_result(aggregate_result: &Value) -> Self::AggregateResult {
        aggregate_result
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap().into())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        field::Field64,
        flp::{
            gadgets::{Mul, ParallelSum},
            types::SumVec,
        },
        vdaf::{
            prio3::{
                Prio3, Prio3Count, Prio3Histogram, Prio3MultihotCountVec, Prio3Sum, Prio3SumVec,
            },
            test_utils::{check_test_vector, check_test_vector_custom_constructor},
            xof::XofTurboShake128,
        },
    };

    #[test]
    fn test_vec_prio3_count() {
        for test_vector_str in [
            include_str!("test_vec/15/Prio3Count_0.json"),
            include_str!("test_vec/15/Prio3Count_1.json"),
            include_str!("test_vec/15/Prio3Count_2.json"),
            include_str!("test_vec/15/Prio3Count_bad_meas_share.json"),
            include_str!("test_vec/15/Prio3Count_bad_wire_seed.json"),
            include_str!("test_vec/15/Prio3Count_bad_gadget_poly.json"),
            include_str!("test_vec/15/Prio3Count_bad_helper_seed.json"),
        ] {
            let test_vector = serde_json::from_str(test_vector_str).unwrap();
            check_test_vector::<Prio3Count, 32, 16>(&test_vector);
        }
    }

    #[test]
    fn test_vec_prio3_sum() {
        for test_vector_str in [
            include_str!("test_vec/15/Prio3Sum_0.json"),
            include_str!("test_vec/15/Prio3Sum_1.json"),
            include_str!("test_vec/15/Prio3Sum_2.json"),
        ] {
            let test_vector = serde_json::from_str(test_vector_str).unwrap();
            check_test_vector::<Prio3Sum, 32, 16>(&test_vector);
        }
    }

    #[test]
    fn test_vec_prio3_sum_vec() {
        for test_vector_str in [
            include_str!("test_vec/15/Prio3SumVec_0.json"),
            include_str!("test_vec/15/Prio3SumVec_1.json"),
        ] {
            let test_vector = serde_json::from_str(test_vector_str).unwrap();
            check_test_vector::<Prio3SumVec, 32, 16>(&test_vector);
        }
    }

    #[test]
    fn test_vec_prio3_sum_vec_multiproof() {
        type Prio3SumVecField64Multiproof =
            Prio3<SumVec<Field64, ParallelSum<Field64, Mul<Field64>>>, XofTurboShake128, 32>;
        let num_proofs = 3;
        let alg_id = 0xFFFFFFFF;

        for test_vector_str in [
            include_str!("test_vec/15/Prio3SumVecWithMultiproof_0.json"),
            include_str!("test_vec/15/Prio3SumVecWithMultiproof_1.json"),
        ] {
            let test_vector = serde_json::from_str(test_vector_str).unwrap();
            check_test_vector_custom_constructor::<Prio3SumVecField64Multiproof, 32, 16>(
                &test_vector,
                |shares, parameters| {
                    let bits = parameters["bits"].as_u64().unwrap().try_into().unwrap();
                    let length = parameters["length"].as_u64().unwrap().try_into().unwrap();
                    let chunk_length = parameters["chunk_length"]
                        .as_u64()
                        .unwrap()
                        .try_into()
                        .unwrap();
                    let sum_vec = SumVec::new(bits, length, chunk_length).unwrap();
                    Prio3::new(shares, num_proofs, alg_id, sum_vec).unwrap()
                },
            );
        }
    }

    #[test]
    fn test_vec_prio3_histogram() {
        for test_vector_str in [
            include_str!("test_vec/15/Prio3Histogram_0.json"),
            include_str!("test_vec/15/Prio3Histogram_1.json"),
            include_str!("test_vec/15/Prio3Histogram_2.json"),
            include_str!("test_vec/15/Prio3Histogram_bad_leader_jr_blind.json"),
            include_str!("test_vec/15/Prio3Histogram_bad_helper_jr_blind.json"),
            include_str!("test_vec/15/Prio3Histogram_bad_public_share.json"),
            include_str!("test_vec/15/Prio3Histogram_bad_prep_msg.json"),
        ] {
            let test_vector = serde_json::from_str(test_vector_str).unwrap();
            check_test_vector::<Prio3Histogram, 32, 16>(&test_vector);
        }
    }

    #[test]
    fn test_vec_prio3_multihot_count_vec() {
        for test_vector_str in [
            include_str!("test_vec/15/Prio3MultihotCountVec_0.json"),
            include_str!("test_vec/15/Prio3MultihotCountVec_1.json"),
            include_str!("test_vec/15/Prio3MultihotCountVec_2.json"),
        ] {
            let test_vector = serde_json::from_str(test_vector_str).unwrap();
            check_test_vector::<Prio3MultihotCountVec, 32, 16>(&test_vector);
        }
    }
}
