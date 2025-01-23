// SPDX-License-Identifier: MPL-2.0

//! The approximate heavy hitters protocol of [[RZCGP24], Section 5].
//!
//! [RZCGP24]: https://eprint.iacr.org/2024/666

use crate::{
    field::{Field64, FieldElement, FieldElementWithInteger},
    flp::{gadgets::PolyEval, Flp, FlpError, Gadget, Type},
    fp::log2,
    vdaf::VdafError,
    vidpf::VidpfInput,
};

use sha3::{Digest, Sha3_256};
use std::iter::once;

use super::{Mastic, MasticAggregationParam};

/// A [`Mastic`] variant for the approximate heavy hitters protocol of [[RZCGP24], Section 5].
///
/// This VDAF uses a single round of aggregation in order to recover to compute an approximation of
/// the heavy hitters of strings held by clients. The VDAF has two parameters: the length in bit of
/// the strings and a number of "buckets". The weight for Mastic is an encoding of the client's bit
/// string; the Mastic input is the bucket for that string, computed by hashing the string.
///
/// The aggregators evaluate the VIDPF at each possible bucket, thereby expanding the client's
/// input into a vector with one non-zero entry. The non-zero entry contains the client's string.
/// The aggregate result counts how many times each string occurs; any entry with count greater
/// than the threshold is a heavy hitter.
///
/// The main advantage of this protocol over [`Poplar1`](crate::vdaf::poplar1::Poplar1) or
/// [`Mastic`] in weighted heavy-hitters mode is that it requires one round of aggregation rather
/// than many. The cost of this efficiency is a larger report. It also has some important caveats:
///
/// * This VDAF doesn't compute the exact heavy hitters. It's possible for two input strings to
///   hash to the same entry. In fact, a misbehaving client can choose whatever bucket it wishes.
///   The weight is encoded in a way that allows the collector to correct for this. However, in
///   rare cases it's possible that a true heavy hitter may not be recovered.
///
/// * This VDAF leaks information about light hitters, similar to
///   [`Poplar1`](crate::vdaf::poplar1::Poplar1) or [`Mastic`] when used in weighted heavy-hitters
///   mode. It will be necessary to figure out some mechanism for differential privacy to use with
///   this scheme.
///
/// [RZCGP24]: https://eprint.iacr.org/2024/666
pub type MasticHeavyHittersSketch = Mastic<HeavyHittersSketch>;

impl MasticHeavyHittersSketch {
    /// Construct a new instance of [`MasticHeavyHittersSketch`].
    pub fn new_heavy_hitters_sketch(
        num_bits: u16,
        num_buckets: u32,
        threshold: usize,
    ) -> Result<Self, VdafError> {
        // TODO Assign a codepoint for this instance of Mastic.
        const ID: u32 = 0xfffffeed;

        let hh = HeavyHittersSketch {
            num_bits,
            num_buckets,
            threshold,
        };

        let bits = log2(u128::from(num_buckets)).try_into().unwrap();
        Mastic::new(ID, hh, bits)
    }

    /// Convert a bit string to a [`Mastic`] measurement.
    pub fn to_mastic_input(&self, measurement: &[bool]) -> Result<VidpfInput, VdafError> {
        let (_sign, bucket) = self.szk.typ.sign_and_bucket(measurement)?;
        Ok(self.bucket_to_mastic_input(bucket))
    }

    /// Return the [`Mastic`] aggregation parameter to use for this variant.
    pub fn get_mastic_agg_param(&self) -> Result<MasticAggregationParam, VdafError> {
        let prefixes: Vec<_> = (0..self.szk.typ.num_buckets)
            .map(|bucket| self.bucket_to_mastic_input(bucket))
            .collect();
        MasticAggregationParam::new(prefixes, true)
    }

    fn bucket_to_mastic_input(&self, bucket: u32) -> VidpfInput {
        let mut bits = Vec::with_capacity(usize::from(self.vidpf.bits));
        for i in (0..self.vidpf.bits).rev() {
            bits.push(bucket >> i & 1 == 1);
        }
        VidpfInput::from_bools(&bits)
    }
}

/// FLP used for [`MasticHeavyHittersSketch`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeavyHittersSketch {
    num_bits: u16,
    num_buckets: u32,
    threshold: usize,
}

impl HeavyHittersSketch {
    /// [RZCGP24], Section 5.
    fn sign_and_bucket(&self, measurement: &[bool]) -> Result<(bool, u32), FlpError> {
        if measurement.len() != usize::from(self.num_bits) {
            return Err(FlpError::Encode("unexpected measurement length".into()));
        }

        // TODO Replace this with something we'd want to standardize.
        let mut hasher = Sha3_256::new();
        hasher.update(self.num_bits.to_be_bytes());
        hasher.update(self.num_buckets.to_be_bytes());
        hasher.update(VidpfInput::from_bools(measurement).to_bytes());
        let hash = hasher.finalize();
        let bucket = u32::from_be_bytes(<[u8; 4]>::try_from(&hash[..4]).unwrap());
        let sign = hash[4] & 1 == 1;
        Ok((sign, bucket))
    }
}

impl Type for HeavyHittersSketch {
    type Measurement = Vec<bool>;
    type AggregateResult = Option<Vec<bool>>;

    fn encode_measurement(&self, measurement: &Vec<bool>) -> Result<Vec<Field64>, FlpError> {
        let (sign, _bucket) = self.sign_and_bucket(measurement)?;

        // [RZCGP24], Figure 5:
        //
        // The first field element, denoted `beta`, is computed by hashing the measurement: if
        // `beta == 0`, then we compliment the bits of the input before encoding.
        //
        // For each bit of the input, if `bit == 1`, then we encoded it as `1`; otherwise we encode
        // it as `-1`.
        //
        // TODO Rewrite this to avoid branching on the value of `sign` or `bit`.
        Ok(once(if sign {
            Field64::one()
        } else {
            -Field64::one()
        })
        .chain(measurement.iter().map(|bit| {
            if bit ^ !sign {
                Field64::one()
            } else {
                -Field64::one()
            }
        }))
        .collect())
    }

    fn truncate(&self, input: Vec<Field64>) -> Result<Vec<Field64>, FlpError> {
        Ok(input)
    }

    fn decode_result(
        &self,
        data: &[Field64],
        num_measurements: usize,
    ) -> Result<Option<Vec<bool>>, FlpError> {
        if num_measurements < self.threshold {
            return Ok(None);
        }

        let m = Field64::modulus() - u64::try_from(num_measurements).unwrap();

        // [RZCGP24], Figure 5: Round the data by mapping each value larger than `n` to `0` and
        // every other value to `1`.
        let sign = (1..m).contains(&u64::from(data[0]));
        let heavy_hitter = data[1..]
            .iter()
            .copied()
            .map(|x| (1..m).contains(&u64::from(x)) ^ !sign)
            .collect();
        Ok(Some(heavy_hitter))
    }

    fn output_len(&self) -> usize {
        self.input_len()
    }
}

impl Flp for HeavyHittersSketch {
    type Field = Field64;

    fn gadget(&self) -> Vec<Box<dyn Gadget<Field64>>> {
        // p(x) = (x+1)(x-1)
        let p = vec![-Field64::one(), Field64::zero(), Field64::one()];
        vec![Box::new(PolyEval::new(p, 1 + usize::from(self.num_bits)))]
    }

    fn num_gadgets(&self) -> usize {
        1
    }

    fn valid(
        &self,
        gadgets: &mut Vec<Box<dyn Gadget<Field64>>>,
        input: &[Field64],
        joint_rand: &[Field64],
        _num_shares: usize,
    ) -> Result<Vec<Field64>, FlpError> {
        self.valid_call_check(input, joint_rand)?;

        // Check that each input is either a `1` or a `-1`.
        input
            .iter()
            .copied()
            .map(|x| gadgets[0].call(&[x]))
            .collect()
    }

    fn input_len(&self) -> usize {
        1 + usize::from(self.num_bits)
    }

    fn proof_len(&self) -> usize {
        2 * ((2 + usize::from(self.num_bits)).next_power_of_two() - 1) + 2
    }

    fn verifier_len(&self) -> usize {
        3
    }

    fn joint_rand_len(&self) -> usize {
        0
    }

    fn eval_output_len(&self) -> usize {
        self.input_len()
    }

    fn prove_rand_len(&self) -> usize {
        1
    }

    fn query_rand_len(&self) -> usize {
        1 + self.input_len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        field::add_vector, flp::test_utils::TypeTest,
        idpf::test_utils::generate_zipf_distributed_batch, vdaf::test_utils::run_vdaf,
    };
    use rand::prelude::*;

    // Default is useful for tests, but we don't want this to be used in release builds.
    #[allow(clippy::derivable_impls)]
    impl Default for HeavyHittersSketch {
        fn default() -> Self {
            Self {
                num_bits: 0,
                num_buckets: 0,
                threshold: 0,
            }
        }
    }

    #[test]
    fn type_roundtrip() {
        let hh = HeavyHittersSketch {
            num_bits: 256,
            ..Default::default()
        };

        let gen_meas = || {
            let mut rng = thread_rng();
            std::iter::repeat_with(|| rng.gen())
                .take(usize::from(hh.num_bits))
                .collect::<Vec<bool>>()
        };

        let heavy_hitter = gen_meas();
        let mut measurements = vec![heavy_hitter.clone(); 2];
        // We should be able to decode even when some of the strings in the bucket don't match the
        // heavy hitter.
        measurements.push(gen_meas());

        let decoded = hh
            .decode_result(
                &measurements
                    .iter()
                    .map(|bits| hh.encode_measurement(bits).unwrap())
                    .reduce(add_vector)
                    .unwrap(),
                measurements.len(),
            )
            .unwrap();
        assert_eq!(decoded, Some(heavy_hitter));
    }

    #[test]
    fn flp() {
        let hh = HeavyHittersSketch {
            num_bits: 256,
            ..Default::default()
        };

        TypeTest::expect_valid::<2>(
            &hh,
            &hh.encode_measurement(&vec![true; usize::from(hh.num_bits)])
                .unwrap(),
            &vec![-Field64::one(); 1 + usize::from(hh.num_bits)],
        );
    }

    #[test]
    fn vdaf() {
        let mastic = Mastic::new_heavy_hitters_sketch(2, 100, 2).unwrap();

        let measurements = generate_zipf_distributed_batch(
            &mut thread_rng(),
            usize::from(mastic.szk.typ.num_bits),
            10,
            5,
            10,
            1.03,
        )
        .0
        .into_iter()
        .map(|input| input.iter().collect::<Vec<bool>>())
        .map(|measurement| (mastic.to_mastic_input(&measurement).unwrap(), measurement))
        .collect::<Vec<_>>();

        let mut count = HashMap::<Vec<bool>, usize>::new();
        for (_input, weight) in measurements.iter() {
            println!("{weight:?}");
            *(count.entry(weight.clone()).or_default()) += 1;
        }
        println!("counts: {count:?}");

        let agg_param = mastic.get_mastic_agg_param().unwrap();

        let heavy_hitters = run_vdaf(b"some application", &mastic, &agg_param, measurements)
            .unwrap()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        println!("heavy hitters: {heavy_hitters:?}");

        // Make sure every output is actually a heavy hitter.
        for heavy_hitter in heavy_hitters.iter() {
            if *count.get(heavy_hitter).unwrap() < mastic.szk.typ.threshold {
                panic!("non-heavy hitter output by Collector: {heavy_hitter:?}");
            }
        }

        // Make sure all heavy hitters are accounted for.
        let num_heavy_hitters = count
            .iter()
            .filter(|(_weight, count)| **count >= mastic.szk.typ.threshold)
            .count();
        assert_eq!(heavy_hitters.len(), num_heavy_hitters);
    }
}
