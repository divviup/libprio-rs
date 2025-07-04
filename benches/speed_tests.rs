// SPDX-License-Identifier: MPL-2.0

#[cfg(feature = "experimental")]
use criterion::Throughput;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "experimental")]
use fixed::types::{I1F15, I1F31};
#[cfg(feature = "experimental")]
use num_bigint::BigUint;
#[cfg(feature = "experimental")]
use num_rational::Ratio;
#[cfg(feature = "experimental")]
use num_traits::ToPrimitive;
#[cfg(feature = "experimental")]
use prio::dp::distributions::DiscreteGaussian;
#[cfg(feature = "experimental")]
use prio::idpf::test_utils::generate_zipf_distributed_batch;
#[cfg(feature = "experimental")]
use prio::vdaf::prio2::Prio2;
#[cfg(feature = "experimental")]
use prio::vidpf::VidpfServerId;
use prio::{
    benchmarked::*,
    field::{Field128 as F, FieldElement},
    flp::gadgets::Mul,
    vdaf::{prio3::Prio3, Aggregator, Client},
};
#[cfg(feature = "experimental")]
use prio::{
    field::{Field255, Field64},
    flp::types::fixedpoint_l2::FixedPointBoundedL2VecSum,
    idpf::{Idpf, IdpfInput, RingBufferCache},
    vdaf::poplar1::{Poplar1, Poplar1AggregationParam, Poplar1IdpfValue},
};
#[cfg(feature = "experimental")]
use rand::{distr::Distribution, random, rngs::StdRng, Rng, SeedableRng};
#[cfg(feature = "experimental")]
use std::iter;
use std::{hint::black_box, time::Duration};

/// Seed for generation of random benchmark inputs.
///
/// A fixed RNG seed is used to generate inputs in order to minimize run-to-run variability. The
/// seed value may be freely changed to get a different set of inputs.
#[cfg(feature = "experimental")]
const RNG_SEED: u64 = 0;

/// Speed test for generating a seed and deriving a pseudorandom sequence of field elements.
fn prng(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand");
    let test_sizes = [16, 256, 1024, 4096];
    for size in test_sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, size| {
            b.iter(|| F::random_vector(*size))
        });
    }
    group.finish();
}

/// Speed test for generating samples from the discrete gaussian distribution using different
/// standard deviations.
#[cfg(feature = "experimental")]
pub fn dp_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("dp_noise");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let test_stds = [
        Ratio::<BigUint>::from_integer(BigUint::from(u128::MAX)).pow(2),
        Ratio::<BigUint>::from_integer(BigUint::from(u64::MAX)),
        Ratio::<BigUint>::from_integer(BigUint::from(u32::MAX)),
        Ratio::<BigUint>::from_integer(BigUint::from(5u8)),
        Ratio::<BigUint>::new(BigUint::from(10000u32), BigUint::from(23u32)),
    ];
    for std in test_stds {
        let sampler = DiscreteGaussian::new(std.clone()).unwrap();
        group.bench_function(
            BenchmarkId::new("discrete_gaussian", std.to_f64().unwrap_or(f64::INFINITY)),
            |b| b.iter(|| sampler.sample(&mut rng)),
        );
    }
    group.finish();
}

/// The asymptotic cost of polynomial multiplication is `O(n log n)` using NTT and `O(n^2)` using
/// the naive method. This benchmark demonstrates that the latter has better concrete performance
/// for small polynomials. The result is used to pick the `NTT_THRESHOLD` constant in
/// `src/flp/gadgets.rs`.
fn poly_mul(c: &mut Criterion) {
    let test_sizes = [1_usize, 30, 60, 90, 120, 150, 255];

    let mut group = c.benchmark_group("poly_mul");
    for size in test_sizes {
        group.bench_with_input(BenchmarkId::new("ntt", size), &size, |b, size| {
            let m = (size + 1).next_power_of_two();
            let mut g: Mul<F> = Mul::new(*size);
            let mut outp = vec![F::zero(); 2 * m];
            let mut inp = vec![];
            inp.push(F::random_vector(m));
            inp.push(F::random_vector(m));

            b.iter(|| {
                benchmarked_gadget_mul_call_poly_ntt(&mut g, &mut outp, &inp).unwrap();
            })
        });

        group.bench_with_input(BenchmarkId::new("direct", size), &size, |b, size| {
            let m = (size + 1).next_power_of_two();
            let mut g: Mul<F> = Mul::new(*size);
            let mut outp = vec![F::zero(); 2 * m];
            let mut inp = vec![];
            inp.push(F::random_vector(m));
            inp.push(F::random_vector(m));

            b.iter(|| {
                benchmarked_gadget_mul_call_poly_direct(&mut g, &mut outp, &inp).unwrap();
            })
        });
    }
    group.finish();
}

/// Benchmark prio2.
#[cfg(feature = "experimental")]
fn prio2(c: &mut Criterion) {
    let mut group = c.benchmark_group("prio2_shard");
    for input_length in [10, 100, 1_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(input_length),
            &input_length,
            |b, input_length| {
                let vdaf = Prio2::new(*input_length).unwrap();
                let measurement = (0..u32::try_from(*input_length).unwrap())
                    .map(|i| i & 1)
                    .collect::<Vec<_>>();
                let nonce = black_box([0u8; 16]);
                b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("prio2_prepare_init");
    for input_length in [10, 100, 1_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(input_length),
            &input_length,
            |b, input_length| {
                let vdaf = Prio2::new(*input_length).unwrap();
                let measurement = (0..u32::try_from(*input_length).unwrap())
                    .map(|i| i & 1)
                    .collect::<Vec<_>>();
                let nonce = black_box([0u8; 16]);
                let verify_key = black_box([0u8; 32]);
                let (public_share, input_shares) = vdaf.shard(b"", &measurement, &nonce).unwrap();
                b.iter(|| {
                    vdaf.prepare_init(
                        &verify_key,
                        b"",
                        0,
                        &(),
                        &nonce,
                        &public_share,
                        &input_shares[0],
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark prio3.
fn prio3(c: &mut Criterion) {
    let num_shares = 2;

    c.bench_function("prio3count_shard", |b| {
        let vdaf = Prio3::new_count(num_shares).unwrap();
        let measurement = black_box(true);
        let nonce = black_box([0u8; 16]);
        b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
    });

    c.bench_function("prio3count_prepare_init", |b| {
        let vdaf = Prio3::new_count(num_shares).unwrap();
        let measurement = black_box(true);
        let nonce = black_box([0u8; 16]);
        let verify_key = black_box([0u8; 32]);
        let (public_share, input_shares) = vdaf.shard(b"", &measurement, &nonce).unwrap();
        b.iter(|| {
            vdaf.prepare_init(
                &verify_key,
                b"",
                0,
                &(),
                &nonce,
                &public_share,
                &input_shares[0],
            )
            .unwrap()
        });
    });

    let mut group = c.benchmark_group("prio3sum_shard");
    for bits in [8, 32] {
        group.bench_with_input(BenchmarkId::from_parameter(bits), &bits, |b, bits| {
            // Doesn't matter for speed what we use for max measurement, or measurement
            let max_measurement = (1 << bits) - 1;
            let vdaf = Prio3::new_sum(num_shares, max_measurement).unwrap();
            let measurement = max_measurement;
            let nonce = black_box([0u8; 16]);
            b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
        });
    }
    group.finish();

    let mut group = c.benchmark_group("prio3sum_prepare_init");
    for bits in [8, 32] {
        group.bench_with_input(BenchmarkId::from_parameter(bits), &bits, |b, bits| {
            let max_measurement = (1 << bits) - 1;
            let vdaf = Prio3::new_sum(num_shares, max_measurement).unwrap();
            let measurement = max_measurement;
            let nonce = black_box([0u8; 16]);
            let verify_key = black_box([0u8; 32]);
            let (public_share, input_shares) = vdaf.shard(b"", &measurement, &nonce).unwrap();
            b.iter(|| {
                vdaf.prepare_init(
                    &verify_key,
                    b"",
                    0,
                    &(),
                    &nonce,
                    &public_share,
                    &input_shares[0],
                )
                .unwrap()
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("prio3sumvec_shard");
    for (input_length, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
        group.bench_with_input(
            BenchmarkId::new("serial", input_length),
            &(input_length, chunk_length),
            |b, (input_length, chunk_length)| {
                let vdaf = Prio3::new_sum_vec(num_shares, 1, *input_length, *chunk_length).unwrap();
                let measurement = (0..u128::try_from(*input_length).unwrap())
                    .map(|i| i & 1)
                    .collect::<Vec<_>>();
                let nonce = black_box([0u8; 16]);
                b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
            },
        );
    }

    #[cfg(feature = "multithreaded")]
    {
        for (input_length, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
            group.bench_with_input(
                BenchmarkId::new("parallel", input_length),
                &(input_length, chunk_length),
                |b, (input_length, chunk_length)| {
                    let vdaf = Prio3::new_sum_vec_multithreaded(
                        num_shares,
                        1,
                        *input_length,
                        *chunk_length,
                    )
                    .unwrap();
                    let measurement = (0..u128::try_from(*input_length).unwrap())
                        .map(|i| i & 1)
                        .collect::<Vec<_>>();
                    let nonce = black_box([0u8; 16]);
                    b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
                },
            );
        }
    }
    group.finish();

    let mut group = c.benchmark_group("prio3sumvec_prepare_init");
    for (input_length, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
        group.bench_with_input(
            BenchmarkId::new("serial", input_length),
            &(input_length, chunk_length),
            |b, (input_length, chunk_length)| {
                let vdaf = Prio3::new_sum_vec(num_shares, 1, *input_length, *chunk_length).unwrap();
                let measurement = (0..u128::try_from(*input_length).unwrap())
                    .map(|i| i & 1)
                    .collect::<Vec<_>>();
                let nonce = black_box([0u8; 16]);
                let verify_key = black_box([0u8; 32]);
                let (public_share, input_shares) = vdaf.shard(b"", &measurement, &nonce).unwrap();
                b.iter(|| {
                    vdaf.prepare_init(
                        &verify_key,
                        b"",
                        0,
                        &(),
                        &nonce,
                        &public_share,
                        &input_shares[0],
                    )
                    .unwrap()
                });
            },
        );
    }

    #[cfg(feature = "multithreaded")]
    {
        for (input_length, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
            group.bench_with_input(
                BenchmarkId::new("parallel", input_length),
                &(input_length, chunk_length),
                |b, (input_length, chunk_length)| {
                    let vdaf = Prio3::new_sum_vec_multithreaded(
                        num_shares,
                        1,
                        *input_length,
                        *chunk_length,
                    )
                    .unwrap();
                    let measurement = (0..u128::try_from(*input_length).unwrap())
                        .map(|i| i & 1)
                        .collect::<Vec<_>>();
                    let nonce = black_box([0u8; 16]);
                    let verify_key = black_box([0u8; 32]);
                    let (public_share, input_shares) =
                        vdaf.shard(b"", &measurement, &nonce).unwrap();
                    b.iter(|| {
                        vdaf.prepare_init(
                            &verify_key,
                            b"",
                            0,
                            &(),
                            &nonce,
                            &public_share,
                            &input_shares[0],
                        )
                        .unwrap()
                    });
                },
            );
        }
    }
    group.finish();

    let mut group = c.benchmark_group("prio3histogram_shard");
    for (input_length, chunk_length) in [
        (10, 3),
        (100, 10),
        (1_000, 31),
        (10_000, 100),
        (100_000, 316),
    ] {
        if input_length >= 100_000 {
            group.measurement_time(Duration::from_secs(15));
        }
        group.bench_with_input(
            BenchmarkId::new("serial", input_length),
            &(input_length, chunk_length),
            |b, (input_length, chunk_length)| {
                let vdaf = Prio3::new_histogram(num_shares, *input_length, *chunk_length).unwrap();
                let measurement = black_box(0);
                let nonce = black_box([0u8; 16]);
                b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
            },
        );
    }

    #[cfg(feature = "multithreaded")]
    {
        for (input_length, chunk_length) in [
            (10, 3),
            (100, 10),
            (1_000, 31),
            (10_000, 100),
            (100_000, 316),
        ] {
            if input_length >= 100_000 {
                group.measurement_time(Duration::from_secs(15));
            }
            group.bench_with_input(
                BenchmarkId::new("parallel", input_length),
                &(input_length, chunk_length),
                |b, (input_length, chunk_length)| {
                    let vdaf = Prio3::new_histogram_multithreaded(
                        num_shares,
                        *input_length,
                        *chunk_length,
                    )
                    .unwrap();
                    let measurement = black_box(0);
                    let nonce = black_box([0u8; 16]);
                    b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
                },
            );
        }
    }
    group.finish();

    let mut group = c.benchmark_group("prio3histogram_prepare_init");
    for (input_length, chunk_length) in [
        (10, 3),
        (100, 10),
        (1_000, 31),
        (10_000, 100),
        (100_000, 316),
    ] {
        if input_length >= 100_000 {
            group.measurement_time(Duration::from_secs(15));
        }
        group.bench_with_input(
            BenchmarkId::new("serial", input_length),
            &(input_length, chunk_length),
            |b, (input_length, chunk_length)| {
                let vdaf = Prio3::new_histogram(num_shares, *input_length, *chunk_length).unwrap();
                let measurement = black_box(0);
                let nonce = black_box([0u8; 16]);
                let verify_key = black_box([0u8; 32]);
                let (public_share, input_shares) = vdaf.shard(b"", &measurement, &nonce).unwrap();
                b.iter(|| {
                    vdaf.prepare_init(
                        &verify_key,
                        b"",
                        0,
                        &(),
                        &nonce,
                        &public_share,
                        &input_shares[0],
                    )
                    .unwrap()
                });
            },
        );
    }

    #[cfg(feature = "multithreaded")]
    {
        for (input_length, chunk_length) in [
            (10, 3),
            (100, 10),
            (1_000, 31),
            (10_000, 100),
            (100_000, 316),
        ] {
            if input_length >= 100_000 {
                group.measurement_time(Duration::from_secs(15));
            }
            group.bench_with_input(
                BenchmarkId::new("parallel", input_length),
                &(input_length, chunk_length),
                |b, (input_length, chunk_length)| {
                    let vdaf = Prio3::new_histogram_multithreaded(
                        num_shares,
                        *input_length,
                        *chunk_length,
                    )
                    .unwrap();
                    let measurement = black_box(0);
                    let nonce = black_box([0u8; 16]);
                    let verify_key = black_box([0u8; 32]);
                    let (public_share, input_shares) =
                        vdaf.shard(b"", &measurement, &nonce).unwrap();
                    b.iter(|| {
                        vdaf.prepare_init(
                            &verify_key,
                            b"",
                            0,
                            &(),
                            &nonce,
                            &public_share,
                            &input_shares[0],
                        )
                        .unwrap()
                    });
                },
            );
        }
    }
    group.finish();

    let mut group = c.benchmark_group("prio3multihotcountvec_shard");
    for (input_length, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
        group.bench_with_input(
            BenchmarkId::new("serial", input_length),
            &(input_length, chunk_length),
            |b, (input_length, chunk_length)| {
                let vdaf =
                    Prio3::new_multihot_count_vec(num_shares, *input_length, 2, *chunk_length)
                        .unwrap();
                let mut measurement = vec![false; *input_length];
                measurement[0] = true;
                measurement[1] = true;
                let measurement = black_box(measurement);
                let nonce = black_box([0u8; 16]);
                b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
            },
        );
    }

    #[cfg(feature = "multithreaded")]
    {
        for (input_length, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
            group.bench_with_input(
                BenchmarkId::new("parallel", input_length),
                &(input_length, chunk_length),
                |b, (input_length, chunk_length)| {
                    let vdaf = Prio3::new_multihot_count_vec_multithreaded(
                        num_shares,
                        *input_length,
                        2,
                        *chunk_length,
                    )
                    .unwrap();
                    let mut measurement = vec![false; *input_length];
                    measurement[0] = true;
                    measurement[1] = true;
                    let measurement = black_box(measurement);
                    let nonce = black_box([0u8; 16]);
                    b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
                },
            );
        }
    }
    group.finish();

    let mut group = c.benchmark_group("prio3multihotcountvec_prepare_init");
    for (input_length, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
        group.bench_with_input(
            BenchmarkId::new("serial", input_length),
            &(input_length, chunk_length),
            |b, (input_length, chunk_length)| {
                let vdaf =
                    Prio3::new_multihot_count_vec(num_shares, *input_length, 2, *chunk_length)
                        .unwrap();
                let mut measurement = vec![false; *input_length];
                measurement[0] = true;
                measurement[1] = true;
                let measurement = black_box(measurement);
                let nonce = black_box([0u8; 16]);
                let verify_key = black_box([0u8; 32]);
                let (public_share, input_shares) = vdaf.shard(b"", &measurement, &nonce).unwrap();
                b.iter(|| {
                    vdaf.prepare_init(
                        &verify_key,
                        b"",
                        0,
                        &(),
                        &nonce,
                        &public_share,
                        &input_shares[0],
                    )
                    .unwrap();
                })
            },
        );
    }

    #[cfg(feature = "multithreaded")]
    {
        for (input_length, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
            group.bench_with_input(
                BenchmarkId::new("parallel", input_length),
                &(input_length, chunk_length),
                |b, (input_length, chunk_length)| {
                    let vdaf = Prio3::new_multihot_count_vec_multithreaded(
                        num_shares,
                        *input_length,
                        2,
                        *chunk_length,
                    )
                    .unwrap();
                    let mut measurement = vec![false; *input_length];
                    measurement[0] = true;
                    measurement[1] = true;
                    let measurement = black_box(measurement);
                    let nonce = black_box([0u8; 16]);
                    let verify_key = black_box([0u8; 32]);
                    let (public_share, input_shares) =
                        vdaf.shard(b"", &measurement, &nonce).unwrap();
                    b.iter(|| {
                        vdaf.prepare_init(
                            &verify_key,
                            b"",
                            0,
                            &(),
                            &nonce,
                            &public_share,
                            &input_shares[0],
                        )
                        .unwrap();
                    })
                },
            );
        }
    }
    group.finish();

    #[cfg(feature = "experimental")]
    {
        const FP16_ZERO: I1F15 = I1F15::lit("0");
        const FP32_ZERO: I1F31 = I1F31::lit("0");
        const FP16_HALF: I1F15 = I1F15::lit("0.5");
        const FP32_HALF: I1F31 = I1F31::lit("0.5");

        let mut group = c.benchmark_group("prio3fixedpointboundedl2vecsum_i1f15_shard");
        for dimension in [10, 100, 1_000] {
            group.bench_with_input(
                BenchmarkId::new("serial", dimension),
                &dimension,
                |b, dimension| {
                    let vdaf: Prio3<FixedPointBoundedL2VecSum<I1F15, _, _>, _, 32> =
                        Prio3::new_fixedpoint_boundedl2_vec_sum(num_shares, *dimension).unwrap();
                    let mut measurement = vec![FP16_ZERO; *dimension];
                    measurement[0] = FP16_HALF;
                    let nonce = black_box([0u8; 16]);
                    b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
                },
            );
        }

        #[cfg(feature = "multithreaded")]
        {
            for dimension in [10, 100, 1_000] {
                group.bench_with_input(
                    BenchmarkId::new("parallel", dimension),
                    &dimension,
                    |b, dimension| {
                        let vdaf: Prio3<FixedPointBoundedL2VecSum<I1F15, _, _>, _, 32> =
                            Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(
                                num_shares, *dimension,
                            )
                            .unwrap();
                        let mut measurement = vec![FP16_ZERO; *dimension];
                        measurement[0] = FP16_HALF;
                        let nonce = black_box([0u8; 16]);
                        b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
                    },
                );
            }
        }
        group.finish();

        let mut group = c.benchmark_group("prio3fixedpointboundedl2vecsum_i1f15_prepare_init");
        for dimension in [10, 100, 1_000] {
            group.bench_with_input(
                BenchmarkId::new("series", dimension),
                &dimension,
                |b, dimension| {
                    let vdaf: Prio3<FixedPointBoundedL2VecSum<I1F15, _, _>, _, 32> =
                        Prio3::new_fixedpoint_boundedl2_vec_sum(num_shares, *dimension).unwrap();
                    let mut measurement = vec![FP16_ZERO; *dimension];
                    measurement[0] = FP16_HALF;
                    let nonce = black_box([0u8; 16]);
                    let verify_key = black_box([0u8; 32]);
                    let (public_share, input_shares) =
                        vdaf.shard(b"", &measurement, &nonce).unwrap();
                    b.iter(|| {
                        vdaf.prepare_init(
                            &verify_key,
                            b"",
                            0,
                            &(),
                            &nonce,
                            &public_share,
                            &input_shares[0],
                        )
                        .unwrap()
                    });
                },
            );
        }

        #[cfg(feature = "multithreaded")]
        {
            for dimension in [10, 100, 1_000] {
                group.bench_with_input(
                    BenchmarkId::new("parallel", dimension),
                    &dimension,
                    |b, dimension| {
                        let vdaf: Prio3<FixedPointBoundedL2VecSum<I1F15, _, _>, _, 32> =
                            Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(
                                num_shares, *dimension,
                            )
                            .unwrap();
                        let mut measurement = vec![FP16_ZERO; *dimension];
                        measurement[0] = FP16_HALF;
                        let nonce = black_box([0u8; 16]);
                        let verify_key = black_box([0u8; 32]);
                        let (public_share, input_shares) =
                            vdaf.shard(b"", &measurement, &nonce).unwrap();
                        b.iter(|| {
                            vdaf.prepare_init(
                                &verify_key,
                                b"",
                                0,
                                &(),
                                &nonce,
                                &public_share,
                                &input_shares[0],
                            )
                            .unwrap()
                        });
                    },
                );
            }
        }
        group.finish();

        let mut group = c.benchmark_group("prio3fixedpointboundedl2vecsum_i1f31_shard");
        for dimension in [10, 100, 1_000] {
            group.bench_with_input(
                BenchmarkId::new("serial", dimension),
                &dimension,
                |b, dimension| {
                    let vdaf: Prio3<FixedPointBoundedL2VecSum<I1F31, _, _>, _, 32> =
                        Prio3::new_fixedpoint_boundedl2_vec_sum(num_shares, *dimension).unwrap();
                    let mut measurement = vec![FP32_ZERO; *dimension];
                    measurement[0] = FP32_HALF;
                    let nonce = black_box([0u8; 16]);
                    b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
                },
            );
        }

        #[cfg(feature = "multithreaded")]
        {
            for dimension in [10, 100, 1_000] {
                group.bench_with_input(
                    BenchmarkId::new("parallel", dimension),
                    &dimension,
                    |b, dimension| {
                        let vdaf: Prio3<FixedPointBoundedL2VecSum<I1F31, _, _>, _, 32> =
                            Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(
                                num_shares, *dimension,
                            )
                            .unwrap();
                        let mut measurement = vec![FP32_ZERO; *dimension];
                        measurement[0] = FP32_HALF;
                        let nonce = black_box([0u8; 16]);
                        b.iter(|| vdaf.shard(b"", &measurement, &nonce).unwrap());
                    },
                );
            }
        }
        group.finish();

        let mut group = c.benchmark_group("prio3fixedpointboundedl2vecsum_i1f31_prepare_init");
        for dimension in [10, 100, 1_000] {
            group.bench_with_input(
                BenchmarkId::new("series", dimension),
                &dimension,
                |b, dimension| {
                    let vdaf: Prio3<FixedPointBoundedL2VecSum<I1F31, _, _>, _, 32> =
                        Prio3::new_fixedpoint_boundedl2_vec_sum(num_shares, *dimension).unwrap();
                    let mut measurement = vec![FP32_ZERO; *dimension];
                    measurement[0] = FP32_HALF;
                    let nonce = black_box([0u8; 16]);
                    let verify_key = black_box([0u8; 32]);
                    let (public_share, input_shares) =
                        vdaf.shard(b"", &measurement, &nonce).unwrap();
                    b.iter(|| {
                        vdaf.prepare_init(
                            &verify_key,
                            b"",
                            0,
                            &(),
                            &nonce,
                            &public_share,
                            &input_shares[0],
                        )
                        .unwrap()
                    });
                },
            );
        }

        #[cfg(feature = "multithreaded")]
        {
            for dimension in [10, 100, 1_000] {
                group.bench_with_input(
                    BenchmarkId::new("parallel", dimension),
                    &dimension,
                    |b, dimension| {
                        let vdaf: Prio3<FixedPointBoundedL2VecSum<I1F31, _, _>, _, 32> =
                            Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(
                                num_shares, *dimension,
                            )
                            .unwrap();
                        let mut measurement = vec![FP32_ZERO; *dimension];
                        measurement[0] = FP32_HALF;
                        let nonce = black_box([0u8; 16]);
                        let verify_key = black_box([0u8; 32]);
                        let (public_share, input_shares) =
                            vdaf.shard(b"", &measurement, &nonce).unwrap();
                        b.iter(|| {
                            vdaf.prepare_init(
                                &verify_key,
                                b"",
                                0,
                                &(),
                                &nonce,
                                &public_share,
                                &input_shares[0],
                            )
                            .unwrap()
                        });
                    },
                );
            }
        }
        group.finish();
    }
}

/// Benchmark IdpfPoplar performance.
#[cfg(feature = "experimental")]
fn idpf(c: &mut Criterion) {
    let test_sizes = [8usize, 8 * 16, 8 * 256];

    let mut group = c.benchmark_group("idpf_gen");
    for size in test_sizes.iter() {
        group.throughput(Throughput::Bytes(*size as u64 / 8));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let bits = iter::repeat_with(random).take(size).collect::<Vec<bool>>();
            let input = IdpfInput::from_bools(&bits);

            let inner_values = Field64::random_vector(size - 1)
                .into_iter()
                .map(|random_element| Poplar1IdpfValue::new([Field64::one(), random_element]))
                .collect::<Vec<_>>();
            let leaf_value =
                Poplar1IdpfValue::new([Field255::one(), Field255::random_vector(1)[0]]);

            let idpf = Idpf::new((), ());
            b.iter(|| {
                idpf.gen(&input, inner_values.clone(), leaf_value, b"", &[0; 16])
                    .unwrap();
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("idpf_eval");
    for size in test_sizes.iter() {
        group.throughput(Throughput::Bytes(*size as u64 / 8));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let bits = iter::repeat_with(random).take(size).collect::<Vec<bool>>();
            let input = IdpfInput::from_bools(&bits);

            let inner_values = Field64::random_vector(size - 1)
                .into_iter()
                .map(|random_element| Poplar1IdpfValue::new([Field64::one(), random_element]))
                .collect::<Vec<_>>();
            let leaf_value =
                Poplar1IdpfValue::new([Field255::one(), Field255::random_vector(1)[0]]);

            let idpf = Idpf::new((), ());
            let (public_share, keys) = idpf
                .gen(&input, inner_values, leaf_value, b"", &[0; 16])
                .unwrap();

            b.iter(|| {
                // This is an aggressively small cache, to minimize its impact on the benchmark.
                // In this synthetic benchmark, we are only checking one candidate prefix per level
                // (typically there are many candidate prefixes per level) so the cache hit rate
                // will be unaffected.
                let mut cache = RingBufferCache::new(1);

                for prefix_length in 1..=size {
                    let prefix = input[..prefix_length].to_owned().into();
                    idpf.eval(
                        0,
                        &public_share,
                        &keys[0],
                        &prefix,
                        b"",
                        &[0; 16],
                        &mut cache,
                    )
                    .unwrap();
                }
            });
        });
    }
    group.finish();
}

/// Benchmark Poplar1.
#[cfg(feature = "experimental")]
fn poplar1(c: &mut Criterion) {
    let test_sizes = [16_usize, 128, 256];

    let mut group = c.benchmark_group("poplar1_shard");
    for size in test_sizes.iter() {
        group.throughput(Throughput::Bytes(*size as u64 / 8));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let vdaf = Poplar1::new_turboshake128(size);
            let mut rng = StdRng::seed_from_u64(RNG_SEED);
            let nonce = rng.random::<[u8; 16]>();
            let bits = iter::repeat_with(|| rng.random())
                .take(size)
                .collect::<Vec<bool>>();
            let measurement = IdpfInput::from_bools(&bits);

            b.iter(|| {
                vdaf.shard(b"", &measurement, &nonce).unwrap();
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("poplar1_prepare_init");
    for size in test_sizes.iter() {
        group.measurement_time(Duration::from_secs(30)); // slower benchmark
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let vdaf = Poplar1::new_turboshake128(size);
            let mut rng = StdRng::seed_from_u64(RNG_SEED);
            let verify_key: [u8; 32] = rng.random();
            let nonce: [u8; 16] = rng.random();

            // Parameters are chosen to match Chris Wood's experimental setup:
            // https://github.com/chris-wood/heavy-hitter-comparison
            let (measurements, prefix_tree) = generate_zipf_distributed_batch(
                &mut rng, // rng
                size,     // bits
                10,       // threshold
                1000,     // number of measurements
                128,      // Zipf support
                1.03,     // Zipf exponent
            );

            // We are benchmarking preparation of a single report. For this test, it doesn't matter
            // which measurement we generate a report for, so pick the first measurement
            // arbitrarily.
            let (public_share, input_shares) = vdaf.shard(b"", &measurements[0], &nonce).unwrap();
            let input_share = input_shares.into_iter().next().unwrap();

            // For the aggregation paramter, we use the candidate prefixes from the prefix tree for
            // the sampled measurements. Run preparation for the last step, which ought to represent
            // the worst-case performance.
            let agg_param =
                Poplar1AggregationParam::try_from_prefixes(prefix_tree[size - 1].clone()).unwrap();

            b.iter(|| {
                vdaf.prepare_init(
                    &verify_key,
                    b"",
                    0,
                    &agg_param,
                    &nonce,
                    &public_share,
                    &input_share,
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

/// Benchmark VIDPF performance.
#[cfg(feature = "experimental")]
fn vidpf(c: &mut Criterion) {
    use prio::vidpf::{Vidpf, VidpfInput, VidpfWeight};

    let test_sizes = [8usize, 8 * 16, 8 * 256];
    const NONCE_SIZE: usize = 16;
    const NONCE: &[u8; NONCE_SIZE] = b"Test Nonce VIDPF";

    let mut group = c.benchmark_group("vidpf_gen");
    for size in test_sizes.iter() {
        group.throughput(Throughput::Bytes(*size as u64 / 8));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let bits = iter::repeat_with(random).take(size).collect::<Vec<bool>>();
            let input = VidpfInput::from_bools(&bits);
            let weight = VidpfWeight::from(vec![Field255::one(), Field255::one()]);

            let vidpf = Vidpf::<VidpfWeight<Field255>>::new(bits.len(), 2).unwrap();

            b.iter(|| {
                let _ = vidpf
                    .gen(b"some application", &input, &weight, NONCE)
                    .unwrap();
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("vidpf_eval");
    for size in test_sizes.iter() {
        group.throughput(Throughput::Bytes(*size as u64 / 8));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let bits = iter::repeat_with(random).take(size).collect::<Vec<bool>>();
            let input = VidpfInput::from_bools(&bits);
            let weight = VidpfWeight::from(vec![Field255::one(), Field255::one()]);
            let vidpf = Vidpf::<VidpfWeight<Field255>>::new(bits.len(), 2).unwrap();

            let (public, keys) = vidpf
                .gen(b"some application", &input, &weight, NONCE)
                .unwrap();

            b.iter(|| {
                let _ = vidpf
                    .eval(
                        b"some application",
                        VidpfServerId::S0,
                        &keys[0],
                        &public,
                        &input,
                        NONCE,
                    )
                    .unwrap();
            });
        });
    }
    group.finish();
}

#[cfg(feature = "experimental")]
criterion_group!(benches, poplar1, prio3, prio2, poly_mul, prng, idpf, dp_noise, vidpf);
#[cfg(not(feature = "experimental"))]
criterion_group!(benches, prio3, prng, poly_mul);

criterion_main!(benches);
